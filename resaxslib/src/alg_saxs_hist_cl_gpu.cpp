///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2016 Lubo Antonov
//
//    This file is part of RESAXS.
//
//    RESAXS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    any later version.
//
//    RESAXS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with RESAXS.  If not, see <http://www.gnu.org/licenses/>.
//
///////////////////////////////////////////////////////////////////////////////

#include "alg_saxs_hist_cl_gpu.hpp"
#include <assert.h>
#include <sstream>
#include <fstream>
#include "utils.hpp"
#include "ff_coef.hpp"

#include <iostream>

using namespace std;
using namespace std::placeholders;

namespace resaxs {
namespace algorithm {
#include "saxs_hist_gpu.cl"

    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::atomic_ff_params::initialize(const std::vector<FLT_T> &qq, std::function<void()> call_set_ext_dirty)
    {
        call_set_ext_dirty_ = call_set_ext_dirty;

        qq_ = &qq;
        atomic_form_factors<FLT_T>::generate(qq, this->factors_);
        this->set_water_ff_index(atoms::OH2);
    }

    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::atomic_ff_params::set_expansion_factor(FLT_T c)
    {
        if (this->expansion_factor_ != c)
        {
            form_factor_params<FLT_T>::set_expansion_factor(c);

            atomic_form_factors<FLT_T>::generate(*qq_, owner_.avg_radius_, c, this->factors_);

            exp_factor_changed_sink_(*this);
        }
    }

    template <typename FLT_T>
    saxs_hist_cl_gpu<FLT_T>::gpu_pt_wf_params::gpu_pt_wf_params(saxs_hist_cl_gpu<FLT_T> &owner) :
        owner_(owner),
        ff_params_(owner, std::bind(&saxs_hist_cl_gpu<FLT_T>::on_exp_factor_changed, &owner, _1)),
        rsasa_alg(rsasa_alg_factory::template create<rsasa_alg_base, rsasa_dot_spheres>())
    {
        this->set_ocl_changed_event_sink(std::bind(&saxs_hist_cl_gpu<FLT_T>::on_ocl_device_changed, &owner, _1));
    }

    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::gpu_pt_wf_params::initialize(const std::vector<FLT_T> &qq, const std::vector<real4> &bodies,
        const std::vector<dev_id> &devices, unsigned int workgroup_size)
    {
        saxs_params<FLT_T>::initialize(qq, bodies, devices, workgroup_size);
        ff_params_.initialize(qq, std::bind(&gpu_pt_wf_params::on_dirty_set, this));
    }

    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::gpu_pt_wf_params::update_rsasa(const std::vector<real4> &bodies)
    {
        if (rsasa_alg->computing())
            rsasa_alg->recalc_rsasa(bodies, this->access_rsasa());
        else
        {
            if (!rsasa_alg->initialized())
            {
                typename rsasa_alg_factory::dot_spheres_cl_params rsasa_params(0.5f, 1.8f, get_ocl_exec_params().devices_, this->n_bodies_);
                rsasa_alg->initialize(&rsasa_params);
            }
            rsasa_alg->calc_rsasa(bodies, this->access_rsasa());
        }
    }


    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::on_exp_factor_changed(const form_factor_params<FLT_T> & ff_params)
    {
        if (this->args_set())
        {
            b_factors_.write_from(params_.get_ff_params().factors_, this->queue_);
            changed_buffers.factors = true;

            const auto workgroup_size = params_.get_ocl_exec_params().workgroup_size_;

            // map factors to bodies for all q
            this->queue_.enqueueNDRangeKernel(kernel_map_factors_, cl::NullRange, cl::NDRange(n_bodies_aln_), cl::NDRange(workgroup_size));
            changed_buffers.body_factors = true;
        }
    }

    //////////////////////////////////////////////////////////////////////////
    ///  Handle change of the OCL devices: move over the program and buffers.
    ///  The algorithm will continue on the new devices.
    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::on_ocl_device_changed(const ocl_exec_params &ocl_params)
    {
        if (!this->args_set())
            return;

        auto old_state = this->state;
        auto old_queue = this->queue_;

        alg_base::initialize(params_.get_ocl_exec_params().devices_);

        load_program();

        /// input OpenCL buffers
        b_qq_.init(this->context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, params_.qq_);
        b_factors_.init(this->context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, params_.get_ff_params().factors_);
        b_bodies_.move(old_queue, this->context_);

        auto & rsasa = params_.get_implicit_water_params().access_rsasa();
        b_body_surface_.init(this->context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rsasa); // aligned to WF_SIZE

                                                                                                /// output OpenCL buffer
        b_Iqq_.move(old_queue, this->context_);

        /// internal device buffers
        b_margin_Iqq_.move(old_queue, this->context_);   // aligned to 4
        b_body_factors_.move(old_queue, this->context_);    // aligned to WF_SIZE
        b_block_sums_.move(old_queue, this->context_);    // aligned to 4x4

        set_kernel_args();

        if (xaction())
        {
            b_saved_bodies_.move(old_queue, this->context_);  // aligned to WF_SIZE
            b_saved_body_surface_.move(old_queue, this->context_); // aligned to WF_SIZE
            b_saved_block_sums_.move(old_queue, this->context_); // aligned to 4x4
            b_saved_factors_.move(old_queue, this->context_);
            b_saved_body_factors_.move(old_queue, this->context_);    // aligned to WF_SIZE
        }

        params_.on_ocl_changed(ocl_params);

        this->state = old_state;

        params_.get_ocl_exec_params().clear_dirty();
    }

    template <typename FLT_T>
    saxs_hist_cl_gpu<FLT_T>::saxs_hist_cl_gpu() : params_(*this), n_q_(0), n_bodies_aln_(0), avg_radius_(1.62f)
    {
        changed_buffers.bodies = false;
        changed_buffers.body_factors = false;
        changed_buffers.factors = false;
        changed_buffers.rsasa = false;
        changed_buffers.sum = false;
    }

    template <typename FLT_T>
    saxs_params<FLT_T> & saxs_hist_cl_gpu<FLT_T>::access_params()
    {
        return params_;
    }

    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::initialize()
    {
        alg_base::initialize(params_.get_ocl_exec_params().devices_);

        load_program();
        set_args();

        if (radii_map_.empty())
            atomic_form_factors<FLT_T>::calc_radii(radii_map_);

        if (xaction())
        {
            b_saved_bodies_.clear();
            b_saved_body_surface_.clear();
            b_saved_block_sums_.clear();
            b_saved_factors_.clear();
            b_saved_body_factors_.clear();
        }
    }

    //////////////////////////////////////////////////////////////////////////
    ///  Loads the program for this algorithm to the devices.
    ///  The algorithm should have been already initialized with devices.
    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::load_program()
    {
        verify(this->initialized(), error::SAXS_ALG_NOT_INITIALIZED);

        // add the wavefront size as a parameter to the program
        ostringstream all_options;
        all_options << "-D GROUP_SIZE=" << params_.get_ocl_exec_params().workgroup_size_ <<
            " -D WATER_FF_INDEX=" << params_.get_ff_params().water_ff_index_;

        this->build_program(saxs_hist_gpu_cl, program_, all_options.str());

        // initialize the kernels
        kernel_map_factors_ = cl::Kernel(program_, "map_factors");
        kernel_calc_ = cl::Kernel(program_, "calc_block");
        kernel_vsum_ = cl::Kernel(program_, "block_v_sum");
        kernel_hsum_ = cl::Kernel(program_, "block_h_sum");
        kernel_recalc_ = cl::Kernel(program_, "recalc_block");
        kernel_recalc_with_pivots_ = cl::Kernel(program_, "recalc_block_with_pivots");

        /*auto a = program_.getInfo<CL_PROGRAM_BINARIES>();
        string s = a[0];

        ofstream f("saxs_cl.bin", ios::trunc | ios::out | ios::binary);
        f.write(s.c_str(), s.size());*/
        this->state = this->alg_program_loaded;
    }

    //////////////////////////////////////////////////////////////////////////
    ///  Sets the arguments for the pending calculations for a set of bodies (a protein).
    ///  These will not change for a particular protein. Calling this again will reset the algorithm for a new protein.
    ///  There needs to be a program already loaded.
    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::set_args()
    {
        verify(this->program_loaded(), error::SAXS_PROGRAM_NOT_LOADED);

        const auto & factors = params_.get_ff_params().factors_;
        auto workgroup_size = params_.get_ocl_exec_params().workgroup_size_;

        n_q_ = int(params_.qq_.size());                                  // number of q values
        n_bodies_aln_ = aligned_num(params_.n_bodies_, workgroup_size); // number of bodies, aligned to the workgroup size
        int n_blocks_aln = aligned_num(n_bodies_aln_ / workgroup_size, 4); // number of blocks in each dimension, aligned to 4

                                                                            /// input OpenCL buffers
        b_qq_.init(this->context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, params_.qq_);
        b_factors_.init(this->context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, factors);
        b_bodies_.init(this->context_, CL_MEM_READ_WRITE, n_bodies_aln_); // aligned to WF_SIZE

        auto & rsasa = params_.get_implicit_water_params().access_rsasa();
        rsasa.resize(n_bodies_aln_);
        b_body_surface_.init(this->context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, rsasa); // aligned to WF_SIZE

                                                                                                /// output OpenCL buffer
        b_Iqq_.init(this->context_, CL_MEM_WRITE_ONLY, n_q_);

        /// internal device buffers
        vector<FLT_T> zero_margin_Iqq(n_q_ * n_blocks_aln, 0);
        b_margin_Iqq_.init(this->context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, zero_margin_Iqq);   // aligned to 4
        b_body_factors_.init(this->context_, CL_MEM_READ_WRITE, n_q_ * n_bodies_aln_);    // aligned to WF_SIZE
        vector<FLT_T> zero_block_sums(n_q_ * n_blocks_aln * n_blocks_aln, 0);
        b_block_sums_.init(this->context_, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, zero_block_sums);    // aligned to 4x4

        set_kernel_args();

        this->state = this->alg_args_set;
    }

    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::set_kernel_args()
    {
        const auto & factors = params_.get_ff_params().factors_;
        auto workgroup_size = params_.get_ocl_exec_params().workgroup_size_;
        int n_factors = int(factors.size() / n_q_);             // number of form factors per q value
        auto & water_params_ = params_.get_implicit_water_params();

        kernel_map_factors_.setArg(ka_mf_n_q, n_q_);
        kernel_map_factors_.setArg(ka_mf_v_bodies, b_bodies_);
        kernel_map_factors_.setArg(ka_mf_n_bodies, params_.n_bodies_);
        kernel_map_factors_.setArg(ka_mf_t_factors, b_factors_);
        kernel_map_factors_.setArg(ka_mf_n_factors, n_factors);
        kernel_map_factors_.setArg(ka_mf_t_body_factors, b_body_factors_);

        kernel_calc_.setArg(ka_calc_v_q, b_qq_);
        kernel_calc_.setArg(ka_calc_v_bodies, b_bodies_);
        kernel_calc_.setArg(ka_calc_t_factors, b_factors_);
        kernel_calc_.setArg(ka_calc_n_factors, n_factors);
        kernel_calc_.setArg(ka_calc_t_body_factors, b_body_factors_);
        kernel_calc_.setArg(ka_calc_v_body_surface, b_body_surface_);        // surface area factors - will be filled-in at each iteration
        kernel_calc_.setArg(ka_calc_water_weight, water_params_.water_weight_);        // weight of the water contribution - will be filled-in later
        kernel_calc_.setArg(ka_calc_t_block_sums, b_block_sums_);

        kernel_vsum_.setArg(ka_vsum_t_block_sums, b_block_sums_);
        kernel_vsum_.setArg(ka_vsum_t_margin_Iq, b_margin_Iqq_);

        kernel_hsum_.setArg(ka_hsum_t_margin_Iq, b_margin_Iqq_);
        kernel_hsum_.setArg(ka_hsum_n_blocks, n_bodies_aln_ / workgroup_size);
        kernel_hsum_.setArg(ka_hsum_out_v_Iq, b_Iqq_);

        kernel_recalc_.setArg(ka_recalc_v_q, b_qq_);
        kernel_recalc_.setArg(ka_recalc_v_bodies, b_bodies_);
        kernel_recalc_.setArg(ka_recalc_t_factors, b_factors_);
        kernel_recalc_.setArg(ka_recalc_n_factors, n_factors);
        kernel_recalc_.setArg(ka_recalc_t_body_factors, b_body_factors_);
        kernel_recalc_.setArg(ka_recalc_v_body_surface, b_body_surface_);      // surface area factors - will be filled-in at each iteration
        kernel_recalc_.setArg(ka_recalc_water_weight, water_params_.water_weight_);        // weight of the water contribution - will be filled-in later
                                                                                            //kernel_recalc_.setArg(ka_recalc_upd_start_aln, 0);                  // update start - will be filled-in later
                                                                                            //kernel_recalc_.setArg(ka_recalc_upd_end_aln, n_bodies_aln_);      // update end - will be filled-in later
        kernel_recalc_.setArg(ka_recalc_t_block_sums, b_block_sums_);

        kernel_recalc_with_pivots_.setArg(ka_recalc_v_q, b_qq_);
        kernel_recalc_with_pivots_.setArg(ka_recalc_v_bodies, b_bodies_);
        kernel_recalc_with_pivots_.setArg(ka_recalc_t_factors, b_factors_);
        kernel_recalc_with_pivots_.setArg(ka_recalc_n_factors, n_factors);
        kernel_recalc_with_pivots_.setArg(ka_recalc_t_body_factors, b_body_factors_);
        kernel_recalc_with_pivots_.setArg(ka_recalc_v_body_surface, b_body_surface_);      // surface area factors - will be filled-in at each iteration
        kernel_recalc_with_pivots_.setArg(ka_recalc_water_weight, water_params_.water_weight_);        // weight of the water contribution - will be filled-in later
                                                                                                        //kernel_recalc_with_pivots_.setArg(ka_recalc_upd_start_aln, 0);                  // update start - will be filled-in later
                                                                                                        //kernel_recalc_with_pivots_.setArg(ka_recalc_upd_end_aln, n_bodies_aln_);      // update end - will be filled-in later
        kernel_recalc_with_pivots_.setArg(ka_recalc_t_block_sums, b_block_sums_);
    }

    //////////////////////////////////////////////////////////////////////////
    ///  Initialize the internal buffers for tracking transactions.
    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::init_xaction()
    {
        b_saved_bodies_.init(this->context_, CL_MEM_READ_WRITE, b_bodies_.size_);  // aligned to WF_SIZE
        b_saved_body_surface_.init(this->context_, CL_MEM_READ_WRITE, b_body_surface_.size_); // aligned to WF_SIZE
        b_saved_block_sums_.init(this->context_, CL_MEM_READ_WRITE, b_block_sums_.size_); // aligned to 4x4
        b_saved_factors_.init(this->context_, CL_MEM_READ_WRITE, b_factors_.size_);
        b_saved_body_factors_.init(this->context_, CL_MEM_READ_WRITE, b_body_factors_.size_);    // aligned to WF_SIZE
    }

    //////////////////////////////////////////////////////////////////////////
    ///  Updates that bodies on the device and the RSASA if necessary.
    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::update_bodies(const std::vector<real4> &bodies, unsigned int &&upd_start, unsigned int &&upd_length)
    {
        assert(bodies.size() >= upd_start + upd_length);
        assert(upd_start + upd_length <= params_.n_bodies_);

        auto & water_params = params_.get_implicit_water_params();
        bool update_rsasa = false;

        if (upd_length > 0)
        {
            // send the moved bodies
            b_bodies_.write_from(bodies, upd_start, upd_length, this->queue_);
            changed_buffers.bodies = true;

            // If the water weigh is 0, then RSASA doesn't matter - the water form factors will be zeroed out
            update_rsasa = water_params.water_weight_ != 0;
        }
        else
        {
            update_rsasa = water_params.is_dirty() && !params_.has_rsasa() && water_params.water_weight_ != 0;
        }

        // check if we should skip RSASA recalc
        if (water_params.no_rsasa_recalc_ && params_.has_rsasa())
            update_rsasa = false;

        if (update_rsasa)
        {
            // for now, change in any rsasa will cause a full recalculation
            //upd_start = 0;
            //upd_length = params_.n_bodies_;

            auto prev_rsasa = water_params.access_rsasa();
            params_.update_rsasa(bodies);
            auto &rsasa = water_params.access_rsasa();

            int first_diff = -1, last_diff = -1;
            for (size_t j = 0; j < prev_rsasa.size(); ++j)
            {
                if (prev_rsasa[j] != rsasa[j])
                {
                    if (first_diff < 0)
                        first_diff = (int)j;
                    last_diff = (int)j;
                }
            }
            ++last_diff;
            if (first_diff < 0)
            {
                //cout << "# DEBUG: SAXS: no RSASA change" << endl;
                return;
            }
            else
            {
                //cout << "# DEBUG: SAXS: RSASA changed between " << first_diff << "-" << last_diff << endl;                
                last_diff = std::max(last_diff, int(upd_start + upd_length));
                upd_start = min((unsigned int)first_diff, upd_start);
                upd_length = last_diff - upd_start;
                //cout << "# DEBUG: SAXS: update between " << upd_start << "-" << upd_start + upd_length << endl;                
            }


            // send all body surface area factors to the device
            // Note: only need to send updated bodies, as the 0-padded aligned buffer was sent at initialization
            b_body_surface_.write_from(water_params.access_rsasa(), upd_start, upd_length, this->queue_);
            changed_buffers.rsasa = true;
        }
    }

    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::update_bodies(const vector<real4> &bodies)
    {
        update_bodies(bodies, 0, (unsigned int)(bodies.size()));
    }

    //////////////////////////////////////////////////////////////////////////
    ///  Performs the initial calculation of the SAXS curve.
    ///  The algorithm arguments should have been set before calling this.
    ///      bodies          - vector of atomic bodies, describing the protein
    ///      out_Iqq         - buffer to receive the result - I(q) - for each value of q
    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::calc_curve(const vector<real4> &bodies, vector<FLT_T> &out_Iqq)
    {
        verify(this->args_set(), error::SAXS_ARGS_NOT_SET);
        verify(bodies.size() == params_.n_bodies_, error::SAXS_INVALID_ARG, "RESAXS::error -- different number of bodies expected in calc_curve()");

        // send all bodies to the device
        //b_bodies_.write_from(bodies, this->queue_);
        update_bodies(bodies);

        avg_radius_ = atomic_form_factors<FLT_T>::calc_avg_radius(bodies, radii_map_);

        const auto workgroup_size = params_.get_ocl_exec_params().workgroup_size_;
        // map factors to bodies for all q
        this->queue_.enqueueNDRangeKernel(kernel_map_factors_, cl::NullRange, cl::NDRange(n_bodies_aln_), cl::NDRange(workgroup_size));
        changed_buffers.body_factors = true;

        auto & water_params_ = params_.get_implicit_water_params();
        kernel_calc_.setArg(ka_calc_water_weight, water_params_.water_weight_);        // weight of the water contribution
        kernel_recalc_.setArg(ka_recalc_water_weight, water_params_.water_weight_);
        kernel_recalc_with_pivots_.setArg(ka_recalc_water_weight, water_params_.water_weight_);

        params_.clear_dirty();

        // reset the pivot params
        auto &pivot_params = params_.get_pivot_params();
        pivot_params.pivot_start_ = pivot_params.pivot_end_ = -1;

        // calculate Debye for all square blocks of size wavefront X wavefront
        this->queue_.enqueueNDRangeKernel(kernel_calc_, cl::NullRange,
            cl::NDRange(n_bodies_aln_, n_bodies_aln_ / workgroup_size, n_q_), cl::NDRange(workgroup_size, 1, 1));
        // sum the Debye results vertically
        this->queue_.enqueueNDRangeKernel(kernel_vsum_, cl::NullRange, cl::NDRange(n_bodies_aln_ / workgroup_size, n_q_), cl::NullRange);
        // sum the Debye results horizontally
        this->queue_.enqueueNDRangeKernel(kernel_hsum_, cl::NullRange, cl::NDRange(n_q_), cl::NullRange);
        changed_buffers.sum = true;

        // retrieve the results I(q)
        out_Iqq.resize(n_q_);
        b_Iqq_.read_to(out_Iqq, this->queue_);

        this->state = this->alg_computing;
    }

    //
    //  Recalculates the SAXS curve that was previously calculated.
    //  Initial calculation should have been performed before calling this.
    //      bodies          - vector of atomic bodies, describing the protein. it will contains both moved and retained bodies.
    //      upd_start       - index of the first moved body
    //      upd_length      - number of moved bodies
    //      out_Iqq         - buffer to receive the result - I(q) - for each value of q
    //
    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::recalc_curve(const std::vector<real4> &bodies, unsigned int upd_start, unsigned int upd_length, std::vector<FLT_T> &out_Iqq)
    {
        verify(this->computing(), error::SAXS_NO_INITIAL_CALC, "RESAXS::error -- recalc_curve() called before an initial call to calc_curve()");
        verify(bodies.size() >= upd_start + upd_length, error::SAXS_INVALID_ARG, "RESAXS::error -- bodies vector too small in recalc_curve()");
        verify(upd_start + upd_length <= params_.n_bodies_, error::SAXS_INVALID_ARG, "RESAXS::error -- update region exceeds the number of bodies in recalc_curve()");

        auto & water_params = params_.get_implicit_water_params();

        update_bodies(bodies, std::move(upd_start), std::move(upd_length));

        if (params_.is_dirty())
        {
            // new params will force a full recalculation
            upd_start = 0;
            upd_length = params_.n_bodies_;

            if (water_params.is_dirty())
            {
                kernel_recalc_.setArg(ka_recalc_water_weight, water_params.water_weight_);        // weight of the water contribution
                kernel_recalc_with_pivots_.setArg(ka_recalc_water_weight, water_params.water_weight_);        // weight of the water contribution
            }

            params_.clear_dirty();
        }

        if (upd_length > 0)
        {
            const auto workgroup_size = params_.get_ocl_exec_params().workgroup_size_;

            /*vector<FLT_T> blocks(b_block_sums_.size_);
            b_block_sums_.read_to(blocks, this->queue_);

            ofstream f("saxs_sums1.dat", ios::trunc);
            int x = 0;
            //for (int j = 0; j < n_q_; ++j)
            for (int i = 0; i < 128;++i)
            {
            for (int t = 0; t < 128; ++t)
            f << blocks[x++] << ' ';
            f << endl;
            }*/

            //cout << "# DEBUG: SAXS: recalc between " << upd_start << "-" << upd_start + upd_length << endl;                

            // align the start index and the length to wavefront_size_
            int upd_start_aln = (upd_start / workgroup_size) * workgroup_size;
            int upd_end_aln = aligned_num(upd_start + upd_length, workgroup_size);

            auto &pivot_params = params_.get_pivot_params();
            int pivot_start_aln = (pivot_params.pivot_start_ / workgroup_size) * workgroup_size;
            int pivot_end_aln = aligned_num(pivot_params.pivot_end_, workgroup_size);
            pivot_params.pivot_start_ = pivot_params.pivot_end_ = -1;

            if (!changed_buffers.rsasa && pivot_params.pivot_start_ >= 0 && (pivot_start_aln > upd_start_aln || pivot_end_aln < upd_end_aln))
            {
                kernel_recalc_with_pivots_.setArg(ka_recalc_upd_start_aln, pivot_start_aln);
                kernel_recalc_with_pivots_.setArg(ka_recalc_upd_end_aln, pivot_end_aln);
                // recalculate blocks affected by the update
                //cl::NDRange offset(0, pivot_start_aln / workgroup_size, 0);
                //cl::NDRange work_size(pivot_end_aln, (n_bodies_aln_ - pivot_start_aln) / workgroup_size, n_q_);
                //this->queue_.enqueueNDRangeKernel(kernel_recalc_with_pivots_, offset, work_size, cl::NDRange(workgroup_size, 1, 1));
                this->queue_.enqueueNDRangeKernel(kernel_recalc_with_pivots_, cl::NullRange, cl::NDRange(n_bodies_aln_, n_bodies_aln_ / workgroup_size, n_q_),
                    cl::NDRange(workgroup_size, 1, 1));
            }
            else
            {
                kernel_recalc_.setArg(ka_recalc_upd_start_aln, upd_start_aln);
                kernel_recalc_.setArg(ka_recalc_upd_end_aln, upd_end_aln);
                // recalculate blocks affected by the update
                /*cl::NDRange offset(0, upd_start_aln / workgroup_size, 0);
                cl::NDRange work_size(n_bodies_aln_, (n_bodies_aln_ - upd_start_aln) / workgroup_size, n_q_);
                this->queue_.enqueueNDRangeKernel(kernel_recalc_, offset, work_size, cl::NDRange(workgroup_size, 1, 1));*/
                this->queue_.enqueueNDRangeKernel(kernel_recalc_, cl::NullRange, cl::NDRange(n_bodies_aln_, n_bodies_aln_ / workgroup_size, n_q_),
                    cl::NDRange(workgroup_size, 1, 1));
            }

            // sum the Debye results vertically
            this->queue_.enqueueNDRangeKernel(kernel_vsum_, cl::NullRange, cl::NDRange(n_bodies_aln_ / workgroup_size, n_q_), cl::NullRange);
            // sum the Debye results horizontally
            this->queue_.enqueueNDRangeKernel(kernel_hsum_, cl::NullRange, cl::NDRange(n_q_), cl::NullRange);
            changed_buffers.sum = true;

            // retrieve the results I(q)
            out_Iqq.resize(n_q_);
            b_Iqq_.read_to(out_Iqq, this->queue_);

            /*b_block_sums_.read_to(blocks, this->queue_);

            ofstream f1("saxs_sums2.dat", ios::trunc);
            x = 0;
            //for (int j = 0; j < n_q_; ++j)
            for (int i = 0; i < 128;++i)
            {
            for (int t = 0; t < 128; ++t)
            f1 << blocks[x++] << ' ';
            f1 << endl;
            }*/
        }
    }

    template <typename FLT_T>
    template <typename U>
    inline void saxs_hist_cl_gpu<FLT_T>::copy_if_changed(Buffer<U> &from, Buffer<U> &to, bool &changed)
    {
        if (changed)
        {
            from.copy_to(to, this->queue_);
            changed = false;
        }
    }

    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::commit()
    {
        verify(this->computing(), error::SAXS_NO_INITIAL_CALC, "RESAXS::error -- commit() called before an initial call to calc_curve()");

        if (!xaction())
            init_xaction();

        copy_if_changed(b_bodies_, b_saved_bodies_, changed_buffers.bodies);
        copy_if_changed(b_block_sums_, b_saved_block_sums_, changed_buffers.sum);
        copy_if_changed(b_body_surface_, b_saved_body_surface_, changed_buffers.rsasa);
        copy_if_changed(b_factors_, b_saved_factors_, changed_buffers.factors);
        copy_if_changed(b_body_factors_, b_saved_body_factors_, changed_buffers.body_factors);
    }

    template <typename FLT_T>
    void saxs_hist_cl_gpu<FLT_T>::revert()
    {
        verify(this->computing(), error::SAXS_NO_INITIAL_CALC, "RESAXS::error -- revert() called before an initial call to calc_curve()");

        if (!xaction())
            return;

        copy_if_changed(b_saved_bodies_, b_bodies_, changed_buffers.bodies);
        copy_if_changed(b_saved_block_sums_, b_block_sums_, changed_buffers.sum);
        copy_if_changed(b_saved_body_surface_, b_body_surface_, changed_buffers.rsasa);
        copy_if_changed(b_saved_factors_, b_factors_, changed_buffers.factors);
        copy_if_changed(b_saved_body_factors_, b_body_factors_, changed_buffers.body_factors);
    }

    template class saxs_hist_cl_gpu<float>;
    template class saxs_hist_cl_gpu<double>;

} // namespace algorithm
} // namespace
