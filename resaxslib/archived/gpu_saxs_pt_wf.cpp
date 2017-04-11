///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2013-2015 Lubo Antonov
//
//    This file is part of ACCSAXS.
//
//    ACCSAXS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    any later version.
//
//    ACCSAXS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with ACCSAXS.  If not, see <http://www.gnu.org/licenses/>.
//
///////////////////////////////////////////////////////////////////////////////

#include "../include/gpu_saxs_pt_wf.hpp"

#include <sstream>
#include <algorithm>
#include <cmath>
#include <iostream>
#include <functional>

#include <limits>

#include "../include/utils.hpp"

namespace accsaxs {

#include "gpu_saxs_pt_wf.cl"

    gpu_saxs_pt_wf_alg::gpu_saxs_pt_wf_alg() : wavefront_size_(SAXS_NV_FERMI_WF_SIZE), water_weight_(0.0), curve_params_updated_(true),
        water_ff_index_(std::numeric_limits<unsigned int>::max()),//!param_b_(0.23f),
        upd_angle_start_aln_(0), upd_angle_end_aln_(0), last_upd_start_(0), last_upd_length_(0), calc_state_(calc_pending), init_state_(init_blank) {}

    void gpu_saxs_pt_wf_alg::initialize(const std::vector<dev_id> &dev_ids, unsigned int water_ff_index)
    {
        base::initialize(dev_ids);

        water_ff_index_ = water_ff_index;
        init_state_ = init_done;
    }

//
//  Loads the program for this algorithm to the devices.
//  The algorithm should have been already initialized with devices.
//      options         - additional options to send to the OpenCL compiler
//                          (defines, optimizations, etc. - see clBuildProgram in the OpenCL Specification section 5.6.3)
//      wavefront_size  - the wavefront size to use in the algorithm.
//                          matching this to the hardware could result in dramatic speedup, depending on the algorithm.
//                          the default of 0 lets the algorithm pick a size.
//                          the current default size is the size of a warp on NVIDIA Fermi devices - 32.
//                          for AMD devices (RV8xx and later) a wavefront size of 64 should be used.
//
void gpu_saxs_pt_wf_alg::load_program(const std::string & options, int wavefront_size)
{
    verify(initialized() && init_state_ == init_done, error::SAXS_ALG_NOT_INITIALIZED);
    
    wavefront_size_ = wavefront_size == 0 ? SAXS_NV_FERMI_WF_SIZE : wavefront_size;

    // add the wavefront size as a parameter to the program
    std::ostringstream all_options;
    if (!options.empty())
        all_options << options << ' ';
    all_options << "-D GROUP_SIZE=" << wavefront_size_ << " -D WATER_FF_INDEX=" << water_ff_index_;

    //saxs_algorithm_base<float, cl_float4>::load_program("gpu_saxs_2d_block.cl", program_, all_options.str().c_str());
    saxs_algorithm_base<float, cl_float4>::build_program(gpu_saxs_pt_wf_cl, program_, all_options.str());

    // initialize the kernels
    kernel_map_factors_ = cl::Kernel(program_, "map_factors");
    kernel_calc_ = cl::Kernel(program_, "calc_block");
    kernel_vsum_ = cl::Kernel(program_, "block_v_sum");
    kernel_hsum_ = cl::Kernel(program_, "block_h_sum");
    kernel_recalc_ = cl::Kernel(program_, "recalc_block");
    kernel_recalc_with_domains = cl::Kernel(program_, "recalc_block_with_domains");

    state = alg_program_loaded;
}

#if 0
// water form factors Fw(q)=Fw(0)*E(q), E(q)=exp(-bq^2), Fw(0)=3.5, b = 0.23 (estimated)
float water_factor_func(float b, float q)
{
    return float(3.5 * std::sqrt(std::exp(-b * q * q)));
}
#endif // 0

//
//  Sets the arguments for the pending calculations for a set of bodies (a protein).
//  These will not change for a particular protein. Calling this again will reset the algorithm for a new protein.
//  There needs to be a program already loaded.
//      qq              - vector of values for the scattering momentum q
//      factors         - (n_factors x n_qq) packed table of form factors for each value of q
//      n_bodies        - number of bodies in the protein
//
void gpu_saxs_pt_wf_alg::set_args(const std::vector<float> & qq, const std::vector<float> & factors, int n_bodies)
{
    verify(program_loaded(), error::SAXS_PROGRAM_NOT_LOADED);

    n_q_ = int(qq.size());                                  // number of q values
    int n_factors = int(factors.size() / n_q_);             // number of form factors per q value
    n_bodies_aln_ = aligned_num(n_bodies, wavefront_size_); // number of bodies, aligned to the wavefront size
    int n_blocks_aln = aligned_num(n_bodies_aln_ / wavefront_size_, 4); // number of blocks in each dimension, aligned to 4

    /// input OpenCL buffers
    b_qq_.init(context_, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, qq);
    b_factors_.init(context_, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, factors);
    b_bodies_.init(context_, CL_MEM_READ_WRITE, n_bodies_aln_); // aligned to WF_SIZE
    
    body_surfaces_.resize(n_bodies_aln_);
    b_body_surface_.init(context_, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, body_surfaces_); // aligned to WF_SIZE
    
    // calculate the water form factors Fw(q)=Fw(0)*E(q), E(q)=exp(-bq^2), Fw(0)=3.5, b = 0.23 (estimated)
    //std::vector<float> water_factors(n_q_);
    //std::transform(qq.begin(), qq.end(), water_factors.begin(), std::bind1st(std::ptr_fun(water_factor_func), param_b_));
    //b_water_factors_ = cl::Buffer(context_, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, n_q_ * sizeof(float), &water_factors.front());
    
    /// output OpenCL buffer
    b_Iqq_.init(context_, CL_MEM_WRITE_ONLY, n_q_);

    /// internal device buffers
    b_margin_Iqq_.init(context_, CL_MEM_READ_WRITE, n_q_ * n_blocks_aln);   // aligned to 4
    b_body_factors_.init(context_, CL_MEM_READ_WRITE, n_q_ * n_bodies_aln_);    // aligned to WF_SIZE
    b_block_sums_.init(context_, CL_MEM_READ_WRITE, n_q_ * n_blocks_aln * n_blocks_aln);    // aligned to 4x4
    b_block_sums_committed_.init(context_, CL_MEM_READ_WRITE, b_block_sums_.size_); // aligned to 4x4
    b_bodies_committed_.init(context_, CL_MEM_READ_WRITE, b_bodies_.size_);  // aligned to WF_SIZE

    kernel_map_factors_.setArg(ka_mf_n_q, n_q_);
    kernel_map_factors_.setArg(ka_mf_v_bodies, b_bodies_);
    kernel_map_factors_.setArg(ka_mf_n_bodies, n_bodies);
    kernel_map_factors_.setArg(ka_mf_t_factors, b_factors_);
    kernel_map_factors_.setArg(ka_mf_n_factors, n_factors);
    kernel_map_factors_.setArg(ka_mf_t_body_factors, b_body_factors_);

    kernel_calc_.setArg(ka_calc_v_q, b_qq_);
    kernel_calc_.setArg(ka_calc_v_bodies, b_bodies_);
    kernel_calc_.setArg(ka_calc_t_factors, b_factors_);
    kernel_calc_.setArg(ka_calc_n_factors, n_factors);
    kernel_calc_.setArg(ka_calc_t_body_factors, b_body_factors_);
    //!kernel_calc_.setArg(3, param_b_);        // weight of the water contribution - will be filled-in later
    //kernel_calc_.setArg(3, b_water_factors_);
    kernel_calc_.setArg(ka_calc_v_body_surface, b_body_surface_);        // surface area factors - will be filled-in at each iteration
    //kernel_calc_.setArg(ka_calc_water_weight, water_weight_);        // weight of the water contribution - will be filled-in later
    kernel_calc_.setArg(ka_calc_t_block_sums, b_block_sums_);

    kernel_vsum_.setArg(ka_vsum_t_block_sums, b_block_sums_);
    kernel_vsum_.setArg(ka_vsum_t_margin_Iq, b_margin_Iqq_);

    kernel_hsum_.setArg(ka_hsum_t_margin_Iq, b_margin_Iqq_);
    kernel_hsum_.setArg(ka_hsum_n_blocks, n_bodies_aln_ / wavefront_size_);
    kernel_hsum_.setArg(ka_hsum_out_v_Iq, b_Iqq_);

    kernel_recalc_.setArg(ka_recalc_v_q, b_qq_);
    kernel_recalc_.setArg(ka_recalc_v_bodies, b_bodies_);
    kernel_recalc_.setArg(ka_recalc_t_factors, b_factors_);
    kernel_recalc_.setArg(ka_recalc_n_factors, n_factors);
    kernel_recalc_.setArg(ka_recalc_t_body_factors, b_body_factors_);
    //!kernel_recalc_.setArg(3, param_b_);        // weight of the water contribution - will be filled-in later
    //kernel_recalc_.setArg(3, b_water_factors_);
    kernel_recalc_.setArg(ka_recalc_v_body_surface, b_body_surface_);      // surface area factors - will be filled-in at each iteration
    //kernel_recalc_.setArg(ka_recalc_water_weight, water_weight_);        // weight of the water contribution - will be filled-in later
    //kernel_recalc_.setArg(ka_recalc_upd_start_aln, 0);                  // update start - will be filled-in later
    //kernel_recalc_.setArg(ka_recalc_upd_end_aln, n_bodies_aln_);      // update end - will be filled-in later
    kernel_recalc_.setArg(ka_recalc_t_block_sums, b_block_sums_);

    upd_angle_start_aln_ = 0;
    upd_angle_end_aln_ = n_bodies_aln_;

    kernel_recalc_with_domains.setArg(ka_rwd_v_q, b_qq_);
    kernel_recalc_with_domains.setArg(ka_rwd_v_bodies, b_bodies_);
    kernel_recalc_with_domains.setArg(ka_rwd_t_factors, b_factors_);
    kernel_recalc_with_domains.setArg(ka_rwd_n_factors, n_factors);
    kernel_recalc_with_domains.setArg(ka_rwd_t_body_factors, b_body_factors_);
    //!kernel_recalc_with_domains.setArg(3, param_b_);        // weight of the water contribution - will be filled-in later
    //kernel_recalc_with_domains.setArg(3, b_water_factors_);
    kernel_recalc_with_domains.setArg(ka_rwd_v_body_surface, b_body_surface_);      // surface area factors - will be filled-in at each iteration
    //kernel_recalc_with_domains.setArg(ka_rwd_water_weight, water_weight_);        // weight of the water contribution - will be filled-in later
    //kernel_recalc_with_domains.setArg(ka_rwd_upd_start_aln, 0);                  // update start - will be filled-in later
    //kernel_recalc_with_domains.setArg(ka_rwd_upd_end_aln, n_bodies_aln_);      // update end - will be filled-in later
    kernel_recalc_with_domains.setArg(ka_rwd_t_block_sums, b_block_sums_);

    state = alg_args_set;
}

#if 0
//
// Calculates the accessible surface factor for all bodies by using the coordination number (CN).
//
void calc_body_surfaces(const std::vector<cl_float4> & bodies, std::vector<float> & body_surface, float w,int & upd_start, int & upd_length)
{
    for (int i = 0; i < int(bodies.size()); ++i)
    {
        const cl_float4 & body1 = bodies[i];
        int CN = 0;
        for (auto & j : bodies)
        {
            float x = body1.s[0] - j.s[0];
            float y = body1.s[1] - j.s[1];
            float z = body1.s[2] - j.s[2];
            float dist = x*x + y*y + z*z;
            if (dist < 169.0)   // 13*13
                ++CN;
        }
        
        float surface;
        const int a = 48;
        const int b = 16;
        if (CN > a)
            surface = 0.0;
        else if (CN < b)
            surface = 1.0;
        else
            surface = float(a - CN) / (a - b);
        if (body_surface[i] != w * surface)
        {
            body_surface[i] = w * surface;
            // expand the update region if necessary
            if (i < upd_start)
                upd_start = i;
            else if (i >= upd_start + upd_length)
                upd_length = i + 1 - upd_start;
        }
        //body_surface[i] = 0.0;
    }
}
#endif // 0

//
//  Performs the initial calculation of the SAXS curve.
//  The algorithm arguments should have been set before calling this.
//      bodies          - vector of atomic bodies, describing the protein
//      out_Iqq         - buffer to receive the result - I(q) - for each value of q
//
void gpu_saxs_pt_wf_alg::calc_curve(const std::vector<cl_float4> & bodies, std::vector<float> & out_Iqq)
{
    verify(args_set(), error::SAXS_ARGS_NOT_SET);

    // send all bodies to the device
    b_bodies_.write_from(bodies, queue_);
    
    // map factors to bodies for all q
    queue_.enqueueNDRangeKernel(kernel_map_factors_, cl::NullRange, cl::NDRange(n_bodies_aln_), cl::NDRange(wavefront_size_));
    
#if 0
    //int d1 = 0, d2 = bodies.size(); // dummy
    //calc_body_surfaces(bodies, body_surface, water_factor_, d1, d2);
#endif // 0
    // send all body surface area factors to the device
    // Note: only need to send # of bodies, as the 0-padded aligned buffer was sent at initialization
    b_body_surface_.write_from(body_surfaces_, 0, bodies.size(), queue_);

    kernel_calc_.setArg(ka_calc_water_weight, water_weight_);        // weight of the water contribution
    kernel_recalc_.setArg(ka_recalc_water_weight, water_weight_);
    kernel_recalc_with_domains.setArg(ka_rwd_water_weight, water_weight_);
    //     kernel_calc_.setArg(3, param_b_);        // weight of the water contribution
    //     kernel_recalc_.setArg(3, param_b_);      // param b of the water form factors
    //     kernel_recalc_with_domains.setArg(3, param_b_);      // param b of the water form factors
    curve_params_updated_ = false;

    // calculate Debye for all square blocks of size wavefront X wavefront
    queue_.enqueueNDRangeKernel(kernel_calc_, cl::NullRange, cl::NDRange(n_bodies_aln_, n_bodies_aln_/wavefront_size_, n_q_), cl::NDRange(wavefront_size_, 1, 1));
    // sum the Debye results vertically
    queue_.enqueueNDRangeKernel(kernel_vsum_, cl::NullRange, cl::NDRange(n_bodies_aln_/wavefront_size_, n_q_), cl::NullRange);
    // sum the Debye results horizontally
    queue_.enqueueNDRangeKernel(kernel_hsum_, cl::NullRange, cl::NDRange(n_q_), cl::NullRange);

    // retrieve the results I(q)
    out_Iqq.resize(n_q_);
    b_Iqq_.read_to(out_Iqq, queue_);
    
    state = alg_computing;
    
    // initialize the committed state
    last_upd_start_ = 0;
    last_upd_length_ = (unsigned int)bodies.size();
    calc_state_ = calc_pending;
    //std::cout << "PENDING" << std::endl;
    //commit_last_calc();
    calc_state_ = calc_pending;     // set back to pending to maintain the logic
    //std::cout << "PENDING" << std::endl;
}

void gpu_saxs_pt_wf_alg::set_updated_angle_range(unsigned int upd_start, unsigned int upd_end)
{
    upd_angle_start_aln_ = (upd_start / wavefront_size_) * wavefront_size_;
    upd_angle_end_aln_ = aligned_num(upd_end, wavefront_size_);
}

//
//  Recalculates the SAXS curve that was previously calculated.
//  Initial calculation should have been performed before calling this.
//      bodies          - vector of atomic bodies, describing the protein. it will contains both moved and retained bodies.
//      upd_start       - index of the first moved body
//      upd_length      - number of moved bodies
//      out_Iqq         - buffer to receive the result - I(q) - for each value of q
//
void gpu_saxs_pt_wf_alg::recalc_curve(const std::vector<cl_float4> & bodies, int upd_start, int upd_length, std::vector<float> & out_Iqq)
{
    verify(computing(), error::SAXS_NO_INITIAL_CALC);

    last_upd_start_ = upd_start;
    last_upd_length_ = upd_length;
    calc_state_ = calc_pending;
    //std::cout << "PENDING" << std::endl;
    
    if (upd_length > 0)
    {
        // send the moved bodies
        b_bodies_.write_from(bodies, upd_start, upd_length, queue_);
        
        //calc_body_surfaces(bodies, body_surface, water_factor_, upd_start, upd_length);
        // send all body surface area factors to the device
        // Note: only need to send updated bodies, as the 0-padded aligned buffer was sent at initialization
        b_body_surface_.write_from(body_surfaces_, upd_start, upd_length, queue_);
    }
    
    if (curve_params_updated_)
    {
        // new params will force a full recalculation
        upd_start = 0;
        upd_length = int(bodies.size());
        kernel_recalc_.setArg(ka_recalc_water_weight, water_weight_);        // weight of the water contribution
        //!kernel_recalc_.setArg(3, param_b_);             // param b of the water form factors
        curve_params_updated_ = false;
        
        upd_angle_start_aln_ = 0;
        upd_angle_end_aln_ = n_bodies_aln_;
        kernel_recalc_with_domains.setArg(ka_rwd_water_weight, water_weight_);        // weight of the water contribution
        //!kernel_recalc_with_domains.setArg(3, param_b_);             // param b of the water form factors
    }
    
    if (upd_length > 0)
    {
        // align the start index and the length to wavefront_size_
        int upd_start_aln = (upd_start / wavefront_size_) * wavefront_size_;
        int upd_end_aln = aligned_num(upd_start + upd_length, wavefront_size_);
        
        if (upd_angle_start_aln_ > upd_start_aln || upd_angle_end_aln_ < upd_end_aln)
        {
            //std::cout << "  USING: " << upd_angle_start_aln_ << ":" << upd_angle_end_aln_ << " instead of: " << upd_start_aln << ":" << upd_end_aln << std::endl;
            kernel_recalc_with_domains.setArg(ka_rwd_upd_start_aln, upd_angle_start_aln_);
            kernel_recalc_with_domains.setArg(ka_rwd_upd_end_aln, upd_angle_end_aln_);
            // recalculate blocks affected by the update
            queue_.enqueueNDRangeKernel(kernel_recalc_with_domains, cl::NullRange, cl::NDRange(n_bodies_aln_, n_bodies_aln_/wavefront_size_, n_q_), cl::NDRange(wavefront_size_, 1, 1));
        }
        else
        {
            kernel_recalc_.setArg(ka_recalc_upd_start_aln, upd_start_aln);
            kernel_recalc_.setArg(ka_recalc_upd_end_aln, upd_end_aln);
            // recalculate blocks affected by the update
            queue_.enqueueNDRangeKernel(kernel_recalc_, cl::NullRange, cl::NDRange(n_bodies_aln_, n_bodies_aln_/wavefront_size_, n_q_), cl::NDRange(wavefront_size_, 1, 1));
        }
        
        // sum the Debye results vertically
        queue_.enqueueNDRangeKernel(kernel_vsum_, cl::NullRange, cl::NDRange(n_bodies_aln_/wavefront_size_, n_q_), cl::NullRange);
        // sum the Debye results horizontally
        queue_.enqueueNDRangeKernel(kernel_hsum_, cl::NullRange, cl::NDRange(n_q_), cl::NullRange);

        // retrieve the results I(q)
        out_Iqq.resize(n_q_);
        b_Iqq_.read_to(out_Iqq, queue_);
    }
}

void gpu_saxs_pt_wf_alg::save_block_cache(unsigned int upd_start, unsigned int upd_end)
{
    verify(computing(), error::SAXS_NO_INITIAL_CALC);
    
    queue_.enqueueBarrier();
    b_block_sums_.copy_to(b_block_sums_committed_, queue_);
    queue_.enqueueBarrier();
}

void gpu_saxs_pt_wf_alg::restore_block_cache(unsigned int upd_start, unsigned int upd_end)
{
    verify(computing(), error::SAXS_NO_INITIAL_CALC);
    
    queue_.enqueueBarrier();
    b_block_sums_committed_.copy_to(b_block_sums_, queue_);
    queue_.enqueueBarrier();
}

void gpu_saxs_pt_wf_alg::commit_last_calc()
{
    return;

    //verify(calc_state_ == calc_pending, SAXS_NO_PENDING_CALC);
    if (calc_state_ != calc_pending)
        return;
    
    queue_.enqueueBarrier();
    if (last_upd_length_ > 0)
        b_bodies_.copy_to(b_bodies_committed_, last_upd_start_, last_upd_length_, queue_);
    
    b_block_sums_.copy_to(b_block_sums_committed_, queue_);
    queue_.enqueueBarrier();
    
    calc_state_ = calc_committed;
    std::cout << "COMMITTED" << std::endl;
}

void gpu_saxs_pt_wf_alg::revert_last_calc()
{
    return;

    //verify(calc_state_ == calc_pending, SAXS_NO_PENDING_CALC);
    if (calc_state_ != calc_pending)
        return;
    
    queue_.enqueueBarrier();
    if (last_upd_length_ > 0)
        b_bodies_committed_.copy_to(b_bodies_, last_upd_start_, last_upd_length_, queue_);
    
    b_block_sums_committed_.copy_to(b_block_sums_, queue_);
    queue_.enqueueBarrier();
    
    calc_state_ = calc_reverted;
    std::cout << "REVERTED" << std::endl;
}

}   // namespace
