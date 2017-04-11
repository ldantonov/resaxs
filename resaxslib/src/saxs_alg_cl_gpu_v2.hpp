///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2015 Lubo Antonov
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

#ifndef RE_SAXS_ALG_CL_GPU_V2_HPP
#define RE_SAXS_ALG_CL_GPU_V2_HPP

#include <functional>
#include <memory>
#include "saxs_algorithm.hpp"
#include "re_cl_algorithm_base.hpp"
#include "rsasa_algorithm.hpp"

namespace resaxs {
    namespace algorithm {

        ///////////////////////////////////////////////////////////////////////////////
        //  SAXS Page-Tile algorithm with hydration layer for GPUs, using a square block and page decomposition and local memory.
        //
        //      Specialized to float, but can be templetized for double, etc.
        //
        template <typename FLT_T>
        class saxs_cl_gpu_v2 : public i_saxs<FLT_T, cl_base<FLT_T>>
        {
            typedef typename real_type<FLT_T>::real4 real4;

            typedef cl_base<FLT_T> alg_base;
            typedef i_saxs<FLT_T, alg_base> base;

            class atomic_ff_params : public form_factor_params<FLT_T>
            {
                std::function<void()> call_set_ext_dirty_;
                std::function<void(const form_factor_params<FLT_T> &)> exp_factor_changed_sink_;
                saxs_cl_gpu_v2 & owner_;
                const std::vector<FLT_T> *qq_;

                virtual void on_dirty_set() override
                {
                    call_set_ext_dirty_();
                }

            public:
                atomic_ff_params(saxs_cl_gpu_v2 &owner, std::function<void(const form_factor_params<FLT_T> &)> exp_factor_changed_sink) :
                    call_set_ext_dirty_([]() {}), exp_factor_changed_sink_(exp_factor_changed_sink),
                    owner_(owner), qq_(NULL) {}

                void initialize(const std::vector<FLT_T> &qq, std::function<void()> call_set_ext_dirty = []() {});

                virtual void set_expansion_factor(FLT_T c) override;
            };

            class common_ocl_params : public ocl_exec_params
            {
                std::function<void(const ocl_exec_params &)> ocl_changed_event_sink_;
            public:
                common_ocl_params() = default;
                void set_ocl_changed_event_sink(std::function<void(const ocl_exec_params &)> ocl_changed_event_sink)
                {
                    ocl_changed_event_sink_ = ocl_changed_event_sink;
                }

                virtual void set_devices(const std::vector<dev_id> &devices) override
                {
                    ocl_exec_params::set_devices(devices);

                    ocl_changed_event_sink_(*this);
                }
            };

            /// Parameters for the atomic saxs OpenCL algorithm.
            ///
            struct gpu_pt_wf_params : public saxs_params<FLT_T>, private implicit_water_params<FLT_T>, private common_ocl_params
            {
            private:
                saxs_cl_gpu_v2 & owner_;
                atomic_ff_params ff_params_;
                pivot_params pivot_params_;

                typedef i_rsasa<FLT_T, FLT_T, alg_base> rsasa_class;
                typedef rsasa<FLT_T, FLT_T> rsasa_alg_factory;
                std::unique_ptr<rsasa_class> rsasa_alg;

                virtual void on_dirty_set()
                {
                    saxs_params<FLT_T>::set_dirty();
                }

                using saxs_params<FLT_T>::set_dirty;

            public:
                void on_ocl_changed(const ocl_exec_params &ocl_params)
                {
                    // If there was a RSASA alg initialized, reinitialize it. Otherwise - delay.
                    // This will avoid allocating OpenCL resources if there is no need for RSASA.
                    if (rsasa_alg->initialized())
                    {
                        typename rsasa_alg_factory::dot_spheres_cl_params rsasa_params(0.5f, 1.8f, ocl_params.devices_, this->n_bodies_);
                        rsasa_alg->initialize(&rsasa_params);
                    }
                }

            public:
                gpu_pt_wf_params(saxs_cl_gpu_v2 &owner);

                bool is_dirty() const { return saxs_params<FLT_T>::is_dirty(); }

                virtual void initialize(const std::vector<FLT_T> &qq, const std::vector<real4> &bodies,
                    const std::vector<dev_id> &devices, unsigned int workgroup_size = SAXS_NV_FERMI_WF_SIZE);

                virtual form_factor_params<FLT_T> & get_ff_params()
                {
                    return ff_params_;
                }
                virtual implicit_water_params<FLT_T> & get_implicit_water_params()
                {
                    return *this;
                }
                virtual ocl_exec_params & get_ocl_exec_params()
                {
                    return *this;
                }
                virtual pivot_params & get_pivot_params()
                {
                    return pivot_params_;
                }

                void clear_dirty()
                {
                    saxs_params<FLT_T>::clear_dirty();
                    implicit_water_params<FLT_T>::clear_dirty();
                    ocl_exec_params::clear_dirty();
                    ff_params_.clear_dirty();
                }

                //
                //  Updates the calculated RSASA using the new body positions.
                //
                void update_rsasa(const std::vector<real4> &bodies);

                bool has_rsasa() const { return rsasa_alg->initialized(); }
            };

            typedef gpu_pt_wf_params params_type;

            void on_exp_factor_changed(const form_factor_params<FLT_T> & ff_params);
            void on_ocl_device_changed(const ocl_exec_params &ocl_params);

        public:
            saxs_cl_gpu_v2();

            virtual saxs_params<FLT_T> & access_params() override;

            virtual void initialize() override;

            //
            //  Performs the initial calculation of the SAXS curve.
            //  The algorithm arguments should have been set before calling this.
            //      bodies          - vector of atomic bodies, describing the protein
            //      out_Iqq         - buffer to receive the result - I(q) - for each value of q
            //
            virtual void calc_curve(const std::vector<real4> &bodies, std::vector<FLT_T> &out_Iqq) override;

            //
            //  Recalculates the SAXS curve that was previously calculated.
            //  Initial calculation should have been performed before calling this.
            //      bodies          - vector of atomic bodies, describing the protein. it will contains both moved and retained bodies.
            //      upd_start       - index of the first moved body
            //      upd_length      - number of moved bodies
            //      out_Iqq         - buffer to receive the result - I(q) - for each value of q
            //
            virtual void recalc_curve(const std::vector<real4> &bodies, unsigned int upd_start, unsigned int upd_length, std::vector<FLT_T> &out_Iqq) override;

            //
            // Commits the last calculation, i.e. the state is checkpointed and can be restored by a revert operation.
            //
            virtual void commit() override;

            //
            // Reverts the last calculation, i.e. restores the state to the last committed state.
            //
            virtual void revert() override;

        private:

            void load_program();
            void set_args();
            void set_kernel_args();

            void update_bodies(const std::vector<real4> &bodies, unsigned int &&upd_start, unsigned int &&upd_length);
            void update_bodies(const std::vector<real4> &bodies);

            bool xaction() const { return b_saved_bodies_; }    // has there been a transaction committed
            void init_xaction();    // initialize the transaction mechanism

                                    // copy the buffer if it has changed, clear the flag
            template <typename U>
            void copy_if_changed(Buffer<U> &from, Buffer<U> &to, bool &changed);

        private:
            params_type params_;


            int n_q_;                           // number of q vector elements
            int n_bodies_aln_;                  // body count aligned to the wavefront size
            std::vector<FLT_T> radii_map_;      // map of atomic radii
            FLT_T avg_radius_;                  // average radius of all bodies

            cl::Program program_;

            cl::Kernel kernel_map_factors_;     // maps all from factors to the bodies for each value of q
            cl::Kernel kernel_calc_;            // calculates the double sum Debye for a square block with side equal to wavefront
            cl::Kernel kernel_vsum_;            // sums a vertical column of blocks into the margin
            cl::Kernel kernel_hsum_;            // sums the margin sums for a value of q
            cl::Kernel kernel_recalc_;          // recalculates the I(q) vector with changed body positions
            cl::Kernel kernel_recalc_with_pivots_;          // recalculates the I(q) vector with changed body positions in a pivot range
            cl::Kernel kernel_calc_distances_;

            enum map_factors_kernel_args
            {
                ka_mf_n_q,
                ka_mf_v_bodies,
                ka_mf_n_bodies,
                ka_mf_t_factors,
                ka_mf_n_factors,
                ka_mf_t_body_factors
            };

            enum calc_kernel_args
            {
                ka_calc_v_q,
                ka_calc_t_dists,
                ka_calc_t_factors,
                ka_calc_n_factors,
                ka_calc_t_body_factors,
                ka_calc_v_body_surface,
                ka_calc_water_weight,
                ka_calc_t_block_sums
            };

            enum vsum_kernel_args
            {
                ka_vsum_t_block_sums,
                ka_vsum_t_margin_Iq
            };

            enum hsum_kernel_args
            {
                ka_hsum_t_margin_Iq,
                ka_hsum_n_blocks,
                ka_hsum_out_v_Iq
            };

            enum recalc_kernel_args
            {
                ka_recalc_v_q,
                ka_recalc_t_dists,
                ka_recalc_t_factors,
                ka_recalc_n_factors,
                ka_recalc_t_body_factors,
                ka_recalc_v_body_surface,
                ka_recalc_water_weight,
                ka_recalc_upd_start_aln,
                ka_recalc_upd_end_aln,
                ka_recalc_t_block_sums
            };

            enum calc_distances_args
            {
                ka_calc_dist_v_bodies,
                ka_calc_dist_t_distances
            };

        private:
            // Input OpenCL buffers
            Buffer<FLT_T> b_qq_;
            Buffer<FLT_T> b_factors_;
            Buffer<real4> b_bodies_;
            Buffer<FLT_T> b_Iqq_;

            Buffer<FLT_T> b_body_surface_;          // surface area factors for each body (vector:bodies)

            // internal device buffers 
            Buffer<FLT_T> b_margin_Iqq_;            // margin sums for each column of blocks
            Buffer<FLT_T> b_body_factors_;          // n_q_ pages of form factors mapped to bodies
            Buffer<FLT_T> b_block_sums_;            // n_q_ pages of square block sum tables
            Buffer<FLT_T> b_distances_;

            Buffer<real4> b_saved_bodies_;          // the last accepted body positions
            Buffer<FLT_T> b_saved_block_sums_;      // the last accepted block sums
            Buffer<FLT_T> b_saved_body_surface_;    // the last accepted surface area
            Buffer<FLT_T> b_saved_factors_;
            Buffer<FLT_T> b_saved_body_factors_;

            // Flags that track which buffers have changed since the last commit
            struct checkpoint_bits
            {
                bool sum;
                bool bodies;
                bool rsasa;
                bool factors;
                bool body_factors;
            } changed_buffers;
        };

    } // namespace algorithm
} // namespace

#endif // RE_SAXS_ALG_CL_GPU_V2_HPP
