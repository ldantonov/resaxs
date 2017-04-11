///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2013 Lubo Antonov
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

#ifndef ACCSAXS_GPU_SAXS_BL_WF_HPP
#define ACCSAXS_GPU_SAXS_BL_WF_HPP

#include "accsaxs.hpp"
#include "saxs_algorithm_base.hpp"

namespace accsaxs {

///////////////////////////////////////////////////////////////////////////////
//  SAXS Page-Tile algorithm with hydration layer for GPUs, using a square block and page decomposition and local memory.
//
//      Specialized to float, but can be templetized for double, etc.
//
class gpu_saxs_pt_wf_alg : public saxs_algorithm_base<float, cl_float4>
{
    typedef saxs_algorithm_base<float, cl_float4> base;

public:
    gpu_saxs_pt_wf_alg();

    void initialize(const std::vector<dev_id> &dev_ids, unsigned int water_ff_index);

    //  OVERRIDE
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
    virtual void load_program(const std::string & options, int wavefront_size = SAXS_NV_FERMI_WF_SIZE);

    //  OVERRIDE
    //  Sets the arguments for the pending calculations for a set of bodies (a protein).
    //  These will not change for a particular protein. Calling this again will reset the algorithm for a new protein.
    //  There needs to be a program already loaded.
    //      qq              - vector of values for the scattering momentum q
    //      factors         - (n_factors x n_qq) packed table of form factors for each value of q
    //      n_bodies        - number of bodies in the protein
    //
    virtual void set_args(const std::vector<float> & qq, const std::vector<float> & factors, int n_bodies);

    //  OVERRIDE
    //  Performs the initial calculation of the SAXS curve.
    //  The algorithm arguments should have been set before calling this.
    //      bodies          - vector of atomic bodies, describing the protein
    //      out_Iqq         - buffer to receive the result - I(q) - for each value of q
    //  
    virtual void calc_curve(const std::vector<cl_float4> & bodies, std::vector<float> & out_Iqq);

    //  OVERRIDE
    //  Recalculates the SAXS curve that was previously calculated.
    //  Initial calculation should have been performed before calling this.
    //      bodies          - vector of atomic bodies, describing the protein. it will contains both moved and retained bodies.
    //      upd_start       - index of the first moved body
    //      upd_length      - number of moved bodies
    //      out_Iqq         - buffer to receive the result - I(q) - for each value of q
    //  
    virtual void recalc_curve(const std::vector<cl_float4> & bodies, int upd_start, int upd_length, std::vector<float> & out_Iqq);

    virtual ~gpu_saxs_pt_wf_alg() {};
    
//     void set_param_b(float b) {
//         if (param_b_ != b)
//         {
//             param_b_ = b;
//             curve_params_updated_ = true;
//         }
//     }
    void set_water_weight(float w) {
        if (water_weight_ != w)
        {
            water_weight_ = w;
            curve_params_updated_ = true;
        }
    }
    
    std::vector<float> & get_body_surfaces() {
        return body_surfaces_;
    }
    
    void set_updated_angle_range(unsigned int upd_start, unsigned int upd_end);
    
    void save_block_cache(unsigned int upd_start, unsigned int upd_end);
    void restore_block_cache(unsigned int upd_start, unsigned int upd_end);
    
    void commit_last_calc();
    void revert_last_calc();

private:
    int wavefront_size_;                // wavefront size to use for kernel compilation and data alignment and decomposition
    int n_q_;                           // number of q vector elements
    int n_bodies_aln_;                  // body count aligned to the wavefront size

    cl::Program program_;

    cl::Kernel kernel_map_factors_;     // maps all from factors to the bodies for each value of q
    cl::Kernel kernel_calc_;            // calculates the double sum Debye for a square block with side equal to wavefront
    cl::Kernel kernel_vsum_;            // sums a vertical column of blocks into the margin
    cl::Kernel kernel_hsum_;            // sums the margin sums for a value of q
    cl::Kernel kernel_recalc_;          // recalculates the I(q) vector with changed body positions
    cl::Kernel kernel_recalc_with_domains;  // recalculates the I(q) vector with changed angles between putative domains

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
        ka_calc_v_bodies,
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
        ka_recalc_v_bodies,
        ka_recalc_t_factors,
        ka_recalc_n_factors,
        ka_recalc_t_body_factors,
        ka_recalc_v_body_surface,
        ka_recalc_water_weight,
        ka_recalc_upd_start_aln,
        ka_recalc_upd_end_aln,
        ka_recalc_t_block_sums
    };

    enum recalc_with_domains_kernel_args
    {
        ka_rwd_v_q,
        ka_rwd_v_bodies,
        ka_rwd_t_factors,
        ka_rwd_n_factors,
        ka_rwd_t_body_factors,
        ka_rwd_v_body_surface,
        ka_rwd_water_weight,
        ka_rwd_upd_start_aln,
        ka_rwd_upd_end_aln,
        ka_rwd_t_block_sums
    };

private:
    // input OpenCL buffers
    Buffer<float> b_body_surface_;          // surface area factors for each body (vector:bodies)
    //cl::Buffer b_water_factors_;          // water form factors for each value of q (vector:q)
    
    // internal device buffers 
    Buffer<float> b_margin_Iqq_;            // margin sums for each column of blocks
    Buffer<float> b_body_factors_;          // n_q_ pages of form factors mapped to bodies
    Buffer<float> b_block_sums_;            // n_q_ pages of square block sum tables
    Buffer<float> b_block_sums_committed_;  // the last accepted block sums
    cl::Event e_block_sums_copy_;           // event for completion of block sum commit/revert
    Buffer<cl_float4> b_bodies_committed_;  // the last accepted body positions
    cl::Event e_bodies_copy;                // event for completion of bodies commit/revert
    
    std::vector<float> body_surfaces_;      // surface area factors for each body
    float water_weight_;                    // weight factor for the water layer
    bool curve_params_updated_;             // flag to track if the curve parameters (water weight factor, b) have been updated, which will cause a full recalculation
    //!float param_b_;                       // parameter b of the Gaussian approximation function for the water layer
    unsigned int water_ff_index_;           // index of the water form factors into the form factor table
    int upd_angle_start_aln_;               // the starting body for updated angles; aligned to group size
    int upd_angle_end_aln_;                 // the end of the stretch of bodies with updated angles; aligned to group size
    
    unsigned int last_upd_start_;           // starting index of the updated bodies in the last recalc
    unsigned int last_upd_length_;          // length of the updated bodies in the last recalc

    enum calc_state
    {
        calc_committed = 0,
        calc_pending,
        calc_reverted
    } calc_state_;
    
    enum calc_error
    {
        SAXS_NO_PENDING_CALC = -100
    };

    enum init_state
    {
        init_blank,
        init_done
    } init_state_;
};

}   // namespace

#endif // ACCSAXS__GPU_SAXS_BL_WF_HPP

