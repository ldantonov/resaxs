#ifndef ACCSAXS_GPU_SAXS_2D_HPP
#define ACCSAXS_GPU_SAXS_2D_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011 Lubo Antonov
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

#include "../include/accsaxs.hpp"
#include "../include/saxs_algorithm_base.hpp"

namespace accsaxs {

///////////////////////////////////////////////////////////////////////////////
//  SAXS algorithm for GPUs, using a column and page decomposition without local memory.
//
//      Specialized to float, but can be templetized for double, etc.
//
class gpu_saxs_2d_alg : public saxs_algorithm_base<float, cl_float4>
{
public:
    gpu_saxs_2d_alg() : wavefront_size_(SAXS_NV_FERMI_WF_SIZE) {}

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
    virtual void load_program(const std::string & options, int wavefront_size = SAXS_NV_FERMI_WF_SIZE);

    //
    //  Sets the arguments for the pending calculations for a set of bodies (a protein).
    //  These will not change for a particular protein. Calling this again will reset the algorithm for a new protein.
    //  There needs to be a program already loaded.
    //      qq              - vector of values for the scattering momentum q
    //      factors         - (n_factors x n_qq) packed table of form factors for each value of q
    //      n_bodies        - number of bodies in the protein
    //
    virtual void set_args(const std::vector<float> & qq, const std::vector<float> & factors, int n_bodies);

    //
    //  Performs the initial calculation of the SAXS curve.
    //  The algorithm arguments should have been set before calling this.
    //      bodies          - vector of atomic bodies, describing the protein
    //      out_Iqq         - buffer to receive the result - I(q) - for each value of q
    //  
    virtual void calc_curve(const std::vector<cl_float4> & bodies, std::vector<float> & out_Iqq);

    //
    //  Recalculates the SAXS curve that was previously calculated.
    //  Initial calculation should have been performed before calling this.
    //      bodies          - vector of atomic bodies, describing the protein. it will contains both moved and retained bodies.
    //      upd_start       - index of the first moved body
    //      upd_length      - number of moved bodies
    //      out_Iqq         - buffer to receive the result - I(q) - for each value of q
    //  
    virtual void recalc_curve(const std::vector<cl_float4> & bodies, int upd_start, int upd_length, std::vector<float> & out_Iqq);

    virtual ~gpu_saxs_2d_alg() {};

private:
    int wavefront_size_;                // wavefront size to use for kernel compilation and data alignment and decomposition
    int n_q_;                           // number of q vector elements
    int n_bodies_aln_;                  // body count aligned to the wavefront size
    
    cl::Program m_program;

    cl::Kernel m_kernel_map_factors;
    cl::Kernel m_kernel_save_prev_bodies;
    cl::Kernel m_kernel_recalc;
    cl::Kernel m_kernel_adjust;
    cl::Kernel m_kernel_sum_phase1;
    cl::Kernel m_kernel_sum_phase2;

    cl::Kernel m_kernel_margin_phase2;
    cl::Kernel m_kernel;
    cl::Kernel m_kernel_sum;

private:
    cl::Buffer b_body_factors;
    cl::Buffer b_prev_bodies;
    cl::Buffer b_sum;

    cl::Buffer b_int_terms;
    cl::Buffer b_margin_Iq;
};

}   // namespace

#endif  // ACCSAXS_GPU_SAXS_2D_HPP