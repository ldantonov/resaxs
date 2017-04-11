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

#include "gpu_saxs_2d_block_local.hpp"

#include <sstream>

#include "../include/utils.hpp"

namespace accsaxs {

#include "gpu_saxs_2d_block.cl"

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
void gpu_saxs_2d_block_local_alg::load_program(const std::string & options, int wavefront_size)
{
    wavefront_size_ = wavefront_size == 0 ? SAXS_NV_FERMI_WF_SIZE : wavefront_size;

    // add the wavefront size as a parameter to the program
    std::ostringstream all_options;
    if (!options.empty())
        all_options << options << ' ';
    all_options << "-D GROUP_SIZE=" << wavefront_size_;

    //saxs_algorithm_base<float, cl_float4>::load_program("gpu_saxs_2d_block.cl", program_, all_options.str().c_str());
    saxs_algorithm_base<float, cl_float4>::build_program(gpu_saxs_2d_block_cl, program_, all_options.str().c_str());

    // initialize the kernels
    kernel_map_factors_ = cl::Kernel(program_, "map_factors");
    kernel_calc_ = cl::Kernel(program_, "calc_block");
    kernel_vsum_ = cl::Kernel(program_, "block_v_sum");
    kernel_hsum_ = cl::Kernel(program_, "block_h_sum");
    kernel_recalc_ = cl::Kernel(program_, "recalc_block");

    state = alg_program_loaded;
}

//
//  Sets the arguments for the pending calculations for a set of bodies (a protein).
//  These will not change for a particular protein. Calling this again will reset the algorithm for a new protein.
//  There needs to be a program already loaded.
//      qq              - vector of values for the scattering momentum q
//      factors         - (n_factors x n_qq) packed table of form factors for each value of q
//      n_bodies        - number of bodies in the protein
//
void gpu_saxs_2d_block_local_alg::set_args(const std::vector<float> & qq, const std::vector<float> & factors, int n_bodies)
{
    verify(program_loaded(), error::SAXS_PROGRAM_NOT_LOADED);

    n_q_ = int(qq.size());                                  // number of q values
    int n_factors = int(factors.size() / n_q_);             // number of form factors per q value
    n_bodies_aln_ = aligned_num(n_bodies, wavefront_size_); // number of bodies, aligned to the wavefront size
    int n_blocks_aln = aligned_num(n_bodies_aln_ / wavefront_size_, 4); // number of blocks in each dimention, aligned to 4

    // input OpenCL buffers
    b_qq_.init(context_, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, qq);
    b_factors_.init(context_, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, factors);
    b_bodies_.init(context_, CL_MEM_READ_WRITE, n_bodies_aln_); // aligned to WF_SIZE
    // output OpenCL buffer
    b_Iqq_.init(context_, CL_MEM_WRITE_ONLY, n_q_);
    

    // internal device buffers
    b_margin_Iqq_.init(context_, CL_MEM_READ_WRITE, n_q_ * n_blocks_aln);   // aligned to 4
    b_body_factors_.init(context_, CL_MEM_READ_WRITE, n_q_ * n_bodies_aln_);    // aligned to WF_SIZE
    b_block_sums_.init(context_, CL_MEM_READ_WRITE, n_q_ * n_blocks_aln * n_blocks_aln);    // aligned to 4x4

    // set the kernel arguments
    kernel_map_factors_.setArg(0, n_q_);
    kernel_map_factors_.setArg(1, b_bodies_);
    kernel_map_factors_.setArg(2, n_bodies);
    kernel_map_factors_.setArg(3, b_factors_);
    kernel_map_factors_.setArg(4, n_factors);
    kernel_map_factors_.setArg(5, b_body_factors_);

    kernel_calc_.setArg(0, b_qq_);
    kernel_calc_.setArg(1, b_bodies_);
    kernel_calc_.setArg(2, b_body_factors_);
    kernel_calc_.setArg(3, b_block_sums_);

    kernel_vsum_.setArg(0, b_block_sums_);
    kernel_vsum_.setArg(1, b_margin_Iqq_);

    kernel_hsum_.setArg(0, b_margin_Iqq_);
    kernel_hsum_.setArg(1, n_bodies_aln_ / wavefront_size_);
    kernel_hsum_.setArg(2, b_Iqq_);

    kernel_recalc_.setArg(0, b_qq_);
    kernel_recalc_.setArg(1, b_bodies_);
    kernel_recalc_.setArg(2, b_body_factors_);
    //kernel_recalc_.setArg(3, 0);                  // update start - will be filled-in later
    //kernel_recalc_.setArg(4, n_bodies_aln_);      // update end - will be filled-in later
    kernel_recalc_.setArg(5, b_block_sums_);

    state = alg_args_set;
}

//
//  Performs the initial calculation of the SAXS curve.
//  The algorithm arguments should have been set before calling this.
//      bodies          - vector of atomic bodies, describing the protein
//      out_Iqq         - buffer to receive the result - I(q) - for each value of q
//
void gpu_saxs_2d_block_local_alg::calc_curve(const std::vector<cl_float4> & bodies, std::vector<float> & out_Iqq)
{
    verify(args_set(), error::SAXS_ARGS_NOT_SET);

    // send all bodies to the device
    b_bodies_.write_from(bodies, queue_);
    
    // map factors to bodies for all q
    queue_.enqueueNDRangeKernel(kernel_map_factors_, cl::NullRange, cl::NDRange(n_bodies_aln_), cl::NDRange(wavefront_size_));
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
}

//
//  Recalculates the SAXS curve that was previously calculated.
//  Initial calculation should have been performed before calling this.
//      bodies          - vector of atomic bodies, describing the protein. it will contains both moved and retained bodies.
//      upd_start       - index of the first moved body
//      upd_length      - number of moved bodies
//      out_Iqq         - buffer to receive the result - I(q) - for each value of q
//
void gpu_saxs_2d_block_local_alg::recalc_curve(const std::vector<cl_float4> & bodies, int upd_start, int upd_length, std::vector<float> & out_Iqq)
{
    verify(computing(), error::SAXS_NO_INITIAL_CALC);

    // send the moved bodies
    b_bodies_.write_from(bodies, upd_start, upd_length, queue_);
    
    // align the start index and the length to wavefront_size_
    int upd_start_aln = (upd_start / wavefront_size_) * wavefront_size_;
    int upd_length_aln = aligned_num(upd_start + upd_length, wavefront_size_) - upd_start_aln;
    kernel_recalc_.setArg(3, upd_start_aln);
    kernel_recalc_.setArg(4, upd_start_aln + upd_length_aln);
    // recalculate blocks affected by the update
    queue_.enqueueNDRangeKernel(kernel_recalc_, cl::NullRange, cl::NDRange(n_bodies_aln_, n_bodies_aln_/wavefront_size_, n_q_), cl::NDRange(wavefront_size_, 1, 1));
    // sum the Debye results vertically
    queue_.enqueueNDRangeKernel(kernel_vsum_, cl::NullRange, cl::NDRange(n_bodies_aln_/wavefront_size_, n_q_), cl::NullRange);
    // sum the Debye results horizontally
    queue_.enqueueNDRangeKernel(kernel_hsum_, cl::NullRange, cl::NDRange(n_q_), cl::NullRange);

    // retrieve the results I(q)
    out_Iqq.resize(n_q_);
    b_Iqq_.read_to(out_Iqq, queue_);
}

}   // namespace
