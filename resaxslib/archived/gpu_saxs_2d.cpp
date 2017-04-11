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

#include "gpu_saxs_2d.hpp"

#include <sstream>

#include "../include/utils.hpp"

namespace accsaxs {

#include "gpu_saxs_2d.cl"

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
void gpu_saxs_2d_alg::load_program(const std::string & options, int wavefront_size)
{
    wavefront_size_ = wavefront_size == 0 ? SAXS_NV_FERMI_WF_SIZE : wavefront_size;

    // add the wavefront size as a parameter to the program
    std::ostringstream all_options;
    if (!options.empty())
        all_options << options << ' ';
    all_options << "-D GROUP_SIZE=" << wavefront_size_;

    //saxs_algorithm_base<float, cl_float4>::load_program("gpu_saxs_2d.cl", m_program, all_options.str().c_str());
    saxs_algorithm_base<float, cl_float4>::build_program(gpu_saxs_2d_cl, m_program, all_options.str().c_str());

    // initialize the kernels
    m_kernel_map_factors = cl::Kernel(m_program, "map_factors");
    m_kernel = cl::Kernel(m_program, "margin_sum_small");
    m_kernel_sum_phase1 = cl::Kernel(m_program, "sum_local_phase1");
    m_kernel_sum = cl::Kernel(m_program, "sum_local_phase2");
    m_kernel_save_prev_bodies = cl::Kernel(m_program, "save_prev_bodies");
    m_kernel_recalc = cl::Kernel(m_program, "recalc_margins_sum");
    m_kernel_adjust = cl::Kernel(m_program, "adjust_Iq");

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
void gpu_saxs_2d_alg::set_args(const std::vector<float> & qq, const std::vector<float> & factors, int n_bodies)
{
    verify(program_loaded(), error::SAXS_PROGRAM_NOT_LOADED);

    n_q_ = int(qq.size());                                  // number of q values
    int n_factors = int(factors.size() / n_q_);             // number of form factors per q value
    n_bodies_aln_ = aligned_num(n_bodies, wavefront_size_); // number of bodies, aligned to the wavefront size
    //int n_blocks_aln = aligned_num(n_bodies_aln_ / wavefront_size_, 4); // number of blocks in each dimention, aligned to 4

    // input OpenCL buffers
    b_qq_.init(context_, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, qq);
    b_factors_.init(context_, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, factors);
    b_bodies_.init(context_, CL_MEM_READ_WRITE, n_bodies_aln_); // aligned to WF_SIZE
    // output OpenCL buffer
    b_Iqq_.init(context_, CL_MEM_WRITE_ONLY, n_q_);
    

    b_margin_Iq = cl::Buffer(context_, CL_MEM_READ_WRITE, n_q_ * n_bodies_aln_ * sizeof(float));    //aligned to GROUP_SIZE

    /*m_kernel_sum.setArg(0, b_margin_Iq);
    m_kernel_sum.setArg(1, n_bodies);
    m_kernel_sum.setArg(2, b_Iq);*/

    // internal device buffers
    b_body_factors = cl::Buffer(context_, CL_MEM_READ_WRITE, n_q_ * n_bodies_aln_ * sizeof(float)); // aligned to WF_SIZE
    b_prev_bodies = cl::Buffer(context_, CL_MEM_READ_WRITE, n_bodies_aln_ * sizeof(cl_float4));    // aligned to WF_SIZE
    b_sum = cl::Buffer(context_, CL_MEM_READ_WRITE, n_bodies_aln_ / wavefront_size_ * n_q_ * sizeof(cl_float));

    b_int_terms = cl::Buffer(context_, CL_MEM_READ_WRITE, n_q_ * n_bodies_aln_ * n_bodies_aln_ / wavefront_size_ * sizeof(cl_float));    // aligned to WF_SIZE

    m_kernel_map_factors.setArg(0, n_q_);
    m_kernel_map_factors.setArg(1, b_bodies_);
    m_kernel_map_factors.setArg(2, n_bodies);
    m_kernel_map_factors.setArg(3, b_factors_);
    m_kernel_map_factors.setArg(4, n_factors);
    m_kernel_map_factors.setArg(5, b_body_factors);
    m_kernel_map_factors.setArg(6, b_prev_bodies);

    m_kernel.setArg(0, b_qq_);
    //m_kernel.setArg(1, n_q);
    m_kernel.setArg(1, b_bodies_);
    m_kernel.setArg(2, n_bodies);
    m_kernel.setArg(3, b_body_factors);
    m_kernel.setArg(4, b_margin_Iq);
    //m_kernel.setArg(4, b_sum);

    m_kernel_sum_phase1.setArg(0, b_margin_Iq);
    m_kernel_sum_phase1.setArg(1, b_sum);

    m_kernel_sum.setArg(0, b_sum);
    m_kernel_sum.setArg(1, n_bodies_aln_ / wavefront_size_);
    m_kernel_sum.setArg(2, b_Iqq_);

    //m_kernel.setArg(6, n_bodies_aln * sizeof(cl_float4), NULL);

    m_kernel_save_prev_bodies.setArg(0, b_bodies_);
    //m_kernel_save_prev_bodies.setArg(1, 0);
    m_kernel_save_prev_bodies.setArg(2, b_prev_bodies);

    m_kernel_recalc.setArg(0, b_qq_);
    //m_kernel_recalc.setArg(1, n_q);
    m_kernel_recalc.setArg(1, b_bodies_);
    //m_kernel_recalc.setArg(2, n_bodies);
    m_kernel_recalc.setArg(2, b_body_factors);
    //m_kernel_recalc.setArg(3, 0);
    //m_kernel_recalc.setArg(4, n_bodies);
    m_kernel_recalc.setArg(5, b_prev_bodies);
    m_kernel_recalc.setArg(6, b_margin_Iq);

    m_kernel_adjust.setArg(0, b_margin_Iq);
    m_kernel_adjust.setArg(1, n_bodies);
    m_kernel_adjust.setArg(2, b_Iqq_);

    state = alg_args_set;
}

//
//  Performs the initial calculation of the SAXS curve.
//  The algorithm arguments should have been set before calling this.
//      bodies          - vector of atomic bodies, describing the protein
//      out_Iqq         - buffer to receive the result - I(q) - for each value of q
//
void gpu_saxs_2d_alg::calc_curve(const std::vector<cl_float4> & bodies, std::vector<float> & out_Iqq)
{
    verify(args_set(), error::SAXS_ARGS_NOT_SET);

    // send all bodies to the device
    b_bodies_.write_from(bodies, queue_);
    
    // map factors to bodies for all q
    queue_.enqueueNDRangeKernel(m_kernel_map_factors, cl::NullRange, cl::NDRange(n_bodies_aln_), cl::NDRange(wavefront_size_));
    // calculate Debye for all square blocks of size wavefront X wavefront
    queue_.enqueueNDRangeKernel(m_kernel, cl::NullRange, cl::NDRange(n_bodies_aln_, n_q_), cl::NDRange(wavefront_size_, 1));
    // sum the Debye results vertically
    queue_.enqueueNDRangeKernel(m_kernel_sum_phase1, cl::NullRange, cl::NDRange(n_bodies_aln_, n_q_), cl::NDRange(wavefront_size_, 1));
    // sum the Debye results horizontally
    queue_.enqueueNDRangeKernel(m_kernel_sum, cl::NullRange, cl::NDRange(n_q_), cl::NullRange);

    // retrieve the results I(q)
    out_Iqq.resize(n_q_);
    queue_.enqueueReadBuffer(b_Iqq_, CL_TRUE, 0, n_q_ * sizeof(float), &out_Iqq.front());

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
void gpu_saxs_2d_alg::recalc_curve(const std::vector<cl_float4> & bodies, int upd_start, int upd_length, std::vector<float> & out_Iqq)
{
    verify(computing(), error::SAXS_NO_INITIAL_CALC);

        // zero out the form factor indices, as we have already mapped them
        std::vector<cl_float4> upd_bodies(&bodies[upd_start], &bodies[upd_start + upd_length]);
        for (std::vector<cl_float4>::iterator iter = upd_bodies.begin(); iter != upd_bodies.end(); iter++)
            (*iter).s[3] = 0;
    // send the moved bodies
    b_bodies_.write_from(&upd_bodies.front(), upd_start, upd_length, queue_);
    
    // align the start index and the length to wavefront_size_
    int upd_start_aln = (upd_start / wavefront_size_) * wavefront_size_;
    int upd_length_aln = aligned_num(upd_start + upd_length, wavefront_size_) - upd_start_aln;
    m_kernel_recalc.setArg(3, upd_start_aln);
    m_kernel_recalc.setArg(4, upd_start_aln + upd_length_aln);
    // recalculate blocks affected by the update
    queue_.enqueueNDRangeKernel(m_kernel_recalc, cl::NullRange, cl::NDRange(n_bodies_aln_, n_q_), cl::NDRange(wavefront_size_, 1));
        // adjust I(q) with calculated deltas
    queue_.enqueueNDRangeKernel(m_kernel_adjust, cl::NullRange, cl::NDRange(n_q_), cl::NullRange);
    // sum the Debye results horizontally
        m_kernel_save_prev_bodies.setArg(1, upd_start_aln);
    queue_.enqueueNDRangeKernel(m_kernel_save_prev_bodies, cl::NullRange, cl::NDRange(upd_length_aln), cl::NDRange(wavefront_size_));

    // retrieve the results I(q)
    out_Iqq.resize(n_q_);
    queue_.enqueueReadBuffer(b_Iqq_, CL_TRUE, 0, n_q_ * sizeof(float), &out_Iqq.front());
}

}   // namespace
