#ifndef ACCSAXS_CONTEXT_HPP
#define ACCSAXS_CONTEXT_HPP

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

#include <string>

#define __CL_ENABLE_EXCEPTIONS

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4290 )   // disable warning about the ignored exception specs in VC++
#endif
#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif
#ifdef _MSC_VER
#pragma warning( pop )
#endif

namespace accsaxs
{

class Context
{
public:
    enum algorithm_type
    {
        CPU_only,
        GPU_only,
    };

    enum device_type
    {
        CPU,
        GPU,
    };

    struct dev_id
    {
        long platform;
        long device;
        device_type type;
    };

    Context();
    void initialize(const std::vector<dev_id> & dev_ids);

protected:
    cl::Context m_context;
    cl::CommandQueue m_queue;
    std::vector<cl::Device> m_run_device;

    void load_program(const std::vector<std::string> & file_names, cl::Program & result, const std::string & options = NULL);
    void load_program(const std::string & file_name, cl::Program & result, const std::string & options = NULL);
};

#define WF_SIZE     32
#define WF_PER_SM   64
#define NUM_SM      8

class GpuSaxsAlg : public Context
{
public:
    void load_kernels();
    void set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies);
    virtual void calc_saxs_curve(cl_float4* v_bodies, int upd_start, int upd_length, float* out_v_Iq) = 0;
    virtual ~GpuSaxsAlg() {};

    static int aligned_num(int num, int stride);
protected:
    cl::Program m_program;
    cl::Kernel m_kernel;

    cl::Kernel m_kernel_sum;

protected:
    cl::Buffer b_q;
    int n_q;
    cl::Buffer b_bodies;
    int n_bodies;
    cl::Buffer b_factors;
    int n_factors;
    cl::Buffer b_Iq;

    cl::Buffer b_margin_Iq;

    int n_bodies_aln;
};

class GpuSaxsBasic : public GpuSaxsAlg
{
public:
    void load_kernels();
    void set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies);
    virtual void calc_saxs_curve(cl_float4* v_bodies, int upd_start, int upd_length, float* out_v_Iq);
};

class GpuSaxsMappedFactors : public GpuSaxsAlg
{
public:
    void load_kernels();
    void set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies);
    virtual void calc_saxs_curve(cl_float4* v_bodies, int upd_start, int upd_length, float* out_v_Iq);

private:
    cl::Kernel m_kernel_map_factors;

private:
    cl::Buffer b_bodies_f;
};

class GpuSaxsMappedFactors1 : public GpuSaxsAlg
{
public:
    void load_kernels();
    void set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies);
    virtual void calc_saxs_curve(cl_float4* v_bodies, int upd_start, int upd_length, float* out_v_Iq);

private:
    cl::Kernel m_kernel_map_factors;

private:
    cl::Buffer b_bodies_f;
};

class GpuSaxsCached : public GpuSaxsAlg
{
public:
    void load_kernels();
    void set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies);
    virtual void calc_saxs_curve(cl_float4* v_bodies, int upd_start, int upd_length, float* out_v_Iq);

private:
    cl::Kernel m_kernel_map_factors;
    cl::Kernel m_kernel_save_prev_bodies;
    cl::Kernel m_kernel_recalc;
    cl::Kernel m_kernel_adjust;

private:
    cl::Buffer b_bodies_f;
    cl::Buffer b_prev_bodies;
};


} // namespace

#endif // ACCSAXS_CONTEXT_HPP
