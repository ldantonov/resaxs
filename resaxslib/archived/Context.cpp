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

#include <cstdlib>

#include "Context.hpp"

#include <iostream>
#include <fstream>
#include <stdexcept>

#include "utils.hpp"

namespace accsaxs
{

Context::Context()
{
}

void Context::initialize(const std::vector<dev_id> & dev_ids)
{
    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    std::vector<cl::Device> devices;
    m_run_device.clear();
    if (dev_ids[0].platform == -1 || dev_ids[0].device == -1)
    {
        cl_device_type dev_type = CL_DEVICE_TYPE_GPU;
        if (dev_ids[0].type == CPU)
            dev_type = CL_DEVICE_TYPE_CPU;

        for (std::vector<cl::Platform>::iterator iter = platforms.begin(); iter != platforms.end(); iter++)
        {
            try
            {
                (*iter).getDevices(dev_type, &devices);
            }
            catch(cl::Error e)
            {
                if (e.err() == CL_DEVICE_NOT_FOUND)
                    continue;
                else
                    throw;
            }

            m_run_device.push_back(devices[0]);
            break;
        }
    }
    else
    {
        platforms[dev_ids[0].platform].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        m_run_device.push_back(devices[dev_ids[0].device]);
    }

    m_context = cl::Context(m_run_device);
    m_queue = cl::CommandQueue(m_context, m_run_device[0]);
}

//
//  Converts the contents of a file into a string
//
std::string convert_to_string(const std::string & filename)
{
    size_t size;
    char*  str;
    std::string s;

    std::fstream f(filename.c_str(), (std::fstream::in | std::fstream::binary));

    if(f.is_open())
    {
        size_t fileSize;
        f.seekg(0, std::fstream::end);
        size = fileSize = (size_t)f.tellg();
        f.seekg(0, std::fstream::beg);

        str = new char[size+1];
        if(!str)
        {
            f.close();
            return NULL;
        }

        f.read(str, fileSize);
        f.close();
        str[size] = '\0';
    
        s = str;
        delete[] str;
        return s;
    }
    else
    {
        throw std::runtime_error("File containg the kernel code(\".cl\") not found.");
    }
    return NULL;
}

void get_program(const std::vector<std::string> & file_names, cl::Context & context, cl::Program & result)
{
    // The cl::Program class has a retarded interface, so we need to keep the sources alive during the constructor call
    std::vector<std::string> sources;
    cl::Program::Sources pr_sources;
    for (std::vector<std::string>::const_iterator iter = file_names.begin(); iter != file_names.end(); iter++)
    {
        sources.push_back(convert_to_string(*iter));
        pr_sources.push_back(std::make_pair(sources.back().c_str(), sources.back().size()));
    }

    result = cl::Program(context, pr_sources);
}

void Context::load_program(const std::vector<std::string> & file_names, cl::Program & result, const std::string & options)
{
    get_program(file_names, m_context, result);

    try
    {
        result.build(m_run_device, options.c_str());
    }
    catch(cl::Error e)
    {
        std::string str = result.getBuildInfo<CL_PROGRAM_BUILD_LOG>(m_run_device[0]);

        std::cerr << " \n\t\t\tBUILD LOG\n";
        std::cerr << " ************************************************\n";
        std::cerr << str.c_str() << std::endl;
        std::cerr << " ************************************************\n";

        throw;
    }
}

void Context::load_program(const std::string & file_name, cl::Program & result, const std::string & options)
{
    std::vector<std::string> file_names(1, file_name);
    load_program(file_names, result, options);
}


////// 

void GpuSaxsAlg::load_kernels()
{
    Context::load_program("gpu_saxs.cl", m_program);

    m_kernel_sum = cl::Kernel(m_program, "sum");
}

int GpuSaxsAlg::aligned_num(int num, int stride)
{
    return ((num - 1) / stride + 1) * stride;
}

void GpuSaxsAlg::set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies)
{
    this->n_q = n_q;
    this->n_bodies = n_bodies;
    this->n_factors = n_factors;

    n_bodies_aln = aligned_num(n_bodies, WF_SIZE);

    b_q = cl::Buffer(m_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, n_q * sizeof(float), v_q);
    b_bodies = cl::Buffer(m_context, CL_MEM_READ_WRITE, n_bodies_aln * sizeof(cl_float4));//n_bodies * sizeof(cl_float4));
    b_factors = cl::Buffer(m_context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, n_q * n_factors * sizeof(float), t_factors);
    b_Iq = cl::Buffer(m_context, CL_MEM_WRITE_ONLY, n_q * sizeof(float));

    b_margin_Iq = cl::Buffer(m_context, CL_MEM_READ_WRITE, n_q * n_bodies * sizeof(float));

    m_kernel_sum.setArg(0, b_margin_Iq);
    m_kernel_sum.setArg(1, n_bodies);
    m_kernel_sum.setArg(2, b_Iq);
}


////// -

void GpuSaxsBasic::set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies)
{
    GpuSaxsAlg::set_args(v_q, n_q, t_factors, n_factors, n_bodies);

    m_kernel.setArg(0, b_q);
    m_kernel.setArg(1, n_q);
    m_kernel.setArg(2, b_bodies);
    m_kernel.setArg(3, n_bodies);
    m_kernel.setArg(4, b_factors);
    m_kernel.setArg(5, n_factors);
    m_kernel.setArg(6, b_margin_Iq);
}

void GpuSaxsBasic::load_kernels()
{
    GpuSaxsAlg::load_kernels();

    m_kernel = cl::Kernel(m_program, "margins");
}

void GpuSaxsBasic::calc_saxs_curve(cl_float4* v_bodies, int upd_start, int upd_length, float* out_v_Iq)
{
    m_queue.enqueueWriteBuffer(b_bodies, CL_TRUE, 0, n_bodies * sizeof(cl_float4), v_bodies);

    // Execute the kernel.
    // 'globalWorkSize' is the 1D dimension of the work-items
    cl::NDRange gthreads = n_bodies;
    m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, gthreads, cl::NullRange);

    m_queue.enqueueNDRangeKernel(m_kernel_sum, cl::NullRange, cl::NDRange(n_q), cl::NullRange);

    // Read the OpenCL output buffer (d_C) to the host output array (C)
    m_queue.enqueueReadBuffer(b_Iq, CL_TRUE, 0, n_q * sizeof(float), out_v_Iq);
}


////// -

void GpuSaxsMappedFactors::load_kernels()
{
    GpuSaxsAlg::load_kernels();

    m_kernel_map_factors = cl::Kernel(m_program, "map_factors");
    m_kernel = cl::Kernel(m_program, "margins_mapped_factors");
}

void GpuSaxsMappedFactors::set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies)
{
    GpuSaxsAlg::set_args(v_q, n_q, t_factors, n_factors, n_bodies);

    b_bodies_f = cl::Buffer(m_context, CL_MEM_READ_WRITE, n_q * n_bodies * sizeof(cl_float4));

    m_kernel_map_factors.setArg(0, n_q);
    m_kernel_map_factors.setArg(1, b_bodies);
    m_kernel_map_factors.setArg(2, n_bodies);
    m_kernel_map_factors.setArg(3, b_factors);
    m_kernel_map_factors.setArg(4, n_factors);
    m_kernel_map_factors.setArg(5, b_bodies_f);

    m_kernel.setArg(0, b_q);
    m_kernel.setArg(1, n_q);
    m_kernel.setArg(2, b_bodies_f);
    m_kernel.setArg(3, n_bodies);
    m_kernel.setArg(4, b_margin_Iq);
}

void GpuSaxsMappedFactors::calc_saxs_curve(cl_float4* v_bodies, int upd_start, int upd_length, float* out_v_Iq)
{
    m_queue.enqueueWriteBuffer(b_bodies, CL_TRUE, 0, n_bodies * sizeof(cl_float4), v_bodies);

    // Execute the kernel.
    // 'globalWorkSize' is the 1D dimension of the work-items
    cl::NDRange gthreads = n_bodies;
    m_queue.enqueueNDRangeKernel(m_kernel_map_factors, cl::NullRange, gthreads, cl::NullRange);
    m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, gthreads, cl::NullRange);

    m_queue.enqueueNDRangeKernel(m_kernel_sum, cl::NullRange, cl::NDRange(n_q), cl::NullRange);

    // Read the OpenCL output buffer (d_C) to the host output array (C)
    m_queue.enqueueReadBuffer(b_Iq, CL_TRUE, 0, n_q * sizeof(float), out_v_Iq);
}


///// -

void GpuSaxsMappedFactors1::load_kernels()
{
    GpuSaxsAlg::load_kernels();

    m_kernel_map_factors = cl::Kernel(m_program, "map_factors1");
    m_kernel = cl::Kernel(m_program, "margins_mapped_factors1");
}

void GpuSaxsMappedFactors1::set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies)
{
    GpuSaxsAlg::set_args(v_q, n_q, t_factors, n_factors, n_bodies);

    b_bodies_f = cl::Buffer(m_context, CL_MEM_READ_WRITE, n_q * n_bodies * sizeof(float));

    m_kernel_map_factors.setArg(0, n_q);
    m_kernel_map_factors.setArg(1, b_bodies);
    m_kernel_map_factors.setArg(2, n_bodies);
    m_kernel_map_factors.setArg(3, b_factors);
    m_kernel_map_factors.setArg(4, n_factors);
    m_kernel_map_factors.setArg(5, b_bodies_f);

    m_kernel.setArg(0, b_q);
    m_kernel.setArg(1, n_q);
    m_kernel.setArg(2, b_bodies);
    m_kernel.setArg(3, n_bodies);
    m_kernel.setArg(4, b_bodies_f);
    m_kernel.setArg(5, b_margin_Iq);
}

void GpuSaxsMappedFactors1::calc_saxs_curve(cl_float4* v_bodies, int upd_start, int upd_length, float* out_v_Iq)
{
    m_queue.enqueueWriteBuffer(b_bodies, CL_TRUE, 0, n_bodies * sizeof(cl_float4), v_bodies);

    // Execute the kernel.
    // 'globalWorkSize' is the 1D dimension of the work-items
    cl::NDRange gthreads = n_bodies;
    m_queue.enqueueNDRangeKernel(m_kernel_map_factors, cl::NullRange, gthreads, cl::NullRange);
    m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, gthreads, cl::NullRange);

    m_queue.enqueueNDRangeKernel(m_kernel_sum, cl::NullRange, cl::NDRange(n_q), cl::NullRange);

    // Read the OpenCL output buffer (d_C) to the host output array (C)
    m_queue.enqueueReadBuffer(b_Iq, CL_TRUE, 0, n_q * sizeof(float), out_v_Iq);
}


///// -

void GpuSaxsCached::load_kernels()
{
    GpuSaxsAlg::load_kernels();

    m_kernel_map_factors = cl::Kernel(m_program, "map_factors2");
    m_kernel = cl::Kernel(m_program, "margins_mapped_factors1");
    m_kernel_save_prev_bodies = cl::Kernel(m_program, "save_prev_bodies");
    m_kernel_recalc = cl::Kernel(m_program, "recalc_margins_mf2");
    m_kernel_adjust = cl::Kernel(m_program, "adjust_Iq");
}

void GpuSaxsCached::set_args(float* v_q, int n_q, float* t_factors, int n_factors, int n_bodies)
{
    GpuSaxsAlg::set_args(v_q, n_q, t_factors, n_factors, n_bodies);

    b_bodies_f = cl::Buffer(m_context, CL_MEM_READ_WRITE, n_q * n_bodies * sizeof(float));
    b_prev_bodies = cl::Buffer(m_context, CL_MEM_READ_WRITE, n_bodies_aln * sizeof(cl_float4));//n_bodies * sizeof(cl_float4));

    m_kernel_map_factors.setArg(0, n_q);
    m_kernel_map_factors.setArg(1, b_bodies);
    m_kernel_map_factors.setArg(2, n_bodies);
    m_kernel_map_factors.setArg(3, b_factors);
    m_kernel_map_factors.setArg(4, n_factors);
    m_kernel_map_factors.setArg(5, b_bodies_f);
    m_kernel_map_factors.setArg(6, b_prev_bodies);

    m_kernel.setArg(0, b_q);
    m_kernel.setArg(1, n_q);
    m_kernel.setArg(2, b_bodies);
    m_kernel.setArg(3, n_bodies);
    m_kernel.setArg(4, b_bodies_f);
    m_kernel.setArg(5, b_margin_Iq);

    m_kernel_save_prev_bodies.setArg(0, b_bodies);
    //m_kernel_save_prev_bodies.setArg(1, 0);
    m_kernel_save_prev_bodies.setArg(2, b_prev_bodies);

    m_kernel_recalc.setArg(0, b_q);
    m_kernel_recalc.setArg(1, n_q);
    m_kernel_recalc.setArg(2, b_bodies);
    m_kernel_recalc.setArg(3, n_bodies);
    m_kernel_recalc.setArg(4, b_bodies_f);
    //m_kernel_recalc.setArg(5, 0);
    //m_kernel_recalc.setArg(6, n_bodies);
    m_kernel_recalc.setArg(7, b_prev_bodies);
    m_kernel_recalc.setArg(8, b_margin_Iq);

    m_kernel_adjust.setArg(0, b_margin_Iq);
    m_kernel_adjust.setArg(1, n_bodies);
    m_kernel_adjust.setArg(2, b_Iq);
}

void GpuSaxsCached::calc_saxs_curve(cl_float4* v_bodies, int upd_start, int upd_length, float* out_v_Iq)
{
    if (upd_start == 0 && upd_length == n_bodies)
    {
        m_queue.enqueueWriteBuffer(b_bodies, CL_TRUE, upd_start * sizeof(cl_float4), upd_length * sizeof(cl_float4), v_bodies);

        // Execute the kernel.
        // 'globalWorkSize' is the 1D dimension of the work-items
        cl::NDRange gthreads = n_bodies;
        m_queue.enqueueNDRangeKernel(m_kernel_map_factors, cl::NullRange, gthreads, cl::NullRange);
        m_queue.enqueueNDRangeKernel(m_kernel, cl::NullRange, gthreads, cl::NullRange);

        m_queue.enqueueNDRangeKernel(m_kernel_sum, cl::NullRange, cl::NDRange(n_q), cl::NullRange);
    }
    else
    {
        std::vector<cl_float4> upd_bodies(v_bodies, v_bodies + upd_length);
        for (std::vector<cl_float4>::iterator iter = upd_bodies.begin(); iter != upd_bodies.end(); iter++)
            (*iter).s[3] = 0;
        m_queue.enqueueWriteBuffer(b_bodies, CL_TRUE, upd_start * sizeof(cl_float4), upd_length * sizeof(cl_float4), &upd_bodies.front());

        m_kernel_recalc.setArg(5, upd_start);
        m_kernel_recalc.setArg(6, upd_start + upd_length);
        m_queue.enqueueNDRangeKernel(m_kernel_recalc, cl::NullRange, cl::NDRange(n_bodies), cl::NullRange);

        m_queue.enqueueNDRangeKernel(m_kernel_adjust, cl::NullRange, cl::NDRange(n_q), cl::NullRange);

        int upd_start_aln = (upd_start / WF_SIZE) * WF_SIZE;
        int upd_length_aln = aligned_num(upd_start + upd_length, WF_SIZE) - upd_start_aln;
        m_kernel_save_prev_bodies.setArg(1, upd_start_aln);
        m_queue.enqueueNDRangeKernel(m_kernel_save_prev_bodies, cl::NullRange, cl::NDRange(upd_length_aln), cl::NDRange(WF_SIZE));
    }

    // Read the OpenCL output buffer (d_C) to the host output array (C)
    m_queue.enqueueReadBuffer(b_Iq, CL_TRUE, 0, n_q * sizeof(float), out_v_Iq);
}



} // namespace
