///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011-2015 Lubo Antonov
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

#include "../include/re_cl_algorithm_base.hpp"

#include <cassert>
#include <iostream>
#include <fstream>

#include "utils.hpp"

using namespace std;

namespace resaxs {
namespace algorithm {

    //
    //  Initializes the algorithm with a list of devices to run on
    //
    template <typename FLT_T>
    void cl_base<FLT_T>::initialize(const std::vector<dev_id> & dev_ids)
    {
        devices_ = get_cl_devices(dev_ids);
        context_ = cl::Context(devices_);
        queue_ = cl::CommandQueue(context_, devices_[0]);

        this->state = this->alg_initialized;
    }

    //
    //  Loads the specified files as a program and builds it.
    //      file_names          - a list of file names to load
    //      result              - a built OpenCL program object on successful completion
    //      options             - command-line options to pass to the OpenCL compiler
    //
    template <typename FLT_T>
    void cl_base<FLT_T>::load_program(const std::vector<std::string> & file_names, cl::Program & result, const std::string & options)
    {
        get_program(file_names, result);

        try
        {
            result.build(devices_, options.c_str());
        }
        catch (cl::Error e)
        {
            std::string str = result.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices_[0]);

            std::cerr << " \n\t\t\tBUILD LOG\n";
            std::cerr << " ************************************************\n";
            std::cerr << str.c_str() << std::endl;
            std::cerr << " ************************************************\n";

            throw;
        }
    }

    //
    //  Loads the specified file as a program and builds it.
    //      file_name           - a single file name to load
    //      result              - a built OpenCL program object on successful completion
    //      options             - command-line options to pass to the OpenCL compiler
    //
    template <typename FLT_T>
    void cl_base<FLT_T>::load_program(const std::string & file_name, cl::Program & result, const std::string & options)
    {
        std::vector<std::string> file_names(1, file_name);
        load_program(file_names, result, options);
    }

    //
    //  Loads the specified files as a program object
    //      file_names          - a list of file names to load
    //      result              - a OpenCL program object on successful completion
    //
    template <typename FLT_T>
    void cl_base<FLT_T>::get_program(const std::vector<std::string> & file_names, cl::Program & result)
    {
        verify(this->initialized(), error::SAXS_ALG_NOT_INITIALIZED);

        // The cl::Program class has a retarded interface, so we need to keep the sources alive during the constructor call
        std::vector<std::string> sources;
        cl::Program::Sources pr_sources;
        for (auto & file_name : file_names)
        {
            sources.push_back(convert_to_string(file_name));
            pr_sources.push_back(std::make_pair(sources.back().c_str(), sources.back().size()));
        }

        result = cl::Program(context_, pr_sources);
    }

    //
    //  Builds a program from a source string.
    //      src                 - program source
    //      result              - a built OpenCL program object on successful completion
    //      options             - command-line options to pass to the OpenCL compiler
    //
    template <typename FLT_T>
    void cl_base<FLT_T>::build_program(const std::string & src, cl::Program & result, const std::string & options)
    {
        verify(this->initialized(), error::SAXS_ALG_NOT_INITIALIZED);

        cl::Program::Sources pr_sources;
        pr_sources.push_back(std::make_pair(src.c_str(), src.size()));
        result = cl::Program(context_, pr_sources);

        try
        {
            result.build(devices_, options.c_str());
        }
        catch (cl::Error e)
        {
            std::string str = result.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices_[0]);

            std::cerr << " \n\t\t\tBUILD LOG\n";
            std::cerr << " ************************************************\n";
            std::cerr << str.c_str() << std::endl;
            std::cerr << " ************************************************\n";

            throw;
        }
    }

    //
    //  Converts the contents of a file into a string
    //
    template <typename FLT_T>
    std::string cl_base<FLT_T>::convert_to_string(const std::string & filename)
    {
        size_t size;
        char*  str;
        std::string s;

        std::fstream f(filename.c_str(), (std::fstream::in | std::fstream::binary));

        if (f.is_open())
        {
            size_t fileSize;
            f.seekg(0, std::fstream::end);
            size = fileSize = (size_t)f.tellg();
            f.seekg(0, std::fstream::beg);

            str = new char[size + 1];
            if (!str)
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
            throw error(error::SAXS_INVALID_ARG, "File containing the kernel code(\".cl\") not found.");
        }
        return NULL;
    }

    template class cl_base<float>;
    template class cl_base<double>;

}
}   // namespace
