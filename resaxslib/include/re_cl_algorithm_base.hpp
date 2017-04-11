#ifndef RE_CL_ALGORITHM_BASE
#define RE_CL_ALGORITHM_BASE

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011-2016 Lubo Antonov
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

#include <algorithm>
#include "re_algorithm.hpp"
#include "re_cl_buffer.hpp"

namespace resaxs {
namespace algorithm {

    ///////////////////////////////////////////////////////////////////////////////
    //  Base for all OpenCL algorithms.
    //
    //      Provides device initialization and program loading.
    //      Instantiated for float and double.
    //
    template <typename FLT_T>
    class cl_base : public algorithm
    {
    public:
        //
        //  Initializes the algorithm with a list of devices to run on
        //
        void initialize(const std::vector<dev_id> & dev_ids);

    protected:
        // OpenCL objects - valid after initialize()
        cl::Context context_;
        cl::CommandQueue queue_;
        std::vector<cl::Device> devices_;

        //
        //  Loads the specified files as a program and builds it.
        //      file_names          - a list of file names to load
        //      file_name           - a single file name to load
        //      result              - a built OpenCL program object on successful completion
        //      options             - command-line options to pass to the OpenCL compiler
        //
        void load_program(const std::vector<std::string> & file_names, cl::Program & result, const std::string & options = NULL);
        void load_program(const std::string & file_name, cl::Program & result, const std::string & options = NULL);

        //
        //  Loads the specified files as a program object
        //      file_names          - a list of file names to load
        //      result              - a OpenCL program object on successful completion
        //
        void get_program(const std::vector<std::string> & file_names, cl::Program & result);

        //
        //  Builds a program from a source string.
        //      src                 - program source
        //      result              - a built OpenCL program object on successful completion
        //      options             - command-line options to pass to the OpenCL compiler
        //      
        void build_program(const std::string & src, cl::Program & result, const std::string & options = NULL);

        //
        //  Converts the contents of a file into a string.
        //
        static std::string convert_to_string(const std::string & filename);
    };

} // namespace algorithm
} // namespace resaxs

#endif  // RE_CL_ALGORITHM_BASE
