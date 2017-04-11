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

#include "re_cl_algorithm_base.hpp"
#include "saxs_algorithm.hpp"
#include "saxs_alg_cl_gpu_pt_wf.hpp"
#include "saxs_alg_cl_gpu_v2.hpp"
#include "alg_saxs_hist_cl_gpu.hpp"

using namespace std;

namespace resaxs {
namespace algorithm {

    //////////////////////////////////////////////////////////////////////////
    /// Creator for the OpenCL GPU saxs_gpu_pt_wf algorithm.
    template <typename FLT_T>
    struct saxs_creator<FLT_T, cl_base<FLT_T>, saxs_enum::saxs_gpu_pt_wf>
    {
        static i_saxs<FLT_T, cl_base<FLT_T>>* create()
        {
            return new saxs_cl_gpu<FLT_T>();
        }
    };

    template <typename FLT_T>
    struct saxs_creator<FLT_T, cl_base<FLT_T>, saxs_enum::saxs_gpu_v2>
    {
        static i_saxs<FLT_T, cl_base<FLT_T>>* create()
        {
            return new saxs_cl_gpu_v2<FLT_T>();
        }
    };

    template <typename FLT_T>
    struct saxs_creator<FLT_T, cl_base<FLT_T>, saxs_enum::saxs_hist_cl_gpu>
    {
        static i_saxs<FLT_T, cl_base<FLT_T>>* create()
        {
            return new saxs_hist_cl_gpu<FLT_T>();
        }
    };

    //////////////////////////////////////////////////////////////////////////
    /// Explicit instantiations for the supported algorithm configurations.
    template struct saxs_creator<float, cl_base<float>, saxs_enum::saxs_gpu_pt_wf>;
    template struct saxs_creator<double, cl_base<double>, saxs_enum::saxs_gpu_pt_wf>;
    template struct saxs_creator<float, cl_base<float>, saxs_enum::saxs_gpu_v2>;
    template struct saxs_creator<double, cl_base<double>, saxs_enum::saxs_gpu_v2>;
    template struct saxs_creator<float, cl_base<float>, saxs_enum::saxs_hist_cl_gpu>;
    template struct saxs_creator<double, cl_base<double>, saxs_enum::saxs_hist_cl_gpu>;
}
}