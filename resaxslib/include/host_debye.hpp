#ifndef RESAXS_HOST_DEBYE_HPP
#define RESAXS_HOST_DEBYE_HPP

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

#include "resaxs_cl_core.hpp"

namespace resaxs
{

typedef void (*CalcSaxsCurveFn)(const float* v_q, int n_q, const cl_float4* v_bodies, int n_bodies, const float* t_factors, int n_factors, float* out_v_Iq);
typedef void (*CalcSaxsCurveFnDP)(const double* v_q, int n_q, const cl_double4* v_bodies, int n_bodies, const double* t_factors, int n_factors, double* out_v_Iq);

template <typename T, typename FLT_T, typename FLT4_T>
class host_debye
{
public:
    typedef void (*CalcSaxsCurveFn)(const FLT_T * v_q, int n_q, const FLT4_T * v_bodies, int n_bodies, const FLT_T * t_factors, int n_factors, FLT_T * out_v_Iq);
    static void calc_curve(const FLT_T * v_q, int n_q, const FLT4_T* v_bodies, int n_bodies, const FLT_T * t_factors, int n_factors, FLT_T * out_v_Iq);
    static void calc_curve(const std::vector<FLT_T> & v_q, const std::vector<FLT4_T> & v_bodies, const std::vector<FLT_T> & t_factors, std::vector<FLT_T> & out_v_Iq);
};

} // namespace

#endif // RESAXS_HOST_DEBYE_HPP