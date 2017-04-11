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

#include <cstdlib>
#include <cmath>

#include "../include/host_debye.hpp"

#include "../include/utils.hpp"

namespace resaxs
{

//
// Compute the Euclidean distance between a and b.
//
template<typename T, typename FLT4_T> T distance(const FLT4_T & a, const FLT4_T & b)
{
    T dx = T(a.s[0]) - b.s[0];
    T dy = T(a.s[1]) - b.s[1];
    T dz = T(a.s[2]) - b.s[2];
    return sqrt(dx*dx + dy*dy + dz*dz);
}

//
// Calculate the SAXS intensity curve I(q)
//
// v_q          - vector of q values
// n_q          - number of q values
// v_bodies     - vector of bodies (x, y, z, factor index)
// n_bodies     - number of bodies
// t_factors    - table of form factors [factor index, q index]
// n_factors    - number of factors (for each q)
// out_v_Iq     - the result - vector I(q) of SAXS intensity curve values
//
template<typename T, typename FLT_T, typename FLT4_T>
void host_debye<T, FLT_T, FLT4_T>::calc_curve(const FLT_T * v_q, int n_q, const FLT4_T* v_bodies, int n_bodies, const FLT_T * t_factors, int n_factors, FLT_T * out_v_Iq)
{
    // pre-compute all distances
    // allocate the full matrix for performance, even though we only need the lower triangle
    std::vector<T> r(n_bodies * n_bodies);
    for (int y = 1; y < n_bodies; y++)
        for (int x = 0; x < y; x++)
            r[y * n_bodies + x] = distance<T>(v_bodies[x], v_bodies[y]);

    std::vector<FLT_T> factors(n_q * n_bodies);
    for (int k = 0; k < n_q; k++)
    {
        for (int i = 0; i < n_bodies; i++)
        {
            int f_index = get_factor_index(v_bodies[i]);
            factors[k * n_bodies + i] = t_factors[k * n_factors + f_index];
        }
    }

    for (int k = 0; k < n_q; k++)
    {
        T q = v_q[k];
        T Iq = 0;
        const FLT_T * Fq = &factors[k * n_bodies];
        // accumulate the autologous cases separately, so that we simply do * 2 for the rest
        T auto_Iq = 0;

        for (int y = 0; y < n_bodies; y++)
        {
            T Fqy = Fq[y];

            auto_Iq += Fqy * Fqy;   // autologous case

            // inner sum; Fi(q) is factored out in front -- I(q) = sumi( Fi(q) * sumj( ... ) )
            // but we skip the autologous case (x = y), since we already did it above
            T inner_sum = 0;
            for (int x = 0; x < y; x++)
            {
                T v1 = 1;   // for q = 0, sin(qr)/qr = 1
                if (q != 0)
                {
                    T qr = q * r[y * n_bodies + x];   // q * r
                    v1 = sinc(qr);
                }

                inner_sum += Fq[x] * v1;
            }

            Iq += inner_sum * Fqy;
        }

        // double the values from the lower-triangle of the matrix and add the autologous ones
        out_v_Iq[k] = FLT_T(Iq * 2 + auto_Iq);
    }
}

template<typename T, typename FLT_T, typename FLT4_T>
void host_debye<T, FLT_T, FLT4_T>::calc_curve(const std::vector<FLT_T> & v_q, const std::vector<FLT4_T> & v_bodies, const std::vector<FLT_T> & t_factors, std::vector<FLT_T> & out_v_Iq)
{
    calc_curve(&v_q.front(), int(v_q.size()), &v_bodies.front(), int(v_bodies.size()), &t_factors.front(), int(t_factors.size() / v_q.size()), &out_v_Iq.front());
}

template class host_debye<double, double, cl_double4>;
template class host_debye<float, float, cl_float4>;
template class host_debye<double, float, cl_float4>;

} // namespace
