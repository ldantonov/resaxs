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

#include "../include/rsasa_algorithm.hpp"

#include "rsasa_cpu_alg.hpp"
#include "constants.hpp"
#include "rsasa_cl_gpu_alg.hpp"

using namespace std;

namespace resaxs {
namespace algorithm {

    // Simplifying the syntax for the vector4 types
    template <typename FLT_T> using real4 = typename real_type<FLT_T>::real4;

    //////////////////////////////////////////////////////////////////////////
    /// Creator for the CPU rsasa_dot_spheres algorithm.
    template <typename FLT_T, typename OUT_FLT_T>
    struct rsasa_creator<FLT_T, OUT_FLT_T, algorithm, rsasa_dot_spheres>
    {
        static i_rsasa<FLT_T, OUT_FLT_T, algorithm>* create()
        {
            return new rsasa_cpu_alg<FLT_T, OUT_FLT_T>();
        }
    };

    //////////////////////////////////////////////////////////////////////////
    /// Creator for the OpenCL GPU rsasa_dot_spheres algorithm.
    template <typename FLT_T, typename OUT_FLT_T>
    struct rsasa_creator<FLT_T, OUT_FLT_T, cl_base<FLT_T>, rsasa_dot_spheres>
    {
        static i_rsasa<FLT_T, OUT_FLT_T, cl_base<FLT_T>>* create()
        {
            return new rsasa_cl_gpu<FLT_T, OUT_FLT_T>();
        }
    };

    template <typename FLT_T>
    const vector<real4<FLT_T>> create_sphere_dots(FLT_T radius, FLT_T n_sect, FLT_T probe_radius)
    {
        vector<real4<FLT_T>> dots;
        const FLT_T n_vert_sect = 0.5f * n_sect;
        const FLT_T ext_radius = radius + probe_radius;

        for (unsigned int i = 0; i < n_vert_sect; i++)
        {
            FLT_T phi = (static_cast<FLT_T>(pi)* i) / n_vert_sect;  // vertical angle
            FLT_T z = cos(phi) * ext_radius;
            FLT_T xy = sin(phi);    // radius of the circular slice at angle phi
            FLT_T horz_count = xy * n_sect;
            for (unsigned int j = 0; j < horz_count - 1; j++)
            {
                FLT_T theta = (2 * static_cast<FLT_T>(pi)* j) / horz_count;     // horizontal angle
                FLT_T x = xy * cos(theta) * ext_radius;
                FLT_T y = xy * sin(theta) * ext_radius;
                dots.push_back({x, y, z, radius});
            }
        }

        return dots;
    }

    template <typename FLT_T, typename OUT_FLT_T>
    const vector<real4<FLT_T>> rsasa<FLT_T, OUT_FLT_T>::dot_spheres_params::create_dots(FLT_T radius) const
    {
        const FLT_T n_sect = 2 * static_cast<FLT_T>(pi)* radius * sqrt_dot_density_;    // number of sectors
        return create_sphere_dots<FLT_T>(radius, n_sect, probe_radius_);
    }

    template <typename FLT_T, typename OUT_FLT_T>
    const vector<real4<FLT_T>> rsasa<FLT_T, OUT_FLT_T>::dot_spheres_params::create_32_dots(FLT_T radius) const
    {
        return create_sphere_dots<FLT_T>(radius, 10.5f, probe_radius_);
    }

    template <typename FLT_T, typename OUT_FLT_T>
    const vector<real4<FLT_T>> rsasa<FLT_T, OUT_FLT_T>::dot_spheres_params::create_fibonacci_dots(FLT_T radius) const
    {
        vector<real4> dots;
        const int n = 32;
        const FLT_T phi = (1.0f + sqrt(5.0f)) / 2.0f;
        const FLT_T ext_radius = radius + probe_radius_;

        for (int i = -(n - 1); i <= n; ++i)
        {
            const FLT_T theta = 2 * static_cast<FLT_T>(pi)* i / phi;
            const FLT_T sphi = static_cast<FLT_T>(i) / n;
            const FLT_T cphi = static_cast<FLT_T>(sqrt((n + i) * (n - i)) / n);

            const FLT_T x = cphi * sin(theta) * ext_radius;
            const FLT_T y = cphi * cos(theta) * ext_radius;
            const FLT_T z = sphi * ext_radius;
            dots.push_back({x, y, z, radius});
        }
        return dots;
    }

    template <typename FLT_T, typename OUT_FLT_T>
    const vector<real4<FLT_T>> rsasa<FLT_T, OUT_FLT_T>::dot_spheres_params::create_dodecahedron_dots(FLT_T radius) const
    {
        const FLT_T ext_radius = radius + probe_radius_;
        const FLT_T phi = (1.0f + sqrt(5.0f)) / 2.0f;
        const FLT_T r_phi = 1 / phi;

        const FLT_T r_sqrt3 = 1 / sqrt(3.0f);

        const FLT_T v1 = r_sqrt3 * ext_radius;
        const FLT_T v2 = r_phi * v1;
        const FLT_T v3 = phi * v1;

        vector<real4> dots {
                {v1, v1, v1, radius},
                {v1, v1, -v1, radius},
                {v1, -v1, v1, radius},
                {v1, -v1, -v1, radius},
                {-v1, v1, v1, radius},
                {-v1, v1, -v1, radius},
                {-v1, -v1, v1, radius},
                {-v1, -v1, -v1, radius},
                {0, v2, v3, radius},
                {0, v2, -v3, radius},
                {0, -v2, v3, radius},
                {0, -v2, -v3, radius},
                {v2, v3, 0, radius},
                {v2, -v3, 0, radius},
                {-v2, v3, 0, radius},
                {-v2, -v3, 0, radius},
                {v3, 0, v2, radius},
                {v3, 0, -v2, radius},
                {-v3, 0, v2, radius},
                {-v3, 0, -v2, radius}
        };
        return dots;
    }

    template <typename FLT_T, typename OUT_FLT_T>
    const vector<real4<FLT_T>> rsasa<FLT_T, OUT_FLT_T>::dot_spheres_params::create_dodeca_icosa_dots(FLT_T radius) const
    {
        const FLT_T ext_radius = radius + probe_radius_;
        const FLT_T phi = (1.0f + sqrt(5.0f)) / 2.0f;
        const FLT_T r_phi = 1 / phi;

        const FLT_T r_sqrt3 = 1 / sqrt(3.0f);

        const FLT_T v1 = r_sqrt3 * ext_radius;
        const FLT_T v2 = r_phi * v1;
        const FLT_T v3 = phi * v1;


        const FLT_T ico_f = FLT_T((sqrt(5.0f) + 5) / 10 / sqrt(3 - 1 / pow(phi*cos(3 * pi / 10), 2)) * ext_radius);

        vector<real4> dots {
                {v1, v1, v1, radius},
                {v1, v1, -v1, radius},
                {v1, -v1, v1, radius},
                {v1, -v1, -v1, radius},
                {-v1, v1, v1, radius},
                {-v1, v1, -v1, radius},
                {-v1, -v1, v1, radius},
                {-v1, -v1, -v1, radius},
                {0, v2, v3, radius},
                {0, v2, -v3, radius},
                {0, -v2, v3, radius},
                {0, -v2, -v3, radius},
                {v2, v3, 0, radius},
                {v2, -v3, 0, radius},
                {-v2, v3, 0, radius},
                {-v2, -v3, 0, radius},
                {v3, 0, v2, radius},
                {v3, 0, -v2, radius},
                {-v3, 0, v2, radius},
                {-v3, 0, -v2, radius},

                {ico_f, 0, phi * ico_f, radius},
                {ico_f, 0, -phi * ico_f, radius},
                {-ico_f, 0, phi * ico_f, radius},
                {-ico_f, 0, -phi * ico_f, radius},
                {0, phi * ico_f, ico_f, radius},
                {0, phi * ico_f, -ico_f, radius},
                {0, -phi * ico_f, ico_f, radius},
                {0, -phi * ico_f, -ico_f, radius},
                {phi * ico_f, ico_f, 0, radius},
                {phi * ico_f, -ico_f, 0, radius},
                {-phi * ico_f, ico_f, 0, radius},
                {-phi * ico_f, -ico_f, 0, radius}
        };
        return dots;
    }

    //////////////////////////////////////////////////////////////////////////
    /// Explicit instantiations for the supported algorithm configurations.
    template struct rsasa_creator < float, float, algorithm, rsasa_dot_spheres >;
    template struct rsasa_creator < float, double, algorithm, rsasa_dot_spheres >;
    template struct rsasa_creator < double, float, algorithm, rsasa_dot_spheres >;
    template struct rsasa_creator < double, double, algorithm, rsasa_dot_spheres >;

    template struct rsasa < float, float>::dot_spheres_params;
    template struct rsasa < float, double>::dot_spheres_params;
    template struct rsasa < double, float>::dot_spheres_params;
    template struct rsasa < double, double>::dot_spheres_params;

    template struct rsasa_creator < float, float, cl_base<float>, rsasa_dot_spheres >;
    template struct rsasa_creator < float, double, cl_base<float>, rsasa_dot_spheres >;
    template struct rsasa_creator < double, float, cl_base<double>, rsasa_dot_spheres >;
    template struct rsasa_creator < double, double, cl_base<double>, rsasa_dot_spheres >;

    template struct rsasa < float, float>::dot_spheres_cl_params;
    template struct rsasa < float, double>::dot_spheres_cl_params;
    template struct rsasa < double, float>::dot_spheres_cl_params;
    template struct rsasa < double, double>::dot_spheres_cl_params;
}
}