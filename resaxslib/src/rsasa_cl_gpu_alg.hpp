///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2015 Lubo Antonov
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

#ifndef RE_RSASA_CL_GPU_ALG_HPP
#define RE_RSASA_CL_GPU_ALG_HPP

#include "rsasa_algorithm.hpp"
#include "re_cl_algorithm_base.hpp"

namespace resaxs {
namespace algorithm {

    ///////////////////////////////////////////////////////////////////////////////
    //  SAXS Page-Tile algorithm with hydration layer for GPUs, using a square block and page decomposition and local memory.
    //
    //      Specialized to float, but can be templetized for double, etc.
    //
    template <typename FLT_T, typename OUT_FLT_T>
    class rsasa_cl_gpu : public i_rsasa<FLT_T, OUT_FLT_T, cl_base<FLT_T>>
    {
        typedef real_type<FLT_T> num_type;
        typedef typename num_type::real4 real4;

        typedef cl_base<FLT_T> alg_base;
        typedef i_rsasa<FLT_T, OUT_FLT_T, alg_base> base;
        typedef typename rsasa<FLT_T, OUT_FLT_T>::dot_spheres_cl_params params_type;

    public:
        rsasa_cl_gpu() : n_atoms_aligned(0) {}

        virtual void initialize(const rsasa_params *);

        /// RSASA (OVERRIDE)
        /// \param[in] atoms Vector of touples of 4 floating-point numbers; the first 3 are the coordinates of an atom, the 4th is an atom-type index.
        /// \param[out] rsasa RSASA for each atom.
        virtual void calc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa);


        /// SASA
        /// \param[in] atoms Vector of touples of 4 floating-point numbers; the first 3 are the coordinates of an atom, the 4th is an atom-type index.
        /// \param[out] rsasa RSASA for each atom.
        virtual void recalc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa);

    private:
        const unsigned int max_dots_per_sphere = 32;

        void create_sphere_dots(const params_type params);
        const std::vector<FLT_T> get_radii(const std::vector<real4> &atoms) const;
        //const std::vector<FLT4_T> get_dots(const std::vector<FLT4_T> &atoms) const;

        params_type params_;
        std::vector<std::vector<real4>> sphere_dots_;
        std::vector<FLT_T> radii_map_;
        std::vector<real4> dots_map_;
        size_t n_atoms_aligned;     // Number of atoms aligned to max_dots_per_sphere

        void load_program();
        void set_args(const cl::CommandQueue &old_queue);
        void do_recalc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa);

        cl::Program program_;

        cl::Kernel kernel_calc_rsasa_;

        enum calc_rsasa_kernel_args
        {
            ka_calc_t_dots_map,
            ka_calc_v_bodies,
            ka_calc_v_radii,
            ka_calc_v_accessible_dots
        };

        Buffer<cl_uint> b_active_dots_;

        Buffer<real4> b_dots_map_;
        Buffer<FLT_T> b_radii_;
        Buffer<real4> b_atoms_;
    };

}
}   // namespace

#endif // RE_RSASA_CL_GPU_ALG_HPP
