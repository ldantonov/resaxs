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

#ifndef RE_RSASA_CPU_ALG_HPP
#define RE_RSASA_CPU_ALG_HPP

#include "rsasa_algorithm.hpp"

namespace resaxs {
namespace algorithm {

    ///////////////////////////////////////////////////////////////////////////////
    //  SAXS Page-Tile algorithm with hydration layer for GPUs, using a square block and page decomposition and local memory.
    //
    //      Specialized to float, but can be templetized for double, etc.
    //
    template <typename FLT_T, typename OUT_FLT_T>
    class rsasa_cpu_alg : public i_rsasa < FLT_T, OUT_FLT_T >
    {
        typedef real_type<FLT_T> num_type;
        typedef typename num_type::real4 real4;

        typedef i_rsasa<FLT_T, OUT_FLT_T> base;
        typedef typename rsasa<FLT_T, OUT_FLT_T>::dot_spheres_params params_type;

    public:
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
        void create_sphere_dots(const params_type params);
        const std::vector<FLT_T> get_radii(const std::vector<real4> &atoms) const;

        params_type params_;
        std::vector<std::vector<real4>> sphere_dots_;
        std::vector<FLT_T> radii_map_;
    };

}
}   // namespace

#endif // RE_RSASA_CPU_ALG_HPP

