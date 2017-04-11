#ifndef SASA_HPP
#define SASA_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2015 Lubo Antonov
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

#include <vector>
#include <unordered_map>

#include "resaxs.hpp"

namespace resaxs
{

    template <typename FLT_T>
    class sasa_atomic
    {
        using real4 = typename real_type<FLT_T>::real4;

    public:
        /// SASA
        /// \param[in] atoms Vector of touples of 4 floating-point numbers; the first 3 are the coordinates of an atom, the 4th is the radius.
        /// \param[in] probe_radius Radius of the probe to roll on the surface.
        /// \param[in] density Sampling density per A^2.
        template <typename OUT_FLT_T>
        void get_rsasa(const std::vector<real4> &atoms, FLT_T probe_radius, FLT_T density, std::vector<OUT_FLT_T> &rsasa);

    private:
        const std::vector<real4> create_sphere_dots(FLT_T radius, FLT_T density, FLT_T probe_radius) const;
        void create_sphere_dots(FLT_T density, FLT_T probe_radius);
        std::vector<FLT_T> get_radii(const std::vector<real4> &atoms) const;

        std::unordered_map<FLT_T, size_t> radii2type_;
        std::vector<std::vector<real4>> sphere_dots_;
        FLT_T density_;
        std::vector<FLT_T> radii_map_;
    };
} // namespace

#endif // SASA_HPP