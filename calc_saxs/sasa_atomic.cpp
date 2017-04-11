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

#include "sasa_atomic.hpp"
#include "constants.hpp"

#include "bb_grid.hpp"
#include "ff_coef.hpp"
#include "utils.hpp"

using namespace std;

namespace resaxs
{
    template <typename FLT_T>
    const std::vector<real4<FLT_T>> sasa_atomic<FLT_T>::create_sphere_dots(FLT_T radius, FLT_T density, FLT_T probe_radius) const
    {
        std::vector<real4> dots;
        const FLT_T n_sect = 2 * static_cast<FLT_T>(pi)* radius * sqrt(density);    // number of sectors
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
                dots.push_back({ x, y, z, 0 });
            }
        }

        return dots;
    }

    template <typename FLT_T>
    void sasa_atomic<FLT_T>::create_sphere_dots(FLT_T density, FLT_T probe_radius)
    {
        // If the atoms and density did not change, no need to generate the dot spheres again
        // (assumes that if atom radii do not change).
        if (!radii2type_.empty() && density_ == density)
            return;
        
        radii2type_.clear();
        sphere_dots_.clear();
        
        density_ = density;

        if (radii_map_.empty())
            atomic_form_factors<FLT_T>::calc_radii(radii_map_);

        for (auto radius : radii_map_)
        {
            size_t type = radii2type_.size();
            radii2type_[radius] = type;
            sphere_dots_.push_back(create_sphere_dots(radius, density, probe_radius));
            //const FLT_T ratio = (radius + 1.8f) / radius;
            //for_each(sphere_dots_[type].begin(), sphere_dots_[type].end(), [ratio](FLT4_T &dot) {dot.s[0] *= ratio; dot.s[1] *= ratio; dot.s[2] *= ratio; });
        }
    }

    template <typename FLT_T> inline
    bool is_intersecting(const num3<FLT_T> &sph_center1, const num3<FLT_T> &sph_center2, const FLT_T radius1, const FLT_T radius2)
    {
        const auto sum_r2 = (radius1 + radius2) * (radius1 + radius2);
        const auto dist2 = distance2(sph_center1, sph_center2);

        return sum_r2 - dist2 > 0.0001;
    }

    template <typename FLT_T> inline
    bool spheres_collide(const num3<FLT_T> &probe_center, FLT_T probe_radius, const vector<real4<FLT_T>> &atoms, const vector<unsigned int> &neighbors,
        const vector<FLT_T> &radii)
    {
        for (const auto neighbor : neighbors)
        {
            if (is_intersecting(probe_center, pos_3d<FLT_T>(atoms[neighbor]).data_, probe_radius, radii[neighbor]))
            {
                return true;
            }
        }
        return false;
    }

    template <typename FLT_T> inline
        num3<FLT_T> fmad(const real4<FLT_T> &p1, FLT_T m, const real4<FLT_T> &p2)
    {
        return { p1.s[0] * m + p2.s[0], p1.s[1] * m + p2.s[1], p1.s[2] * m + p2.s[2] };
    }

    template <typename FLT_T> inline
        num3<FLT_T> fadd(const real4<FLT_T> &p1, const real4<FLT_T> &p2)
    {
        return{ p1.s[0] + p2.s[0], p1.s[1] + p2.s[1], p1.s[2] + p2.s[2] };
    }

    /// Get the radii of atoms based on their types.
    ///
    template <typename FLT_T> inline
        std::vector<FLT_T> sasa_atomic<FLT_T>::get_radii(const vector<real4> &atoms) const
    {
        const size_t n_atoms = atoms.size();
        vector<FLT_T> radii(n_atoms);
        for (unsigned int i = 0; i < n_atoms; ++i)
        {
            unsigned int index = get_factor_index(atoms[i]);
            radii[i] = radii_map_[index];
        }
        return radii;
    }

    template <typename FLT_T>
    template <typename OUT_FLT_T>
    void sasa_atomic<FLT_T>::get_rsasa(const std::vector<real4> &atoms, FLT_T probe_radius, FLT_T density, std::vector<OUT_FLT_T> &rsasa)
    {
        const size_t n_atoms = atoms.size();
        rsasa.resize(n_atoms);

        // Generate spheres of dots for all radii present in the atoms
        create_sphere_dots(density, probe_radius);

        const vector<FLT_T> &radii = get_radii(atoms);

        // Put the positions in a voxel grid
        // Side of 6A should be bigger than the max radius + 2*probe radius, so it should improve matching speed
        voxel_grid<FLT_T> grid(static_cast<FLT_T>(6.0), bounding_box_3d<FLT_T>(atoms));
        for (unsigned int i = 0; i < n_atoms; ++i)
        {
            const auto &index = grid.get_virtual_index(atoms[i]); // we know this index is bounded, so use this call directly
            grid[index].push_back(i);
        }

        // Neighbor voxels - preallocate for performance
        vector<unsigned int> neighbors1(256), neighbors2(256);

        // compute the rsasa
        const FLT_T max_radius = 1.5f;
        for (unsigned int i = 0; i < n_atoms; ++i)
        {
            const FLT_T atom_radius = radii[i];
            const FLT_T radius = atom_radius + 2 * probe_radius + max_radius;

            bounding_box_3d<FLT_T> bbox_atom(atoms[i]);   // bounding box in angstrom around atom
            bbox_atom += radius;    // expand the box
            const num3<int> &li = grid.get_nearest_bounded_index(bbox_atom.lc_);   // grid index of the lower corner
            const num3<int> &ui = grid.get_nearest_bounded_index(bbox_atom.uc_);   // grid index of the upper corner

            // find the neighbor voxels
            neighbors1.clear();
            neighbors2.clear();
            const auto &end_it = grid.end(ui);  // let's not create this more than once
            for (auto it = grid.begin(li, ui); it != end_it; ++it)
            {
                 for (const auto pt_idx : *it)
                 {
                     const FLT_T radius_sum1 = atom_radius + radii[pt_idx];
                     const FLT_T radius_sum2 = radius_sum1 + 2 * probe_radius;
                     const FLT_T dist2 = distance2(atoms[i], atoms[pt_idx]);

                     if (dist2 < radius_sum1 * radius_sum1)
                         neighbors1.push_back(pt_idx);
                     else if (dist2 < radius_sum2 * radius_sum2)
                         neighbors2.push_back(pt_idx);
                 }
            }

            //const FLT_T ratio = (atom_radius + probe_radius) / atom_radius;
            const vector<real4> &sdots = sphere_dots_[get_factor_index(atoms[i])];     // the idx is the same for factors and dot spheres

            int dot_num = 0;
            for (const auto &sdot : sdots)
            {
                //const num3<FLT_T> & probe_center = fmad(sdot, ratio, atoms[i]);
                const num3<FLT_T> & probe_center = fadd<FLT_T>(sdot, atoms[i]);

                bool collides = spheres_collide(probe_center, probe_radius, atoms, neighbors1, radii);
                if (!collides)
                    collides = spheres_collide(probe_center, probe_radius, atoms, neighbors2, radii);
                if (!collides)
                    ++dot_num;
            }
            rsasa[i] = static_cast<OUT_FLT_T>(dot_num) / sdots.size();
        }
    }

    template void sasa_atomic < float>::get_rsasa<float>(const std::vector<cl_float4> &atoms, float probe_radius, float density, std::vector<float> &rsasa);
    template void sasa_atomic < double>::get_rsasa<float>(const std::vector<cl_double4> &atoms, double probe_radius, double density, std::vector<float> &rsasa);

    template class sasa_atomic < float> ;
    template class sasa_atomic < double> ;



} // namespace
