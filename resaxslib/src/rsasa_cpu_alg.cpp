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

#include "rsasa_cpu_alg.hpp"
#include "constants.hpp"

#include "bb_grid.hpp"
#include "ff_coef.hpp"
#include "utils.hpp"

using namespace std;

namespace resaxs {
namespace algorithm {

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cpu_alg<FLT_T, OUT_FLT_T>::initialize(const rsasa_params *base_params)
    {
        auto params = static_cast<const typename rsasa<FLT_T, OUT_FLT_T>::dot_spheres_params *>(base_params);

        // Generate spheres of dots for the radii of the known atom types
        create_sphere_dots(*params);

        algorithm::initialize();
    }

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cpu_alg<FLT_T, OUT_FLT_T>::create_sphere_dots(const typename rsasa<FLT_T, OUT_FLT_T>::dot_spheres_params params)
    {
        // If the density did not change, no need to generate the dot spheres again
        if (!sphere_dots_.empty() && params_ == params)
            return;

        params_ = params;

        if (radii_map_.empty())
            atomic_form_factors<FLT_T>::calc_radii(radii_map_);

        sphere_dots_.resize(radii_map_.size());
        transform(radii_map_.begin(), radii_map_.end(), sphere_dots_.begin(),
            [this](FLT_T radius) { return params_.create_dots(radius); });
    }

    template <typename FLT_T> inline
        bool is_intersecting(const num3<FLT_T> &sph_center1, const num3<FLT_T> &sph_center2, const FLT_T radius1, const FLT_T radius2)
    {
        const auto sum_r2 = (radius1 + radius2) * (radius1 + radius2);
        const auto dist2 = distance2(sph_center1, sph_center2);

        return sum_r2 - dist2 > 0.0001;
    }

    template <typename FLT_T> inline
        bool spheres_collide(const num3<FLT_T> &probe_center, FLT_T probe_radius, const vector<typename real_type<FLT_T>::real4> &atoms,
        const vector<unsigned int> &neighbors, const vector<FLT_T> &radii)
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
        num3<FLT_T> fmad(const typename real_type<FLT_T>::real4 &p1, FLT_T m, const typename real_type<FLT_T>::real4 &p2)
    {
        return{ p1.s[0] * m + p2.s[0], p1.s[1] * m + p2.s[1], p1.s[2] * m + p2.s[2] };
    }

    template <typename FLT_T> inline
        num3<FLT_T> fadd(const typename real_type<FLT_T>::real4 &p1, const typename real_type<FLT_T>::real4 &p2)
    {
        return{ p1.s[0] + p2.s[0], p1.s[1] + p2.s[1], p1.s[2] + p2.s[2] };
    }

    /// Get the radii of atoms based on their types.
    ///
    template <typename FLT_T, typename OUT_FLT_T> inline
    const std::vector<FLT_T> rsasa_cpu_alg<FLT_T, OUT_FLT_T>::get_radii(const vector<real4> &atoms) const
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

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cpu_alg<FLT_T, OUT_FLT_T>::calc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa)
    {
        verify(this->initialized(), error::SAXS_ALG_NOT_INITIALIZED);
        verify(rsasa.size() >= atoms.size(), error::SAXS_INVALID_ARG);

        const size_t n_atoms = atoms.size();
        //rsasa.resize(n_atoms);

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
            const FLT_T radius = atom_radius + 2 * params_.probe_radius_ + max_radius;

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
                    const FLT_T radius_sum2 = radius_sum1 + 2 * params_.probe_radius_;
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

                bool collides = spheres_collide(probe_center, params_.probe_radius_, atoms, neighbors1, radii);
                if (!collides)
                    collides = spheres_collide(probe_center, params_.probe_radius_, atoms, neighbors2, radii);
                if (!collides)
                    ++dot_num;
            }
            rsasa[i] = static_cast<OUT_FLT_T>(dot_num) / sdots.size();
        }

        this->state = this->alg_computing;
    }

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cpu_alg<FLT_T, OUT_FLT_T>::recalc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa)
    {
        verify(this->computing(), error::SAXS_ARGS_NOT_SET);

        calc_rsasa(atoms, rsasa);
    }

    template class rsasa_cpu_alg<float, float>;
    template class rsasa_cpu_alg<float, double> ;
    template class rsasa_cpu_alg<double, double>;
    template class rsasa_cpu_alg<double, float> ;

}
} // namespace
