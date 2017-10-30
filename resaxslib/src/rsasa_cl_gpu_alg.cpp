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

#include "rsasa_cl_gpu_alg.hpp"
#include "constants.hpp"

#include <sstream>
#include <fstream>

#include "bb_grid.hpp"
#include "ff_coef.hpp"
#include "utils.hpp"

using namespace std;

namespace resaxs {
namespace algorithm {
    #include "rsasa_dot_spheres.cl"

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cl_gpu<FLT_T, OUT_FLT_T>::initialize(const rsasa_params *base_params)
    {
        auto params = static_cast<const params_type *>(base_params);

        // Generate spheres of dots for the radii of the known atom types
        create_sphere_dots(*params);

        auto old_state = this->state;
        auto old_queue = this->queue_;

        alg_base::initialize(params_.devices_);

        load_program();
        set_args(old_queue);
        
        // If this is a reinitialization, we may be in a more advanced state
        if (old_state > this->state)
            this->state = old_state;
    }

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cl_gpu<FLT_T, OUT_FLT_T>::create_sphere_dots(const typename rsasa<FLT_T, OUT_FLT_T>::dot_spheres_cl_params params)
    {
        // If the density did not change, no need to generate the dot spheres again
        if (!sphere_dots_.empty() && params_ == params)
            return;

        params_ = params;

        if (radii_map_.empty())
            atomic_form_factors<FLT_T>::calc_radii(radii_map_);

        sphere_dots_.resize(radii_map_.size());
        transform(radii_map_.begin(), radii_map_.end(), sphere_dots_.begin(),
            [this](FLT_T radius) { return params_.create_dodeca_icosa_dots(radius); });

        dots_map_.resize(radii_map_.size() * max_dots_per_sphere);
        //dots_map_.resize(max_dots_per_sphere);
        const real4 sentinel_dot {-numeric_limits<FLT_T>::infinity(), -numeric_limits<FLT_T>::infinity(),
            -numeric_limits<FLT_T>::infinity(), -numeric_limits<FLT_T>::infinity()};

        for (size_t i = 0; i < radii_map_.size(); ++i)
        {
            const vector<real4> &atom_dots = sphere_dots_[i];
            for (unsigned int dot_i = 0; dot_i < max_dots_per_sphere; ++dot_i)
            {
                dots_map_[i * max_dots_per_sphere + dot_i] = (dot_i < atom_dots.size()) ? atom_dots[dot_i] : sentinel_dot;
            }
        }
        //dots_map_ = create_32_dots<FLT_T, FLT4_T>();
    }

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cl_gpu<FLT_T, OUT_FLT_T>::load_program()
    {
        verify(this->initialized(), error::SAXS_ALG_NOT_INITIALIZED);
        
        // add the wavefront size as a parameter to the program
        ostringstream all_options;
        all_options << "-D GROUP_SIZE=" << max_dots_per_sphere << " -D PROBE_RADIUS=" << params_.probe_radius_ << " -D N_BODIES=" << params_.n_atoms_ <<
            " -D N_DOTS=" << sphere_dots_[0].size();

        this->build_program(rsasa_dot_spheres_cl, program_, all_options.str());

        kernel_calc_rsasa_ = cl::Kernel(program_, "calc_rsasa");

        /*auto a = program_.getInfo<CL_PROGRAM_BINARIES>();
        string s = a[0];

        ofstream f("test.ptx", ios::trunc | ios::out | ios::binary);
        f.write(s.c_str(), s.size());*/
        this->state = this->alg_program_loaded;
    }

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cl_gpu<FLT_T, OUT_FLT_T>::set_args(const cl::CommandQueue &old_queue)
    {
        verify(this->program_loaded(), error::SAXS_PROGRAM_NOT_LOADED);
        
        // pre-init atoms to blanks to make sure extra atoms for alignment are properly initialized
        const real4 blank_atom {-numeric_limits<FLT_T>::infinity(), -numeric_limits<FLT_T>::infinity(), -numeric_limits<FLT_T>::infinity(), -numeric_limits<FLT_T>::infinity()};
        const size_t n_atoms = params_.n_atoms_;
        n_atoms_aligned = aligned_num(n_atoms, max_dots_per_sphere);
        vector<real4> atoms(n_atoms_aligned, blank_atom);
        b_atoms_.init_or_switch_context(old_queue, this->context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, atoms);

        b_dots_map_.init_or_switch_context(old_queue, this->context_, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, dots_map_);
        b_radii_.init_or_switch_context(old_queue, this->context_, CL_MEM_READ_ONLY, n_atoms_aligned);
        b_active_dots_.init_or_switch_context(old_queue, this->context_, CL_MEM_WRITE_ONLY, n_atoms_aligned);

        kernel_calc_rsasa_.setArg(ka_calc_t_dots_map, b_dots_map_);
        kernel_calc_rsasa_.setArg(ka_calc_v_bodies, b_atoms_);
        kernel_calc_rsasa_.setArg(ka_calc_v_radii, b_radii_);
        kernel_calc_rsasa_.setArg(ka_calc_v_accessible_dots, b_active_dots_);

        this->state = this->alg_args_set;
    }

    /// Get the radii of atoms based on their types.
    ///
    template <typename FLT_T, typename OUT_FLT_T>
    const std::vector<FLT_T> rsasa_cl_gpu<FLT_T, OUT_FLT_T>::get_radii(const vector<real4> &atoms) const
    {
        const size_t n_atoms = params_.n_atoms_;
        vector<FLT_T> radii(n_atoms_aligned, -numeric_limits<FLT_T>::infinity());   // -Inf radii for alignment atoms
        for (unsigned int i = 0; i < n_atoms; ++i)
        {
            unsigned int index = get_factor_index(atoms[i]);
            radii[i] = radii_map_[index];// +params_.probe_radius_;
            //radii[i] *= radii[i];
        }
        return radii;
    }

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cl_gpu<FLT_T, OUT_FLT_T>::calc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa)
    {
        verify(this->args_set(), error::SAXS_ARGS_NOT_SET);
        verify(rsasa.size() >= atoms.size(), error::SAXS_INVALID_ARG);

        const auto &radii = get_radii(atoms);
        b_radii_.write_from(radii, this->queue_);

        //if (rsasa.size() < atoms.size())
        //    rsasa.resize(atoms.size());
        do_recalc_rsasa(atoms, rsasa);

        this->state = this->alg_computing;
    }

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cl_gpu<FLT_T, OUT_FLT_T>::recalc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa)
    {
        verify(this->computing(), error::SAXS_NO_INITIAL_CALC);

        do_recalc_rsasa(atoms, rsasa);
    }

    template <typename FLT_T, typename OUT_FLT_T>
    void rsasa_cl_gpu<FLT_T, OUT_FLT_T>::do_recalc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa)
    {
        b_atoms_.write_from(atoms, this->queue_);

        // # of workgroups - max_dots_per_sphere, each workgroup is the dots for one atom
        this->queue_.enqueueNDRangeKernel(kernel_calc_rsasa_, cl::NullRange,
            cl::NDRange(n_atoms_aligned * max_dots_per_sphere), cl::NDRange(max_dots_per_sphere));

        const size_t n_atoms = params_.n_atoms_;
        vector<cl_uint> active_dots(n_atoms);
        b_active_dots_.read_to(active_dots, this->queue_);

        for (size_t i = 0; i < n_atoms; ++i)
        {
            unsigned int index = get_factor_index(atoms[i]);

            const vector<real4> &sdots = sphere_dots_[index];
            rsasa[i] = static_cast<OUT_FLT_T>(active_dots[i]) / sdots.size();
        }
    }

    template class rsasa_cl_gpu<float, float>;
    template class rsasa_cl_gpu<float, double>;
    template class rsasa_cl_gpu<double, float>;
    template class rsasa_cl_gpu<double, double>;

}
} // namespace
