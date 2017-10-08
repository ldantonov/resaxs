#ifndef CALC_SAXS_HPP
#define CALC_SAXS_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2014-2015 Lubo Antonov
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

#include <vector>
#include <string>
#include <memory>
#include "resaxs.hpp"
#include "utils.hpp"
#include "host_debye.hpp"
#include "saxs_algorithm.hpp"
#include "saxs_profile.hpp"

namespace resaxs
{

    template <typename FLT_T>
    struct calc_saxs
    {
        using real4 = typename real_type<FLT_T>::real4;

        enum verbose_levels
        {
            QUIET = 0,
            NORMAL,
            DETAILS,
            DEBUG
        };

        struct profile_param
        {
        public:
            profile_param(const FLT_T value, bool fit) : value_(value), fit_(fit) {}
            profile_param(const profile_param&) = default;

            operator const FLT_T () const { return value_; }

        private:
            bool fit_;
            FLT_T value_;
        };

        struct profile_params
        {
        public:
            profile_param scale_;           // scale of the profile relative to the reference
            profile_param water_weight_;    // weight parameter for the water layer
            saxs_profile<FLT_T> ref_profile;       // reference (e.g. experimental) profile
        };

        calc_saxs(const std::vector<std::string> & bodies_filenames, const std::string & exe_base_path, bool atomic, FLT_T q_min, FLT_T q_max, unsigned int q_n,
            profile_params params, verbose_levels verbose_lvl);
        calc_saxs(const calc_saxs & other);
        void set_verbose_level(verbose_levels verbose_lvl = NORMAL) { verbose_lvl_ = verbose_lvl; }

        void cl_saxs(algorithm::saxs_enum alg_pick, const std::string & dev_spec, unsigned int wf_size);
        void cl_saxs_ensemble(algorithm::saxs_enum alg_pick, const std::string &dev_spec, unsigned int wf_size);
        void host_saxs();
        void verify_result();

        std::vector<std::vector<real4>> v_models_;

        std::vector<real4> v_bodies_;
        std::vector<FLT_T> v_q_;
        unsigned int n_factors_;
        std::vector<FLT_T> t_factors_;
        std::vector<FLT_T> v_Iq_;

    private:
#if 0
        void parse_bodies(const std::string & filename);
#endif
        void load_pdb(const std::string & filename);
        const std::vector<std::vector<real4>> load_pdb_atomic(const std::string & filename) const;
        void load_pdb_atomic(const std::vector<std::string> & filenames);

        profile_params params_;
        verbose_levels verbose_lvl_;
    };

} // namepsace

#endif // CALC_SAXS_HPP