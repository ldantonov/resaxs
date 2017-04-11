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

#include <memory>
#include "resaxs.hpp"
#include "utils.hpp"
#include "host_debye.hpp"
#include "saxs_algorithm.hpp"

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

        calc_saxs(const std::string & bodies_filename, const std::string & exe_base_path, bool atomic, FLT_T q_min, FLT_T q_max, unsigned int q_n,
            FLT_T water_weight, verbose_levels verbose_lvl);
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
        void load_pdb_atomic(const std::string & filename);

        FLT_T water_weight_;
        verbose_levels verbose_lvl_;
    };

} // namepsace

#endif // CALC_SAXS_HPP