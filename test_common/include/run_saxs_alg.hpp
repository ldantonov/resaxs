#ifndef RUN_SAXS_ALG_HPP
#define RUN_SAXS_ALG_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2014 Lubo Antonov
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
#include <ctime>
#include <random>
#include "resaxs.hpp"
#include "utils.hpp"
#include "host_debye.hpp"
#include "saxs_algorithm.hpp"

namespace resaxs
{

template <typename FLT_T, typename FLT4_T>
struct alg_test_data
{
    alg_test_data(unsigned long seed, unsigned int n_bodies);
    alg_test_data(const alg_test_data & other);

    bool show_timings_;

    void cl_saxs(algorithm::saxs_enum alg_pick, const std::string & dev_spec, int repeats, unsigned int wf_size);
    void host_saxs(int repeats, typename host_debye<FLT_T, FLT_T, FLT4_T>::CalcSaxsCurveFn calc_fn);

protected:
    std::mt19937 rand_engine_;
    std::vector<FLT4_T> v_bodies_;
    std::vector<FLT_T> v_q_;
    unsigned int n_factors_;
    std::vector<FLT_T> t_factors_;
    std::vector<FLT_T> v_Iq_;

    //virtual void set_args(saxs_algorithm<FLT_T, FLT4_T> & algorithm) {}
    virtual void move_bodies(int & index, int & length) {}

private:
    void init_internal_saxs_params();
    void init_bodies();
};

} // namespace

#endif // RUN_SAXS_ALG_HPP