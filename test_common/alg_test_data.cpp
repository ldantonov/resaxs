///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011 Lubo Antonov
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

#include <iostream>
#include <memory>
#include <string>

#include "run_saxs_alg.hpp"
#include "saxs_params_io.hpp"
#include "random_helpers.hpp"
#include "re_cl_algorithm_base.hpp"

using namespace std;

namespace resaxs
{

template <typename FLT_T, typename FLT4_T>
alg_test_data<FLT_T, FLT4_T>::alg_test_data(unsigned long seed, unsigned int n_bodies) : show_timings_(true)
{
    rand_engine_.seed(seed);
    try
    {
        saxs_params_io::load_reals("form_factors-q_vector.dat", v_q_);
        saxs_params_io::load_reals("form_factors-two_body_model.dat", t_factors_);
    }
    catch (const error & )
    {
        // saxs param files failed to load, so generate internal values
        init_internal_saxs_params();
    }

    n_factors_ = (unsigned int)(t_factors_.size() / v_q_.size());
    intensity_.resize(v_q_.size());

    v_bodies_.resize(n_bodies);  // 1888
    init_bodies();
}

template <typename FLT_T, typename FLT4_T>
alg_test_data<FLT_T, FLT4_T>::alg_test_data(const alg_test_data & other) : rand_engine_(other.rand_engine_),
    v_bodies_(other.v_bodies_), v_q_(other.v_q_), n_factors_(other.n_factors_), t_factors_(other.t_factors_), intensity_(other.intensity_),
    show_timings_(other.show_timings_)
{
}

template <typename FLT_T, typename FLT4_T>
void alg_test_data<FLT_T, FLT4_T>::init_internal_saxs_params()
{
    // populate q vector (from 0.0 to 0.75 in 0.015 increments)
    v_q_.resize(51);
    float next_q = 0.0;
    for (auto & q : v_q_)
    {
        q = next_q;
        next_q += 0.015f;
    }

    t_factors_.resize(21 * v_q_.size());    // 21 * 51
    // generate random factors (always the same sequence)
    rand_factor_func<FLT_T> rand_factor(rand_engine_);
    generate(t_factors_.begin(), t_factors_.end(), rand_factor);
}

template <typename FLT_T, typename FLT4_T>
void alg_test_data<FLT_T, FLT4_T>::init_bodies()
{
    // generate random bodies
    rand_body_func<FLT_T, FLT4_T> rand_body(rand_engine_, n_factors_);
    generate(v_bodies_.begin(), v_bodies_.end(), rand_body);
}


template <typename FLT_T, typename FLT4_T>
void alg_test_data<FLT_T, FLT4_T>::cl_saxs(algorithm::saxs_enum alg_pick, const std::string & dev_spec, int repeats, unsigned int wf_size)
{
    std::vector<dev_id> devices;
    parse_dev_spec(dev_spec, devices);

    using alg_base = algorithm::cl_base<FLT_T>;
    using saxs_class = algorithm::i_saxs<FLT_T, alg_base>;

    unique_ptr<saxs_class> saxs_alg(algorithm::saxs<FLT_T>::template create<alg_base>(alg_pick));

    auto & saxs_params = saxs_alg->access_params();
    saxs_params.initialize(v_q_, v_bodies_, devices, wf_size);

    auto & water_params = saxs_params.get_implicit_water_params();
    water_params.set_water_weight(0);

    saxs_alg->initialize();

    clock_t t0 = clock();
    saxs_alg->calc_curve(v_bodies_, intensity_);
    clock_t t1 = clock();

    if (show_timings_)
        std::cout << "OpenCL full SAXS calculation time: " << double(t1 - t0) * 1000 / CLOCKS_PER_SEC << "ms\n" << std::endl;

    if (repeats > 0)
    {
        clock_t t_total = 0;
        std::cout << "Random moves..." << std::endl;
        clock_t t2 = clock();
        for (int i = 0; i < repeats; i++)
        {
            int index, len;
            move_bodies(index, len);
            saxs_alg->recalc_curve(v_bodies_, index, len, intensity_);
        }
        t_total = clock() - t2;
        if (show_timings_)
            std::cout << std::endl << "OpenCL SAXS step time: " << double(t_total) * 1000 / CLOCKS_PER_SEC / repeats << "ms/iteration\n" << std::endl;
    }

    //     auto & rsasa = water_params.access_rsasa();
    // 
    //     ofstream log("rsasa.log");
    //     for_each(rsasa.begin(), rsasa.end(), [&log](FLT_T x){log << x << endl; });
}

template <typename FLT_T, typename FLT4_T>
void alg_test_data<FLT_T, FLT4_T>::host_saxs(int repeats, typename host_debye<FLT_T, FLT_T, FLT4_T>::CalcSaxsCurveFn calc_fn)
{
    if (repeats > 0)
    {
        cout << "Randomizing moves..." << endl;
        for (int i = 0; i < repeats; i++)
        {
            int index, len;
            move_bodies(index, len);
        }
    }

    clock_t t0 = clock();
    calc_fn(&v_q_.front(), int(v_q_.size()), &v_bodies_.front(), int(v_bodies_.size()), &t_factors_.front(), n_factors_, &intensity_.front());
    clock_t t1 = clock();
    if (show_timings_)
        std::cout << "Host CPU full SAXS calculation time: " << double(t1-t0) * 1000 / CLOCKS_PER_SEC  << "ms" << std::endl << std::endl;
}


template struct alg_test_data<float, cl_float4>;
template struct alg_test_data<double, cl_double4>;

} // namespace