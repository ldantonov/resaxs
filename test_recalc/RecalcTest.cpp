#include "RecalcTest.h"

#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "random_helpers.hpp"

using namespace std;

namespace resaxs
{

std::unordered_map<std::string, algorithm::saxs_enum> RecalcTest::alg_map;

template <typename FLT_T, typename FLT4_T>
void move_bodies(FLT4_T* v_bodies, int n_bodies, rand_body_func<FLT_T, FLT4_T> & rand_body)
{
    for (int i = 0; i < n_bodies; i++)
    {
        FLT4_T body = rand_body();
        body.s[3] = v_bodies[i].s[3];   // preserve the form factor index
        v_bodies[i] = body;
    }
}

template <typename FLT_T, typename FLT4_T>
void recalc_test<FLT_T, FLT4_T>::alg_test_data::move_bodies(int & index, int & length)
{
    std::uniform_int_distribution<int> rand_len(0, change_len_);
    length = rand_len(this->rand_engine_);
    std::uniform_int_distribution<int> rand_index(0, (int)this->v_bodies_.size() - length - 1);
    index = rand_index(this->rand_engine_);

    rand_body_func<FLT_T, FLT4_T> rand_body(this->rand_engine_, this->n_factors_);
    resaxs::move_bodies(&this->v_bodies_[index], length, rand_body);
}

template <typename FLT_T, typename FLT4_T>
recalc_test<FLT_T, FLT4_T>::recalc_test(unsigned int n_bodies, int wf_size) :
    data_(42, n_bodies), wf_size_(wf_size)
{
    initialize();
}

template <typename FLT_T, typename FLT4_T>
recalc_test<FLT_T, FLT4_T>::recalc_test(const resaxs::alg_test_data<FLT_T, FLT4_T> & data, int wf_size) :
    data_(data), wf_size_(wf_size)
{
    initialize();
}

template <typename FLT_T, typename FLT4_T>
void recalc_test<FLT_T, FLT4_T>::initialize()
{
    data_.show_timings_ = true;

    if (RecalcTest::alg_map.empty())
    {
        RecalcTest::alg_map["pt_wf"] = algorithm::saxs_enum::saxs_gpu_pt_wf;
    }
}

template <typename FLT_T, typename FLT4_T>
void recalc_test<FLT_T, FLT4_T>::run(const string & alg_spec, const string & dev_spec, unsigned int n_steps, bool test)
{
    cout << "*** Executing with device spec: " << dev_spec << endl << endl <<
        resetiosflags(ios_base::fixed);

    if (alg_spec.compare("std") == 0)
    {
        data_.host_saxs(n_steps, host_debye<double, FLT_T, FLT4_T>::calc_curve);
    }
    else
    {
        try
        {
            auto algorithm = RecalcTest::alg_map.at(alg_spec);
            data_.cl_saxs(algorithm, dev_spec, n_steps, wf_size_);
        }
        catch(const out_of_range & e)
        {
            cerr << "!ERROR! - " << e.what() << endl;
        }
    }
}

template class recalc_test<float, cl_float4>;
//template class recalc_test<double , cl_double4>;

} // namespace
