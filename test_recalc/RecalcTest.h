#pragma once

#include <vector>
#include <unordered_map>

#include "utils.hpp"
#include "resaxs.hpp"
#include "run_saxs_alg.hpp"

namespace resaxs
{

class RecalcTest
{
public:
    static std::unordered_map<std::string, algorithm::saxs_enum> alg_map;
};

template <typename FLT_T, typename FLT4_T>
class recalc_test
{
public:
    recalc_test(unsigned int n_bodies, int wf_size);
    recalc_test(const alg_test_data<FLT_T, FLT4_T> & data, int wf_size);
    ~recalc_test() {}
    void run(const std::string & alg_spec, const std::string & dev_spec, unsigned int n_steps, bool test = false);

private:
    struct alg_test_data : public resaxs::alg_test_data<FLT_T, FLT4_T>
    {
        typedef resaxs::alg_test_data<FLT_T, FLT4_T> base;
        
        alg_test_data(unsigned long seed, unsigned int n_bodies) :
            resaxs::alg_test_data<FLT_T, FLT4_T>(seed, n_bodies), change_len_(int(n_bodies * 0.4)) {}
        alg_test_data(const resaxs::alg_test_data<FLT_T, FLT4_T> & data) :
            resaxs::alg_test_data<FLT_T, FLT4_T>(data), change_len_(int(this->v_bodies_.size() * 0.4)) {}

        int change_len_;

        //virtual void set_args(saxs_algorithm<FLT_T, FLT4_T> & algorithm) {}
        virtual void move_bodies(int & index, int & length);
    } data_;

    int wf_size_;

    void initialize();
};

} // namespace
