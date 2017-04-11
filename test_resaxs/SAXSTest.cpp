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

#include "SAXSTest.hpp"

#include <random>

#include <algorithm>
#include <iostream>
#include <iomanip>
#include <memory>
#include <cassert>
#include <fstream>
#include <iterator>
#include <limits>

//#include "Context.hpp"
#include "FPCompare.hpp"
#include "utils.hpp"

#include "resaxs.hpp"
#include "host_debye.hpp"

#include "re_cl_algorithm_base.hpp"

using namespace std;

namespace resaxs
{

void SAXSTest::initialize()
{
    if (!file_root_.empty())
        initialize_from_files();
    else
    {
        if (length_ == 0)
            length_ = 1888;

        n_factors = 21;
        v_q.resize(51); // 51
        v_bodies.resize(length_);  // 1888
        t_factors.resize(n_factors * v_q.size());    //21*51
        v_Iq.resize(v_q.size());

        populate();
    }
}

void SAXSTest::initialize_from_files()
{
    assert(!file_root_.empty());

    std::ifstream file_q_vec(file_root_ + ".q_vec.txt");
    verify(!file_q_vec.bad(), SAXS_TEST_FILE_NOT_FOUND, file_root_ + "q_vec.txt file not found");
    std::copy(std::istream_iterator<float>(file_q_vec), std::istream_iterator<float>(), std::back_inserter(v_q));

    std::ifstream file_ff_db(file_root_ + ".ff_db.txt");
    verify(!file_ff_db.bad(), SAXS_TEST_FILE_NOT_FOUND, file_root_ + "ff_db.txt file not found");
    int n_q = int(v_q.size());
    parse_factors(file_ff_db, n_q);
    n_factors = int(t_factors.size() / v_q.size());

    std::ifstream file_pos_matrix(file_root_ + ".pos_matrix.txt");
    verify(!file_pos_matrix.bad(), SAXS_TEST_FILE_NOT_FOUND, file_root_ + ".pos_matrix.txt file not found");
    parse_bodies(file_pos_matrix);
    length_ = int(v_bodies.size());

    v_Iq.resize(v_q.size());
}

void SAXSTest::parse_factors(std::ifstream & stream, int & n_vec)
{
    char next = stream.peek();
    while (!stream.eof() && n_vec > 0)
    {
        switch(next)
        {
        case '[':
            stream.get();
            parse_factors(stream, n_vec);
            break;
        case ']':
            stream.get();
            n_vec--;
            return;
        case ',':
            stream.get();
            break;
        default:
            if (isalnum(next))
            {
                float value;
                stream >> value;
                t_factors.push_back(value);
            }
            else
                stream.get();
            break;
        }
        next = stream.peek();
    }
}

//
// Set the index in the body data (the space is for a float, so we need to trick the compiler).
//
inline void set_factor_index(cl_float4 & body, int index)
{
    memcpy(&body.s[3], &index, sizeof(int));
}

void SAXSTest::parse_bodies(std::ifstream & stream)
{
    while (!stream.eof())
    {
        cl_float4 body;
        int index;
        stream >> body.s[0] >> body.s[1] >> body.s[2] >> index;
        if (stream.good())
        {
            set_factor_index(body, index);
            v_bodies.push_back(body);
        }
    }
}

void SAXSTest::parse_Iqq(std::ifstream & stream)
{
    v_Iq.clear();
    for (unsigned int i = 0; i < v_q.size(); i++)
    {
        float q, Iq;
        stream >> q >> Iq;
        if (stream.good())
            v_Iq.push_back(Iq);
        else
            break;
    }
}

class rand_body_func
{
public:
    rand_body_func(std::mt19937 & engine, int n_factors) : rand_engine(engine),
        rand_body_coord(-200,200), rand_factor_index(0, n_factors - 1) {}

    cl_float4 operator()()
    {
        cl_float4 body;
        body.s[0] = rand_body_coord(rand_engine);
        body.s[1] = rand_body_coord(rand_engine);
        body.s[2] = rand_body_coord(rand_engine);
        int index = rand_factor_index(rand_engine);
        set_factor_index(body, index);

        return body;
    }

private:
    std::mt19937 & rand_engine;
    std::uniform_real_distribution<float> rand_body_coord;
    std::uniform_int_distribution<int> rand_factor_index;
};

class rand_factor_func
{
public:
    rand_factor_func(std::mt19937 & engine) : rand_engine(engine), rand_factor(0, 40) {}

    float operator()()
    {
        float factor = rand_factor(rand_engine);
        return factor;
    }

private:
    std::mt19937 & rand_engine;
    std::uniform_real_distribution<float> rand_factor;
};

void SAXSTest::populate()
{
    // populate q vector (from 0.0 to 0.75 in 0.015 increments)
    float q = 0.0;
    for (std::vector<float>::iterator i = v_q.begin(); i != v_q.end(); i++)
    {
        (*i) = q;
        q += 0.015f;
    }

    // engine for random numbers (always generates the same sequence)
    std::mt19937 rand_engine;
    rand_engine.seed(42);

    // generate random bodies
    rand_body_func rand_body(rand_engine, n_factors);
    std::generate(v_bodies.begin(), v_bodies.end(), rand_body);

    /*std::cout << "Bodies:" << std::endl;
    for (auto i = v_bodies.begin(); i != v_bodies.end(); i++)
    {
        std::cout << "  0:" << i->s[0] << "  1:" << i->s[1] << "  2:" << i->s[2] << "  Index:" << get_factor_index(*i) << std::endl;
    }*/
    /*int idx = 0; q = 0.0f;
    for (std::vector<cl_float4>::iterator i = v_bodies.begin(); i != v_bodies.end(); i++)
    {
        i->s[0] = q; q += 0.015f;
        i->s[1] = q; q += 0.015f;
        i->s[2] = q; q += 0.015f;
        i->s[3] = 0; idx++;
    }*/

    // generate random factors (always the same sequence)
    rand_factor_func rand_factor(rand_engine);
    std::generate(t_factors.begin(), t_factors.end(), rand_factor);
    /*q = 0.0f;
    for (std::vector<float>::iterator i = t_factors.begin(); i != t_factors.end(); i++)
    {
        (*i) = q;q += 0.02f;
    }*/
}

void move_bodies(cl_float4* v_bodies, int n_bodies, rand_body_func & rand_body)
{
    for (int i = 0; i < n_bodies; i++)
    {
        cl_float4 body = rand_body();
        body.s[3] = v_bodies[i].s[3];   // preserve the form factor index
        v_bodies[i] = body;
    }
}

void SAXSTest::host_saxs(int repeats, int change_len, CalcSaxsCurveFn calc_fn)
{
    if (repeats > 0 && file_root_.empty())
    {
        std::mt19937 rand_engine;
        std::uniform_int_distribution<int> rand_index(0, (int)v_bodies.size() - change_len - 1);

        // average over iterations
        rand_engine.seed(43);
        std::cout << "Randomizing indices.";
        for (int i = 0; i < repeats; i++)
        {
            int index = rand_index(rand_engine);
            rand_body_func rand_body(rand_engine, n_factors);
            move_bodies(&v_bodies[index], change_len, rand_body);
        }
        std::cout << std::endl;
    }

    clock_t t0 = clock();
    calc_fn(&v_q.front(), int(v_q.size()), &v_bodies.front(), int(v_bodies.size()), &t_factors.front(), n_factors, &v_Iq.front());
    clock_t t1 = clock();
    std::cout << "Host CPU full SAXS calculation time: " << double(t1-t0) * 1000 / CLOCKS_PER_SEC  << "ms" << std::endl << std::endl;
}

void SAXSTest::cl_saxs(algorithm::saxs_enum alg_pick, const std::string & dev_spec, int repeats, int change_len)
{
    std::vector<dev_id> devices;
    parse_dev_spec(dev_spec, devices);

    using alg_base = algorithm::cl_base<float>;
    using saxs_class = algorithm::i_saxs<float, alg_base>;

    unique_ptr<saxs_class> saxs_alg(algorithm::saxs<float>::template create<alg_base>(alg_pick));
    auto & saxs_params = saxs_alg->access_params();
    saxs_params.initialize(v_q, v_bodies, devices, wf_size_);

    auto & water_params = saxs_params.get_implicit_water_params();
    water_params.set_water_weight(0);

    saxs_alg->initialize();

    clock_t t0 = clock();
    saxs_alg->calc_curve(v_bodies, v_Iq);
    clock_t t1 = clock();

    std::cout << "OpenCL full SAXS calculation time: " << double(t1 - t0) * 1000 / CLOCKS_PER_SEC << "ms" << std::endl << std::endl;

    if (repeats > 0)
    {
        std::mt19937 rand_engine;
        std::uniform_int_distribution<int> rand_index(0, (int)v_bodies.size() - change_len - 1);

        rand_engine.seed(43);
        clock_t t_total = 0;
        std::cout << "Random moves..." << std::endl;
        clock_t t2 = clock();
        for (int i = 0; i < repeats; i++)
        {
            int index = rand_index(rand_engine);
            if (file_root_.empty())
            {
                rand_body_func rand_body(rand_engine, n_factors);
                move_bodies(&v_bodies[index], change_len, rand_body);
            }
            saxs_alg->recalc_curve(v_bodies, index, change_len, v_Iq);
        }
        t_total = clock() - t2;
        std::cout << std::endl << "OpenCL SAXS step time: " << double(t_total) * 1000 / CLOCKS_PER_SEC / repeats  << "ms/iteration" << std::endl << std::endl;
    }

//     auto & rsasa = water_params.access_rsasa();
// 
//     ofstream log("rsasa.log");
//     for_each(rsasa.begin(), rsasa.end(), [&log](FLT_T x){log << x << endl; });
}

void print_Iqq(const std::vector<float> & Iqq)
{
    std::cout << "I(q) = " << std::endl;
    std::cout << std::setiosflags(std::ios_base::fixed);
    for (std::vector<float>::const_iterator i = Iqq.begin(); i != Iqq.end(); i++)
    {
        float Iq = (*i);

        std::cout << std::setw(20) << Iq;
    }

    std::cout << std::endl << std::endl;
}

void SAXSTest::run(const std::string & dev_spec, bool test)
{
    initialize();

    int change_len = int(length_ * 0.4); // 740

    std::vector<float> host_Iq;
    if (test)
    {
        if (file_root_.empty())
        {
            run("dbl_host");
            host_Iq.resize(v_Iq.size());
            host_Iq.assign(v_Iq.begin(), v_Iq.end());   // copy Iq computed on the host

            initialize();
        }
        else
        {
            std::ifstream file_Iq_vec(file_root_ + ".Iq_vec.txt");
            verify(!file_Iq_vec.bad(), SAXS_TEST_FILE_NOT_FOUND, file_root_ + ".Iq_vec.txt file not found");
            parse_Iqq(file_Iq_vec);
            host_Iq.resize(v_Iq.size());
            host_Iq.assign(v_Iq.begin(), v_Iq.end());   // copy Iq computed on the host
            std::cout << "*** Precalculated protein:" << std::endl << std::endl;
            print_Iqq(host_Iq);
        }
    }

    std::cout << "*** Executing with device spec: " << dev_spec << std::endl << std::endl <<
        std::resetiosflags(std::ios_base::fixed);

    if (dev_spec.compare("dbl_host") == 0)
    {
        host_saxs(repeats_, change_len, host_debye<double, float, cl_float4>::calc_curve);
    }
    else if (dev_spec.compare("host") == 0)
    {
        host_saxs(repeats_, change_len, host_debye<float, float, cl_float4>::calc_curve);
    }
    else
    {
        cl_saxs(algorithm::saxs_enum::saxs_gpu_pt_wf, dev_spec, repeats_, change_len);
    }
//     else if (dev_spec.compare("gpu") == 0)
//     {
//         cl_saxs(saxs_algorithm_factory::gpu_saxs_2d_block_local, dev_spec, repeats_, change_len);
//     }
//     else if (dev_spec.compare("gpu_old") == 0)
//     {
//         cl_saxs(saxs_algorithm_factory::gpu_saxs_2d, "gpu", repeats_, change_len);
//     }
//     else if (dev_spec.compare("gpu_wf") == 0)
//     {
//         cl_saxs(saxs_algorithm_factory::gpu_saxs_pt_wf, "cpu", repeats_, change_len);
//     }
//     else if (dev_spec.compare("cpu") == 0)
//     {
//         cl_saxs(saxs_algorithm_factory::gpu_saxs_2d_block_local, dev_spec, repeats_, change_len);
//     }
//     else
//     {
//         cl_saxs(saxs_algorithm_factory::gpu_saxs_2d_block_local, dev_spec, repeats_, change_len);
//     }

    if (test)
        print_distance(v_Iq, host_Iq);

    print_Iqq(v_Iq);
}

} // namespace
