#ifndef RANDOM_HELPERS_HPP
#define RANDOM_HELPERS_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2014 Lubo Antonov
//
//    This file is part of ACCSAXS.
//
//    ACCSAXS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    any later version.
//
//    ACCSAXS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with ACCSAXS.  If not, see <http://www.gnu.org/licenses/>.
//
///////////////////////////////////////////////////////////////////////////////

#ifdef __GNUC__
#include <tr1/random>
#else
#include <random>
#endif

namespace resaxs
{

template <typename FLT_T>
class rand_factor_func
{
public:
    rand_factor_func(std::mt19937 & engine) : rand_engine_(engine), rand_factor_(0, 20) {}

    FLT_T operator()()
    {
        FLT_T factor = rand_factor_(rand_engine_);
        return factor;
    }

private:
    std::mt19937 & rand_engine_;
    std::uniform_real_distribution<FLT_T> rand_factor_;
};

template <typename FLT_T, typename FLT4_T>
class rand_body_func
{
public:
    rand_body_func(std::mt19937 & engine, int n_factors) : rand_engine(engine),
        rand_body_coord(-100,100), rand_factor_index(0, n_factors - 1) {}

    FLT4_T operator()()
    {
        FLT4_T body;
        body.s[0] = rand_body_coord(rand_engine);
        body.s[1] = rand_body_coord(rand_engine);
        body.s[2] = rand_body_coord(rand_engine);
        int index = rand_factor_index(rand_engine);
        set_factor_index(body, index);

        return body;
    }

private:
    std::mt19937 & rand_engine;
    std::uniform_real_distribution<FLT_T> rand_body_coord;
    std::uniform_int_distribution<int> rand_factor_index;
};

} // namespace

#endif // RANDOM_HELPERS_HPP