#ifndef SAXSTEST_HPP
#define SAXSTEST_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011 Lubo Antonov
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

#include <string>
#include <vector>

#include "resaxs.hpp"
#include "host_debye.hpp"
#include "saxs_algorithm.hpp"

namespace resaxs
{

class SAXSTest
{
public:
    SAXSTest(int length, int repeats, int wf_size) : length_(length), repeats_(repeats), wf_size_(wf_size) { if (wf_size_ == 0) wf_size_ = SAXS_NV_FERMI_WF_SIZE; }
    SAXSTest(const std::string & file_root, int repeats, int wf_size) : repeats_(repeats), wf_size_(wf_size), file_root_(file_root) {}
    void initialize();
    void run(const std::string & dev_spec, bool test = false);

private:
    std::vector<float> v_q;
    std::vector<cl_float4> v_bodies;
    std::vector<float> t_factors;
    int n_factors;
    std::vector<float> v_Iq;
    int length_;
    int repeats_;
    int wf_size_;
    std::string file_root_;

    void initialize_from_files();
    void parse_factors(std::ifstream & stream, int & n_vec);
    void parse_bodies(std::ifstream & stream);
    void parse_Iqq(std::ifstream & stream);
    void populate();
    void host_saxs(int repeats, int change_len, CalcSaxsCurveFn calc_fn);
    void cl_saxs(algorithm::saxs_enum alg_pick, const std::string & dev_spec, int repeats, int change_len);
};

#define SAXS_TEST_FILE_NOT_FOUND        -1000   // a test file could not be located

} // namespace

#endif // SAXSTEST_HPP
