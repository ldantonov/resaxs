#ifndef RESAXS_UTILS_HPP
#define RESAXS_UTILS_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011-2015 Lubo Antonov
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

#include <cstdlib>
#include <map>
#include "resaxs_cl_core.hpp"
#include "constants.hpp"

namespace resaxs {

struct dev_id;

//
//  Report details on the OpenCL platforms and devices to cout.
//  Useful for command line utils
//
void report_opencl_caps();

//
//  Parse a string with device specifications for algorithm execution.
//  NOTE: only a single dev spec is currenty supported.
//      dev_spec            - sequence of 'gpu | cpu | <platform>-<device>', separated by whitespace
//      dev_ids             - result usable for initializing algorithms
//
void parse_dev_spec(const std::string & dev_spec, std::vector<dev_id> & dev_ids);

std::vector<cl::Device> get_cl_devices(const std::vector<dev_id> & dev_ids);

//
// A simple implementation of the sinc function for use on the host side
//
template<class T> inline T sinc(T x)
{
    return sin(x) / x;
}

//
// Get the index out of the body data (it's coded as a float, so we need to trick the compiler).
//
inline unsigned int get_factor_index(const cl_float4 & body)
{
    unsigned int index;
    memcpy(&index, &body.s[3], sizeof(cl_float));
    return index;
}

inline unsigned int get_factor_index(const cl_double4 & body)
{
    unsigned long long index;
    memcpy(&index, &body.s[3], sizeof(cl_double));
    return (unsigned int)index;
}

//
// Access the index of the body into the FF table (the space is for a float, so we need to reinterpret the bits).
// memcpy, or char* is the only way not to break strict aliasing rules.
//
inline void set_factor_index(cl_float4 & body, unsigned int index)
{
    memcpy(&body.s[3], &index, sizeof(unsigned int));
}

inline void set_factor_index(cl_double4 & body, unsigned int index)
{
    unsigned long long l_index = index;
    memcpy(&body.s[3], &l_index, sizeof(unsigned long long));
}

//
// Align the number to the next largest increment of the stride.
//
template <typename T>
inline T aligned_num(T num, unsigned int stride)
{
    return ((num - 1) / stride + 1) * stride;
}

extern const std::map<std::string, atoms::atom_type> pdb_to_ff_simple_atom_map;

atoms::atom_type map_pdb_atom_to_ff_type(const std::string &atom_label, const std::string &res_type);

atoms::atom_type map_pdb_atom_to_ff_type(atoms::atom_type element, const std::string &atom_label, const std::string &res_type);

//
// Trim leading and trailing white space from a string
//
std::string trim(const std::string& str);

//
// Trim leading and trailing characters from a string
//
std::string trim(const std::string& str, const std::string& delim);

///
/// Split a string at a specified delimiter, returning a vector of strings
///
std::vector<std::string> split(const std::string &s, char delim);

} // namespace

#endif // RESAXS_UTILS_HPP
