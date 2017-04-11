#ifndef SAXS_PARAMS_IO_HPP
#define SAXS_PARAMS_IO_HPP

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

#include <string>
#include <fstream>
#include <algorithm>
#include <iterator>

#include "resaxs.hpp"

#define SAXS_TEST_FILE_NOT_FOUND        -1000   // a test file could not be located

namespace resaxs
{

class saxs_params_io
{
public:
    template <typename FLT_T>
    static void load_pure_vector(const std::string & file_name, std::vector<FLT_T> & out_vec)
    {
        std::ifstream file(file_name);
        verify(file.is_open(), SAXS_TEST_FILE_NOT_FOUND, file_name + " file not found");
        std::copy(std::istream_iterator<FLT_T>(file), std::istream_iterator<FLT_T>(), std::back_inserter(out_vec));
    }

    template <typename FLT_T>
    static void load_reals(const std::string & file_name, std::vector<FLT_T> & out_reals)
    {
        std::ifstream file(file_name);
        verify(file.is_open(), SAXS_TEST_FILE_NOT_FOUND, file_name + " file not found");

        while (!file.eof())
        {
            // this weird construct is to solve an anomaly with operator>> below
            // the >> operator doesn't advance the stream position, so it enters and endless loop
            if (!isdigit(file.peek()))
            {
                file.get();
                continue;
            }
            FLT_T elem;
            file >> elem;
            if (!file)
                continue;
        
            out_reals.push_back(elem);
        }
    }
};

}

#endif // SAXS_PARAMS_IO_HPP