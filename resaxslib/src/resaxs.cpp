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

#include "../include/resaxs.hpp"

namespace resaxs
{

//
//  Verify the condition and throw an error exception, if false.
//      condition           - condition to verify
//      err                 - error ID to use on failure
//      err_str             - optional error string to include in the exception
//
void verify(bool condition, int err, const char* err_str)
{
    if (!condition)
    {
        if (err_str != NULL)
            throw error(err, err_str);
        else
            throw error(err, "RESAXS::error");
    }
}
void verify(bool condition, int err, const std::string & err_str)
{
    verify(condition, err, err_str.c_str());
}

}
