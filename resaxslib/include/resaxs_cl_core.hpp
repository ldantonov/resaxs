#ifndef RESAXS_RESAXS_CL_CORE_HPP
#define RESAXS_RESAXS_CL_CORE_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2015 Lubo Antonov
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

#define CL_USE_DEPRECATED_OPENCL_1_1_APIS

// Enable exceptions from the OpenCL C++ wrappers
#define __CL_ENABLE_EXCEPTIONS

#ifdef _WIN32
// Disable the problematic min and max macros
#define NOMINMAX
//#pragma push_macro("max")
//#pragma push_macro("min")
//#undef max
//#undef min
#endif

// Disable some useless warnings for OpenCL headers
#ifdef _MSC_VER
#pragma warning( push )
#pragma warning( disable : 4290 )   // disable warning about the ignored exception specs in VC++ (seems to be pre-VS2013 only)
#elif __GNUC__ * 100 + __GNUC_MINOR__ > 406 // GCC 4.6 needed
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcomment"  // disable the "comment-in-comment" warning for opencl.h from CUDA 
#endif

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include <CL/cl.hpp>
#endif

// Restore the warning state
#ifdef _MSC_VER
#pragma warning( pop )
#elif __GNUC__ * 100 + __GNUC_MINOR__ > 406
#pragma GCC diagnostic pop
#endif

// If for some reason the Win32 macros are needed, you can reenable them with the following:
// #ifdef _WIN32
// #pragma pop_macro("min")
// #pragma pop_macro("max")
// #endif

// This macro will allow stringifying text that has commas in it
//#ifdef _MSC_VER
#define CL_STRINGIFY(...) #__VA_ARGS__
//#else
//#define CL_STRINGIFY_1(x...) #x
//#define CL_STRINGIFY(x...) CL_STRINGIFY_1(x)
//#endif

#endif // RESAXS_RESAXS_CL_CORE_HPP