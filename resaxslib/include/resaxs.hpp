#ifndef RESAXS_RESAXS_HPP
#define RESAXS_RESAXS_HPP

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

#include <string>

#include "resaxs_cl_core.hpp"

// Wavefront size for some types devices
#define SAXS_NV_FERMI_WF_SIZE       64      // Default wavefront is the Fermi size
#define SAXS_AMD_WF_SIZE            64      // for RV8xx devices and later
#define SAXS_AMD_LOW_END_WF_SIZE    32      // for some AMD low-end devices

namespace resaxs {

    template <typename T>
    union num3
    {
        typedef T num;
        num s[3];
        struct
        {
            num x;
            num y;
            num z;
        };

        num3(num x = 0, num y = 0, num z = 0) : x(x), y(y), z(z) {}
        
        num & operator[](unsigned int dim)
        {
            return s[dim];
        }

        const num & operator[](unsigned int dim) const
        {
            return s[dim];
        }

        bool operator==(const num3 &right) const
        {
            return x == right.x && y == right.y && z == right.z;
        }

        bool operator!=(const num3 & right) const
        {
            return !operator==(right);
        }

        num3 & operator+=(num delta)
        {
            x += delta;
            y += delta;
            z += delta;
            return *this;
        }

        num3 & operator-=(num delta)
        {
            return operator+=(-delta);
        }

        num3 operator+(num delta) const
        {
            return num3(x + delta, y + delta, z + delta);
        }

        num3 operator-(num delta) const
        {
            return operator+(-delta);
        }

        num3 operator+(const num3 &right) const
        {
            return num3(x + right.x, y + right.y, z + right.z);
        }

        num3 operator*(num factor) const
        {
            return num3(x * factor, y * factor, z * factor);
        }

        template <typename Func>
        void apply(Func func)
        {
            s[0] = func(s[0], 0);
            s[1] = func(s[1], 1);
            s[2] = func(s[2], 2);
        }

        template <typename U, typename Func>
        num3<U> generate(Func func) const
        {
            return num3<U>(func(s[0], 0), func(s[1], 1), func(s[2], 2));
        }
    };

    template <typename FLT_T>
    union real_type
    {
    };

    template <>
    union real_type <float>
    {
        typedef float real;
        typedef cl_float4 real4;
        typedef num3<float> real3;
        typedef cl_float4 cl_real4;
        typedef cl_float3 cl_real3;
    };

    inline bool operator==(const cl_float4 & o1, const cl_float4 & o2)
    {
        return memcmp(&o1, &o2, sizeof(o1));
    }

    template <>
    union real_type <double>
    {
        typedef double real;
        typedef cl_double4 real4;
        typedef num3<double> real3;
        typedef cl_double4 cl_real4;
        typedef cl_double3 cl_real3;
    };

    // Simplifying the syntax for the vector4 types
    template <typename FLT_T>
    using real4 = typename real_type<FLT_T>::real4;

    // types of acceleration devices supported
    enum device_type
    {
        CPU,
        GPU,
        ANY
    };

    // descriptor for a device
    // use report_opencl_caps() to see the full list on a system
    struct dev_id
    {
        long platform;          // platform number, as reported by OpenCL; -1 means any platform
        long device;            // device number, as reported by OpenCL; -1 means any device
        device_type type;       // if platform or device are -1, the type of device to look for
        
        bool operator==(const dev_id &other) const
        {
            return platform == other.platform && device == other.device && type == other.type;
        }
        bool operator!=(const dev_id &other) const
        {
            return !operator==(other);
        }
        std::string to_string() const
        {
            return std::to_string(platform) + '-' + std::to_string(device);
        }
    };


///////////////////////////////////////////////////////////////////////////////
//  Error handling
//

class error : public std::exception
{
private:
    int err_;
    std::string err_str_;
public:
    //  Create a new exception for a given error code and corresponding message.
    error(int err) : err_(err) {}
    error(int err, const char* err_str) : err_(err), err_str_(err_str) {}

    ~error() throw () {}

    //  Gets an error string associated with exception.
    //  Returns a pointer to the error message string.
    virtual const char* what() const throw ()
    {
        if (err_str_.empty())
            return "empty";
        else
            return err_str_.c_str();
    }

    //  Get the error code associated with the exception
    int err() const { return err_; }

    enum code
    {
        SAXS_INVALID_DEVICE =           -1,     // an invalid device was specified
        SAXS_ALG_NOT_INITIALIZED =      -2,     // the algorithm has not been initialized
        SAXS_INVALID_ARG =              -3,     // an invalid function argument was used
        SAXS_PROGRAM_NOT_LOADED =       -4,     // the program for the algorithm has not been loaded
        SAXS_ARGS_NOT_SET =             -5,     // the arguments for the algorithm have not been set
        SAXS_NO_INITIAL_CALC =          -6,     // the initial calc_curve has not been performed
    };
};

//
//  Verify the condition and throw an error exception, if false.
//      condition           - condition to verify
//      err                 - error ID to use on failure
//      err_str             - optional error string to include in the exception
//
void verify(bool condition, int err, const char* err_str = NULL);
void verify(bool condition, int err, const std::string & err_str);

}   // namespace

#endif  // RESAXS_RESAXS_HPP
