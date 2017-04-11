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

#ifdef SAXS_USE_DOUBLE
    typedef double real;
    typedef double4 real4;
#else
    typedef float real;
    typedef float4 real4;
#endif

#define mad_int(x, y, z)    ((x) * (y) + (z))

int aligned_num(int num, int stride)
{
    return ((num - 1) / stride + 1) * stride;
}

int count_strides(int num, int stride)
{
    return (num - 1) / stride + 1;
}

real sinc_qr(real q, real4 bodyx, real4 bodyy)
{
#ifdef SAXS_USE_CL_LENGTH
    // the distance() (and length()) function is 4x slower on NVIDIA than the manual computation
    // (but it may be slightly more precise)
    real qr = q * distance(bodyx, bodyy);
#else
    real4 diff = bodyx - bodyy;
    real qr = q * native_sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
#endif
    real v1 = native_divide(native_sin(qr), qr);
    return isnan(v1) ? 1.0f : v1;   // for q = 0 or r = 0, sin(qr)/qr = 1
}


