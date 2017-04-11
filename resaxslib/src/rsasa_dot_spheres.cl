///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2013-2015 Lubo Antonov
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

const char * rsasa_dot_spheres_cl = CL_STRINGIFY(

\n#ifdef USE_DOUBLE\n
    typedef double real;
    typedef double4 real4;
\n#else\n
    typedef float real;
    typedef float4 real4;
\n#endif\n

\n#define mad_int(x, y, z)    ((x) * (y) + (z))\n

inline int aligned_num(int num, int stride)
{
    return ((num - 1) / stride + 1) * stride;
}

inline int count_strides(int num, int stride)
{
    return (num - 1) / stride + 1;
}

inline real len2(const real4 v)
{
    return v.x * v.x + v.y * v.y + v.z * v.z;
}

\n#define R_SQRT3     0.577350269\n
\n#define PHI         1.618033989\n
\n#define R_PHI       0.618033989\n

\n#define DOD1        R_PHI * R_SQRT3\n
\n#define DOD2        PHI * R_SQRT3\n
\n#define ICO1        0.525731112\n
\n#define ICO2        PHI * ICO1\n

constant real4 platonic_solid_dots[] = {
        {R_SQRT3, R_SQRT3, R_SQRT3, 1},
        {R_SQRT3, R_SQRT3, -R_SQRT3, 1},
        {R_SQRT3, -R_SQRT3, R_SQRT3, 1},
        {R_SQRT3, -R_SQRT3, -R_SQRT3, 1},
        {-R_SQRT3, R_SQRT3, R_SQRT3, 1},
        {-R_SQRT3, R_SQRT3, -R_SQRT3, 1},
        {-R_SQRT3, -R_SQRT3, R_SQRT3, 1},
        {-R_SQRT3, -R_SQRT3, -R_SQRT3, 1},
        {0, DOD1, DOD2, 1},
        {0, DOD1, -DOD2, 1},
        {0, -DOD1, DOD2, 1},
        {0, -DOD1, -DOD2, 1},
        {DOD1, DOD2, 0, 1},
        {DOD1, -DOD2, 0, 1},
        {-DOD1, DOD2, 0, 1},
        {-DOD1, -DOD2, 0, 1},
        {DOD2, 0, DOD1, 1},
        {DOD2, 0, -DOD1, 1},
        {-DOD2, 0, DOD1, 1},
        {-DOD2, 0, -DOD1, 1},

        {ICO1, 0, ICO2, 1},
        {ICO1, 0, -ICO2, 1},
        {-ICO1, 0, ICO2, 1},
        {-ICO1, 0, -ICO2, 1},
        {0, ICO2, ICO1, 1},
        {0, ICO2, -ICO1, 1},
        {0, -ICO2, ICO1, 1},
        {0, -ICO2, -ICO1, 1},
        {ICO2, ICO1, 0, 1},
        {ICO2, -ICO1, 0, 1},
        {-ICO2, ICO1, 0, 1},
        {-ICO2, -ICO1, 0, 1},
};

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void calc_rsasa(
    global const real4 *restrict v_dots_map,
    global const real4 *restrict v_bodies,
    global const real *restrict v_radii,
    global unsigned int *restrict v_accessible_dots
    )
{
    const size_t body_idx = get_group_id(0);  // body index

    const real4 body = v_bodies[body_idx];
    //const real b_radius = v_radii[body_idx];

    const size_t lid = get_local_id(0);     // dot #
    real4 sdot = isfinite(body.w) ? v_dots_map[as_int(body.w) * GROUP_SIZE + lid] : body.w;
    //real4 sdot = platonic_solid_dots[lid];// *(b_radius + PROBE_RADIUS);
    sdot += body;                           // shift the generic dot relative to the position of the body
    //real4 sdot = v_dots_map[lid];
    //sdot = fma(sdot, 1 + PROBE_RADIUS));

    unsigned int bit = isfinite(sdot.x);    // if this is an alignment body, or dot - the count will be 0

    // process bodies in groups
    local real4 bodies[GROUP_SIZE];
    for (unsigned int i = 0; i < N_BODIES; i += GROUP_SIZE)
    {
        real4 b = v_bodies[i + lid];            // stream in the bodies in the group
        const real radius = v_radii[i + lid];   // and the radii
        b.w = (PROBE_RADIUS + radius) * (PROBE_RADIUS + radius);    // store the r^2 in the 4th element
        bodies[lid] = b;                        // stream the modified bodies to local memory
        barrier(CLK_LOCAL_MEM_FENCE);           // finish loading before the reading loop

        for (unsigned int j = 0; j < GROUP_SIZE; ++j)
        {
            const real4 b2 = bodies[j];         // x,y,z - position; w - r^2
            const real4 diff = sdot - b2;
            const real dist2 = len2(diff);

            bit = isgreater(b2.w - dist2, 0.0001f) ? 0 : bit;
        }
        barrier(CLK_LOCAL_MEM_FENCE);           // finish reading before loading in the next iteration
    }

    /*for (unsigned int i = 0; i < N_BODIES; i++)
    {
        const real radius2 = /*v_radii[i];*//*(PROBE_RADIUS + v_radii[i]) * (PROBE_RADIUS + v_radii[i]);
        const real4 diff = sdot - v_bodies[i];
        const real dist2 = len2(diff);

        bit = (radius2 - dist2 > 0.0001) ? 0 : bit;
    }*/

    // Count the number of accessible dots in the workgroup
    local unsigned int bits[GROUP_SIZE];
    bits[lid] = bit;
    barrier(CLK_LOCAL_MEM_FENCE);           // make sure all bits are stored

    // do a multi-stage reduction for the workgroup in local memory
    // in the last stage only the first work item is active
    for (int s = GROUP_SIZE >> 1; s > 0; s >>= 1)
    {
        if (lid < s)
        {
            bits[lid] += bits[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);       // make sure all sums are stored
    }

    v_accessible_dots[body_idx] = bits[0];
}

);
