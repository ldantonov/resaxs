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

const char * gpu_saxs_2d_cl = CL_STRINGIFY(

\n#ifdef SAXS_USE_DOUBLE\n
    typedef double real;
    typedef double4 real4;
\n#else\n
    typedef float real;
    typedef float4 real4;
\n#endif\n

\n#define mad_int(x, y, z)    ((x) * (y) + (z))\n

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
\n#ifdef SAXS_USE_CL_LENGTH\n
    // the distance() (and length()) function is 4x slower on NVIDIA than the manual computation
    // (but it may be slightly more precise)
    real qr = q * distance(bodyx, bodyy);
\n#else\n
    real4 diff = bodyx - bodyy;
    real qr = q * native_sqrt(diff.x * diff.x + diff.y * diff.y + diff.z * diff.z);
\n#endif\n
    real v1 = native_divide(native_sin(qr), qr);
    return isnan(v1) ? 1.0f : v1;   // for q = 0 or r = 0, sin(qr)/qr = 1
}

//#include "saxs_common.cl"

// v_bodies must be aligned to WF size = local_size (also v_prev_bodies and t_bodies_f)
//
kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void map_factors(const int n_q,
    global float4* v_bodies, const int n_bodies,
    const global float* t_factors, const int n_factors,
    global float* t_body_factors,       // buffer to store a table of body factors for each q
    global float4* v_prev_bodies)   // the body positions for the previous step (it's simply initialized here)
{
    int idx = get_global_id(0);
    int n_bodies_aln = get_global_size(0);

    float4 body = v_bodies[idx];
    int factor_index = as_int(body.w);
    body = (idx < n_bodies) ? body : (float4)(0.0f);

\n#ifdef SAXS_USE_CL_LENGTH\n
    // we need to zero the .w component when using the built-in vector functions
    body.w = 0.0f;
    v_bodies[idx] = body;
\n#endif\n

    v_prev_bodies[idx] = body;  // save a copy in the vector of previous positions

    for (int k = 0; k < n_q; k++)
    {
        // populate a table of form factors with axes (body index, q index)
        t_body_factors[k * n_bodies_aln + idx] = (idx < n_bodies) ? t_factors[k * n_factors + factor_index] : 0.0f;
    }
}

//
// the global NDRange MUST be (aligned body count, block count (=aligned body count/GROUP_SIZE), q count)
// the local NDRange MUST be (GROUP_SIZE, 1, 1)
//
kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void margin_sum_small1(const global float* v_q,
            const global float4* v_bodies,                  // aligned to GROUP_SIZE
            const global float* t_body_factors,             // aligned to GROUP_SIZE
            global float* t_int_terms)                      // aligned to GROUP_SIZE
{
    int idx = get_global_id(0);             // body index
    int idy = get_global_id(1);             // v body index / GROUP_SIZE = block index
    int idz = get_global_id(2);             // q index
    int n_bodies_aln = get_global_size(0);  // should be the aligned body count
    int n_blocks = get_global_size(1);      // should be the number of vertical blocks of GROUP_SIZE
    //int gidx = get_group_id(0);             // x index of the workgroup
    //int lid = get_local_id(0);              // body index within the workgroup

    float q = v_q[idz];
    float4 bodyx = v_bodies[idx];

    // unrolling the loop for better performance
    float inner_sum = 0.0f;
    for (int y = idy * GROUP_SIZE; y < (idy + 1) * GROUP_SIZE; y += 4)
    {
        float v1 = sinc_qr(q, bodyx, v_bodies[y]);

        inner_sum = fma(t_body_factors[mad_int(idz, n_bodies_aln, y)], v1, inner_sum);
        v1 = sinc_qr(q, bodyx, v_bodies[y+1]);
        inner_sum = fma(t_body_factors[mad_int(idz, n_bodies_aln, y+1)], v1, inner_sum);
        v1 = sinc_qr(q, bodyx, v_bodies[y+2]);
        inner_sum = fma(t_body_factors[mad_int(idz, n_bodies_aln, y+2)], v1, inner_sum);
        v1 = sinc_qr(q, bodyx, v_bodies[y+3]);
        inner_sum = fma(t_body_factors[mad_int(idz, n_bodies_aln, y+3)], v1, inner_sum);
    }

    t_int_terms[idz * n_bodies_aln * n_blocks + idy * n_bodies_aln + idx] = inner_sum;
}

//
// the global NDRange MUST be (aligned body count, q count)
// the local NDRange MUST be (GROUP_SIZE, 1)
//
kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void margin_sum_small2(
            const global float* t_body_factors,             // aligned to GROUP_SIZE
            const global float* t_int_terms,                // aligned to GROUP_SIZE
            global float* t_margin_Iq)                      // aligned to GROUP_SIZE
{
    int idx = get_global_id(0);             // body index
    int idy = get_global_id(1);             // q index
    int n_bodies_aln = get_global_size(0);  // should be the aligned body count
    int n_v_groups = n_bodies_aln / GROUP_SIZE;

    // unrolling the loop for better performance
    float inner_sum = 0.0f;
    for (int y = 0; y < n_v_groups; y++)
    {
        inner_sum += t_int_terms[idy * n_bodies_aln * n_v_groups + y * n_bodies_aln + idx];
    }

    t_margin_Iq[mad_int(idy, n_bodies_aln, idx)] = inner_sum * t_body_factors[mad_int(idy, n_bodies_aln, idx)];
}

//----
//
// the global NDRange MUST be (aligned body count, q count)
// the local NDRange MUST be (GROUP_SIZE, 1)
//
kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void margin_sum_small(const global float* v_q,
            const global float4* v_bodies, int n_bodies,    // aligned to GROUP_SIZE
            const global float* t_body_factors,             // aligned to GROUP_SIZE
            global float* t_margin_Iq)                      // aligned to GROUP_SIZE
{
    int idx = get_global_id(0);             // body index
    int idy = get_global_id(1);             // q index
    int n_bodies_aln = get_global_size(0);  // should be the aligned body count
    //int n_q = get_global_size(1);           // should be the q count
    //int gidx = get_group_id(0);             // x index of the workgroup
    //int lid = get_local_id(0);              // body index within the workgroup

    float q = v_q[idy];
    float4 bodyx = v_bodies[idx];

    // unrolling the loop for better performance
    float inner_sum = 0.0f;
    for (int y = 0; y < n_bodies; y += 4)
    {
        float v1 = sinc_qr(q, bodyx, v_bodies[y]);

        inner_sum = fma(t_body_factors[mad_int(idy, n_bodies_aln, y)], v1, inner_sum);
        v1 = sinc_qr(q, bodyx, v_bodies[y+1]);
        inner_sum = fma(t_body_factors[mad_int(idy, n_bodies_aln, y+1)], v1, inner_sum);
        v1 = sinc_qr(q, bodyx, v_bodies[y+2]);
        inner_sum = fma(t_body_factors[mad_int(idy, n_bodies_aln, y+2)], v1, inner_sum);
        v1 = sinc_qr(q, bodyx, v_bodies[y+3]);
        inner_sum = fma(t_body_factors[mad_int(idy, n_bodies_aln, y+3)], v1, inner_sum);
    }

    t_margin_Iq[mad_int(idy, n_bodies_aln, idx)] = inner_sum * t_body_factors[mad_int(idy, n_bodies_aln, idx)];
}


kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void recalc_margins_sum(const global float* v_q,
            const global float4* v_bodies,
            const global float* t_body_factors,
            int upd_start, int upd_end,
            const global float4* v_prev_bodies,
            global float* t_margin_Iq)
{
    int idx = get_global_id(0);
    int idy = get_global_id(1);             // q index
    int n_bodies_aln = get_global_size(0);  // should be the aligned body count

    int page_base = idy * n_bodies_aln;     // the index at which the current 'q' page starts
    float q = v_q[idy];
    if (isnan(0 / q))     // for q = 0, the sum never changes
    {
        t_margin_Iq[page_base + idx] = 0.0f;
        return;
    }

    float mult = idx < upd_start ? 2.0f : 1.0f;
    mult = idx >= upd_end ? 2.0f : mult;

    float4 prev_bodyx = v_prev_bodies[idx];
    float4 bodyx = v_bodies[idx];

    float inner_delta = 0.0f;
    for (int y = upd_start; y < upd_end; y += 4)
    {
        inner_delta += t_body_factors[page_base + y] * (sinc_qr(q, bodyx, v_bodies[y]) - sinc_qr(q, prev_bodyx, v_prev_bodies[y]));
        inner_delta += t_body_factors[page_base + y + 1] * (sinc_qr(q, bodyx, v_bodies[y + 1]) - sinc_qr(q, prev_bodyx, v_prev_bodies[y + 1]));
        inner_delta += t_body_factors[page_base + y + 2] * (sinc_qr(q, bodyx, v_bodies[y + 2]) - sinc_qr(q, prev_bodyx, v_prev_bodies[y + 2]));
        inner_delta += t_body_factors[page_base + y + 3] * (sinc_qr(q, bodyx, v_bodies[y + 3]) - sinc_qr(q, prev_bodyx, v_prev_bodies[y + 3]));

        //local float4 prev_bodyy;
        //local float4 bodyy;
        /*local float y_factor;
        if (get_local_id(0) == 0)
        {
            //prev_bodyy = v_prev_bodies[y];
            //bodyy = v_bodies[y];
            y_factor = t_body_factors[ii + y];//t_bodies_f[k * n_bodies + y];
        }*/
        //barrier(CLK_LOCAL_MEM_FENCE);
        //float fac = y_factor;
    }

    // save the delta for this column of the matrix
    t_margin_Iq[page_base + idx] = inner_delta * t_body_factors[page_base + idx] * mult;
}

//
// the global NDRange MUST be (aligned body count, q count)
// the local NDRange MUST be (GROUP_SIZE, 1)
//
kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void sum_local_phase1(const global float* t_margin_Iq,  // matrix of margin sums for each body, per q (aligned n(bodies), n(q)) 
            global float* buf_sum)                      // matrix to pass to phase 2 (n(q), aligned n(bodies) / GROUP_SIZE)
{
    int idx = get_global_id(0);             // body index
    int idy = get_global_id(1);             // q index
    int n_bodies_aln = get_global_size(0);  // should be the aligned body count
    int n_q = get_global_size(1);           // should be the q count
    int gidx = get_group_id(0);             // x index of the workgroup
    int lid = get_local_id(0);              // body index within the workgroup

    local float scratch[GROUP_SIZE];
    scratch[lid] = t_margin_Iq[idy * n_bodies_aln + idx];
    barrier(CLK_LOCAL_MEM_FENCE);

    // do reduction for the workgroup in shared mem
    for (int s = GROUP_SIZE / 2; s > 0; s >>= 1) 
    {
        if(lid < s) 
        {
            scratch[lid] += scratch[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // write the result for this workgroup to the internal buffer
    if (lid == 0)
        buf_sum[gidx * n_q + idy] = scratch[0];
}

//
// the global NDRange MUST be (q count)
// the local NDRange should be (1)
//
kernel_exec(1, float)
void sum_local_phase2(const global float* buf_sum,  // matrix computed in phase 1 (n(q), aligned n(bodies) / GROUP_SIZE)
            const int n_parts,                      // aligned n(bodies) / GROUP_SIZE
            global float* out_v_Iq)                 // result vector of I(q)
{
    int idx = get_global_id(0);     // q index
    int n_q = get_global_size(0);   // should be the q count

    // sum our column of the matrix
    float sum = 0.0f;
    for (int i = 0; i < n_parts; i++)
    {
        sum += buf_sum[i * n_q + idx];
    }

    // save the total the the I(q) vector
    out_v_Iq[idx] = sum;
}

//
// the global NDRange should be (aligned moved count)
// the local NDRange should be (GROUP_SIZE)
kernel_exec(GROUP_SIZE, float4)
void save_prev_bodies(const global float4* v_bodies,
            int upd_offset,
            global float4* v_prev_bodies)
{
    int idx = get_global_id(0) + upd_offset;
    v_prev_bodies[idx] = v_bodies[idx];
}

//
// Adjusts the computed I(q) by the marginal deltas for the bodies for a given q.
// Each work item is assigned to a index of q.
//
kernel_exec(1, float)
void adjust_Iq(const global float* t_margin_Iq,
            int n_bodies,
            global float* out_v_Iq)
{
    int idx = get_global_id(0);     // global work item id is q index
    int n_q = get_global_size(0);   // should be q count

    float adj = 0.0f;
    for (int i = 0; i < aligned_num(n_bodies, GROUP_SIZE); i++)
    {
        adj += t_margin_Iq[mad_int(idx, aligned_num(n_bodies, GROUP_SIZE), i)];
    }
    out_v_Iq[idx] += adj; // save the adjusted I(q)
}

);
