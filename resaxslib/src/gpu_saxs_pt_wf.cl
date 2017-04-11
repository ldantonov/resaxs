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

const char * gpu_saxs_pt_wf_cl = CL_STRINGIFY(

\n#ifdef SAXS_USE_DOUBLE\n
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

inline real sinc_qr(real q, real4 bodyx, real4 bodyy)
{
\n#ifdef SAXS_USE_CL_LENGTH\n
    // the distance() (and length()) function is 4x slower on NVIDIA than the manual computation
    // (but it may be slightly more precise)
    real qr = q * fast_distance(bodyx, bodyy);
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
    global real4 *restrict v_bodies,
    const int n_bodies,
    const global real *restrict t_factors,           // packed table of form factors (n_q x n_factors)
    const int n_factors,
    global real *restrict t_body_factors)       // buffer to store a table of body factors for each q
{
    int idx = get_global_id(0);
    int n_bodies_aln = get_global_size(0);

    real4 body = v_bodies[idx];
    int factor_index = as_int(body.w);

\n#ifdef SAXS_USE_CL_LENGTH\n
    // we need to zero the .w component when using the built-in vector functions
    body.w = 0.0f;
    v_bodies[idx] = body;
\n#endif\n

    for (int k = 0; k < n_q; k++)
    {
        // populate a table of form factors with axes (body index, q index)
        t_body_factors[k * n_bodies_aln + idx] = (idx < n_bodies) ? t_factors[k * n_factors + factor_index] : 0.0f;
    }
}

//
// The global NDRange MUST be (aligned body count, block count (=aligned body count/GROUP_SIZE), q count)
// The local NDRange MUST be (GROUP_SIZE, 1, 1)
//
// IMPORTANT NOTE: an OpenCL limitation does not allow a kernel to call another kernel if the latter is using local memory.
//          This is why we need to pass the local memory as parameters.
//
void do_calc_block(const global real *restrict v_q,
            const global real4 *restrict v_bodies,           // bodies vector        - aligned to GROUP_SIZE
            const global real *restrict t_factors,           // packed table of form factors (n_q x n_factors) - needed only for the water form factors
            const unsigned int n_factors,           // number of factors for each q in the t_factors table
            const global real *restrict t_body_factors,      // (factor, q) table    - aligned to GROUP_SIZE
            //! const real param_b,	- edited: switched to water FF calc on the host
            //const global real* v_water_factors,    // ++(q) vector of water form factors
            const global real *restrict v_body_surface,      // ++(body index) vector of surface area factor for each body - aligned to GROUP_SIZE
            const real water_weight,                // weight factor for the water contribution
            
            // local memory caches for workgroup use
            local real4 *restrict y_bodies,                  // bodies corresponding to the y-dimension of the block
            local real *restrict item_sum,                   // buffer for storing the vertical sum for each work item_sum
            local real *restrict y_factors,                  // form factors for the y-dimension bodies
            
            global real *restrict t_buf_blocks)              // internal buffer for storing the block sums
                                                    //                      - aligned to 4x4
{
    int idx = get_global_id(0);             // body index
    int idy = get_global_id(1);             // block index (= v body index / GROUP_SIZE)
    int idz = get_global_id(2);             // q index
    int n_bodies_aln = get_global_size(0);  // should be the aligned body count
    int n_blocks = get_global_size(1);      // should be the number of vertical blocks of GROUP_SIZE
    int lid = get_local_id(0);              // body index within the workgroup

    // skip the calculation for mirror blocks (upper diagonal) in the matrix
    int x_block = get_group_id(0);
    if (x_block > idy)
        return;

    real q = v_q[idz];
    //real water_factor = v_water_factors[idz] * water_weight;
    // water form factor: Fw(q)=Fw(0)*E(q), E(q)=sqrt(exp(-bq^2)), Fw(0)=3.5
    //real water_factor = 3.5 * native_sqrt(native_exp(-param_b * q * q)) * water_weight; - edited: switched to water FF calc on the host
    const real water_factor = t_factors[idz * n_factors + WATER_FF_INDEX] * water_weight;
    const global real* factors_page = t_body_factors + idz * n_bodies_aln; // pointer to the correct q page of factors
        
    // stream the data with the whole workgroup
    y_bodies[lid] = v_bodies[idy * GROUP_SIZE + lid];
    item_sum[lid] = factors_page[idx];      // preload the factors for the x-dimension bodies
    item_sum[lid] += v_body_surface[idx] * water_factor;    // add water contribution
    y_factors[lid] = factors_page[idy * GROUP_SIZE + lid];
    y_factors[lid] += v_body_surface[idy * GROUP_SIZE + lid] * water_factor;   // add water contributions

    real4 bodyx = v_bodies[idx];           // this is the body in the x-dimension for this work item

    // unrolling the loop for better performance
    real block_sum = 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);           // make sure the locals are loaded
    for (int y = 0; y < GROUP_SIZE; y += 4)
    {
        real v1 = sinc_qr(q, bodyx, y_bodies[y]);
        block_sum = fma(y_factors[y], v1, block_sum);
        v1 = sinc_qr(q, bodyx, y_bodies[y + 1]);
        block_sum = fma(y_factors[y + 1], v1, block_sum);
        v1 = sinc_qr(q, bodyx, y_bodies[y + 2]);
        block_sum = fma(y_factors[y + 2], v1, block_sum);
        v1 = sinc_qr(q, bodyx, y_bodies[y + 3]);
        block_sum = fma(y_factors[y + 3], v1, block_sum);
    }

    // multiply the sum by the x body factor and store back
    item_sum[lid] *= block_sum;
    barrier(CLK_LOCAL_MEM_FENCE);           // make sure all sums are stored

    // do a multi-stage reduction for the workgroup in local memory
    // in the last stage only the first work item is active
    for (int s = GROUP_SIZE >> 1; s > 0; s >>= 1) 
    {
        if(lid < s) 
        {
            item_sum[lid] += item_sum[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);       // make sure all sums are stored
    }

    // write the result for this workgroup to the internal buffer
    int n_blocks_aln = aligned_num(n_blocks, 4);
    if (lid == 0)
    {
        t_buf_blocks[idz * n_blocks_aln * n_blocks_aln + idy * n_blocks_aln + x_block] = item_sum[0];
        // duplicate the result for the mirror block (upper diagonal) in the matrix
        if (x_block != idy)
            t_buf_blocks[idz * n_blocks_aln * n_blocks_aln + x_block * n_blocks_aln + idy] = item_sum[0];
    }
}

//
// The global NDRange MUST be (aligned body count, block count (=aligned body count/GROUP_SIZE), q count)
// The local NDRange MUST be (GROUP_SIZE, 1, 1)
//
// IMPORTANT NOTE: an OpenCL limitation does not allow a kernel to call another kernel if the latter is using local memory.
//
kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void calc_block(const global real *restrict v_q,
            const global real4 *restrict v_bodies,          // bodies vector        - aligned to GROUP_SIZE
            const global real *restrict t_factors,           // packed table of form factors (n_q x n_factors) - needed only for the water form factors
            const unsigned int n_factors,           // number of factors for each q in the t_factors table
            const global real *restrict t_body_factors,     // (factor, q) table    - aligned to GROUP_SIZE
            //!const real param_b, - edited: switched to water FF calc on the host
            //const global real* v_water_factors,     // ++(q) vector of water form factors
            const global real *restrict v_body_surface,     // ++(body index) vector of surface area factor for each body - aligned to GROUP_SIZE
            const real water_weight,                // weight factor for the water contribution
            global real *restrict t_buf_blocks)             // internal buffer for storing the block sums
                                                    //                      - aligned to 4x4
{
    // Local memory caches for workgroup use.
    // They have to be allocated at the top kernel level and then passed to the calc function,
    // since a kernel should not call another kernel that uses local memory (the behavior is undefined -
    // it seems to work on NVIDIA, not on AMD).
    local real4 y_bodies[GROUP_SIZE];      // bodies corresponding to the y-dimension of the block
    local real item_sum[GROUP_SIZE];       // buffer for storing the vertical sum for each work item
    local real y_factors[GROUP_SIZE];      // form factors for the y-dimension bodies

    do_calc_block(v_q, v_bodies, t_factors, n_factors, t_body_factors, /*param_b,*/ v_body_surface, water_weight, y_bodies, item_sum, y_factors, t_buf_blocks);
}

//
// the global NDRange MUST be (block count, q count)
//
kernel
void block_v_sum(
            const global real *restrict t_buf_blocks,               // aligned to GROUP_SIZE
            global real *restrict t_margin_Iq)                      // aligned to 4
{
    int idx = get_global_id(0);             // x block index
    int idy = get_global_id(1);             // q index
    int n_blocks = get_global_size(0);      // 

    int n_blocks_aln = aligned_num(n_blocks, 4);
    const global real* block_page = t_buf_blocks + idy * n_blocks_aln * n_blocks_aln;

    // unrolling the loop for better performance
    real sum = 0.0f;
    for (int y = 0; y < n_blocks; y++)
    {
        sum += block_page[y * n_blocks_aln + idx];
    }

    t_margin_Iq[idy * n_blocks_aln + idx] = sum;
}

//
// the global NDRange MUST be (q count)
//
kernel
void block_h_sum(
            const global real *restrict t_margin_Iq,               // aligned to GROUP_SIZE
            const int n_blocks,
            global real *restrict out_v_Iq)
{
    int idx = get_global_id(0);
    //int n_q = get_global_size(0);

    int n_blocks_aln = aligned_num(n_blocks, 4);

    real Iq = 0.0f;
    for (int i = 0; i < n_blocks; i++)
    {
        Iq += t_margin_Iq[idx * n_blocks_aln + i];
    }
    out_v_Iq[idx] = Iq;
}

//
// The global NDRange MUST be (aligned body count, block count (=aligned body count/GROUP_SIZE), q count)
// The local NDRange MUST be (GROUP_SIZE, 1, 1)
//
kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void recalc_block(const global real *restrict v_q,
            const global real4 *restrict v_bodies,          // bodies vector        - aligned to GROUP_SIZE
            const global real *restrict t_factors,           // packed table of form factors (n_q x n_factors) - needed only for the water form factors
            const unsigned int n_factors,           // number of factors for each q in the t_factors table
            const global real *restrict t_body_factors,     // (factor, q) table    - aligned to GROUP_SIZE
            //!const real param_b, - edited: switched to water FF calc on the host
            //const global real* v_water_factors,     // ++(q) vector of water form factors
            const global real *restrict v_body_surface,     // ++(body index) vector of surface area factor for each body - aligned to GROUP_SIZE
            const real water_weight,                // weight factor for the water contribution
            int upd_start_aln, int upd_end_aln,
            global real *restrict t_buf_blocks)             // internal buffer for storing the block sums
                                                    //                      - aligned to 4x4
{
    int idx = get_global_id(0);             // body index
    int idy = get_global_id(1);             // block index (= v body index / GROUP_SIZE)

    // no need to recalculate blocks made of bodies from outside the update region
    int block_clean = (idx < upd_start_aln || idx >= upd_end_aln) && (idy < upd_start_aln / GROUP_SIZE || idy >= upd_end_aln / GROUP_SIZE);
    if (block_clean)
        return;
    // now we know that this block needs recalculation

    // Local memory caches for workgroup use.
    // They have to be allocated at the top kernel level and then passed to the calc function,
    // since a kernel should not call another kernel that uses local memory (the behavior is undefined -
    // it seems to work on NVIDIA, not on AMD).
    local real4 y_bodies[GROUP_SIZE];      // bodies corresponding to the y-dimension of the block
    local real item_sum[GROUP_SIZE];       // buffer for storing the vertical sum for each work item
    local real y_factors[GROUP_SIZE];      // form factors for the y-dimension bodies

    do_calc_block(v_q, v_bodies, t_factors, n_factors, t_body_factors, /*param_b,*/ v_body_surface, water_weight, y_bodies, item_sum, y_factors, t_buf_blocks);
}

kernel __attribute__((reqd_work_group_size(GROUP_SIZE, 1, 1)))
void recalc_block_with_pivots(const global real *restrict v_q,
            const global real4 *restrict v_bodies,          // bodies vector        - aligned to GROUP_SIZE
            const global real *restrict t_factors,           // packed table of form factors (n_q x n_factors) - needed only for the water form factors
            const unsigned int n_factors,           // number of factors for each q in the t_factors table
            const global real *restrict t_body_factors,     // (factor, q) table    - aligned to GROUP_SIZE
            //!const real param_b, - edited: switched to water FF calc on the host
            //const global real* v_water_factors,     // ++(q) vector of water form factors
            const global real *restrict v_body_surface,     // ++(body index) vector of surface area factor for each body - aligned to GROUP_SIZE
            const real water_weight,                // weight factor for the water contribution
            int upd_start_aln, int upd_end_aln,
            global real *restrict t_buf_blocks)             // internal buffer for storing the block sums
                                                    //                      - aligned to 4x4
{
    int idx = get_global_id(0);             // body index
    int idy = get_global_id(1);             // block index (= v body index / GROUP_SIZE)

    // no need to recalculate blocks made of bodies from outside the update region
    int block_clean = ((idx < upd_start_aln) && (idy < (upd_start_aln / GROUP_SIZE))) || ((idx >= upd_end_aln) && (idy >= (upd_end_aln / GROUP_SIZE)));
    //int block_clean = (idx < upd_start_aln || idx >= upd_end_aln) && (idy < upd_start_aln / GROUP_SIZE || idy >= upd_end_aln / GROUP_SIZE);
    if (block_clean)
        return;
    // now we know that this block needs recalculation

    // Local memory caches for workgroup use.
    // They have to be allocated at the top kernel level and then passed to the calc function,
    // since a kernel should not call another kernel that uses local memory (the behavior is undefined -
    // it seems to work on NVIDIA, not on AMD).
    local real4 y_bodies[GROUP_SIZE];      // bodies corresponding to the y-dimension of the block
    local real item_sum[GROUP_SIZE];       // buffer for storing the vertical sum for each work item
    local real y_factors[GROUP_SIZE];      // form factors for the y-dimension bodies

    do_calc_block(v_q, v_bodies, t_factors, n_factors, t_body_factors, /*param_b,*/ v_body_surface, water_weight, y_bodies, item_sum, y_factors, t_buf_blocks);
}

);
