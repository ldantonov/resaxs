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

#ifdef cl_amd_printf
#pragma OPENCL EXTENSION cl_amd_printf : enable
#endif

#define mad_int(x, y, z)    ((x) * (y) + (z))

kernel
void map_factors2(int n_q,
    global float4* v_bodies, int n_bodies,
    const global float* t_factors, int n_factors,
    global float* t_bodies_f,   // buffer to store a table of body factors for each q
    global float4* v_prev_bodies)   // the body positions for the previous step (it's simply initialized here)
{
    size_t idx = get_global_id(0);

    float4 body = v_bodies[idx];
    int factor_index = as_int(body.w);
    body.w = 0.0f;
    v_prev_bodies[idx] = body;  // save a copy in the vector of previous positions
    v_bodies[idx].w = 0.0f;    // zero the .w component for correct distance calculation

    for (int k = 0; k < n_q; k++)
    {
        // populate a table of form factors with axes (body index, q index)
        t_bodies_f[k * n_bodies + idx] = t_factors[k * n_factors + factor_index];
    }
}

kernel
void save_prev_bodies(const global float4* v_bodies, int upd_offset,
    global float4* v_prev_bodies)
{
    int idx = get_global_id(0) + upd_offset;
    v_prev_bodies[idx] = v_bodies[idx];
}

float sinc_qr(float q, float4 bodyx, float4 bodyy)
{
    float qr = q * distance(bodyx, bodyy);   // q * r
    float v1 = native_divide(native_sin(qr), qr);
    return isnan(v1) ? 1.0f : v1;   // for q = 0 or r = 0, sin(qr)/qr = 1
}

kernel
void recalc_margins_mf2(const global float* v_q, int n_q,
            const global float4* v_bodies, int n_bodies,
            const global float* t_bodies_f,
            int upd_start, int upd_end, const global float4* v_prev_bodies,
            global float* t_margin_Iq)
{
    int idx = get_global_id(0);

    float mult = idx < upd_start ? 2.0f : 1.0f;
    mult = idx >= upd_end ? 2.0f : mult;

    for (int k = 0; k < n_q; k++)
    {
        float q = v_q[k];
        //if (isnan(0 / q))     // improves running time on CPUs
        //    continue;

        float4 prev_bodyx = v_prev_bodies[idx];
        float4 bodyx = v_bodies[idx];

        float inner_delta = 0.0f;
        for (int y = upd_start; y < upd_end; y++)
        {
            local float4 prev_bodyy;
            local float4 bodyy;
            local float y_factor;
            if (get_local_id(0) == 0)
            {
                prev_bodyy = v_prev_bodies[y];
                bodyy = v_bodies[y];
                y_factor = t_bodies_f[k * n_bodies + y];
            }
            barrier(CLK_LOCAL_MEM_FENCE);
            float fac = y_factor;

            /*float qr = q * distance(prev_bodyx, prev_bodyy);   // q * r
            //float qr = q * distance(prev_bodyx, v_prev_bodies[y]);   // q * r
            float v1 = native_divide(native_sin(qr), qr);
            v1 = isnan(v1) ? 1 : v1;   // for q = 0 or r = 0, sin(qr)/qr = 1*/

            //inner_delta -= t_bodies_f[k * n_bodies + y] * v1;
            inner_delta -= fac * sinc_qr(q, prev_bodyx, prev_bodyy);//v1;

            /*qr = q * distance(bodyx, bodyy);   // q * r
            //qr = q * distance(bodyx, v_bodies[y]);   // q * r
            v1 = native_divide(native_sin(qr), qr);
            v1 = isnan(v1) ? 1 : v1;   // for q = 0 or r = 0, sin(qr)/qr = 1*/

            //inner_delta += t_bodies_f[k * n_bodies + y] * v1;
            inner_delta += fac * sinc_qr(q, bodyx, bodyy);//v1;
        }

        // save the delta for this column of the matrix
        //t_margin_Iq[k * n_bodies + idx] = inner_delta * t_bodies_f[k * n_bodies + idx] * mult;
        t_margin_Iq[mad_int(k, n_bodies, idx)] = inner_delta * t_bodies_f[mad_int(k, n_bodies, idx)] * mult;
    }
}

//
// Adjusts the computed I(q) by the marginal deltas for the bodies for a give q.
// Each work item is assigned to a index of q.
//
kernel
void adjust_Iq(const global float* t_margin_Iq, int n_bodies,
            global float* out_v_Iq)
{
    int idx = get_global_id(0); // global work item id is q index

    float adj = 0.0f;
    for (int i = 0; i < n_bodies; i++)
    {
        adj += t_margin_Iq[mad_int(idx, n_bodies, i)];
    }
    out_v_Iq[idx] += adj; // save the adjusted I(q)
}

///------------------

kernel
void map_factors1(int n_q,
    global float4* v_bodies, int n_bodies,
    const global float* t_factors, int n_factors,
    global float* t_bodies_f)   // buffer to store a table of body factors for each q
{
    size_t idx = get_global_id(0);

    float4 body = v_bodies[idx];
    int factor_index = as_int(body.w);
    v_bodies[idx].w = 0.0f;    // zero the .w component for correct distance calculation

    for (int k = 0; k < n_q; k++)
    {
        t_bodies_f[k * n_bodies + idx] = t_factors[k * n_factors + factor_index];
    }
}

kernel
void margins_mapped_factors1(const global float* v_q, int n_q,
            const global float4* v_bodies, int n_bodies,
            const global float* t_bodies_f,
            global float* t_margin_Iq)
{
    int idx = get_global_id(0);

    for (int k = 0; k < n_q; k++)
    {
        float q = v_q[k];
        float4 bodyx = v_bodies[idx];

        float inner_sum = 0.0f;
        for (int y = 0; y < n_bodies; y++)
        {
            /*float qr = q * distance(bodyx, v_bodies[y]);   // q * r
            float v1 = native_divide(native_sin(qr), qr);
            v1 = isnan(v1) ? 1 : v1;   // for q = 0 or r = 0, sin(qr)/qr = 1*/
            float v1 = sinc_qr(q, bodyx, v_bodies[y]);

            // form factor for body y
            inner_sum = fma(t_bodies_f[k * n_bodies + y], v1, inner_sum);
        }
        t_margin_Iq[mad24(k, n_bodies, idx)] = inner_sum * t_bodies_f[mad24(k, n_bodies, idx)];
    }
}



kernel
void map_factors(int n_q,
    const global float4* v_bodies, int n_bodies,
    const global float* t_factors, int n_factors,
    global float4* v_bodies_f)
{
    size_t idx = get_global_id(0);

    float4 body = v_bodies[idx];
    int factor_index = as_int(body.w);

    for (int k = 0; k < n_q; k++)
    {
        body.w = t_factors[factor_index + k * n_factors];
        v_bodies_f[k * n_bodies + idx] = body;
    }
}

kernel
void margins_mapped_factors(const global float* v_q, int n_q,
            global float4* v_bodies_f, int n_bodies,
            global float* t_margin_Iq)
{
    int idx = get_global_id(0);

    for (int k = 0; k < n_q; k++)
    {
        float q = v_q[k];
        float4 bodyx = v_bodies_f[k * n_bodies + idx];
        float Fqx = bodyx.w;
        bodyx.w = 0;
        //const global float* Fq = t_factors + mul24(k, n_factors);  // form factors for this q

        float inner_sum = 0;
        for (int y = 0; y < n_bodies; y++)
        {
            float4 bodyy = v_bodies_f[k * n_bodies + y];
            float Fqy = bodyy.w;
            bodyy.w = 0;
            float qr = q * distance(bodyx, bodyy);   // q * r
            float v1 = native_divide(native_sin(qr), qr);
            v1 = isnan(v1) ? 1 : v1;   // for q = 0 or r = 0, sin(qr)/qr = 1

            // form factor for body y
            inner_sum += Fqy * v1;
            //inner_sum = fma(Fqy, v1, inner_sum);
        }
        t_margin_Iq[k * n_bodies + idx] = inner_sum * Fqx;
    }
}

kernel
void margins(const global float* v_q, int n_q,
            const global float4* v_bodies, int n_bodies,
            const global float* t_factors, int n_factors,
//            global float* out_v_Iq,
            global float* t_margin_Iq)
{
    //local float fac0[24];
    //event_t ev = async_work_group_copy(fac0, t_factors, 21, 0);
    //wait_group_events(1, &ev);

    int idx = get_global_id(0);
#ifdef cl_amd_printf
    //printf("%d\n", idx);
#endif

    // form factor for body x
    int fx_index = as_int(v_bodies[idx].w);

    for (int k = 0; k < n_q; k++)
    {
        float q = v_q[k];
        const global float* Fq = t_factors + mul24(k, n_factors);  // form factors for this q
        float Fqx = Fq[fx_index];

        float inner_sum = 0;
        for (int y = 0; y < n_bodies; y++)
        {
            float qr = q * distance(v_bodies[idx], v_bodies[y]);   // q * r
            float v1 = native_divide(native_sin(qr), qr);
            v1 = isnan(v1) ? 1 : v1;   // for q = 0 or r = 0, sin(qr)/qr = 1

            // form factor for body y
            int fy_index = as_int(v_bodies[y].w);
            //inner_sum += Fq[fy_index] * v1;
            inner_sum = fma(Fq[fy_index], v1, inner_sum);
        }
        t_margin_Iq[idx + k * n_bodies] = inner_sum * Fqx;
    }
}

kernel
void sum(global float* t_margin_Iq, int n_bodies,
            global float* out_v_Iq)
{
    int idx = get_global_id(0);

    float Iq = 0;
    for (int i = 0; i < n_bodies; i++)
    {
        Iq += t_margin_Iq[mad_int(idx, n_bodies, i)];
    }
    out_v_Iq[idx] = Iq;
}

kernel
void sum(global float* t_margin_Iq, int n_bodies, int n_q,
            global float* out_v_Iq,
            local float* margin_buf)
{
    int idx = get_global_id(0);
    int tid = get_local_id(0);
    int localSize = get_local_size(0);

    // Number of tiles we need to iterate
    unsigned int n_tiles = n_bodies / localSize;

    for(int i = 0; i < n_tiles; ++i)
    {
        // load one tile into local memory

        int idx = i * localSize + tid;

        localPos[tid] = pos[idx];



        // Synchronize to make sure data is available for processing

        barrier(CLK_LOCAL_MEM_FENCE);

    for (int i = 0; i < n_q; i++)
    {
    }

    float Iq = 0;
    for (int i = 0; i < n_bodies; i++)
    {
        Iq += t_margin_Iq[mad_int(idx, n_bodies, i)];
    }
    out_v_Iq[idx] = Iq;
}