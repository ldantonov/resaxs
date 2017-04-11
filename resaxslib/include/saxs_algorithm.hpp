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

#ifndef RE_SAXS_ALGORITHM_HPP
#define RE_SAXS_ALGORITHM_HPP

#include <cmath>

#include "re_algorithm.hpp"
#include "re_cl_algorithm_base.hpp"

namespace resaxs {
namespace algorithm {

    class ocl_exec_params : public params_base
    {
    public:
        std::vector<dev_id> devices_;
        unsigned int workgroup_size_ = 64;

        virtual void set_devices(const std::vector<dev_id> &devices)
        {
            devices_ = devices;
            set_dirty();
        }

        void set_workgroup_size(unsigned int size)
        {
            if (workgroup_size_ != size)
            {
                workgroup_size_ = size;
                set_dirty();
            }
        }
    };

    template <typename FLT_T>
    class implicit_water_params : public params_base
    {
    protected:
        std::vector<FLT_T> rsasa_;
    public:
        FLT_T water_weight_ = 0;
        bool no_rsasa_recalc_ = false;

        void set_water_weight(FLT_T weight)
        {
            if (water_weight_ != weight)
            {
                water_weight_ = weight;
                set_dirty();
            }
        }

        std::vector<FLT_T> & access_rsasa()
        {
            return rsasa_;
        }
    };

    template <typename FLT_T>
    class form_factor_params : public params_base
    {
    public:
        std::vector<FLT_T> factors_;
        int water_ff_index_ = -1;
        FLT_T expansion_factor_ = 1;

        void set_water_ff_index(int index)
        {
            if (water_ff_index_ != index)
            {
                water_ff_index_ = index;
                set_dirty();
            }
        }

        std::vector<FLT_T> & access_factors()
        {
            return factors_;
        }

        virtual void set_expansion_factor(FLT_T c)
        {
            if (expansion_factor_ != c)
            {
                expansion_factor_ = c;
                set_dirty();
            }
        }
    };
    
    class pivot_params : public params_base
    {
    public:
        int pivot_start_ = -1;
        int pivot_end_ = -1;
    };

    template <typename FLT_T>
    struct saxs_params : public params_base
    {
        std::vector<FLT_T> qq_;
        unsigned int n_bodies_ = 0;

        virtual void initialize(const std::vector<FLT_T> &qq, const std::vector<real4<FLT_T>> &bodies,
            const std::vector<dev_id> &devices, unsigned int workgroup_size = SAXS_NV_FERMI_WF_SIZE)
        {
            qq_ = qq;
            n_bodies_ = (unsigned int)bodies.size();
            get_ocl_exec_params().set_devices(devices);
            get_ocl_exec_params().set_workgroup_size(workgroup_size);
            set_dirty();
        }

        virtual form_factor_params<FLT_T> & get_ff_params() = 0;
        virtual implicit_water_params<FLT_T> & get_implicit_water_params() = 0;
        virtual ocl_exec_params & get_ocl_exec_params() = 0;
        virtual pivot_params & get_pivot_params() = 0;
    };

    //////////////////////////////////////////////////////////////////////////
    /// Interface to all SAXS algorithm classes
    ///
    ///     It is not ref counted, so it needs to be explicitly managed with delete.
    ///     The template parameters specify the underlying floating point type to use
    ///         (and its associated 4-value vector - cl_float4 or cl_double4).
    ///     It is specialized on float and double, could support others (long double?).
    template <typename FLT_T, typename BASE = algorithm>
    struct i_saxs : public BASE
    {
        typedef typename real_type<FLT_T>::real4 real4;

    public:
        virtual saxs_params<FLT_T> & access_params() = 0;

        virtual void initialize() = 0;
        //
        //  Performs the initial calculation of the SAXS curve.
        //  The algorithm arguments should have been set before calling this.
        //      bodies          - vector of atomic bodies, describing the protein
        //      out_Iqq         - buffer to receive the result - I(q) - for each value of q
        //
        virtual void calc_curve(const std::vector<real4> &bodies, std::vector<FLT_T> &out_Iqq) = 0;

        //
        //  Recalculates the SAXS curve that was previously calculated.
        //  Initial calculation should have been performed before calling this.
        //      bodies          - vector of atomic bodies, describing the protein. it will contains both moved and retained bodies.
        //      upd_start       - index of the first moved body
        //      upd_length      - number of moved bodies
        //      out_Iqq         - buffer to receive the result - I(q) - for each value of q
        //
        virtual void recalc_curve(const std::vector<real4> &bodies, unsigned int upd_start, unsigned int upd_length, std::vector<FLT_T> &out_Iqq) = 0;
        
        //
        // Commits the last calculation, i.e. the state is checkpointed and can be restored by a revert operation.
        //
        virtual void commit() {};
        
        //
        // Reverts the last calculation, i.e. restores the state to the last committed state.
        //
        virtual void revert() {};
    };
    
    //////////////////////////////////////////////////////////////////////////
    /// SAXS algorithms provided in the library.
    ///
    enum class saxs_enum
    {
        saxs_gpu_pt_wf,       // decomposition by 2D square blocks, using local memory; implicit hydration layer through modified form factors.
        saxs_gpu_v2,
        saxs_hist_cl_gpu,
    };

    //////////////////////////////////////////////////////////////////////////
    /// Algorithm creator class that is specialized on the different algorithms.
    ///     We need this in addition to saxs::create(), since member functions can't be partially specialized
    ///     and we want to specialize on the algorithm and base only, while leaving the REAL types as template args.
    template <typename FLT_T, typename BASE, saxs_enum alg>
    struct saxs_creator
    {
        static i_saxs<FLT_T, BASE>* create();
    };

    ///////////////////////////////////////////////////////////////////////////////
    /// Factory for creating the predefined algorithms.
    ///
    template <typename FLT_T>
    class saxs
    {
        typedef typename real_type<FLT_T>::real4 real4;

    public:
        /// Creates a built-in algorithm instance.
        /// Delegates to the creator classes.
        template <typename BASE, saxs_enum alg>
        static i_saxs<FLT_T, BASE>* create()
        {
            return saxs_creator<FLT_T, BASE, alg>::create();
        }

        /// Convenience function - allows selecting an algorithm at run-time through a single interface
        template <typename BASE>
        static i_saxs<FLT_T, BASE>* create(saxs_enum alg)
        {
            switch (alg)
            {
            case saxs_enum::saxs_gpu_pt_wf:
                return create<BASE, saxs_enum::saxs_gpu_pt_wf>();
            case saxs_enum::saxs_gpu_v2:
                return create<BASE, saxs_enum::saxs_gpu_v2>();
            case saxs_enum::saxs_hist_cl_gpu:
                return create<BASE, saxs_enum::saxs_hist_cl_gpu>();
            default:
                return NULL;
        }
    }
    };

    template <typename FLT_T = float>
    i_saxs<FLT_T, cl_base<FLT_T> >* create_saxs_ocl()
    {
        return saxs<FLT_T>::template create<cl_base<FLT_T>, saxs_enum::saxs_gpu_pt_wf>();
    }
}   // namespace algorithm
}   // namespace

#endif // RE_SAXS_ALGORITHM_HPP
