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

#ifndef RE_RSASA_ALGORITHM_HPP
#define RE_RSASA_ALGORITHM_HPP

#include <cmath>

#include "re_algorithm.hpp"

namespace resaxs {
namespace algorithm {

    //////////////////////////////////////////////////////////////////////////
    /// Base class for RSASA algorithm parameter structures.
    ///
    struct rsasa_params
    {
    };

    //////////////////////////////////////////////////////////////////////////
    /// Base interface to RSASA algorithms.
    ///
    template <typename FLT_T, typename OUT_FLT_T = FLT_T, typename BASE = algorithm>
    class i_rsasa : public BASE
    {
        typedef typename real_type<FLT_T>::real4 real4;

    public:
        /// Algorithm initialization.
        /// \param[in]  A structure of algorithm parameters.
        virtual void initialize(const rsasa_params *) = 0;

        /// SASA
        /// \param[in] atoms Vector of touples of 4 floating-point numbers; the first 3 are the coordinates of an atom, the 4th is an atom-type index.
        /// \param[out] rsasa RSASA for each atom.
        virtual void calc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa) = 0;

        /// SASA
        /// \param[in] atoms Vector of touples of 4 floating-point numbers; the first 3 are the coordinates of an atom, the 4th is an atom-type index.
        /// \param[out] rsasa RSASA for each atom.
        virtual void recalc_rsasa(const std::vector<real4> &atoms, std::vector<OUT_FLT_T> &rsasa) = 0;
    };

    //////////////////////////////////////////////////////////////////////////
    /// RSASA algorithms provided in the library.
    ///
    enum rsasa_enum
    {
        rsasa_dot_spheres,       // RSASA by using spheres of dots around atoms.
    };

    //////////////////////////////////////////////////////////////////////////
    /// Algorithm creator class that is specialized on the different algorithms.
    ///     We need this in addition to rsasa_algorithms::create(), since member functions can't be partially specialized
    ///     and we want to specialize on the algorithm and base only, while leaving the REAL types as template args.
    template <typename FLT_T, typename OUT_FLT_T, typename BASE, unsigned int alg>
    struct rsasa_creator
    {
        static i_rsasa<FLT_T, OUT_FLT_T, BASE>* create();
    };

    ///////////////////////////////////////////////////////////////////////////////
    /// Factory for creating the predefined algorithms.
    ///
    template <typename FLT_T, typename OUT_FLT_T = FLT_T>
    class rsasa
    {
        typedef typename real_type<FLT_T>::real4 real4;
    
    public:
        /// Parameters for the rsasa_dot_spheres algorithm.
        ///
        struct dot_spheres_params : public rsasa_params
        {
            FLT_T sqrt_dot_density_;
            FLT_T probe_radius_;

            dot_spheres_params() : sqrt_dot_density_(0), probe_radius_(0) {}
            dot_spheres_params(FLT_T dot_density, FLT_T probe_radius) : sqrt_dot_density_(sqrt(dot_density)), probe_radius_(probe_radius) {}

            bool operator==(const dot_spheres_params &other) const
            {
                return sqrt_dot_density_ == other.sqrt_dot_density_ && probe_radius_ == other.probe_radius_;
            }
            bool operator!=(const dot_spheres_params &other) const
            {
                return !operator==(other);
            }

            const std::vector<real4> create_dots(FLT_T radius) const;
            const std::vector<real4> create_32_dots(FLT_T radius) const;
            const std::vector<real4> create_fibonacci_dots(FLT_T radius) const;
            const std::vector<real4> create_dodecahedron_dots(FLT_T radius) const;
            const std::vector<real4> create_dodeca_icosa_dots(FLT_T radius) const;
            
            virtual ~dot_spheres_params() {}    // needed
        };

        struct dot_spheres_cl_params : public dot_spheres_params
        {
            std::vector<dev_id> devices_;
            unsigned int n_atoms_;

            dot_spheres_cl_params() : dot_spheres_params() {}
            dot_spheres_cl_params(FLT_T dot_density, FLT_T probe_radius, const std::vector<dev_id> &devices, unsigned int n_atoms) :
                dot_spheres_params(dot_density, probe_radius), devices_(devices), n_atoms_(n_atoms) {}
        };

        /// Creates a built-in algorithm instance.
        /// Delegates to the creator classes.
        template <typename BASE, unsigned int alg>
        static i_rsasa<FLT_T, OUT_FLT_T, BASE>* create()
        {
            return rsasa_creator<FLT_T, OUT_FLT_T, BASE, alg>::create();
        }

        template <typename BASE>
        static i_rsasa<FLT_T, OUT_FLT_T, BASE>* create(rsasa_enum alg)
        {
            switch (alg)
            {
            case rsasa_dot_spheres:
                return create<BASE, rsasa_dot_spheres>();
            default:
                return NULL;
            }
        }
    };

}   // namespace
}   // namespace

#endif // RE_RSASA_ALGORITHM_HPP

