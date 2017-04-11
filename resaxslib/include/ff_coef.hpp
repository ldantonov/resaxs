#ifndef FF_COEF_HPP
#define FF_COEF_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2014-2015 Lubo Antonov
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

#include <array>
#include <cmath>
#include <functional>

#include "resaxs.hpp"
#include "constants.hpp"

namespace resaxs
{

    namespace coefficients
    {
        // Cromer-Mann coefficients, from http://staff.chess.cornell.edu/~smilgies/x-rays/f0_CromerMann.txt
        // a1  a2  a3  a4  c  b1  b2  b3  b4
        const float cromer_mann_coefficients[][9] = {
                { 0.4930020f, 0.3229120f, 0.1401910f, 4.0810000E-02f, 3.0380001E-03f, 10.51090f, 26.12570f, 3.142360f, 57.79970f }, // H
                { 2.310000f, 1.020000f, 1.588600f, 0.8650000f, 0.2156000f, 20.84390f, 10.20750f, 0.5687000f, 51.65120f }, // C
                { 12.21260f, 3.132200f, 2.012500f, 1.166300f, -11.52900f, 5.7000001E-03f, 9.893300f, 28.99750f, 0.5826000f }, // N
                { 3.048500f, 2.286800f, 1.546300f, 0.8670000f, 0.2508000f, 13.27710f, 5.701100f, 0.3239000f, 32.90890f }, // O
                { 6.434500f, 4.179100f, 1.780000f, 1.490800f, 1.114900f, 1.906700f, 27.15700f, 0.5260000f, 68.16450f }, // P
                { 6.905300f, 5.203400f, 1.437900f, 1.586300f, 0.8669000f, 1.467900f, 22.21510f, 0.2536000f, 56.17200f }, // S
                { 14.07430f, 7.031800f, 5.165200f, 2.410000f, 1.304100f, 3.265500f, 0.2333000f, 10.31630f, 58.70970f }, // Zn
                { 36.02280f, 23.41280f, 14.94910f, 4.188000f, 13.39660f, 0.5293000f, 3.325300f, 16.09270f, 100.6130f }, // U 92
        };

        /// Efficiently manages access to one entry in the Cromer-Mann coefficient table.
        /// Equivalent to direct indexing into a local pointer to a sub-array.
        ///
        class cromer_mann
        {
        private:
            /// Offsets of each coefficient range into the table entry
            enum coef_type
            {
                CM_A = 0,
                CM_C = 4,
                CM_B = 5
            };

            /// 
            const float* coef;

            const float operator[] (unsigned int index) { return coef[index]; }

        public:
            cromer_mann(unsigned int index) : coef(cromer_mann_coefficients[index]) {}

            const float A(unsigned int index) { return (*this)[index + CM_A]; }
            const float B(unsigned int index) { return (*this)[index + CM_B]; }
            const float C() { return (*this)[CM_C]; }
        };
    }

    /// atomic ff(k) = c + SUM(a_i * exp(-b_i * k^2)) | i = [1, 4], k = sin(theta) / lambda = q / 4pi
    /// @param type type of atom
    /// @param k_sq k^2 for k = sin(theta) / lambda = q / 4pi
    template <typename FLT_T>
    inline FLT_T calc_vacuum_form_factor(atoms::atom_type type, FLT_T k_sq)
    {
        coefficients::cromer_mann coef(type);

        FLT_T result = coef.C();
        for (int i = 0; i < 4; i++)
        {
            result += coef.A(i) * exp(-coef.B(i) * k_sq);
        }

        return result;
    }

    /// excluded volume ff(k) = rho * V * exp(-pi * V^(2/3) * k^2), k = sin(theta) / lambda = q / 4pi
    /// [Fraser, MacRae and Suzuki (1978)]
    ///
    template <typename FLT_T>
    inline FLT_T calc_excl_vol_form_factor(atoms::atom_type type, FLT_T k_sq)
    {
        return default_solvent_rho * atoms::excluded_volume[type] * exp(-static_cast<FLT_T>(pi) * pow(atoms::excluded_volume[type], 2.0f / 3.0f) * k_sq);
    }

    template <typename FLT_T>
    class atomic_form_factors
    {
    public:
        /// Generates a table of atomic form factors for a range of scattering momenta.
        /// f(q) = f_vacuum(q) - f_excluded_volume(q)
        /// \param v_q A vector of scattering momenta
        /// \param t_factors Table to receive the generated form factors - row major, with q as the rows.    
        static void generate(const std::vector<FLT_T> &v_q, std::vector<FLT_T>& t_factors)
        {
            // generate with excluded volume expansion factor of 1
            generate(v_q, [](FLT_T k_sq){ return FLT_T(1); }, t_factors);
        }

        static void generate(const std::vector<FLT_T> &v_q, FLT_T avg_radius, FLT_T c, std::vector<FLT_T>& t_factors)
        {
            // calculate the excluded volume expansion factor from c and the average atom radius
            if (c != 1)
                generate(v_q, std::bind(&excl_vol_expansion_factor, c, std::placeholders::_1, avg_radius), t_factors);
            else
                generate(v_q, t_factors);
        }

        static void generate(const std::vector<FLT_T> &v_q, std::function<FLT_T(FLT_T k_sq)> excl_vol_exp_fact_func, std::vector<FLT_T>& t_factors)
        {
            t_factors.resize(v_q.size() * atoms::ATOM_TYPE_COUNT);
            for (size_t i = 0; i < v_q.size(); i++)
            {
                FLT_T* factor_page = &t_factors[i * atoms::ATOM_TYPE_COUNT];

                // Convert the scattering momentum q = 4pi*sin(theta) / lambda to k = sin(theta) / lambda,
                // as needed for computing atomic form factors
                const FLT_T k = v_q[i] / (4 * static_cast<FLT_T>(pi));
                const FLT_T k_sq = static_cast<float>(k * k);
                const FLT_T excl_vol_exp_fact = excl_vol_exp_fact_func(k_sq);

                // Calculate the form factors for all simple atom types at this q
                for (unsigned int j = 0; j < atoms::SIMPLE_ATOM_TYPE_COUNT; ++j)
                {
                    const FLT_T f_vac = calc_vacuum_form_factor(static_cast<atoms::atom_type>(j), k_sq);
                    const FLT_T f_excl = calc_excl_vol_form_factor(static_cast<atoms::atom_type>(j), k_sq) * excl_vol_exp_fact;
                    factor_page[j] = f_vac - f_excl;
                }

                // Calculate the form factors for all complex atom types at this q.
                // f(q) = f(simple_type,q) + number_or_Hydrogens * f(H,q)
                FLT_T H_factor = factor_page[atoms::H];
                for (unsigned int j = atoms::SIMPLE_ATOM_TYPE_COUNT; j < atoms::ATOM_TYPE_COUNT; ++j)
                {
                    const unsigned int* atom_structure = atoms::complex_atom_stucture[j - atoms::SIMPLE_ATOM_TYPE_COUNT];
                    factor_page[j] = factor_page[atom_structure[0]] + atom_structure[1] * H_factor;
                }
            }
//             ofstream log("ff.log");
//             //ostream & log = cerr;
//             log << "Form factors for " << v_q.size() << " q values and c=:\n\n";// << c << ":\n";
//             for_each(t_factors.begin(), t_factors.end(), [&log](FLT_T x){log << x << ' '; });
//             log << "\n\n";
        }

        static void calc_radii(std::vector<FLT_T> &radii)
        {
            std::array<FLT_T, atoms::ATOM_TYPE_COUNT> ff_excl;

            // Calculate the excluded volume form factors for all simple atom types at 0
            for (unsigned int i = 0; i < atoms::SIMPLE_ATOM_TYPE_COUNT; ++i)
            {
                ff_excl[i] = calc_excl_vol_form_factor(static_cast<atoms::atom_type>(i), FLT_T(0));
            }

            // Calculate the excluded volume form factors for all complex atom types at 0.
            // f(q) = f(simple_type,q) + number_or_Hydrogens * f(H,q)
            FLT_T H_factor = ff_excl[atoms::H];
            for (unsigned int i = atoms::SIMPLE_ATOM_TYPE_COUNT; i < atoms::ATOM_TYPE_COUNT; ++i)
            {
                const unsigned int* atom_structure = atoms::complex_atom_stucture[i - atoms::SIMPLE_ATOM_TYPE_COUNT];
                ff_excl[i] = ff_excl[atom_structure[0]] + atom_structure[1] * H_factor;
            }

            radii.resize(atoms::ATOM_TYPE_COUNT);
            for (unsigned int i = 0; i < atoms::ATOM_TYPE_COUNT; ++i)
            {
                radii[i] = excl_vol_ff0_to_radius(ff_excl[i]);
            }
        }

        static FLT_T calc_avg_radius(const std::vector<real4<FLT_T>> &bodies, const std::vector<FLT_T> &radii)
        {
            FLT_T result = 0;
            for (size_t i = 0; i < bodies.size(); ++i)
            {
                result += radii[get_factor_index(bodies[i])];
            }
            return result / bodies.size();
        }

        static FLT_T excl_vol_expansion_factor(FLT_T c, FLT_T k_sq, FLT_T rm)
        {
            const FLT_T c_sq = c * c;
            static const FLT_T _4div3pi = pow(4 * static_cast<FLT_T>(pi) / 3, 2.0f / 3.0f);
            return c_sq * c * exp(-static_cast<FLT_T>(pi) * _4div3pi * k_sq * rm * rm * (c_sq - 1));
        }

    private:
        static FLT_T excl_vol_ff0_to_radius(FLT_T ff)
        {
            return std::pow(excl_vol_to_r3_coef * ff, static_cast<FLT_T>(1.0 / 3));
        }
    };

} // namespace

#endif // FF_COEF_HPP
