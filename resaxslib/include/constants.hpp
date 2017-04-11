#ifndef RESAXS_CONSTANTS_HPP
#define RESAXS_CONSTANTS_HPP

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


namespace resaxs
{
    const double pi = 3.14159265358979323846;

    /// Solvent density - default is HOH = 0.334 e/A^3
    const float default_solvent_rho = 0.334f;

    const float excl_vol_to_r3_coef = static_cast<float>(3.0 / (4.0 * pi * default_solvent_rho));

    namespace atoms
    {
        enum atom_type
        {
            H,
            C,
            N,
            O,
            P,
            S,
            Zn,
            SIMPLE_ATOM_TYPE_LAST = Zn,
            SIMPLE_ATOM_TYPE_COUNT,
            CH = SIMPLE_ATOM_TYPE_COUNT,
            CH2,
            CH3,
            NH,
            NH2,
            NH3,
            OH,
            OH2,
            SH,
            COMPLEX_ATOM_TYPE_LAST = SH,
            ATOM_TYPE_COUNT
        };
        
        const float excluded_volume[] = {
            5.15f,   // H
            16.44f,  // C
            2.49f,   // N
            9.13f,   // O
            5.73f,   // P
            19.86f,  // S
            9.85f,   // Zn
        };
        
        const unsigned int complex_atom_stucture[][2] = {
            { C, 1 },  // CH
            { C, 2 },  // CH2
            { C, 3 },  // CH3
            { N, 1 },  // NH
            { N, 2 },  // NH2
            { N, 3 },  // NH3
            { O, 1 },  // OH
            { O, 2 },  // OH2
            { S, 1 }   // SH
        };
    } // namespace atoms
    
} // namespace

#endif // RESAXS_CONSTANTS_HPP
