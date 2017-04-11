#ifndef PARSE_PDB_HPP
#define PARSE_PDB_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2014-2015 Lubo Antonov
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

#include <string>
#include <vector>

#include "resaxs.hpp"

#define SAXS_PDB_FILE_NOT_FOUND     -1100

namespace resaxs
{

    template <typename FLT_T>
    class parse_pdb
    {
        using real4 = typename real_type<FLT_T>::real4;

    public:
        enum verbose_levels
        {
            QUIET = 0,
            NORMAL,
            DETAILS,
            DEBUG
        };

        parse_pdb(const std::string & pdb_filename) : pdb_filename_(pdb_filename) {}
        void set_verbose_level(verbose_levels verbose_lvl = NORMAL) { verbose_lvl_ = verbose_lvl; }

        auto first_model_as_complex_atoms() -> std::vector<real4>;
        auto all_models_as_complex_atoms() -> std::vector<std::vector<real4>>;
    private:
        template <size_t size>
        void start_model(char (&line)[size]) {}
        template <size_t size>
        void parse_atom(char(&line)[size], std::vector<real4> & out_bodies);

        const std::string pdb_filename_;
        int cur_res_index = 0;
        verbose_levels verbose_lvl_ = NORMAL;
    };

} // namepsace

#endif // PARSE_PDB_HPP