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

#include <fstream>
#include <iostream>
#include <map>

#include "parse_pdb.hpp"
#include "utils.hpp"

#include "ff_coef.hpp"

using namespace std;

namespace resaxs
{
    template <typename FLT_T>
    auto parse_pdb<FLT_T>::first_model_as_complex_atoms() -> vector<real4>
    {
        ifstream file(pdb_filename_);
        verify(file.is_open(), SAXS_PDB_FILE_NOT_FOUND, pdb_filename_ + " file not found");

        vector<real4> bodies;
        do
        {
            char line[100];
            file.getline(line, sizeof(line));
            if (!file)
                break;

            string rec_type(trim(string(line, 6)));
            if (rec_type == "END")
                break;
            else if (rec_type == "ENDMDL")
                break;
            else if (rec_type == "MODEL")
                start_model(line);
            else if (rec_type == "ATOM" || rec_type == "HETATM")
                parse_atom(line, bodies);
        } while (true);

        return bodies;
    }

    template <typename FLT_T>
    auto parse_pdb<FLT_T>::all_models_as_complex_atoms() -> vector<vector<real4>>
    {
        ifstream file(pdb_filename_);
        verify(file.is_open(), SAXS_PDB_FILE_NOT_FOUND, pdb_filename_ + " file not found");

        vector<vector<real4>> models;
        bool in_model = false;
        do
        {
            char line[100];
            file.getline(line, sizeof(line));
            if (file.bad() || file.eof())
                break;

            // the fail flag is set if the whole buffer was filled, i.e. the line is longer than 99 chars
            if (file.fail() && file.gcount() > 0)
            {
                file.clear();
                // skip the rest of the line
                file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
            }

            string rec_type(trim(string(line, 6)));
            if (rec_type == "END")
                break;  // the end - exit
            else if (rec_type == "ENDMDL")
                in_model = false;   // just mark that the current model is finished
            else if (rec_type == "MODEL")
            {
                models.resize(models.size() + 1);
                in_model = true;    // now we have an active model
            }
            else if (rec_type == "ATOM" || rec_type == "HETATM")
            {
                if (!in_model)
                {
                    models.resize(models.size() + 1);
                    in_model = true;    // now we have an active model
                }
                parse_atom(line, models.back());
            }
        } while (true);

        return models;
    }

    // Map converts PDB reader simple atom types to form factor atom types
    //
/*
    const map<string, atoms::atom_type> pdb_to_ff_simple_atom_map = {
            { "H", atoms::H },
            { "C", atoms::C },
            { "N", atoms::N },
            { "O", atoms::O },
            { "P", atoms::P },
            { "S", atoms::S },
            { "Zn", atoms::Zn },
    };

    // Converts PDB reader atom types (within residues) to form factor types
    //
    const map<string, atoms::atom_type> pdb_to_ff_complex_atom_map = {
            { "CH2", atoms::CH2 },
            { "CH3", atoms::CH3 },
            { "CA", atoms::CH },
            { "CB", atoms::CH2 },
            { "CG", atoms::CH2 },
            { "CG2", atoms::CH3 },
            { "CD", atoms::CH2 },
            { "N", atoms::NH },
    };

    typedef pair<string, string> pdb_atom_type;

    // Converts PDB reader atom types for specific residues to form factor types
    //
    const map<pdb_atom_type, atoms::atom_type> pdb_to_ff_complex_atom_special_map = {
            { { "CA", "GLY" }, atoms::CH2 },
            { { "CB", "ALA" }, atoms::CH3 },
            { { "CB", "ILE" }, atoms::CH },
            { { "CB", "THR" }, atoms::CH },
            { { "CB", "VAL" }, atoms::CH },
            { { "CG", "ASN" }, atoms::C },
            { { "CG", "ASP" }, atoms::C },
            { { "CG", "HIS" }, atoms::C },
            { { "CG", "LEU" }, atoms::CH },
            { { "CG", "PHE" }, atoms::C },
            { { "CG", "TRP" }, atoms::C },
            { { "CG", "TYR" }, atoms::C },
            { { "CG1", "ILE" }, atoms::CH2 },
            { { "CG1", "VAL" }, atoms::CH3 },
            { { "CD", "GLN" }, atoms::C },
            { { "CD", "GLU" }, atoms::C },
            { { "CD1", "ILE" }, atoms::CH3 },
            { { "CD1", "LEU" }, atoms::CH3 },
            { { "CD1", "PHE" }, atoms::CH },
            { { "CD1", "TRP" }, atoms::CH },
            { { "CD1", "TYR" }, atoms::CH },
            { { "CD2", "HIS" }, atoms::CH },
            { { "CD2", "LEU" }, atoms::CH3 },
            { { "CD2", "PHE" }, atoms::CH },
            { { "CD2", "TYR" }, atoms::CH },
            { { "CE", "LYS" }, atoms::CH2 },
            { { "CE", "MET" }, atoms::CH3 },
            { { "CE1", "HIS" }, atoms::CH },
            { { "CE1", "PHE" }, atoms::CH },
            { { "CE1", "TYR" }, atoms::CH },
            { { "CE2", "PHE" }, atoms::CH },
            { { "CE2", "TYR" }, atoms::CH },
            { { "CE3", "TRP" }, atoms::CH },
            { { "CZ", "PHE" }, atoms::CH },
            { { "CZ2", "TRP" }, atoms::CH },
            { { "CZ3", "TRP" }, atoms::CH },
            { { "N", "PRO" }, atoms::N },
            { { "ND1", "HIS" }, atoms::NH },    // TODO: just N? check foxs
            { { "ND2", "ASN" }, atoms::NH2 },
            { { "NH1", "ARG" }, atoms::NH2 },
            { { "NH2", "ARG" }, atoms::NH2 },
            { { "NE", "ARG" }, atoms::NH },
            { { "NE1", "TRP" }, atoms::NH },
            { { "NE2", "GLN" }, atoms::NH2 },
            { { "NZ", "LYS" }, atoms::NH3 },    // TODO: why not NH2?
            { { "OG", "SER" }, atoms::OH },
            { { "OG1", "THR" }, atoms::OH },
            { { "OH", "TYR" }, atoms::OH },
            { { "SG", "CYS" }, atoms::SH },
            // TODO: OD1, OD2 on ASP - should there be an H on one of them? half on each? or nothing?
    };

    atoms::atom_type map_pdb_atom_to_ff_type(atoms::atom_type element, const string& atom_label, const string& res_type)
    {
        // first map all atoms to their simple types (elements)
        atoms::atom_type complex_atom_type = element;

        // handle any complex atoms that have a default type other than their element
        auto complex_atom = pdb_to_ff_complex_atom_map.find(atom_label);
        if (complex_atom != pdb_to_ff_complex_atom_map.end())
            complex_atom_type = (*complex_atom).second;

        // handle any exceptions to the default types
        auto complex_atom_special = pdb_to_ff_complex_atom_special_map.find(pdb_atom_type(atom_label, res_type));
        if (complex_atom_special != pdb_to_ff_complex_atom_special_map.end())
            complex_atom_type = (*complex_atom_special).second;

        return complex_atom_type;
    }
*/

    template <typename FLT_T>
    struct pdb_line_format;
    
    template <>
    struct pdb_line_format<float>
    {
        static const char * str()
        {
            return "%6c%5d%*1c%4c%1c%3c%*1c%1c%4d%1c%*3c%8f%8f%8f%6f%6f%*10c%2c%2c";
        }
    };

    template <>
    struct pdb_line_format<double>
    {
        static const char * str()
        {
            return "%6c%5d%*1c%4c%1c%3c%*1c%1c%4d%1c%*3c%8lf%8lf%8lf%6lf%6lf%*10c%2c%2c";
        }
    };

    /// Represents the information in an ATOM or HETATM line of a PDB file.
    /// See http://www.wwpdb.org/documentation/format33/sect9.html#ATOM
    ///
    template <typename FLT_T, size_t size>
    struct pdb_atom_line
    {
        FLT_T x;
        FLT_T y;
        FLT_T z;
        FLT_T occupancy;
        FLT_T b_factor;
        int atom_n;
        int res_n;
        char alt_loc;
        char chain_id;
        char ins_code;

        string atom_type_;
        string res_type_;
        string element_;

        pdb_atom_line(char (&line)[size])
        {
            fill0(r_rec_name);
            fill0(r_atom_type);
            fill0(r_res_type);
            fill0(r_element);
            fill0(r_charge);

#ifdef _MSC_VER
#pragma warning( push )
#pragma warning(disable : 4996)    // suppress warning about sscanf being insecure in VS
#endif
            // initialize the fields by parsing the atom line
            sscanf(line, pdb_line_format<FLT_T>::str(),
                r_rec_name, &atom_n, r_atom_type, &alt_loc, r_res_type, &chain_id, &res_n,
                &ins_code, &x, &y, &z, &occupancy, &b_factor, r_element, r_charge);
#ifdef _MSC_VER
#pragma warning( pop )
#endif

            atom_type_ = trim(r_atom_type);
            res_type_ = trim(r_res_type);
            element_ = trim(r_element);
            if (element_.empty())
            {
                // element entry is missing - determine from atom type
                element_ = atom_type_.substr(0, 1);
            }
        }

        /// The easiest way to have a templatized friend is to have it inline
        friend ostream& operator<< (ostream& os, const pdb_atom_line<FLT_T, size>& data)
        {
            os << data.r_rec_name << ' ' << data.atom_n << ' ' << data.r_atom_type << ' ' <<
                data.alt_loc << data.r_res_type << ' ' << data.chain_id << ' ' << data.res_n << ' ' << data.ins_code << ' ' <<
                data.x << ' ' << data.y << ' ' << data.z << ' ' << data.occupancy << ' ' << ' ' << data.b_factor << ' ' << data.r_element << data.r_charge;
            return os;
        }

    private:
        /// Fills the fixed-size buffer with \0.
        ///
        template <size_t buf_size>
        static void fill0(char (&buf)[buf_size])
        {
            fill(buf, buf + buf_size, '\0');
        }

        char r_rec_name[7];
        char r_atom_type[5];
        char r_res_type[4];
        char r_element[3];
        char r_charge[3];
    };

    template <typename FLT_T> template <size_t size>
    void parse_pdb<FLT_T>::parse_atom(char(&line)[size], std::vector<real4> & out_bodies)
    {
        pdb_atom_line<FLT_T, size> data(line);

        if (data.res_type_ == "HOH")    // skip waters
        {
            if (verbose_lvl_ >= DEBUG)
                cout << "Skipped HOH molecule " << data.atom_n << endl;
            return;
        }

        if (data.alt_loc != ' ' && data.alt_loc != 'A')   // ignore extra alt locations
        {
            if (verbose_lvl_ >= DETAILS)
                cout << "Skipped alt loc " << data << endl;
            return;
        }

        // First map all atoms to their simple types (elements)
        atoms::atom_type atom_type;
        auto element_entry = pdb_to_ff_simple_atom_map.find(data.element_);
        if (element_entry != pdb_to_ff_simple_atom_map.end())
            atom_type = element_entry->second;
        else
        {
            if (verbose_lvl_ >= NORMAL)
                cerr << "Missing form factor for element " << data.element_ << " in residue " << data.res_type_ << "(" << data.res_n << ") - using N instead" << endl;
            atom_type = atoms::N;
        }

        if (atom_type == atoms::H)  // skip hydrogens - only complex atoms are used
        {
            if (verbose_lvl_ >= DETAILS)
                cout << "Skipped H atom " << data.atom_n << endl;
            return;
        }

        // map the atom type within the residue to a form factor atom type
        atom_type = map_pdb_atom_to_ff_type(atom_type, data.atom_type_, data.res_type_);

        // set the position of the complex atom and its form factor index
        real4 body;
        body.s[0] = data.x;
        body.s[1] = data.y;
        body.s[2] = data.z;
        set_factor_index(body, atom_type);

        // add to the list of bodies
        out_bodies.push_back(body);

        if (verbose_lvl_ >= DEBUG)
        {
            cout << "Added body " << out_bodies.size() - 1 << " of type " << atom_type << ' ' << data << endl;
        }
    }

    template class parse_pdb<float>;
    template class parse_pdb<double>;
} // namespace
