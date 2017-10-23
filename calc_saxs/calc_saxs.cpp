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

#include <iostream>
#include <memory>
#include <string>

#include "calc_saxs.hpp"
#include "saxs_params_io.hpp"
#include <dsrpdb/Protein.h>
#include <dsrpdb/../../src/Residue_data.h>
#include "host_debye.hpp"

#include "ff_coef.hpp"
#include "parse_pdb.hpp"
#include "sasa_atomic.hpp"
#include "re_cl_algorithm_base.hpp"

using namespace std;

namespace resaxs
{

template <typename FLT_T>
calc_saxs<FLT_T>::calc_saxs(const vector<string> & bodies_filenames, const string & exe_base_path, bool atomic, FLT_T q_min, FLT_T q_max, unsigned int q_n,
    const profile_params<FLT_T> & params, verbose_levels verbose_lvl) :
    calc_saxs(bodies_filenames, exe_base_path, atomic, q_min, q_max, q_n, profile_params<FLT_T>(params), verbose_lvl)
{}

template <typename FLT_T>
calc_saxs<FLT_T>::calc_saxs(const vector<string> & bodies_filenames, const string & exe_base_path, bool atomic, FLT_T q_min, FLT_T q_max, unsigned int q_n,
    profile_params<FLT_T> && params, verbose_levels verbose_lvl) :
    params_(move(params)), verbose_lvl_(verbose_lvl)
{
    // get the q-values from the ref profile if possible
    if (params_.ref_profile_)
        v_q_ = params_.ref_profile_.v_q_;
    else
    {
        // generate the q-values
        v_q_.resize(q_n);
        FLT_T step = (q_n > 0) ? (q_max - q_min) / (q_n - 1) : 0;
        for (unsigned int i = 0; i < q_n; ++i)
            v_q_[i] = q_min + i * step;
    }
    v_Iq_.resize(v_q_.size());

    try
    {
        if (atomic)
            ;
        else
        {
            saxs_params_io::load_reals(exe_base_path + "form_factors-q_vector.dat", v_q_);
            saxs_params_io::load_reals(exe_base_path + "form_factors-two_body_model.dat", t_factors_);
        }
    }
    catch (const error & )
    {
        // saxs param files failed to load
        throw;
    }

    n_factors_ = (unsigned int)(t_factors_.size() / v_q_.size());

    if (atomic)
        load_pdb_atomic(bodies_filenames);
    else
    {
        load_pdb(bodies_filenames[0]);
        v_models_.push_back(v_bodies_);
    }
}

template <typename FLT_T>
void calc_saxs<FLT_T>::host_saxs()
{
    atomic_form_factors<FLT_T>::generate(v_q_, t_factors_);
    host_debye<double, FLT_T, real4>::calc_curve(v_q_, v_bodies_, t_factors_, v_Iq_);
}

template <typename FLT_T>
void calc_saxs<FLT_T>::verify_result()
{
    calc_saxs<FLT_T> host_calc(*this);

    // zero the result in the host calc
    host_calc.v_Iq_.clear();
    host_calc.v_Iq_.resize(v_q_.size());
    host_calc.host_saxs();

    double chisq = 0;
    for (auto i1 = v_Iq_.begin(), i2 = host_calc.v_Iq_.begin(); i1 != v_Iq_.end(), i2 != host_calc.v_Iq_.end(); ++i1, ++i2)
    {
        chisq += (*i1 - *i2) * (*i1 - *i2) / *i2;
    }

    cout << "Chi square: " << chisq << endl;
}

#if 0
// This code is for reading body information from a regular text file (not PDB). It is basically obsolete.
template <typename FLT_T, typename FLT4_T>
void calc_saxs<FLT_T, FLT4_T>::parse_bodies(const string & filename)
{
    ifstream file(filename);
    verify(file.is_open(), SAXS_TEST_FILE_NOT_FOUND, filename + " file not found");

    v_bodies_.clear();
    while (!file.eof())
    {
        FLT4_T body;
        int index;
        file >> body.s[0] >> body.s[1] >> body.s[2] >> index;
        if (file.good())
        {
            set_factor_index(body, index);
            v_bodies_.push_back(body);
        }
    }
}
#endif

    // Order of the residues in the Form Factor database
    enum TWO_BODIES_FF_NAMES
    {
        BACKBONE_NORM=0, // normal backbone *with* C_beta
        ALA_COMPLETE, ARG_SC, ASN_SC,       ASP_SC, CYS_SC,  // GLY and ALA don't have a side chain
        GLN_SC,       GLU_SC, GLY_COMPLETE, HIS_SC, ILE_SC,
        LEU_SC,       LYS_SC, MET_SC,       PHE_SC, PRO_SC,
        SER_SC,       THR_SC, TRP_SC,       TYR_SC, VAL_SC
    };

    // Order of the residues in the dsrpdb library
        /*enum Type { GLY=0, ALA, VAL, LEU, ILE,
        SER, THR, CYS, MET, PRO,
        ASP, ASN, GLU, GLN, LYS,
        ARG, HIS, PHE, TYR, TRP,
        ACE, NH2, INV };*/

    // Maps residues from dsrpdb enum to Form Factor database index
    const TWO_BODIES_FF_NAMES pdb_to_ff_res_map[20] = {GLY_COMPLETE, ALA_COMPLETE, VAL_SC, LEU_SC, ILE_SC,
        SER_SC, THR_SC, CYS_SC, MET_SC, PRO_SC,
        ASP_SC, ASN_SC, GLU_SC, GLN_SC, LYS_SC,
        ARG_SC, HIS_SC, PHE_SC, TYR_SC, TRP_SC};

double get_mass(dsrpdb::Atom::Type type)
{
    double mass = 0.;
    switch (type)
    {
    case dsrpdb::Atom::Type::C:
        mass = 12.0107;
        break;
    case dsrpdb::Atom::Type::N:
        mass = 14.0067;
        break;
    case dsrpdb::Atom::Type::O:
        mass = 15.9994;
        break;
    case dsrpdb::Atom::Type::S:
        mass = 32.0655;
        break;
    }

    return mass;
}

template <typename FLT_T>
void calc_saxs<FLT_T>::load_pdb(const string & filename)
{
    ifstream file(filename);
    verify(file.is_open(), SAXS_TEST_FILE_NOT_FOUND, filename + " file not found");

    dsrpdb::Protein prot(file);
    v_bodies_.clear();
    for (auto res_it = prot.residues_begin(); res_it != prot.residues_end(); ++res_it)
    {
        const dsrpdb::Residue::Type res_type = res_it->type();
        if (res_type >= sizeof(pdb_to_ff_res_map) / sizeof(TWO_BODIES_FF_NAMES))
            continue;

        real4 bbody; bbody.s[0] = bbody.s[1] = bbody.s[2] = 0.;
        real4 sbody; sbody.s[0] = sbody.s[1] = sbody.s[2] = 0.;
        double bx = 0., by = 0., bz = 0.;
        double wb = 0.;
        double sx = 0., sy = 0., sz = 0.;
        double ws = 0.;

        for (auto atom_it = res_it->atoms_begin(); atom_it != res_it->atoms_end(); ++atom_it)
        {
            if (atom_it->second.type() != dsrpdb::Atom::H)
            {
                const dsrpdb::Atom & atom = atom_it->second;

                double mass = get_mass(atom.type());

                const dsrpdb::Point & pt = atom.cartesian_coords();
                switch (atom_it->first)
                {
                case dsrpdb::Residue::AL_C:
                case dsrpdb::Residue::AL_CA:
                case dsrpdb::Residue::AL_N:
                case dsrpdb::Residue::AL_CB:
                case dsrpdb::Residue::AL_O:
                    // backbone
                    wb += mass;
                    bx += mass * pt.x();
                    by += mass * pt.y();
                    bz += mass * pt.z();
                    break;
                    // ignore CB, just like Phaistos - NOT ANYMORE
                    break;
                default:
                    // assume everything else is sidechain
                    ws += mass;
                    sx += mass * pt.x();
                    sy += mass * pt.y();
                    sz += mass * pt.z();
                    break;
                }
            }
        }

        if (wb != 0)
        {
            bx /= wb;
            by /= wb;
            bz /= wb;
        }
        bbody.s[0] = FLT_T(bx);
        bbody.s[1] = FLT_T(by);
        bbody.s[2] = FLT_T(bz);

        unsigned int index = 0;     // backbone body index
        if (res_type == dsrpdb::Residue::Type::ALA || res_type == dsrpdb::Residue::Type::GLY)
            index = pdb_to_ff_res_map[res_type];
        set_factor_index(bbody, index);
        v_bodies_.push_back(bbody);

        if (res_type != dsrpdb::Residue::Type::ALA && res_type != dsrpdb::Residue::Type::GLY)
        {
            if (ws != 0)
            {
                sx /= ws;
                sy /= ws;
                sz /= ws;
            }
            sbody.s[0] = FLT_T(sx);
            sbody.s[1] = FLT_T(sy);
            sbody.s[2] = FLT_T(sz);

            set_factor_index(sbody, pdb_to_ff_res_map[res_type]);
            v_bodies_.push_back(sbody);
        }
    }
}

/// Load atomic models from a PDB file.
template <typename FLT_T>
decltype(auto) calc_saxs<FLT_T>::load_pdb_atomic(const string & filename) const
{
    parse_pdb<FLT_T> parser(filename);
    parser.set_verbose_level(typename decltype(parser)::verbose_levels(verbose_lvl_));
    return parser.all_models_as_complex_atoms();
}

/// Load atomic model from a collection of PDB files.
template <typename FLT_T>
void calc_saxs<FLT_T>::load_pdb_atomic(const std::vector<std::string> & filenames)
{
    v_models_.clear();

    for (const auto & filename : filenames)
    {
        const auto & models = load_pdb_atomic(filename);
        // append the models from this PDB with move semantics
        for (auto & model : models)
        {
            v_models_.push_back(move(model));
        }
    }

    v_bodies_ = v_models_[0];
}


template struct calc_saxs<float>;
template struct calc_saxs<double>;

} // namespace
