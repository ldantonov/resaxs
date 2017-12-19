///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011 Lubo Antonov
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
#include <memory>
#include <string>
#include <sstream>
#include <ctime>
#include <regex>

#include "utils.hpp"
#include <tclap/CmdLine.h>
#include "calc_saxs.hpp"

using namespace std;
using namespace resaxs;

#ifdef _WIN32
#include <windows.h>
string getexepath()
{
    vector<char> path(MAX_PATH);
    DWORD ch_read = 0;
    while (true)
    {
        ch_read = GetModuleFileName(NULL, &path[0], static_cast<DWORD>(path.size()));
        if (ch_read < path.size())
            break;
        path.resize(path.size() * 2);
    }
    string result(path.begin(), path.begin() + ch_read);
    auto found = result.find_last_of("\\");
    return result.substr(0, found) + "\\";
}
#else
#include <dlfcn.h>

string getexepath()
{
    Dl_info rInfo;
    memset(&rInfo, 0, sizeof(rInfo));

    if (!dladdr(reinterpret_cast<void *>(getexepath), &rInfo) || !rInfo.dli_fname)
    {

        // Return empty string if failed
        return "";
    }

    string result(rInfo.dli_fname);
    auto found = result.find_last_of("\\");
    return result.substr(0, found) + "\\";
}
#endif


struct water_weight_range
{
    constexpr static float min_ = -2.0f;
    constexpr static float max_ = 4.0f;
    constexpr static float default_ = 0;
};

struct exp_factor_range
{
    constexpr static float min_ = 0.95f;
    constexpr static float max_ = 1.05f;
    constexpr static float default_ = 1;
};

template <typename RANGE>
struct range_arg : public RANGE
{
    float lo_ = this->default_;
    float hi_ = this->default_;

    bool singular() const { return lo_ == hi_; }

    string min_max_str() const
    {
        return string("[") + to_string(this->min_) + ',' + to_string(this->max_) + ']';
    }

    range_arg& operator=(const string& str)
    {
        string s = trim(str, "[](){} ");
        auto elems = split(s, ',');

        if (elems.size() == 2)
        {
            // a range between two numbers
            elems[0] = trim(elems[0]);
            elems[1] = trim(elems[1]);
            std::istringstream iss(elems[0]);
            if (!(iss >> lo_))
                throw TCLAP::ArgParseException(elems[0] + " is not a real number");
            iss.seekg(0);
            iss.str(elems[1]);
            if (!(iss >> hi_))
                throw TCLAP::ArgParseException(elems[1] + " is not a real number");
            if (lo_ < this->min_ || hi_ > this->max_ || lo_ > hi_)
                throw TCLAP::ArgParseException(s + " is not a valid range in " + min_max_str());
        }
        else if (elems.size() == 1)
        {
            // a single value
            elems[0] = trim(elems[0]);
            std::istringstream iss(elems[0]);
            if (!(iss >> lo_))
                throw TCLAP::ArgParseException(elems[0] + " is not a real number");
            hi_ = lo_;
            if (lo_ < this->min_ || hi_ > this->max_)
                throw TCLAP::ArgParseException(elems[0] + " is not in the valid range " + min_max_str());
        }
        else
            throw TCLAP::ArgParseException(s + " is not a real number or a valid range");

        return *this;
    }
};

namespace TCLAP {

    template<> struct ArgTraits<range_arg<water_weight_range>> {
        typedef StringLike ValueCategory;
    };

    template<> struct ArgTraits<range_arg<exp_factor_range>> {
        typedef StringLike ValueCategory;
    };

}

/// Create a profile_param out of command line arguments. The fit option will be set appropriately.
///     arg             - command line argument
///     fit_filename    - command line argument for the intensity file to use for fitting
template <typename FLT_T>
profile_param<FLT_T> make_param(const TCLAP::ValueArg<float> & arg, const TCLAP::ValueArg<string> & fit_filename)
{
    bool fit = !arg.isSet() && !const_cast<TCLAP::ValueArg<string> &>(fit_filename).getValue().empty();
    return fit ? profile_param<FLT_T>() : profile_param<FLT_T>(const_cast<TCLAP::ValueArg<float> &>(arg).getValue());
};

template <typename FLT_T, typename RANGE>
profile_param<FLT_T> make_param(const TCLAP::ValueArg<range_arg<RANGE>> & arg, const TCLAP::ValueArg<string> & fit_filename)
{
    const auto & value = const_cast<TCLAP::ValueArg<range_arg<RANGE>> &>(arg).getValue();

    // if arg is default and there's a fit file, fit over the default range
    bool default_range = !arg.isSet() && !const_cast<TCLAP::ValueArg<string> &>(fit_filename).getValue().empty();
    return { default_range ? value.min_ : value.lo_, default_range ? value.max_ : value.hi_ };
};

int main(int argc, char ** argv)
{
    //report_opencl_caps();

    try
    {
        TCLAP::CmdLine cmd("----- Testing SAXS curve PARTIAL RECALCULATION -----", ' ', "0.1");

        //vector<string> alg_choice;
        //alg_choice.push_back("std");
        //alg_choice.push_back("pt_wf");
        //TCLAP::ValuesConstraint<string> alg_constr(alg_choice);
        //TCLAP::ValueArg<string> alg_arg("a","algorithm","Algorithm to run", false, "std", &alg_constr);
        //cmd.add(alg_arg);
        vector<string> verbose_choice;
        verbose_choice.push_back("quiet");
        verbose_choice.push_back("normal");
        verbose_choice.push_back("details");
        verbose_choice.push_back("debug");
        TCLAP::ValuesConstraint<string> verbose_constr(verbose_choice);
        TCLAP::ValueArg<string> verbose_lvl("v", "verbose", "Verbosity level", false, "normal", &verbose_constr, cmd);

        TCLAP::SwitchArg verify_res("t", "test", "Verify the results against the simple host standard.", false);
        cmd.add(verify_res);

        TCLAP::SwitchArg dp_alg("", "double", "Turn on double precision.", false);
        cmd.add(dp_alg);

        TCLAP::ValueArg<string> dev_arg("d", "device", "OpenCL device to use", false, "cpu", "device descriptor: {platform-id}-{device-id}, e.g. 0-0; cpu; gpu", cmd);

        TCLAP::ValueArg<string> fit_filename("f", "fit", "Fit to experimental profile", false, "", "file name", cmd);

        TCLAP::ValueArg<float> scale("s", "scale", "SAXS profile intensity scale.", false, 1.0, "floating-point value", cmd);

        TCLAP::ValueArg<range_arg<water_weight_range>> water_weight("w", "water", "Weight of the water layer contribution.", false,
            range_arg<water_weight_range>(), "range [,] or floating_point value", cmd);

        TCLAP::ValueArg<range_arg<exp_factor_range>> exp_factor("e", "exp_factor", "Excluded volume expansion factor.", false,
            range_arg<exp_factor_range>(), "range [,] or floating_point value", cmd);

        TCLAP::ValueArg<unsigned int> q_n("n", "q_n", "Number of q values.", false, 500, "whole number", cmd);
        TCLAP::ValueArg<float> q_max("", "q_max", "Maximum value for q.", false, 0.75, "floating-point value", cmd);
        TCLAP::ValueArg<float> q_min("", "q_min", "Starting value for q.", false, 0, "floating-point value", cmd);

        // A list of input PDB files
        TCLAP::UnlabeledMultiArg<string> pdb_filenames("pdb_files", "input PDB file names", true, "", cmd);

        TCLAP::ValueArg<string> out_filename("o", "outfile", "output file of SAXS intensities", true, "", "file name", cmd);

        cmd.parse(argc, argv);

        if (!water_weight.getValue().singular() && fit_filename.getValue().empty())
            throw TCLAP::ArgParseException("Range can't be used without an experimental profile; use a single value instead.");

        if (!exp_factor.getValue().singular() && fit_filename.getValue().empty())
            throw TCLAP::ArgParseException("Range can't be used without an experimental profile; use a single value instead.");

        unsigned int n_verbose_lvl = static_cast<unsigned int>(distance(verbose_choice.cbegin(), find(verbose_choice.cbegin(), verbose_choice.cend(), verbose_lvl.getValue())));

        //unsigned int n_steps = n_steps_arg.getValue();
        string device = dev_arg.getValue();
        //string algorithm = alg_arg.getValue();
        bool dp = dp_alg.getValue();

        if (n_verbose_lvl >= calc_saxs<float>::NORMAL)
            std::cout << std::endl << "----- Calculating SAXS curve -----" << std::endl << std::endl;

        if (dp)
        {
            calc_params<double> params{ make_param<double>(scale, fit_filename), 
                make_param<double>(water_weight, fit_filename),
                make_param<double>(exp_factor, fit_filename),
                saxs_profile<double>::read_from_file(fit_filename.getValue()) };
            calc_saxs<double> calc(pdb_filenames.getValue(), getexepath(), true, q_min.getValue(), q_max.getValue(), q_n.getValue(),
                move(params), calc_saxs<double>::verbose_levels(n_verbose_lvl));
            if (device == "host")
                calc.host_saxs();

            if (n_verbose_lvl > calc_saxs<float>::QUIET)
                cout << calc.v_bodies_.size() << " bodies generated." << endl;

            ofstream outfile(out_filename.getValue());
            for (unsigned int i = 0; i < calc.v_q_.size(); ++i)
            {
                outfile << fixed << calc.v_q_[i] << "     \t" << calc.intensity_[i] << endl;
            }
        }
        else
        {
            clock_t t1 = clock();

            calc_params<float> params{ make_param<float>(scale, fit_filename),
                make_param<float>(water_weight, fit_filename),
                make_param<float>(exp_factor, fit_filename),
                saxs_profile<float>::read_from_file(fit_filename.getValue()) };
            calc_saxs<float> calc(pdb_filenames.getValue(), getexepath(), true, q_min.getValue(), q_max.getValue(), q_n.getValue(),
                move(params), calc_saxs<float>::verbose_levels(n_verbose_lvl));

            if (n_verbose_lvl > calc_saxs<float>::QUIET)
                cout << calc.v_bodies_.size() << " bodies generated." << endl;

            cout << endl << "Preprocessing time: " << double(clock() - t1) * 1000 / CLOCKS_PER_SEC << "ms" << endl << endl;
            t1 = clock();

            fitted_params<float> best_params;

            if (device == "host")
                calc.host_saxs();
            else
            {
                calc_cl_saxs<float> evaluator(algorithm::saxs_enum::saxs_gpu_pt_wf, device, 64);
                best_params = calc.fit_ensemble(evaluator);
                calc.intensity_ = best_params.intensity_; // TODO: remove usage of calc.intensity_;
            }
            
            cout << endl << "SAXS calc time: " << double(clock() - t1) * 1000 / CLOCKS_PER_SEC << "ms" << endl << endl;

            ofstream outfile(out_filename.getValue());
            outfile << "# scale: " << best_params.scale_ << ", water weight: " << best_params.water_weight_
                << ", expansion factor: " << best_params.exp_factor_ << ", Chi: " << best_params.chi_ << endl;
            for (unsigned int i = 0; i < calc.v_q_.size(); ++i)
            {
                outfile << fixed << calc.v_q_[i] << "     \t" << calc.intensity_[i] << endl;
            }

            if (verify_res.getValue())
                calc.verify_result();
        }
    }
    catch (const TCLAP::ArgException & e)  // catch any exceptions
    {
        std::cerr << "error: " << e.error() << " for arg " << e.argId() << std::endl;
    }
    catch(int)
    {
        // something is wrong with the command line - just report system caps
        report_opencl_caps();
    }
    catch(resaxs::error e)
    {
        std::cerr << "ERROR: " << e.what() << "(" << e.err() << ")" << std::endl;
        return -1;
    }
    catch(cl::Error e)
    {
        std::cerr << "ERROR: " << e.what() << "(" << e.err() << ")" << std::endl;
        return -1;
    }
    catch(std::exception e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
