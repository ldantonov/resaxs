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
        TCLAP::ValueArg<string> verbose_lvl("v", "verbose", "Verbosity level", false, "normal", &verbose_constr);
        cmd.add(verbose_lvl);

        TCLAP::SwitchArg verify_res("t", "test", "Verify the results against the simple host standard.", false);
        cmd.add(verify_res);

        TCLAP::SwitchArg dp_alg("", "double", "Turn on double precision.", false);
        cmd.add(dp_alg);
        TCLAP::ValueArg<string> dev_arg("d", "device", "OpenCL device to use", false, "cpu", "device descriptor: {platform-id}-{device-id}, e.g. 0-0");
        cmd.add(dev_arg);

        TCLAP::ValueArg<float> water_weight("w", "water", "Weight of the water layer contribution.", false, 0, "floating-point value");
        cmd.add(water_weight);

        TCLAP::ValueArg<unsigned int> q_n("n", "q_n", "Number of q values.", false, 500, "floating-point value");
        cmd.add(q_n);
        TCLAP::ValueArg<float> q_max("", "q_max", "Maximum value for q.", false, 0.75, "floating-point value");
        cmd.add(q_max);
        TCLAP::ValueArg<float> q_min("", "q_min", "Starting value for q.", false, 0, "floating-point value");
        cmd.add(q_min);

        TCLAP::SwitchArg ensemble_arg("e", "ensemble", "Calculate ensemble average, if more than one conformation is available.", false);
        cmd.add(ensemble_arg);

        TCLAP::ValueArg<string> int_filename("o", "outfile", "output file of SAXS intensities", true, "", "file name");
        cmd.add(int_filename);

        TCLAP::UnlabeledMultiArg<string> pdb_filenames("pdb_files", "input PDB file names", true, "");
        cmd.add(pdb_filenames);

        cmd.parse(argc, argv);
        unsigned int n_verbose_lvl = static_cast<unsigned int>(distance(verbose_choice.cbegin(), find(verbose_choice.cbegin(), verbose_choice.cend(), verbose_lvl.getValue())));

        //unsigned int n_steps = n_steps_arg.getValue();
        string device = dev_arg.getValue();
        //string algorithm = alg_arg.getValue();
        bool dp = dp_alg.getValue();

        if (n_verbose_lvl >= calc_saxs<float>::NORMAL)
            std::cout << std::endl << "----- Calculating SAXS curve -----" << std::endl << std::endl;

        if (dp)
        {
            calc_saxs<double> calc(pdb_filenames.getValue()[0], getexepath(), true, q_min.getValue(), q_max.getValue(), q_n.getValue(),
                water_weight.getValue(),
                calc_saxs<double>::verbose_levels(n_verbose_lvl));
            if (device == "host")
                calc.host_saxs();

            cout << calc.v_bodies_.size() << " bodies generated." << endl;
            ofstream outfile(int_filename.getValue());
            for (unsigned int i = 0; i < calc.v_q_.size(); ++i)
            {
                outfile << fixed << calc.v_q_[i] << "     \t" << calc.v_Iq_[i] << endl;
            }
        }
        else
        {
            calc_saxs<float> calc(pdb_filenames.getValue()[0], getexepath(), true, q_min.getValue(), q_max.getValue(), q_n.getValue(),
                water_weight.getValue(),
                calc_saxs<float>::verbose_levels(n_verbose_lvl));
            if (device == "host")
                calc.host_saxs();
            else if (ensemble_arg.getValue())
                calc.cl_saxs_ensemble(algorithm::saxs_enum::saxs_gpu_pt_wf, device, 64);
            else
                calc.cl_saxs(algorithm::saxs_enum::saxs_gpu_pt_wf, device, 64);
            //alg_test_data<float, cl_float4> init_data(42, 1888);
            //recalc_test<float, cl_float4> tester(init_data, 64);
            //tester.run(algorithm, device, n_steps);

            if (n_verbose_lvl > 0)
                cout << calc.v_bodies_.size() << " bodies generated." << endl;

            ofstream outfile(int_filename.getValue());
            for (unsigned int i = 0; i < calc.v_q_.size(); ++i)
            {
                outfile << fixed << calc.v_q_[i] << "     \t" << calc.v_Iq_[i] << endl;
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
