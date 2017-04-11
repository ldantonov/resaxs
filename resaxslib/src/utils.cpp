///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011-2015 Lubo Antonov
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
#include <cassert>
#include <vector>

#include "../include/utils.hpp"

#include "../include/resaxs.hpp"

using namespace std;

namespace resaxs
{

    //
    // Print out some basic information about a device
    //
    void show_device_info(int platform_idx, int idx, const cl::Device & device)
    {
        std::cout << "\tDevice " << platform_idx << '-' << idx << ':' << std::endl;
        std::cout << "\t\tName: " << device.getInfo<CL_DEVICE_NAME>(NULL) << std::endl;
        std::cout << "\t\tVendor: " << device.getInfo<CL_DEVICE_VENDOR>(NULL) << std::endl;
        std::string s;
        if (device.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>(NULL) == CL_LOCAL)
            s = "dedicated";
        else
            s = "emulated";
        std::cout << "\t\tLocal memory: " << s << std::endl;
    }

    //
    // Print out some basic information about a platform
    //
    void show_platform_info(int idx, const cl::Platform & platform)
    {
        std::cout << "Platform " << idx << ':' << std::endl;
        std::cout << "\tName: " << platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
        std::cout << "\tVendor: " << platform.getInfo<CL_PLATFORM_VENDOR>() << std::endl;
        std::cout << "\tVersion: " << platform.getInfo<CL_PLATFORM_VERSION>() << std::endl;

        std::vector<cl::Device> devices;
        platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
        if (devices.size() == 0)
            std::cout << "\tNo devices detected." << std::endl;
        else
        {
            std::cout << '\t' << devices.size() << " devices detected." << std::endl;
            int device_idx = 0;
            for (std::vector<cl::Device>::const_iterator i = devices.begin(); i != devices.end(); i++)
                show_device_info(idx, device_idx++, *i);
        }
    }

    //
    //  Report details on the OpenCL platforms and devices to cout.
    //  Useful for command line utils
    //
    void report_opencl_caps()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0)
            std::cout << "No platforms detected." << std::endl;
        else
        {
            std::cout << platforms.size() << " platforms detected." << std::endl;
            int idx = 0;
            for (std::vector<cl::Platform>::const_iterator i = platforms.begin(); i != platforms.end(); i++)
                show_platform_info(idx++, *i);
        }
    }

    //
    //  Parse a string with device specifications for algorithm execution.
    //  NOTE: only a single dev spec is currenty supported.
    //      dev_spec            - sequence of 'gpu | cpu | <platform>-<device>', separated by whitespace
    //      dev_ids             - result usable for initializing algorithms
    //
    void parse_dev_spec(const std::string & dev_spec, std::vector<dev_id> & dev_ids)
    {
        dev_id device;
        device.platform = -1;
        device.device = -1;
        device.type = GPU;

        if (dev_spec.compare("gpu") == 0)
            device.type = GPU;
        else if (dev_spec.compare("cpu") == 0)
            device.type = CPU;
        else
        {
            // the device spec has a format of [platform-device]
            std::string::size_type sep_idx = dev_spec.find('-');
            if (sep_idx == std::string::npos)
                throw -1;

            // extract the platform and device separately
            std::string s_platform = dev_spec.substr(0, sep_idx);
            std::string::size_type bropen_idx = dev_spec.find('(', sep_idx + 1);
            std::string s_device;
            if (bropen_idx == std::string::npos)
                s_device = dev_spec.substr(sep_idx + 1);
            else
            {
                s_device = dev_spec.substr(sep_idx + 1, bropen_idx - sep_idx - 1);
                std::string s_dev_type = dev_spec.substr(bropen_idx + 1);
                if (s_dev_type.find("cpu") == 0)
                    device.type = CPU;
            }
            if (s_platform.empty() || s_device.empty())
                throw -1;

            // convert to indices
            char* end;
            device.platform = strtol(s_platform.c_str(), &end, 0);
            device.device = strtol(s_device.c_str(), &end, 0);
        }

        dev_ids.push_back(device);
    }

    vector<cl::Device> get_cl_devices(const vector<dev_id> & dev_ids)
    {
        verify(!dev_ids.empty(), error::SAXS_INVALID_DEVICE);

        assert(dev_ids.size() == 1);    // current implementation only uses the first device

        vector<cl::Device> result;

        vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);

        vector<cl::Device> devices;
        if (dev_ids[0].platform == -1 || dev_ids[0].device == -1)   // -1 is a wildcard
        {
            // look for the first device of the correct type
            cl_device_type dev_type;
            switch (dev_ids[0].type)
            {
            case CPU:
                dev_type = CL_DEVICE_TYPE_CPU;
                break;
            case GPU:
                dev_type = CL_DEVICE_TYPE_GPU;
                break;
            default:
                dev_type = CL_DEVICE_TYPE_ALL;
                break;
            }

            for (auto & platform : platforms)
            {
                try
                {
                    platform.getDevices(dev_type, &devices);
                }
                catch (cl::Error e)
                {
                    continue;   // check other platforms
                }

                result.push_back(devices[0]);    // right now only save the first device
                break;
            }
        }
        else
        {
            // a device was specified by ID, so find it
            verify(dev_ids[0].platform < static_cast<long>(platforms.size()), error::SAXS_INVALID_DEVICE);

            platforms[dev_ids[0].platform].getDevices(CL_DEVICE_TYPE_ALL, &devices);
            result.push_back(devices[dev_ids[0].device]);
        }

        verify(!result.empty(), error::SAXS_INVALID_DEVICE);
        return result;
    }

    // Map converts PDB reader simple atom types to form factor atom types
    //
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
    
    atoms::atom_type map_pdb_atom_to_ff_type(const string& atom_label, const string& res_type)
    {
        atoms::atom_type element;
        auto element_entry = pdb_to_ff_simple_atom_map.find(atom_label.substr(0, 1));
        if (element_entry != pdb_to_ff_simple_atom_map.end())
            element = element_entry->second;
        else
        {
            cerr << "Missing form factor for element " << atom_label.substr(0, 1) << " in residue " << res_type <<  " - using N instead\n";
            element = atoms::N;
        }
        
        return map_pdb_atom_to_ff_type(element, atom_label, res_type);
    }
    
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
    
    //
    // Trim leading and trailing white space from a string
    //
    string trim(const string& str)
    {
        string::size_type pos1 = str.find_first_not_of(' ');
        string::size_type pos2 = str.find_last_not_of(' ');
        return str.substr(pos1 == string::npos ? 0 : pos1,
            pos2 == string::npos ? str.length() - 1 : pos2 - pos1 + 1);
    }
    
} // namespace
