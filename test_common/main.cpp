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

#include <iostream>
#include <memory>
#include <string>

#include "utils.hpp"
#include <tclap/CmdLine.h>

using namespace std;
using namespace resaxs;

int main(int argc, char ** argv)
{
    std::cout << std::endl << "----- Testing SAXS curve PARTIAL RECALCULATION -----" << std::endl << std::endl;

    try
    {
        TCLAP::CmdLine cmd("----- Testing SAXS curve PARTIAL RECALCULATION -----", ' ', "0.1");

        TCLAP::ValueArg<unsigned int> n_steps_arg("n","n_steps","Number of steps", true, 0, "non-negative integer");
        cmd.add(n_steps_arg);
        TCLAP::ValueArg<string> dev_arg("d","device","OpenCL device to use", false, "gpu", "device descriptor: {platform-id}-{device-id}, e.g. 0-0");
        cmd.add(dev_arg);

        cmd.parse(argc, argv);

        unsigned int n_steps = n_steps_arg.getValue();
        string device = dev_arg.getValue();
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
/*    catch(resaxs::error e)
    {
        std::cerr << "ERROR: " << e.what() << "(" << e.err() << ")" << std::endl;
        return -1;
    }
    catch(cl::Error e)
    {
        std::cerr << "ERROR: " << e.what() << "(" << e.err() << ")" << std::endl;
        return -1;
    }*/
    catch(std::exception e)
    {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}
