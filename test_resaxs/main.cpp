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

#include "utils.hpp"
#include "SAXSTest.hpp"

using namespace resaxs;

int main(int argc, char ** argv)
{
    std::cout << std::endl << "----- Testing SAXS curve program -----" << std::endl << std::endl;

    // 1. process the command line
    try
    {
        if (argc <= 1)
            throw -1;

        std::string dev_spec;
        std::string file_root;
        bool ask_test = false;
        int length = 0;
        int repeats = 0;
        int wf_size = 0;
        for (int i = 1; i < argc; i++)
        {
            std::string par = argv[i];
            if (par.find("-dev:") == 0)
            {
                dev_spec = par.substr(5);
            }
            else if (par.find("-test") == 0)
                ask_test = true;
            else if (par.find("-l:") == 0)
            {
                char* end;
                length = int(strtol(&par[3], &end, 10));
            }
            else if (par.find("-r:") == 0)
            {
                char* end;
                repeats = int(strtol(&par[3], &end, 10));
            }
            else if (par.find("-f:") == 0)
            {
                file_root = par.substr(3);
            }
            else if (par.find("-w:") == 0)
            {
                char* end;
                wf_size = int(strtol(&par[3], &end, 10));
            }
        }

        if (!dev_spec.empty())
        {
            SAXSTest* raw_test = file_root.empty() ? new SAXSTest(length, repeats, wf_size) : new SAXSTest(file_root, repeats, wf_size);
            std::unique_ptr<SAXSTest> test(raw_test);
            test->run(dev_spec, ask_test);
        }
        else
            throw -1;
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
