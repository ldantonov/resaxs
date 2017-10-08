///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2014-2017 Lubo Antonov
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

#include <vector>
#include <random>

#define SAXS_PROFILE_FILE_NOT_FOUND     -1101

namespace resaxs {

template <typename FLT_T>
class saxs_profile
{
public:
    bool initialized() const
    {
        return v_q_.size() > 0;
    }

    explicit operator bool() const
    {
        return initialized();
    }

    std::vector<FLT_T> v_q_;
    std::vector<FLT_T> v_Iq_;
    std::vector<FLT_T> v_error_;

    /// read a SAXS profile from a text file. Expected format is 3 columns: q, I(q) and sigma(q).
    /// Comment lines start with #. 
    static saxs_profile read_from_file(std::string filename)
    {
        saxs_profile profile;
        if (filename.empty())
            return profile;

        std::ifstream profile_file(filename.c_str());
        verify(!profile_file, SAXS_PROFILE_FILE_NOT_FOUND, filename + " file not found");

        std::string line;
        while (!profile_file.eof())
        {
            getline(profile_file, line);
            line = trim(line);
            // skip comments and blank lines
            if (line.empty() || line.find_first_of('#') == 0)
                continue;

            FLT_T q_elem, iq_elem, error_elem;
            istringstream line_s(line);
            line_s >> q_elem >> iq_elem;
            // skip misformed lines
            if (!line_s)
                continue;
            profile.v_q_.push_back(q_elem);
            profile.v_Iq_.push_back(iq_elem);

            // try to read the error
            line_s >> error_elem;
            if (line_s)
            {
                profile.v_error_.push_back(error_elem);
            }
        }

        // if some errors were missing, generate all
        if (profile.v_error_.size() < profile.v_Iq_.size())
            profile.generate_errors();

        return profile;
    }

private:
    void generate_errors()
    {
        v_error_.resize(v_q_.size());

        std::default_random_engine generator;
        std::poisson_distribution<int> poisson(10.0);

        for (unsigned int i = 0; i < v_q_.size(); ++i)
        {
            // Error is 3% , scaled by Poisson factors and 5 * q
            auto rnd = std::abs(poisson(generator) / 10.0 - 1.0) + 1.0;
            v_error_[i] = FLT_T(0.03 * v_Iq_[i] * 5.0 * (v_q_[i] + 0.001) * rnd);
        }
    }
};

}