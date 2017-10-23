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
#include <algorithm>
#include <random>

#define SAXS_PROFILE_FILE_NOT_FOUND     -1101

namespace resaxs {

template <typename FLT_T>
class saxs_profile
{
public:
    saxs_profile() = default;                       // an empty profile is valid
    saxs_profile(saxs_profile &&) = default;        // all of the data is movable
    saxs_profile(const saxs_profile &) = default;   // force default copy constructor, since custom move constructor disables it
    saxs_profile & operator=(const saxs_profile &) = default; // force default assignment, since custom move constructor disables it

    /// use move semantics to initialize from temp buffers
    saxs_profile(std::vector<FLT_T> && v_q, std::vector<FLT_T> && v_Iq, std::vector<FLT_T> && v_error) :
        v_q_(std::move(v_q)), v_Iq_(std::move(v_Iq)), v_error_(std::move(v_error))
    {
        // if some errors were missing, generate all
        if (v_error_.size() < v_q_.size())
            generate_errors();

        // generate squared precision as 1 / error^2
        v_precision2_.resize(v_error_.size());
        std::transform(v_error_.begin(), v_error_.end(), v_precision2_.begin(),
            [](FLT_T e) -> FLT_T { return 1 / (e * e); });
    }

    void initialize(std::vector<FLT_T> && v_q)
    {
        v_q_ = std::move(v_q);
        v_Iq_.resize(v_q_.size());
    }

    void initialize(const std::vector<FLT_T> & v_q)
    {
        // use a temporary to invoke move version
        initialize(std::vector<FLT_T>(v_q));
    }

    size_t size() const
    {
        return v_q_.size();
    }

    bool initialized() const
    {
        return size() > 0;
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
    static saxs_profile read_from_file(const std::string & filename)
    {
        std::ifstream profile_file(filename.c_str());
        verify(bool(profile_file), SAXS_PROFILE_FILE_NOT_FOUND, filename + " file not found");

        // temp buffers
        std::vector<FLT_T> v_q;
        std::vector<FLT_T> v_Iq;
        std::vector<FLT_T> v_error;

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
            v_q.push_back(q_elem);
            v_Iq.push_back(iq_elem);

            // try to read the error
            line_s >> error_elem;
            if (line_s)
            {
                v_error.push_back(error_elem);
            }
        }

        return saxs_profile(std::move(v_q), std::move(v_Iq), std::move(v_error));
    }

    /// Find the scale factor for another profile that optimizes its fit with this one,
    /// using linear least squares.
    /// \param[in] other_Iq An intensity profile vector for the same scattering angles as our profile.
    FLT_T optimize_scale_for(const std::vector<FLT_T> &other_Iq) const
    {
        // argmin(c) sum((a - cb)^2) = sum(ab) / sum(b^2)
        //   a = I(q)/e, b = other_I(q)/e
        FLT_T num = 0;
        FLT_T denom = 0;
        for (auto i = 0; i < v_Iq_.size(); ++i)
        {
            num += v_Iq_[i] * other_Iq[i] * v_precision2_[i];
            denom += other_Iq[i] * other_Iq[i] * v_precision2_[i];
        }
        return num / denom;
    }

private:
    std::vector<FLT_T> v_precision2_;       // precision^2 = 1 / error^2

    /// Generate simulated errors based on the intensities and a Poisson distribution.
    void generate_errors()
    {
        v_error_.resize(size());

        std::default_random_engine generator;
        std::poisson_distribution<int> poisson(10.0);

        for (unsigned int i = 0; i < v_q_.size(); ++i)
        {
            // Error is 15%, scaled by Poisson factors and q
            auto rnd = std::abs(poisson(generator) / 10.0 - 1.0) + 1.0;
            v_error_[i] = FLT_T(0.15 * v_Iq_[i] * (v_q_[i] + 0.001) * rnd);
        }
    }
};

}