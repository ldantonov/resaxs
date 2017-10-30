#ifndef CALC_SAXS_HPP
#define CALC_SAXS_HPP

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

#include <vector>
#include <string>
#include <memory>
#include <cassert>
#include "resaxs.hpp"
#include "utils.hpp"
#include "host_debye.hpp"
#include "saxs_algorithm.hpp"
#include "saxs_profile.hpp"

namespace resaxs
{

    template <typename FLT_T>
    struct profile_param
    {
    public:
        profile_param(FLT_T value, bool fit = false) : value_(value), fit_(fit) {}
        profile_param(const profile_param &) = default;

        operator const FLT_T () const { return value_; }
        bool fit() const { return fit_; }

        void fix(FLT_T value) { value_ = value; fit_ = false; }

    private:
        bool fit_;
        FLT_T value_;
    };

    template <typename FLT_T>
    struct profile_params
    {
    public:
        profile_param<FLT_T> scale_ = { 1, false };           // scale of the profile relative to the reference
        profile_param<FLT_T> water_weight_ = { 0, false };    // weight parameter for the water layer
        saxs_profile<FLT_T> ref_profile_;       // reference (e.g. experimental) profile
    };

    template <typename FLT_T>
    class calc_cl_saxs
    {
        using real4 = typename real_type<FLT_T>::real4;
        using alg_base = algorithm::cl_base<FLT_T>;
        using saxs_class = algorithm::i_saxs<FLT_T, alg_base>;

    public:
        profile_params<FLT_T> params_;

    public:
        calc_cl_saxs(algorithm::saxs_enum alg_pick, const std::string & dev_spec, unsigned int wf_size) :
            wf_size_(wf_size), saxs_alg_(algorithm::create_saxs_ocl<FLT_T>())
        {
            parse_dev_spec(dev_spec, devices_);
        }

        /// re-initialize the calculation for a new set of bodies
        ///
        void reinit(const std::vector<real4> & bodies)
        {
            assert(params_.ref_profile_.initialized());

            auto & saxs_params = saxs_alg_->access_params();
            saxs_params.initialize(params_.ref_profile_.v_q_, bodies, devices_, wf_size_);

            set_params();
            saxs_alg_->initialize();
        }

        /// calculate for the specified body positions
        ///
        void calc(const std::vector<real4> & bodies, std::vector<FLT_T> & v_Iq)
        {
            if (saxs_alg_->computing())
            {
                set_params();
                bool pos_changed = memcmp(bodies.data(), prev_bodies.data(), bodies.size() * sizeof(real4)) != 0;
                saxs_alg_->recalc_curve(bodies, 0, pos_changed ? (unsigned int)bodies.size() : 0, v_Iq);
                if (pos_changed)
                    prev_bodies = bodies;
            }
            else
            {
                if (saxs_alg_->initialized())
                    set_params();
                else
                    reinit(bodies);
                saxs_alg_->calc_curve(bodies, v_Iq);
                prev_bodies = bodies;
            }
        }

    private:
        std::vector<dev_id> devices_;
        unsigned int wf_size_;
        std::unique_ptr<saxs_class> saxs_alg_;
        std::vector<real4> prev_bodies;

    private:
        void set_params()
        {
            auto & saxs_params = saxs_alg_->access_params();
            auto & water_params = saxs_params.get_implicit_water_params();
            water_params.set_water_weight(params_.water_weight_);
            //saxs_params.get_ff_params().set_expansion_factor(1.04f);
        }
    };

    template <typename FLT_T>
    struct calc_saxs
    {
        using real4 = typename real_type<FLT_T>::real4;

        enum verbose_levels
        {
            QUIET = 0,
            NORMAL,
            DETAILS,
            DEBUG
        };

        calc_saxs(const std::vector<std::string> & bodies_filenames, const std::string & exe_base_path, bool atomic, FLT_T q_min, FLT_T q_max, unsigned int q_n,
            const profile_params<FLT_T> & params, verbose_levels verbose_lvl);
        calc_saxs(const std::vector<std::string> & bodies_filenames, const std::string & exe_base_path, bool atomic, FLT_T q_min, FLT_T q_max, unsigned int q_n,
            profile_params<FLT_T> && params, verbose_levels verbose_lvl);
        calc_saxs(const calc_saxs & other) = default;
        void set_verbose_level(verbose_levels verbose_lvl = NORMAL) { verbose_lvl_ = verbose_lvl; }

        template <typename CALC_T>
        profile_params<FLT_T> fit_ensemble(CALC_T & eval)
        {
            if (verbose_lvl_ >= NORMAL)
                cout << "Calculating ensemble average for " << v_models_.size() << " conformations.\n";

            eval.params_.ref_profile_.initialize(v_q_);

            profile_params<FLT_T> best_params{ params_.scale_, params_.water_weight_ };
            FLT_T low_chi2 = std::numeric_limits<FLT_T>::max();

            FLT_T c2 = params_.water_weight_.fit() ? FLT_T(-2.0) : params_.water_weight_;
            FLT_T max_c2 = params_.water_weight_.fit() ? FLT_T(4.0) : params_.water_weight_;
            do
            {
                eval.params_.scale_ = params_.scale_;   // reset scale param
                eval.params_.water_weight_.fix(c2);

                avg_ensemble(eval);

                FLT_T chi2 = params_.ref_profile_.chi_square(v_Iq_);
                if (chi2 < low_chi2)
                {
                    low_chi2 = chi2;
                    best_params.scale_ = eval.params_.scale_;
                    best_params.water_weight_ = c2;
                }
                c2 += FLT_T(0.1);
            } while (c2 <= max_c2);

            if (c2 > best_params.water_weight_ + FLT_T(0.1))
            {
                eval.params_.scale_.fix(best_params.scale_);
                eval.params_.water_weight_.fix(best_params.water_weight_);

                avg_ensemble(eval);
            }
            return best_params;
        }

        /// compute the ensemble average using the supplied evaluator and current parameters.
        ///
        template <typename CALC_T>
        void avg_ensemble(CALC_T & eval)
        {
            // average the intensity for all models
            v_Iq_.assign(v_Iq_.size(), FLT_T(0));
            vector<FLT_T> Iq(v_Iq_.size());
            for (const auto & bodies : v_models_)
            {
                eval.calc(bodies, Iq);
                for (auto i = 0U; i < Iq.size(); ++i)
                    v_Iq_[i] += Iq[i];
            }
            for (auto & iq : v_Iq_)
                iq /= v_models_.size();

            // fit the scale parameter, if needed, and scale the intensity
            if (params_.scale_.fit())
                eval.params_.scale_.fix(params_.ref_profile_.optimize_scale_for(v_Iq_));
            for (auto & iq : v_Iq_)
                iq *= eval.params_.scale_;
        }

        void host_saxs();
        void verify_result();

        std::vector<std::vector<real4>> v_models_;

        std::vector<real4> v_bodies_;
        std::vector<FLT_T> v_q_;
        unsigned int n_factors_;
        std::vector<FLT_T> t_factors_;
        std::vector<FLT_T> v_Iq_;
        //FLT_T scale_ = 1;

    private:
#if 0
        void parse_bodies(const std::string & filename);
#endif
        void load_pdb(const std::string & filename);
        decltype(auto) load_pdb_atomic(const std::string & filename) const;
        void load_pdb_atomic(const std::vector<std::string> & filenames);

        profile_params<FLT_T> params_;
        verbose_levels verbose_lvl_;
    };

} // namepsace

#endif // CALC_SAXS_HPP