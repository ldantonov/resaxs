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

    template <typename T, typename U,
        typename = std::enable_if_t<std::is_arithmetic<T>::value>,
        typename = std::enable_if_t<std::is_arithmetic<U>::value>>
        inline std::vector<T> & operator*=(std::vector<T> & v, U c)
    {
        for (auto & e : v)
            e *= c;
        return v;
    }

    template <typename T, typename U,
        typename = std::enable_if_t<std::is_arithmetic<T>::value>,
        typename = std::enable_if_t<std::is_arithmetic<U>::value>>
        inline std::vector<T> & operator/=(std::vector<T> & v, U c)
    {
        for (auto & e : v)
            e /= c;
        return v;
    }

    template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
    inline std::vector<T> & operator+=(std::vector<T> & v1, const std::vector<T> & v2)
    {
        for (auto i = 0U; i < v1.size(); ++i)
            v1[i] += v2[i];
        return v1;
    }


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
    struct calc_params
    {
    public:
        profile_param<FLT_T> scale_ = { 1, false };           // scale of the profile relative to the reference
        profile_param<FLT_T> water_weight_ = { 0, false };    // weight parameter for the water layer
        profile_param<FLT_T> exp_factor_ = { 1, false };      // expansion factor parameter for the excluded volume
        saxs_profile<FLT_T> ref_profile_;       // reference (e.g. experimental) profile
    };

    template <typename FLT_T>
    struct fitted_params
    {
        FLT_T scale_ = 1;           // profile scaling factor
        FLT_T water_weight_ = 0;    // weight parameter for the water layer
        FLT_T exp_factor_ = 1;        // expansion factor parameter for the excluded volume
        FLT_T chi2_ = std::numeric_limits<FLT_T>::max();
        std::vector<FLT_T> intensity_;

        bool operator ==(const fitted_params<FLT_T> & other) const
        {
            return chi2_ == other.chi2_;
        }

        bool operator <(const fitted_params<FLT_T> & other) const
        {
            return chi2_ < other.chi2_;
        }

        bool operator <=(const fitted_params<FLT_T> & other) const
        {
            return chi2_ <= other.chi2_;
        }

        bool operator >(const fitted_params<FLT_T> & other) const
        {
            return !operator <=(other);
        }

        bool operator >=(const fitted_params<FLT_T> & other) const
        {
            return !operator <(other);
        }

        template <typename FLT_T>
        friend std::ostream & operator <<(std::ostream & os, const fitted_params<FLT_T> & params);
    };

    template <typename FLT_T>
    inline std::ostream & operator <<(std::ostream & os, const fitted_params<FLT_T> & params)
    {
        os << "# scale: " << params.scale_ << ", water weight: " << params.water_weight_
            << ", expansion factor: " << params.exp_factor_ << ", Chi: " << sqrt(params.chi2_) << endl;
        return os;
    }

    template <typename FLT_T>
    class calc_cl_saxs
    {
        using real4 = typename real_type<FLT_T>::real4;
        using alg_base = algorithm::cl_base<FLT_T>;
        using saxs_class = algorithm::i_saxs<FLT_T, alg_base>;

    public:
        calc_params<FLT_T> params_;
        unsigned int calc_count = 0;

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
            ++calc_count;
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
            saxs_params.get_ff_params().set_expansion_factor(params_.exp_factor_);
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
            const calc_params<FLT_T> & params, verbose_levels verbose_lvl);
        calc_saxs(const std::vector<std::string> & bodies_filenames, const std::string & exe_base_path, bool atomic, FLT_T q_min, FLT_T q_max, unsigned int q_n,
            calc_params<FLT_T> && params, verbose_levels verbose_lvl);
        calc_saxs(const calc_saxs & other) = default;
        void set_verbose_level(verbose_levels verbose_lvl = NORMAL) { verbose_lvl_ = verbose_lvl; }

        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble(CALC_T & eval)
        {
            if (verbose_lvl_ >= NORMAL)
                cout << "Calculating ensemble average for " << v_models_.size() << " conformations.\n";

            eval.params_.ref_profile_.initialize(v_q_);

            FLT_T min_c2 = params_.water_weight_.fit() ? FLT_T(-2.0) : params_.water_weight_;
            FLT_T max_c2 = params_.water_weight_.fit() ? FLT_T(4.0) : params_.water_weight_;

            eval.params_.exp_factor_.fix(params_.exp_factor_);

            fitted_params<FLT_T> left_point = fit_ensemble(eval, min_c2);
            fitted_params<FLT_T> right_point = fit_ensemble(eval, max_c2);

            fitted_params<FLT_T> best_params = fit_ensemble(eval, left_point, right_point);

            cout << eval.calc_count << " calculations." << endl;

            return best_params;
        }

        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble(CALC_T & eval, const fitted_params<FLT_T> & fit_min, const fitted_params<FLT_T> & fit_max)
        {
            auto ww_delta = std::fabs(fit_min.water_weight_ - fit_max.water_weight_);
            auto chi_delta = std::fabs(std::sqrt(fit_min.chi2_) - std::sqrt(fit_max.chi2_)) / std::sqrt(fit_max.chi2_);
            if (ww_delta < 0.001 && chi_delta < 0.0001)
            {
                cout << "found min: " << (fit_min < fit_max ? fit_min : fit_max);
                return fit_min < fit_max ? fit_min : fit_max;
            }

            fitted_params<FLT_T> mid_point = fit_ensemble(eval, fit_min.water_weight_ + ww_delta / 2);

            if (fit_min > mid_point && mid_point > fit_max)
                return fit_ensemble(eval, mid_point, fit_max);

            if (fit_min < mid_point && mid_point < fit_max)
                return fit_ensemble(eval, fit_min, mid_point);

            fitted_params<FLT_T> left_fit = fit_ensemble(eval, fit_min, mid_point);
            fitted_params<FLT_T> right_fit = fit_ensemble(eval, mid_point, fit_max);
            return left_fit < right_fit ? left_fit : right_fit;
       }

        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble(CALC_T & eval, FLT_T ww)
        {
            eval.params_.scale_ = params_.scale_;   // reset scale param
            eval.params_.water_weight_.fix(ww);

            fitted_params<FLT_T> fit_params;
            fit_params.water_weight_ = ww;
            fit_params.exp_factor_ = params_.exp_factor_;
            fit_params.intensity_ = avg_ensemble(eval);

            // fit the scale parameter, if needed, and scale the intensity
            fit_params.scale_ = params_.scale_.fit() ? params_.ref_profile_.optimize_scale_for(fit_params.intensity_) : params_.scale_;
            fit_params.intensity_ *= fit_params.scale_;

            fit_params.chi2_ = params_.ref_profile_.chi_square(fit_params.intensity_);

            return fit_params;
        }

        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble1(CALC_T & eval)
        {
            if (verbose_lvl_ >= NORMAL)
                cout << "Calculating ensemble average for " << v_models_.size() << " conformations.\n";

            eval.params_.ref_profile_.initialize(v_q_);

            fitted_params<FLT_T> best_params;
            FLT_T prev_chi2 = best_params.chi2_;

            const unsigned int default_slices = 4;
            const unsigned int fixed_value_slices = 0;
            unsigned int ef_slices = params_.exp_factor_.fit() ? default_slices : fixed_value_slices;
            FLT_T min_c1 = params_.exp_factor_.fit() ? FLT_T(0.95) : params_.exp_factor_;
            FLT_T max_c1 = params_.exp_factor_.fit() ? FLT_T(1.05) : params_.exp_factor_;
            FLT_T delta_c1 = (max_c1 - min_c1) / ef_slices;
            unsigned int ww_slices = params_.water_weight_.fit() ? default_slices : fixed_value_slices;
            FLT_T min_c2 = params_.water_weight_.fit() ? FLT_T(-2.0) : params_.water_weight_;
            FLT_T max_c2 = params_.water_weight_.fit() ? FLT_T(4.0) : params_.water_weight_;
            FLT_T delta_c2 = (max_c2 - min_c2) / ww_slices;

            // adjust starting range to include endpoints
            min_c2 -= delta_c2;
            max_c2 += delta_c2;
            ww_slices += 2;

            eval.params_.exp_factor_.fix(params_.exp_factor_);

            int count = 0;
            do
            {
                for (unsigned int j = 1; j < ww_slices; ++j)
                {
                    FLT_T c2 = min_c2 + j * delta_c2;

                    fitted_params<FLT_T> new_params = fit_ensemble(eval, c2);
                    ++count;
                    if (new_params.chi2_ < best_params.chi2_)
                    {
                        FLT_T prev_chi2 = best_params.chi2_;

                        best_params = std::move(new_params);
                        cout << "found min: " << best_params;
                        if (std::fabs(std::sqrt(best_params.chi2_) - std::sqrt(prev_chi2)) / std::sqrt(prev_chi2) < 0.0001)
                        {
                            delta_c1 = delta_c2 = 0;
                            break;
                        }
                    }
                }

                ww_slices = default_slices;     // reset slices to exclude endpoints
                min_c2 = std::max(best_params.water_weight_ - delta_c2, FLT_T(-2.0));
                max_c2 = std::min(best_params.water_weight_ + delta_c2, FLT_T(4.0));
                delta_c2 = (max_c2 - min_c2) / ww_slices;
            } while (delta_c2 > 0.001);

            cout << count << " SAXS evals." << endl;

            return best_params;
        }

        /// compute the ensemble average using the supplied evaluator and current parameters.
        ///
        template <typename CALC_T>
        std::vector<FLT_T> avg_ensemble(CALC_T & eval) const
        {
            // average the intensity for all models
            std::vector<FLT_T> avg_iq(v_q_.size());
            std::vector<FLT_T> iq(avg_iq.size());
            for (const auto & bodies : v_models_)
            {
                eval.calc(bodies, iq);
                avg_iq += iq;
            }
            avg_iq /= v_models_.size();
            return avg_iq;
        }

        void host_saxs();
        void verify_result();

        std::vector<std::vector<real4>> v_models_;

        std::vector<real4> v_bodies_;
        std::vector<FLT_T> v_q_;
        unsigned int n_factors_;
        std::vector<FLT_T> t_factors_;
        std::vector<FLT_T> intensity_;

    private:

#if 0
        void parse_bodies(const std::string & filename);
#endif
        void load_pdb(const std::string & filename);
        decltype(auto) load_pdb_atomic(const std::string & filename) const;
        void load_pdb_atomic(const std::vector<std::string> & filenames);

        calc_params<FLT_T> params_;
        verbose_levels verbose_lvl_;
    };

} // namepsace

#endif // CALC_SAXS_HPP