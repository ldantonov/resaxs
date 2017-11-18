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
#include <initializer_list>
#include <type_traits>
#include "resaxs.hpp"
#include "utils.hpp"
#include "host_debye.hpp"
#include "saxs_algorithm.hpp"
#include "saxs_profile.hpp"

namespace resaxs
{

    /// Enhanced operations for std::vector
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

    /// Check if the list contains the value.
    template <typename T>
    inline bool contains(std::initializer_list<T> il, const T& value)
    {
        auto last = std::end(il);
        return std::find(std::begin(il), last, value) != last;
    }

    /// If the value falls below the low threshold clamp it down to zero.
    template <typename T, typename = std::enable_if_t<std::is_arithmetic<T>::value>>
    constexpr T clamp_to_zero(T value, T lo)
    {
        return value > lo ? value : 0;
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
        saxs_profile<FLT_T> ref_profile_;                     // reference (e.g. experimental) profile
    };

    template <typename FLT_T>
    struct fitted_params
    {
        FLT_T scale_ = 1;           // profile scaling factor
        FLT_T water_weight_ = 0;    // weight parameter for the water layer
        FLT_T exp_factor_ = 1;      // expansion factor parameter for the excluded volume
        FLT_T chi_ = std::numeric_limits<FLT_T>::max(); // chi goodness of fit
        std::vector<FLT_T> intensity_;  // SAXS intensity

        /// tests for identity, not just equal chi
        ///
        bool operator ==(const fitted_params<FLT_T> & other) const
        {
            return scale_ == other.scale_ && water_weight_ == other.water_weight_ && exp_factor_ == other.exp_factor_;
        }

        /// tests for equal chi
        ///
        bool same_fit(const fitted_params<FLT_T> & other) const
        {
            return chi_ == other.chi_;
        }

        bool operator <(const fitted_params<FLT_T> & other) const
        {
            return chi_ < other.chi_;
        }

        bool operator <=(const fitted_params<FLT_T> & other) const
        {
            return chi_ <= other.chi_;
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
            << ", expansion factor: " << params.exp_factor_ << ", Chi: " << params.chi_ << endl;
        return os;
    }

    template <typename FLT_T>
    class calc_cl_saxs
    {
        using real4 = typename real_type<FLT_T>::real4;
        using alg_base = algorithm::cl_base<FLT_T>;
        using saxs_class = algorithm::i_saxs<FLT_T, alg_base>;

    public:
        calc_params<FLT_T> params_;     // current params for the calculation
        unsigned int calc_count = 0;    // number of calculations performed

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

        /// Fit the ensemble parameters by depth-first binary search of the solution space.
        /// \param[in] eval SAXS evaluator object
        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble_binary(CALC_T & eval)
        {
            if (verbose_lvl_ >= NORMAL)
                cout << "Calculating ensemble average for " << v_models_.size() << " conformations.\n";

            eval.params_.ref_profile_.initialize(v_q_);

            FLT_T min_ef = params_.exp_factor_.fit() ? FLT_T(0.95) : params_.exp_factor_;
            FLT_T max_ef = params_.exp_factor_.fit() ? FLT_T(1.05) : params_.exp_factor_;
            FLT_T min_ww = params_.water_weight_.fit() ? FLT_T(-2.0) : params_.water_weight_;
            FLT_T max_ww = params_.water_weight_.fit() ? FLT_T(4.0) : params_.water_weight_;

            fitted_params<FLT_T> fit_min = fit_ensemble(eval, min_ef, min_ww);
            fitted_params<FLT_T> fit_max = fit_ensemble(eval, max_ef, max_ww);

            fitted_params<FLT_T> best_params = fit_ensemble_binary_convex(eval, fit_min, fit_max);

            cout << eval.calc_count << " calculations." << endl;

            return best_params;
        }

        /// Fit the ensemble parameters by FULL depth-first binary search of the solution space.
        /// This will find the true optimum, but is extremely slow.
        /// \param[in] eval SAXS evaluator object
        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble_binary_full(CALC_T & eval, const fitted_params<FLT_T> & fit_min, const fitted_params<FLT_T> & fit_max)
        {
            auto ef_delta = std::fabs(fit_min.exp_factor_ - fit_max.exp_factor_);
            auto ww_delta = std::fabs(fit_min.water_weight_ - fit_max.water_weight_);
            auto chi_delta = std::fabs(fit_min.chi_ - fit_max.chi_) / fit_max.chi_;
            if (ef_delta < 0.0001 && ww_delta < 0.001 || chi_delta < 0.0001)
            {
                cout << "found min: " << (fit_min < fit_max ? fit_min : fit_max);
                return fit_min < fit_max ? fit_min : fit_max;
            }

            if (ef_delta < 0.0001)
            {
                fitted_params<FLT_T> mid_point = fit_ensemble(eval,  fit_min.exp_factor_, fit_min.water_weight_ + ww_delta / 2);

                fitted_params<FLT_T> left_fit = fit_ensemble_binary_full(eval, fit_min, mid_point);
                fitted_params<FLT_T> right_fit = fit_ensemble_binary_full(eval, mid_point, fit_max);
                return left_fit < right_fit ? left_fit : right_fit;
            }

            if (ww_delta < 0.001)
            {
                fitted_params<FLT_T> mid_point = fit_ensemble(eval, fit_min.exp_factor_ + ef_delta / 2, fit_min.water_weight_);

                fitted_params<FLT_T> left_fit = fit_ensemble_binary_full(eval, fit_min, mid_point);
                fitted_params<FLT_T> right_fit = fit_ensemble_binary_full(eval, mid_point, fit_max);
                return left_fit < right_fit ? left_fit : right_fit;
            }

            fitted_params<FLT_T> mid_point = fit_ensemble(eval, fit_min.exp_factor_ + ef_delta / 2, fit_min.water_weight_ + ww_delta / 2);
            fitted_params<FLT_T> s_mid_point = fit_ensemble(eval, fit_min.exp_factor_ + ef_delta / 2, fit_min.water_weight_);
            fitted_params<FLT_T> w_mid_point = fit_ensemble(eval, fit_min.exp_factor_, fit_min.water_weight_ + ww_delta / 2);
            fitted_params<FLT_T> n_mid_point = fit_ensemble(eval, fit_min.exp_factor_ + ef_delta / 2, fit_max.water_weight_);
            fitted_params<FLT_T> e_mid_point = fit_ensemble(eval, fit_max.exp_factor_, fit_min.water_weight_ + ww_delta / 2);

            fitted_params<FLT_T> sw_fit = fit_ensemble_binary_full(eval, fit_min, mid_point);
            fitted_params<FLT_T> nw_fit = fit_ensemble_binary_full(eval, w_mid_point, n_mid_point);
            fitted_params<FLT_T> ne_fit = fit_ensemble_binary_full(eval, mid_point, fit_max);
            fitted_params<FLT_T> se_fit = fit_ensemble_binary_full(eval, s_mid_point, e_mid_point);

            return std::min({ sw_fit, nw_fit, ne_fit, se_fit });
        }

        /// Fit the ensemble parameters by convex depth-first binary search of the solution space.
        /// Assumes that the solution space is convex to speed things up.
        /// \param[in] eval SAXS evaluator object
        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble_binary_convex(CALC_T & eval, const fitted_params<FLT_T> & fit_min, const fitted_params<FLT_T> & fit_max)
        {
            auto ef_delta = std::fabs(fit_min.exp_factor_ - fit_max.exp_factor_);
            auto ww_delta = std::fabs(fit_min.water_weight_ - fit_max.water_weight_);
            auto chi_delta = std::fabs(fit_min.chi_ - fit_max.chi_) / fit_max.chi_;
            if (ef_delta < 0.0001 && ww_delta < 0.001 || chi_delta < 0.0001)
            {
                cout << "found min: " << (fit_min < fit_max ? fit_min : fit_max);
                return fit_min < fit_max ? fit_min : fit_max;
            }

            if (ef_delta < 0.0001)
            {
                fitted_params<FLT_T> mid_point = fit_ensemble(eval, fit_min.exp_factor_, fit_min.water_weight_ + ww_delta / 2);

                if (fit_min > mid_point && mid_point > fit_max)
                return fit_ensemble_binary_convex(eval, mid_point, fit_max);

                if (fit_min < mid_point && mid_point < fit_max)
                return fit_ensemble_binary_convex(eval, fit_min, mid_point);

                fitted_params<FLT_T> left_fit = fit_ensemble_binary_convex(eval, fit_min, mid_point);
                fitted_params<FLT_T> right_fit = fit_ensemble_binary_convex(eval, mid_point, fit_max);
                return left_fit < right_fit ? left_fit : right_fit;
            }

            if (ww_delta < 0.001)
            {
                fitted_params<FLT_T> mid_point = fit_ensemble(eval, fit_min.exp_factor_ + ef_delta / 2, fit_min.water_weight_);

                if (fit_min > mid_point && mid_point > fit_max)
                return fit_ensemble_binary_convex(eval, mid_point, fit_max);

                if (fit_min < mid_point && mid_point < fit_max)
                return fit_ensemble_binary_convex(eval, fit_min, mid_point);

                fitted_params<FLT_T> left_fit = fit_ensemble_binary_convex(eval, fit_min, mid_point);
                fitted_params<FLT_T> right_fit = fit_ensemble_binary_convex(eval, mid_point, fit_max);
                return left_fit < right_fit ? left_fit : right_fit;
            }

            fitted_params<FLT_T> mid_point = fit_ensemble(eval, fit_min.exp_factor_ + ef_delta / 2, fit_min.water_weight_ + ww_delta / 2);
            fitted_params<FLT_T> s_mid_point = fit_ensemble(eval, fit_min.exp_factor_ + ef_delta / 2, fit_min.water_weight_);
            fitted_params<FLT_T> w_mid_point = fit_ensemble(eval, fit_min.exp_factor_, fit_min.water_weight_ + ww_delta / 2);
            fitted_params<FLT_T> n_mid_point = fit_ensemble(eval, fit_min.exp_factor_ + ef_delta / 2, fit_max.water_weight_);
            fitted_params<FLT_T> e_mid_point = fit_ensemble(eval, fit_max.exp_factor_, fit_min.water_weight_ + ww_delta / 2);

            fitted_params<FLT_T> min_point = std::min({ fit_min, fit_max, mid_point, s_mid_point, w_mid_point, n_mid_point, e_mid_point });

            fitted_params<FLT_T> bad_fit;

            fitted_params<FLT_T> sw_fit = contains({ fit_min, w_mid_point, mid_point, s_mid_point }, min_point) ? fit_ensemble_binary_convex(eval, fit_min, mid_point) : bad_fit;
            fitted_params<FLT_T> nw_fit = contains({ w_mid_point, mid_point, n_mid_point }, min_point) ? fit_ensemble_binary_convex(eval, w_mid_point, n_mid_point) : bad_fit;
            fitted_params<FLT_T> ne_fit = contains({ fit_max, n_mid_point, mid_point, e_mid_point }, min_point) ? fit_ensemble_binary_convex(eval, mid_point, fit_max) : bad_fit;
            fitted_params<FLT_T> se_fit = contains({ s_mid_point, mid_point, e_mid_point }, min_point) ? fit_ensemble_binary_convex(eval, s_mid_point, e_mid_point) : bad_fit;

            return std::min({ sw_fit, nw_fit, ne_fit, se_fit });
        }

        /// Evaluate the ensemble with the specified params and fit the scale.
        /// \param[in] eval SAXS evaluator object
        /// \param[in] exp_factor Expansion factor
        /// \param[in] water_weight Weight of the water layer
        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble(CALC_T & eval, FLT_T exp_factor, FLT_T water_weight)
        {
            eval.params_.scale_ = params_.scale_;   // reset scale param
            eval.params_.exp_factor_.fix(exp_factor);
            eval.params_.water_weight_.fix(water_weight);

            fitted_params<FLT_T> fit_params;
            fit_params.exp_factor_ = exp_factor;
            fit_params.water_weight_ = water_weight;
            fit_params.intensity_ = avg_ensemble(eval);

            // fit the scale parameter, if needed, and scale the intensity
            fit_params.scale_ = params_.scale_.fit() ? params_.ref_profile_.optimize_scale_for(fit_params.intensity_) : params_.scale_;
            fit_params.intensity_ *= fit_params.scale_;

            fit_params.chi_ = params_.ref_profile_.chi_square_chi(fit_params.intensity_);

            return fit_params;
        }

        /// Fit the ensemble parameters by a grid search of the (assumed convex) solution space.
        /// \param[in] eval SAXS evaluator object
/*        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble1(CALC_T & eval)
        {
            if (verbose_lvl_ >= NORMAL)
                cout << "Calculating ensemble average for " << v_models_.size() << " conformations.\n";

            eval.params_.ref_profile_.initialize(v_q_);

            constexpr FLT_T ef_range_min = FLT_T(0.95);
            constexpr FLT_T ef_range_max = FLT_T(1.05);
            constexpr FLT_T ww_range_min = FLT_T(-2.0);
            constexpr FLT_T ww_range_max = FLT_T(4.0);

            constexpr FLT_T ef_epsilon = FLT_T(0.0001);
            constexpr FLT_T ww_epsilon = FLT_T(0.001);
            constexpr FLT_T chi_threshold = FLT_T(0.00001);


            constexpr FLT_T x0_e = (ef_range_max - ef_range_min) / 2;
            constexpr FLT_T x0_w = (ww_range_max - ww_range_min) / 2;
            constexpr FLT_T c = 1;

            constexpr FLT_T b = c * (std::sqrt(3) - 1) / (2 * std::sqrt(2));
            constexpr FLT_T a = b + c / std::sqrt(2);

            FLT_T x1_e = x0_e + a;
            FLT_T x1_w = x0_w + b;
            FLT_T x2_e = x0_e + b;
            FLT_T x2_w = x0_w + a;





            fitted_params<FLT_T> best_params;
            FLT_T prev_round_chi = best_params.chi_;
            bool chi_threshold_reached = false;
            do
            {
                prev_round_chi = best_params.chi_;
                ef_delta = (ef_max - ef_min) / ef_grid_slices;
                ww_delta = (ww_max - ww_min) / ww_grid_slices;

                // sample a sparse square grid from min to max values
                for (unsigned int i = 0; i <= ef_grid_slices; ++i)
                {
                    FLT_T prev_pt_chi = std::numeric_limits<FLT_T>::max();
                    for (unsigned int j = 0; j <= ww_grid_slices; ++j)
                    {
                        fitted_params<FLT_T> new_params = fit_ensemble(eval, ef_min + i * ef_delta, ww_min + j * ww_delta);

                        // if we are getting away from the minimum, skip the rest of the points along this line
                        if (prev_pt_chi < new_params.chi_)
                            break;
                        prev_pt_chi = new_params.chi_;

                        if (new_params < best_params)
                            best_params = std::move(new_params);
                    }
                }

                if (verbose_lvl_ >= DETAILS && best_params.chi_ < prev_round_chi)
                    cout << "found min: " << best_params;

                // update slices and deltas for the next round
                ef_grid_slices = ef_delta > ef_epsilon ? default_grid_slices : fixed_value_slices;
                ww_grid_slices = ww_delta > ww_epsilon ? default_grid_slices : fixed_value_slices;
                ef_delta = clamp_to_zero(ef_delta, ef_epsilon);
                ww_delta = clamp_to_zero(ww_delta, ww_epsilon);

                // search space for next round is +/-delta around the current minimum
                ef_min = std::max(best_params.exp_factor_ - ef_delta, ef_range_min);
                ef_max = std::min(best_params.exp_factor_ + ef_delta, ef_range_max);
                ww_min = std::max(best_params.water_weight_ - ww_delta, ww_range_min);
                ww_max = std::min(best_params.water_weight_ + ww_delta, ww_range_max);

                auto chi_delta = std::fabs(prev_round_chi - best_params.chi_);
                chi_threshold_reached = chi_delta / prev_round_chi < chi_threshold && chi_delta < 0.0001;
            } while ((ef_delta > ef_epsilon || ww_delta > ww_epsilon) && !chi_threshold_reached);

            if (verbose_lvl_ >= DETAILS)
                cout << eval.calc_count << " calculations." << endl;

            return best_params;
        }*/

        /// Fit the ensemble parameters by a grid search of the (assumed convex) solution space.
        /// \param[in] eval SAXS evaluator object
        template <typename CALC_T>
        fitted_params<FLT_T> fit_ensemble(CALC_T & eval)
        {
            if (verbose_lvl_ >= NORMAL)
                cout << "Calculating ensemble average for " << v_models_.size() << " conformations.\n";

            eval.params_.ref_profile_.initialize(v_q_);

            constexpr FLT_T ef_range_min = FLT_T(0.95);
            constexpr FLT_T ef_range_max = FLT_T(1.05);
            constexpr FLT_T ww_range_min = FLT_T(-2.0);
            constexpr FLT_T ww_range_max = FLT_T(4.0);

            constexpr FLT_T ef_epsilon = FLT_T(0.0001);
            constexpr FLT_T ww_epsilon = FLT_T(0.001);
            constexpr FLT_T chi_threshold = FLT_T(0.00001);

            const unsigned int default_grid_slices = 5;
            const unsigned int first_round_grid_slices = 5;
            const unsigned int fixed_value_slices = 1;

            // define the search space and how it will be sliced up
            unsigned int ef_grid_slices = params_.exp_factor_.fit() ? first_round_grid_slices : fixed_value_slices;
            FLT_T ef_min = params_.exp_factor_.fit() ? ef_range_min : params_.exp_factor_;
            FLT_T ef_max = params_.exp_factor_.fit() ? ef_range_max : params_.exp_factor_;
            FLT_T ef_delta;
            unsigned int ww_grid_slices = params_.water_weight_.fit() ? first_round_grid_slices : fixed_value_slices;
            FLT_T ww_min = params_.water_weight_.fit() ? ww_range_min : params_.water_weight_;
            FLT_T ww_max = params_.water_weight_.fit() ? ww_range_max : params_.water_weight_;
            FLT_T ww_delta;

            fitted_params<FLT_T> best_params;
            FLT_T prev_round_chi = best_params.chi_;
            bool chi_threshold_reached = false;
            do
            {
                prev_round_chi = best_params.chi_;
                ef_delta = (ef_max - ef_min) / ef_grid_slices;
                ww_delta = (ww_max - ww_min) / ww_grid_slices;

                // sample a sparse square grid from min to max values
                for (unsigned int i = 0; i <= ef_grid_slices; ++i)
                {
                    FLT_T prev_pt_chi = std::numeric_limits<FLT_T>::max();
                    for (unsigned int j = 0; j <= ww_grid_slices; ++j)
                    {
                        fitted_params<FLT_T> new_params = fit_ensemble(eval, ef_min + i * ef_delta, ww_min + j * ww_delta);

                        // if we are getting away from the minimum, skip the rest of the points along this line
                        if (prev_pt_chi < new_params.chi_)
                            break;
                        prev_pt_chi = new_params.chi_;

                        if (new_params < best_params)
                            best_params = std::move(new_params);
                    }
                }

                if (verbose_lvl_ >= DETAILS && best_params.chi_ < prev_round_chi)
                    cout << "found min: " << best_params;

                // update slices and deltas for the next round
                ef_grid_slices = ef_delta > ef_epsilon ? default_grid_slices : fixed_value_slices;
                ww_grid_slices = ww_delta > ww_epsilon ? default_grid_slices : fixed_value_slices;
                ef_delta = clamp_to_zero(ef_delta, ef_epsilon);
                ww_delta = clamp_to_zero(ww_delta, ww_epsilon);

                // search space for next round is +/-delta around the current minimum
                ef_min = std::max(best_params.exp_factor_ - ef_delta, ef_range_min);
                ef_max = std::min(best_params.exp_factor_ + ef_delta, ef_range_max);
                ww_min = std::max(best_params.water_weight_ - ww_delta, ww_range_min);
                ww_max = std::min(best_params.water_weight_ + ww_delta, ww_range_max);

                auto chi_delta = std::fabs(prev_round_chi - best_params.chi_);
                chi_threshold_reached = chi_delta / prev_round_chi < chi_threshold && chi_delta < 0.0001;
            } while ((ef_delta > ef_epsilon || ww_delta > ww_epsilon) && !chi_threshold_reached);

            if (verbose_lvl_ >= DETAILS)
                cout << eval.calc_count << " calculations." << endl;

            return best_params;
        }

        /// Compute the ensemble average using the supplied evaluator and current parameters.
        /// \param[in] eval SAXS evaluator object
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