#ifndef BB_GRID_HPP
#define BB_GRID_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2015 Lubo Antonov
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

#include <limits>
#include <algorithm>

namespace resaxs
{

    /// Calculates squared Euclidean distance between the two data points
    template <typename T> inline
        auto distance2(const T & pt1, const T & pt2) -> decltype(pt1.s[0] + pt2.s[0])
    {
        auto x_diff = pt1.s[0] - pt2.s[0];
        auto y_diff = pt1.s[1] - pt2.s[1];
        auto z_diff = pt1.s[2] - pt2.s[2];
        return x_diff * x_diff + y_diff * y_diff + z_diff * z_diff;
    }


    template <typename FLT_T>
    class pos_3d
    {
        typedef real_type<FLT_T> num_type;
        typedef typename num_type::real real;

    public:
        typename num_type::real3 data_;

        pos_3d() : pos_3d(0) {}
        pos_3d(real val) : data_({ val, val, val }) {}
        pos_3d(const typename num_type::real4 &r4_value) : data_({ r4_value.s[0], r4_value.s[1], r4_value.s[2] }) {}

        real & operator[](unsigned int dim)
        {
            return data_[dim];
        }

        const real & operator[](unsigned int dim) const
        {
            return data_[dim];
        }

        pos_3d & operator+=(real delta)
        {
            data_ += delta;
            return *this;
        }

        pos_3d & operator-=(real delta)
        {
            return operator+=(-delta);
        }

        template <typename Func>
        void apply(Func func)
        {
            data_.apply(func);
        }
    };

    template <typename FLT_T>
    class bounding_box_3d
    {
    public:
        typedef real_type<FLT_T> num_type;
        typedef typename num_type::real real;
        typedef typename num_type::real4 real4;

        pos_3d<FLT_T> lc_;
        pos_3d<FLT_T> uc_;

        bounding_box_3d()
        {
            lc_ = pos_3d<FLT_T> { std::numeric_limits<real>::max() };
            uc_ = pos_3d<FLT_T> { std::numeric_limits<real>::lowest() };
        }

        /// Create a bounding box around the given points
        bounding_box_3d(const std::vector<real4> &points) : bounding_box_3d()
        {
            for (const auto &point : points)
            {
                operator +=(point);
            }
        }

        /// Create a bounding box for one point
        bounding_box_3d(const real4 &pt) : lc_(pt), uc_(pt) {}

        /// Adjust the bounding box to include the given point
        bounding_box_3d & operator +=(const real4 &point)
        {
            lc_.apply([&point](const real v1, unsigned int dim) { return std::min(v1, point.s[dim]); });
            uc_.apply([&point](const real v1, unsigned int dim) { return std::max(v1, point.s[dim]); });
            return *this;
        }

        // Grow the bounding box by a margin equally on all sides
        bounding_box_3d & operator +=(real delta)
        {
            lc_ -= delta;
            uc_ += delta;
            return *this;
        }
    };


    template <typename FLT_T>
    class voxel_grid
    {
        typedef real_type<FLT_T> num_type;
        typedef typename num_type::real real;
        typedef typename num_type::real4 real4;

        void get_ns1(real u, real l, real ds, int &dd) const
        {
            real box_side = u - l;
            real d = box_side / ds;
            real cd = std::ceil(d);
            dd = std::max(1, static_cast<int>(cd));
        }

        template <typename NS>
        const num3<int> get_ns(const NS &ds, const bounding_box_3d<real> &bbox) const
        {
            num3<int> dd;
            get_ns1(bbox.uc_[0], bbox.lc_[0], ds[0], dd[0]);
            get_ns1(bbox.uc_[1], bbox.lc_[1], ds[1], dd[1]);
            get_ns1(bbox.uc_[2], bbox.lc_[2], ds[2], dd[2]);
            return dd;
        }

    public:
        voxel_grid(real side, const bounding_box_3d<real> &bbox) : bbox_(bbox), n_voxels_(get_ns(std::vector<real>(3, side), bbox)), origin_(bbox.lc_),
            side_(side), rside_(1 / side),
            data_(n_voxels_[0] * n_voxels_[1] * n_voxels_[2])
        {
            origin_ = bbox_.lc_;
        }

        num3<int> get_nearest_bounded_index(const pos_3d<real> &pt) const
        {
            num3<int> index = get_virtual_index(pt);

            // restrict the index to the bounding box
            index.apply([](const int v, unsigned int) { return std::max(v, 0); });
            index.apply([this](const int v, unsigned int dim) { return std::min(v, n_voxels_[dim] - 1); });
            return index;
        }

        num3<int> get_virtual_index(const pos_3d<real> &pt) const
        {
            // calculate the index: (pt - origin) / side
            num3<real> d = pt.data_.template generate<real>([this](const real v, unsigned int dim) { return (v - origin_[dim]) * rside_; });
            return d.template generate<int>([](const real v, unsigned int) { return static_cast<int>(std::floor(v)); });
        }

        std::vector<unsigned int> & operator [](const num3<int> &index)
        {
            return data_[index.z * n_voxels_.x * n_voxels_.y + index.y * n_voxels_.x + index.x];
        }

        class iterator
        {
        public:
            iterator(voxel_grid &grid, const num3<int> &begin_index, const num3<int> &end_index) :
                grid_(grid), cur_index_(begin_index), begin_index_(begin_index), end_index_(end_index) {}

            iterator & operator ++()     // preincrement
            {
                ++cur_index_.x;
                if (cur_index_.x == end_index_.x)
                {
                    cur_index_.x = begin_index_.x;
                    ++cur_index_.y;
                    if (cur_index_.y == end_index_.y)
                    {
                        cur_index_.y = begin_index_.y;
                        ++cur_index_.z;
                        if (cur_index_.z >= end_index_.z)
                        {
                            cur_index_ = end_index_;
                        }
                    }
                }
                return *this;
            }

            const std::vector<unsigned int> & operator*() const
            {
                return grid_[cur_index_];
            }

            std::vector<unsigned int> & operator*()
            {
                return grid_[cur_index_];
            }

            iterator operator ++(int)
            {
                iterator tmp = *this;
                ++*this;
                return tmp;
            }

            bool operator==(const iterator &right) const
            {
                return cur_index_ == right.cur_index_;
            }

            bool operator!=(const iterator &right) const
            {
                return !operator==(right);
            }

        private:
            voxel_grid & grid_;
            num3<int> cur_index_;
            const num3<int> begin_index_;
            const num3<int> end_index_;
        };

        iterator begin(const num3<int> &lc, const num3<int> &uc)
        {
            const num3<int> &end_index = uc + 1;
            return{ *this, lc, end_index };
        }

        iterator end(const num3<int> &uc)
        {
            const num3<int> &end_index = uc + 1;
            return { *this, end_index, end_index };
        }

    private:
        const bounding_box_3d<real> &bbox_;
        num3<int> n_voxels_;
        pos_3d<real> origin_;

        real side_;
        real rside_;

        std::vector<std::vector<unsigned int>> data_;
    };

 } // namespace

#endif // BB_GRID_HPP
