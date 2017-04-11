#ifndef RE_ALGORITHM_HPP
#define RE_ALGORITHM_HPP

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011-2016 Lubo Antonov
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

#include "resaxs.hpp"

namespace resaxs {
namespace algorithm {

    ///////////////////////////////////////////////////////////////////////////////
    //  Interface to all SAXS algorithm classes
    //
    //      It is not ref counted, so it needs to be explicitly managed with delete.
    struct algorithm
    {
    protected:
        enum algorithm_state
        {
            alg_blank = 0,
            alg_initialized,
            alg_program_loaded,
            alg_args_set,
            alg_computing
        } state = alg_blank;

    public:
        virtual ~algorithm() = default;   // objects with this interface may be managed polymorphically

    public:
        // Algorithm state
        bool initialized() const { return state != alg_blank; }
        bool program_loaded() const { return state >= alg_program_loaded; }
        bool args_set() const { return state >= alg_args_set; }
        bool computing() const { return state == alg_computing; }

        //  Initializes the algorithm
        void initialize()
        {
            state = alg_initialized;
        }
    };

    //////////////////////////////////////////////////////////////////////////
    /// Base class for algorithm parameter structures.
    ///
    class params_base
    {
    public:
        virtual ~params_base() = default;   // objects with this interface may be managed polymorphically

        bool is_dirty() const { return dirty_; }
        void clear_dirty() { dirty_ = false; }

    private:
        bool dirty_ = false;

    protected:
        // provides a way to propagate the dirty state to other places (i.e. externally)
        virtual void on_dirty_set() {}

        void set_dirty()
        {
            if (!is_dirty())
            {
                dirty_ = true;
                on_dirty_set();
            }
        }
    };

}
}   // namespace

#endif  // RE_ALGORITHM_HPP
