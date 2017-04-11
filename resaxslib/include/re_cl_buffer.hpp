#ifndef RE_CL_BUFFER_HPP
#define RE_CL_BUFFER_HPP

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

namespace resaxs {

    ///////////////////////////////////////////////////////////////////////////////
    //  Encapsulates cl::Buffer objects.
    template <typename S>
    class Buffer
    {
    public:
        cl::Buffer cl_buffer_;
        ::size_t size_ = 0;
        cl_mem_flags flags_ = 0;


        Buffer() = default;
        Buffer(const Buffer & other) = default;
        Buffer(const cl::Buffer & buffer, ::size_t size) : cl_buffer_(buffer), size_(size) {}

        Buffer & operator=(const Buffer & rhs) = default;

        operator const cl::Buffer &() const { return cl_buffer_; }

        operator bool() const { return cl_buffer_() != NULL; }

        ::size_t size_in_bytes() const { return size_in_bytes(size_); }
        static ::size_t size_in_bytes(::size_t size) { return size * sizeof(S); }

        /// Release buffer resources
        void clear()
        {
            cl_buffer_ = NULL;
            size_ = 0;
            flags_ = 0;
        }

        /// Initializes the buffer.
        /// \param context The owning context
        /// \param flags Memory buffer flags
        /// \param size Buffer size
        /// \param host_ptr Host memory pointer to use with this buffer
        void init(const cl::Context& context, cl_mem_flags flags, ::size_t size, S * host_ptr = NULL)
        {
            size_ = size;
            cl_buffer_ = cl::Buffer(context, flags, size_in_bytes(), host_ptr);
            flags_ = flags;
        }

        /// Initializes the buffer from a vector by copying the data.
        /// \param context The owning context
        /// \param flags Memory buffer flags; CL_MEM_COPY_HOST_PTR will be added automatically
        /// \param host_vec Vector to use as source to initialize the buffer
        void init(const cl::Context& context, cl_mem_flags flags, const std::vector<S> & host_vec)
        {
            flags |= CL_MEM_COPY_HOST_PTR;
            init(context, flags, host_vec.size(), const_cast<S*>(&host_vec.front()));
         }

        /// Initializes the buffer with a vector.
        /// \param context The owning context
        /// \param flags Memory buffer flags
        /// \param host_vec Vector to use
        void init(const cl::Context& context, cl_mem_flags flags, std::vector<S> & host_vec)
        {
            init(context, flags, host_vec.size(), &host_vec.front());
        }

        /// Enqueues a write to the buffer from a memory pointer.
        /// \param ptr Source memory to read from
        /// \param offset Offset into the buffer to start writing to
        /// \param size Number of elements to write
        /// \param queue The command queue to use
        /// \param blocking Optional - whether the write is blocking or asynchronous
        /// \param events Optional list of events that need to complete before this command gets executed
        /// \param event Optional event object that will get initialized to an event that indicates the completion of this command
        cl_int write_from(const S * ptr, ::size_t offset, ::size_t size,
            const cl::CommandQueue & queue, cl_bool blocking = CL_TRUE,
            const std::vector<cl::Event> * events = NULL, cl::Event * event = NULL) const
        {
            return queue.enqueueWriteBuffer(*this, blocking, size_in_bytes(offset), size_in_bytes(size), ptr, events, event);
        }

        /// Enqueues a write to the buffer from a vector. The same offset is used for the source and the destination.
        /// \param vec Source vector to read from
        /// \param offset Offset into the vector to start reading from, as well as into the buffer to start writing to
        /// \param size Number of elements to write
        /// \param queue The command queue to use
        /// \param blocking Optional - whether the write is blocking or asynchronous
        /// \param events Optional list of events that need to complete before this command gets executed
        /// \param event Optional event object that will get initialized to an event that indicates the completion of this command
        cl_int write_from(const std::vector<S> & vec, ::size_t offset, ::size_t size,
            const cl::CommandQueue & queue, cl_bool blocking = CL_TRUE,
            const std::vector<cl::Event> * events = NULL, cl::Event * event = NULL) const
        {
            return write_from(&vec[offset], offset, std::min(size_, std::min(size, vec.size())), queue, blocking, events, event);
        }

        /// Enqueues a write to the buffer from a vector. The entire vector is transfered.
        /// \param vec Source vector to read from
        /// \param queue The command queue to use
        /// \param blocking Optional - whether the write is blocking or asynchronous
        /// \param events Optional list of events that need to complete before this command gets executed
        /// \param event Optional event object that will get initialized to an event that indicates the completion of this command
        cl_int write_from(const std::vector<S> & vec,
            const cl::CommandQueue & queue, cl_bool blocking = CL_TRUE,
            const std::vector<cl::Event> * events = NULL, cl::Event * event = NULL) const
        {
            return write_from(vec, 0, vec.size(), queue, blocking, events, event);
        }

        /// Enqueues a buffer copy command to another buffer.
        /// \param dst The destination buffer
        /// \param src_offset Offset into the source buffer (this)
        /// \param dst_offset Offset into the destination buffer
        /// \param size Number of elements to copy
        /// \param queue The command queue to use
        /// \param events Optional list of events that need to complete before this command gets executed
        /// \param event Optional event object that will get initialized to an event that indicates the completion of this command
        cl_int copy_to(const Buffer & dst, ::size_t src_offset, ::size_t dst_offset, ::size_t size,
            const cl::CommandQueue & queue,
            const std::vector<cl::Event> * events = NULL, cl::Event * event = NULL) const
        {
            return queue.enqueueCopyBuffer(*this, dst, size_in_bytes(src_offset), dst.size_in_bytes(dst_offset), size_in_bytes(size), events, event);
        }

        /// Enqueues a buffer copy command to another buffer.
        /// \param dst The destination buffer
        /// \param offset Offset into the source and destination buffers
        /// \param size Number of elements to copy
        /// \param queue The command queue to use
        /// \param events Optional list of events that need to complete before this command gets executed
        /// \param event Optional event object that will get initialized to an event that indicates the completion of this command
        cl_int copy_to(const Buffer & dst, ::size_t offset, ::size_t size,
            const cl::CommandQueue & queue,
            const std::vector<cl::Event> * events = NULL, cl::Event * event = NULL) const
        {
            return copy_to(dst, offset, offset, size, queue, events, event);
        }

        /// Enqueues a buffer copy command to another buffer. Copies the entire buffer.
        /// \param dst The destination buffer
        /// \param queue The command queue to use
        /// \param events Optional list of events that need to complete before this command gets executed
        /// \param event Optional event object that will get initialized to an event that indicates the completion of this command
        cl_int copy_to(const Buffer & dst,
            const cl::CommandQueue & queue,
            const std::vector<cl::Event> * events = NULL, cl::Event * event = NULL) const
        {
            return copy_to(dst, 0, size_, queue, events, event);
        }

        /// Enqueues a read from the buffer to a memory pointer.
        /// \param ptr Destination memory to read to
        /// \param offset Offset into the buffer to start reading from
        /// \param size Number of elements to read
        /// \param queue The command queue to use
        /// \param blocking Optional - whether the write is blocking or asynchronous
        /// \param events Optional list of events that need to complete before this command gets executed
        /// \param event Optional event object that will get initialized to an event that indicates the completion of this command
        cl_int read_to(S * ptr, ::size_t offset, ::size_t size,
            const cl::CommandQueue & queue, cl_bool blocking = CL_TRUE,
            const std::vector<cl::Event> * events = NULL, cl::Event * event = NULL) const
        {
            return queue.enqueueReadBuffer(*this, blocking, size_in_bytes(offset), size_in_bytes(size), ptr, events, event);
        }

        /// Enqueues a read from the buffer to a vector. The same offset is used for the source and the destination.
        /// \param vec Destination vector to read to
        /// \param offset Offset into the buffer to start reading from, as well as into the vector to start reading to
        /// \param size Number of elements to read
        /// \param queue The command queue to use
        /// \param blocking Optional - whether the write is blocking or asynchronous
        /// \param events Optional list of events that need to complete before this command gets executed
        /// \param event Optional event object that will get initialized to an event that indicates the completion of this command
        cl_int read_to(std::vector<S> & vec, ::size_t offset, ::size_t size,
            const cl::CommandQueue & queue, cl_bool blocking = CL_TRUE,
            const std::vector<cl::Event> * events = NULL, cl::Event * event = NULL) const
        {
            return read_to(&vec[offset], offset, std::min(size_, std::min(size, vec.size())), queue, blocking, events, event);
        }

        /// Enqueues a read from the buffer to a vector. The entire buffer is transfered.
        /// \param vec Destination vector to read to
        /// \param queue The command queue to use
        /// \param blocking Optional - whether the write is blocking or asynchronous
        /// \param events Optional list of events that need to complete before this command gets executed
        /// \param event Optional event object that will get initialized to an event that indicates the completion of this command
        cl_int read_to(std::vector<S> & vec,
            const cl::CommandQueue & queue, cl_bool blocking = CL_TRUE,
            const std::vector<cl::Event> * events = NULL, cl::Event * event = NULL) const
        {
            return read_to(vec, 0, size_, queue, blocking, events, event);
        }

        /// Moves the buffer to a new context.
        /// \param from_queue Command queue to use for reading the old buffer
        /// \param context The new context
        void move(const cl::CommandQueue &from_queue, const cl::Context &context)
        {
            std::vector<S> vec(size_);
            read_to(vec, from_queue);
            init(context, flags_ | CL_MEM_COPY_HOST_PTR, vec);
        }

        /// Initializes the buffer, or switches it to a new context if it is not empty.
        /// \param context The owning context
        /// \param flags Memory buffer flags
        /// \param size Buffer size
        /// \param host_ptr Host memory pointer to use with this buffer
        void init_or_switch_context(const cl::CommandQueue &from_queue, const cl::Context& context, cl_mem_flags flags, ::size_t size, S * host_ptr = NULL)
        {
            if (*this)
                move(from_queue, context);
            else
                init(context, flags, size, host_ptr);
        }

        /// Initializes the buffer from a vector, or switches it to a new context if it is not empty.
        /// \param context The owning context
        /// \param flags Memory buffer flags; CL_MEM_COPY_HOST_PTR will be added automatically
        /// \param host_vec Vector to use as source to initialize the buffer
        void init_or_switch_context(const cl::CommandQueue &from_queue, const cl::Context& context, cl_mem_flags flags, const std::vector<S> & host_vec)
        {
            if (*this)
                move(from_queue, context);
            else
                init(context, flags, host_vec);
        }

        /// Initializes the buffer with a vector, or switches it to a new context if it is not empty.
        /// \param context The owning context
        /// \param flags Memory buffer flags
        /// \param host_vec Vector to use
        void init_or_switch_context(const cl::CommandQueue &from_queue, const cl::Context& context, cl_mem_flags flags, std::vector<S> & host_vec)
        {
            if (*this)
                move(from_queue, context);
            else
                init(context, flags, host_vec);
        }

    };

}   // namespace

namespace cl
{

#if 1
    namespace detail {
        // This is the way to specialize KernelArgumentHandler for the resaxs Buffer class, but
        // only if it is a top-level class. Otherwise we have the fully specialized versions below; the problem with them is
        // that the types not covered here will not work properly.
        template <typename S>
        struct KernelArgumentHandler<resaxs::Buffer<S> >
        {
            static ::size_t size(const typename resaxs::Buffer<S> &) { return sizeof(cl::Buffer); }
            static const cl::Buffer * ptr(const typename resaxs::Buffer<S> & value) { return &value.cl_buffer_; }
        };
    }   // namespace detail
#endif

#if 0
        // float and double Buffers
    template <>
    inline cl_int Kernel::setArg<resaxs::Buffer<float> >(
        cl_uint index, resaxs::Buffer<float> value)
    {
        return setArg(index, value.cl_buffer_);
    }

    template <>
    inline cl_int Kernel::setArg<resaxs::Buffer<cl_float4> >(
        cl_uint index, resaxs::Buffer<cl_float4> value)
    {
        return setArg(index, value.cl_buffer_);
    }

    template <>
    inline cl_int Kernel::setArg<resaxs::Buffer<double> >(
        cl_uint index, resaxs::Buffer<double> value)
    {
        return setArg(index, value.cl_buffer_);
    }

    template <>
    inline cl_int Kernel::setArg<resaxs::Buffer<cl_double4> >(
        cl_uint index, resaxs::Buffer<cl_double4> value)
    {
        return setArg(index, value.cl_buffer_);
    }

    // int/unsigned int buffers
    template <>
    inline cl_int Kernel::setArg<resaxs::Buffer<int> >(
        cl_uint index, resaxs::Buffer<int> value)
    {
        return setArg(index, value.cl_buffer_);
    }

    template <>
    inline cl_int Kernel::setArg<resaxs::Buffer<unsigned int> >(
        cl_uint index, resaxs::Buffer<unsigned int> value)
    {
        return setArg(index, value.cl_buffer_);
    }

#endif // Disabled code

}

#endif RE_CL_BUFFER_HPP
