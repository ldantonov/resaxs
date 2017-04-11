#ifndef ACCSAXS_SAXS_ALGORITHM_BASE
#define ACCSAXS_SAXS_ALGORITHM_BASE

///////////////////////////////////////////////////////////////////////////////
//
//              Copyright 2011 Lubo Antonov
//
//    This file is part of ACCSAXS.
//
//    ACCSAXS is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    any later version.
//
//    ACCSAXS is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with ACCSAXS.  If not, see <http://www.gnu.org/licenses/>.
//
///////////////////////////////////////////////////////////////////////////////

#include "accsaxs.hpp"

namespace accsaxs {

///////////////////////////////////////////////////////////////////////////////
//  Base for all SAXS algorithms.
//
//      Provides device initialization and program loading.
//      Instantiated for float and double.
//
template <typename T, typename T4>
class saxs_algorithm_base: public saxs_algorithm<T, T4>
{
public:
    //
    //  Initializes the algorithm with a list of devices to run on
    //
    virtual void initialize(const std::vector<dev_id> & dev_ids);

    virtual ~saxs_algorithm_base() {};

protected:
    // OpenCL objects - valid after initialize()
    cl::Context context_;
    cl::CommandQueue queue_;
    std::vector<cl::Device> devices_;

    //
    //  Loads the specified files as a program and builds it.
    //      file_names          - a list of file names to load
    //      file_name           - a single file name to load
    //      result              - a built OpenCL program object on successful completion
    //      options             - command-line options to pass to the OpenCL compiler
    //
    void load_program(const std::vector<std::string> & file_names, cl::Program & result, const std::string & options = NULL);
    void load_program(const std::string & file_name, cl::Program & result, const std::string & options = NULL);

    //
    //  Loads the specified files as a program object
    //      file_names          - a list of file names to load
    //      result              - a OpenCL program object on successful completion
    //
    void get_program(const std::vector<std::string> & file_names, cl::Program & result);

    //
    //  Builds a program from a source string.
    //      src                 - program source
    //      result              - a built OpenCL program object on successful completion
    //      options             - command-line options to pass to the OpenCL compiler
    //      
    void build_program(const std::string & src, cl::Program & result, const std::string & options = NULL);

    //
    //  Converts the contents of a file into a string.
    //
    static std::string convert_to_string(const std::string & filename);

public:
    
    template <typename S>
    class Buffer
    {
    public:
        cl::Buffer cl_buffer_;
        ::size_t size_;
        
        Buffer() : size_(0) {}
        Buffer(const Buffer & other) : cl_buffer_(other.cl_buffer_), size_(other.size_) {}
        Buffer(const cl::Buffer & buffer, ::size_t size) : cl_buffer_(buffer), size_(size) {}
        
        Buffer & operator = (const Buffer & rhs)
        {
            if (this != &rhs)
            {
                cl_buffer_ = rhs.cl_buffer_;
                size_ = rhs.size_;
            }
            return *this;
        }
        
        operator const cl::Buffer &() const { return cl_buffer_; }
        
        ::size_t size_in_bytes() const { return size_in_bytes(size_); }
        static ::size_t size_in_bytes(::size_t size) { return size * sizeof(S); }
        
        /// Initializes the buffer.
        /// \param context The owning context
        /// \param flags Memory buffer flags
        /// \param size Buffer size
        /// \param host_ptr Host memory pointer to use with this buffer
        void init(const cl::Context& context, cl_mem_flags flags, ::size_t size, S * host_ptr = NULL)
        {
            size_ = size;
            cl_buffer_ = cl::Buffer(context, flags, size_in_bytes(), host_ptr);
        }
        
        /// Initializes the buffer from a vector by copying the data.
        /// \param context The owning context
        /// \param flags Memory buffer flags; CL_MEM_COPY_HOST_PTR will be added automatically
        /// \param host_vec Vector to use as source to initialize the buffer
        void init(const cl::Context& context, cl_mem_flags flags, const std::vector<S> & host_vec)
        {
            flags |= CL_MEM_COPY_HOST_PTR;
            init(context, flags, host_vec.size(), const_cast<S*>(&host_vec.front()));
            //size_ = host_vec.size();
            //cl_buffer_ = cl::Buffer(context, flags, size_ * sizeof(S), const_cast<S*>(&host_vec.front()), err);
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
            return write_from(&vec[offset], offset, size, queue, blocking, events, event);
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
            return read_to(&vec[offset], offset, size, queue, blocking, events, event);
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
    };
    
protected:
    // OpenCL memory buffers
    Buffer<T> b_qq_;
    Buffer<T> b_factors_;
    Buffer<T4> b_bodies_;
    Buffer<T> b_Iqq_;
};

}   // namespace

namespace cl
{

#if 0
// This SHOULD be the way to specialize KernelArgumentHandler for the accsaxs Buffer class, but
// it does not compile. So instead we have the fully specialized versions below. The only problem is
// that the type combinations not covered here will not work properly.
template <typename T, typename T4, typename S>
struct detail::KernelArgumentHandler<accsaxs::saxs_algorithm_base<T, T4>::Buffer<S> >
{
    static ::size_t size(const accsaxs::saxs_algorithm_base<T, T4>::Buffer<S> &) { return sizeof(cl::Buffer); }
    static cl::Buffer * ptr(accsaxs::saxs_algorithm_base<T, T4>::Buffer<S> & value) { return &value.cl_buffer_; }
};
#endif // Disabled code
    
template <>
inline cl_int Kernel::setArg<accsaxs::saxs_algorithm_base<float, cl_float4>::Buffer<float> >(
    cl_uint index, accsaxs::saxs_algorithm_base<float, cl_float4>::Buffer<float> value)
{
    return setArg(index, value.cl_buffer_);
}

template <>
inline cl_int Kernel::setArg<accsaxs::saxs_algorithm_base<float, cl_float4>::Buffer<cl_float4> >(
    cl_uint index, accsaxs::saxs_algorithm_base<float, cl_float4>::Buffer<cl_float4> value)
{
    return setArg(index, value.cl_buffer_);
}

template <>
inline cl_int Kernel::setArg<accsaxs::saxs_algorithm_base<double, cl_double4>::Buffer<double> >(
    cl_uint index, accsaxs::saxs_algorithm_base<double, cl_double4>::Buffer<double> value)
{
    return setArg(index, value.cl_buffer_);
}

template <>
inline cl_int Kernel::setArg<accsaxs::saxs_algorithm_base<double, cl_double4>::Buffer<cl_double4> >(
    cl_uint index, accsaxs::saxs_algorithm_base<double, cl_double4>::Buffer<cl_double4> value)
{
    return setArg(index, value.cl_buffer_);
}

}
#endif  // ACCSAXS_SAXS_ALGORITHM_BASE