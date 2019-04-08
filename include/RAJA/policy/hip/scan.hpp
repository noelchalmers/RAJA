/*!
******************************************************************************
*
* \file
*
* \brief   Header file providing RAJA scan declarations.
*
******************************************************************************
*/

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
//
// Produced at the Lawrence Livermore National Laboratory
//
// LLNL-CODE-689114
//
// All rights reserved.
//
// This file is part of RAJA.
//
// For details about use and distribution, please read RAJA/LICENSE.
//
//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//

#ifndef RAJA_scan_hip_HPP
#define RAJA_scan_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <iterator>
#include <type_traits>

#if defined(__HCC__)
#define ROCPRIM_HIP_API 1
#include "rocprim/device/device_scan_hip.hpp"
#elif defined(__CUDACC__)
#include "cub/device/device_scan.cuh"
#include "cub/util_allocator.cuh"
#endif

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"

namespace RAJA
{
namespace impl
{
namespace scan
{

/*!
        \brief explicit inclusive inplace scan given range, function, and
   initial value
*/
template <size_t BLOCK_SIZE, bool Async, typename InputIter, typename Function>
void inclusive_inplace(const ::RAJA::hip_exec<BLOCK_SIZE, Async>&,
                       InputIter begin,
                       InputIter end,
                       Function binary_op)
{
  hipStream_t stream = 0;

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HCC__)
  hipErrchk(::rocprim::inclusive_scan(d_temp_storage,
                                      temp_storage_bytes,
                                      begin,
                                      begin,
                                      len,
                                      binary_op,
                                      stream));
#elif defined(__CUDACC__)
  hipErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage,
                                             temp_storage_bytes,
                                             begin,
                                             begin,
                                             binary_op,
                                             len,
                                             stream));
#endif

  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
#if defined(__HCC__)
  hipErrchk(::rocprim::inclusive_scan(d_temp_storage,
                                      temp_storage_bytes,
                                      begin,
                                      begin,
                                      len,
                                      binary_op,
                                      stream));
#elif defined(__CUDACC__)
  hipErrchk(::cub::DeviceScan::InclusiveScan(d_temp_storage,
                                             temp_storage_bytes,
                                             begin,
                                             begin,
                                             binary_op,
                                             len,
                                             stream));
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  hip::launch(stream);
  if (!Async) hip::synchronize(stream);
}

/*!
        \brief explicit exclusive inplace scan given range, function, and
   initial value
*/
template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename Function,
          typename T>
void exclusive_inplace(const ::RAJA::hip_exec<BLOCK_SIZE, Async>&,
                       InputIter begin,
                       InputIter end,
                       Function binary_op,
                       T init)
{
  hipStream_t stream = 0;

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HCC__)
  hipErrchk(::rocprim::exclusive_scan(d_temp_storage,
                                      temp_storage_bytes,
                                      begin,
                                      begin,
                                      init,
                                      len,
                                      binary_op,
                                      stream));
#elif defined(__CUDACC__)
  hipErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                             temp_storage_bytes,
                                             begin,
                                             begin,
                                             binary_op,
                                             init,
                                             len,
                                             stream));
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
#if defined(__HCC__)
  hipErrchk(::rocprim::exclusive_scan(d_temp_storage,
                                              temp_storage_bytes,
                                              begin,
                                              begin,
                                              init,
                                              len,
                                              binary_op,
                                              stream));
#elif defined(__CUDACC__)
  hipErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                             temp_storage_bytes,
                                             begin,
                                             begin,
                                             binary_op,
                                             init,
                                             len,
                                             stream));
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  hip::launch(stream);
  if (!Async) hip::synchronize(stream);
}

/*!
        \brief explicit inclusive scan given input range, output, function, and
   initial value
*/
template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename OutputIter,
          typename Function>
void inclusive(const ::RAJA::hip_exec<BLOCK_SIZE, Async>&,
               InputIter begin,
               InputIter end,
               OutputIter out,
               Function binary_op)
{
  hipStream_t stream = 0;

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HCC__)
  hipErrchk(::rocprim::inclusive_scan(
      d_temp_storage, temp_storage_bytes, begin, out, len, binary_op, stream));
#elif defined(__CUDACC__)
  hipErrchk(::cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, begin, out, binary_op, len, stream));
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
#if defined(__HCC__)
  hipErrchk(::rocprim::inclusive_scan(
      d_temp_storage, temp_storage_bytes, begin, out, len, binary_op, stream));
#elif defined(__CUDACC__)
  hipErrchk(::cub::DeviceScan::InclusiveScan(
      d_temp_storage, temp_storage_bytes, begin, out, binary_op, len, stream));
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  hip::launch(stream);
  if (!Async) hip::synchronize(stream);
}

/*!
        \brief explicit exclusive scan given input range, output, function, and
   initial value
*/
template <size_t BLOCK_SIZE,
          bool Async,
          typename InputIter,
          typename OutputIter,
          typename Function,
          typename T>
void exclusive(const ::RAJA::hip_exec<BLOCK_SIZE, Async>&,
               InputIter begin,
               InputIter end,
               OutputIter out,
               Function binary_op,
               T init)
{
  hipStream_t stream = 0;

  int len = std::distance(begin, end);
  // Determine temporary device storage requirements
  void* d_temp_storage = nullptr;
  size_t temp_storage_bytes = 0;
#if defined(__HCC__)
  hipErrchk(::rocprim::exclusive_scan(d_temp_storage,
                                      temp_storage_bytes,
                                      begin,
                                      out,
                                      init,
                                      len,
                                      binary_op,
                                      stream));
#elif defined(__CUDACC__)
  hipErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                             temp_storage_bytes,
                                             begin,
                                             out,
                                             binary_op,
                                             init,
                                             len,
                                             stream));
#endif
  // Allocate temporary storage
  d_temp_storage =
      hip::device_mempool_type::getInstance().malloc<unsigned char>(
          temp_storage_bytes);
  // Run
#if defined(__HCC__)
  hipErrchk(::rocprim::exclusive_scan(d_temp_storage,
                                      temp_storage_bytes,
                                      begin,
                                      out,
                                      init,
                                      len,
                                      binary_op,
                                      stream));
#elif defined(__CUDACC__)
  hipErrchk(::cub::DeviceScan::ExclusiveScan(d_temp_storage,
                                             temp_storage_bytes,
                                             begin,
                                             out,
                                             binary_op,
                                             init,
                                             len,
                                             stream));
#endif
  // Free temporary storage
  hip::device_mempool_type::getInstance().free(d_temp_storage);

  hip::launch(stream);
  if (!Async) hip::synchronize(stream);
}

}  // namespace scan

}  // namespace impl

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
