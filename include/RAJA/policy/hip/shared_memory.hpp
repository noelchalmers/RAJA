/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing HIP shared memory object and policy
 *
 ******************************************************************************
 */

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~//
// Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

#ifndef RAJA_policy_hip_shared_memory_HPP
#define RAJA_policy_hip_shared_memory_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "RAJA/pattern/shared_memory.hpp"

#include "RAJA/policy/hip/policy.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{


namespace internal
{

template <typename T>
RAJA_INLINE RAJA_DEVICE T *hip_get_shmem_ptr(size_t byte_offset = 0)
{
  HIP_DYNAMIC_SHARED( char, my_ptr)
  return reinterpret_cast<T *>(&my_ptr[byte_offset]);
}
}  // namespace internal


/*!
 * Shared Memory object for HIP kernels.
 *
 * Indexing into this is [0, N), regardless of what block or thread you are.
 *
 * The data is always in HIP shared memory, so it's block-local.
 */
template <typename T, size_t NumElem>
struct SharedMemory<hip_shmem, T, NumElem>
    : public internal::SharedMemoryBase {
  using self = SharedMemory<hip_shmem, T, NumElem>;
  using element_t = T;

  static constexpr size_t size = NumElem;
  static constexpr size_t num_bytes = NumElem * sizeof(T);

  int offset;  // offset into dynamic shared memory, in bytes

  RAJA_INLINE
  RAJA_HOST_DEVICE
  size_t shmem_setup_buffer(size_t offset0)
  {
    offset = offset0;
    return num_bytes;
  }

  template <typename OffsetTuple>
  RAJA_INLINE RAJA_HOST_DEVICE void shmem_set_window(OffsetTuple const &)
  {
  }

  template <typename IDX>
  RAJA_INLINE RAJA_DEVICE T &operator[](IDX i) const
  {
    T *ptr = internal::hip_get_shmem_ptr<T>(offset);
    return ptr[i];
  }
};


}  // namespace RAJA

#endif  // RAJA_ENABLE_HIP

#endif
