/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA HIP policy definitions.
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

#ifndef RAJA_policy_hip_HPP
#define RAJA_policy_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "RAJA/pattern/reduce.hpp"

#include "RAJA/policy/PolicyBase.hpp"

#include "RAJA/util/Operators.hpp"
#include "RAJA/util/types.hpp"

namespace RAJA
{

#if defined(RAJA_ENABLE_CLANG_HIP)
using hip_dim_t = uint3;
#else
using hip_dim_t = dim3;
#endif


///
/////////////////////////////////////////////////////////////////////
///
/// Generalizations of HIP dim3 x, y and z used to describe
/// sizes and indices for threads and blocks.
///
/////////////////////////////////////////////////////////////////////
///

struct Dim3x {
  __host__ __device__ inline unsigned int &operator()(hip_dim_t &dim)
  {
    return dim.x;
  }

  __host__ __device__ inline unsigned int operator()(hip_dim_t const &dim)
  {
    return dim.x;
  }
};
///
struct Dim3y {
  __host__ __device__ inline unsigned int &operator()(hip_dim_t &dim)
  {
    return dim.y;
  }

  __host__ __device__ inline unsigned int operator()(hip_dim_t const &dim)
  {
    return dim.y;
  }
};
///
struct Dim3z {
  __host__ __device__ inline unsigned int &operator()(hip_dim_t &dim)
  {
    return dim.z;
  }

  __host__ __device__ inline unsigned int operator()(hip_dim_t const &dim)
  {
    return dim.z;
  }
};

//
/////////////////////////////////////////////////////////////////////
//
// Execution policies
//
/////////////////////////////////////////////////////////////////////
//

///
/// Segment execution policies
///

namespace detail
{
template <bool Async>
struct get_launch {
  static constexpr RAJA::Launch value = RAJA::Launch::async;
};

template <>
struct get_launch<false> {
  static constexpr RAJA::Launch value = RAJA::Launch::sync;
};
}  // end namespace detail

namespace policy
{
namespace hip
{

template <size_t BLOCK_SIZE, bool Async = false>
struct hip_exec : public RAJA::make_policy_pattern_launch_platform_t<
                      RAJA::Policy::hip,
                      RAJA::Pattern::forall,
                      detail::get_launch<Async>::value,
                      RAJA::Platform::hip> {
};


/*
 * Policy for on-device loop with a __syncthreads() after each iteration
 */
struct hip_seq_syncthreads_exec
    : public RAJA::make_policy_pattern_launch_platform_t<RAJA::Policy::hip,
                                                         RAJA::Pattern::forall,
                                                         RAJA::Launch::sync,
                                                         RAJA::Platform::hip> {
};

//
// NOTE: There is no Index set segment iteration policy for HIP
//

///
///////////////////////////////////////////////////////////////////////
///
/// Reduction reduction policies
///
///////////////////////////////////////////////////////////////////////
///

template <bool maybe_atomic>
struct hip_reduce_base
    : public RAJA::
          make_policy_pattern_launch_platform_t<RAJA::Policy::hip,
                                                RAJA::Pattern::reduce,
                                                detail::get_launch<false>::value,
                                                RAJA::Platform::hip> {
};

using hip_reduce = hip_reduce_base<false>;

using hip_reduce_atomic = hip_reduce_base<true>;


template <typename POL>
struct HipPolicy : public RAJA::make_policy_pattern_launch_platform_t<
                        RAJA::Policy::hip,
                        RAJA::Pattern::forall,
                        RAJA::Launch::undefined,
                        RAJA::Platform::hip> {

  using hip_exec_policy = POL;
};

//
// Operations in the included files are parametrized using the following
// values for HIP warp size and max block size.
//
constexpr const RAJA::Index_type WARP_SIZE = 32;
constexpr const RAJA::Index_type MAX_BLOCK_SIZE = 1024;
constexpr const RAJA::Index_type MAX_WARPS = MAX_BLOCK_SIZE / WARP_SIZE;
static_assert(WARP_SIZE >= MAX_WARPS,
              "RAJA Assumption Broken: WARP_SIZE < MAX_WARPS");
static_assert(MAX_BLOCK_SIZE % WARP_SIZE == 0,
              "RAJA Assumption Broken: MAX_BLOCK_SIZE not "
              "a multiple of WARP_SIZE");

struct hip_synchronize : make_policy_pattern_launch_t<Policy::hip,
                                                       Pattern::synchronize,
                                                       Launch::sync> {
};

}  // end namespace hip
}  // end namespace policy

using policy::hip::hip_exec;

template <size_t BLOCK_SIZE>
using hip_exec_async = policy::hip::hip_exec<BLOCK_SIZE, true>;

using policy::hip::hip_seq_syncthreads_exec;
using policy::hip::hip_reduce_base;
using policy::hip::hip_reduce;
using policy::hip::hip_reduce_atomic;
using policy::hip::HipPolicy;

using policy::hip::hip_synchronize;

/*!
 * \brief Struct that contains two HIP dim3's that represent the number of
 * thread block and the number of blocks.
 *
 * This is passed to the execution policies to setup the kernel launch.
 */
struct HipDim {
  hip_dim_t num_threads;
  hip_dim_t num_blocks;

  RAJA_HOST_DEVICE void print(void) const
  {
    printf("<<< (%d,%d,%d), (%d,%d,%d) >>>\n",
           (int)num_blocks.x,
           (int)num_blocks.y,
           (int)num_blocks.z,
           (int)num_threads.x,
           (int)num_threads.y,
           (int)num_threads.z);
  }
};


RAJA_INLINE
constexpr RAJA::Index_type numBlocks(HipDim const &dim)
{
  return dim.num_blocks.x * dim.num_blocks.y * dim.num_blocks.z;
}

RAJA_INLINE
constexpr RAJA::Index_type numThreads(HipDim const &dim)
{
  return dim.num_threads.x * dim.num_threads.y * dim.num_threads.z;
}


template <typename POL, typename IDX>
struct HipIndexPair : public POL {
  template <typename IS>
  RAJA_INLINE constexpr HipIndexPair(HipDim &dims, IS const &is)
      : POL(dims, is)
  {
  }

  typedef IDX INDEX;
};

/** Provides a range from 0 to N_iter - 1
 *
 */
template <typename VIEWDIM, size_t threads_per_block>
struct HipThreadBlock {
  RAJA::Index_type distance;

  VIEWDIM view;

  template <typename Iterable>
  HipThreadBlock(HipDim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  __device__ inline RAJA::Index_type operator()(void)
  {
    RAJA::Index_type idx =
        (RAJA::Index_type)view(blockIdx) * (RAJA::Index_type)threads_per_block +
        (RAJA::Index_type)view(threadIdx);

    if (idx >= distance) {
      idx = RAJA::operators::limits<RAJA::Index_type>::min();
    }

    return idx;
  }

  void inline setDims(HipDim &dims)
  {
    RAJA::Index_type n = distance;
    if (n < threads_per_block) {
      view(dims.num_threads) = n;
      view(dims.num_blocks) = 1;
    } else {
      view(dims.num_threads) = threads_per_block;

      RAJA::Index_type blocks = n / threads_per_block;
      if (n % threads_per_block) {
        ++blocks;
      }
      view(dims.num_blocks) = blocks;
    }
  }
};

/*
 * These execution policies map a loop nest to the block and threads of a
 * given dimension with the number of THREADS per block specifies.
 */

template <size_t THREADS>
using hip_threadblock_x_exec = HipPolicy<HipThreadBlock<Dim3x, THREADS>>;

template <size_t THREADS>
using hip_threadblock_y_exec = HipPolicy<HipThreadBlock<Dim3y, THREADS>>;

template <size_t THREADS>
using hip_threadblock_z_exec = HipPolicy<HipThreadBlock<Dim3z, THREADS>>;

template <typename VIEWDIM>
struct HipThread {
  RAJA::Index_type distance;

  VIEWDIM view;

  template <typename Iterable>
  HipThread(HipDim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  __device__ inline RAJA::Index_type operator()(void)
  {
    RAJA::Index_type idx = view(threadIdx);
    if (idx >= distance) {
      return RAJA::operators::limits<RAJA::Index_type>::min();
    }
    return idx;
  }

  void inline setDims(HipDim &dims) { view(dims.num_threads) = distance; }
};

/* These execution policies map the given loop nest to the threads in the
   specified dimensions (not blocks)
 */
using hip_thread_x_exec = HipPolicy<HipThread<Dim3x>>;

using hip_thread_y_exec = HipPolicy<HipThread<Dim3y>>;

using hip_thread_z_exec = HipPolicy<HipThread<Dim3z>>;

template <typename VIEWDIM>
struct HipBlock {
  RAJA::Index_type distance;

  VIEWDIM view;

  template <typename Iterable>
  HipBlock(HipDim &dims, Iterable const &i)
      : distance(std::distance(std::begin(i), std::end(i)))
  {
    setDims(dims);
  }

  __device__ inline RAJA::Index_type operator()(void)
  {
    RAJA::Index_type idx = view(blockIdx);
    if (idx >= distance) {
      return RAJA::operators::limits<RAJA::Index_type>::min();
    }
    return idx;
  }

  void inline setDims(HipDim &dims) { view(dims.num_blocks) = distance; }
};

/* These execution policies map the given loop nest to the blocks in the
   specified dimensions (not threads)
 */
using hip_block_x_exec = HipPolicy<HipBlock<Dim3x>>;

using hip_block_y_exec = HipPolicy<HipBlock<Dim3y>>;

using hip_block_z_exec = HipPolicy<HipBlock<Dim3z>>;


///
///////////////////////////////////////////////////////////////////////
///
/// Shared memory policies
///
///////////////////////////////////////////////////////////////////////
///

/*!
 * HIP shared memory
 */

struct hip_shmem {
};


/*!
 * HIP shared memory that allows global indexing into a block's shmem
 */
template <typename DimView>
struct block_map_shmem {

  template <typename T>
  RAJA_INLINE RAJA_DEVICE static T apply(ptrdiff_t dim_size, T idx)
  {
    DimView dim_view;
    ptrdiff_t block_offset = dim_view(blockIdx) * dim_size;
    return idx - block_offset;
  }
};

using block_map_x_shmem = block_map_shmem<Dim3x>;
using block_map_y_shmem = block_map_shmem<Dim3y>;
using block_map_z_shmem = block_map_shmem<Dim3z>;


}  // namespace RAJA

#endif  // RAJA_ENABLE_HIP
#endif
