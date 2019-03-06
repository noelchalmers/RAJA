/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel
 *          traversals on GPU with HIP.
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
//////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2018,2019 Advanced Micro Devices, Inc.
//////////////////////////////////////////////////////////////////////////////

#ifndef RAJA_policy_hip_kernel_HipKernel_HPP
#define RAJA_policy_hip_kernel_HipKernel_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel.hpp"
#include "RAJA/pattern/kernel/For.hpp"
#include "RAJA/pattern/kernel/Lambda.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"

#include "RAJA/internal/LegacyCompatibility.hpp"

#include "RAJA/policy/hip/kernel/internal.hpp"

namespace RAJA
{

/*!
 * HIP kernel launch policy where the user specifies the number of physical
 * thread blocks and threads per block.
 */
template <bool async0, int num_blocks, int num_threads>
struct hip_explicit_launch {};


/*!
 * HIP kernel launch policy where the number of physical blocks and threads
 * are determined by the HIP occupancy calculator.
 */
template <int num_threads0, bool async0>
struct hip_occ_calc_launch {};


namespace statement
{

/*!
 * A RAJA::kernel statement that launches a HIP kernel.
 *
 *
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct HipKernelExt
    : public internal::Statement<hip_exec<0>, EnclosedStmts...> {
};


/*!
 * A RAJA::kernel statement that launches a HIP kernel using the
 * HIP occupancy calculator to determine the optimal number of threads.
 * The kernel launch is synchronous.
 */
template <typename... EnclosedStmts>
using HipKernelOcc =
    HipKernelExt<hip_occ_calc_launch<1024, false>, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a HIP kernel using the
 * HIP occupancy calculator to determine the optimal number of threads.
 * Thre kernel launch is asynchronous.
 */
template <typename... EnclosedStmts>
using HipKernelOccAsync =
    HipKernelExt<hip_occ_calc_launch<1024, true>, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a HIP kernel with a fixed
 * number of threads (specified by num_threads)
 * Thre kernel launch is synchronous.
 */
template <int num_threads, typename... EnclosedStmts>
using HipKernelFixed =
    HipKernelExt<hip_explicit_launch<false, 0, num_threads>,
                  EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a HIP kernel with a fixed
 * number of threads (specified by num_threads)
 * Thre kernel launch is asynchronous.
 */
template <int num_threads, typename... EnclosedStmts>
using HipKernelFixedAsync =
    HipKernelExt<hip_explicit_launch<true, 0, num_threads>, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a HIP kernel with 1024 threads
 * Thre kernel launch is synchronous.
 */
template <typename... EnclosedStmts>
using HipKernel = HipKernelFixed<1024, EnclosedStmts...>;

/*!
 * A RAJA::kernel statement that launches a HIP kernel with 1024 threads
 * Thre kernel launch is asynchronous.
 */
template <typename... EnclosedStmts>
using HipKernelAsync = HipKernelFixedAsync<1024, EnclosedStmts...>;

}  // namespace statement

namespace internal
{


/*!
 * HIP global function for launching HipKernel policies
 */
template <typename Data, typename Exec>
__global__ void HipKernelLauncher(Data data)
{

  using data_t = camp::decay<Data>;
  data_t private_data = data;

  Exec::exec(private_data, true);
}


/*!
 * HIP global function for launching HipKernel policies
 * This is annotated to gaurantee that device code generated
 * can be launched by a kernel with BlockSize number of threads.
 *
 * This launcher is used by the HipKerelFixed policies.
 */
template <size_t BlockSize, typename Data, typename Exec>
__launch_bounds__(BlockSize, 1) __global__
    void HipKernelLauncherFixed(Data data)
{

  using data_t = camp::decay<Data>;
  data_t private_data = data;

  // execute the the object
  Exec::exec(private_data, true);
}

/*!
 * Helper class that handles HIP kernel launching, and computing
 * maximum number of threads/blocks
 */
template<typename LaunchPolicy, typename StmtList, typename Data>
struct HipLaunchHelper;


/*!
 * Helper class specialization to use the HIP occupancy calculator to
 * determine the number of threads and blocks
 */
template<int num_threads, bool async0, typename StmtList, typename Data>
struct HipLaunchHelper<hip_occ_calc_launch<num_threads, async0>,StmtList,Data>
{
  static constexpr bool async = async0;

  using executor_t = internal::hip_statement_list_executor_t<StmtList, Data>;

  inline static void max_blocks_threads(int shmem_size,
      int &max_blocks, int &max_threads)
  {

    auto func = internal::HipKernelLauncher<Data, executor_t>;

    hipOccupancyMaxPotentialBlockSize(&max_blocks,
                                       &max_threads,
                                       func,
                                       shmem_size);

  }

  static void launch(Data const &data,
                     internal::LaunchDims launch_dims,
                     size_t shmem,
                     hipStream_t stream)
  {

    auto func = internal::HipKernelLauncher<Data, executor_t>;

    hipLaunchKernelGGL((func), dim3(launch_dims.blocks), dim3(launch_dims.threads),
                        shmem, stream, data);
  }
};



/*!
 * Helper class specialization to use the HIP device properties and a user
 * specified number of threads to compute the number of blocks/threads
 */
template<bool async0, int num_blocks, int num_threads, typename StmtList, typename Data>
struct HipLaunchHelper<hip_explicit_launch<async0, num_blocks, num_threads>,StmtList,Data>
{
  static constexpr bool async = async0;

  using executor_t = internal::hip_statement_list_executor_t<StmtList, Data>;

  inline static void max_blocks_threads(int shmem_size,
      int &max_blocks, int &max_threads)
  {

    max_blocks = num_blocks;
    max_threads = num_threads;

    // Use maximum number of blocks if user didn't specify
    if (num_blocks <= 0) {
      max_blocks = RAJA::hip::internal::getMaxBlocks();
    }

  }

  static void launch(Data const &data,
                     internal::LaunchDims launch_dims,
                     size_t shmem,
                     hipStream_t stream)
  {

    auto func = internal::HipKernelLauncherFixed<num_threads,Data, executor_t>;

    hipLaunchKernelGGL((func), dim3(launch_dims.blocks), dim3(launch_dims.threads),
                        shmem, stream, data);
  }
};

/*!
 * Helper function that is used to compute either the number of blocks
 * or threads that get launched.
 * It takes the max threads (limit), the requested number (result),
 * and a minimum limit (minimum).
 *
 * The algorithm is greedy (and probably could be improved), and favors
 * maximizing the number of threads (or blocks) in x, y, then z.
 */
inline
hip_dim_t fitHipDims(int limit, hip_dim_t result, hip_dim_t minimum = hip_dim_t()){


  // clamp things to at least 1
  result.x = result.x ? result.x : 1;
  result.y = result.y ? result.y : 1;
  result.z = result.z ? result.z : 1;

  minimum.x = minimum.x ? minimum.x : 1;
  minimum.y = minimum.y ? minimum.y : 1;
  minimum.z = minimum.z ? minimum.z : 1;

  // if we are under the limit, we're done
  if(result.x * result.y * result.z <= limit) return result;

  // Can we reduce z to fit?
  if(result.x * result.y * minimum.z < limit){
    // compute a new z
    result.z = limit / (result.x*result.y);
    return result;
  }
  // we don't fit, so reduce z to it's minimum and continue on to y
  result.z = minimum.z;


  // Can we reduce y to fit?
  if(result.x * minimum.y * result.z < limit){
    // compute a new y
    result.y = limit / (result.x*result.z);
    return result;
  }
  // we don't fit, so reduce y to it's minimum and continue on to x
  result.y = minimum.y;


  // Can we reduce y to fit?
  if(minimum.x * result.y * result.z < limit){
    // compute a new x
    result.x = limit / (result.y*result.z);
    return result;
  }
  // we don't fit, so we'll return the smallest possible thing
  result.x = minimum.x;

  return result;
}


/*!
 * Specialization that launches HIP kernels for RAJA::kernel from host code
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct StatementExecutor<
    statement::HipKernelExt<LaunchConfig, EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using StatementType =
      statement::HipKernelExt<LaunchConfig, EnclosedStmts...>;

  template <typename Data>
  static inline void exec(Data &&data)
  {

    using data_t = camp::decay<Data>;
    using executor_t = hip_statement_list_executor_t<stmt_list_t, data_t>;
    using launch_t = HipLaunchHelper<LaunchConfig, stmt_list_t, data_t>;


    //
    // Setup shared memory buffers
    //
    int shmem = 0;
    hipStream_t stream = 0;


    //
    // Compute the MAX physical kernel dimensions
    //
    int max_blocks, max_threads;
    launch_t::max_blocks_threads(shmem, max_blocks, max_threads);


    //
    // Privatize the LoopData, using make_launch_body to setup reductions
    //
    auto hip_data = RAJA::hip::make_launch_body(
        max_blocks, max_threads, shmem, stream, data);


    //
    // Compute the requested kernel dimensions
    //
    LaunchDims launch_dims = executor_t::calculateDimensions(data);


    // Only launch kernel if we have something to iterate over
    int num_blocks = launch_dims.num_blocks();
    int num_threads = launch_dims.num_threads();
    if (num_blocks > 0 || num_threads > 0) {

      //
      // Fit the requested threads an blocks
      //
      launch_dims.blocks = fitHipDims(max_blocks, launch_dims.blocks);
      launch_dims.threads = fitHipDims(max_threads, launch_dims.threads, launch_dims.min_threads);

      // make sure that we fit
      if(launch_dims.num_blocks() > max_blocks){
        RAJA_ABORT_OR_THROW("RAJA::kernel exceeds max num blocks");
      }
      if(launch_dims.num_threads() > max_threads){
        RAJA_ABORT_OR_THROW("RAJA::kernel exceeds max num threads");
      }

      //
      // Launch the kernels
      //
      launch_t::launch(hip_data, launch_dims, shmem, stream);


      //
      // Check for errors
      //
      RAJA::hip::peekAtLastError();


      //
      // Synchronize
      //
      RAJA::hip::launch(stream);

      if (!launch_t::async) {
        RAJA::hip::synchronize(stream);
      }
    }
  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
