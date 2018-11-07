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

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{


namespace internal
{
  /*!
   * HIP global function for launching HipKernel policies
   */
  template <typename StmtList, typename Data, typename Exec>
  __global__ void HipKernelLauncher(Data data, Exec executor, int num_logical_blocks)
  {
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    using exec_t = camp::decay<Exec>;
    exec_t private_executor = executor;

    private_executor.initThread(data);

    // execute the the object
    private_executor.exec(private_data, num_logical_blocks, -1);
  }


  template <size_t BlockSize, typename StmtList, typename Data, typename Exec>
  __launch_bounds__(BlockSize, 1)
  __global__
  void HipKernelLauncherFixed(Data data, Exec executor, int num_logical_blocks)
  {
    using data_t = camp::decay<Data>;
    data_t private_data = data;

    using exec_t = camp::decay<Exec>;
    exec_t private_executor = executor;

    private_executor.initThread(data);

    // execute the the object
    private_executor.exec(private_data, num_logical_blocks, -1);
  }

} //namespace internal

/*!
 * HIP kernel launch policy where the user specifies the number of physical
 * thread blocks and threads per block.
 */
template <bool async0, int num_blocks, int num_threads>
struct hip_explicit_launch {

  static constexpr bool async = async0;

  template <typename Func>
  RAJA_INLINE static internal::LaunchDim calc_max_physical(Func const &, int)
  {
    int nblocks = num_blocks;

    // Use maximum number of blocks if user didn't specify
    if(num_blocks <= 0){
      nblocks = RAJA::hip::internal::getMaxBlocks();
    }

    return internal::LaunchDim(nblocks, num_threads);
  }


  template<typename StmtList, typename Data, typename Exec>
  static void launch(Data const &hip_data, Exec &exec, internal::LaunchDim launch_dims,
                     internal::LaunchDim logical_dims,
                     size_t shmem, hipStream_t stream)
  {
    // launch using global function that fixes the number of threads at
    // compile time.  This guarantees that the compiler will generate a kernel
    // that will fit in a block of num_threads.
    hipLaunchKernelGGL((RAJA::internal::HipKernelLauncherFixed<num_threads, StmtList>), dim3(launch_dims.blocks), dim3(launch_dims.threads), shmem, stream,
            hip_data, exec, logical_dims.blocks);
  }
};


/*!
 * HIP kernel launch policy where the number of physical blocks and threads
 * are determined by the HIP occupancy calculator.
 */
template <int num_threads0, bool async0>
struct hip_occ_calc_launch {

  static constexpr bool async = async0;

  static constexpr int num_threads = num_threads0;

  template <typename Func>
  RAJA_INLINE static internal::LaunchDim calc_max_physical(Func const &func,
                                                           int shmem_size)
  {

    int occ_blocks = -1, occ_threads = -1;

    hipOccupancyMaxPotentialBlockSize(&occ_blocks,
                                       &occ_threads,
                                       func,
                                       shmem_size);

    return internal::LaunchDim(occ_blocks, occ_threads);
  }

  template<typename StmtList, typename Data, typename Exec>
  static void launch(Data const &hip_data, Exec &exec, internal::LaunchDim launch_dims,
                     internal::LaunchDim logical_dims,
                     size_t shmem, hipStream_t stream)
  {
    hipLaunchKernelGGL((RAJA::internal::HipKernelLauncher<StmtList>), dim3(launch_dims.blocks), dim3(launch_dims.threads), shmem, stream,
              hip_data, exec, logical_dims.blocks);
  }

};

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
 * A RAJA::kernel statement that launches a HIP kernel.
 *
 *
 */
template <typename... EnclosedStmts>
using HipKernelOcc =
    HipKernelExt<hip_occ_calc_launch<1024, false>, EnclosedStmts...>;

template <typename... EnclosedStmts>
using HipKernelOccAsync =
    HipKernelExt<hip_occ_calc_launch<1024, true>, EnclosedStmts...>;

template <int num_threads, typename... EnclosedStmts>
using HipKernelFixed =
    HipKernelExt<hip_explicit_launch<false, 0, num_threads>, EnclosedStmts...>;

template <int num_threads, typename... EnclosedStmts>
using HipKernelFixedAsync =
    HipKernelExt<hip_explicit_launch<true, 0, num_threads>, EnclosedStmts...>;

template <typename... EnclosedStmts>
using HipKernel = HipKernelFixed<1024, EnclosedStmts...>;

template <typename... EnclosedStmts>
using HipKernelAsync = HipKernelFixedAsync<1024, EnclosedStmts...>;

}  // namespace statement

namespace internal
{






/*!
 * Specialization that launches HIP kernels for RAJA::kernel from host code
 */
template <typename LaunchConfig, typename... EnclosedStmts>
struct StatementExecutor<statement::HipKernelExt<LaunchConfig,
                                                  EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using StatementType =
      statement::HipKernelExt<LaunchConfig, EnclosedStmts...>;

  template <typename Data>
  static RAJA_INLINE void exec(Data &&data)
  {

    using data_t = camp::decay<Data>;
    using executor_t = hip_statement_list_executor_t<stmt_list_t, data_t>;


    int shmem = (int)RAJA::internal::shmem_setup_buffers(data.param_tuple);
    hipStream_t stream = 0;



    //
    // Instantiate an executor object
    //
    executor_t executor;



    //
    // Compute the MAX physical kernel dimensions
    //
    LaunchDim max_physical = LaunchConfig::calc_max_physical(
        HipKernelLauncher<StatementList<EnclosedStmts...>, data_t, executor_t>, shmem);



    //
    // Privatize the LoopData, using make_launch_body to setup reductions
    //
    auto hip_data = RAJA::hip::make_launch_body(
        max_physical.blocks, max_physical.threads, shmem, stream, data);



    //
    // Compute the logical kernel dimensions
    //
    LaunchDim logical_dims = executor.calculateDimensions(data, max_physical);



    //
    // Compute the actual physical kernel dimensions
    //
    LaunchDim launch_dims;
    launch_dims.blocks = std::min(max_physical.blocks, logical_dims.blocks);
    launch_dims.threads = std::min(max_physical.threads, logical_dims.threads);


    // Only launch kernel if we have something to iterate over
    bool at_least_one_iter = launch_dims.blocks > 0 || launch_dims.threads > 0;
    bool is_degenerate =     launch_dims.blocks < 0 || launch_dims.threads < 0;
    if (at_least_one_iter && !is_degenerate) {

      //
      // Make sure that having either 0 blocks or 0 threads get bumped to 1
      //
      launch_dims.blocks = std::max(launch_dims.blocks, (int)1);
      launch_dims.threads = std::max(launch_dims.threads, (int)1);

      //
      // Launch the kernels
      //
//      printf("Launching kernel b=%d, t=%d\n",
//          (int)launch_dims.blocks, (int)launch_dims.threads);
      LaunchConfig::template launch<StatementList<EnclosedStmts...>>(
          hip_data, executor, launch_dims, logical_dims, shmem, stream);


      //
      // Check for errors
      //
      RAJA::hip::peekAtLastError();


      //
      // Synchronize
      //
      RAJA::hip::launch(stream);

      if (!LaunchConfig::async) {
        RAJA::hip::synchronize(stream);
      }
    }

  }
};


}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
