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


#ifndef RAJA_policy_hip_kernel_Sync_HPP
#define RAJA_policy_hip_kernel_Sync_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cassert>
#include <climits>

#include "camp/camp.hpp"

#include "RAJA/pattern/kernel.hpp"

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"


namespace RAJA
{
namespace statement
{

/*!
 * A RAJA::kernel statement that performs a HIP __syncthreads().
 */
struct HipSyncThreads : public internal::Statement<camp::nil> {
};

}  // namespace statement

namespace internal
{

template <typename Data>
struct HipStatementExecutor<Data, statement::HipSyncThreads> {

  static
  inline
  RAJA_DEVICE
  void exec(Data &, bool) { __syncthreads(); }


  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    return LaunchDims();
  }
};

}  // namespace internal
}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
