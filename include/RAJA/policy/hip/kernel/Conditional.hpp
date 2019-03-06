/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for HIP kernel conditional methods.
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

#ifndef RAJA_policy_hip_kernel_Conditional_HPP
#define RAJA_policy_hip_kernel_Conditional_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/Conditional.hpp"

#include "RAJA/policy/hip/kernel/internal.hpp"

namespace RAJA
{
namespace internal
{


template <typename Data,
          typename Conditional,
          typename... EnclosedStmts>
struct HipStatementExecutor<Data,
                             statement::If<Conditional, EnclosedStmts...>> {

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t>;


  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    if (Conditional::eval(data)) {

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }



  static
  inline
  LaunchDims calculateDimensions(Data const &data)
  {
    return enclosed_stmts_t::calculateDimensions(data);
  }
};


}  // namespace internal
}  // end namespace RAJA


#endif
