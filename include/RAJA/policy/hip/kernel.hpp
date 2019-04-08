/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run kernel::forall
 *          traversals on GPU with CUDA.
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


#ifndef RAJA_policy_hip_kernel_HPP
#define RAJA_policy_hip_kernel_HPP

#include "RAJA/policy/hip/kernel/Conditional.hpp"
#include "RAJA/policy/hip/kernel/For.hpp"
#include "RAJA/policy/hip/kernel/ForICount.hpp"
#include "RAJA/policy/hip/kernel/HipKernel.hpp"
#include "RAJA/policy/hip/kernel/Hyperplane.hpp"
#include "RAJA/policy/hip/kernel/InitLocalMem.hpp"
#include "RAJA/policy/hip/kernel/Lambda.hpp"
#include "RAJA/policy/hip/kernel/Reduce.hpp"
#include "RAJA/policy/hip/kernel/Sync.hpp"
#include "RAJA/policy/hip/kernel/Tile.hpp"
#include "RAJA/policy/hip/kernel/TileTCount.hpp"
#include "RAJA/policy/hip/kernel/internal.hpp"

#endif  // closing endif for header file include guard
