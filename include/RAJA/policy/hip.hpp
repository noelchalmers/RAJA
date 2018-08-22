/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing RAJA headers for HIPCC HIP execution.
 *
 *          These methods work only on platforms that support HIP.
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

#ifndef RAJA_hip_HPP
#define RAJA_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <hip/hip_runtime.h>

#include "RAJA/policy/hip/atomic.hpp"
#include "RAJA/policy/hip/forall.hpp"
#include "RAJA/policy/hip/policy.hpp"
#include "RAJA/policy/hip/reduce.hpp"
//#if defined(__NVCC__)
//#include "RAJA/policy/hip/scan.hpp"
//#endif
#include "RAJA/policy/hip/synchronize.hpp"
#include "RAJA/policy/hip/forallN.hpp"
#include "RAJA/policy/hip/shared_memory.hpp"
#include "RAJA/policy/hip/kernel.hpp"

#endif  // closing endif for if defined(RAJA_ENABLE_HIP)

#endif  // closing endif for header file include guard
