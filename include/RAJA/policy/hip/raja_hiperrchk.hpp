/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file containing utility methods used in HIP operations.
 *
 *          These methods work only on platforms that support HIP.
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

#ifndef RAJA_raja_hiperrchk_HPP
#define RAJA_raja_hiperrchk_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <iostream>
#include <string>

#include <hip/hip_runtime.h>

#include "RAJA/util/macros.hpp"

namespace RAJA
{

///
///////////////////////////////////////////////////////////////////////
///
/// Utility assert method used in HIP operations to report HIP
/// error codes when encountered.
///
///////////////////////////////////////////////////////////////////////
///
#define hipErrchk(ans)                            \
  {                                                \
    ::RAJA::hipAssert((ans), __FILE__, __LINE__); \
  }

inline void hipAssert(hipError_t code,
                       const char *file,
                       int line,
                       bool abort = true)
{
  if (code != hipSuccess) {
    fprintf(
        stderr, "HIPassert: %s %s %d\n", hipGetErrorString(code), file, line);
    if (abort) RAJA_ABORT_OR_THROW("HIPassert");
  }
}

}  // namespace RAJA

#endif  // closing endif for if defined(RAJA_ENABLE_HIP)

#endif  // closing endif for header file include guard
