/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief  Header file for HIP synchronize method.
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

#ifndef RAJA_synchronize_hip_HPP
#define RAJA_synchronize_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "RAJA/policy/hip/raja_hiperrchk.hpp"

namespace RAJA
{

namespace policy
{

namespace hip
{

/*!
 * \brief Synchronize the current HIP device.
 */
RAJA_INLINE
void synchronize_impl(const hip_synchronize&)
{
  hipErrchk(hipDeviceSynchronize());
}


}  // end of namespace hip
}  // namespace policy
}  // end of namespace RAJA

#endif  // defined(RAJA_ENABLE_HIP)

#endif  // RAJA_synchronize_hip_HPP
