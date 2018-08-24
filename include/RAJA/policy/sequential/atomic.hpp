/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file defining sequential atomic operations.
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

#ifndef RAJA_policy_sequential_atomic_HPP
#define RAJA_policy_sequential_atomic_HPP

#include "RAJA/config.hpp"

#include "RAJA/util/macros.hpp"

namespace RAJA
{
namespace atomic
{

struct seq_atomic {
};


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicAdd(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc += value;
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicSub(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc -= value;
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicMin(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc = ret < value ? ret : value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicMax(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc = ret > value ? ret : value;
  return ret;
}


RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicInc(seq_atomic, T volatile *acc)
{
  T ret = *acc;
  (*acc) += 1;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicInc(seq_atomic, T volatile *acc, T val)
{
  T old = *acc;
  (*acc) = ((old >= val) ? 0 : (old + 1));
  return old;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicDec(seq_atomic, T volatile *acc)
{
  T ret = *acc;
  (*acc) -= 1;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicDec(seq_atomic, T volatile *acc, T val)
{
  T old = *acc;
  (*acc) = (((old == 0) | (old > val)) ? val : (old - 1));
  return old;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicAnd(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc &= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicOr(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc |= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicXor(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc ^= value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicExchange(seq_atomic, T volatile *acc, T value)
{
  T ret = *acc;
  *acc = value;
  return ret;
}

RAJA_SUPPRESS_HD_WARN
template <typename T>
RAJA_INLINE T RAJA_HOST_DEVICE atomicCAS(seq_atomic, T volatile *acc, T compare, T value)
{
  T ret = *acc;
  *acc = ret == compare ? value : ret;
  return ret;
}


}  // namespace atomic
}  // namespace RAJA


#endif  // guard
