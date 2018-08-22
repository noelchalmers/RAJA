/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   RAJA header file containing constructs used to run forallN
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

#ifndef RAJA_forallN_hip_HPP
#define RAJA_forallN_hip_HPP

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include <cassert>
#include <climits>

#include "RAJA/util/types.hpp"

#include "RAJA/policy/hip/MemUtils_HIP.hpp"
#include "RAJA/policy/hip/policy.hpp"

#include "RAJA/internal/ForallNPolicy.hpp"
#include "RAJA/internal/LegacyCompatibility.hpp"


namespace RAJA
{

/*!
 * \brief Functor that binds the first argument of a callable.
 *
 * This version has host-device constructor and device-only operator.
 */
template <typename BODY>
struct ForallN_BindFirstArg_Device {
  BODY const &body;
  size_t i;

  RAJA_INLINE
  RAJA_DEVICE
  constexpr ForallN_BindFirstArg_Device(BODY &b, size_t i0) : body(b), i(i0) {}

  template <typename... ARGS>
  RAJA_INLINE RAJA_DEVICE void operator()(ARGS... args) const
  {
    body(i, args...);
  }
};


template <typename HIP_EXEC, typename Iterator>
struct HipIterableWrapper {
  HIP_EXEC pol_;
  Iterator i_;
  constexpr HipIterableWrapper(const HIP_EXEC &pol, const Iterator &i)
      : pol_(pol), i_(i)
  {
  }

  __device__ inline decltype(i_[0]) operator()()
  {
    auto val = pol_();
    return val > INT_MIN ? i_[pol_()] : INT_MIN;
  }
};

template <typename HIP_EXEC, typename Iterator>
auto make_hip_iter_wrapper(const HIP_EXEC &pol, const Iterator &i)
    -> HipIterableWrapper<HIP_EXEC, Iterator>
{
  return HipIterableWrapper<HIP_EXEC, Iterator>(pol, i);
}

/*!
 * \brief  Function to check indices for out-of-bounds
 *
 */
template <typename BODY, typename... ARGS>
RAJA_INLINE __device__ void hipCheckBounds(BODY &body, int i, ARGS... args)
{
  if (i > INT_MIN) {
    ForallN_BindFirstArg_Device<BODY> bound(body, i);
    hipCheckBounds(bound, args...);
  }
}

template <typename BODY>
RAJA_INLINE __device__ void hipCheckBounds(BODY &body, int i)
{
  if (i > INT_MIN) {
    body(i);
  }
}

/*!
 * \brief Launcher that uses execution policies to map blockIdx and threadIdx to
 * map
 * to N-argument function
 */
template <typename BODY, typename... CARGS>
__global__ void hipLauncherN(BODY loop_body, CARGS... cargs)
{
  // force reduction object copy constructors and destructors to run
  auto body = loop_body;

  // Compute indices and then pass through the bounds-checking mechanism
  hipCheckBounds(body, (cargs())...);
}

template <bool device,
          typename CuARG0,
          typename ISET0,
          typename CuARG1,
          typename ISET1,
          typename... CuARGS,
          typename... ISETS>
struct ForallN_Executor<device,
                        ForallN_PolicyPair<HipPolicy<CuARG0>, ISET0>,
                        ForallN_PolicyPair<HipPolicy<CuARG1>, ISET1>,
                        ForallN_PolicyPair<HipPolicy<CuARGS>, ISETS>...> {
  ForallN_PolicyPair<HipPolicy<CuARG0>, ISET0> iset0;
  ForallN_PolicyPair<HipPolicy<CuARG1>, ISET1> iset1;
  std::tuple<ForallN_PolicyPair<HipPolicy<CuARGS>, ISETS>...> isets;

  ForallN_Executor(
      ForallN_PolicyPair<HipPolicy<CuARG0>, ISET0> const &iset0_,
      ForallN_PolicyPair<HipPolicy<CuARG1>, ISET1> const &iset1_,
      ForallN_PolicyPair<HipPolicy<CuARGS>, ISETS> const &... isets_)
      : iset0(iset0_), iset1(iset1_), isets(isets_...)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY body) const
  {
    unpackIndexSets(body, VarOps::make_index_sequence<sizeof...(CuARGS)>{});
  }

  template <typename BODY, size_t... N>
  RAJA_INLINE void unpackIndexSets(BODY body,
                                   VarOps::index_sequence<N...>) const
  {
    HipDim dims;

    callLauncher(dims,
                 body,
                 make_hip_iter_wrapper(CuARG0(dims, iset0), std::begin(iset0)),
                 make_hip_iter_wrapper(CuARG1(dims, iset1), std::begin(iset1)),
                 make_hip_iter_wrapper(CuARGS(dims, std::get<N>(isets)),
                                        std::begin(std::get<N>(isets)))...);
  }

  template <typename BODY, typename... CARGS>
  RAJA_INLINE void callLauncher(HipDim const &dims,
                                BODY loop_body,
                                CARGS const &... cargs) const
  {
    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {

      bool Async = true;
      hipStream_t stream = 0;

      hipLaunchKernelGGL((hipLauncherN), dim3(dims.num_blocks), dim3(dims.num_threads), 0, stream,
          RAJA::hip::make_launch_body(dims.num_blocks,
                                       dims.num_threads,
                                       0,
                                       stream,
                                       std::move(loop_body)),
          cargs...);
      RAJA::hip::peekAtLastError();

      RAJA::hip::launch(stream);
      if (!Async) RAJA::hip::synchronize(stream);
    }
  }
};

template <bool device, typename CuARG0, typename ISET0>
struct ForallN_Executor<device, ForallN_PolicyPair<HipPolicy<CuARG0>, ISET0>> {
  ISET0 iset0;

  ForallN_Executor(ForallN_PolicyPair<HipPolicy<CuARG0>, ISET0> const &iset0_)
      : iset0(iset0_)
  {
  }

  template <typename BODY>
  RAJA_INLINE void operator()(BODY loop_body) const
  {
    HipDim dims;
    auto c0 = make_hip_iter_wrapper(CuARG0(dims, iset0), std::begin(iset0));

    if (numBlocks(dims) > 0 && numThreads(dims) > 0) {

      bool Async = true;
      hipStream_t stream = 0;

      hipLaunchKernelGGL((hipLauncherN), dim3(dims.num_blocks), dim3(dims.num_threads), 0, stream,
          RAJA::hip::make_launch_body(dims.num_blocks,
                                       dims.num_threads,
                                       0,
                                       stream,
                                       std::move(loop_body)),
          c0);
      RAJA::hip::peekAtLastError();

      RAJA::hip::launch(stream);
      if (!Async) RAJA::hip::synchronize(stream);
    }
  }
};

}  // namespace RAJA

#endif  // closing endif for RAJA_ENABLE_HIP guard

#endif  // closing endif for header file include guard
