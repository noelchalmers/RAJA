/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for HIP statement executors.
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


#ifndef RAJA_policy_hip_kernel_ForICount_HPP
#define RAJA_policy_hip_kernel_ForICount_HPP

#include "RAJA/config.hpp"

#include "RAJA/policy/hip/kernel/internal.hpp"


namespace RAJA
{

namespace internal
{



/*
 * Executor for thread work sharing loop inside HipKernel.
 * Mapping directly from threadIdx.xyz to indices
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop count to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int ThreadDim,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::hip_thread_xyz_direct<ThreadDim>, EnclosedStmts...>>
    : public HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_thread_xyz_direct<ThreadDim>, EnclosedStmts...>> {

  using Base = HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_thread_xyz_direct<ThreadDim>, EnclosedStmts...>>;

  using typename Base::enclosed_stmts_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    auto len = segment_length<ArgumentId>(data);
    auto i = get_hip_dim<ThreadDim>(dim3(threadIdx.x,threadIdx.y,threadIdx.z));

    // assign thread id directly to offset
    data.template assign_offset<ArgumentId>(i);
    data.template assign_param<ParamId>(i);

    // execute enclosed statements if in bounds
    enclosed_stmts_t::exec(data, thread_active && (i<len));

  }
};




/*
 * Executor for thread work sharing loop inside HipKernel.
 * Provides a block-stride loop (stride of blockDim.xyz) for
 * each thread in xyz.
 * Assigns the loop iterate to offset ArgumentId
 * Assigns the loop offset to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int ThreadDim,
          int MinThreads,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::hip_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>>
    : public HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>> {

  using Base = HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_thread_xyz_loop<ThreadDim, MinThreads>, EnclosedStmts...>>;

  using typename Base::enclosed_stmts_t;

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // block stride loop
    int len = segment_length<ArgumentId>(data);
    auto i0 = get_hip_dim<ThreadDim>(dim3(threadIdx.x,threadIdx.y,threadIdx.z));
    auto i_stride = get_hip_dim<ThreadDim>(dim3(blockDim.x,blockDim.y,blockDim.z));
    auto i = i0;
    for(;i < len;i += i_stride){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
    // do we need one more masked iteration?
    if(i - i0 < len)
    {
      // execute enclosed statements one more time, but masking them off
      // this is because there's at least one thread that isn't masked off
      // that is still executing the above loop
      enclosed_stmts_t::exec(data, false);
    }
  }
};




/*
 * Executor for block work sharing inside HipKernel.
 * Provides a grid-stride loop (stride of gridDim.xyz) for
 * each block in xyz.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          int BlockDim,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, RAJA::hip_block_xyz_loop<BlockDim>, EnclosedStmts...>>
    : public HipStatementExecutor<
        Data,
        statement::For<ArgumentId, RAJA::hip_block_xyz_loop<BlockDim>, EnclosedStmts...>> {

  using Base = HipStatementExecutor<
      Data,
      statement::For<ArgumentId, RAJA::hip_block_xyz_loop<BlockDim>, EnclosedStmts...>>;

  using typename Base::enclosed_stmts_t;

  static
  inline RAJA_DEVICE void exec(Data &data, bool thread_active)
  {
    // grid stride loop
    int len = segment_length<ArgumentId>(data);
    auto i0 = get_hip_dim<BlockDim>(dim3(blockIdx.x,blockIdx.y,blockIdx.z));
    auto i_stride = get_hip_dim<BlockDim>(dim3(gridDim.x,gridDim.y,gridDim.z));
    for(auto i = i0;i < len;i += i_stride){

      // Assign the x thread to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }
};


/*
 * Executor for sequential loops inside of a HipKernel.
 *
 * This is specialized since it need to execute the loop immediately.
 * Assigns the loop index to offset ArgumentId
 * Assigns the loop index to param ParamId
 */
template <typename Data,
          camp::idx_t ArgumentId,
          typename ParamId,
          typename... EnclosedStmts>
struct HipStatementExecutor<
    Data,
    statement::ForICount<ArgumentId, ParamId, seq_exec, EnclosedStmts...> >
    : public HipStatementExecutor<
        Data,
        statement::For<ArgumentId, seq_exec, EnclosedStmts...> > {

  using Base = HipStatementExecutor<
      Data,
      statement::For<ArgumentId, seq_exec, EnclosedStmts...> >;

  using typename Base::enclosed_stmts_t;

  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {
    using idx_type = camp::decay<decltype(camp::get<ArgumentId>(data.offset_tuple))>;

    idx_type len = segment_length<ArgumentId>(data);

    for(idx_type i = 0;i < len;++ i){
      // Assign i to the argument
      data.template assign_offset<ArgumentId>(i);
      data.template assign_param<ParamId>(i);

      // execute enclosed statements
      enclosed_stmts_t::exec(data, thread_active);
    }
  }
};





}  // namespace internal
}  // end namespace RAJA


#endif /* RAJA_pattern_kernel_HPP */
