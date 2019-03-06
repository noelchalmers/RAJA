/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Header file for HIP shared memory window executors.
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


#ifndef RAJA_policy_hip_kernel_InitLocalMem_HPP
#define RAJA_policy_hip_kernel_InitLocalMem_HPP

#include "RAJA/config.hpp"

#include <iostream>
#include <type_traits>

#include "RAJA/util/macros.hpp"
#include "RAJA/util/types.hpp"

#include "RAJA/pattern/kernel/InitLocalMem.hpp"
#include "RAJA/policy/hip/kernel/internal.hpp"

namespace RAJA
{
namespace internal
{

//Intialize thread shared array
template <typename Data, camp::idx_t... Indices, typename... EnclosedStmts>
struct HipStatementExecutor<Data, statement::InitLocalMem<RAJA::hip_shared_mem, camp::idx_seq<Indices...>, EnclosedStmts...>>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t>;


  //Launch loops
  template<camp::idx_t Pos>
  static
  inline
  RAJA_DEVICE
  void initMem(Data &data, bool thread_active)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::element_t;
    const camp::idx_t NumElem = camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::NumElem;

    __shared__ varType Array[NumElem];
    camp::get<Pos>(data.param_tuple).m_arrayPtr = Array;

    enclosed_stmts_t::exec(data, thread_active);
  }

  //Intialize local array
  //Identifies type + number of elements needed
  template<camp::idx_t Pos, camp::idx_t Pos2, camp::idx_t... others>
  static
  inline
  RAJA_DEVICE
  void initMem(Data &data, bool thread_active)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::element_t;
    const camp::idx_t NumElem = camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::NumElem;

    __shared__ varType Array[NumElem];
    camp::get<Pos>(data.param_tuple).m_arrayPtr = Array;
    initMem<Pos2, others...>(data, thread_active);
  }

  //Set pointer to null base case
  template<camp::idx_t Pos>
  static
  inline
  RAJA_DEVICE
  void setPtrToNull(Data &data)
  {

    camp::get<Pos>(data.param_tuple).m_arrayPtr = nullptr;
  }


  //Set pointer to null recursive case
  template<camp::idx_t Pos, camp::idx_t Pos2, camp::idx_t... others>
  static
  inline
  RAJA_DEVICE
  void setPtrToNull(Data &data)
  {

    camp::get<Pos>(data.param_tuple).m_arrayPtr = nullptr;
    setPtrToNull<Pos2, others...>(data);
  }


  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {

    //Intialize scoped arrays + launch loops
    initMem<Indices...>(data, thread_active);

    //set pointers in scoped arrays to null
    setPtrToNull<Indices...>(data);
  }


  inline
  static
  LaunchDims calculateDimensions(Data const &data)
  {
    return enclosed_stmts_t::calculateDimensions(data);
  }

};

//Intialize thread private array
template <typename Data, camp::idx_t... Indices, typename... EnclosedStmts>
struct HipStatementExecutor<Data, statement::InitLocalMem<RAJA::hip_thread_mem, camp::idx_seq<Indices...>, EnclosedStmts...>>
{

  using stmt_list_t = StatementList<EnclosedStmts...>;
  using enclosed_stmts_t = HipStatementListExecutor<Data, stmt_list_t>;


  //Launch loops
  template<camp::idx_t Pos>
  static
  inline
  RAJA_DEVICE
  void initMem(Data &data, bool thread_active)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::element_t;
    const camp::idx_t NumElem = camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::NumElem;

    varType Array[NumElem];
    camp::get<Pos>(data.param_tuple).m_arrayPtr = Array;

    enclosed_stmts_t::exec(data, thread_active);
  }

  //Intialize local array
  //Identifies type + number of elements needed
  template<camp::idx_t Pos, camp::idx_t Pos2, camp::idx_t... others>
  static
  inline
  RAJA_DEVICE
  void initMem(Data &data, bool thread_active)
  {
    using varType = typename camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::element_t;
    const camp::idx_t NumElem = camp::tuple_element_t<Pos, typename camp::decay<Data>::param_tuple_t>::NumElem;

    varType Array[NumElem];
    camp::get<Pos>(data.param_tuple).m_arrayPtr = Array;
    initMem<Pos2, others...>(data, thread_active);
  }

  //Set pointer to null base case
  template<camp::idx_t Pos>
  static
  inline
  RAJA_DEVICE
  void setPtrToNull(Data &data)
  {

    camp::get<Pos>(data.param_tuple).m_arrayPtr = nullptr;
  }


  //Set pointer to null recursive case
  template<camp::idx_t Pos, camp::idx_t Pos2, camp::idx_t... others>
  static
  inline
  RAJA_DEVICE
  void setPtrToNull(Data &data)
  {

    camp::get<Pos>(data.param_tuple).m_arrayPtr = nullptr;
    setPtrToNull<Pos2, others...>(data);
  }


  static
  inline
  RAJA_DEVICE
  void exec(Data &data, bool thread_active)
  {

    //Intialize scoped arrays + launch loops
    initMem<Indices...>(data, thread_active);

    //set pointers in scoped arrays to null
    setPtrToNull<Indices...>(data);
  }


  inline
  static
  LaunchDims calculateDimensions(Data const &data)
  {
    return enclosed_stmts_t::calculateDimensions(data);
  }

};



}  // namespace internal
}  // end namespace RAJA


#endif
