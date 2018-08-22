/*!
 ******************************************************************************
 *
 * \file
 *
 * \brief   Implementation file for routines used to manage
 *          memory for HIP reductions and other operations.
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

#include "RAJA/config.hpp"

#if defined(RAJA_ENABLE_HIP)

#include "RAJA/policy/hip/MemUtils_HIP.hpp"

#include "RAJA/policy/hip/raja_hiperrchk.hpp"



namespace RAJA
{

namespace hip
{

namespace detail
{
//
/////////////////////////////////////////////////////////////////////////////
//
// Variables representing the state of execution.
//
/////////////////////////////////////////////////////////////////////////////
//

//! State of the host code globally
hipInfo g_status;

//! State of the host code in this thread
hipInfo tl_status;
#if defined(RAJA_ENABLE_OPENMP) && defined(_OPENMP)
#pragma omp threadprivate(tl_status)
#endif

//! State of raja hip stream synchronization for hip reducer objects
std::unordered_map<hipStream_t, bool> g_stream_info_map{ {hipStream_t(0), true} };


}  // closing brace for detail namespace

}  // closing brace for hip namespace

}  // closing brace for RAJA namespace


#endif  // if defined(RAJA_ENABLE_HIP)



