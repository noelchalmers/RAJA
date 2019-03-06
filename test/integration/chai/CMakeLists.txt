###############################################################################
#
# Copyright (c) 2016-19, Lawrence Livermore National Security, LLC.
#
# Produced at the Lawrence Livermore National Laboratory
#
# LLNL-CODE-689114
#
# All rights reserved.
#
# This file is part of RAJA.
#
# For details about use and distribution, please read RAJA/LICENSE.
#
###############################################################################
###############################################################################
# Copyright (c) 2018,2019 Advanced Micro Devices, Inc.
###############################################################################

if (ENABLE_CUDA OR ENABLE_HIP)
  raja_add_test(
    NAME test-chai-nested
    SOURCES chai-nested.cpp)
endif(ENABLE_CUDA OR ENABLE_HIP)

raja_add_test(
  NAME test-chai-policy
  SOURCES chai-policy-tests.cpp)

raja_add_test(
  NAME test-chai
  SOURCES chai-tests.cpp)
