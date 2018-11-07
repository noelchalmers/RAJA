###############################################################################
#
# Copyright (c) 2016-18, Lawrence Livermore National Security, LLC.
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

raja_add_test(
  NAME test-hip-nested
  SOURCES test-nested.cpp)

# raja_add_test(
#   NAME test-hip-scan
#   SOURCES test-scan.cpp)

raja_add_test(
  NAME test-hip-reduce-sum
  SOURCES test-reduce-sum.cpp)

raja_add_test(
  NAME test-hip-reduce-min
  SOURCES test-reduce-min.cpp)

raja_add_test(
  NAME test-hip-reduce-loc
  SOURCES test-reduce-loc.cpp)

raja_add_test(
  NAME test-hip-reduce-max
  SOURCES test-reduce-max.cpp)

raja_add_test(
  NAME test-hip-forall
  SOURCES test-forall.cpp)

raja_add_test(
  NAME test-hip-nested-strided
  SOURCES test-nested-strided.cpp)

raja_add_test(
  NAME test-hip-forall-view
  SOURCES test-forall-view.cpp)

raja_add_test(
  NAME test-hip-synchronize
  SOURCES test-synchronize.cpp)
