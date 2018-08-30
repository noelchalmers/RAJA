#!/bin/bash

##
## Copyright (c) 2017-18, Lawrence Livermore National Security, LLC.
##
## Produced at the Lawrence Livermore National Laboratory.
##
## LLNL-CODE-738930
##
## All rights reserved.
##
## This file is part of the RAJA Performance Suite.
##
## For details about use and distribution, please read raja-perfsuite/LICENSE.
##

rm -rf build_ubuntu_hipcc_clang-coral-2018.04.28 >/dev/null
mkdir build_ubuntu_hipcc_clang-coral-2018.04.28 && cd build_ubuntu_hipcc_clang-coral-2018.04.28

#module load cmake/3.7.2

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -C ../host-configs/hipcc_clang_coral_2018_08_28.cmake \
  -DENABLE_OPENMP=Off \
  -DENABLE_HIP=On \
  "$@" \
  ..
