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

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include "RAJA/RAJA.hpp"
#include "RAJA/util/Timer.hpp"
/*
 *  Simd benchmark 5 - vector addition
 */

//#define ADD_ALIGN_HINT

#if defined(ADD_ALIGN_HINT)
#define VEC_ADD_BODY \
  z[i + j*N] = x[i + j*N] + y[i + j*N];
#else
#define VEC_ADD_BODY \
  c[i + j*N] = a[i + j*N] + b[i + j*N];
#endif


using realType = double;
using TFloat = realType * const RAJA_RESTRICT;

RAJA_INLINE
void vec_add_noVec(TFloat a, TFloat b, TFloat c, RAJA::Index_type N, RAJA::Index_type M) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

#pragma omp parallel for
  for(RAJA::Index_type j=0; j<M; j++){

    RAJA_NO_SIMD
    for(RAJA::Index_type i=0; i<N; ++i){
      VEC_ADD_BODY
     }
  }

}

RAJA_INLINE
void vec_add_native(TFloat a, TFloat b, TFloat c,  RAJA::Index_type N, RAJA::Index_type M) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

#pragma omp parallel for
  for(RAJA::Index_type j=0; j<M; j++){

    for(RAJA::Index_type i=0; i<N; ++i){
      VEC_ADD_BODY
     }
  }

}

RAJA_INLINE
void vec_add_simd(TFloat a, TFloat b, TFloat c, RAJA::Index_type N, RAJA::Index_type M) 
{  

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);
#endif

#pragma omp parallel for
  for(RAJA::Index_type j=0; j<M; j++){
    
    RAJA_SIMD    
    for(RAJA::Index_type i=0; i<N; ++i){
      VEC_ADD_BODY
     }
  }
  
}

template<typename POL>
RAJA_INLINE
void vec_add_RAJA(TFloat a, TFloat b, TFloat c,  RAJA::Index_type N, RAJA::Index_type M) 
{

#if defined(ADD_ALIGN_HINT)
  realType *x = RAJA::align_hint(a);
  realType *y = RAJA::align_hint(b);
  realType *z = RAJA::align_hint(c);  
#endif

  RAJA::kernel<POL>
    (RAJA::make_tuple(RAJA::RangeSegment(0, N), RAJA::RangeSegment(0, M)),  
     [=](int i, int j) {
      VEC_ADD_BODY
    });
  
}

void checkResult(TFloat b, RAJA::Index_type len);

int main(int argc, char *argv[])
{

  if(argc !=2 ){
  exit(-1);
  }

//
// Define vector length
//
  RAJA::Timer::ElapsedType runTime; 
  const RAJA::Index_type N = atoi(argv[1]); 
  const RAJA::Index_type M = 16;

#if defined(ADD_ALIGN_HINT)
  std::cout << "\n\nRAJA omp reduction benchmark with alignment hint...\n";
#else
  std::cout << "\n\nRAJA omp reduction product addition benchmark...\n";
#endif
  std::cout<<"No of entries "<<N<<" x "<<M<<"\n\n"<<std::endl;

  auto timer = RAJA::Timer();
  const RAJA::Index_type Niter = 50000;

  TFloat a = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*M*sizeof(realType));  
  TFloat b = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*M*sizeof(realType));  
  TFloat c = RAJA::allocate_aligned_type<realType>(RAJA::DATA_ALIGN, N*M*sizeof(realType));  
  
  //intialize memory
#pragma omp parallel for 
  for(int j=0; j<M; ++j){    
    for(int i=0; i<N; ++i){
      a[i + j*N] = 1;
      b[i + j*N] = 1;
      c[i + j*N] = 0.0;      
    }    
  }

  //---------------------------------------------------------
  std::cout<<"Native C - strictly sequential"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
#pragma omp parallel for 
    for(int j=0; j<M; ++j){    
      for(int i=0; i<N; ++i){
        c[i + j*N] = 0.0;      
      }    
    }    
    timer.start();
    vec_add_noVec(a, b, c, N, M);
    timer.stop();    
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, M);

  //---------------------------------------------------------
  std::cout<<"Native C - raw loop"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
#pragma omp parallel for 
  for(int j=0; j<M; ++j){    
    for(int i=0; i<N; ++i){
      c[i + j*N] = 0.0;      
    }    
  }    
    timer.start();
    vec_add_native(a, b, c, N, M);
    timer.stop();
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, M);

  //---------------------------------------------------------
  std::cout<<"Native C - with vectorization hint"<<std::endl;
  //---------------------------------------------------------
  for(RAJA::Index_type it = 0; it < Niter; ++it){
#pragma omp parallel for 
  for(int j=0; j<M; ++j){    
    for(int i=0; i<N; ++i){
      c[i + j*N] = 0.0;      
    }    
  }    
    timer.start();
    vec_add_simd(a, b, c, N, M);
    timer.stop();
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, M);
  

  //---------------------------------------------------------
  std::cout<<"RAJA - strictly sequential"<<std::endl;
  //---------------------------------------------------------
  using NESTED_EXEC_POL = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<0, RAJA::seq_exec,
          RAJA::statement::Lambda<0>
        >
      >  
    >;

  for(RAJA::Index_type it = 0; it < Niter; ++it){
#pragma omp parallel for 
    for(int j=0; j<M; ++j){    
      for(int i=0; i<N; ++i){
        c[i + j*N] = 0.0;      
      }    
    }    
    timer.start();
    vec_add_RAJA<NESTED_EXEC_POL>(a, b, c, N, M);
    timer.stop();
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, M);


  //---------------------------------------------------------
  std::cout<<"RAJA - raw loop"<<std::endl;
  //---------------------------------------------------------
  using NESTED_EXEC_POL_2 = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<0, RAJA::loop_exec,
          RAJA::statement::Lambda<0>
        >
      >  
    >;

  for(RAJA::Index_type it = 0; it < Niter; ++it){
#pragma omp parallel for 
    for(int j=0; j<M; ++j){    
      for(int i=0; i<N; ++i){
        c[i + j*N] = 0.0;      
      }    
    }    
    timer.start();
    vec_add_RAJA<NESTED_EXEC_POL_2>(a, b, c, N, M);
    timer.stop();
  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, M);

  //---------------------------------------------------------
  std::cout<<"RAJA - with vectorization hint"<<std::endl;
  //---------------------------------------------------------
  using NESTED_EXEC_POL_3 = 
    RAJA::KernelPolicy<
      RAJA::statement::For<1, RAJA::omp_parallel_for_exec,
        RAJA::statement::For<0, RAJA::simd_exec,
          RAJA::statement::Lambda<0>
        >
      >  
    >;
  for(RAJA::Index_type it = 0; it < Niter; ++it){
#pragma omp parallel for 
    for(int j=0; j<M; ++j){    
      for(int i=0; i<N; ++i){
        c[i + j*N] = 0.0;      
      }    
    }    
    timer.start();
    vec_add_RAJA<NESTED_EXEC_POL_3>(a, b, c, N, M);
    timer.stop();

  }
  runTime = timer.elapsed();
  timer.reset();
  std::cout<< "\trun time : " << runTime << " seconds" << std::endl;
  checkResult(c, M);
  //---------------------------------------------------------
  std::cout << "\n DONE!...\n";


}

void checkResult(TFloat c, RAJA::Index_type len){
  bool correct = true;
  for (RAJA::Index_type i = 0; i < len; i++) {
    if ( std::abs( c[i] - 2) > 1e-9 ) { correct = false; }
  }

  if ( correct ) {
    std::cout << "\t result -- PASS\n\n";
  } else {
    std::cout << "\t result -- FAIL\n\n";
  }

}
