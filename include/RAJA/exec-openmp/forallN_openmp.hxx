//AUTOGENERATED BY gen_forallN_generic.py
/*
 * Copyright (c) 2016, Lawrence Livermore National Security, LLC.
 * Produced at the Lawrence Livermore National Laboratory.
 *
 * All rights reserved.
 *
 * This source code cannot be distributed without permission and
 * further review from Lawrence Livermore National Laboratory.
 */
  
#ifndef RAJA_forallN_openmp_HXX__
#define RAJA_forallN_openmp_HXX__

#include<RAJA/config.hxx>
#include<RAJA/int_datatypes.hxx>

namespace RAJA {



/******************************************************************
 *  ForallN OpenMP Parallel Region policies
 ******************************************************************/

// Begin OpenMP Parallel Region
struct Forall2_OMP_Parallel_Tag {};
template<typename NEXT=Forall2_Execute>
struct Forall2_OMP_Parallel {
  typedef Forall2_OMP_Parallel_Tag PolicyTag;
  typedef NEXT NextPolicy;
};

// Begin OpenMP Parallel Region
struct Forall3_OMP_Parallel_Tag {};
template<typename NEXT=Forall3_Execute>
struct Forall3_OMP_Parallel {
  typedef Forall3_OMP_Parallel_Tag PolicyTag;
  typedef NEXT NextPolicy;
};

// Begin OpenMP Parallel Region
struct Forall4_OMP_Parallel_Tag {};
template<typename NEXT=Forall4_Execute>
struct Forall4_OMP_Parallel {
  typedef Forall4_OMP_Parallel_Tag PolicyTag;
  typedef NEXT NextPolicy;
};

// Begin OpenMP Parallel Region
struct Forall5_OMP_Parallel_Tag {};
template<typename NEXT=Forall5_Execute>
struct Forall5_OMP_Parallel {
  typedef Forall5_OMP_Parallel_Tag PolicyTag;
  typedef NEXT NextPolicy;
};


/******************************************************************
 *  forallN Executor OpenMP auto-collapse rules
 ******************************************************************/

// OpenMP Executor with collapse(2) for omp_parallel_for_exec
template<>
class Forall2Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

#pragma omp parallel for schedule(static) collapse(2)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          body(i, j);
      } } 
    }
};

// OpenMP Executor with collapse(2) for omp_for_nowait_exec
template<>
class Forall2Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

#pragma omp for schedule(static) collapse(2) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          body(i, j);
      } } 
    }
};


// OpenMP Executor with collapse(2) for omp_parallel_for_exec
template<typename POLICY_K, typename TK>
class Forall3Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, POLICY_K, RAJA::RangeSegment, RAJA::RangeSegment, TK> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, TK const &is_k, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

#pragma omp parallel for schedule(static) collapse(2)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          RAJA::forall<POLICY_K>(is_k, [=](Index_type k){
            body(i, j, k);
          });
      } } 
    }
};

// OpenMP Executor with collapse(3) for omp_parallel_for_exec
template<>
class Forall3Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

#pragma omp parallel for schedule(static) collapse(3)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            body(i, j, k);
      } } } 
    }
};

// OpenMP Executor with collapse(2) for omp_for_nowait_exec
template<typename POLICY_K, typename TK>
class Forall3Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, POLICY_K, RAJA::RangeSegment, RAJA::RangeSegment, TK> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, TK const &is_k, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

#pragma omp for schedule(static) collapse(2) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          RAJA::forall<POLICY_K>(is_k, [=](Index_type k){
            body(i, j, k);
          });
      } } 
    }
};

// OpenMP Executor with collapse(3) for omp_for_nowait_exec
template<>
class Forall3Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

#pragma omp for schedule(static) collapse(3) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            body(i, j, k);
      } } } 
    }
};


// OpenMP Executor with collapse(2) for omp_parallel_for_exec
template<typename POLICY_K, typename POLICY_L, typename TK, typename TL>
class Forall4Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, POLICY_K, POLICY_L, RAJA::RangeSegment, RAJA::RangeSegment, TK, TL> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, TK const &is_k, TL const &is_l, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

#pragma omp parallel for schedule(static) collapse(2)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          exec(is_k, is_l, [=](Index_type k, Index_type l){
            body(i, j, k, l);
          });
      } } 
    }

  private:
    Forall2Executor<POLICY_K, POLICY_L, TK, TL> exec;
};

// OpenMP Executor with collapse(3) for omp_parallel_for_exec
template<typename POLICY_L, typename TL>
class Forall4Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, POLICY_L, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, TL> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, TL const &is_l, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

#pragma omp parallel for schedule(static) collapse(3)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            RAJA::forall<POLICY_L>(is_l, [=](Index_type l){
              body(i, j, k, l);
            });
      } } } 
    }
};

// OpenMP Executor with collapse(4) for omp_parallel_for_exec
template<>
class Forall4Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, RAJA::RangeSegment const &is_l, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

      Index_type const l_start = is_l.getBegin();
      Index_type const l_end   = is_l.getEnd();

#pragma omp parallel for schedule(static) collapse(4)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            for(Index_type l = l_start;l < l_end;++ l){
              body(i, j, k, l);
      } } } } 
    }
};

// OpenMP Executor with collapse(2) for omp_for_nowait_exec
template<typename POLICY_K, typename POLICY_L, typename TK, typename TL>
class Forall4Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, POLICY_K, POLICY_L, RAJA::RangeSegment, RAJA::RangeSegment, TK, TL> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, TK const &is_k, TL const &is_l, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

#pragma omp for schedule(static) collapse(2) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          exec(is_k, is_l, [=](Index_type k, Index_type l){
            body(i, j, k, l);
          });
      } } 
    }

  private:
    Forall2Executor<POLICY_K, POLICY_L, TK, TL> exec;
};

// OpenMP Executor with collapse(3) for omp_for_nowait_exec
template<typename POLICY_L, typename TL>
class Forall4Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, POLICY_L, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, TL> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, TL const &is_l, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

#pragma omp for schedule(static) collapse(3) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            RAJA::forall<POLICY_L>(is_l, [=](Index_type l){
              body(i, j, k, l);
            });
      } } } 
    }
};

// OpenMP Executor with collapse(4) for omp_for_nowait_exec
template<>
class Forall4Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, RAJA::RangeSegment const &is_l, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

      Index_type const l_start = is_l.getBegin();
      Index_type const l_end   = is_l.getEnd();

#pragma omp for schedule(static) collapse(4) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            for(Index_type l = l_start;l < l_end;++ l){
              body(i, j, k, l);
      } } } } 
    }
};


// OpenMP Executor with collapse(2) for omp_parallel_for_exec
template<typename POLICY_K, typename POLICY_L, typename POLICY_M, typename TK, typename TL, typename TM>
class Forall5Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, POLICY_K, POLICY_L, POLICY_M, RAJA::RangeSegment, RAJA::RangeSegment, TK, TL, TM> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, TK const &is_k, TL const &is_l, TM const &is_m, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

#pragma omp parallel for schedule(static) collapse(2)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          exec(is_k, is_l, is_m, [=](Index_type k, Index_type l, Index_type m){
            body(i, j, k, l, m);
          });
      } } 
    }

  private:
    Forall3Executor<POLICY_K, POLICY_L, POLICY_M, TK, TL, TM> exec;
};

// OpenMP Executor with collapse(3) for omp_parallel_for_exec
template<typename POLICY_L, typename POLICY_M, typename TL, typename TM>
class Forall5Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, POLICY_L, POLICY_M, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, TL, TM> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, TL const &is_l, TM const &is_m, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

#pragma omp parallel for schedule(static) collapse(3)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            exec(is_l, is_m, [=](Index_type l, Index_type m){
              body(i, j, k, l, m);
            });
      } } } 
    }

  private:
    Forall2Executor<POLICY_L, POLICY_M, TL, TM> exec;
};

// OpenMP Executor with collapse(4) for omp_parallel_for_exec
template<typename POLICY_M, typename TM>
class Forall5Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, POLICY_M, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, TM> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, RAJA::RangeSegment const &is_l, TM const &is_m, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

      Index_type const l_start = is_l.getBegin();
      Index_type const l_end   = is_l.getEnd();

#pragma omp parallel for schedule(static) collapse(4)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            for(Index_type l = l_start;l < l_end;++ l){
              RAJA::forall<POLICY_M>(is_m, [=](Index_type m){
                body(i, j, k, l, m);
              });
      } } } } 
    }
};

// OpenMP Executor with collapse(5) for omp_parallel_for_exec
template<>
class Forall5Executor<RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::omp_parallel_for_exec, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, RAJA::RangeSegment const &is_l, RAJA::RangeSegment const &is_m, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

      Index_type const l_start = is_l.getBegin();
      Index_type const l_end   = is_l.getEnd();

      Index_type const m_start = is_m.getBegin();
      Index_type const m_end   = is_m.getEnd();

#pragma omp parallel for schedule(static) collapse(5)
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            for(Index_type l = l_start;l < l_end;++ l){
              for(Index_type m = m_start;m < m_end;++ m){
                body(i, j, k, l, m);
      } } } } } 
    }
};

// OpenMP Executor with collapse(2) for omp_for_nowait_exec
template<typename POLICY_K, typename POLICY_L, typename POLICY_M, typename TK, typename TL, typename TM>
class Forall5Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, POLICY_K, POLICY_L, POLICY_M, RAJA::RangeSegment, RAJA::RangeSegment, TK, TL, TM> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, TK const &is_k, TL const &is_l, TM const &is_m, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

#pragma omp for schedule(static) collapse(2) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          exec(is_k, is_l, is_m, [=](Index_type k, Index_type l, Index_type m){
            body(i, j, k, l, m);
          });
      } } 
    }

  private:
    Forall3Executor<POLICY_K, POLICY_L, POLICY_M, TK, TL, TM> exec;
};

// OpenMP Executor with collapse(3) for omp_for_nowait_exec
template<typename POLICY_L, typename POLICY_M, typename TL, typename TM>
class Forall5Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, POLICY_L, POLICY_M, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, TL, TM> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, TL const &is_l, TM const &is_m, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

#pragma omp for schedule(static) collapse(3) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            exec(is_l, is_m, [=](Index_type l, Index_type m){
              body(i, j, k, l, m);
            });
      } } } 
    }

  private:
    Forall2Executor<POLICY_L, POLICY_M, TL, TM> exec;
};

// OpenMP Executor with collapse(4) for omp_for_nowait_exec
template<typename POLICY_M, typename TM>
class Forall5Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, POLICY_M, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, TM> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, RAJA::RangeSegment const &is_l, TM const &is_m, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

      Index_type const l_start = is_l.getBegin();
      Index_type const l_end   = is_l.getEnd();

#pragma omp for schedule(static) collapse(4) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            for(Index_type l = l_start;l < l_end;++ l){
              RAJA::forall<POLICY_M>(is_m, [=](Index_type m){
                body(i, j, k, l, m);
              });
      } } } } 
    }
};

// OpenMP Executor with collapse(5) for omp_for_nowait_exec
template<>
class Forall5Executor<RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::omp_for_nowait_exec, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment, RAJA::RangeSegment> {
  public:  
    template<typename BODY>
    inline void operator()(RAJA::RangeSegment const &is_i, RAJA::RangeSegment const &is_j, RAJA::RangeSegment const &is_k, RAJA::RangeSegment const &is_l, RAJA::RangeSegment const &is_m, BODY body) const {
      Index_type const i_start = is_i.getBegin();
      Index_type const i_end   = is_i.getEnd();

      Index_type const j_start = is_j.getBegin();
      Index_type const j_end   = is_j.getEnd();

      Index_type const k_start = is_k.getBegin();
      Index_type const k_end   = is_k.getEnd();

      Index_type const l_start = is_l.getBegin();
      Index_type const l_end   = is_l.getEnd();

      Index_type const m_start = is_m.getBegin();
      Index_type const m_end   = is_m.getEnd();

#pragma omp for schedule(static) collapse(5) nowait
      for(Index_type i = i_start;i < i_end;++ i){
        for(Index_type j = j_start;j < j_end;++ j){
          for(Index_type k = k_start;k < k_end;++ k){
            for(Index_type l = l_start;l < l_end;++ l){
              for(Index_type m = m_start;m < m_end;++ m){
                body(i, j, k, l, m);
      } } } } } 
    }
};



/******************************************************************
 *  forallN_policy(), OpenMP Parallel Region execution
 ******************************************************************/


/*!
 * \brief OpenMP Parallel Region Section policy function.
 */
template<typename POLICY, typename PolicyI, typename PolicyJ, typename TI, typename TJ, typename BODY>
RAJA_INLINE void forall2_policy(Forall2_OMP_Parallel_Tag, TI const &is_i, TJ const &is_j, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // create OpenMP Parallel Region
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // execute the next policy
    forall2_policy<NextPolicy, PolicyI, PolicyJ>(NextPolicyTag(), is_i, is_j, body);
  }
}


/*!
 * \brief OpenMP Parallel Region Section policy function.
 */
template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename TI, typename TJ, typename TK, typename BODY>
RAJA_INLINE void forall3_policy(Forall3_OMP_Parallel_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // create OpenMP Parallel Region
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // execute the next policy
    forall3_policy<NextPolicy, PolicyI, PolicyJ, PolicyK>(NextPolicyTag(), is_i, is_j, is_k, body);
  }
}


/*!
 * \brief OpenMP Parallel Region Section policy function.
 */
template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename TI, typename TJ, typename TK, typename TL, typename BODY>
RAJA_INLINE void forall4_policy(Forall4_OMP_Parallel_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // create OpenMP Parallel Region
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // execute the next policy
    forall4_policy<NextPolicy, PolicyI, PolicyJ, PolicyK, PolicyL>(NextPolicyTag(), is_i, is_j, is_k, is_l, body);
  }
}


/*!
 * \brief OpenMP Parallel Region Section policy function.
 */
template<typename POLICY, typename PolicyI, typename PolicyJ, typename PolicyK, typename PolicyL, typename PolicyM, typename TI, typename TJ, typename TK, typename TL, typename TM, typename BODY>
RAJA_INLINE void forall5_policy(Forall5_OMP_Parallel_Tag, TI const &is_i, TJ const &is_j, TK const &is_k, TL const &is_l, TM const &is_m, BODY body){
  typedef typename POLICY::NextPolicy            NextPolicy;
  typedef typename POLICY::NextPolicy::PolicyTag NextPolicyTag;

  // create OpenMP Parallel Region
#ifdef _OPENMP
#pragma omp parallel
#endif
  {
    // execute the next policy
    forall5_policy<NextPolicy, PolicyI, PolicyJ, PolicyK, PolicyL, PolicyM>(NextPolicyTag(), is_i, is_j, is_k, is_l, is_m, body);
  }
}



} // namespace RAJA
  
#endif
