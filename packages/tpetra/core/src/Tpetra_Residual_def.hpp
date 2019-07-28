/*
//@HEADER
// ***********************************************************************
//
//       Ifpack2: Templated Object-Oriented Algebraic Preconditioner Package
//                 Copyright (2009) Sandia Corporation
//
// Under terms of Contract DE-AC04-94AL85000, there is a non-exclusive
// license for use of this work by or on behalf of the U.S. Government.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// ***********************************************************************
//@HEADER
*/

#ifndef TPETRA_RESIDUAL_DEF_HPP
#define TPETRA_RESIDUAL_DEF_HPP

#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Operator.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_withLocalAccess_MultiVector.hpp"
#include "Tpetra_Export_decl.hpp"
#include "Tpetra_Import_decl.hpp"
#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "Teuchos_Assert.hpp"
#include <type_traits>

namespace Tpetra {
namespace Details {

/// \brief Functor for computing W := B - A*X and norm(W).
///
/// This is an implementation detail of
/// localResidualVectorAndNormSquared, which in turn is an
/// implementation detail of Residual.
template<class WVector,
         class BVector,
         class AMatrix,
         class XVector>
struct ResidualFunctor {
  static_assert (static_cast<int> (WVector::Rank) == 1,
                 "WVector must be a rank 1 View.");
  static_assert (static_cast<int> (BVector::Rank) == 1,
                 "BVector must be a rank 1 View.");
  static_assert (static_cast<int> (XVector::Rank) == 1,
                 "XVector must be a rank 1 View.");

  using execution_space = typename AMatrix::execution_space;
  using LO = typename AMatrix::non_const_ordinal_type;
  using team_policy = typename Kokkos::TeamPolicy<execution_space>;
  using team_member = typename team_policy::member_type;
  using IPST = Kokkos::InnerProductSpaceTraits<residual_value_type>;
  using norm_type = typename IPST::val_type;

  WVector m_w;
  BVector m_b;
  AMatrix m_A;
  XVector m_x;

  const LO rows_per_team;

  ResidualFunctor (const WVector& w,
                   const BVector& b,
                   const AMatrix& A,
                   const XVector& x,
                   const int rowsPerTeam) :
    m_w (w),
    m_b (b),
    m_A (A),
    m_x (x),
    rows_per_team (rowsPerTeam)
  {
    const size_t numRows = m_A.numRows ();
    const size_t numCols = m_A.numCols ();

    TEUCHOS_ASSERT( m_w.extent (0) == m_d.extent (0) );
    TEUCHOS_ASSERT( m_w.extent (0) == m_b.extent (0) );
    TEUCHOS_ASSERT( numRows == size_t (m_w.extent (0)) );
    TEUCHOS_ASSERT( numCols <= size_t (m_x.extent (0)) );
  }

  // Residual vector only; no norm
  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member& dev) const
  {
    using residual_value_type = typename BVector::non_const_value_type;
    using KAT = Kokkos::ArithTraits<residual_value_type>;

    Kokkos::parallel_for
      (Kokkos::TeamThreadRange (dev, 0, rows_per_team),
       [&] (const LO loop) {
         const LO lclRow =
           static_cast<LO> (dev.league_rank ()) * rows_per_team + loop;
         if (lclRow >= m_A.numRows ()) {
           return;
         }
         const auto A_row = m_A.rowConst(lclRow);
         const LO row_length = static_cast<LO> (A_row.length);
         residual_value_type A_x = KAT::zero ();

         Kokkos::parallel_reduce
           (Kokkos::ThreadVectorRange (dev, row_length),
            [&] (const LO iEntry, residual_value_type& lsum) {
              const auto A_val = A_row.value(iEntry);
              lsum += A_val * m_x(A_row.colidx(iEntry));
            }, A_x);

         Kokkos::single (Kokkos::PerThread (dev),
           [&] () {
             m_w(lclRow) = m_b(lclRow) - A_x;
           });
       });
  }

  // Residual vector + its norm (squared)
  KOKKOS_INLINE_FUNCTION
  void operator() (const team_member& dev, norm_type& nrmSqrd) const
  {
    using residual_value_type = typename BVector::non_const_value_type;

    norm_type lclNrmSqrd = Kokkos::ArithTraits<norm_type>::zero ();
    Kokkos::parallel_reduce
      (Kokkos::TeamThreadRange (dev, 0, rows_per_team),
       [&] (const LO loop, norm_type& myNrmSqrd) {
         const LO lclRow =
           static_cast<LO> (dev.league_rank ()) * rows_per_team + loop;
         if (lclRow >= m_A.numRows ()) {
           return;
         }
         const auto A_row = m_A.rowConst(lclRow);
         const LO row_length = static_cast<LO> (A_row.length);
         residual_value_type A_x =
           Kokkos::ArithTraits<residual_value_type>::zero ();

         Kokkos::parallel_reduce
           (Kokkos::ThreadVectorRange (dev, row_length),
            [&] (const LO iEntry, residual_value_type& lsum) {
              const auto A_val = A_row.value(iEntry);
              lsum += A_val * m_x(A_row.colidx(iEntry));
            }, A_x);

         Kokkos::single (Kokkos::PerThread (dev),
           [&] () {
             m_w(lclRow) = m_b(lclRow) - A_x;
             myNrmSqrd += IPST::norm (m_w(lclRow));
           });
       }, lclNrmSqrd);
    nrmSqrd += lclNrmSqrd;
  }
};

template<class ExecutionSpace>
int64_t
residual_launch_parameters (int64_t numRows,
                            int64_t nnz,
                            int64_t rows_per_thread,
                            int& team_size,
                            int& vector_length)
{
  using execution_space = typename ExecutionSpace::execution_space;

  int64_t rows_per_team;
  int64_t nnz_per_row = nnz/numRows;

  if (nnz_per_row < 1) {
    nnz_per_row = 1;
  }

  if (vector_length < 1) {
    vector_length = 1;
    while (vector_length<32 && vector_length*6 < nnz_per_row) {
      vector_length *= 2;
    }
  }

  // Determine rows per thread
  if (rows_per_thread < 1) {
#ifdef KOKKOS_ENABLE_CUDA
    if (std::is_same<Kokkos::Cuda, execution_space>::value) {
      rows_per_thread = 1;
    }
    else
#endif
    {
      if (nnz_per_row < 20 && nnz > 5000000) {
        rows_per_thread = 256;
      }
      else {
        rows_per_thread = 64;
      }
    }
  }

#ifdef KOKKOS_ENABLE_CUDA
  if (team_size < 1) {
    if (std::is_same<Kokkos::Cuda,execution_space>::value) {
      team_size = 256/vector_length;
    }
    else {
      team_size = 1;
    }
  }
#endif

  rows_per_team = rows_per_thread * team_size;

  if (rows_per_team < 0) {
    int64_t nnz_per_team = 4096;
    int64_t conc = execution_space::concurrency ();
    while ((conc * nnz_per_team * 4 > nnz) &&
           (nnz_per_team > 256)) {
      nnz_per_team /= 2;
    }
    rows_per_team = (nnz_per_team + nnz_per_row - 1) / nnz_per_row;
  }

  return rows_per_team;
}

// W := B - A*X; return square of two-norm of W.
template<class ExecutionSpace,
         class WVector,
         class BVector,
         class AMatrix,
         class XVector>
typename Kokkos::InnerProductSpaceTraits<typename WVector::non_const_value_type>::norm_type
localResidualVectorAndNormSquared (const char kernel_label[],
                                   ExecutionSpace execSpace,
                                   WVector w,
                                   BVector b,
                                   AMatrix A,
                                   XVector x)
{
  using execution_space = typename AMatrix::execution_space;

  if (A.numRows () == 0) {
    return;
  }

  int team_size = -1;
  int vector_length = -1;
  int64_t rows_per_thread = -1;

  const int64_t rows_per_team =
    residual_launch_parameters<execution_space>
      (A.numRows (), A.nnz (), rows_per_thread, team_size, vector_length);
  int64_t worksets = (b.extent (0) + rows_per_team - 1) / rows_per_team;

  using Kokkos::Dynamic;
  using Kokkos::Schedule;
  using Kokkos::TeamPolicy;
  using policy_type = TeamPolicy<execution_space, Schedule<Dynamic>>;
  policy_type policy = (team_size < 0) ?
    policy_type (execSpace, worksets, Kokkos::AUTO, vector_length) :
    policy_type (execSpace, worksets, team_size, vector_length);

  // Canonicalize template arguments to avoid redundant instantiations.
  using w_vec_type = Kokkos::View<typename WVector::non_const_data_type,
                                  typename WVector::array_layout,
                                  typename WVector::device_type,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using b_vec_type = Kokkos::View<typename BVector::const_data_type,
                                  typename BVector::array_layout,
                                  typename BVector::device_type,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using matrix_type = AMatrix;
  using x_vec_type = Kokkos::View<typename XVector::const_data_type,
                                  typename XVector::array_layout,
                                  typename XVector::device_type,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using functor_type =
    ResidualFunctor<w_vec_type, b_vec_type, matrix_type, x_vec_type>;
  functor_type func (w, b, A, x, rows_per_team);
  using norm_type = typename functor_type::norm_type;
  norm_type lclNrmSqrd = Kokkos::ArithTraits<norm_type>::zero ();
  Kokkos::parallel_reduce (kernel_label, policy, func, lclNrmSqrd);
  return lclNrmSqrd;
}

// W := B - A*X.
template<class ExecutionSpace,
         class WVector,
         class BVector,
         class AMatrix,
         class XVector>
void
localResidualVector (const char kernel_label[],
                     ExecutionSpace execSpace,
                     WVector w,
                     BVector b,
                     AMatrix A,
                     XVector x)
{
  if (A.numRows () == 0) {
    return;
  }

  int team_size = -1;
  int vector_length = -1;
  int64_t rows_per_thread = -1;

  const int64_t rows_per_team =
    residual_launch_parameters<ExecutionSpace>
      (A.numRows (), A.nnz (), rows_per_thread, team_size, vector_length);
  int64_t worksets = (b.extent (0) + rows_per_team - 1) / rows_per_team;

  using Kokkos::Dynamic;
  using Kokkos::Schedule;
  using Kokkos::TeamPolicy;
  using policy_type = TeamPolicy<ExecutionSpace, Schedule<Dynamic>>;
  policy_type policy = (team_size < 0) ?
    policy_type (execSpace, worksets, Kokkos::AUTO, vector_length) :
    policy_type (execSpace, worksets, team_size, vector_length);

  // Canonicalize template arguments to avoid redundant instantiations.
  using w_vec_type = Kokkos::View<typename WVector::non_const_data_type,
                                  typename WVector::array_layout,
                                  typename WVector::device_type,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using b_vec_type = Kokkos::View<typename BVector::const_data_type,
                                  typename BVector::array_layout,
                                  typename BVector::device_type,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using matrix_type = AMatrix;
  using x_vec_type = Kokkos::View<typename XVector::const_data_type,
                                  typename XVector::array_layout,
                                  typename XVector::device_type,
                                  Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using functor_type =
    ResidualFunctor<w_vec_type, b_vec_type, matrix_type, x_vec_type>;
  functor_type func (w, b, A, x, rows_per_team);
  Kokkos::parallel_for (kernel_label, policy, func);
}

} // namespace Details

template<class SC, class LO, class GO, class NT>
Residual<SC, LO, GO, NT>::
Residual (const Teuchos::RCP<const operator_type>& A)
{
  setMatrix (A);
}

template<class SC, class LO, class GO, class NT>
void
Residual<SC, LO, GO, NT>::
setMatrix (const Teuchos::RCP<const operator_type>& A)
{
  if (A_op_.get () != A.get ()) {
    A_op_ = A;
    // Realloc'd on demand
    X_colMap_ = std::unique_ptr<vector_type> (nullptr);

    using Teuchos::rcp_dynamic_cast;
    Teuchos::RCP<const crs_matrix_type> A_crs =
      rcp_dynamic_cast<const crs_matrix_type> (A);
    if (A_crs.is_null ()) {
      A_crs_ = Teuchos::null;
      imp_ = Teuchos::null;
      exp_ = Teuchos::null;
    }
    else {
      TEUCHOS_ASSERT( A_crs->isFillComplete () );
      A_crs_ = A_crs;
      auto G = A_crs->getCrsGraph ();
      imp_ = G->getImporter ();
      exp_ = G->getExporter ();
    }
  }
}

template<class SC, class LO, class GO, class NT>
Teuchos::ArrayView<typename Vector<SC, LO, GO, NT>::mag_type>
Residual<SC, LO, GO, NT>::
computeWithNorms (multivector_type& W,
                  multivector_type& B,
                  multivector_type& X)
{
  const char kernel_label[] = "Tpetra::Residual::computeWithNorms";
  
  if (canFuse (B)) {
    using Teuchos::RCP;
    // "nonconst" here has no effect other than on the return type.
    RCP<vector_type> W_vec = W.getVectorNonConst (0);
    RCP<vector_type> B_vec = B.getVectorNonConst (0);
    RCP<vector_type> X_vec = X.getVectorNonConst (0);
    TEUCHOS_ASSERT( ! A_crs_.is_null () );
    return fusedCaseWithNorms (kernel_label, *W_vec, *B_vec, *A_crs_, *X_vec);
  }
  else {
    TEUCHOS_ASSERT( ! A_op_.is_null () );
    return unfusedCaseWithNorms (kernel_label, W, B, *A_op_, X);
  }
}

template<class SC, class LO, class GO, class NT>
void
Residual<SC, LO, GO, NT>::
computeWithoutNorms (multivector_type& W,
                     multivector_type& B,
                     multivector_type& X)
{
  const char kernel_label[] = "Tpetra::Residual::computeWithoutNorms";
  
  if (canFuse (B)) {
    using Teuchos::RCP;
    // "nonconst" here has no effect other than on the return type.
    RCP<vector_type> W_vec = W.getVectorNonConst (0);
    RCP<vector_type> B_vec = B.getVectorNonConst (0);
    RCP<vector_type> X_vec = X.getVectorNonConst (0);
    TEUCHOS_ASSERT( ! A_crs_.is_null () );
    fusedCaseWithoutNorms (kernel_label, *W_vec, *B_vec,
                           *A_crs_, *X_vec);
  }
  else {
    TEUCHOS_ASSERT( ! A_op_.is_null () );
    unfusedCaseWithoutNorms (kernel_label, W, B, *A_op_, X);
  }
}

template<class SC, class LO, class GO, class NT>
typename Residual<SC, LO, GO, NT>::vector_type
Residual<SC, LO, GO, NT>::
importVector (vector_type& X_domMap)
{
  if (imp_.is_null ()) {
    return X_domMap;
  }
  else if (! A_crs_.is_null ()) {
    auto X_colMap_mv = A_crs_->getColumnMapMultiVector (X_domMap);
    TEUCHOS_ASSERT( ! X_colMap_mv.is_null () );
    vector_type X_colMap (*X_colMap_mv, 0);
    X_colMap.doImport (X_domMap, *imp_, Tpetra::REPLACE);
    return X_colMap;
    
    // if (X_colMap_.get () == nullptr) {
    //   using V = vector_type;
    //   X_colMap_ = std::unique_ptr<V> (new V (imp_->getTargetMap ()));
    // }
    // X_colMap_->doImport (X_domMap, *imp_, Tpetra::REPLACE);
    // return *X_colMap_;
  }
}

template<class SC, class LO, class GO, class NT>
bool
Residual<SC, LO, GO, NT>::
canFuse (const multivector_type& B) const
{
  return B.getNumVectors () == size_t (1) &&
    ! A_crs_.is_null () &&
    exp_.is_null ();
}

template<class SC, class LO, class GO, class NT>
Teuchos::ArrayView<const typename Vector<SC, LO, GO, NT>::mag_type>
Residual<SC, LO, GO, NT>::
unfusedCaseWithNorms (const char /* kernel_label */ [],
                      multivector_type& W,
                      multivector_type& B,
                      const operator_type& A,
                      multivector_type& X)
{
  const SC one = Teuchos::ScalarTraits<SC>::one ();
  Tpetra::deep_copy (W, B);
  A.apply (X, W, Teuchos::NO_TRANS, -one, one);

  if (size_t (norms_.size ()) != W.getNumVectors ()) {
    std::vector<norm_type> newNorms (W.getNumVectors ());
    std::swap (norms_, newNorms);
  }
  using Teuchos::ArrayView;
  ArrayView<norm_type> norms_av (norms_.data (), norms.size ());
  W.norm2 (norms_av);
  return ArrayView<const norm_type> (norms_av);
}

template<class SC, class LO, class GO, class NT>
void
Residual<SC, LO, GO, NT>::
unfusedCaseWithoutNorms (const char /* kernel_label */ [],
                         multivector_type& W,
                         multivector_type& B,
                         const operator_type& A,
                         multivector_type& X)
{
  const SC one = Teuchos::ScalarTraits<SC>::one ();
  Tpetra::deep_copy (W, B);
  A.apply (X, W, Teuchos::NO_TRANS, -one, one);
}

template<class SC, class LO, class GO, class NT>
Teuchos::ArrayView<const typename Vector<SC, LO, GO, NT>::mag_type>
Residual<SC, LO, GO, NT>::
fusedCaseWithNorms (const char kernel_label[],
                    vector_type& W,
                    vector_type& B,
                    const crs_matrix_type& A,
                    vector_type& X)
{
  vector_type& X_colMap = importVector (X);

  // Only need these aliases because we lack C++14 generic lambdas.
  using Tpetra::with_local_access_function_argument_type;
  using ro_lcl_vec_type =
    with_local_access_function_argument_type<
      decltype (readOnly (B))>;
  using wo_lcl_vec_type =
    with_local_access_function_argument_type<
      decltype (writeOnly (B))>;

  using Tpetra::withLocalAccess;
  using Tpetra::readOnly;
  using Tpetra::writeOnly;
  auto A_lcl = A.getLocalMatrix ();
  
  using Details::localResidualVectorAndNormSquared;
  using norm_type = typename vector_type::mag_type;

  norm_type lclNrmSqrd = Kokkos::ArithTraits<norm_type>::zero ();
  withLocalAccess
    ([&] (const wo_lcl_vec_type& W_lcl,
          const ro_lcl_vec_type& B_lcl,
          const ro_lcl_vec_type& X_lcl) {
       lclNrmSqrd = localResidualVectorAndNormSquared (kernel_label,
                                                       W_lcl, B_lcl,
                                                       A_lcl, X_lcl);
     },
     writeOnly (W),
     readOnly (B),
     readOnly (X_colMap));

  auto map = W.getMap ();
  auto comm = map.is_null () ? Teuchos::null : map->getComm ();
  norm_type gblNrmSqrd = lclNrmSqrd;
  if (! comm.is_null ()) {
    Teuchos::reduceAll (*comm, Teuchos::REDUCE_SUM, lclNrmSqrd,
                        Teuchos::outArg (gblNrmSqrd));
  }
  if (size_t (norms_.size ()) != size_t (1)) {
    std::vector<norm_type> newNorms (W.getNumVectors ());
    std::swap (norms_, newNorms);
  }
  norms_[0] = gblNrmSqrd;
  using Teuchos::ArrayView;
  return ArrayView<const norm_type> (norms_.data (), norms_.size ());
}

template<class SC, class LO, class GO, class NT>
void
Residual<SC, LO, GO, NT>::
fusedCaseWithoutNorms (const char kernel_label[],
                       vector_type& W,
                       vector_type& B,
                       const crs_matrix_type& A,
                       vector_type& X)
{
  vector_type& X_colMap = importVector (X);

  // Only need these aliases because we lack C++14 generic lambdas.
  using Tpetra::with_local_access_function_argument_type;
  using ro_lcl_vec_type =
    with_local_access_function_argument_type<
      decltype (readOnly (B))>;
  using wo_lcl_vec_type =
    with_local_access_function_argument_type<
      decltype (writeOnly (B))>;

  using Tpetra::withLocalAccess;
  using Tpetra::readOnly;
  using Tpetra::writeOnly;
  auto A_lcl = A.getLocalMatrix ();
  
  withLocalAccess
    ([&] (const wo_lcl_vec_type& W_lcl,
          const ro_lcl_vec_type& B_lcl,
          const ro_lcl_vec_type& X_lcl) {
       Details::localResidualVector (kernel_label, W_lcl,
                                     B_lcl, A_lcl, X_lcl);
   },
   writeOnly (W),
   readOnly (B),
   readOnly (X_colMap));
}

} // namespace Tpetra

//
// Explicit instantiation macro
//
// Must be expanded from within the Tpetra namespace!
//

#define TPETRA_RESIDUAL_INSTANT(SC,LO,GO,NT) \
  template class Residual<SC, LO, GO, NT>;

#endif // TPETRA_RESIDUAL_DEF_HPP
