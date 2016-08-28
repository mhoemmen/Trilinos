/*
// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
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
// Questions? Contact Michael A. Heroux (maherou@sandia.gov)
//
// ************************************************************************
// @HEADER
*/

/// \file MKL_TRSM.cpp
/// \brief Unit tests for KokkosSparse::Impl::Mkl::Trsm

#include "Teuchos_UnitTestHarness.hpp"
#include "Teuchos_TypeNameTraits.hpp"
#include "Kokkos_ArithTraits.hpp"
#include "Kokkos_Sparse_CrsMatrix.hpp"
#include "Kokkos_Sparse_impl_MKL.hpp"
#include "Kokkos_Sparse_impl_MKL_trsm.hpp"
#include <limits>

namespace { // (anonymous)

using std::endl;

// Reference implementation of sparse triangular solve with multiple
// vectors.  That is, solve A*X = B for X, where A is either upper or
// lower triangular.  We don't implement the transpose or conjugate
// transpose cases here.
template<class CrsMatrixType, class MultiVectorType>
void
TRSM (Teuchos::FancyOStream& out,
      MultiVectorType& X,
      const CrsMatrixType& A,
      MultiVectorType& B,
      const char uplo[],
      const char trans[],
      const char diag[])
{
  typedef typename Kokkos::Details::ArithTraits<typename CrsMatrixType::value_type>::val_type val_type;
  typedef Kokkos::Details::ArithTraits<val_type> KAT;
  typedef typename CrsMatrixType::ordinal_type LO;
  typedef typename CrsMatrixType::size_type offset_type;
  typedef Kokkos::pair<offset_type, offset_type> range_type;

  Teuchos::OSTab tab0 (out);
  out << "Reference TRSM implementation" << endl;
  Teuchos::OSTab tab1 (out);

  bool upper = false;
  if (uplo[0] == 'U' || uplo[0] == 'u') {
    upper = true;
  }
  else if (uplo[0] == 'L' || uplo[0] == 'l') {
    upper = false; // lower instead
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid uplo");
  }

  bool transpose = false;
  bool conjugate = false;
  if (trans[0] == 'T' || trans[0] == 't') {
    transpose = true;
    conjugate = false;
  }
  else if (trans[0] == 'C' || trans[0] == 'c') {
    transpose = true;
    conjugate = true;
  }
  else if (trans[0] == 'H' || trans[0] == 'h') {
    transpose = true;
    conjugate = true;
  }
  else if (trans[0] == 'N' || trans[0] == 'n') {
    transpose = false;
    conjugate = false;
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid trans");
  }

  TEUCHOS_TEST_FOR_EXCEPTION
    (transpose || conjugate, std::logic_error, "Transpose and conjugate "
     "transpose cases not implemented");

  bool implicitUnitDiag = false;
  if (diag[0] == 'U' || diag[0] == 'u') {
    implicitUnitDiag = true;
  }
  else if (diag[0] == 'N' || diag[0] == 'n') {
    implicitUnitDiag = false;
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION(true, std::invalid_argument, "Invalid diag");
  }

  out << "upper: " << (upper ? "true" : "false")
      << "transpose: " << (transpose ? "true" : "false")
      << ", conjugate: " << (conjugate ? "true" : "false")
      << ", implicitUnitDiag: " << (implicitUnitDiag ? "true" : "false")
      << endl;

  const LO lclNumRows = (A.graph.row_map.dimension_0 () == 0) ?
    static_cast<LO> (0) :
    static_cast<LO> (A.graph.row_map.dimension_0 () - 1);

  Kokkos::deep_copy (X, KAT::zero ());

  // Assume that the rows and columns have the same (local) indexing,
  // at least relative to the diagonal entry.

  if (upper) {
    // NOTE (mfh 23 Aug 2016) In the upper triangular case, we count
    // from N down to 1, to avoid an infinite loop if LO is unsigned.
    // (Unsigned numbers are ALWAYS >= 0.)
    for (LO lclRowPlusOne = lclNumRows;
         lclRowPlusOne > static_cast<LO> (0);
         --lclRowPlusOne) {
      const LO lclRow = lclRowPlusOne - static_cast<LO> (1);
      const range_type range (A.graph.row_map[lclRow],
                              A.graph.row_map[lclRow+1]);
      const LO numEnt = static_cast<LO> (range.second - range.first);
      auto lclColInds = Kokkos::subview (A.graph.entries, range);
      auto vals = Kokkos::subview (A.values, range);

      for (LO j = 0; j < static_cast<LO> (B.dimension_0 ()); ++j) {
        // Go through the row.  It doesn't matter if the matrix is upper
        // or lower triangular here; what matters is distinguishing the
        // diagonal entry.
        val_type X_ij = static_cast<val_type> (B(lclRow, j));
        val_type diagVal = KAT::one ();
        for (LO k = 0; k < numEnt; ++k) {
          if (lclColInds[k] == lclRow) {
            TEUCHOS_TEST_FOR_EXCEPT( implicitUnitDiag );
            diagVal = vals[k];
          }
          else {
            const LO lclCol = lclColInds[k];
            X_ij = X_ij - vals[k] * X(lclCol, j);
          }
        }

        // Update the output entry
        {
          const LO lclCol = lclRow;
          X(lclCol, j) = X_ij / diagVal;
        }
      } // for each column of B (and X)
    } // for each local row
  }
  else { // lower triangular
    for (LO lclRow = 0; lclRow < lclNumRows; ++lclRow) {
      const range_type range (A.graph.row_map[lclRow],
                              A.graph.row_map[lclRow+1]);
      const LO numEnt = static_cast<LO> (range.second - range.first);
      auto lclColInds = Kokkos::subview (A.graph.entries, range);
      auto vals = Kokkos::subview (A.values, range);

      for (LO j = 0; j < static_cast<LO> (B.dimension_0 ()); ++j) {
        // Go through the row.  It doesn't matter if the matrix is upper
        // or lower triangular here; what matters is distinguishing the
        // diagonal entry.
        val_type X_ij = static_cast<val_type> (B(lclRow, j));
        val_type diagVal = KAT::one ();
        for (LO k = 0; k < numEnt; ++k) {
          if (lclColInds[k] == lclRow) {
            TEUCHOS_TEST_FOR_EXCEPT( implicitUnitDiag );
            diagVal = vals[k];
          }
          else {
            const LO lclCol = lclColInds[k];
            X_ij = X_ij - vals[k] * X(lclCol, j);
          }
        }

        // Update the output entry
        {
          const LO lclCol = lclRow;
          X(lclCol, j) = X_ij / diagVal;
        }
      } // for each column of B (and X)
    } // for each local row
  } // upper or lower triangular
}

// Consider a real arrow matrix (arrow pointing down and right) with
// diagonal entries d and other nonzero entries 1.  Here is a 4 x 4
// example:
//
// [d     1]
// [  d   1]
// [    d 1]
// [1 1 1 d]
//
// Compute its LU factorization without pivoting, assuming that all
// the values exist:
//
// [1            ] [d        1      ]
// [    1        ] [   d     1      ]
// [        1    ] [      d  1      ]
// [1/d 1/d 1/d 1] [         d - 3/d]
//
// Generalize the pattern: the off-diagonal nonzero entries of the L
// factor are all 1/d, and the lower right entry of U is d - (n-1)/d,
// where the original matrix A is n by n.  If d is positive and big
// enough, say d >= 2n, then all diagonal entries of U will be
// sufficiently large for this factorization to make sense.
// Furthermore, if d is a power of 2, then 1/d is exact in binary
// floating-point arithmetic (if it doesn't overflow), as is (1/d +
// 1/d).  This lets us easily check our work.
//
// Suppose that we want to solve Ax=b for b = [1 2 ... n]^T.
// For c = Ux, first solve Lc = b:
//
// c = [1, 2, ..., n-1, n - n(n-1)/(2d)]^T
//
// and then solve Ux = c.  First,
//
// x_n = c_n / (d - (n-1)/d).
//
// Then, for k = 1, ..., n-1, dx_k + x_n = k, so
//
// x_k = (k - x_n) / d, for k = 1, ..., n-1.
//
// Now, multiply b through by d - (n-1)/d.  This completely avoids
// rounding error, as long as no quantities overflow.  To get the
// right answer, multiply both c and x through by the same scaling
// factor.

template<class SC, class LO, class OffsetType, class MultiVectorLayoutType>
void
testArrowMatrix (bool& success,
                 Teuchos::FancyOStream& out,
                 const LO numVecs)
{
  typedef typename Kokkos::Details::ArithTraits<SC>::val_type val_type;
  typedef Kokkos::Details::ArithTraits<val_type> KAT;
  typedef typename KAT::mag_type mag_type;
  typedef typename Kokkos::View<val_type*>::HostMirror::execution_space host_execution_space;
  typedef Kokkos::HostSpace host_memory_space;
  typedef Kokkos::Device<host_execution_space, host_memory_space> HDT;
  typedef KokkosSparse::CrsMatrix<SC, LO, HDT, void, OffsetType> local_matrix_type;
  typedef typename local_matrix_type::StaticCrsGraphType local_graph_type;
  typedef typename local_matrix_type::row_map_type::non_const_type row_offsets_type;
  typedef typename local_graph_type::entries_type::non_const_type col_inds_type;
  typedef typename local_matrix_type::values_type::non_const_type values_type;
  typedef Kokkos::View<val_type**, MultiVectorLayoutType, HDT> MV;

  const bool explicitlyStoreUnitDiagonalOfL = false;

  Teuchos::OSTab tab0 (out);
  out << "Test MKL TRSM with arrow matrix" << endl;
  Teuchos::OSTab tab1 (out);

  const LO lclNumRows = 8; // power of two (see above)
  const LO lclNumCols = lclNumRows;

  row_offsets_type L_ptr ("L_ptr", lclNumRows + 1);
  row_offsets_type U_ptr ("U_ptr", lclNumRows + 1);

  // The local number of _entries_ could in theory require 64 bits
  // even if LO is 32 bits.  This example doesn't require it, but why
  // not be general if there is no serious cost?  We use ptrdiff_t
  // because it is signed.
  const ptrdiff_t L_lclNumEnt = explicitlyStoreUnitDiagonalOfL ?
    (2*lclNumRows - 1) :
    (lclNumRows - 1);
  const ptrdiff_t U_lclNumEnt = 2*lclNumRows - 1;

  col_inds_type L_ind ("L_ind", L_lclNumEnt);
  values_type L_val ("L_val", L_lclNumEnt);

  col_inds_type U_ind ("U_ind", U_lclNumEnt);
  values_type U_val ("U_val", U_lclNumEnt);

  const val_type ONE = KAT::one ();
  const val_type TWO = KAT::one () + KAT::one ();
  // Don't cast directly from an integer type to val_type,
  // since if val_type is complex, that cast may not exist.
  const val_type N = static_cast<val_type> (static_cast<mag_type> (lclNumRows));
  const val_type d = TWO * N;

  ptrdiff_t L_curPos = 0;
  for (LO i = 0; i < lclNumRows; ++i) {
    L_ptr[i] = L_curPos;

    if (i + 1 == lclNumRows) {
      // Last row: Add the off-diagonal entries
      for (LO j = 0; j + 1 < lclNumCols; ++j) {
        L_ind[L_curPos] = j;
        L_val[L_curPos] = ONE / d;
        ++L_curPos;
      }
    }
    if (explicitlyStoreUnitDiagonalOfL) {
      // Add the diagonal entry
      L_ind[L_curPos] = i;
      L_val[L_curPos] = ONE;
      ++L_curPos;
    }
  }
  L_ptr[lclNumRows] = L_curPos;

  ptrdiff_t U_curPos = 0;
  for (LO i = 0; i < lclNumRows; ++i) {
    U_ptr[i] = U_curPos;

    if (i + 1 < lclNumRows) {
      // Add the diagonal entry (first in the row)
      U_ind[U_curPos] = i;
      U_val[U_curPos] = d;
      ++U_curPos;

      // Add the last entry in the row
      U_ind[U_curPos] = lclNumCols - 1;
      U_val[U_curPos] = ONE;
      ++U_curPos;
    }
    else if (i + 1 == lclNumRows) {
      // Add the last row's diagonal entry (only entry in this row)
      U_ind[U_curPos] = lclNumCols - 1;
      U_val[U_curPos] = d - (N - ONE) / d;
      ++U_curPos;
    }
  }
  U_ptr[lclNumRows] = U_curPos;

  // Make sure that we counted the number of entries correctly.
  TEST_ASSERT( L_curPos == L_lclNumEnt );
  TEST_ASSERT( U_curPos == U_lclNumEnt );
  if (! success) {
    out << "Aborting test" << endl;
    return;
  }

  out << "Create the lower triangular sparse matrix L" << endl;
  local_matrix_type L ("L", lclNumRows, lclNumCols, L_lclNumEnt, L_val, L_ptr, L_ind);

  out << "Create the upper triangular sparse matrix U" << endl;
  local_matrix_type U ("U", lclNumRows, lclNumCols, U_lclNumEnt, U_val, U_ptr, U_ind);

  typedef ::KokkosSparse::Impl::Mkl::Trsm<local_matrix_type, MV, MV> solver_type;

  out << "Set up the solver for L" << endl;
  typename solver_type::handle_type L_handle;
  typename solver_type::hints_type L_hints;

  const bool L_isImplemented = solver_type::isImplemented ("L", "N", "U");
  TEST_ASSERT( L_isImplemented );
  if (L_isImplemented) {
    TEST_NOTHROW( solver_type::getDefaultHints (L_hints) );
    TEST_NOTHROW( solver_type::symbolicSetup (L_handle, "L", "N", "U", L.graph, L_hints) );
    TEST_NOTHROW( solver_type::numericSetup (L_handle, "L", "N", "U", L, L_hints) );
    solver_type::printHandle (out, L_handle);
  }

  out << "Set up the solver for U" << endl;
  typename solver_type::handle_type U_handle;
  typename solver_type::hints_type U_hints;
  const bool U_isImplemented = solver_type::isImplemented ("U", "N", "N");
  TEST_ASSERT( U_isImplemented );
  if (U_isImplemented) {
    TEST_NOTHROW( solver_type::getDefaultHints (U_hints) );
    TEST_NOTHROW( solver_type::symbolicSetup (U_handle, "U", "N", "N", U.graph, U_hints) );
    TEST_NOTHROW( solver_type::numericSetup (U_handle, "U", "N", "N", U, U_hints) );
    solver_type::printHandle (out, U_handle);
  }

  if (! success) {
    return;
  }

  const val_type scalingFactor = d - (N - ONE) / d;
  const bool scaleProblem = true;//false;

  // Set up the right-hand sides B.
  MV B ("B", lclNumRows, numVecs);
  for (LO i = 0; i < lclNumRows; ++i) {
    // Don't cast directly from an integer type to val_type,
    // since if val_type is complex, that cast may not exist.
    const val_type K = static_cast<val_type> (static_cast<mag_type> (i+1));
    if (scaleProblem) {
      for (LO j = 0; j < numVecs; ++j) {
        B(i,j) = scalingFactor * K;
      }
    }
    else {
      for (LO j = 0; j < numVecs; ++j) {
        B(i,j) = K;
      }
    }
  }

  // We solve AX=B (with A = LU) by first solving LC = B, and then
  // solving UX = C.
  MV C ("C", lclNumRows, numVecs);
  MV X ("X", lclNumRows, numVecs);

  out << "Solve LC = B for C" << endl;
  TEST_NOTHROW( solver_type::apply (L_handle, B, C) );
  if (! success) {
    return;
  }

  // Test the entries of C for correctness.  These are EXACT tests,
  // which we may do since the solves should not have committed any
  // rounding error.  See discussion above.
  //
  // NOTE (mfh 21 Aug 2016) This won't work if we accept approximate
  // sparse triangular solves.

  const val_type c_n_unscaled_expected = N - ((N - ONE)*N) / (TWO * d);
  const val_type c_n_expected = scaleProblem ?
    (scalingFactor * c_n_unscaled_expected) :
    c_n_unscaled_expected;
  const val_type x_n_unscaled_expected = c_n_unscaled_expected / scalingFactor;
  const val_type x_n_expected = scaleProblem ? c_n_unscaled_expected : x_n_unscaled_expected;

  out << "Test entries of C (solution of LC=B)" << endl;
  {
    Teuchos::OSTab tab2 (out);

    for (LO i = 0; i + 1 < lclNumRows; ++i) {
      // Don't cast directly from an integer type to val_type,
      // since if val_type is complex, that cast may not exist.
      const val_type K = static_cast<val_type> (static_cast<mag_type> (i+1));
      const val_type c_i_expected = scaleProblem ? (scalingFactor * K) : K;
      for (LO j = 0; j < numVecs; ++j) {
        TEST_EQUALITY( C(i,j), c_i_expected );
      }
    }
    for (LO j = 0; j < numVecs; ++j) {
      TEST_EQUALITY( C(lclNumRows-1, j), c_n_expected );
    }
  }

  out << "Solve UX = C for x" << endl;
  TEST_NOTHROW( solver_type::apply (U_handle, C, X) );

  // Test the entries of X for correctness.  These are EXACT tests,
  // which we may do since the solves should not have committed any
  // rounding error.  See discussion above.
  //
  // NOTE (mfh 21 Aug 2016) This won't work if we accept approximate
  // sparse triangular solves.

  out << "Test entries of X (solution of UX=C)" << endl;
  {
    Teuchos::OSTab tab2 (out);

    for (LO i = 0; i + 1 < lclNumRows; ++i) {
      // Don't cast directly from an integer type to val_type, since
      // if val_type is complex, that cast may not exist.
      const val_type K = static_cast<val_type> (static_cast<mag_type> (i+1));
      const val_type x_i_expected = scaleProblem ?
        ((scalingFactor * (K - x_n_unscaled_expected)) / d) :
        ((K - x_n_unscaled_expected) / d);
      for (LO j = 0; j < numVecs; ++j) {
        TEST_EQUALITY( X(i,j), x_i_expected );
      }
    }
    for (LO j = 0; j < numVecs; ++j) {
      TEST_EQUALITY( X(lclNumRows-1, j), x_n_expected );
    }
  }

  out << "Test against a reference sparse triangular solver" << endl;

  Kokkos::deep_copy (C, KAT::zero ());
  Kokkos::deep_copy (X, KAT::zero ());

  const std::string unitDiagL = explicitlyStoreUnitDiagonalOfL ?
    "No unit diagonal" : "Unit diagonal";
  TRSM<local_matrix_type, MV> (out, C, L, B, "Lower triangular",
                               "No transpose", unitDiagL.c_str ());
  out << "Test entries of C (solution of LC=B)" << endl;
  {
    Teuchos::OSTab tab2 (out);

    for (LO i = 0; i + 1 < lclNumRows; ++i) {
      // Don't cast directly from an integer type to val_type, since
      // if val_type is complex, that cast may not exist.
      const val_type K = static_cast<val_type> (static_cast<mag_type> (i+1));
      const val_type c_i_expected = scaleProblem ? (scalingFactor * K) : K;
      for (LO j = 0; j < numVecs; ++j) {
        TEST_EQUALITY( C(i,j), c_i_expected );
      }
    }
    for (LO j = 0; j < numVecs; ++j) {
      TEST_EQUALITY( C(lclNumRows-1, j), c_n_expected );
    }
  }

  TRSM<local_matrix_type, MV> (out, X, U, C, "Upper triangular",
                               "No transpose", "No unit diagonal");
  out << "Test entries of X (solution of UX=C)" << endl;
  {
    Teuchos::OSTab tab2 (out);

    for (LO i = 0; i + 1 < lclNumRows; ++i) {
      // Don't cast directly from an integer type to val_type,
      // since if val_type is complex, that cast may not exist.
      const val_type K = static_cast<val_type> (static_cast<mag_type> (i+1));
      const val_type x_i_expected = scaleProblem ?
        ((scalingFactor * (K - x_n_unscaled_expected)) / d) :
        ((K - x_n_unscaled_expected) / d);
      for (LO j = 0; j < numVecs; ++j) {
        TEST_EQUALITY( X(i,j), x_i_expected );
      }
    }
    for (LO j = 0; j < numVecs; ++j) {
      TEST_EQUALITY( X(lclNumRows-1, j), x_n_expected );
    }
  }
}


template<class SC, class LO, class OffsetType>
void
testAllLayouts (bool& success,
                Teuchos::FancyOStream& out,
                const LO numVecs,
                const bool onlyTestLayoutLeft = false)
{
  {
    out << "Test LayoutLeft" << endl;
    Teuchos::OSTab tab1 (out);
    bool curSuccess = true;
    testArrowMatrix<SC, LO, OffsetType, Kokkos::LayoutLeft> (curSuccess, out, numVecs);
    success = success && curSuccess;
  }
  if (! onlyTestLayoutLeft) {
    out << "Test LayoutRight" << endl;
    Teuchos::OSTab tab1 (out);
    bool curSuccess = true;
    testArrowMatrix<SC, LO, OffsetType, Kokkos::LayoutRight> (curSuccess, out, numVecs);
    success = success && curSuccess;
  }
}


template<class SC, class LO>
void
testAllOffsetTypes (bool& success,
                    Teuchos::FancyOStream& out,
                    const LO numVecs,
                    const bool onlyTestIntInt = false,
                    const bool onlyTestLayoutLeft = false)
{
  {
    out << "Test OffsetType=int" << endl;
    Teuchos::OSTab tab1 (out);
    typedef int offset_type;
    bool curSuccess = true;
    testAllLayouts<SC, LO, offset_type> (curSuccess, out, numVecs, onlyTestLayoutLeft);
    success = success && curSuccess;
  }
  if (! onlyTestIntInt) {
    out << "Test OffsetType=size_t" << endl;
    Teuchos::OSTab tab1 (out);
    typedef size_t offset_type;
    bool curSuccess = true;
    testAllLayouts<SC, LO, offset_type> (curSuccess, out, numVecs, onlyTestLayoutLeft);
    success = success && curSuccess;
  }
}


template<class SC>
void
testAllOrdinalTypes (bool& success,
                     Teuchos::FancyOStream& out,
                     const int numVecs,
                     const bool onlyTestIntInt = false,
                     const bool onlyTestLayoutLeft = false)
{
  {
    out << "Test LO=int" << endl;
    Teuchos::OSTab tab1 (out);
    typedef int ordinal_type;
    const ordinal_type nv = static_cast<ordinal_type> (numVecs);
    bool curSuccess = true;
    testAllOffsetTypes<SC, ordinal_type> (curSuccess, out, nv, onlyTestIntInt, onlyTestLayoutLeft);
    success = success && curSuccess;
  }
  if (! onlyTestIntInt) {
    out << "Test LO=long long" << endl;
    Teuchos::OSTab tab1 (out);
    typedef long long ordinal_type;
    const ordinal_type nv = static_cast<ordinal_type> (numVecs);
    bool curSuccess = true;
    testAllOffsetTypes<SC, ordinal_type> (curSuccess, out, nv, onlyTestIntInt, onlyTestLayoutLeft);
    success = success && curSuccess;
  }
}


void
testAllScalarTypes (bool& success,
                    Teuchos::FancyOStream& out,
                    const int numVecs,
                    const bool onlyTestDouble = false,
                    const bool onlyTestIntInt = false,
                    const bool onlyTestLayoutLeft = false)
{
  {
    out << "Test SC=double" << endl;
    Teuchos::OSTab tab1 (out);
    typedef double scalar_type;
    bool curSuccess = true;
    testAllOrdinalTypes<scalar_type> (curSuccess, out, numVecs, onlyTestIntInt, onlyTestLayoutLeft);
    success = success && curSuccess;
  }

  if (! onlyTestDouble) {
    {
      out << "Test SC=float" << endl;
      Teuchos::OSTab tab1 (out);
      typedef float scalar_type;
      bool curSuccess = true;
      testAllOrdinalTypes<scalar_type> (curSuccess, out, numVecs, onlyTestIntInt, onlyTestLayoutLeft);
      success = success && curSuccess;
    }
    {
      out << "Test SC=std::complex<double>" << endl;
      Teuchos::OSTab tab1 (out);
      typedef std::complex<double> scalar_type;
      bool curSuccess = true;
      testAllOrdinalTypes<scalar_type> (curSuccess, out, numVecs, onlyTestIntInt, onlyTestLayoutLeft);
      success = success && curSuccess;
    }
    {
      out << "Test SC=std::complex<float>" << endl;
      Teuchos::OSTab tab1 (out);
      typedef std::complex<float> scalar_type;
      bool curSuccess = true;
      testAllOrdinalTypes<scalar_type> (curSuccess, out, numVecs, onlyTestIntInt, onlyTestLayoutLeft);
      success = success && curSuccess;
    }
  }
}


void
testEverything (bool& success,
                Teuchos::FancyOStream& out,
                const bool onlyTestDouble = false,
                const bool onlyTestIntInt = false,
                const bool onlyTestLayoutLeft = false)
{
  Teuchos::OSTab tab0 (out);

  const int numNumVecs = 3;
  const int numVecsValues[] = {1, 3, 0};

  for (int k = 0; k < numNumVecs; ++k) {
    const int numVecs = numVecsValues[k];
    out << "Test numVecs = " << numVecs << endl;
    Teuchos::OSTab tab1 (out);
    bool curSuccess = true;
    testAllScalarTypes (curSuccess, out, numVecs, onlyTestDouble,
                        onlyTestIntInt, onlyTestLayoutLeft);
    success = success && curSuccess;
  }
}

} // namespace (anonymous)



int
main (int argc, char* argv[])
{
  using std::endl;

  Teuchos::RCP<Teuchos::FancyOStream> outPtr =
    Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cout));
  Teuchos::FancyOStream& out = *outPtr;

  out << "Call Kokkos::initialize" << endl;
  Kokkos::initialize (argc, argv);

  bool success = true;
  out << "Run test" << endl;

  const bool onlyTestDouble = true;
  const bool onlyTestIntInt = true;
  const bool onlyTestLayoutLeft = true;
  testEverything (success, out, onlyTestDouble, onlyTestIntInt, onlyTestLayoutLeft);

  out << "Call Kokkos::finalize" << endl;
  Kokkos::finalize ();

  if (success) {
    out << "End Result: TEST PASSED" << endl;
    return EXIT_SUCCESS;
  }
  else {
    out << "End Result: TEST FAILED" << endl;
    return EXIT_FAILURE;
  }
}
