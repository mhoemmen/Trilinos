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

#include "Teuchos_UnitTestHarness.hpp"
#include "TpetraCore_ETIHelperMacros.h"
#include "Tpetra_DefaultPlatform.hpp"
#include "Tpetra_Map.hpp"
#include "Kokkos_Core.hpp"
#include "Kokkos_ArithTraits.hpp"
#include "Kokkos_InnerProductSpaceTraits.hpp"
#include "Tpetra_Details_crsMatrixAssembleElement.hpp"
#include "Kokkos_Blas1_MV.hpp"
#include <utility> // std::pair

namespace { // (anonymous)

  // Kokkos::parallel_for functor for testing
  // Tpetra::Details::crsMatrixAssembleElement_sortedLinear.
  // MUST be run over an index "range" of a single index, i = 0.
  template<class SparseMatrixType,
           class VectorViewType,
           class RhsViewType,
           class LhsViewType>
  class TestCrsMatrixAssembleElementSortedLinear {
  public:
    typedef typename SparseMatrixType::ordinal_type ordinal_type;
    typedef typename SparseMatrixType::device_type device_type;

    TestCrsMatrixAssembleElementSortedLinear (const SparseMatrixType& A,
                                              const VectorViewType& x,
                                              const Kokkos::View<ordinal_type*, device_type>& lids,
                                              const Kokkos::View<ordinal_type*, device_type>& sortPerm,
                                              const RhsViewType& rhs,
                                              const LhsViewType& lhs,
                                              const bool forceAtomic,
                                              const bool checkInputIndices) :
      A_ (A),
      x_ (x),
      lids_ (lids),
      sortPerm_ (sortPerm),
      rhs_ (rhs),
      lhs_ (lhs),
      forceAtomic_ (forceAtomic),
      checkInputIndices_ (checkInputIndices),
      result_ ("result")
    {
      static_assert (Kokkos::Impl::is_view<VectorViewType>::value,
                     "VectorViewType must be a Kokkos::View specialization.");
      static_assert (Kokkos::Impl::is_view<RhsViewType>::value,
                     "RhsViewType must be a Kokkos::View specialization.");
      static_assert (Kokkos::Impl::is_view<LhsViewType>::value,
                     "LhsViewType must be a Kokkos::View specialization.");
      static_assert (static_cast<int> (RhsViewType::rank) == 1,
                     "RhsViewType must be a rank-1 Kokkos::View.");
      static_assert (static_cast<int> (LhsViewType::rank) == 2,
                     "LhsViewType must be a rank-2 Kokkos::View.");
      static_assert (std::is_integral<ordinal_type>::value,
                     "SparseMatrixType::ordinal_type must be a built-in integer type.");
    }

  // Only meant to be called for i = 0.
  KOKKOS_FUNCTION void
  operator() (const ordinal_type& i) const
  {
    using ::Tpetra::Details::crsMatrixAssembleElement_sortedLinear;

    if (i == 0) {
      const ordinal_type retval =
        crsMatrixAssembleElement_sortedLinear (A_, x_, lids_.ptr_on_device (),
                                               sortPerm_.ptr_on_device (),
                                               rhs_, lhs_, forceAtomic_,
                                               checkInputIndices_);
      result_() = retval;
    }
  }

  ordinal_type
  getReturnValue () const
  {
    auto result_h = Kokkos::create_mirror_view (result_);
    Kokkos::deep_copy (result_h, result_);
    return result_h();
  }

  private:
    SparseMatrixType A_;
    VectorViewType x_;
    Kokkos::View<ordinal_type*, device_type> lids_;
    Kokkos::View<ordinal_type*, device_type> sortPerm_;
    typename RhsViewType::const_type rhs_;
    typename LhsViewType::const_type lhs_;
    bool forceAtomic_;
    bool checkInputIndices_;
    Kokkos::View<ordinal_type, device_type> result_;
  };

  // Call Tpetra::Details::crsMatrixAssembleElement_sortedLinear on
  // device with the given input, and return the function's return
  // value.
  template<class SparseMatrixType,
           class VectorViewType,
           class RhsViewType,
           class LhsViewType>
  //std::pair<typename SparseMatrixType::ordinal_type, std::pair<bool, bool> >
  typename SparseMatrixType::ordinal_type
  testCrsMatrixAssembleElementSortedLinear (const SparseMatrixType& A,
                                            const VectorViewType& x,
                                            const Kokkos::View<typename SparseMatrixType::ordinal_type*, typename SparseMatrixType::device_type>& lids,
                                            const Kokkos::View<typename SparseMatrixType::ordinal_type*, typename SparseMatrixType::device_type>& sortPerm,
                                            const RhsViewType& rhs,
                                            const LhsViewType& lhs,
                                            const typename RhsViewType::const_type& expectedVectorValues,
                                            const typename SparseMatrixType::values_type::const_type& expectedMatrixValues,
                                            const bool forceAtomic =
#ifdef KOKKOS_HAVE_SERIAL
                                            ! std::is_same<typename SparseMatrixType::device_type::execution_space, Kokkos::Serial>::value,
#else // NOT KOKKOS_HAVE_SERIAL
                                            false,
#endif // KOKKOS_HAVE_SERIAL
                                            const bool checkInputIndices = true)
  {
    static_assert (Kokkos::Impl::is_view<VectorViewType>::value,
                   "VectorViewType must be a Kokkos::View specialization.");
    static_assert (Kokkos::Impl::is_view<RhsViewType>::value,
                   "RhsViewType must be a Kokkos::View specialization.");
    static_assert (Kokkos::Impl::is_view<LhsViewType>::value,
                   "LhsViewType must be a Kokkos::View specialization.");
    static_assert (static_cast<int> (RhsViewType::rank) == 1,
                   "RhsViewType must be a rank-1 Kokkos::View.");
    static_assert (static_cast<int> (LhsViewType::rank) == 2,
                   "LhsViewType must be a rank-2 Kokkos::View.");
    typedef TestCrsMatrixAssembleElementSortedLinear<SparseMatrixType,
      VectorViewType, RhsViewType, LhsViewType> functor_type;
    typedef typename SparseMatrixType::value_type SC;
    typedef typename SparseMatrixType::ordinal_type ordinal_type;
    static_assert (std::is_integral<ordinal_type>::value,
                   "SparseMatrixType::ordinal_type must be a built-in integer type.");
    typedef typename SparseMatrixType::device_type device_type;
    typedef typename device_type::execution_space execution_space;
    typedef typename Kokkos::Details::InnerProductSpaceTraits<SC>::mag_type mag_type;
    typedef Kokkos::RangePolicy<execution_space, ordinal_type> range_type;
    const SC ONE = Kokkos::ArithTraits<SC>::one ();

    TEUCHOS_TEST_FOR_EXCEPTION
      (expectedVectorValues.dimension_0 () != x.dimension_0 (),
       std::invalid_argument,
       "expectedVectorValues.dimension_0() = " << expectedVectorValues.dimension_0 ()
       << " != x.dimension_0() = " << x.dimension_0 () << ".");
    TEUCHOS_TEST_FOR_EXCEPTION
      (expectedMatrixValues.dimension_0 () != A.values.dimension_0 (),
       std::invalid_argument,
       "expectedMatrixValues.dimension_0() = " << expectedMatrixValues.dimension_0 ()
       << " != A.values.dimension_0() = " << A.values.dimension_0 () << ".");

    functor_type functor (A, x, lids, sortPerm, rhs, lhs, forceAtomic, checkInputIndices);
    // It's a "parallel" loop with one loop iteration.  The point is
    // to run on device.
    Kokkos::parallel_for (range_type (0, 1), functor);
#if 0
    // Space for checking norm of result diffs.
    Kokkos::View<mag_type, device_type> nrmResult ("nrmResult");
    auto nrmResult_h = Kokkos::create_mirror_view (nrmResult);

    // Check expected vector ("right-hand side") values.
    typename RhsViewType::non_const_type rhsDiff ("rhsDiff", x.dimension_0 ());
    Kokkos::deep_copy (rhsDiff, rhs);
    KokkosBlas::axpby (-ONE, expectedVectorValues, ONE, rhsDiff); // rhsDiff = rhsDiff - expectedVectorValues
    KokkosBlas::nrmInf (nrmResult, rhsDiff);
    Kokkos::deep_copy (nrmResult_h, nrmResult);
    const bool rhsSuccess = (nrmResult_h() == Kokkos::ArithTraits<mag_type>::eps ());

    // Check expected matrix ("left-hand side") values.
    typedef typename SparseMatrixType::values_type matrix_values_type;
    typename matrix_values_type::non_const_type lhsDiff ("lhsDiff", A.values.dimension_0 ());
    Kokkos::deep_copy (lhsDiff, A.values);
    KokkosBlas::axpby (-ONE, expectedMatrixValues, ONE, lhsDiff); // lhsDiff = lhsDiff - expectedMatrixValues
    KokkosBlas::nrmInf (nrmResult, lhsDiff);
    Kokkos::deep_copy (nrmResult_h, nrmResult);
    const bool lhsSuccess = (nrmResult_h() == Kokkos::ArithTraits<mag_type>::eps ());

    return std::make_pair (functor.getReturnValue (),
                           std::make_pair (rhsSuccess, lhsSuccess));
#endif // 0
    return functor.getReturnValue ();
  }

  TEUCHOS_UNIT_TEST_TEMPLATE_1_DECL( CrsMatrix, assembleElement, ScalarType )
  {
    using std::endl;
    typedef typename Kokkos::ArithTraits<ScalarType>::val_type SC;
    typedef Tpetra::Map<>::local_ordinal_type LO;
    //typedef Tpetra::Map<>::global_ordinal_type GO;
    typedef Tpetra::Map<>::device_type DT;
    typedef Tpetra::Map<> map_type;
    typedef KokkosSparse::CrsMatrix<SC, LO, DT, void> sparse_matrix_type;
    typedef typename sparse_matrix_type::size_type offset_type;
    typedef typename sparse_matrix_type::StaticCrsGraphType sparse_graph_type;

    out << "Test crsMatrixAssembleElement" << endl;
    Teuchos::OSTab tab1 (out);

    out << "Create Map just to initialize Kokkos correctly" << endl;
    auto comm = Tpetra::DefaultPlatform::getDefaultPlatform ().getComm ();
    map_type mapToInitKokkos (Tpetra::global_size_t (100), 0, comm);

    // Dimension of the elements to test.
    constexpr LO eltDim = 4;
    // Sparsity pattern of elements to use for scattering into the matrix.
    const LO eltSparsityPattern[eltDim] = {3, 5, 8, 11};

    // Values for the element to scatter into the matrix.  Choose
    // unique values, to make sure we get the indices right.
    const SC ONE = Kokkos::ArithTraits<SC>::one ();

    //out << "Constructing element matrix" << endl;
    std::cerr << "Constructing element matrix" << endl;

    //typename Kokkos::View<SC**, DT>::HostMirror lhs_h ("lhs_h", eltDim, eltDim);
    Kokkos::View<SC**, typename Kokkos::View<SC**, DT>::array_layout,
      Kokkos::HostSpace> lhs_h ("lhs_h", eltDim, eltDim);
    {
      SC curVal = ONE;
      for (LO i = 0; i < eltDim; ++i) {
        for (LO j = 0; j < eltDim; ++j) {
          lhs_h(i,j) = curVal;
          curVal = curVal + ONE;
        }
      }
      out << "Element matrix (lhs): [";
      for (LO i = 0; i < static_cast<LO> (lhs_h.dimension_0 ()); ++i) {
        for (LO j = 0; j < static_cast<LO> (lhs_h.dimension_1 ()); ++j) {
          constexpr int width = Kokkos::ArithTraits<SC>::is_complex ? 7 : 3;
          out << std::setw (width) << lhs_h(i,j);
          if (j + LO (1) < static_cast<LO> (lhs_h.dimension_1 ())) {
            out << " ";
          }
        }
        if (i + LO (1) < static_cast<LO> (lhs_h.dimension_0 ())) {
          out << endl;
        }
      }
      out << "]" << endl;
    }

    //out << "Constructing element vector" << endl;
    std::cerr << "Constructing element vector" << endl;
    //typename Kokkos::View<SC*, DT>::HostMirror rhs_h ("rhs_h", eltDim);
    Kokkos::View<SC*, typename Kokkos::View<SC*, DT>::array_layout,
      Kokkos::HostSpace> rhs_h ("rhs_h", eltDim);
    {
      SC curVal = ONE;
      for (LO i = 0; i < rhs_h.dimension_0 (); ++i) {
        rhs_h(i) = -curVal;
        curVal = curVal + ONE;
      }
    }

    // Number of rows in the sparse matrix A, and number of entries in
    // the dense vector b.  This is a constexpr so we can easily
    // construct arrays with it.
    constexpr LO numRows = 13;
    // Number of columns in the sparse matrix A, and number of entries
    // in the dense vector b.
    //const LO numCols = numRows;

    // When defining the matrix sparsity pattern, make sure that some
    // rows not in the above element sparsity pattern just happen to
    // have some column indices that match those in the element
    // sparsity pattern.
    const LO matSparsityPattern[] = {
      0, 4, 5, 7, 10,            // row 0 (matches none of the indices)
                                 // row 1 is empty
      2, 3, 7, 9, 11,            // row 2 (matches some of the indices)
      0, 3, 5, 7, 8, 10, 11, 12, // row 3 (all indices, plus extra)
      5, 6, 8,                   // row 4 (matches some of the indices)
      1, 3, 5, 6, 8, 11,         // row 5 (all indices, plus extra)
      2, 10, 12,                 // row 6 (matches none of the indices)
      3, 5, 8, 11,               // row 7 (happens to match all the indices)
      3, 5, 8, 11,               // row 8 (exactly the indices, no more)
                                 // row 9 is empty
                                 // row 10 is empty (two consecutive empty rows)
      0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, // row 11 (all indices, plus extra)
      5, 10                      // row 12 (matches some of the indices)
    };

    // Number of entries in each row of the sparse matrix.
    const offset_type numEntPerRow[] = {
      5,
      0,
      5,
      8,
      3,
      6,
      3,
      4,
      4,
      0,
      0,
      13,
      2
    };

#if 0
    // Pattern of entries in the matrix that an assembleElement
    // operation should have changed.
    const LO matChangedPattern[] = {
      0, 0, 0, 0, 0,             // row 0 (matches none of the indices)
                                 // row 1 is empty
      0, 0, 0, 0, 0,             // row 2 (matches some of the indices)
      0, 1, 1, 0, 1, 0, 1, 0,    // row 3 (all indices, plus extra)
      0, 0, 0,                   // row 4 (matches some of the indices)
      0, 1, 1, 0, 1, 1,         // row 5 (all indices, plus extra)
      0, 0, 0,                   // row 6 (matches none of the indices)
      0, 0, 0, 0,                // row 7 (happens to match all the indices)
      1, 1, 1, 1,               // row 8 (exactly the indices, no more)
                                 // row 9 is empty
                                 // row 10 is empty (two consecutive empty rows)
      0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, // row 11 (all indices, plus extra)
      0, 0                      // row 12 (matches some of the indices)
    };
#endif // 0

    // Total number of entries in the sparse matrix.
    const offset_type numEnt = 53;
    offset_type ptr[numRows+1];
    ptr[0] = 0;
    for (offset_type i = 1; i <= numRows; ++i) {
      ptr[i] = ptr[i-1] + numEntPerRow[i-1];
    }
    TEST_EQUALITY( ptr[numRows], numEnt );
    if (ptr[numRows] != numEnt) {
      out << "The sparse matrix for the test was not constructed "
          << "correctly, since the last entry of the offsets array, "
          << "ptr[numRows=" << numRows << "] != numEnt = " << numEnt
          << ".  Please go back and fix it." << endl;
      return; // no sense in continuing the test at this point
    }

    // Expected values in the matrix after the assembleElement operation.
    out << "Make the array of expected matrix entries" << endl;
    typedef typename sparse_matrix_type::values_type::non_const_type values_type;
    values_type expectedMatrixValues ("expectedMatrixValues", numEnt);
    {
      auto expectedMatrixValues_h = Kokkos::create_mirror_view (expectedMatrixValues);
      Kokkos::deep_copy (expectedMatrixValues_h, Kokkos::ArithTraits<SC>::zero ());
      SC curVal = ONE;
      expectedMatrixValues_h(11) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(12) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(14) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(16) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(22) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(23) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(25) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(26) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(34) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(35) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(36) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(37) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(41) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(43) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(46) = curVal;
      curVal += ONE;
      expectedMatrixValues_h(49) = curVal;
      //curVal += ONE;
      Kokkos::deep_copy (expectedMatrixValues, expectedMatrixValues_h);
    }

    // Expected values in the right-hand-side vector after the
    // assembleElement operation.
    out << "Make the array of expected right-hand-side vector entries" << endl;
    Kokkos::View<SC*, DT> expectedVectorValues ("", numRows);
    {
      auto expectedVectorValues_h = Kokkos::create_mirror_view (expectedVectorValues);
      Kokkos::deep_copy (expectedVectorValues_h, Kokkos::ArithTraits<SC>::zero ());
      SC curVal = ONE;
      expectedVectorValues_h(3) = -curVal;
      curVal += ONE;
      expectedVectorValues_h(5) = -curVal;
      curVal += ONE;
      expectedVectorValues_h(8) = -curVal;
      curVal += ONE;
      expectedVectorValues_h(11) = -curVal;
      //curVal += ONE;
      Kokkos::deep_copy (expectedVectorValues, expectedVectorValues_h);
    }

    out << "Create the sparse matrix A, and fill its values with zeros" << endl;
    typename sparse_graph_type::row_map_type::non_const_type
      A_ptr ("A_ptr", numRows+1);
    typename sparse_graph_type::row_map_type::HostMirror::const_type
      A_ptr_host (ptr, numRows+1);
    Kokkos::deep_copy (A_ptr, A_ptr_host);
    typename sparse_graph_type::entries_type::non_const_type
      A_ind ("A_ind", numEnt);
    typename sparse_graph_type::entries_type::HostMirror::const_type
      A_ind_host (matSparsityPattern, numEnt);
    Kokkos::deep_copy (A_ind, A_ind_host);
    sparse_graph_type A_graph (A_ind, A_ptr);
    sparse_matrix_type A ("A", A_graph);
    Kokkos::deep_copy (A.values, Kokkos::ArithTraits<SC>::zero ());

    out << "Create the \"right-hand side\" vector b, and fill it with zeros" << endl;
    Kokkos::View<SC*, DT> b ("b", numRows);
    Kokkos::deep_copy (b, Kokkos::ArithTraits<SC>::zero ());

    out << "Make the element matrix (lhs) and vector (rhs)" << endl;
    Kokkos::View<SC**, DT> lhs_d ("lhs_d", eltDim, eltDim);
    Kokkos::deep_copy (lhs_d, lhs_h);
    Kokkos::View<SC*, DT> rhs_d ("rhs_d", eltDim, eltDim);
    Kokkos::deep_copy (rhs_d, rhs_h);

    out << "Make the list of indices input/output array" << endl;
    Kokkos::View<LO*, DT> lids_d ("lids", eltDim);
    {
      Kokkos::View<const LO*, typename Kokkos::View<LO*, DT>::array_layout,
        Kokkos::HostSpace, Kokkos::MemoryUnmanaged> lids_h (eltSparsityPattern, eltDim);
      //typename Kokkos::View<const LO*, DT>::HostMirror lids_h (eltSparsityPattern, eltDim);
      Kokkos::deep_copy (lids_d, lids_h);
    }

    out << "Make the sort permutation output array" << endl;
    Kokkos::View<LO*, DT> sortPerm_d ("sortPerm", eltDim);

    out << "Call the function to test" << endl;
    auto retval =
      testCrsMatrixAssembleElementSortedLinear (A, b, lids_d, sortPerm_d,
                                                rhs_d, lhs_d,
                                                expectedVectorValues,
                                                expectedMatrixValues);
    const LO numEntFound = retval; // retval.first;
    TEST_EQUALITY( numEntFound, eltDim*eltDim );
    // TEST_ASSERT( retval.second.first );
    // TEST_ASSERT( retval.second.second );
    out << "Function returned numEntFound=" << numEntFound << endl;

    {
      auto A_val_h = Kokkos::create_mirror_view (A.values);
      Kokkos::deep_copy (A_val_h, A.values);
      auto val_h = Kokkos::create_mirror_view (expectedMatrixValues);
      Kokkos::deep_copy (val_h, expectedMatrixValues);
      TEST_EQUALITY( A_val_h.dimension_0 (), val_h.dimension_0 () );
      if (A_val_h.dimension_0 () == val_h.dimension_0 ()) {
        bool same = true;
        const offset_type len = A_val_h.dimension_0 ();
        for (offset_type k = 0; k < len; ++k) {
          if (A_val_h(k) != val_h(k)) {
            same = false;
            break;
          }
        }
        TEST_ASSERT( same );
      }
      out << "A.values            : [";
      for (offset_type k = 0; k < numEnt; ++k) {
        constexpr int width = Kokkos::ArithTraits<SC>::is_complex ? 7 : 3;
        out << std::setw (width) << A_val_h(k);
        if (k + offset_type (1) < numEnt) {
          out << ",";
        }
      }
      out << "]" << endl;
      out << "expectedMatrixValues: [";
      for (offset_type k = 0; k < numEnt; ++k) {
        constexpr int width = Kokkos::ArithTraits<SC>::is_complex ? 7 : 3;
        out << std::setw (width) << val_h(k);
        if (k + offset_type (1) < numEnt) {
          out << ",";
        }
      }
      out << "]" << endl;
    }
    {
      auto b_h = Kokkos::create_mirror_view (b);
      Kokkos::deep_copy (b_h, b);
      auto b_exp_h = Kokkos::create_mirror_view (expectedVectorValues);
      Kokkos::deep_copy (b_exp_h, expectedVectorValues);
      TEST_EQUALITY( b_h.dimension_0 (), b_exp_h.dimension_0 () );
      if (b_h.dimension_0 (), b_exp_h.dimension_0 ()) {
        bool same = true;
        const offset_type len = b_h.dimension_0 ();
        for (offset_type k = 0; k < len; ++k) {
          if (b_h(k) != b_exp_h(k)) {
            same = false;
            break;
          }
        }
        TEST_ASSERT( same );
      }
      out << "Actual output b  : [";
      for (offset_type k = 0;
           k < static_cast<offset_type> (b_h.dimension_0 ()); ++k) {
        out << b_h(k);
        if (k + offset_type (1) <
            static_cast<offset_type> (b_h.dimension_0 ())) {
          out << ",";
        }
      }
      out << "]" << endl;
      out << "Expected output b: [";
      for (offset_type k = 0;
           k < static_cast<offset_type> (b_exp_h.dimension_0 ()); ++k) {
        out << b_exp_h(k);
        if (k + offset_type (1) <
            static_cast<offset_type> (b_exp_h.dimension_0 ())) {
          out << ",";
        }
      }
      out << "]" << endl;
    }
  }

  //
  // INSTANTIATIONS
  //

#define UNIT_TEST_GROUP( SCALAR ) \
  TEUCHOS_UNIT_TEST_TEMPLATE_1_INSTANT( CrsMatrix, assembleElement, SCALAR )

  TPETRA_ETI_MANGLING_TYPEDEFS()

  TPETRA_INSTANTIATE_S_NO_ORDINAL_SCALAR( UNIT_TEST_GROUP )

} // namespace (anonymous)

