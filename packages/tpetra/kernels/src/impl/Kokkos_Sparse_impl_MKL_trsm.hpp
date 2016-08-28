/*
//@HEADER
// ************************************************************************
//
//          Kokkos: Node API and Parallel Node Kernels
//              Copyright (2008) Sandia Corporation
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
//@HEADER
*/

#ifndef KOKKOS_SPARSE_IMPL_MKL_TRSM_HPP_
#define KOKKOS_SPARSE_IMPL_MKL_TRSM_HPP_

#include "TpetraKernels_config.h"
#include "Kokkos_Sparse_impl_MKL.hpp"
#include <ostream>

// Use TRSV instead of TRSM with MKL
#define KOKKOSSPARSE_IMPL_USE_MKL_TRSV 1

namespace KokkosSparse {
namespace Impl {
namespace Mkl {

template<class ValueType>
struct RawTplInspect {
  static sparse_status_t
  setTrsmOpHint (sparse_matrix_t A,
                 sparse_operation_t op,
                 matrix_descr descr,
                 sparse_layout_t layout,
                 MKL_INT dense_matrix_size,
                 MKL_INT expected_calls)
  {
    if (SupportsValueType<ValueType>::value) {
#ifdef HAVE_TPETRAKERNELS_MKL
      return mkl_sparse_set_sm_hint (A, op, descr, layout, dense_matrix_size,
                                     expected_calls);
#else // NOT HAVE_TPETRAKERNELS_MKL
      return tplStatusNotSupported ();
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else {
      return tplStatusNotSupported ();
    }
  }

  static sparse_status_t
  setTrsvOpHint (sparse_matrix_t A,
                 sparse_operation_t op,
                 matrix_descr descr,
                 MKL_INT expected_calls)
  {
    if (SupportsValueType<ValueType>::value) {
#ifdef HAVE_TPETRAKERNELS_MKL
      return mkl_sparse_set_sv_hint (A, op, descr, expected_calls);
#else // NOT HAVE_TPETRAKERNELS_MKL
      return tplStatusNotSupported ();
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else {
      return tplStatusNotSupported ();
    }
  }

  /// \brief Set memory usage hint
  ///
  /// \param A [in/out] Sparse matrix handle
  /// \param policy [in] Valid arguments: SPARSE_MEMORY_NONE,
  ///   SPARSE_MEMORY_AGGRESSIVE
  static sparse_status_t
  setMemoryHint (sparse_matrix_t A,
                 sparse_memory_usage_t policy)
  {
    if (SupportsValueType<ValueType>::value) {
#ifdef HAVE_TPETRAKERNELS_MKL
      return mkl_sparse_set_memory_hint (A, policy);
#else // NOT HAVE_TPETRAKERNELS_MKL
      return tplStatusNotSupported ();
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else {
      return tplStatusNotSupported ();
    }
  }

  //! Optimize, using previously applied hints.
  static sparse_status_t
  optimize (sparse_matrix_t A)
  {
    if (SupportsValueType<ValueType>::value) {
#ifdef HAVE_TPETRAKERNELS_MKL
      return mkl_sparse_optimize (A);
#else // NOT HAVE_TPETRAKERNELS_MKL
      return tplStatusNotSupported ();
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else {
      return tplStatusNotSupported ();
    }
  }
};

template<class ValueType>
struct RawTplTrsmExecute {
  typedef ValueType value_type;
  typedef value_type internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t /* operation */,
           const internal_value_type /* alpha */,
           const sparse_matrix_t /* A */,
           struct matrix_descr /* descr */,
           sparse_layout_t /* layout */,
           const internal_value_type* /* x */,
           const MKL_INT /* columns */,
           const MKL_INT /* ldx */,
           internal_value_type* /* y */,
           const MKL_INT /* ldy */)
  {
    return tplStatusNotSupported ();
  }
};

template<class ValueType>
struct RawTplTrsvExecute {
  typedef ValueType value_type;
  typedef value_type internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t /* operation */,
           const internal_value_type /* alpha */,
           const sparse_matrix_t /* A */,
           struct matrix_descr /* descr */,
           const internal_value_type* /* x */,
           internal_value_type* /* y */)
  {
    return tplStatusNotSupported ();
  }
};


#ifdef HAVE_TPETRAKERNELS_MKL
template<>
struct RawTplTrsmExecute<double> {
  typedef double value_type;
  typedef value_type internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t operation,
           internal_value_type alpha,
           const sparse_matrix_t A,
           struct matrix_descr descr,
           sparse_layout_t layout,
           const internal_value_type* x,
           MKL_INT columns,
           MKL_INT ldx,
           internal_value_type* y,
           MKL_INT ldy)
  {
    return mkl_sparse_d_trsm (operation, alpha, A, descr, layout, x, columns, ldx, y, ldy);
  }
};

template<>
struct RawTplTrsvExecute<double> {
  typedef double value_type;
  typedef value_type internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t operation,
           const internal_value_type alpha,
           const sparse_matrix_t A,
           struct matrix_descr descr,
           const internal_value_type* x,
           internal_value_type* y)
  {
    return mkl_sparse_d_trsv (operation, alpha, A, descr, x, y);
  }
};

template<>
struct RawTplTrsmExecute<float> {
  typedef float value_type;
  typedef value_type internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t operation,
           internal_value_type alpha,
           const sparse_matrix_t A,
           struct matrix_descr descr,
           sparse_layout_t layout,
           const internal_value_type* x,
           MKL_INT columns,
           MKL_INT ldx,
           internal_value_type* y,
           MKL_INT ldy)
  {
    return mkl_sparse_s_trsm (operation, alpha, A, descr, layout, x, columns, ldx, y, ldy);
  }
};

template<>
struct RawTplTrsvExecute<float> {
  typedef float value_type;
  typedef value_type internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t operation,
           const internal_value_type alpha,
           const sparse_matrix_t A,
           struct matrix_descr descr,
           const internal_value_type* x,
           internal_value_type* y)
  {
    return mkl_sparse_s_trsv (operation, alpha, A, descr, x, y);
  }
};

template<>
struct RawTplTrsmExecute< ::Kokkos::complex<double> > {
  typedef ::Kokkos::complex<double> value_type;
  typedef MKL_Complex16 internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t operation,
           internal_value_type alpha,
           const sparse_matrix_t A,
           struct matrix_descr descr,
           sparse_layout_t layout,
           const internal_value_type* x,
           MKL_INT columns,
           MKL_INT ldx,
           internal_value_type* y,
           MKL_INT ldy)
  {
    return mkl_sparse_z_trsm (operation, alpha, A, descr, layout, x, columns, ldx, y, ldy);
  }
};

template<>
struct RawTplTrsvExecute< ::Kokkos::complex<double> > {
  typedef ::Kokkos::complex<double> value_type;
  typedef MKL_Complex16 internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t operation,
           const internal_value_type alpha,
           const sparse_matrix_t A,
           struct matrix_descr descr,
           const internal_value_type* x,
           internal_value_type* y)
  {
    return mkl_sparse_z_trsv (operation, alpha, A, descr, x, y);
  }
};

template<>
struct RawTplTrsmExecute< ::Kokkos::complex<float> > {
  typedef ::Kokkos::complex<float> value_type;
  typedef MKL_Complex8 internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t operation,
           internal_value_type alpha,
           const sparse_matrix_t A,
           struct matrix_descr descr,
           sparse_layout_t layout,
           const internal_value_type* x,
           MKL_INT columns,
           MKL_INT ldx,
           internal_value_type* y,
           MKL_INT ldy)
  {
    return mkl_sparse_c_trsm (operation, alpha, A, descr, layout, x, columns, ldx, y, ldy);
  }
};

template<>
struct RawTplTrsvExecute< ::Kokkos::complex<float> > {
  typedef ::Kokkos::complex<float> value_type;
  typedef MKL_Complex8 internal_value_type;

  static sparse_status_t
  execute (sparse_operation_t operation,
           const internal_value_type alpha,
           const sparse_matrix_t A,
           struct matrix_descr descr,
           const internal_value_type* x,
           internal_value_type* y)
  {
    return mkl_sparse_c_trsv (operation, alpha, A, descr, x, y);
  }
};
#endif // HAVE_TPETRAKERNELS_MKL



template<class MatrixType,
         class BMV,
         class XMV,
         const bool implemented =
           SupportsValueType<typename MatrixType::non_const_value_type>::value &&
           std::is_same<typename MatrixType::non_const_value_type,
                        typename XMV::non_const_value_type>::value &&
           std::is_same<typename MatrixType::non_const_value_type,
                        typename BMV::non_const_value_type>::value &&
           static_cast<int> (XMV::rank) == 2 >
struct Trsm {
  /// \brief Type of the handle used to store sparse TRSM setup data.
  ///
  /// Please treat this as an opaque data structure.
  /// Names and types of its fields are implementation details.
  typedef struct {
    //! MKL's matrix handle, wrapped.  Call getHandle() to unwrap it.
    std::shared_ptr<WrappedTplMatrixHandle<MatrixType> > matrix;
    //! What operation to perform.
    sparse_operation_t op;
    //! Description of the matrix's shape.
    matrix_descr descr;
    //! Whether the dense matrices are stored row major or column major.
    sparse_layout_t layout;
  } handle_type;

  static void
  printHandle (std::ostream& out,
               const handle_type& handle,
               const bool oneLine = false)
  {
    using std::endl;

    out << "MKL TRSM handle:";
    std::string indent;
    if (oneLine) {
      out << " {";
    }
    else {
      out << endl;
      indent = " ";
    }

    out << indent << "Wrapped matrix handle: ";
    if (handle.matrix.get () == NULL) {
      out << "NULL";
    }
    else {
      out << "Allocated";
    }
    if (oneLine) {
      out << ", ";
    }
    else {
      out << endl;
    }

    out << indent << "Operation: " << sparseOperationToString (handle.op);
    if (oneLine) {
      out << ", ";
    }
    else {
      out << endl;
    }

    out << indent << "Descriptor: ";
    if (! oneLine) {
      out << endl;
    }
    out << matrixDescriptorToString (handle.descr, " ", oneLine, false);
    if (oneLine) {
      out << ", ";
    }
    // Above already adds a newline if oneLine == true.
    // else {
    //   out << endl;
    // }

    out << indent << "Layout: " << sparseLayoutToString (handle.layout);
    if (oneLine) {
      out << "}";
    }
    else {
      out << endl;
    }
  }

  //! Tuning hints for this kernel.
  typedef struct {
    //! Number of columns in the input and output multivectors.
    MKL_INT dense_matrix_size;
    //! Expected number of times the method will be called.
    MKL_INT expected_calls;
    //! Memory usage policy.
    sparse_memory_usage_t policy;
    /// \brief Whether we should assume that the graph of the matrix
    ///   never changes, even if the matrix pointer changes.
    bool reuseGraph;
  } hints_type;

  //! Type of each entry (value) in the sparse matrix.
  typedef typename MatrixType::value_type value_type;

  /// \brief Fill the given struct of tuning hints with default values
  ///   of tuning parameters.
  ///
  /// \param hints [out] Struct of hints to fill.
  static void
  getDefaultHints (hints_type& /* hints */) {}

  /// \brief Whether the desired case is implemented.
  ///
  /// This method only reads at most the first character of each
  /// string, so the string may be as long as you like.
  ///
  /// \param uplo [in] "U" for upper triangular, "L" for lower
  ///   triangular.
  /// \param transA [in] "C" for conjugate transpose, "T" for
  ///   transpose, "N" for neither.
  /// \param diag [in] "U" for an implicitly stored unit diagonal, "N"
  ///   for all diagonal entries assumed to be explicitly stored.
  ///
  /// \return Whether the desired case is implemented.
  static bool
  isImplemented (const char /* uplo */ [],
                 const char /* transA */ [],
                 const char /* diag */ [])
  {
    // Default is not to use MKL, because MKL only supports certain types.
    return false;
  }

  static void
  symbolicSetup (handle_type& /* handle */,
                 const char /* uplo */ [],
                 const char /* trans */ [],
                 const char /* diag */ [],
                 const typename MatrixType::StaticCrsGraphType& /* G */,
                 hints_type& /* hints */)
  {
    throw std::logic_error ("Not implemented!");
  }

  static void
  numericSetup (handle_type& /* handle */,
                const char /* uplo */ [],
                const char /* trans */ [],
                const char /* diag */ [],
                const MatrixType& /* A */,
                hints_type& /* hints */)
  {
    throw std::logic_error ("Not implemented!");
  }

  static void
  apply (handle_type& handle,
         const BMV& /* B */,
         const XMV& /* X */,
         const value_type& /* alpha */ = Kokkos::Details::ArithTraits<value_type>::one ())
  {
    throw std::logic_error ("Not implemented!");
  }
};


template<class MatrixType,
         class BMV,
         class XMV>
struct Trsm<MatrixType, BMV, XMV, true> {
  /// \brief Type of the handle used to store sparse TRSM setup data.
  ///
  /// Please treat this as an opaque data structure.
  /// Names and types of its fields are implementation details.
  typedef struct {
    //! MKL's matrix handle, wrapped.  Call getHandle() to unwrap it.
    std::shared_ptr<WrappedTplMatrixHandle<MatrixType> > matrix;
    //! What operation to perform.
    sparse_operation_t op;
    //! Description of the matrix's shape.
    matrix_descr descr;
    //! Whether the dense matrices are stored row major or column major.
    sparse_layout_t layout;
  } handle_type;

  static void
  printHandle (std::ostream& out,
               const handle_type& handle,
               const bool oneLine = false)
  {
    using std::endl;

    out << "MKL TRSM handle:";
    std::string indent;
    if (oneLine) {
      out << " {";
    }
    else {
      out << endl;
      indent = " ";
    }

    out << indent << "Wrapped matrix handle: ";
    if (handle.matrix.get () == NULL) {
      out << "NULL";
    }
    else {
      out << "Allocated";
    }
    if (oneLine) {
      out << ", ";
    }
    else {
      out << endl;
    }

    out << indent << "Operation: " << sparseOperationToString (handle.op);
    if (oneLine) {
      out << ", ";
    }
    else {
      out << endl;
    }

    out << indent << "Descriptor: ";
    if (! oneLine) {
      out << endl;
    }
    out << matrixDescriptorToString (handle.descr, " ", oneLine, false);
    if (oneLine) {
      out << ", ";
    }
    // Above already adds a newline if oneLine == true.
    // else {
    //   out << endl;
    // }

    out << indent << "Layout: " << sparseLayoutToString (handle.layout);
    if (oneLine) {
      out << "}";
    }
    else {
      out << endl;
    }
  }

  //! Tuning hints for this kernel.
  typedef struct {
    //! Number of columns in the input and output multivectors.
    MKL_INT dense_matrix_size;
    //! Expected number of times the method will be called.
    MKL_INT expected_calls;
    //! Memory usage policy.
    sparse_memory_usage_t policy;
    /// \brief Whether we should assume that the graph of the matrix
    ///   never changes, even if the matrix pointer changes.
    bool reuseGraph;
  } hints_type;

  //! Type of each entry (value) in the sparse matrix.
  typedef typename MatrixType::value_type value_type;

  /// \brief Fill the given struct of tuning hints with default values
  ///   of tuning parameters.
  ///
  /// \param hints [out] Struct of hints to fill.
  static void
  getDefaultHints (hints_type& hints) {
    // FIXME (mfh 26 Aug 2016) Does the number of right-hand sides
    // have to be exact?
    //hints.dense_matrix_size = 10;
    hints.dense_matrix_size = 3;
    hints.expected_calls = 30;
#ifdef HAVE_TPETRAKERNELS_MKL
    hints.policy = SPARSE_MEMORY_NONE; //SPARSE_MEMORY_AGGRESSIVE;
#else
    hints.policy = 0;
#endif // HAVE_TPETRAKERNELS_MKL
    hints.reuseGraph = false;
  }

  /// \brief Whether the desired case is implemented.
  ///
  /// This method only reads at most the first character of each
  /// string, so the string may be as long as you like.
  ///
  /// \param uplo [in] "U" for upper triangular, "L" for lower
  ///   triangular.
  /// \param transA [in] "C" for conjugate transpose, "T" for
  ///   transpose, "N" for neither.
  /// \param diag [in] "U" for an implicitly stored unit diagonal, "N"
  ///   for all diagonal entries assumed to be explicitly stored.
  ///
  /// \return Whether the desired case is implemented.
  static bool
  isImplemented (const char /* uplo */ [],
                 const char /* transA */ [],
                 const char /* diag */ [])
  {
    // MKL implements both lower and upper triangular cases, all three
    // "transpose" cases, and both unit diagonal cases.
    return true;
  }

  static void
  symbolicSetup (handle_type& /* handle */,
                 const char /* uplo */ [],
                 const char /* trans */ [],
                 const char /* diag */ [],
                 const typename MatrixType::StaticCrsGraphType& /* G*/,
                 hints_type& /* hints */ )
  {
    // MKL does not distinguish between symbolic and numeric setup; it
    // only has one setup phase.  Thus, for correctness, we must push
    // all setup into numericSetup.
  }

private:
  static void
  fillHandle (handle_type& handle,
              const char uplo[],
              const char trans[],
              const char diag[])
  {
    if (trans[0] == 'C' || trans[0] == 'c' ||
        trans[0] == 'H' || trans[0] == 'h') {
#ifdef HAVE_TPETRAKERNELS_MKL
      handle.op = SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
#else
      handle.op = 2;
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else if (trans[0] == 'T' || trans[0] == 't') {
#ifdef HAVE_TPETRAKERNELS_MKL
      handle.op = SPARSE_OPERATION_TRANSPOSE;
#else
      handle.op = 1;
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else if (trans[0] == 'N' || trans[0] == 'n') {
#ifdef HAVE_TPETRAKERNELS_MKL
      handle.op = SPARSE_OPERATION_NON_TRANSPOSE;
#else
      handle.op = 0;
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else {
      std::ostringstream os;
      os << "Invalid transpose argument \"" << trans << "\". "
        "Valid arguments are 'N', 'T', and 'C'.";
      throw std::invalid_argument (os.str ());
    }

#ifdef HAVE_TPETRAKERNELS_MKL
    handle.descr.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
#else
    handle.descr.type = 0;
#endif // HAVE_TPETRAKERNELS_MKL

    if (uplo[0] == 'U' || uplo[0] == 'u') {
#ifdef HAVE_TPETRAKERNELS_MKL
      handle.descr.mode = SPARSE_FILL_MODE_UPPER;
#else
      handle.descr.mode = 1;
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else if (uplo[0] == 'L' || uplo[0] == 'l') {
#ifdef HAVE_TPETRAKERNELS_MKL
      handle.descr.mode = SPARSE_FILL_MODE_LOWER;
#else
      handle.descr.mode = 0;
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else {
      std::ostringstream os;
      os << "Invalid uplo argument \"" << uplo << "\". "
        "Valid arguments are 'U' and 'L'.";
      throw std::invalid_argument (os.str ());
    }

    if (diag[0] == 'U' || diag[0] == 'u') {
#ifdef HAVE_TPETRAKERNELS_MKL
      handle.descr.diag = SPARSE_DIAG_UNIT;
#else
      handle.descr.diag = 1;
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else if (diag[0] == 'N' || diag[0] == 'n') {
#ifdef HAVE_TPETRAKERNELS_MKL
      handle.descr.diag = SPARSE_DIAG_NON_UNIT;
#else
      handle.descr.diag = 0;
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else {
      std::ostringstream os;
      os << "Invalid diag argument \"" << diag << "\". "
        "Valid arguments are 'U' and 'N'.";
      throw std::invalid_argument (os.str ());
    }

    if (std::is_same<typename BMV::array_layout, Kokkos::LayoutLeft>::value &&
        std::is_same<typename XMV::array_layout, Kokkos::LayoutLeft>::value) {
#ifdef HAVE_TPETRAKERNELS_MKL
      handle.layout = SPARSE_LAYOUT_COLUMN_MAJOR; // Tpetra default
#else
      handle.layout = 0;
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else if (std::is_same<typename BMV::array_layout, Kokkos::LayoutRight>::value &&
             std::is_same<typename XMV::array_layout, Kokkos::LayoutRight>::value) {
#ifdef HAVE_TPETRAKERNELS_MKL
      handle.layout = SPARSE_LAYOUT_ROW_MAJOR;
#else
      handle.layout = 1;
#endif // HAVE_TPETRAKERNELS_MKL
    }
    else {
      throw std::invalid_argument ("BMV and/or XMV have an unsupported layout.");
    }
  }

public:

  /// \brief Call this method when the matrix's structure and values
  ///   are fixed, and you want to prepare for apply().
  ///
  /// \param handle [in/out] On input: Handle on which symbolicSetup
  ///   has been called.  On output: Handle that is ready for use in
  ///   apply().
  ///
  /// \param uplo [in] "L" if the matrix is lower triangular, or "U"
  ///   if the matrix is upper triangular.  Must be the same argument
  ///   as that given to symbolicSetup().
  ///
  /// \param trans [in] "N" if not applying the transpose, "T" if
  ///   applying the transpose, or "C" if applying the conjugate
  ///   transpose.  Must be the same argument as that given to
  ///   symbolicSetup().
  ///
  /// \param diag [in] "U" if the matrix has an implictly stored unit
  ///   diagonal, or "N" otherwise.  Must be the same argument as that
  ///   given to symbolicSetup().
  ///
  /// \param A [in] The sparse matrix; a specialization of
  ///   KokkosSparse::CrsMatrix.
  ///
  /// \param hints [in/out] Previously initialized optimization hints.
  ///   If you don't know how to set the hints, use getDefaultHints()
  ///   to initialize the hints struct before using it here.  This
  ///   method reserves the right to change any input hints in place.
  static void
  numericSetup (handle_type& handle,
                const char uplo[],
                const char trans[],
                const char diag[],
                const MatrixType& A,
                hints_type& hints)
  {
    typedef WrappedTplMatrixHandle<MatrixType> matrix_type;

    if (handle.matrix.get () == NULL) {
      handle.matrix =
        std::shared_ptr<matrix_type> (new matrix_type (A, hints.reuseGraph));
    }
    else {
      handle.matrix->setMatrix (A, hints.reuseGraph);
    }
    fillHandle (handle, uplo, trans, diag);

    sparse_matrix_t rawMatrix = handle.matrix->getHandle ();

#ifdef KOKKOSSPARSE_IMPL_USE_MKL_TRSV
    // Use TRSV, as long as both BMV and XMV are LayoutLeft (column major).
    constexpr bool useTrsv =
      std::is_same<typename BMV::array_layout, Kokkos::LayoutLeft>::value &&
      std::is_same<typename XMV::array_layout, Kokkos::LayoutLeft>::value;
#else // NOT KOKKOSSPARSE_IMPL_USE_MKL_TRSV
    // Never use TRSV; use TRSM instead.
    constexpr bool useTrsv = false;
#endif // KOKKOSSPARSE_IMPL_USE_MKL_TRSV

    sparse_status_t status;
    if (useTrsv) {
      status =
        RawTplInspect<value_type>::setTrsvOpHint (rawMatrix,
                                                  handle.op,
                                                  handle.descr,
                                                  hints.expected_calls);
    }
    else {
      status =
        RawTplInspect<value_type>::setTrsmOpHint (rawMatrix,
                                                  handle.op,
                                                  handle.descr,
                                                  handle.layout,
                                                  hints.dense_matrix_size,
                                                  hints.expected_calls);
    }
    if (status != tplStatusSuccessful ()) {
      std::ostringstream os;
      os << "Failed to set operation hint for MKL "
         << (useTrsv ? "TRSV" : "TRSM") << ".  Returned status: "
         << tplStatusToString (status);
      throw std::runtime_error (os.str ());
    }

    status = RawTplInspect<value_type>::setMemoryHint (rawMatrix,
                                                       hints.policy);
    if (status != tplStatusSuccessful ()) {
      std::ostringstream os;
      os << "Failed to set memory usage hint.  Returned status: "
         << tplStatusToString (status);
      throw std::runtime_error (os.str ());
    }

    // This is the call that actually optimizes the computational kernel.
    status = RawTplInspect<value_type>::optimize (rawMatrix);
    if (status != tplStatusSuccessful ()) {
      std::ostringstream os;
      os << "MKL sparse kernel optimization failed.  Returned status: "
         << tplStatusToString (status);
      throw std::runtime_error (os.str ());
    }
  }

  /// \brief Solve the sparse triangular system(s) A*X = alpha*B for X.
  ///
  /// You may call this method as many times as you like, but only
  /// after calling symbolicSetup() and numericSetup(), in that order.
  /// If the matrix's values (but not its graph structure) may have
  /// changed, you must call numericSetup() with the matrix, before
  /// you may call apply() again.  If the matrix's graph structure
  /// (and perhaps also its values) may have changed, you must call
  /// symbolicSetup and numericSetup(), in that order, before you may
  /// call apply() again.
  ///
  /// \param handle [in/out] Handle that was given to symbolicSetup()
  ///   and numericSetup().  This stores the sparse matrix, in MKL's
  ///   native representation.
  ///
  /// \param X [in/out] Output (multi)vector.
  /// \param B [in] Input (multi)vector.
  /// \param alpha [in] Coefficient by which to multiply the result X.
  static void
  apply (handle_type& handle,
         const XMV& X, // output
         const BMV& B, // input
         const value_type& alpha = Kokkos::Details::ArithTraits<value_type>::one ())
  {
    static_assert (std::is_same<typename BMV::array_layout,
                     typename XMV::array_layout>::value,
                   "BMV and XMV must have the same layout.");
    if (handle.matrix.get () == NULL) {
      throw std::invalid_argument ("apply: You must call symbolicSetup and "
                                   "numericSetup with the input matrix, "
                                   "before you may call apply().");
    }
    const MKL_INT numVecs = static_cast<MKL_INT> (X.dimension_1 ());

    if (numVecs != static_cast<MKL_INT> (B.dimension_1 ())) {
      std::ostringstream os;
      os << "X and B must have the same number of columns.  "
         << "X.dimension_1() = " << numVecs
         << " != B.dimension_1() = " << B.dimension_1 () << ".";
      throw std::invalid_argument (os.str ());
    }

#ifdef KOKKOSSPARSE_IMPL_USE_MKL_TRSV
    // Use TRSV, as long as both BMV and XMV are LayoutLeft (column major).
    constexpr bool useTrsv =
      std::is_same<typename BMV::array_layout, Kokkos::LayoutLeft>::value &&
      std::is_same<typename XMV::array_layout, Kokkos::LayoutLeft>::value;
#else // NOT KOKKOSSPARSE_IMPL_USE_MKL_TRSV
    // Never use TRSV; use TRSM instead.
    constexpr bool useTrsv = false;
#endif // KOKKOSSPARSE_IMPL_USE_MKL_TRSV

    MKL_INT LDX = 0;
    MKL_INT LDB = 0;

    if (! useTrsv) {
      if (std::is_same<typename BMV::array_layout, Kokkos::LayoutLeft>::value) {
        MKL_INT x_strides[8];
        MKL_INT b_strides[8];
        X.stride (x_strides);
        LDX = (X.dimension_1 () == 0) ? 0 : x_strides[1];
        B.stride (b_strides);
        LDB = (X.dimension_1 () == 0) ? 0 : b_strides[1];
      }
      else if (std::is_same<typename BMV::array_layout, Kokkos::LayoutRight>::value) {
        MKL_INT x_strides[8];
        MKL_INT b_strides[8];
        X.stride (x_strides);
        LDX = (X.dimension_0 () == 0) ? 0 : x_strides[0];
        B.stride (b_strides);
        LDB = (B.dimension_0 () == 0) ? 0 : b_strides[0];
      }
      else {
        throw std::invalid_argument ("BMV and XMV have an unsupported layout.");
      }
    }

    typedef typename RawTplMatrixHandle<value_type>::internal_value_type IVT;
    const IVT alpha_ivt =
      RawTplMatrixHandle<value_type>::convertToInternalValue (alpha);

    if (useTrsv) {
      // Use single-vector triangular solve, one column at a time.
      for (MKL_INT j = 0; j < numVecs; ++j) {
        auto B_j = Kokkos::subview (B, Kokkos::ALL (), j);
        auto X_j = Kokkos::subview (X, Kokkos::ALL (), j);
        const IVT* B_j_raw = reinterpret_cast<const IVT*> (B_j.ptr_on_device ());
        IVT* X_j_raw = reinterpret_cast<IVT*> (X_j.ptr_on_device ());
        const sparse_status_t status =
          RawTplTrsvExecute<value_type>::execute (handle.op, alpha_ivt,
                                                  handle.matrix->getHandle (),
                                                  handle.descr, B_j_raw,
                                                  X_j_raw);
        if (status != tplStatusSuccessful ()) {
          std::ostringstream os;
          os << "MKL TRSV failed on column " << (j+1) << " of " << numVecs;
          throw std::runtime_error (os.str ());
        }
      }
    }
    else { // use TRSM
      const IVT* B_raw = reinterpret_cast<const IVT*> (B.ptr_on_device ());
      IVT* X_raw = reinterpret_cast<IVT*> (X.ptr_on_device ());
      const sparse_status_t status =
        RawTplTrsmExecute<value_type>::execute (handle.op, alpha_ivt,
                                                handle.matrix->getHandle (),
                                                handle.descr, handle.layout,
                                                B_raw, numVecs, LDB,
                                                X_raw, LDX);
      if (status != tplStatusSuccessful ()) {
        throw std::runtime_error ("MKL TRSM failed!");
      }
    }
  }
};

} // namespace Mkl
} // namespace Impl
} // namespace KokkosSparse

#endif // KOKKOS_SPARSE_IMPL_MKL_TRSM_HPP_
