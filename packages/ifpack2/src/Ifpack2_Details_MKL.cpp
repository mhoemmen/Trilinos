/*@HEADER
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

#include "Ifpack2_Details_MKL.hpp"
#ifdef HAVE_IFPACK2_MKL
#include "mkl_spblas.h"

namespace Ifpack2 {
namespace Details {
namespace MKL {
namespace Impl {

MklSparseMatrixHandle::
MklSparseMatrixHandle () :
  A_ (reinterpret_cast<void*> (new sparse_matrix_t)),
  initialized_ (false)
{}

MklSparseMatrixHandle::
~MklSparseMatrixHandle ()
{
  reset ();
  if (A_ != nullptr) {
    sparse_matrix_t* A = reinterpret_cast<sparse_matrix_t*> (A_);
    delete A;
    A_ = nullptr;
  }
}

void
MklSparseMatrixHandle::
reset ()
{
  if (initialized_) {
    sparse_matrix_t A;
    memcpy (&A, A_, sizeof (sparse_matrix_t));
    // There's no rational way to recover from a error here,
    // so there's no reason to check the returned error code.
    (void) mkl_sparse_destroy (A);
    initialized_ = false;
  }
}

bool
MklSparseMatrixHandle::
initialized () const {
  return initialized_;
}

sparse_operation_t
transpose_operation (const Teuchos::ETransp trans)
{
  if (trans == Teuchos::CONJ_TRANS) {
    return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
  }
  else if (trans == Teuchos::TRANS) {
    return SPARSE_OPERATION_TRANSPOSE;
  }
  else {
    return SPARSE_OPERATION_NON_TRANSPOSE;
  }
}

sparse_fill_mode_t
upper_or_lower_triangular (const Teuchos::EUplo ul)
{
  return ul == Teuchos::UPPER_TRI ? SPARSE_FILL_MODE_UPPER :
    SPARSE_FILL_MODE_LOWER;
}

sparse_diag_type_t
unit_or_nonunit_diagonal (const Teuchos::EDiag d)
{
  return d == Teuchos::UNIT_DIAG ? SPARSE_DIAG_UNIT :
    SPARSE_DIAG_NON_UNIT;
}

struct matrix_descr
triangular_solve_descr (const Teuchos::EUplo ul,
                        const Teuchos::EDiag di)
{
  struct matrix_descr d;
  d.type = SPARSE_MATRIX_TYPE_TRIANGULAR;
  d.mode = upper_or_lower_triangular (ul);
  d.diag = unit_or_nonunit_diagonal (di);
  return d;
}

void
check_error (const sparse_status_t status,
             const char funcName[])
{
  TEUCHOS_TEST_FOR_EXCEPTION
    (status == SPARSE_STATUS_NOT_INITIALIZED ||
     status == SPARSE_STATUS_INVALID_VALUE ||
     status == SPARSE_STATUS_NOT_SUPPORTED, std::logic_error,
     funcName << " says that we did something wrong.");
  TEUCHOS_TEST_FOR_EXCEPTION
    (status == SPARSE_STATUS_ALLOC_FAILED ||
     status == SPARSE_STATUS_EXECUTION_FAILED ||
     status == SPARSE_STATUS_INTERNAL_ERROR, std::runtime_error,
     funcName << " says that it failed to do something.");
  TEUCHOS_TEST_FOR_EXCEPTION
    (status != SPARSE_STATUS_SUCCESS, std::runtime_error,
     funcName << " says that something else bad happened "
     "that its documentation doesn't explain.");
}

// void
// reorder_columns (sparse_matrix_t A)
// {
//   const sparse_status_t status = mkl_sparse_order (A);
//   check_error (status, "mkl_sparse_order");
// }

bool
mklSupportsValueType (const std::type_info& val_type)
{
  return val_type == typeid (double) ||
    val_type == typeid (float) ||
    val_type == typeid (Kokkos::complex<double>) ||
    val_type == typeid (Kokkos::complex<float>);
}

void
MklSparseMatrixHandle::
setCrsMatrix (int num_rows,
              int num_cols,
              int* beg,
              int* end,
              int* ind,
              void* val,
              const std::type_info& val_type)
{
  static_assert (std::is_same<MKL_INT, int>::value, "MKL_INT != int");
  reset ();

  sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
  sparse_matrix_t A;
  sparse_status_t status;
  if (val_type == typeid (double)) {
    double* values = reinterpret_cast<double*> (val);
    status = mkl_sparse_d_create_csr (&A, indexing,
                                      num_rows, num_cols,
                                      beg, end, ind, values);
  }
  else if (val_type == typeid (float)) {
    float* values = reinterpret_cast<float*> (val);
    status = mkl_sparse_s_create_csr (&A, indexing,
                                      num_rows, num_cols,
                                      beg, end, ind, values);
  }
  else if (val_type == typeid (Kokkos::complex<double>)) {
    MKL_Complex16* values = reinterpret_cast<MKL_Complex16*> (val);
    status = mkl_sparse_z_create_csr (&A, indexing,
                                      num_rows, num_cols,
                                      beg, end, ind, values);
  }
  else if (val_type == typeid (Kokkos::complex<float>)) {
    MKL_Complex8* values = reinterpret_cast<MKL_Complex8*> (val);
    status = mkl_sparse_c_create_csr (&A, indexing,
                                      num_rows, num_cols,
                                      beg, end, ind, values);
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION
      (true, std::logic_error, "Not implemented for the "
       "given Scalar type " << val_type.name ());
  }
  check_error (status, "mkl_sparse_?_create_csr");
  memcpy (A_, &A, sizeof (sparse_matrix_t));
  initialized_ = true;
}


void
MklSparseMatrixHandle::
setTriangularSolveHints (const Teuchos::ETransp trans,
                         const Teuchos::EUplo uplo,
                         const Teuchos::EDiag diag,
                         const int num_expected_calls)
{
  if (! initialized_) {
    return; // ignore the hints; this is harmless
  }

  sparse_matrix_t A;
  memcpy (&A, A_, sizeof (sparse_matrix_t));

  const auto op = transpose_operation (trans);
  const auto descr = triangular_solve_descr (uplo, diag);

  sparse_status_t status =
    mkl_sparse_set_sv_hint (A, op, descr, num_expected_calls);
  check_error (status, "mkl_sparse_set_sv_hint");

  // SPARSE_MEMORY_AGRESSIVE should be default; other option is
  // SPARSE_MEMORY_NONE.
  sparse_memory_usage_t mem_policy = SPARSE_MEMORY_AGGRESSIVE;
  status = mkl_sparse_set_memory_hint (A, mem_policy);
  check_error (status, "mkl_sparse_set_memory_hint");
}

void
MklSparseMatrixHandle::
optimize ()
{
  if (! initialized_) {
    return; // this is harmless
  }
  sparse_matrix_t A;
  memcpy (&A, A_, sizeof (sparse_matrix_t));

  const sparse_status_t status = mkl_sparse_optimize (A);
  check_error (status, "mkl_sparse_optimize");
}

void
MklSparseMatrixHandle::
trsm (const Teuchos::ETransp trans,
      Teuchos::any alpha,
      const Teuchos::EUplo uplo,
      const Teuchos::EDiag diag,
      const void* x,
      int num_cols,
      int ldx,
      void* y,
      int ldy,
      const std::type_info& val_type)
{
  TEUCHOS_TEST_FOR_EXCEPTION
    (! initialized_, std::logic_error, "You may not call trsm "
     "unless the matrix is initialized.");

  sparse_matrix_t A;
  memcpy (&A, A_, sizeof (sparse_matrix_t));

  const auto op = transpose_operation (trans);
  const auto descr = triangular_solve_descr (uplo, diag);
  sparse_layout_t layout = SPARSE_LAYOUT_COLUMN_MAJOR;

  sparse_status_t status;
  if (val_type == typeid (double)) {
    double alpha_val = Teuchos::any_cast<double> (alpha);
    const double* x_val = reinterpret_cast<const double*> (x);
    double* y_val = reinterpret_cast<double*> (y);
    status = mkl_sparse_d_trsm (op, alpha_val, A, descr, layout,
                                x_val, num_cols, ldx, y_val, ldy);
  }
  else if (val_type == typeid (float)) {
    float alpha_val = Teuchos::any_cast<float> (alpha);
    const float* x_val = reinterpret_cast<const float*> (x);
    float* y_val = reinterpret_cast<float*> (y);
    status = mkl_sparse_s_trsm (op, alpha_val, A, descr, layout,
                                x_val, num_cols, ldx, y_val, ldy);
  }
  else if (val_type == typeid (Kokkos::complex<double>)) {
    MKL_Complex16 alpha_val = Teuchos::any_cast<MKL_Complex16> (alpha);
    const MKL_Complex16* x_val = reinterpret_cast<const MKL_Complex16*> (x);
    MKL_Complex16* y_val = reinterpret_cast<MKL_Complex16*> (y);
    status = mkl_sparse_z_trsm (op, alpha_val, A, descr, layout,
                                x_val, num_cols, ldx, y_val, ldy);
  }
  else if (val_type == typeid (Kokkos::complex<float>)) {
    MKL_Complex8 alpha_val = Teuchos::any_cast<MKL_Complex8> (alpha);
    const MKL_Complex8* x_val = reinterpret_cast<const MKL_Complex8*> (x);
    MKL_Complex8* y_val = reinterpret_cast<MKL_Complex8*> (y);
    status = mkl_sparse_c_trsm (op, alpha_val, A, descr, layout,
                                x_val, num_cols, ldx, y_val, ldy);
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION
      (true, std::logic_error, "Not implemented for the "
       "given Scalar type " << val_type.name ());
  }
  check_error (status, "mkl_sparse_?_trsm");
}

void
MklSparseMatrixHandle::
trsv (const Teuchos::ETransp trans,
      Teuchos::any alpha,
      const Teuchos::EUplo uplo,
      const Teuchos::EDiag diag,
      const void* x,
      void* y,
      const std::type_info& val_type)
{
  TEUCHOS_TEST_FOR_EXCEPTION
    (! initialized_, std::logic_error, "You may not call trsv "
     "unless the matrix is initialized.");

  sparse_matrix_t A;
  memcpy (&A, A_, sizeof (sparse_matrix_t));

  const auto op = transpose_operation (trans);
  const auto descr = triangular_solve_descr (uplo, diag);

  sparse_status_t status;
  if (val_type == typeid (double)) {
    double alpha_val = Teuchos::any_cast<double> (alpha);
    const double* x_val = reinterpret_cast<const double*> (x);
    double* y_val = reinterpret_cast<double*> (y);
    status = mkl_sparse_d_trsv (op, alpha_val, A, descr, x_val, y_val);
  }
  else if (val_type == typeid (float)) {
    float alpha_val = Teuchos::any_cast<float> (alpha);
    const float* x_val = reinterpret_cast<const float*> (x);
    float* y_val = reinterpret_cast<float*> (y);
    status = mkl_sparse_s_trsv (op, alpha_val, A, descr, x_val, y_val);
  }
  else if (val_type == typeid (Kokkos::complex<double>)) {
    MKL_Complex16 alpha_val = Teuchos::any_cast<MKL_Complex16> (alpha);
    const MKL_Complex16* x_val = reinterpret_cast<const MKL_Complex16*> (x);
    MKL_Complex16* y_val = reinterpret_cast<MKL_Complex16*> (y);
    status = mkl_sparse_z_trsv (op, alpha_val, A, descr, x_val, y_val);
  }
  else if (val_type == typeid (Kokkos::complex<float>)) {
    MKL_Complex8 alpha_val = Teuchos::any_cast<MKL_Complex8> (alpha);
    const MKL_Complex8* x_val = reinterpret_cast<const MKL_Complex8*> (x);
    MKL_Complex8* y_val = reinterpret_cast<MKL_Complex8*> (y);
    status = mkl_sparse_c_trsv (op, alpha_val, A, descr, x_val, y_val);
  }
  else {
    TEUCHOS_TEST_FOR_EXCEPTION
      (true, std::logic_error, "Not implemented for the "
       "given Scalar type " << val_type.name ());
  }
  check_error (status, "mkl_sparse_?_trsv");
}

} // namespace Impl
} // namespace MKL
} // namespace Details
} // namespace Ifpack2

#endif // HAVE_IFPACK2_MKL
