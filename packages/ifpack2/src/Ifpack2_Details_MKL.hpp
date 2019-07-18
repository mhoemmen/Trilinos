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

#ifndef IFPACK2_DETAILS_MKL_HPP
#define IFPACK2_DETAILS_MKL_HPP

#include "Ifpack2_config.h"
#ifdef HAVE_IFPACK2_MKL

#include "Tpetra_Details_copyOffsets.hpp"
#include "Teuchos_any.hpp"
#include "Teuchos_BLAS_types.hpp"
#include <memory>
#include <typeinfo>

namespace Ifpack2 {
namespace Details {
namespace MKL {
namespace Impl {

template<class InputViewType>
void
copyOffsets (int* dst, const InputViewType& src)
{
  using output_view_type =
    Kokkos::View<int*,
                 Kokkos::DefaultHostExecutionSpace,
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  output_view_type dst_v (dst, src.extent (0));
  Tpetra::Details::copyOffsets (dst_v, src);
}

//! Whether MKL supports the value type for sparse matrix operations.
bool
mklSupportsValueType (const std::type_info& val_type);

/// \brief Opaque handle representing MKL's sparse_matrix_t.
///
/// sparse_matrix_t can represent a few different sparse matrix
/// storage formats, including compressed sparse row (CSR, or what
/// Trilinos insists on calling CRS for historical reasons).
class MklSparseMatrixHandle {
public:
  MklSparseMatrixHandle ();

  MklSparseMatrixHandle (const MklSparseMatrixHandle&) = delete;
  MklSparseMatrixHandle& operator= (const MklSparseMatrixHandle&) = delete;
  MklSparseMatrixHandle (MklSparseMatrixHandle&&) = delete;
  MklSparseMatrixHandle& operator= (MklSparseMatrixHandle&&) = delete;

  ~MklSparseMatrixHandle ();

  /// \brief Make an MKL sparse_matrix_t that represents a sparse
  ///   matrix stored in compressed sparse row format.
  //
  /// Trilinos insists on calling abbreviating compressed sparse row
  /// as CRS for historical reasons.
  void
  setCrsMatrix (int num_rows,
                int num_cols,
                int* beg,
                int* end,
                int* ind,
                void* val,
                const std::type_info& val_type);
  void reset ();
  void
  setTriangularSolveHints (const Teuchos::ETransp trans,
                           const Teuchos::EUplo uplo,
                           const Teuchos::EDiag diag,
                           const int num_expected_calls);
  void optimize ();

  bool initialized () const;

  //! Sparse triangular solve with column-major multivectors.
  void
  trsm (const Teuchos::ETransp trans,
        Teuchos::any alpha,
        const Teuchos::EUplo uplo,
        const Teuchos::EDiag diag,
        const void* x,
        int num_cols,
        int ldx,
        void* y,
        int ldy,
        const std::type_info& val_type);

  //! Sparse triangular solve with contiguous vectors.
  void
  trsv (const Teuchos::ETransp trans,
        Teuchos::any alpha,
        const Teuchos::EUplo uplo,
        const Teuchos::EDiag diag,
        const void* x,
        void* y,
        const std::type_info& val_type);

private:
  void* A_;
  bool initialized_ = false;
};

} // namespace Impl
} // namespace MKL
} // namespace Details
} // namespace Ifpack2

#endif // HAVE_IFPACK2_MKL

#endif // IFPACK2_DETAILS_MKL_HPP
