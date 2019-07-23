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

#ifndef IFPACK2_DETAILS_MKLCRSMATRIXTRIANGULARSOLVER_HPP
#define IFPACK2_DETAILS_MKLCRSMATRIXTRIANGULARSOLVER_HPP

#include "Ifpack2_config.h"
#ifdef HAVE_IFPACK2_MKL

#include "Ifpack2_Details_MKL.hpp"
#include "Ifpack2_Details_LocalSolver.hpp"
#include "Kokkos_Core.hpp"
#include "Teuchos_any.hpp"

namespace Ifpack2 {
namespace Details {
namespace MKL {

/// \brief Wrapper for a local sparse triangular solver that uses MKL
///   with a sparse matrix stored in compressed sparse row format.
///
/// \tparam LocalCrsMatrixType Specialization of
///   KokkosSparse::CrsMatrix.
///
/// Trilinos insists on calling abbreviating compressed sparse row as
/// CRS for historical reasons.
template<class LocalCrsMatrixType>
class MklCrsMatrixTriangularSolver :
    public ::Ifpack2::Details::LocalSolver<LocalCrsMatrixType> {
  static_assert (std::is_same<typename LocalCrsMatrixType::ordinal_type,
                              int>::value, "ordinal_type != int");
  using nc_offset_view_type =
    Kokkos::View<int*, Kokkos::DefaultHostExecutionSpace>;
  using offset_view_type =
    Kokkos::View<const int*, Kokkos::DefaultHostExecutionSpace>;

public:
  using device_type = typename LocalCrsMatrixType::device_type;
  using value_type = typename LocalCrsMatrixType::value_type;

  MklCrsMatrixTriangularSolver () = delete;

  MklCrsMatrixTriangularSolver (const LocalCrsMatrixType& A,
                                const Teuchos::ETransp trans,
                                const Teuchos::EUplo uplo,
                                const Teuchos::EDiag diag,
                                const int numExpectedCalls)
    : A_ (A),
      handle_ (new Impl::MklSparseMatrixHandle ()),
      trans_ (trans),
      uplo_ (uplo),
      diag_ (diag),
      numExpectedCalls_ (numExpectedCalls)
  {}

  ~MklCrsMatrixTriangularSolver () override = default;

  void setMatrix (const LocalCrsMatrixType& A) override {
    A_ = A;
    ptr_.reset ();
    handle_->reset ();
  }

  void initialize () override {
    ptr_ = makePtr (A_);
    handle_->reset (); // yes, -> not .
  }

  void compute () override
  {
    // MKL doesn't offer separate interfaces for symbolic vs. numeric
    // setup, so we defer all setup (except ptr_) until "compute".
    handle_->setCrsMatrix (A_.numRows (), A_.numCols (),
                           ptr_.get (), ptr_.get () + 1,
                           A_.graph.entries.data (),
                           A_.values.data (), typeid (value_type));
    handle_->setTriangularSolveHints (trans_, uplo_, diag_,
                                      numExpectedCalls_);
    handle_->optimize ();
  }

  void
  apply (Kokkos::View<const value_type**, Kokkos::LayoutLeft,
           device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> in,
         Kokkos::View<value_type**, Kokkos::LayoutLeft,
           device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> out,
         const Teuchos::ETransp mode,
         const value_type alpha,
         const value_type beta) const override
  {
    TEUCHOS_TEST_FOR_EXCEPTION
      (! handle_->initialized (), std::logic_error, "MKL handle is "
       "not initialized.  You must call initialize and compute "
       "before calling apply.");

    using execution_space = typename device_type::execution_space;
    using range_type = Kokkos::RangePolicy<execution_space, int>;
    using KAT = Kokkos::ArithTraits<value_type>;

    const int numCols (in.extent (1));
    Teuchos::any alpha_in (1.0);

    if (alpha == KAT::zero ()) {
      if (beta == KAT::zero ()) {
        Kokkos::deep_copy (out, KAT::zero ());
      }
      else if (beta != KAT::one ()) {
        Kokkos::parallel_for
          ("Scale triangular solve result",
           range_type (0, int (out.extent (0))),
           [&] (const int i) {
            for (int j = 0; j < numCols; ++j) {
              out(i,j) *= beta;
            }
          });
      }
      return;
    }

    if (beta == KAT::zero ()) {
      if (mode == Teuchos::NO_TRANS) {
        applyNoTrans (in, out, alpha_in, numCols);
      }
      else {
        applyTrans (in, out, alpha_in, mode, numCols);
      }
      if (alpha != KAT::one ()) {
        Kokkos::parallel_for
          ("Scale triangular solve result",
           range_type (0, int (out.extent (0))),
           [&] (const int i) {
            for (int j = 0; j < numCols; ++j) {
              out(i,j) *= alpha;
            }
          });
      }
    }
    else {
      if (out_tmp_.extent (0) != out.extent (0) ||
          out_tmp_.extent (1) != out.extent (1)) {
        using view_type =
          Kokkos::View<value_type**, Kokkos::LayoutLeft, device_type>;
        using Kokkos::view_alloc;
        using Kokkos::WithoutInitializing;
        out_tmp_ = view_type ();
        out_tmp_ = view_type (view_alloc ("out_tmp", WithoutInitializing),
                              out.extent (0), out.extent (1));
      }

      if (mode == Teuchos::NO_TRANS) {
        applyNoTrans (in, out_tmp_, alpha_in, numCols);
      }
      else {
        applyTrans (in, out_tmp_, alpha_in, mode, numCols);
      }

      Kokkos::parallel_for
        ("Scale triangular solve result",
         range_type (0, int (out_tmp_.extent (0))),
         [&] (const int i) {
          for (int j = 0; j < numCols; ++j) {
            out(i,j) = beta * out(i,j) + alpha * out_tmp_(i,j);
          }
        });
    }
  }

private:
  LocalCrsMatrixType A_;
  std::unique_ptr<int[]> ptr_;
  std::unique_ptr<Impl::MklSparseMatrixHandle> handle_;
  mutable Kokkos::View<value_type**, Kokkos::LayoutLeft, device_type> out_tmp_;

  Teuchos::ETransp trans_ = Teuchos::NO_TRANS;
  Teuchos::EUplo uplo_ = Teuchos::UNDEF_TRI;
  Teuchos::EDiag diag_ = Teuchos::UNIT_DIAG;
  int numExpectedCalls_ = 0;

  void
  applyNoTrans (Kokkos::View<const value_type**, Kokkos::LayoutLeft,
                  device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> in,
                Kokkos::View<value_type**, Kokkos::LayoutLeft,
                  device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> out,
                Teuchos::any& alpha,
                const int numCols) const
  {
    using Teuchos::NO_TRANS;
    if (numCols == 1) {
      handle_->trsv (NO_TRANS, alpha, uplo_, diag_,
                     in.data (), out.data (),
                     typeid (value_type));
    }
    else {
      handle_->trsm (NO_TRANS, alpha, uplo_, diag_,
                     in.data (), numCols, getStride (in),
                     out.data (), getStride (out),
                     typeid (value_type));
    }
  }

  void
  applyTrans (Kokkos::View<const value_type**, Kokkos::LayoutLeft,
                device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> in,
              Kokkos::View<value_type**, Kokkos::LayoutLeft,
                device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> out,
              Teuchos::any& alpha,
              const Teuchos::ETransp mode,
              const int numCols) const
  {
    // sequential fallback
    {
      const std::string trans_str = (trans_ == Teuchos::CONJ_TRANS) ?
        "C" : (mode == Teuchos::TRANS ? "T" : "N");
      const std::string uplo_str =
        uplo_ == Teuchos::UPPER_TRI ? "U" : "L";
      const std::string diag_str = diag_ == Teuchos::UNIT_DIAG ? "U" : "N";
      KokkosSparse::trsv (uplo_str.c_str (), trans_str.c_str (),
                          diag_str.c_str (), A_, in, out);
      return;
    }

#if 0
    if (numCols == 1) {
      handle_->trsv (mode, alpha, uplo, diag_, in.data (),
                     out.data (), typeid (value_type));
    }
    else {
      handle_->trsm (mode, alpha, uplo_, diag_, in.data (), numCols,
                     getStride (in), out.data (), getStride (out),
                     typeid (value_type));
    }
#endif // 0
  }

  static int
  getStride (Kokkos::View<const value_type**, Kokkos::LayoutLeft,
               device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> in)
  {
    return in.stride (1) == 0 ? 1 : int (in.stride (1));
  }

  static std::unique_ptr<int[]>
  makePtr (const LocalCrsMatrixType& A)
  {
    std::unique_ptr<int[]> ptr (new int [A.graph.row_map.extent (0)]);
    Impl::copyOffsets (ptr.get (), A.graph.row_map);
    return ptr;
  }
};

} // namespace MKL
} // namespace Details
} // namespace Ifpack2

#endif // HAVE_IFPACK2_MKL

#endif // IFPACK2_DETAILS_MKLCRSMATRIXTRIANGULARSOLVER_HPP
