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

#include "Ifpack2_Details_MKL.hpp"
#ifdef HAVE_IFPACK2_MKL

#include "Kokkos_Core.hpp"
#include "Teuchos_any.hpp"
#include "Tpetra_Details_localExplicitTranspose.hpp"

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
class MklCrsMatrixTriangularSolver {
  static_assert (std::is_same<typename LocalCrsMatrixType::ordinal_type,
                              int>::value, "ordinal_type != int");
  using nc_offset_view_type =
    Kokkos::View<int*, Kokkos::DefaultHostExecutionSpace>;
  using offset_view_type =
    Kokkos::View<const int*, Kokkos::DefaultHostExecutionSpace>;

  static Teuchos::EUplo
  transposeUplo (const Teuchos::EUplo uplo)
  {
    if (uplo == Teuchos::UPPER_TRI) {
      return Teuchos::LOWER_TRI;
    }
    else if (uplo == Teuchos::LOWER_TRI) {
      return Teuchos::UPPER_TRI;
    }
    else {
      return uplo;
    }
  }

public:
  using device_type = typename LocalCrsMatrixType::device_type;
  using value_type = typename LocalCrsMatrixType::value_type;

private:
  //static constexpr bool use_explicit_transpose = false;
  static constexpr bool use_explicit_transpose = true;
  static constexpr bool is_complex =
    std::is_same<value_type, Kokkos::complex<double>>::value ||
    std::is_same<value_type, Kokkos::complex<float>>::value;

public:
  MklCrsMatrixTriangularSolver () = delete;

  MklCrsMatrixTriangularSolver (const LocalCrsMatrixType& A,
                                const Teuchos::ETransp trans,
                                const Teuchos::EUplo uplo,
                                const Teuchos::EDiag diag,
                                const int numExpectedCalls)
    : A_ (A),
      handle_ (new Impl::MklSparseMatrixHandle ()),
      xpose_handle_ (new Impl::MklSparseMatrixHandle ()),
      trans_ (trans),
      uplo_ (uplo),
      diag_ (diag),
      numExpectedCalls_ (numExpectedCalls)
  {}

  void setMatrix (const LocalCrsMatrixType& A) {
    A_ = A;
    ptr_.reset ();
    A_xpose_.reset ();
    xpose_ptr_.reset ();
    handle_->reset ();
    xpose_handle_->reset ();
  }

  void initialize () {
    ptr_ = makePtr (A_);
    A_xpose_.reset ();
    xpose_ptr_.reset ();
    handle_->reset (); // yes, -> not .
    xpose_handle_->reset (); // yes, -> not .
  }

  void compute ()
  {
    // MKL doesn't offer separate interfaces for symbolic vs. numeric
    // setup, so we defer all setup (except ptr_) until "compute".
    handle_->setCrsMatrix (A_.numRows (), A_.numCols (),
                           ptr_.get (), ptr_.get () + 1,
                           A_.graph.entries.data (),
                           A_.values.data (), typeid (value_type));
    handle_->setTriangularSolveHints (Teuchos::NO_TRANS, uplo_,
                                      diag_, numExpectedCalls_);
    handle_->optimize ();

    if (use_explicit_transpose) {
      computeExplicitTranspose ();
    }
  }

  void
  apply (Kokkos::View<const value_type**, Kokkos::LayoutLeft,
           device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> in,
         Kokkos::View<value_type**, Kokkos::LayoutLeft,
           device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> out,
         const Teuchos::ETransp mode,
         const value_type alpha) const
  {
    TEUCHOS_TEST_FOR_EXCEPTION
      (! handle_->initialized (), std::logic_error, "MKL handle is "
       "not initialized.  You must call initialize and compute "
       "before calling apply.");

    using KAT = Kokkos::ArithTraits<value_type>;
    if (alpha == KAT::zero ()) {
      Kokkos::deep_copy (out, KAT::zero ());
      return;
    }

    const int numCols (in.extent (1));
    Teuchos::any alpha_in (1.0);

    if (mode == Teuchos::NO_TRANS) {
      applyNoTrans (in, out, alpha_in, numCols);
    }
    else {
      applyTrans (in, out, alpha_in, mode, numCols);
    }

    if (alpha != Kokkos::ArithTraits<value_type>::one ()) {
      using execution_space = typename device_type::execution_space;
      using range_type = Kokkos::RangePolicy<execution_space, int>;
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

private:
  LocalCrsMatrixType A_;
  std::unique_ptr<LocalCrsMatrixType> A_xpose_;
  std::unique_ptr<int[]> ptr_;
  std::unique_ptr<int[]> xpose_ptr_;
  std::unique_ptr<Impl::MklSparseMatrixHandle> handle_;
  std::unique_ptr<Impl::MklSparseMatrixHandle> xpose_handle_;
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
    using Teuchos::NO_TRANS;
    const Teuchos::EUplo uplo = use_explicit_transpose ?
      transposeUplo (uplo_) : uplo_;

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
      if (use_explicit_transpose) {
        xpose_handle_->trsv (NO_TRANS, alpha, uplo, diag_,
                             in.data (), out.data (),
                             typeid (value_type));
      }
      else {
        handle_->trsv (mode, alpha, uplo, diag_, in.data (),
                       out.data (), typeid (value_type));
      }
    }
    else {
      if (use_explicit_transpose) {
        xpose_handle_->trsm (NO_TRANS, alpha, uplo, diag_, in.data (),
                             numCols, getStride (in), out.data (),
                             getStride (out), typeid (value_type));
      }
      else {
        handle_->trsm (mode, alpha, uplo_, diag_, in.data (), numCols,
                       getStride (in), out.data (), getStride (out),
                       typeid (value_type));
      }
    }
#endif // 0
  }

  void computeExplicitTranspose ()
  {
    // FIXME (mfh 22 Jul 2019) This approach is broken for complex
    // with trans_ == TRANS.
    TEUCHOS_ASSERT( ! (is_complex && trans_ == Teuchos::TRANS) );

    using Tpetra::Details::localExplicitTranspose;
    using LCMT = LocalCrsMatrixType;
    const bool conjugate =
      is_complex && trans_ == Teuchos::CONJ_TRANS;
    A_xpose_ = std::unique_ptr<LCMT>
      (new LCMT (localExplicitTranspose (A_, conjugate)));
    xpose_ptr_ = makePtr (*A_xpose_);
    xpose_handle_->setCrsMatrix (A_xpose_->numRows (),
                                 A_xpose_->numCols (),
                                 xpose_ptr_.get (),
                                 xpose_ptr_.get () + 1,
                                 A_xpose_->graph.entries.data (),
                                 A_xpose_->values.data (),
                                 typeid (value_type));
    const Teuchos::ETransp transposeTrans =
      is_complex ? Teuchos::CONJ_TRANS : Teuchos::TRANS;
    xpose_handle_->setTriangularSolveHints (transposeTrans,
                                            transposeUplo (uplo_),
                                            diag_,
                                            numExpectedCalls_);
    xpose_handle_->optimize ();
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
