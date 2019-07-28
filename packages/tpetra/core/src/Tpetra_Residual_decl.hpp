// @HEADER
// ***********************************************************************
//
//          Tpetra: Templated Linear Algebra Services Package
//                 Copyright (2008) Sandia Corporation
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

/// \file Tpetra_Residual_decl.hpp
/// \brief Declaration of Tpetra::Residual

#ifndef TPETRA_RESIDUAL_DECL_HPP
#define TPETRA_RESIDUAL_DECL_HPP

#include "Tpetra_CrsMatrix_fwd.hpp"
#include "Tpetra_MultiVector_fwd.hpp"
#include "Tpetra_Operator_fwd.hpp"
#include "Tpetra_Vector_fwd.hpp"
#include "Tpetra_Export_fwd.hpp"
#include "Tpetra_Import_fwd.hpp"
#include "Teuchos_RCP.hpp"
#include <memory>

namespace Tpetra {

/// \brief Compute residual vector and (optionally) 2-norm.
template<class SC, class LO, class GO, class NT>
class Residual {
private:
  using crs_matrix_type = Tpetra::CrsMatrix<SC, LO, GO, NT>;
  using multivector_type = Tpetra::MultiVector<SC, LO, GO, NT>;
  using operator_type = Tpetra::Operator<SC, LO, GO, NT>;
  using vector_type = Tpetra::Vector<SC, LO, GO, NT>;

public:
  Residual (const Teuchos::RCP<const operator_type>& A);

  void setMatrix (const Teuchos::RCP<const operator_type>& A);

  //! Compute W := B - A*X, and return 2-norm(s) of column(s) of W.
  Teuchos::ArrayView<const typename vector_type::mag_type>
  computeWithNorms (multivector_type& W,
                    multivector_type& B,
                    multivector_type& X);

  //! Compute W := B - A*X.
  void
  computeWithoutNorms (multivector_type& W,
                       multivector_type& B,
                       multivector_type& X);

private:
  using import_type = Tpetra::Import<LO, GO, NT>;
  using export_type = Tpetra::Export<LO, GO, NT>;

  Teuchos::RCP<const operator_type> A_op_;
  Teuchos::RCP<const crs_matrix_type> A_crs_;
  Teuchos::RCP<const import_type> imp_;
  Teuchos::RCP<const export_type> exp_;
  //std::unique_ptr<vector_type> X_colMap_;
  std::vector<typename vector_type::mag_type> norms_;

  // Do the Import, if needed, and return the column Map version of X.
  vector_type importVector (vector_type& X_domMap);

  bool canFuse (const multivector_type& B) const;

  Teuchos::ArrayView<const typename vector_type::mag_type>
  unfusedCaseWithNorms (const char kernel_label[],
                        multivector_type& W,
                        multivector_type& B,
                        const operator_type& A,
                        multivector_type& X);
  void
  unfusedCaseWithoutNorms (const char kernel_label[],
                           multivector_type& W,
                           multivector_type& B,
                           const operator_type& A,
                           multivector_type& X);

  Teuchos::ArrayView<const typename vector_type::mag_type>
  fusedCaseWithNorms (const char kernel_label[],
                      vector_type& W,
                      vector_type& B,
                      const crs_matrix_type& A,
                      vector_type& X);
  void
  fusedCaseWithoutNorms (const char kernel_label[],
                         vector_type& W,
                         vector_type& B,
                         const crs_matrix_type& A,
                         vector_type& X);
};

} // namespace Tpetra

#endif // TPETRA_RESIDUAL_DECL_HPP
