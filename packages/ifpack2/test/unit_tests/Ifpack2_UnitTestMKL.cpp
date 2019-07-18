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

#include "Ifpack2_config.h"
#ifdef HAVE_IFPACK2_MKL
#  include "Teuchos_UnitTestHarness.hpp"
#  include "Ifpack2_Details_MKL.hpp"

namespace { // (anonymous)

TEUCHOS_UNIT_TEST(TPL_MKL, SparseTriangularSolve)
{
  using Ifpack2::Details::MKL::Impl::mklSupportsValueType;
  TEST_ASSERT( mklSupportsValueType (typeid (double)) );
  TEST_ASSERT( mklSupportsValueType (typeid (float)) );
  TEST_ASSERT( mklSupportsValueType (typeid (Kokkos::complex<double>)) );
  TEST_ASSERT( mklSupportsValueType (typeid (Kokkos::complex<float>)) );

  const int numExpectedCalls = 100;
  using Teuchos::NO_TRANS;
  using Teuchos::TRANS;
  using Teuchos::UPPER_TRI;
  using Teuchos::LOWER_TRI;
  using Teuchos::NON_UNIT_DIAG;
  using Teuchos::UNIT_DIAG;

  {
    // Upper triangular matrix [1, 1; 0, 1].
    const int num_rows = 2;
    const int num_cols = 2;
    std::vector<int> ptr {{0, 2, 3}};
    std::vector<int> ind {{0, 1, 1}};
    std::vector<double> val {{1.0, 1.0, 1.0}};

    using Ifpack2::Details::MKL::Impl::MklSparseMatrixHandle;
    MklSparseMatrixHandle handle;
    handle.setCrsMatrix (2, 2, ptr.data (), ptr.data () + 1,
                         ind.data (), val.data (), typeid (double));
    handle.setTriangularSolveHints (NO_TRANS, UPPER_TRI,
                                    NON_UNIT_DIAG, numExpectedCalls);

    std::vector<double> x {{667.0, 1.0}};
    std::vector<double> y (num_cols);
    Teuchos::any alpha (1.0);
    handle.trsv (NO_TRANS, alpha, UPPER_TRI, NON_UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == 666.0 );
    TEST_ASSERT( y[1] == 1.0 );

    handle.trsv (TRANS, alpha, UPPER_TRI, NON_UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == 667.0 );
    TEST_ASSERT( y[1] == -666.0 );
  }

  {
    // Upper triangular matrix [1, 1; 0, 1], with implicit unit
    // diagonal.
    const int num_rows = 2;
    const int num_cols = 2;
    std::vector<int> ptr {{0, 1, 1}};
    std::vector<int> ind {{1}};
    std::vector<double> val {{1.0}};

    using Ifpack2::Details::MKL::Impl::MklSparseMatrixHandle;
    MklSparseMatrixHandle handle;
    handle.setCrsMatrix (2, 2, ptr.data (), ptr.data () + 1,
                         ind.data (), val.data (), typeid (double));
    handle.setTriangularSolveHints (NO_TRANS, UPPER_TRI,
                                    UNIT_DIAG, numExpectedCalls);

    std::vector<double> x {{667.0, 1.0}};
    std::vector<double> y (num_cols);
    Teuchos::any alpha (1.0);
    handle.trsv (NO_TRANS, alpha, UPPER_TRI, UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == 666.0 );
    TEST_ASSERT( y[1] == 1.0 );

    handle.trsv (TRANS, alpha, UPPER_TRI, UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == 667.0 );
    TEST_ASSERT( y[1] == -666.0 );
  }

  {
    // Lower triangular matrix [1, 0; 1, 1].
    const int num_rows = 2;
    const int num_cols = 2;
    std::vector<int> ptr {{0, 1, 3}};
    std::vector<int> ind {{0, 0, 1}};
    std::vector<double> val {{1.0, 1.0, 1.0}};

    using Ifpack2::Details::MKL::Impl::MklSparseMatrixHandle;
    MklSparseMatrixHandle handle;
    handle.setCrsMatrix (2, 2, ptr.data (), ptr.data () + 1,
                         ind.data (), val.data (), typeid (double));
    handle.setTriangularSolveHints (NO_TRANS, LOWER_TRI,
                                    NON_UNIT_DIAG, numExpectedCalls);

    std::vector<double> x {{667.0, 1.0}};
    std::vector<double> y (num_cols);
    Teuchos::any alpha (1.0);
    handle.trsv (NO_TRANS, alpha, LOWER_TRI, NON_UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == 667.0 );
    TEST_ASSERT( y[1] == -666.0 );

    handle.trsv (TRANS, alpha, LOWER_TRI, NON_UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == 666.0 );
    TEST_ASSERT( y[1] == 1.0 );
  }

  {
    // Lower triangular matrix [1, 0; 1, 1], with implicit unit
    // diagonal.
    const int num_rows = 2;
    const int num_cols = 2;
    std::vector<int> ptr {{0, 0, 1}};
    std::vector<int> ind {{0}};
    std::vector<double> val {{1.0}};

    using Ifpack2::Details::MKL::Impl::MklSparseMatrixHandle;
    MklSparseMatrixHandle handle;
    handle.setCrsMatrix (2, 2, ptr.data (), ptr.data () + 1,
                         ind.data (), val.data (), typeid (double));
    handle.setTriangularSolveHints (NO_TRANS, LOWER_TRI,
                                    UNIT_DIAG, numExpectedCalls);

    std::vector<double> x {{667.0, 1.0}};
    std::vector<double> y (num_cols);
    Teuchos::any alpha (1.0);
    handle.trsv (NO_TRANS, alpha, LOWER_TRI, UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == 667.0 );
    TEST_ASSERT( y[1] == -666.0 );

    handle.trsv (TRANS, alpha, LOWER_TRI, UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == 666.0 );
    TEST_ASSERT( y[1] == 1.0 );

    alpha = Teuchos::any (2.0);
    handle.trsv (TRANS, alpha, LOWER_TRI, UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == 1332.0 );
    TEST_ASSERT( y[1] == 2.0 );

    alpha = Teuchos::any (-1.0);
    handle.trsv (TRANS, alpha, LOWER_TRI, UNIT_DIAG,
                 x.data (), y.data (), typeid (double));

    out << "y: " << y[0] << ", " << y[1] << std::endl;

    TEST_ASSERT( y[0] == -666.0 );
    TEST_ASSERT( y[1] == -1.0 );
  }
}

} // namespace (anonymous)

#endif // HAVE_IFPACK2_MKL
