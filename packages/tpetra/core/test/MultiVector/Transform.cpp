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

#include "Tpetra_TestingUtilities.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_Map.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Vector.hpp"
#include "Tpetra_transform_MultiVector.hpp"
#include "Tpetra_Details_Behavior.hpp"
#include "Kokkos_ArithTraits.hpp"

namespace { // (anonymous)

  using Tpetra::getDefaultComm;
  using Teuchos::outArg;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::REDUCE_MIN;
  using Teuchos::reduceAll;
  using std::endl;
  using GST = Tpetra::global_size_t;
  using map_type = Tpetra::Map<>;
  using multivec_type = Tpetra::MultiVector<>;
  using vec_type = Tpetra::Vector<>;
  using GO = map_type::global_ordinal_type;

  //
  // UNIT TESTS
  //

  TEUCHOS_UNIT_TEST( VectorHarness, Transform )
  {
    using Tpetra::transform;
    using Kokkos::ALL;
    using std::endl;
    using device_execution_space =
      typename multivec_type::device_type::execution_space;
    using LO = typename multivec_type::local_ordinal_type;
    const bool debug = ::Tpetra::Details::Behavior::debug ();
    int lclSuccess = 0;
    int gblSuccess = 0;

    RCP<Teuchos::FancyOStream> outPtr = debug ?
      Teuchos::getFancyOStream (Teuchos::rcpFromRef (std::cerr)) :
      Teuchos::rcpFromRef (out);
    Teuchos::FancyOStream& myOut = *outPtr;

    myOut << "Test Tpetra::transform" << endl;
    Teuchos::OSTab tab0 (myOut);

    myOut << "Create a Map" << endl;
    auto comm = getDefaultComm ();
    const auto INVALID = Teuchos::OrdinalTraits<GST>::invalid ();
    const size_t numLocal = 13;
    const size_t numVecs  = 3;
    const GO indexBase = 0;
    auto map = rcp (new map_type (INVALID, numLocal, indexBase, comm));

    myOut << "Create MultiVectors" << endl;
    multivec_type X (map, numVecs);
    TEST_EQUALITY( X.getNumVectors (), numVecs );
    multivec_type Y (map, numVecs);
    TEST_EQUALITY( Y.getNumVectors (), numVecs );

    out << "Set entries of X and Y" << endl;
    constexpr double flagValue = -1.0;
    X.putScalar (flagValue);
    Y.putScalar (666.0);

    out << "transform on default execution space: -1 -> (666+1=667)" << endl;
    transform ("-1 -> (666+1=667)", Y, X,
               KOKKOS_LAMBDA (const double& X_ij) { return X_ij + 1.0; });
    {
      X.sync_host ();
      auto X_lcl = X.getLocalViewHost ();
      for (LO j = 0; j < LO (X.getNumVectors ()); ++j) {
        out << "Column " << j << endl;
        bool ok = true;
        for (LO i = 0; i < LO (X.getLocalLength ()); ++i) {
          const double expectedVal = 667.0;
          if (X_lcl(i,j) != expectedVal) {
            out << "X_lcl(" << i << "," << j << ") = " << X_lcl(i,j)
                << " != " << expectedVal << endl;
            ok = false;
          }
        }
        TEST_ASSERT( ok );
      }
    }

    lclSuccess = success ? 1 : 0;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
    if (gblSuccess != 1) {
      out << "Returning early" << endl;
      return;
    }

    Y.putScalar (418.0);
    out << "transform on default host execution space: 418 -> 419" << endl;

    transform ("418 -> 419", Kokkos::DefaultHostExecutionSpace (), Y, X,
               KOKKOS_LAMBDA (const double& X_ij) { return X_ij + 1.0; });
    {
      // no promise that Y was actually sync'd to host, merely that Y
      // could be accessed from the host execution space
      Y.sync_host ();
      
      auto Y_lcl = Y.getLocalViewHost ();
      for (LO j = 0; j < LO (Y.getNumVectors ()); ++j) {
        out << "Column " << j << endl;
        bool ok = true;
        for (LO i = 0; i < LO (Y.getLocalLength ()); ++i) {
          const double expectedVal = 418.0;
          if (Y_lcl(i,j) != expectedVal) {
            out << "Y_lcl(" << i << "," << j << ") = " << Y_lcl(i,j)
                << " != " << expectedVal << endl;
            ok = false;
          }
        }
        TEST_ASSERT( ok );
      }
      
      //X.sync_host (); // should not be needed here
      
      auto X_lcl = X.getLocalViewHost ();
      for (LO j = 0; j < LO (X.getNumVectors ()); ++j) {
        out << "Column " << j << endl;
        bool ok = true;
        for (LO i = 0; i < LO (X.getLocalLength ()); ++i) {
          const double expectedVal = 419.0;
          if (X_lcl(i,j) != expectedVal) {
            out << "X_lcl(" << i << "," << j << ") = " << X_lcl(i,j)
                << " != " << expectedVal << endl;
            ok = false;
          }
        }
        TEST_ASSERT( ok );
      }
    }

    lclSuccess = success ? 1 : 0;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
    if (gblSuccess != 1) {
      out << "Returning early" << endl;
      return;
    }

    out << "transform on MultiVector's execution space: 418 -> 777" << endl;
    transform ("419 -> 777", device_execution_space (), Y, X,
               KOKKOS_LAMBDA (const double& X_ij) { return X_ij + 359.0; });
    {
      Y.sync_host ();
      auto Y_lcl = Y.getLocalViewHost ();
      for (LO j = 0; j < LO (Y.getNumVectors ()); ++j) {
        out << "Column " << j << endl;
        bool ok = true;
        for (LO i = 0; i < LO (Y.getLocalLength ()); ++i) {
          const double expectedVal = 418.0;
          if (Y_lcl(i,j) != expectedVal) {
            out << "Y_lcl(" << i << "," << j << ") = " << Y_lcl(i,j)
                << " != " << expectedVal << endl;
            ok = false;
          }
        }
        TEST_ASSERT( ok );
      }
      
      X.sync_host ();
      auto X_lcl = X.getLocalViewHost ();
      for (LO j = 0; j < LO (X.getNumVectors ()); ++j) {
        out << "Column " << j << endl;
        bool ok = true;
        for (LO i = 0; i < LO (X.getLocalLength ()); ++i) {
          const double expectedVal = 777.0;
          if (X_lcl(i,j) != expectedVal) {
            out << "X_lcl(" << i << "," << j << ") = " << X_lcl(i,j)
                << " != " << expectedVal << endl;
            ok = false;
          }
        }
        TEST_ASSERT( ok );
      }
    }

    lclSuccess = success ? 1 : 0;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_ASSERT( gblSuccess == 1 );
    if (gblSuccess != 1) {
      out << "Returning early" << endl;
      return;
    }
  }

} // namespace (anonymous)

int
main (int argc, char* argv[])
{
  Tpetra::ScopeGuard tpetraScope (&argc, &argv);
  const int errCode =
    Teuchos::UnitTestRepository::runUnitTestsFromMain (argc, argv);
  return errCode;
}
