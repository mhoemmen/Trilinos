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

#include "Tpetra_TestingUtilities.hpp"
#include "Tpetra_Details_CooMatrix.hpp"
#include "Tpetra_Details_gathervPrint.hpp"
#ifdef HAVE_TPETRACORE_MPI
#  include "Tpetra_Details_extractMpiCommFromTeuchos.hpp"
#  include "Teuchos_DefaultMpiComm.hpp"
#endif // HAVE_TPETRACORE_MPI

namespace { // (anonymous)

using Teuchos::Comm;
using Teuchos::RCP;
using Teuchos::rcp;
using std::endl;
typedef double SC;
typedef Tpetra::DistObject<char>::local_ordinal_type LO;
typedef Tpetra::DistObject<char>::global_ordinal_type GO;
typedef Tpetra::global_size_t GST;
typedef Tpetra::Export<> export_type;
typedef Tpetra::Map<> map_type;

void
testCooMatrix (bool& success,
               Teuchos::FancyOStream& out,
               const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  using Tpetra::Details::CooMatrix;
  using Teuchos::outArg;
  using Teuchos::REDUCE_MIN;
  using Teuchos::reduceAll;
  int lclSuccess = 1;
  int gblSuccess = 0; // output argument

  out << "Test CooMatrix" << endl;
  Teuchos::OSTab tab1 (out);

  TEST_ASSERT( comm->getSize () >= 2 );
  if (comm->getSize () < 2) {
    out << "This test needs at least 2 MPI processes!" << endl;
    return;
  }

  out << "CooMatrix default constructor" << endl;
  CooMatrix<SC, LO, GO> A_in;
  TEST_ASSERT( A_in.getMap ().is_null () );
  TEST_EQUALITY( A_in.getLclNumEntries (), static_cast<std::size_t> (0) );

  out << "Add entries locally to CooMatrix" << endl;
  const int myRank = comm->getRank ();
  if (myRank == 0) {
    A_in.sumIntoGlobalValues ({666, 31, 31, 31}, {11, 6, 5, 6}, {-1.0, 1.0, 2.0, 111.0});
    TEST_EQUALITY( A_in.getLclNumEntries (), static_cast<std::size_t> (3) );
  }
  else if (myRank == 1) {
    A_in.sumIntoGlobalValues ({418, 31}, {11, 5}, {11.0, 5.0});
    TEST_EQUALITY( A_in.getLclNumEntries (), static_cast<std::size_t> (2) );
  }

  lclSuccess = success ? 1 : 0;
  gblSuccess = 0; // output argument
  reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
  TEST_EQUALITY( gblSuccess, 1 );
  if (gblSuccess != 1) {
    out << "A_in not in a consistent state before fillComplete, "
      "so don't bother continuing the test." << endl;
    return;
  }

  out << "Call fillComplete on CooMatrix" << endl;
  A_in.fillComplete (comm);
  TEST_ASSERT( ! A_in.getMap ().is_null () );

  out << "Create output Map" << endl;
  RCP<const map_type> outMap;
  const GO indexBase = 31; // the smallest global index in the Map
  const GST numGblInds = 3;
  if (myRank == 0) {
    const GO myGblInds[] = {418, 666};
    const LO numLclInds = 2;
    outMap = rcp (new map_type (numGblInds, myGblInds, numLclInds, indexBase, comm));
  }
  else if (myRank == 1) {
    const GO myGblInds[] = {31};
    const LO numLclInds = 1;
    outMap = rcp (new map_type (numGblInds, myGblInds, numLclInds, indexBase, comm));
  }
  else {
    const GO* myGblInds = NULL;
    const LO numLclInds = 0;
    outMap = rcp (new map_type (numGblInds, myGblInds, numLclInds, indexBase, comm));
  }

  out << "Create output CooMatrix" << endl;
  CooMatrix<SC, LO, GO> A_out (outMap);
  TEST_EQUALITY( A_out.getLclNumEntries (), static_cast<std::size_t> (0) );
  TEST_ASSERT( ! A_out.getMap ().is_null () );
  const bool outMapsSame = outMap->isSameAs (* (A_out.getMap ()));
  TEST_ASSERT( outMapsSame );
  const bool outMapIsOneToOne = outMap->isOneToOne ();
  TEST_ASSERT( outMapIsOneToOne );

  out << "Create Export object" << endl;
  export_type exporter (A_in.getMap (), A_out.getMap ());

  out << "Call doExport on CooMatrix" << endl;
  A_out.doExport (A_in, exporter, Tpetra::ADD);

  out << "Test global success" << endl;
  lclSuccess = (success && ! A_out.localError ()) ? 1 : 0;
  gblSuccess = 0; // output argument
  reduceAll<int, int> (*comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));

  TEST_EQUALITY( gblSuccess, 1 );
  if (gblSuccess != 1) {
    std::ostringstream os;
    os << "Process " << myRank << ": " << A_out.errorMessages () << endl;
    Tpetra::Details::gathervPrint (out, os.str (), *comm);
  }
}

TEUCHOS_UNIT_TEST( CooMatrix, doubleIntLongLong )
{
  using Tpetra::TestingUtilities::getDefaultComm;
  RCP<const Comm<int> > comm = getDefaultComm ();
  TEST_ASSERT( ! comm.is_null () );
  if (comm.is_null ()) {
    return;
  }

  // Throw away map, just to make sure that Kokkos is initialized
  RCP<const map_type> throwaway_map;
  throwaway_map = rcp(new map_type(static_cast<GST>(0),
                                   static_cast<GO>(0),
                                   comm));

#ifdef HAVE_TPETRACORE_MPI
  // Set the MPI error handler so that errors return, instead of
  // immediately causing MPI_Abort.  This will help us catch any bugs
  // with how we use, e.g., MPI_Pack and MPI_Unpack.
  {
    using Teuchos::MpiComm;
    using Teuchos::rcp_const_cast;
    using Teuchos::rcp_dynamic_cast;

    constexpr bool throwOnFail = true;
    auto mpiComm = rcp_dynamic_cast<const MpiComm<int> > (comm, throwOnFail);
    // We have to cast away const to call setErrorHandler.
    auto mpiCommNonConst = rcp_const_cast<MpiComm<int> > (mpiComm);
    auto errHandler =
      rcp (new Teuchos::OpaqueWrapper<MPI_Errhandler> (MPI_ERRORS_RETURN));
    mpiCommNonConst->setErrorHandler (errHandler);
  }
#endif // HAVE_TPETRACORE_MPI


  testCooMatrix (success, out, comm);
}

std::ostream&
operator<< (std::ostream& out, const Tpetra::Details::EReadEntryResult result)
{
  if (result == Tpetra::Details::READ_ENTRY_VALID) {
    out << "READ_ENTRY_VALID";
  }
  else if (result == Tpetra::Details::READ_ENTRY_EMPTY) {
    out << "READ_ENTRY_EMPTY";
  }
  else if (result == Tpetra::Details::READ_ENTRY_ERROR) {
    out << "READ_ENTRY_ERROR";
  }
  return out;
}

#define TPETRA_DETAILS_TEST_READENTRY( expectedResult, expectedRowInd, expectedColInd, expectedVal ) \
  do { \
    out << "Expect readCooMatrixEntry to return " << expectedResult; \
    Teuchos::OSTab tabInner (out); \
    if (expectedResult == Tpetra::Details::READ_ENTRY_VALID) {          \
      out << ", " << expectedRowInd << ", " << expectedColInd << ", " << expectedVal; \
    } \
    out << std::endl; \
    std::getline (is, line); \
    out << "Got line: " << line << std::endl; \
    EReadEntryResult result; \
    bool threw = false; \
    try { \
      result = readCooMatrixEntry (rowInd, colInd, val, line); \
    } \
    catch (std::exception& e) { \
      threw = true; \
      out << "readCooMatrixEntry threw an exception: " << e.what () << std::endl; \
    } \
    catch (...) { \
      threw = true; \
      out << "readCooMatrixEntry threw an exception not a subclass of std::exception" << std::endl; \
    } \
    TEST_ASSERT( ! threw ); \
    if (! threw) { \
      TEST_ASSERT( result == expectedResult ); \
      if (result != expectedResult) { \
	out << "Got wrong result " << result << std::endl; \
      } \
      if (result == READ_ENTRY_VALID) { \
	out << "Got entry: " << rowInd << ", " << colInd << ", " << val << std::endl; \
      } \
      if (expectedResult == READ_ENTRY_VALID) { \
        TEST_ASSERT( rowInd == expectedRowInd && colInd == expectedColInd && val == expectedVal ); \
      } \
    } \
  } while (false);

TEUCHOS_UNIT_TEST( CooMatrix, ReadEntry_iid )
{
  using Tpetra::Details::EReadEntryResult;
  using Tpetra::Details::READ_ENTRY_VALID;
  using Tpetra::Details::READ_ENTRY_ERROR;
  using Tpetra::Details::READ_ENTRY_EMPTY;
  using Tpetra::Details::readCooMatrixEntry;
  using Tpetra::Details::Impl::is_whitespace;

  TEST_ASSERT( is_whitespace ('\t') );
  TEST_ASSERT( is_whitespace (' ') );
  TEST_ASSERT( ! is_whitespace ('8') );
  TEST_ASSERT( ! is_whitespace ('a') );
  TEST_ASSERT( ! is_whitespace ('A') );
  TEST_ASSERT( ! is_whitespace ('-') );
  TEST_ASSERT( ! is_whitespace ('.') );

  const char mockFile_iid[] =
    "42 43 -100.0\n"       // valid
    "-50 0 2000.0\n"       // valid iff index type is signed
    "# comment line 1\n"   // empty
    "54 55 3.0e+4\n"       // valid
    "\n"                   // empty
    "%% comment line 2\n"  // empty
    "\t  \t\t\n"           // empty
    "ABCDE FGHIJ 4.0e+5\n" // error (invalid indices)
    "66 67 FGHIJ4.0e+5\n"  // not actually error; strtod calls this 0
    "78 79 -Inf\n"         // valid (Inf is still a valid value)
    "5 4 418.666\n"        // valid
    "8 42.0\n"             // error (missing index)
    "9 10\n"               // error (missing value)
    "4 5 666.418\n";       // valid

  int rowInd, colInd;
  double val;
  const std::string mockFile_s (mockFile_iid);
  std::istringstream is (mockFile_s);
  const size_t lineMaxLen = 100;
  std::string line (lineMaxLen, ' ');

  const double neg_Inf_val = std::strtod ("-Inf", nullptr);
  
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, int (42), int (43), double (-100.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, int (-50), int (0), double (2000.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, int (0), int (0), double (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, int (54), int (55), double (3.0e+4) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, int (0), int (0), double (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, int (0), int (0), double (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, int (0), int (0), double (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, int (0), int (0), double (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, int (66), int (67), double (0.0) ); // surprising, but true
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, int (78), int (79), neg_Inf_val );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, int (5), int (4), double (418.666) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, int (0), int (0), double (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, int (0), int (0), double (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, int (4), int (5), double (666.418) );
}

TEUCHOS_UNIT_TEST( CooMatrix, ReadEntry_llc )
{
  using Tpetra::Details::EReadEntryResult;
  using Tpetra::Details::READ_ENTRY_VALID;
  using Tpetra::Details::READ_ENTRY_ERROR;
  using Tpetra::Details::READ_ENTRY_EMPTY;
  using Tpetra::Details::readCooMatrixEntry;
  using Tpetra::Details::Impl::is_whitespace;
  using index_type = long;
  using val_type = std::complex<float>;

  const char mockFile_llc[] =
    "42 43 -100.0 -100.1\n"           // valid
    "-50 0 2000.0 2000.1\n"           // valid iff index type is signed
    "# comment line 1\n"              // empty
    "54 55 3.0e+4 3.1e+4\n"           // valid
    "\n"                              // empty
    "%% comment line 2\n"             // empty
    "\t  \t\t\n"                      // empty
    "ABCDE FGHIJ 4.0e+5 4.1e+5\n"     // error (invalid indices)
    "66 67 FGHIJ4.0e+5 FGHIJ4.1e+5\n" // error
    "78 79 -Inf -Inf\n"               // valid (Inf a valid value)
    "5 4 418.666 418.667\n"           // valid
    "8 42.0 42.1\n"                   // error (missing index)
    "9 10\n"                          // error (missing value)
    "4 5 666.418 666.419\n";          // valid

  index_type rowInd, colInd;
  val_type val;
  const std::string mockFile_s (mockFile_llc);
  std::istringstream is (mockFile_s);
  const size_t lineMaxLen = 100;
  std::string line (lineMaxLen, ' ');

  const float neg_Inf = std::strtof ("-Inf", nullptr);
  
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (42), index_type (43), val_type (-100.0, -100.1) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (-50), index_type (0), val_type (2000.0, 2000.1) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (54), index_type (55), val_type (3.0e+4, 3.1e+4) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, index_type (66), index_type (67), val_type (0.0, 0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (78), index_type (79), val_type (neg_Inf, neg_Inf) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (5), index_type (4), val_type (418.666, 418.667) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (4), index_type (5), val_type (666.418, 666.419) );
}

TEUCHOS_UNIT_TEST( CooMatrix, ReadEntry_uuf )
{
  using Tpetra::Details::EReadEntryResult;
  using Tpetra::Details::READ_ENTRY_VALID;
  using Tpetra::Details::READ_ENTRY_ERROR;
  using Tpetra::Details::READ_ENTRY_EMPTY;
  using Tpetra::Details::readCooMatrixEntry;
  using Tpetra::Details::Impl::is_whitespace;
  using index_type = unsigned long long;
  using val_type = float;
  
  const char mockFile_uuf[] =
    "42 43 -100.0\n"       // valid
    "-50 0 2000.0\n"       // invalid (index type unsigned)
    "# comment line 1\n"   // empty
    "54 55 3.0e+4\n"       // valid
    "\n"                   // empty
    "%% comment line 2\n"  // empty
    "\t  \t\t\n"           // empty
    "ABCDE FGHIJ 4.0e+5\n" // error (invalid indices)
    "66 67 FGHIJ4.0e+5\n"  // not actually error; strtod calls this 0
    "78 79 -Inf\n"         // valid (Inf is still a valid value)
    "5 4 418.666\n"        // valid
    "8 42.0\n"             // error (missing index)
    "9 10\n"               // error (missing value)
    "4 5 666.418\n";       // valid

  index_type rowInd, colInd;
  val_type val;
  const std::string mockFile_s (mockFile_uuf);
  std::istringstream is (mockFile_s);
  const size_t lineMaxLen = 100;
  std::string line (lineMaxLen, ' ');

  const double neg_Inf_val = std::strtof ("-Inf", nullptr);
  
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (42), index_type (43), val_type (-100.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, index_type (0), index_type (0), val_type (2000.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (54), index_type (55), val_type (3.0e+4) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_EMPTY, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (66), index_type (67), val_type (0.0) ); // surprising, but true
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (78), index_type (79), neg_Inf_val );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (5), index_type (4), val_type (418.666) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_ERROR, index_type (0), index_type (0), val_type (0.0) );
  TPETRA_DETAILS_TEST_READENTRY( READ_ENTRY_VALID, index_type (4), index_type (5), val_type (666.418) );
}

} // namespace (anonymous)
