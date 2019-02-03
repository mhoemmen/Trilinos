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

#ifndef TPETRA_DETAILS_READTRIPLES_HPP
#define TPETRA_DETAILS_READTRIPLES_HPP

/// \file Tpetra_Details_ReadTriples.hpp
/// \brief Declaration and definition of
///   Tpetra::Details::readAndDealOutTriples, which reads a Matrix
///   Market file or input stream on one process, and distributes the
///   resulting sparse matrix entries to the other processes.
///
/// \warning This is an implementation detail of Tpetra.
///   Users must not rely on this file or its contents.

#include "TpetraCore_config.h"
#include "Tpetra_Details_PackTriples.hpp"
#include "Tpetra_Details_readCooMatrixEntry.hpp"
#include "Kokkos_ArithTraits.hpp"
#include "Teuchos_MatrixMarket_generic.hpp"
#include "Teuchos_CommHelpers.hpp"
#include <iostream>
#include <memory>
#include <typeinfo>

namespace Tpetra {
namespace Details {

//
// Search for "SKIP DOWN TO HERE" (omit quotes) for the "public"
// interface.  I put "public" in quotes because it's public only for
// Tpetra developers, NOT for Tpetra users.
//

namespace Impl {

// mfh 01 Feb 2017: Unfortunately,
// Teuchos::MatrixMarket::readComplexData requires Teuchos to have
// complex arithmetic support enabled.  To avoid this issue, I
// reimplement the function here.  It's not very long.

/// \brief Read "<rowIndex> <colIndex> <realPart> <imagPart>" from a line.
///
/// Matrix Market files that store a sparse matrix with complex values
/// do so with one sparse matrix entry per line.  It is stored as
/// space-delimited ASCII text: the row index, the column index, the
/// real part, and the imaginary part, in that order.  Both the row
/// and column indices are 1-based.  This function attempts to read
/// one line from the given input stream istr, extract the row and
/// column indices and the real and imaginary parts, and write them to
/// the corresponding output variables.
///
/// \param istr [in/out] Input stream from which to attempt to read
///   one line.
/// \param rowIndex [out] On output: if successful, the row index read
///   from the line.
/// \param colIndex [out] On output: if successful, the column index
///   read from the line.
/// \param realPart [out] On output: if successful, the real part of
///   the matrix entry's value read from the line.
/// \param imagPart [out] On output: if successful, the imaginary part
///   of the matrix entry's value read from the line.
/// \param lineNumber [in] The current line number.  Used only for
///   diagnostic error messages.
///
/// \return True if this function successfully read the line from istr
///   and extracted all the output data, false otherwise.
template<class OrdinalType, class RealType>
bool
readComplexData (std::istream& istr,
                 OrdinalType& rowIndex,
                 OrdinalType& colIndex,
                 RealType& realPart,
                 RealType& imagPart,
                 const std::size_t lineNumber)
{
  using ::Teuchos::MatrixMarket::readRealData;

  RealType the_realPart, the_imagPart;
  if (! readRealData (istr, rowIndex, colIndex, the_realPart, lineNumber)) {
    return false;
  }
  if (istr.eof ()) {
    return false;
  }
  istr >> the_imagPart;
  if (istr.fail ()) {
    return false;
  }
  realPart = the_realPart;
  imagPart = the_imagPart;
  return true;
}


/// \brief Implementation of the readLine stand-alone function in this
///   namespace (see below).
///
/// Implementations are specialized on whether or not SC is a
/// complex-valued type.
///
/// \tparam SC The type of the value of each matrix entry.
/// \tparam GO The type of each (global) index of each matrix entry.
/// \tparam isComplex Whether SC is a complex-valued type.
template<class SC,
         class GO,
         const bool isComplex = ::Kokkos::Details::ArithTraits<SC>::is_complex>
struct ReadLine {
  /// \brief Take a line from the Matrix Market file or input stream,
  ///   and process the sparse matrix entry in that line.
  ///
  /// \param processTriple [in] Closure, generally with side effects,
  ///   that takes in and stores off a sparse matrix entry.  First
  ///   argument is the (global) row index, second argument is the
  ///   (global) column index, and third argument is the value of the
  ///   entry.  The closure must NOT do MPI communication.  Return
  ///   value is an error code, that is zero if and only if the
  ///   closure succeeded.
  /// \param line [in] Current line of the Matrix Market file or input
  ///   stream to read.
  /// \param lineNumber [in] Current line number in the file or input
  ///   stream.
  /// \param errStrm [in] If not NULL, print any error messages to
  ///   this stream.
  /// \param debug [in] If true, print debug messages to \c *errStrm.
  ///
  /// \return Error code; 0 if and only if success.
  static int
  readLine (std::function<int (const GO, const GO, const SC&)> processTriple,
            const std::string& line,
            const std::size_t lineNumber,
            std::ostream* errStrm = NULL,
            const bool debug = false);
};

/// \brief Complex-arithmetic partial specialization of ReadLine.
///
/// This helps implement the readLine stand-alone function in this
/// namespace (see below).
///
/// \tparam SC The type of the value of each matrix entry.
/// \tparam GO The type of each (global) index of each matrix entry.
template<class SC, class GO>
struct ReadLine<SC, GO, true> {
  /// \brief Take a line from the Matrix Market file or input stream,
  ///   and process the sparse matrix entry in that line.
  ///
  /// \param processTriple [in] Closure, generally with side effects,
  ///   that takes in and stores off a sparse matrix entry.  First
  ///   argument is the (global) row index, second argument is the
  ///   (global) column index, and third argument is the value of the
  ///   entry.  The closure must NOT do MPI communication.  Return
  ///   value is an error code, that is zero if and only if the
  ///   closure succeeded.
  /// \param line [in] Current line of the Matrix Market file or input
  ///   stream to read.
  /// \param lineNumber [in] Current line number in the file or input
  ///   stream.
  /// \param errStrm [in] If not NULL, print any error messages to
  ///   this stream.
  /// \param debug [in] If true, print debug messages to \c *errStrm.
  ///
  /// \return Error code; 0 if and only if success.
  static int
  readLine (std::function<int (const GO, const GO, const SC&)> processTriple,
            const std::string& line,
            const std::size_t lineNumber,
            std::ostream* errStrm = nullptr)
  {
    using ::Teuchos::MatrixMarket::checkCommentLine;
    typedef typename ::Kokkos::Details::ArithTraits<SC>::mag_type real_type;
    using std::endl;
    const char rawPrefix[] = "readLine (complex): ";

    GO rowInd, colInd;
    real_type realPart, imagPart;
    std::istringstream istr (line);
    bool success = true;
    bool threw = false;
    try {
      // Use the version of this function in this file, not the
      // version in Teuchos_MatrixMarket_generic.hpp, because the
      // latter only exists if HAVE_TEUCHOS_COMPLEX is defined.
      success = readComplexData (istr, rowInd, colInd, realPart, imagPart,
                                 lineNumber);
    }
    catch (std::exception& e) {
      threw = true;
      if (errStrm != nullptr) {
        std::ostringstream os;
        os << rawPrefix << "readComplexData threw: " << e.what () << endl;
        *errStrm << os.str ();
      }
    }

    if (errStrm != nullptr) {
      std::ostringstream os;
      if (success) {
	os << rawPrefix << "readComplex got entry: "
	   << "row: " << rowInd
	   << ", col: " << colInd
	   << ", realPart: " << realPart
	   << ", imagPart: " << imagPart
	   << endl;
      }
      else if (! threw) {
	os << rawPrefix << "readComplex returned but failed" << endl;
      }
      *errStrm << os.str ();
    }
    
    if (success) {
      // The user's closure may have side effects.
      const int errCode =
        processTriple (rowInd, colInd, SC (realPart, imagPart));
      if (errStrm != nullptr) {
        std::ostringstream os;	
	os << rawPrefix << "processTriple returned " << errCode << endl;
        *errStrm << os.str ();
      }
      return errCode;
    }
    else {
      return -1;
    }
  }
};

/// \brief Real-arithmetic partial specialization of ReadLine.
///
/// This helps implement the readLine stand-alone function in this
/// namespace (see below).
///
/// \tparam SC The type of the value of each matrix entry.
/// \tparam GO The type of each (global) index of each matrix entry.
template<class SC, class GO>
struct ReadLine<SC, GO, false> {
  /// \brief Take a line from the Matrix Market file or input stream,
  ///   and process the sparse matrix entry in that line.
  ///
  /// \param processTriple [in] Closure, generally with side effects,
  ///   that takes in and stores off a sparse matrix entry.  First
  ///   argument is the (global) row index, second argument is the
  ///   (global) column index, and third argument is the value of the
  ///   entry.  The closure must NOT do MPI communication.  Return
  ///   value is an error code, that is zero if and only if the
  ///   closure succeeded.
  /// \param line [in] Current line of the Matrix Market file or input
  ///   stream to read.
  /// \param lineNumber [in] Current line number in the file or input
  ///   stream.
  /// \param errStrm [in] If not NULL, print any error messages to
  ///   this stream.
  /// \param debug [in] If true, print debug messages to \c *errStrm.
  ///
  /// \return Error code; 0 if and only if success.
  static int
  readLine (std::function<int (const GO, const GO, const SC&)> processTriple,
            const std::string& line,
            const std::size_t lineNumber,
            std::ostream* errStrm = NULL,
            const bool debug = false)
  {
    using ::Teuchos::MatrixMarket::checkCommentLine;
    using ::Teuchos::MatrixMarket::readRealData;
    using std::endl;

    GO rowInd, colInd;
    SC val;
    std::istringstream istr (line);
    bool success = true;
    try {
      success = readRealData (istr, rowInd, colInd, val, lineNumber);
    }
    catch (std::exception& e) {
      success = false;
      if (errStrm != NULL) {
        std::ostringstream os;
        os << "readLine: readRealData threw an exception: " << e.what ()
           << endl;
        *errStrm << os.str ();
      }
    }

    if (success) {
      if (debug && errStrm != NULL) {
        std::ostringstream os;
        os << "readLine: Got entry: row=" << rowInd << ", col=" << colInd
           << ", val=" << val << std::endl;
        *errStrm << os.str ();
      }
      // This line may have side effects.
      const int errCode = processTriple (rowInd, colInd, val);
      if (errCode != 0 && errStrm != NULL) {
        std::ostringstream os;
        os << "readLine: processTriple returned " << errCode << " != 0."
           << endl;
        *errStrm << os.str ();
      }
      return errCode;
    }
    else {
      return -1;
    }
  }
};

/// \brief Take a line from the Matrix Market file or input stream,
///   and process the sparse matrix entry in that line.
///
/// The line must be a valid Matrix Market line, not a comment.
///
/// \tparam SC The type of the value of each matrix entry.
/// \tparam GO The type of each (global) index of each matrix entry.
///
/// \param processTriple [in] Closure, generally with side effects,
///   that takes in and stores off a sparse matrix entry.  First
///   argument is the (global) row index, second argument is the
///   (global) column index, and third argument is the value of the
///   entry.  The closure must NOT do MPI communication.  Return value
///   is an error code, that is zero if and only if the closure
///   succeeded.
/// \param line [in] The line from the Matrix Market file or input
///   stream to read.
/// \param lineNumber [in] Current line number in the file or input
///   stream.
/// \param errStrm [in] If not NULL, print any error messages to this
///   stream.
/// \param debug [in] If true, print debug messages to \c *errStrm.
///
/// \return Error code; 0 if and only if success.
template<class SC, class GO>
int
readLine (std::function<int (const GO, const GO, const SC&)> processTriple,
          const std::string& line,
          const std::size_t lineNumber,
          std::ostream* errStrm = NULL,
          const bool debug = false)
{
  return ReadLine<SC, GO>::readLine (processTriple, line, lineNumber,
                                     errStrm, debug);
}

/// \brief Read sparse matrix entries, one per line, from the given stream.
///
/// With respect to MPI, this is a <i>local</i> operation.  That is:
///
/// <ul>
/// <li>You may call this on <i>any</i> MPI process, as long as that
///   process can read from the given input stream.</li>
///
/// <li>The calling process will read entries from the stream and
///   add them to its local data structure, as if calling
///   sumIntoGlobalValue.</li>
///
/// <li>It's a really bad idea to use the same input stream on all
///   processes.  That will give you incorrect results, since each
///   process will read the same entries from the same file and sum
///   them in redundantly. </li>
/// </ul>
///
/// \tparam SC The type of the value of each matrix entry.
/// \tparam GO The type of each (global) index of each matrix entry.
///
/// \param inputStream [in/out] Input stream from which to read.
/// \param processTriple [in] Closure, generally with side effects,
///   that takes in and stores off a sparse matrix entry.  First
///   argument is the (global) row index, second argument is the
///   (global) column index, and third argument is the value of the
///   entry.  The closure must NOT do MPI communication.  Return value
///   is an error code, that is zero if and only if the closure
///   succeeded.
/// \param curLineNum [in/out] Current line number in the input stream.
/// \param numValid [in/out] Running total number of valid entries
///   (triples) read.  This counts entries with the same row and
///   column index as separate.
/// \param numInvalid [in/out] Running total number of invalid,
///   erroneous entries read.
/// \param numErr [in/out] Running total number of errors reported by
///   the closure \c processTriple.
/// \param maxNumTriplesToRead [in] Maximum number of triples to read
///   from the input stream on this call of the function.  This is a
///   strict upper bound for numTriplesRead (see above).
template<class SC, class GO>
void
readTriples (std::istream& inputStream,
             std::function<int (const GO, const GO, const SC&)> processTriple,
             std::size_t& curLineNum,
             std::size_t& numValid,
	     std::size_t& numInvalid,
	     std::size_t& numErr,
             const std::size_t maxNumEntriesToRead)
{
  GO row;
  GO col;
  SC val;
  std::string line;

  if (inputStream.eof () || inputStream.fail ()) {
    return;
  }
  while (numValid < maxNumEntriesToRead && std::getline (inputStream, line)) {
    ++curLineNum; // we did actually get a line
    const EReadEntryResult result = readCooMatrixEntry (row, col, val, line);
    if (result == READ_ENTRY_VALID) {
      ++numValid;
      const int errCode = processTriple (row, col, val);
      numErr += (errCode == 0 ? 0 : 1);
    }
    else if (result == READ_ENTRY_ERROR) {
      ++numInvalid;
    }
  }
}

/// \brief Read at most maxNumEntPerMsg sparse matrix entries from the
///   input stream, and send them to the process with rank destRank.
///
/// To be called only by the sending process.
///
/// \tparam SC The type of the value of each matrix entry.
/// \tparam GO The type of each (global) index of each matrix entry.
///
/// \param inputStream [in/out] Input stream from which to read.
/// \param curLineNum [in/out] Current line number in the input stream.
/// \param numValid [in/out] Running total number of valid entries
///   (triples) read.  This counts entries with the same row and
///   column index as separate.
/// \param numInvalid [in/out] Running total number of invalid,
///   erroneous entries read.
/// \param sizeBuf [in/out] Array of length 1, for sending message size.
/// \param msgBuf [in/out] Message buffer; to be resized as needed.
/// \param rowInds [out] Row indices read from the file.
/// \param colInds [out] Column indices read from the file.
/// \param vals [out] Matrix values read from the file.
/// \param maxNumEntPerMsg [in] Maximum number of sparse matrix
///   entries to read from the input stream on this call to the
///   function.
/// \param destRank [in] Rank of the process where to send the triples.
/// \param comm [in] Communicator to use for sending the triples.
/// \param errStr [in] If not nullptr, print debug output to this
///   stream.
template<class SC, class GO>
int
readAndSendOneBatchOfTriples (std::istream& inputStream,
                              std::size_t& curLineNum,
                              std::size_t& numValid,
			      std::size_t& numInvalid,
                              ::Teuchos::ArrayRCP<int>& sizeBuf,
                              ::Teuchos::ArrayRCP<char>& msgBuf,
                              std::vector<GO>& rowInds,
                              std::vector<GO>& colInds,
                              std::vector<SC>& vals,
                              const std::size_t maxNumEntPerMsg,
                              const int destRank,
                              const ::Teuchos::Comm<int>& comm,
                              std::ostream* errStrm = nullptr)
{
  using ::Tpetra::Details::countPackTriplesCount;
  using ::Tpetra::Details::countPackTriples;
  using ::Tpetra::Details::packTriplesCount;
  using ::Tpetra::Details::packTriples;
  using ::Kokkos::ArithTraits;  
  using ::Teuchos::isend;
  using std::endl;
  const char rawPrefix[] = "readAndSendOneBatchOfTriples: ";

  std::unique_ptr<std::string> prefix;
  if (errStrm != nullptr) {
    std::ostringstream os;
    os << "Proc " << comm.getRank () << ": " << rawPrefix;
    prefix = std::unique_ptr<std::string> (new std::string (os.str ()));
    os << "Start" << endl;
    *errStrm << os.str () << endl;
  }

  constexpr int sizeTag = 42;
  constexpr int msgTag = 43;
  int errCode = 0;

  // This doesn't actually deallocate memory; it just changes the size
  // back to zero, so that push_back starts over from the beginning.
  rowInds.resize (0);
  colInds.resize (0);
  vals.resize (0);
  // Closure that adds the new matrix entry to the above temp arrays.
  auto processTriple = [&rowInds, &colInds, &vals]
    (const GO rowInd, const GO colInd, const SC& val) {
      try {
        rowInds.push_back (rowInd);
        colInds.push_back (colInd);
        vals.push_back (val);
      }
      catch (...) {
        return -1;
      }
      return 0;
    };
  std::size_t numProcessTripleErrs = 0;
  readTriples<SC, GO> (inputStream, processTriple, curLineNum,
		       numValid, numInvalid, numProcessTripleErrs,
		       maxNumEntPerMsg);
  if (errStrm != nullptr) {
    std::ostringstream os;
    os << *prefix << "readTriples reports: numValid: " << numValid
       << ", numInvalid: "  << numInvalid
       << ", numProcessTripleErrs: " << numProcessTripleErrs << endl;
    *errStrm << os.str ();
  }
  if (numInvalid != 0) {  
    errCode = errCode - 1;
  }
  if (numProcessTripleErrs != 0) {
    errCode = errCode - 2;
  }

  // We don't consider reading having "failed" if we've reached
  // end-of-file before reading maxNumEntPerMsg entries.  It's OK if
  // we got fewer triples than that.  Furthermore, we have to send at
  // least one message to the destination process, even if the read
  // from the file failed.  Always send any triples if we have them,
  // even if there was an error reading.  However, errors in packing
  // make it impossible or unsafe to send anything.

  if (numValid == 0) {
    // Tell the receiving process that we have no triples to send.
    sizeBuf[0] = 0;
    send (sizeBuf.getRawPtr (), 1, destRank, sizeTag, comm);
    return errCode;
  }

  // We read a nonzero # of triples.
  const int numEnt = static_cast<int> (numValid);
  int countSize = 0; // output argument
  int triplesSize = 0; // output argument

  int countErrCode = countPackTriplesCount (comm, countSize, errStrm);
  // countSize should never be nonpositive.
  if (countErrCode != 0 || countSize <= 0) {
    errCode = errCode - 4;
    // Send zero to the receiving process, to tell it about the error.
    sizeBuf[0] = 0;
    send (sizeBuf.getRawPtr (), 1, destRank, sizeTag, comm);
    return errCode;
  }

  // countPackTriplesCount succeeded
  countErrCode = countPackTriples<SC, GO> (numEnt, comm, triplesSize, errStrm);
  if (countErrCode != 0) {
    errCode = errCode - 8;
    // Send zero to the receiving process, to tell it about the error.
    sizeBuf[0] = 0;
    send (sizeBuf.getRawPtr (), 1, destRank, sizeTag, comm);
    return errCode;
  }

  // countPackTriples succeeded; message packed & ready to send.
  
  // Send the message size (in bytes).  We can use a nonblocking
  // send here, and try to overlap with message packing.
  const int outBufSize = countSize + triplesSize;
  sizeBuf[0] = outBufSize;
  auto sizeReq = isend<int, int> (sizeBuf, destRank, sizeTag, comm);

  msgBuf.resize (outBufSize);
  char* outBuf = msgBuf.getRawPtr ();

  // If anything goes wrong with packing, send the pack buffer
  // anyway, since the receiving process expects a message.
  int outBufCurPos = 0; // input/output argument
  countErrCode = packTriplesCount (numEnt, outBuf, outBufSize,
				   outBufCurPos, comm, errStrm);
  if (countErrCode != 0) {
    errCode = errCode - 16;
  }
  
  if (errCode == 0) {
    countErrCode = packTriples<SC, GO> (rowInds.data (), colInds.data (),
					vals.data (), numEnt, outBuf,
					outBufSize, outBufCurPos, comm,
					errStrm);
    if (countErrCode != 0) {
      errCode = errCode - 32;
    }
  }
  auto msgReq = isend<int, char> (msgBuf, destRank, msgTag, comm);

  // Wait on the two messages.  It doesn't matter in what order we
  // send them, because they have different tags.  The receiving
  // process will wait on the first message first, in order to get the
  // size of the second message.
  sizeReq->wait ();
  msgReq->wait ();

  // This doesn't actually deallocate; it just resets sizes to zero.
  rowInds.clear ();
  colInds.clear ();
  vals.clear ();

  if (errStrm != nullptr) {
    std::ostringstream os;
    os << *prefix << "Done!" << endl;
    *errStrm << os.str ();
  }
  return errCode;
}

/// \brief Read at most maxNumEntPerMsg sparse matrix entries from the
///   input stream, and send them to the process with rank destRank.
///
/// To be called only by the sending process.
///
/// \tparam SC The type of the value of each matrix entry.
/// \tparam GO The type of each (global) index of each matrix entry.
/// \tparam CommRequestPtr The type of a (smart) pointer to a
///   "communication request" returned by, e.g., ::Teuchos::ireceive.
///   It must implement operator= and operator->, and the thing to
///   which it points must implement <tt>void wait()</tt>.  A model
///   for this is ::Teuchos::RCP< ::Teuchos::CommRequest<int> >.
///
/// \param rowInds [out] Row indices to receive.  Will be resized as
///   needed.
/// \param colInds [out] Column indices to receive.  Will be resized
///   as needed.
/// \param vals [out] Matrix values to receive.  Will be resized as
///   needed.
/// \param numEnt [out] Number of matrix entries (triples) received.
/// \param sizeBuf [in/out] Array of length 1, for receiving message
///   size (size in bytes of \c msgBuf).
/// \param msgBuf [in/out] Message buffer; to be resized as needed.
/// \param sizeReq [in/out] Preposed receive request for message size.
///   After waiting on this, you may read the contents of sizeBuf.
///   This is a nonconst (smart) pointer reference, so that we can
///   assign to it.  A model for this is
///   ::Teuchos::RCP< ::Teuchos::CommRequest<int> >&.
/// \param srcRank [in] Rank of the process from which to receive the
///   matrix entries (triples).
/// \param comm [in] Communicator to use for receiving the triples.
/// \param errStrm [in] If not NULL, print any error messages to this
///   stream.
template<class SC, class GO, class CommRequestPtr>
int
recvOneBatchOfTriples (std::vector<GO>& rowInds,
                       std::vector<GO>& colInds,
                       std::vector<SC>& vals,
                       int& numEnt,
                       ::Teuchos::ArrayRCP<int>& sizeBuf,
                       ::Teuchos::ArrayRCP<char>& msgBuf,
                       CommRequestPtr& sizeReq,
                       const int srcRank,
                       const ::Teuchos::Comm<int>& comm,
                       std::ostream* errStrm = nullptr)
{
  using ::Tpetra::Details::unpackTriplesCount;
  using ::Tpetra::Details::unpackTriples;
  using ::Kokkos::ArithTraits;
  using std::endl;
  const char rawPrefix[] = "recvOneBatchOfTriples: ";

  std::unique_ptr<std::string> prefix;
  if (errStrm != nullptr) {
    std::ostringstream os;
    os << "Proc " << comm.getRank () << ": " << rawPrefix;
    prefix = std::unique_ptr<std::string> (new std::string (os.str ()));
    os << "Start" << endl;
    *errStrm << os.str () << endl;
  }

  constexpr int msgTag = 43;
  int errCode = 0; // return value
  numEnt = 0; // output argument

  // Wait on the ireceive we preposted before calling this function.
  sizeReq->wait ();
  sizeReq = CommRequestPtr (nullptr);
  const int inBufSize = sizeBuf[0];

  if (errStrm != nullptr) {
    std::ostringstream os;
    os << "inBufSize: " << inBufSize << endl;
    *errStrm << os.str ();
  }

  if (inBufSize == 0) {
    numEnt = 0;
    rowInds.resize (0);
    colInds.resize (0);
    vals.resize (0);
  }
  else {
    msgBuf.resize (inBufSize);
    char* inBuf = msgBuf.getRawPtr ();

    if (errStrm != nullptr) {
      std::ostringstream os;
      os << "Post irecv for data" << endl;
      *errStrm << os.str ();
    }
    auto msgReq = ::Teuchos::ireceive (msgBuf, srcRank, msgTag, comm);
    if (errStrm != nullptr) {
      std::ostringstream os;
      os << "Wait on irecv for data" << endl;
      *errStrm << os.str ();
    }
    msgReq->wait ();

    if (errStrm != nullptr) {
      std::ostringstream os;
      os << "Call unpackTriplesCount" << endl;
      *errStrm << os.str ();
    }
    int inBufCurPos = 0; // output argument
    errCode = unpackTriplesCount (inBuf, inBufSize, inBufCurPos,
                                  numEnt, comm, errStrm);
    if (errCode == 0) {
      rowInds.resize (numEnt);
      colInds.resize (numEnt);
      vals.resize (numEnt);
      errCode = unpackTriples<SC, GO> (inBuf, inBufSize, inBufCurPos,
                                       rowInds.data (), colInds.data (),
                                       vals.data (), numEnt, comm, errStrm);
    }
  }

  if (errStrm != nullptr) {
    std::ostringstream os;
    os << "Done! errCode: " << errCode << endl;
    *errStrm << os.str ();
  }
  return errCode;
}

} // namespace Impl

//
// SKIP DOWN TO HERE FOR "PUBLIC" INTERFACE
//

/// \brief On Process 0 in the given communicator, read sparse matrix
///   entries (in chunks of at most maxNumEntPerMsg entries at a time)
///   from the input stream, and "deal them out" to all other
///   processes in the communicator.
///
/// This is a collective over the communicator.
///
/// \tparam SC The type of the value of each matrix entry.
/// \tparam GO The type of each (global) index of each matrix entry.
///
/// \param inputStream [in/out] Input stream from which to read Matrix
///   Market - format matrix entries ("triples").  Only Process 0 in
///   the communicator needs to be able to access this.
/// \param curLineNum [in/out] On both input and output, the
///   current line number in the input stream.  (In the Matrix Market
///   format, sparse matrix entries cannot start until at least line 3
///   of the file.)  This is only valid on Process 0.
/// \param numValid [out] Total number of valid matrix entries
///   (triples) read on Process 0.  This is only valid on Process 0.
/// \param numInvalid [out] Total number of invalid, erroneous entries
///   read.  This is only valid on Process 0.
/// \param processTriple [in] Closure, generally with side effects,
///   that takes in and stores off a sparse matrix entry.  First
///   argument is the (global) row index, second argument is the
///   (global) column index, and third argument is the value of the
///   entry.  The closure must NOT do MPI communication.  Return value
///   is an error code, that is zero if and only if the closure
///   succeeded.  We intend for you to use this to call
///   CooMatrix::insertEntry.
/// \param comm [in] Communicator to use for receiving the triples.
/// \param errStrm [in] If not NULL, print any error messages to this
///   stream.
///
/// \return Error code; 0 if and only if success.
template<class SC, class GO>
int
readAndDealOutTriples (std::istream& inputStream,
                       std::size_t& curLineNum,
                       std::size_t& numValid,
		       std::size_t& numInvalid,
                       std::function<int (const GO, const GO, const SC&)> processTriple,
                       const std::size_t maxNumEntPerMsg,
                       const ::Teuchos::Comm<int>& comm,
                       std::ostream* errStrm = NULL,
                       const bool debug = false)
{
  using Impl::readAndSendOneBatchOfTriples;
  using Impl::readTriples;
  using Kokkos::ArithTraits;
  using std::endl;
  using std::size_t;
  const char rawPrefix[] = "readAndDealOutTriples: ";

  constexpr int srcRank = 0;
  constexpr int sizeTag = 42;
  const int myRank = comm.getRank ();
  const int numProcs = comm.getSize ();
  int errCode = 0;

  std::unique_ptr<std::string> prefix;
  if (debug && errStrm != nullptr) {
    std::ostringstream os;
    os << "Proc " << myRank << ": " << rawPrefix;
    prefix = std::unique_ptr<std::string> (new std::string (os.str ()));
    os << "Start" << endl;
    *errStrm << os.str () << endl;
  }

  ::Teuchos::ArrayRCP<int> sizeBuf (1);
  ::Teuchos::ArrayRCP<char> msgBuf; // to be resized as needed

  // Temporary storage for reading & packing (on Process srcRank) or
  // unpacking (every other process) triples.
  std::vector<GO> rowInds;
  std::vector<GO> colInds;
  std::vector<SC> vals;
  rowInds.reserve (maxNumEntPerMsg);
  colInds.reserve (maxNumEntPerMsg);
  vals.reserve (maxNumEntPerMsg);

  numValid = 0;
  numInvalid = 0;
  if (myRank == srcRank) {
    // Loop around through all the processes, including this one, over
    // and over until we reach the end of the file, or an error occurs.
    int destRank = 0;
    bool lastMessageWasLegitZero = false;
    for ( ;
          ! inputStream.eof () && errCode == 0;
         destRank = (destRank + 1) % numProcs) {

      if (destRank == srcRank) {
	if (debug && errStrm != nullptr) {
	  std::ostringstream os;
	  os << *prefix << "Call readTriples" << endl;
	  *errStrm << os.str () << endl;
	}
        // We can read and process the triples directly.  We don't
        // need to use intermediate storage, because we don't need to
        // pack and send the triples.
	std::size_t numErr = 0;
	readTriples (inputStream, processTriple, curLineNum, numValid,
		     numInvalid, numErr, maxNumEntPerMsg);
	if (numErr != 0) {
	  errCode = errCode - 64;
	}
      }
      else {
	if (debug && errStrm != nullptr) {
	  std::ostringstream os;
	  os << *prefix << "Call readAndSendOneBatchOfTriples" << endl;
	  *errStrm << os.str () << endl;
	}
        // Read, pack, and send the triples to destRank.
        const int readAndSendErrCode =
          readAndSendOneBatchOfTriples (inputStream, curLineNum,
					numValid, numInvalid,
					sizeBuf, msgBuf,
					rowInds, colInds, vals,
					maxNumEntPerMsg, destRank,
					comm, errStrm);
	if (readAndSendErrCode != 0) {
	  errCode = errCode - 128;
	}
        if (readAndSendErrCode == 0 && numValid == 0) {
          lastMessageWasLegitZero = true;
        }
      }
    } // loop around through processes until done reading file, or error

    // Loop around through the remaining processes, and tell them that
    // we're done, by sending zero.  If the last message we sent to
    // destRank was zero, then skip that process, since it only
    // expects one message of size zero.  Note that destRank got
    // incremented mod numProcs at end of loop, so we have to
    // decrement it mod numProcs.
    destRank = (destRank - 1) % numProcs;
    if (destRank < 0) { // C mod operator does not promise positivity
      destRank = destRank + numProcs;
    }

    const int startRank = lastMessageWasLegitZero ? (destRank+1) : destRank;
    for (int outRank = startRank; outRank < numProcs; ++outRank) {
      if (outRank != srcRank) {
        sizeBuf[0] = 0;
        ::Teuchos::send (sizeBuf.getRawPtr (), 1, outRank, sizeTag, comm);
      }
    }
  }
  else {
    while (true) {
      if (debug && errStrm != nullptr) {
	std::ostringstream os;
	os << *prefix << "Prepost irecv" << endl;
	*errStrm << os.str () << endl;
      }
      // Prepost a message to receive the size (in bytes) of the
      // incoming packet.
      sizeBuf[0] = 0; // superfluous, but safe
      auto sizeReq = ::Teuchos::ireceive (sizeBuf, srcRank, sizeTag, comm);
      if (debug && errStrm != nullptr) {
	std::ostringstream os;
	os << *prefix << "Call recvOneBatchOfTriples" << endl;
	*errStrm << os.str () << endl;
      }
      int numEnt = 0; // output argument
      const int recvErrCode =
        Impl::recvOneBatchOfTriples (rowInds, colInds, vals, numEnt, sizeBuf,
                                     msgBuf, sizeReq, srcRank, comm, errStrm);
      if (debug && errStrm != nullptr) {
	std::ostringstream os;
	os << *prefix << "recvErrCode: " << recvErrCode
	   << ", numEnt: " << numEnt << endl;
	*errStrm << os.str () << endl;
      }
      errCode = (recvErrCode != 0) ? recvErrCode : errCode;

      if (numEnt != static_cast<int> (rowInds.size ()) ||
          numEnt != static_cast<int> (colInds.size ()) ||
          numEnt != static_cast<int> (vals.size ())) {
        errCode = (errCode == 0) ? -1 : errCode;
        if (errStrm != nullptr) {
          *errStrm << "recvOneBatchOfTriples produced inconsistent data sizes.  "
                   << "numEnt = " << numEnt
                   << ", rowInds.size() = " << rowInds.size ()
                   << ", colInds.size() = " << colInds.size ()
                   << ", vals.size() = " << vals.size () << "."
                   << endl;
        }
      } // if sizes inconsistent

      // Sending zero items is how Process srcRank tells this process
      // that it (Process srcRank) is done sending out data.
      if (numEnt == 0) {
        break;
      }

      if (debug && errStrm != nullptr) {
	std::ostringstream os;
	os << *prefix << "Process triples" << endl;
	*errStrm << os.str () << endl;
      }
      for (int k = 0; k < numEnt && errCode == 0; ++k) {
        const int curErrCode = processTriple (rowInds[k], colInds[k], vals[k]);
        errCode = (curErrCode == 0) ? errCode : curErrCode;
      }
    } // while we still get messages from srcRank
  }

  if (debug && errStrm != nullptr) {
    std::ostringstream os;
    os << *prefix << "All-reduce for error code" << endl;
    *errStrm << os.str () << endl;
  }

  // Do a bitwise OR to get an error code that is nonzero if and only
  // if any process' local error code is nonzero.
  using ::Teuchos::outArg;
  using ::Teuchos::REDUCE_BOR;
  using ::Teuchos::reduceAll;
  const int lclErrCode = errCode;
  reduceAll<int, int> (comm, REDUCE_BOR, lclErrCode, outArg (errCode));

  if (debug && errStrm != nullptr) {
    std::ostringstream os;
    os << *prefix << "Done! errCode: " << errCode << endl;
    *errStrm << os.str () << endl;
  }
  return errCode;
}

} // namespace Details
} // namespace Tpetra

#endif // TPETRA_DETAILS_READTRIPLES_HPP
