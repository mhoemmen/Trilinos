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

#ifndef TPETRA_DETAILS_READENTRY_HPP
#define TPETRA_DETAILS_READENTRY_HPP

/// \file Tpetra_Details_readEntry.hpp
/// \brief Declaration and definition of functions for reading a
///   sparse matrix entry from a line of a file.

#include "TpetraCore_config.h"
#include <algorithm>
#include <complex>
#include <limits>
#include <sstream>
#include <string>
#include <type_traits>

namespace Tpetra {
namespace Details {

//! Result of readCooMatrixEntry and related functions.
enum EReadEntryResult {
  READ_ENTRY_VALID, ///< Got a valid entry or other thing
  READ_ENTRY_EMPTY, ///< Got a comment or empty line
  READ_ENTRY_ERROR  ///< Got an error
};

// Implementation details of Tpetra::Details.
// So, users REALLY should not use anything in here.
namespace Impl {

//! Whether \c ch is a whitespace character.
bool is_whitespace (const char ch);

/// \brief Read a scalar value.
///
/// This is an implementation detail of readCooMatrixEntry.  This is
/// also a customization point for Tpetra developers.  Implementers
/// may overload this function for their own Scalar types.
///
/// \param val [out] If successful, the value read.
/// \param next [out] This works like the output parameter of the C
///   Standard Library function \c strtod.
/// \param beg [in] Same as first (input) parameter of \c strtod.
/// \param end [in] One past the last entry of the string.
///
/// Implementations must offer the basic exception guarantee, but need
/// not offer any more than that.  In particular, implementations may
/// change <tt>val</tt> even if they return something other than
/// <tt>READ_ENTRY_VALID</tt>.
template<class ScalarType>
EReadEntryResult
readScalar (ScalarType& val, char** next, const char* beg, const char* end)
{
  if (beg == end) {
    return READ_ENTRY_EMPTY;
  }
  else {
    std::istringstream is (std::string (beg, end));
    is >> val;

    if (! is) {
      return READ_ENTRY_EMPTY;
    }
    else {
      *next = const_cast<char*> (beg + is.tellg ());
      return READ_ENTRY_VALID;
    }
  }
}

//! Overload of readScalar for ScalarType = double.
EReadEntryResult
readScalar (double& val, char** next,
            const char* beg, const char* end);

//! Overload of readScalar for ScalarType = float.
EReadEntryResult
readScalar (float& val, char** next,
            const char* beg, const char* end);

//! Overload of readScalar for ScalarType = std::complex<double>.
EReadEntryResult
readScalar (std::complex<double>& val, char** next,
            const char* beg, const char* end);

//! Overload of readScalar for ScalarType = std::complex<float>.
EReadEntryResult
readScalar (std::complex<float>& val, char** next,
            const char* beg, const char* end);

/// \brief Read index as a long long value.
///
/// \param ind [out] If successful, the value read.
/// \param next [out] This works like the output parameter of the C
///   Standard Library function \c strtol.
/// \param beg [in] Same as first (input) parameter of \c strtol.
/// \param end [in] One past the last entry of the string.
/// \param minAllowedIndex [in] Minimum allowed value of ind.
/// \param maxAllowedIndex [in] Maximum allowed value of ind.
///
/// Implementations must offer the basic exception guarantee, but need
/// not offer any more than that.  In particular, implementations may
/// change <tt>ind</tt> even if they return something other than
/// <tt>READ_ENTRY_VALID</tt>.
///
/// The lower and upper bounds are useful for readIndex below, if
/// IndexType is smaller than <tt>long long</tt>.
EReadEntryResult
readLongLongIndex (long long& ind, char** next,
                   const char* beg, const char* end,
                   const long long minAllowedIndex,
                   const long long maxAllowedIndex);

/// \brief Read an integer index as an IndexType value.
///
/// \param ind [out] If successful, the value read.
/// \param next [out] This works like the output parameter of the C
///   Standard Library function \c strtol.
/// \param beg [in] Same as first (input) parameter of \c strtol.
/// \param end [in] One past the last entry of the string.
///
/// Implementations must offer the basic exception guarantee, but need
/// not offer any more than that.  In particular, implementations may
/// change <tt>ind</tt> even if they return something other than
/// <tt>READ_ENTRY_VALID</tt>.
template<class IndexType>
EReadEntryResult
readIndex (IndexType& ind, char** next, const char* beg, const char* end)
{
  constexpr bool use_strtoll = std::is_integral<IndexType>::value &&
    (sizeof (IndexType) < sizeof (long long) ||
     (sizeof (IndexType) == sizeof (long long) && std::is_signed<IndexType>::value));

  if (use_strtoll) {
    using nli = std::numeric_limits<IndexType>;
    const long long minInd = static_cast<long long> (nli::min ());
    const long long maxInd = static_cast<long long> (nli::max ());
    long long ind_ll;
    const EReadEntryResult result =
      readLongLongIndex (ind_ll, next, beg, end, minInd, maxInd);
    // These functions only promise the basic exception guarantee,
    // so we don't have to test for success before assigning to ind.
    ind = static_cast<IndexType> (ind_ll);
    return result;
  }
  else {
    return readScalar (ind, next, beg, end);
  }
}

} // namespace Impl

/// \brief Read a CooMatrix (sparse matrix) entry from a string.
///
/// \param rowInd [out] Global row index.
/// \param colInd [out] Global column index.
/// \param val [out] Matrix entry value.
/// \param line [in] String from which to read entry.
///
/// Implementations must offer the basic exception guarantee, but need
/// not offer any more than that.  In particular, implementations may
/// change the output arguments, even if they return something other
/// than <tt>READ_ENTRY_VALID</tt>.
template<class IndexType, class ValueType>
EReadEntryResult
readCooMatrixEntry (IndexType& rowInd,
                    IndexType& colInd,
                    ValueType& val,
                    const std::string& line)
{
  using Impl::is_whitespace;
  using Impl::readIndex;
  using Impl::readScalar;
  const char* beg = line.data ();
  const size_t len = line.size ();
  const char* end = beg + len;
  const char* pos = std::find_if_not (beg, end, is_whitespace);

  if (pos == end || // only white space
      *pos == '%' || *pos == '#') { // starts with comment
    return READ_ENTRY_EMPTY;
  }

  // In what follows, there should be at least two whitespace
  // characters, or three if ValueType is complex.  We say "at least"
  // because there may be extra whitespace in between, or at the end.
  const ptrdiff_t num_spaces_left = std::count_if (pos, end, is_whitespace);
  constexpr ptrdiff_t required_num_spaces_left =
    std::is_same<ValueType, std::complex<float>>::value ||
    std::is_same<ValueType, std::complex<double>>::value ||
    std::is_same<ValueType, std::complex<long double>>::value ?
    ptrdiff_t (3) : ptrdiff_t (2);
  if (num_spaces_left < required_num_spaces_left) {
    return READ_ENTRY_ERROR;
  }

  char* next = nullptr;

  const EReadEntryResult result_r = readIndex (rowInd, &next, pos, end);
  if (result_r != READ_ENTRY_VALID) {
    return READ_ENTRY_ERROR;
  }
  pos = next;

  pos = std::find_if_not (pos, end, is_whitespace);
  if (pos == end) {
    return READ_ENTRY_ERROR; // nothing after the first index
  }

  const EReadEntryResult result_c = readIndex (colInd, &next, pos, end);
  if (result_c != READ_ENTRY_VALID) {
    return READ_ENTRY_ERROR;
  }
  pos = next;

  pos = std::find_if_not (pos, end, is_whitespace);
  if (pos == end) {
    return READ_ENTRY_ERROR; // nothing after the second index
  }

  return readScalar (val, &next, pos, end);
}

} // namespace Details
} // namespace Tpetra

#endif // TPETRA_DETAILS_READENTRY_HPP
