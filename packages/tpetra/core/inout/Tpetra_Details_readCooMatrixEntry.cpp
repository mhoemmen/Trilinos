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

#include "Tpetra_Details_CooMatrix.hpp"
#include <algorithm>
#include <cctype>
#include <cerrno>

namespace Tpetra {
namespace Details {
namespace Impl {

bool is_whitespace (const char ch)
{
  return std::isspace (static_cast<unsigned char> (ch));
}

EReadEntryResult
readScalar (double& val, char** next, const char* beg, const char* end)
{
  const char* start_pos = std::find_if_not (beg, end, is_whitespace);
  if (start_pos == end) {
    return READ_ENTRY_EMPTY;
  }

  errno = 0;
  val = std::strtod (start_pos, next);
  return (errno == 0) ? READ_ENTRY_VALID : READ_ENTRY_ERROR;
}

EReadEntryResult
readScalar (float& val, char** next, const char* beg, const char* end)
{
  const char* start_pos = std::find_if_not (beg, end, is_whitespace);
  if (start_pos == end) {
    return READ_ENTRY_EMPTY;
  }

  errno = 0;
  val = std::strtof (start_pos, next);
  return (errno == 0) ? READ_ENTRY_VALID : READ_ENTRY_ERROR;
}

EReadEntryResult
readLongLongIndex (long long& ind, char** next,
		   const char* beg, const char* end,
		   const long long minAllowedIndex,
		   const long long maxAllowedIndex)
{
  errno = 0;
  ind = std::strtoll (beg, next, 10);
  if (errno != 0 ||
      (ind > maxAllowedIndex || ind < minAllowedIndex)) {
    errno = 0;
    return READ_ENTRY_ERROR;
  }
  return READ_ENTRY_VALID;
}

template<class T>
EReadEntryResult
readComplexScalar (std::complex<T>& val, char** next, const char* beg, const char* end)
{
  if (beg == end) {
    return READ_ENTRY_EMPTY;
  }

  T real;
  const char* pos = beg;
  const EReadEntryResult realResult = readScalar (real, next, pos, end);
  if (realResult != READ_ENTRY_VALID) {
    return realResult;
  }

  pos = *next;
  // This character actually needs to _be_ whitespace.
  if (pos == end || ! is_whitespace (*pos)) {
    return READ_ENTRY_ERROR;
  }
  pos = std::find_if_not (pos, end, is_whitespace);
  if (pos == end) {
    return READ_ENTRY_ERROR;
  }

  T imag;
  const EReadEntryResult imagResult = readScalar (imag, next, pos, end);
  if (imagResult != READ_ENTRY_VALID) {
    return imagResult;
  }

  val = std::complex<T> (real, imag);
  return READ_ENTRY_VALID;
}

EReadEntryResult
readScalar (std::complex<float>& val, char** next, const char* beg, const char* end)
{
  return readComplexScalar<float> (val, next, beg, end);
}

EReadEntryResult
readScalar (std::complex<double>& val, char** next, const char* beg, const char* end)
{
  return readComplexScalar<double> (val, next, beg, end);
}

} // namespace Impl
} // namespace Details
} // namespace Tpetra
