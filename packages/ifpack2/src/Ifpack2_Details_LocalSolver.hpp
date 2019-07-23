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

#ifndef IFPACK2_DETAILS_LOCALSOLVER_HPP
#define IFPACK2_DETAILS_LOCALSOLVER_HPP

#include "Ifpack2_config.h"
#include "Kokkos_Core.hpp"

namespace Ifpack2 {
namespace Details {

template<class LocalCrsMatrixType>
class LocalSolver {
  using device_type = typename LocalCrsMatrixType::device_type;
  using value_type = typename LocalCrsMatrixType::value_type;

public:
  virtual ~LocalSolver () = default;
  virtual void setMatrix (const LocalCrsMatrixType& A) = 0;
  virtual void initialize () = 0;
  virtual void compute () = 0;
  virtual void
  apply (Kokkos::View<const value_type**, Kokkos::LayoutLeft,
           device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> in,
         Kokkos::View<value_type**, Kokkos::LayoutLeft,
           device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>> out,
         const Teuchos::ETransp mode,
         const value_type alpha,
         const value_type beta) const = 0;
};

} // namespace Details
} // namespace Ifpack2

#endif // IFPACK2_DETAILS_LOCALSOLVER_HPP
