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

// This test tests whether the MPI implementation that Trilinos uses
// is CUDA aware.  See Trilinos GitHub issues #1571 and #1088 to learn
// what it means for an MPI implementation to be "CUDA aware," and why
// this matters for performance.
//
// The test will only build if CUDA is enabled.  If you want to
// exercise this test, you must build Tpetra with CUDA enabled, and
// set the environment variable TPETRA_ASSUME_CUDA_AWARE_MPI to some
// true value (e.g., "1" or "TRUE").  If you set the environment
// variable to some false value (e.g., "0" or "FALSE"), the test will
// run but will pass trivially.  This means that you may control the
// test's behavior at run time.

#include "Tpetra_TestingUtilities.hpp"
#include "Tpetra_Details_Behavior.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Kokkos_ArithTraits.hpp"
#include "Kokkos_DualView.hpp"
#include "Kokkos_TeuchosCommAdapters.hpp"

namespace { // (anonymous)
  using std::endl;

#if ! defined(KOKKOS_ENABLE_CUDA) || ! defined(HAVE_TPETRA_INST_CUDA)
#  error "Building this test requires that Trilinos was built with CUDA enabled, and that Tpetra_INST_CUDA:BOOL=ON.  The latter should be true by default if the former is true.  Thus, if Trilinos was built with CUDA enabled, then you must have set some nondefault CMake option."
#endif // ! defined(KOKKOS_ENABLE_CUDA) && ! defined(HAVE_TPETRA_INST_CUDA)

  void
  tpetraGblTest (bool& success,
                 Teuchos::FancyOStream& out,
                 const Teuchos::Comm<int>& comm)
  {
    using Teuchos::outArg;
    using Teuchos::REDUCE_MIN;
    using Teuchos::reduceAll;

    const int lclSuccess = success ? 1 : 0;
    int gblSuccess = 0;
    reduceAll<int, int> (comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
    TEST_EQUALITY_CONST( gblSuccess, 1 );
    if (gblSuccess != 1) {
      out << "FAILED on some process." << endl;
    }
    else {
      out << "OK thus far on all processes." << endl;
    }
  }

  template<class OutputScalarType, class InputIntegerType>
  KOKKOS_INLINE_FUNCTION OutputScalarType
  toScalar (const InputIntegerType x)
  {
    using KAT = Kokkos::ArithTraits<OutputScalarType>;
    using mag_type = typename KAT::mag_type;
    return static_cast<OutputScalarType> (static_cast<mag_type> (x));
  }

  template<class BufferDualViewType>
  void
  packAndPrepare (BufferDualViewType& sendBuf,
                  const typename BufferDualViewType::non_const_value_type&
                    startValue,
                  const bool fillOnHost)
  {
    using index_type = decltype (sendBuf.extent (0));
    using value_type =
      typename BufferDualViewType::t_dev::non_const_value_type;
    // This line of code also helps keep references out of the lambda.
    const value_type startVal = static_cast<value_type> (startValue);

    sendBuf.clear_sync_state ();
    if (fillOnHost) {
      sendBuf.modify_host ();
      using execution_space =
        typename BufferDualViewType::t_host::execution_space;
      using range_type = Kokkos::RangePolicy<execution_space, index_type>;
      Kokkos::parallel_for
        ("Fill sendBuf on host", range_type (0, sendBuf.extent (0)),
         [=] (const index_type k) {
          sendBuf.h_view(k) = startVal + toScalar<value_type> (k);
        });
    }
    else {
      sendBuf.modify_device ();
      using execution_space =
        typename BufferDualViewType::t_dev::execution_space;
      using range_type = Kokkos::RangePolicy<execution_space, index_type>;
      Kokkos::parallel_for
        ("Fill sendBuf on device", range_type (0, sendBuf.extent (0)),
         KOKKOS_LAMBDA (const index_type k) {
          sendBuf.d_view(k) = startVal + toScalar<value_type> (k);
        });
      // MPI will access the data on host, so we need to make sure the
      // device kernel is done before we return.
      execution_space::fence ();
    }
  }

  // Return whether the values are as they should be.
  template<class BufferDualViewType>
  bool
  unpackAndCombine (BufferDualViewType& recvBuf,
                    const typename BufferDualViewType::non_const_value_type&
                      expectedStartValue,
                    const bool checkOnHost)
  {
    using index_type = decltype (recvBuf.extent (0));
    using value_type =
      typename BufferDualViewType::t_dev::non_const_value_type;
    // This line of code also helps keep references out of the lambda.
    const value_type startVal = static_cast<value_type> (expectedStartValue);

    if (checkOnHost) {
      if (recvBuf.need_sync_host ()) {
        recvBuf.sync_host ();
      }
      using execution_space =
        typename BufferDualViewType::t_host::execution_space;
      using range_type = Kokkos::RangePolicy<execution_space, index_type>;
      int allSame = 0;
      Kokkos::parallel_reduce
        ("Check recvBuf on host", range_type (0, recvBuf.extent (0)),
         [=] (const index_type k, int& curResult) {
          const value_type expectedVal = startVal + toScalar<value_type> (k);
          const int equal = (recvBuf.h_view(k) == expectedVal) ? 1 : 0;
          curResult = curResult && equal;
        }, Kokkos::LAnd<int> (allSame));
      return allSame == 1;
    }
    else {
      if (recvBuf.need_sync_device ()) {
        recvBuf.sync_device ();
      }
      using execution_space =
        typename BufferDualViewType::t_dev::execution_space;
      using range_type = Kokkos::RangePolicy<execution_space, index_type>;
      int allSame = 0;
      Kokkos::parallel_reduce
        ("Check recvBuf on device", range_type (0, recvBuf.extent (0)),
         KOKKOS_LAMBDA (const index_type k, int& curResult) {
          const value_type expectedVal = startVal + toScalar<value_type> (k);
          const int equal = (recvBuf.d_view(k) == expectedVal) ? 1 : 0;
          curResult = curResult && equal;
        }, Kokkos::LAnd<int> (allSame));
      return allSame == 1;
    }
  }

  template<class BufferDualViewType>
  void
  exchangeMessages (const BufferDualViewType& recvBuf,
                    const BufferDualViewType& sendBuf,
                    const int recvProc,
                    const int sendProc,
                    const int msgTag,
                    const Teuchos::Comm<int>& comm,
                    const bool useHostBuf)
  {
    using request_type = Teuchos::RCP<Teuchos::CommRequest<int>>;
    Teuchos::Array<request_type> requests (2);

    if (useHostBuf) {
      requests[0] = Teuchos::ireceive (recvBuf.h_view, recvProc, msgTag, comm);
      requests[1] = Teuchos::isend (sendBuf.h_view, sendProc, msgTag, comm);
    }
    else {
      requests[0] = Teuchos::ireceive (recvBuf.d_view, recvProc, msgTag, comm);
      requests[1] = Teuchos::isend (sendBuf.d_view, sendProc, msgTag, comm);
    }
    Teuchos::waitAll (comm, requests ());
  }

  template<class BufferDualViewType>
  void
  fillWithFlagValue (BufferDualViewType& buf,
                     const typename BufferDualViewType::non_const_value_type
                       flagValue)
  {
    buf.clear_sync_state ();
    buf.modify_host ();
    Kokkos::deep_copy (buf.h_view, flagValue);
    buf.sync_device ();
  }

  //
  // UNIT TESTS
  //

  TEUCHOS_UNIT_TEST( Comm, HostPinnedBoundaryExchange )
  {
    using packet_type = int;
    using buffer_execution_space = Kokkos::Cuda;
    using buffer_memory_space = Kokkos::CudaHostPinnedSpace;
    using buffer_device_type = Kokkos::Device<buffer_execution_space, buffer_memory_space>;
    using buffer_dual_view_type = Kokkos::DualView<packet_type*, buffer_device_type>;

    static_assert
      (std::is_same<typename buffer_dual_view_type::t_dev::execution_space,
       buffer_execution_space>::value,
       "Device execution space isn't what we thought");
    static_assert
      (std::is_same<typename buffer_dual_view_type::t_host::execution_space,
       Kokkos::DefaultHostExecutionSpace>::value,
       "Host execution space isn't what we thought");
    static_assert
      (std::is_same<typename buffer_dual_view_type::t_dev::memory_space,
       buffer_memory_space>::value,
       "Device memory space isn't what we thought");
    static_assert
      (std::is_same<typename buffer_dual_view_type::t_host::memory_space,
       buffer_memory_space>::value,
       "Host memory space should be same as device memory space");

    out << "Testing host-pinned buffers for MPI communication" << endl;
    Teuchos::OSTab tab1 (out);

    const bool assumeMpiIsCudaAware =
      Tpetra::Details::Behavior::assumeMpiIsCudaAware ();
    out << "Assuming that MPI is " << (assumeMpiIsCudaAware ? "" : "NOT ")
        << "CUDA aware.  That should not be not relevant to this test, "
      "but it's handy to know." << endl;

    const Teuchos::RCP<const Teuchos::Comm<int>> comm =
      Tpetra::getDefaultComm ();
    const int myRank = comm->getRank ();
    const int numProcs = comm->getSize ();

    TEST_ASSERT( numProcs >= 2 );
    if (numProcs < 2) {
      out << "This test is more meaningful if run with at least 2 MPI "
        "processes.  You ran with only 1 MPI process." << endl;
      return;
    }
    const std::string prefix ([=] { // compare to Lisp's LET (yay const!)
        std::ostringstream os;
        os << "Proc " << myRank << ": ";
        return os.str ();
      } ());

    const int bufLen = 100;
    buffer_dual_view_type recvBuf ("recvBuf", bufLen);
    buffer_dual_view_type sendBuf ("sendBuf", bufLen);

    const packet_type flagValue = -1;
    fillWithFlagValue (recvBuf, flagValue);
    fillWithFlagValue (sendBuf, flagValue);

    tpetraGblTest (success, out, *comm);
    if (! success) {
      return;
    }

    {
      out << "Test self-messages" << endl;
      Teuchos::OSTab tab2 (out);

      const int recvProc = myRank;
      const int sendProc = myRank;
      const int msgTag = 11;

      for (bool workOnHost : {false, true}) {
        const packet_type startValue = workOnHost ? 111 : 666;
        packAndPrepare (sendBuf, startValue, workOnHost);
        exchangeMessages (recvBuf, sendBuf, recvProc, sendProc,
                          msgTag, *comm, workOnHost);
        const bool ok = unpackAndCombine (recvBuf, startValue, workOnHost);
        TEST_ASSERT( ok );

        fillWithFlagValue (recvBuf, flagValue);
        fillWithFlagValue (sendBuf, flagValue);
      }
    }

    tpetraGblTest (success, out, *comm);
    if (! success) {
      return;
    }

    {
      out << "Test messages between processes" << endl;
      Teuchos::OSTab tab2 (out);

      const int recvProc = (myRank + 1) % numProcs;
      const int sendProc = (myRank + 1) % numProcs;
      const int msgTag = 31;

      const packet_type sendStartValue = (myRank == 0) ? 93 : 418;
      const packet_type recvStartValue = (myRank == 0) ? 418 : 93;

      for (bool workOnHost : {false, true}) {
        packAndPrepare (sendBuf, sendStartValue, workOnHost);
        exchangeMessages (recvBuf, sendBuf, recvProc, sendProc,
                          msgTag, *comm, workOnHost);
        const bool ok = unpackAndCombine (recvBuf, recvStartValue, workOnHost);
        TEST_ASSERT( ok );

        fillWithFlagValue (recvBuf, flagValue);
        fillWithFlagValue (sendBuf, flagValue);
      }
    }

    tpetraGblTest (success, out, *comm);
  }

} // namespace (anonymous)


int
main (int argc, char* argv[])
{
  // Initialize MPI (if enabled) before initializing Kokkos.  This
  // lets MPI control things like pinning processes to sockets.
  Teuchos::GlobalMPISession mpiSession (&argc, &argv);
  Kokkos::initialize (argc, argv);
  const int errCode =
    Teuchos::UnitTestRepository::runUnitTestsFromMain (argc, argv);
  Kokkos::finalize ();
  return errCode;
}
