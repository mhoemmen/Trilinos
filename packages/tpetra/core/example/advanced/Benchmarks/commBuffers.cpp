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

#include "Tpetra_Core.hpp"
#include "Tpetra_Details_Behavior.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_FancyOStream.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_TimeMonitor.hpp"
#include "Teuchos_TypeNameTraits.hpp"
#include "Kokkos_ArithTraits.hpp"
#include "Kokkos_DualView.hpp"
#include "Kokkos_TeuchosCommAdapters.hpp"
#include <type_traits>

namespace { // (anonymous)

#if ! defined(KOKKOS_ENABLE_CUDA) || ! defined(HAVE_TPETRA_INST_CUDA)
#  error "Building this test requires that Trilinos was built with CUDA enabled, and that Tpetra_INST_CUDA:BOOL=ON.  The latter should be true by default if the former is true.  Thus, if Trilinos was built with CUDA enabled, then you must have set some nondefault CMake option."
#endif // ! defined(KOKKOS_ENABLE_CUDA) && ! defined(HAVE_TPETRA_INST_CUDA)

  using std::endl;

  // void
  // tpetraGblTest (bool& success,
  //                Teuchos::FancyOStream& out,
  //                const Teuchos::Comm<int>& comm)
  // {
  //   using Teuchos::outArg;
  //   using Teuchos::REDUCE_MIN;
  //   using Teuchos::reduceAll;

  //   const int lclSuccess = success ? 1 : 0;
  //   int gblSuccess = 0;
  //   reduceAll<int, int> (comm, REDUCE_MIN, lclSuccess, outArg (gblSuccess));
  //   TEST_EQUALITY_CONST( gblSuccess, 1 );
  //   if (gblSuccess != 1) {
  //     out << "FAILED on some process." << endl;
  //   }
  //   else {
  //     out << "OK thus far on all processes." << endl;
  //   }
  // }

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
                  const bool fenceDeviceBeforeMpi)
  {
    using index_type = decltype (sendBuf.extent (0));
    using value_type =
      typename BufferDualViewType::t_dev::non_const_value_type;
    // This line of code also helps keep references out of the lambda.
    const value_type startVal = static_cast<value_type> (startValue);

    sendBuf.clear_sync_state ();
    sendBuf.modify_device ();
    using execution_space =
      typename BufferDualViewType::t_dev::execution_space;
    using range_type = Kokkos::RangePolicy<execution_space, index_type>;
    Kokkos::parallel_for
      ("Fill sendBuf on device", range_type (0, sendBuf.extent (0)),
       KOKKOS_LAMBDA (const index_type k) {
        sendBuf.d_view(k) = startVal + toScalar<value_type> (k);
      });

    if (std::is_same<execution_space, Kokkos::Cuda>::value &&
        fenceDeviceBeforeMpi) {
      execution_space::fence ();
    }
  }

  // Return whether the values are as they should be.  We don't have
  // to use the return value; the point is to simulate some kind of
  // unpacking work, and to make it a reduction, so that we know the
  // kernel is done (without an extra fence).
  template<class BufferDualViewType>
  bool
  unpackAndCombine (BufferDualViewType& recvBuf,
                    const typename BufferDualViewType::non_const_value_type&
                      expectedStartValue)
  {
    using index_type = decltype (recvBuf.extent (0));
    using value_type =
      typename BufferDualViewType::t_dev::non_const_value_type;
    // This line of code also helps keep references out of the lambda.
    const value_type startVal = static_cast<value_type> (expectedStartValue);

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
      TEUCHOS_ASSERT( ! recvBuf.need_sync_host () );
      TEUCHOS_ASSERT( ! sendBuf.need_sync_host () );
      requests[0] = Teuchos::ireceive (recvBuf.h_view, recvProc, msgTag, comm);
      requests[1] = Teuchos::isend (sendBuf.h_view, sendProc, msgTag, comm);
    }
    else {
      TEUCHOS_ASSERT( ! recvBuf.need_sync_device () );
      TEUCHOS_ASSERT( ! sendBuf.need_sync_device () );
      requests[0] = Teuchos::ireceive (recvBuf.d_view, recvProc, msgTag, comm);
      requests[1] = Teuchos::isend (sendBuf.d_view, sendProc, msgTag, comm);
    }
    Teuchos::waitAll (comm, requests ());
  }

  struct CmdLineArgs {
    int numTrials = 1000;
    int bufSize = 1000;
    bool fenceDeviceBeforeMpi = true;
    bool success = true;
    bool printedHelp = false;
  };

  CmdLineArgs getCmdLineArgs (const int argc, char** argv)
  {
    using Teuchos::CommandLineProcessor;

    CmdLineArgs args;
    CommandLineProcessor cmdp;
    cmdp.setOption ("num-trials", &args.numTrials,
                    "Number of times to repeat each "
                    "operation in a timing loop");
    cmdp.setOption ("buffer-size", &args.bufSize,
                    "Communication buffer size on each process");
    cmdp.setOption ("sync-before-mpi", "no-sync-before-mpi",
                    &args.fenceDeviceBeforeMpi,
                    "Whether to sync (Kokkos fence) the device "
                    "before MPI accesses communication buffers");

    const CommandLineProcessor::EParseCommandLineReturn parseResult =
      cmdp.parse (argc, argv);
    if (parseResult == CommandLineProcessor::PARSE_HELP_PRINTED) {
      // The user specified --help at the command line to print help
      // with command-line arguments.
      args.success = true;
      args.printedHelp = true;
    }
    else {
      args.printedHelp = false;
      args.success =
        parseResult == CommandLineProcessor::PARSE_SUCCESSFUL;
    }
    return args;
  }

  template<class BufferExecutionSpace, class BufferMemorySpace>
  void
  runBenchmark (Teuchos::FancyOStream& out,
                const CmdLineArgs& args,
                const Teuchos::Comm<int>& comm)
  {
    using Teuchos::TimeMonitor;
    using Teuchos::TypeNameTraits;
    using packet_type = int;
    using buffer_execution_space =
      typename BufferExecutionSpace::execution_space;
    using buffer_memory_space = typename BufferMemorySpace::memory_space;
    using buffer_device_type =
      Kokkos::Device<buffer_execution_space, buffer_memory_space>;
    using buffer_dual_view_type =
      Kokkos::DualView<packet_type*, buffer_device_type>;

    const std::string execSpaceName =
      TypeNameTraits<BufferExecutionSpace>::name ();
    const std::string memSpaceName =
      TypeNameTraits<BufferMemorySpace>::name ();
    out << "Execution space: " << execSpaceName << endl
        << "Memory space: " << memSpaceName << endl;
    Teuchos::OSTab tab1 (out);

    const int myRank = comm.getRank ();
    const int numProcs = comm.getSize ();

    buffer_dual_view_type recvBuf ("recvBuf", args.bufSize);
    buffer_dual_view_type sendBuf ("sendBuf", args.bufSize);

    {
      out << "Test self-messages" << endl;
      Teuchos::OSTab tab2 (out);

      const int recvProc = myRank;
      const int sendProc = myRank;
      const int msgTag = 11;
      const packet_type startValue = 666;
      const bool makeMpiUseHostBuf = false;

      auto wholeLoopTimer = TimeMonitor::getNewCounter
        (execSpaceName + ", " + memSpaceName + ", self messages: All");
      auto exchangeTimer = TimeMonitor::getNewCounter
        (execSpaceName + ", " + memSpaceName + ", self messages: MPI only");

      TimeMonitor wholeLoopTimeMon (*wholeLoopTimer);
      for (int trial = 0; trial < args.numTrials; ++trial) {
        packAndPrepare (sendBuf, startValue,
                        args.fenceDeviceBeforeMpi);
        {
          TimeMonitor exchangeTimeMon (*exchangeTimer);
          exchangeMessages (recvBuf, sendBuf, recvProc, sendProc,
                            msgTag, comm, makeMpiUseHostBuf);
        }
        (void) unpackAndCombine (recvBuf, startValue);
      }
    }

    if (numProcs > 1) {
      out << "Test messages between processes" << endl;
      Teuchos::OSTab tab2 (out);

      const int recvProc = (myRank + 1) % numProcs;
      const int sendProc = (myRank + 1) % numProcs;
      const int msgTag = 31;
      const packet_type sendStartValue = (myRank == 0) ? 93 : 418;
      const packet_type recvStartValue = (myRank == 0) ? 418 : 93;
      const bool makeMpiUseHostBuf = false;

      auto wholeLoopTimer = TimeMonitor::getNewCounter
        (execSpaceName + ", " + memSpaceName + ", process pairs: All");
      auto exchangeTimer = TimeMonitor::getNewCounter
        (execSpaceName + ", " + memSpaceName + ", process pairs: MPI only");

      TimeMonitor wholeLoopTimeMon (*wholeLoopTimer);
      for (int trial = 0; trial < args.numTrials; ++trial) {
        packAndPrepare (sendBuf, sendStartValue,
                        args.fenceDeviceBeforeMpi);
        {
          TimeMonitor exchangeTimeMon (*exchangeTimer);
          exchangeMessages (recvBuf, sendBuf, recvProc, sendProc,
                            msgTag, comm, makeMpiUseHostBuf);
        }
        (void) unpackAndCombine (recvBuf, recvStartValue);
      }
    }
  }

  Teuchos::RCP<Teuchos::FancyOStream>
  getOutputStream (const Teuchos::Comm<int>& comm)
  {
    Teuchos::RCP<std::ostream> outPtr;
    if (comm.getRank () == 0) {
      outPtr = Teuchos::rcpFromRef (std::cout);
    }
    else {
      outPtr = Teuchos::rcp (new Teuchos::oblackholestream ());
    }
    return Teuchos::getFancyOStream (outPtr);
  }

  void
  runBenchmarks (Teuchos::FancyOStream& out,
                 const CmdLineArgs& args,
                 const Teuchos::Comm<int>& comm)
  {
    const int myRank = comm.getRank ();
    const int numProcs = comm.getSize ();

    out << "MPI communication buffer benchmarks" << endl;
    Teuchos::OSTab tab1 (out);

    out << "Number of MPI processes: " << numProcs << endl;
    const bool assumeMpiIsCudaAware =
      Tpetra::Details::Behavior::assumeMpiIsCudaAware ();
    out << "May we assume that MPI is CUDA aware? "
        << (assumeMpiIsCudaAware ? "YES" : "NO") << endl;

    // Using host-pinned memory does not require MPI to be CUDA aware.
    {
      using buf_exec_space = Kokkos::Cuda;
      using buf_mem_space = Kokkos::CudaHostPinnedSpace;
      runBenchmark<buf_exec_space, buf_mem_space> (out, args, comm);
    }
    if (assumeMpiIsCudaAware) {
      using buf_exec_space = Kokkos::Cuda;
      using buf_mem_space = Kokkos::CudaSpace;
      runBenchmark<buf_exec_space, buf_mem_space> (out, args, comm);
    }
    if (assumeMpiIsCudaAware) {
      using buf_exec_space = Kokkos::Cuda;
      using buf_mem_space = Kokkos::CudaUVMSpace;
      runBenchmark<buf_exec_space, buf_mem_space> (out, args, comm);
    }
    {
      using buf_exec_space = Kokkos::DefaultHostExecutionSpace;
      using buf_mem_space = Kokkos::HostSpace;
      runBenchmark<buf_exec_space, buf_mem_space> (out, args, comm);
    }
  }

} // namespace (anonymous)




int
main (int argc, char* argv[])
{
  int errCode = 0;

  // Initialize MPI (if enabled) before initializing Kokkos.  This
  // lets MPI control things like pinning processes to sockets.
  Teuchos::GlobalMPISession mpiScope (&argc, &argv);
  Kokkos::ScopeGuard kokkosScope (argc, argv);

  const CmdLineArgs parsedArgs = getCmdLineArgs (argc, argv);
  errCode = parsedArgs.success ? 0 : -1;
  if (parsedArgs.success && ! parsedArgs.printedHelp) {
    const auto comm = Tpetra::getDefaultComm ();
    auto fancyOutPtr = getOutputStream (*comm);
    auto& out = *fancyOutPtr;

    runBenchmarks (out, parsedArgs, *comm);
    Teuchos::TimeMonitor::report (comm.ptr (), out);
  }
  return errCode;
}
