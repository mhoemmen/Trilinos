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

#include "TpetraNew_Map.hpp"
#include "TpetraNew_Directory.hpp"
#include "Tpetra_Details_FixedHashTable.hpp"
// FIXME mfh 09 Sep 2018 I would really rather not have to include
// this file.  It suggests that I haven't quite hooked up TpetraNew
// correctly to TpetraCore yet.
#include "Tpetra_Details_FixedHashTable_def.hpp"
#include "Tpetra_Details_gathervPrint.hpp"
#include "Tpetra_Details_printOnce.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_Util.hpp"
#include "Teuchos_as.hpp"
#include "Teuchos_TypeNameTraits.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Tpetra_Details_mpiIsInitialized.hpp"
#include "Tpetra_Details_extractMpiCommFromTeuchos.hpp" // teuchosCommIsAnMpiComm
#include "Tpetra_Details_initializeKokkos.hpp"
#include <stdexcept>
#include <typeinfo>

namespace TpetraNew {

  Map::Map () :
    comm_ (new Teuchos::SerialComm<int> ()),
    indexBase_ (0),
    globalNumIndices_ (0),
    myNumIndices_ (0),
    minMyGID_ (::Tpetra::Details::OrdinalTraits<global_ordinal_type>::invalid ()),
    maxMyGID_ (::Tpetra::Details::OrdinalTraits<global_ordinal_type>::invalid ()),
    minAllGID_ (::Tpetra::Details::OrdinalTraits<global_ordinal_type>::invalid ()),
    maxAllGID_ (::Tpetra::Details::OrdinalTraits<global_ordinal_type>::invalid ()),
    firstContiguousGID_ (::Tpetra::Details::OrdinalTraits<global_ordinal_type>::invalid ()),
    lastContiguousGID_ (::Tpetra::Details::OrdinalTraits<global_ordinal_type>::invalid ()),
    uniform_ (false), // trivially
    contiguous_ (false),
    distributed_ (false), // no communicator yet
    directory_ (new Directory ())
  {
    ::Tpetra::Details::initializeKokkos ();
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> > &comm,
       const ::Tpetra::LocalGlobal lOrG) :
    comm_ (comm),
    uniform_ (true),
    directory_ (new Directory ())
  {
    using Teuchos::as;
    using Teuchos::broadcast;
    using Teuchos::outArg;
    using Teuchos::reduceAll;
    using Teuchos::REDUCE_MIN;
    using Teuchos::REDUCE_MAX;
    using Teuchos::typeName;
    using LO = local_ordinal_type;
    using GO = global_ordinal_type;
    const GO GINV = ::Tpetra::Details::OrdinalTraits<GO>::invalid ();

    ::Tpetra::Details::initializeKokkos ();

#ifdef HAVE_TPETRA_DEBUG
    // In debug mode only, check whether numGlobalElements and
    // indexBase are the same over all processes in the communicator.
    {
      GO proc0NumGlobalElements = numGlobalElements;
      broadcast<int, GO> (*comm_, 0, outArg (proc0NumGlobalElements));
      GO minNumGlobalElements = numGlobalElements;
      GO maxNumGlobalElements = numGlobalElements;
      reduceAll<int, GO> (*comm, REDUCE_MIN, numGlobalElements, outArg (minNumGlobalElements));
      reduceAll<int, GO> (*comm, REDUCE_MAX, numGlobalElements, outArg (maxNumGlobalElements));
      TEUCHOS_TEST_FOR_EXCEPTION(
        minNumGlobalElements != maxNumGlobalElements || numGlobalElements != minNumGlobalElements,
        std::invalid_argument,
        "Tpetra::Map constructor: All processes must provide the same number "
        "of global elements.  Process 0 set numGlobalElements = "
        << proc0NumGlobalElements << ".  The calling process "
        << comm->getRank () << " set numGlobalElements = " << numGlobalElements
        << ".  The min and max values over all processes are "
        << minNumGlobalElements << " resp. " << maxNumGlobalElements << ".");

      GO proc0IndexBase = indexBase;
      broadcast<int, GO> (*comm_, 0, outArg (proc0IndexBase));
      GO minIndexBase = indexBase;
      GO maxIndexBase = indexBase;
      reduceAll<int, GO> (*comm, REDUCE_MIN, indexBase, outArg (minIndexBase));
      reduceAll<int, GO> (*comm, REDUCE_MAX, indexBase, outArg (maxIndexBase));
      TEUCHOS_TEST_FOR_EXCEPTION(
        minIndexBase != maxIndexBase || indexBase != minIndexBase,
        std::invalid_argument,
        "Tpetra::Map constructor: "
        "All processes must provide the same indexBase argument.  "
        "Process 0 set indexBase = " << proc0IndexBase << ".  The calling "
        "process " << comm->getRank () << " set indexBase = " << indexBase
        << ".  The min and max values over all processes are "
        << minIndexBase << " resp. " << maxIndexBase << ".");
    }
#endif // HAVE_TPETRA_DEBUG

    // Distribute the elements across the processes in the given
    // communicator so that global IDs (GIDs) are
    //
    // - Nonoverlapping (only one process owns each GID)
    // - Contiguous (the sequence of GIDs is nondecreasing, and no two
    //   adjacent GIDs differ by more than one)
    // - As evenly distributed as possible (the numbers of GIDs on two
    //   different processes do not differ by more than one)

    // All processes have the same numGlobalElements, but we still
    // need to check that it is valid.  numGlobalElements must be
    // positive and not the "invalid" value (GINV).
    //
    // This comparison looks funny, but it avoids compiler warnings
    // just in case GO is unsigned.
    TEUCHOS_TEST_FOR_EXCEPTION
      ((numGlobalElements < 1 && numGlobalElements != 0),
       std::invalid_argument,
       "Tpetra::Map constructor: numGlobalElements (= "
       << numGlobalElements << ") must be nonnegative.");

    TEUCHOS_TEST_FOR_EXCEPTION
      (numGlobalElements == GINV, std::invalid_argument,
       "Tpetra::Map constructor: You provided numGlobalElements = Teuchos::"
       "OrdinalTraits<Map::global_ordinal_type>::invalid().  This constructor "
       "requires a valid value of numGlobalElements.  You probably mistook "
       "this constructor for the \"contiguous nonuniform\" constructor, "
       "which can compute the global number of elements for you "
       "if you set numGlobalElements to that value.");

    LO numLocalElements = 0; // will set below
    if (lOrG == ::Tpetra::GloballyDistributed) {
      // Compute numLocalElements:
      //
      // If numGlobalElements == numProcs * B + remainder,
      // then Proc r gets B+1 elements if r < remainder,
      // and B elements if r >= remainder.
      //
      // This strategy is valid for any value of numGlobalElements and
      // numProcs, including the following border cases:
      //   - numProcs == 1
      //   - numLocalElements < numProcs
      //
      // In the former case, remainder == 0 && numGlobalElements ==
      // numLocalElements.  In the latter case, remainder ==
      // numGlobalElements && numLocalElements is either 0 or 1.
      const GO numProcs = static_cast<GO> (comm_->getSize ());
      const GO myRank = static_cast<GO> (comm_->getRank ());
      const GO quotient  = numGlobalElements / numProcs;
      const GO remainder = numGlobalElements - quotient * numProcs;

      GO startIndex;
      if (myRank < remainder) {
        numLocalElements = static_cast<LO> (1) + static_cast<LO> (quotient);
        // myRank was originally an int, so it should never overflow
        // reasonable GO types.
        startIndex = static_cast<GO> (myRank) *
	  static_cast<GO> (numLocalElements);
      }
      else {
        numLocalElements = static_cast<LO> (quotient);
        startIndex = static_cast<GO> (myRank) *
	  static_cast<GO> (numLocalElements) +
          static_cast<GO> (remainder);
      }

      minMyGID_  = indexBase + startIndex;
      maxMyGID_  = indexBase + startIndex + numLocalElements - 1;
      minAllGID_ = indexBase;
      maxAllGID_ = indexBase + numGlobalElements - 1;
      distributed_ = (numProcs > 1);
    }
    else {  // lOrG == LocallyReplicated
      numLocalElements = static_cast<LO> (numGlobalElements);
      minMyGID_ = indexBase;
      maxMyGID_ = indexBase + numGlobalElements - 1;
      distributed_ = false;
    }

    minAllGID_ = indexBase;
    maxAllGID_ = indexBase + numGlobalElements - 1;
    indexBase_ = indexBase;
    globalNumIndices_ = numGlobalElements;
    myNumIndices_ = numLocalElements;
    firstContiguousGID_ = minMyGID_;
    lastContiguousGID_ = maxMyGID_;
    contiguous_ = true;

    // Create the Directory on demand in getRemoteIndexList().
    //setupDirectory ();
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const local_ordinal_type numLocalElements,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> > &comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    using Teuchos::broadcast;
    using Teuchos::outArg;
    using Teuchos::reduceAll;
    using Teuchos::REDUCE_MIN;
    using Teuchos::REDUCE_MAX;
    using Teuchos::REDUCE_SUM;
    using Teuchos::scan;
    using GO = global_ordinal_type;
    const GO GINV = ::Tpetra::Details::OrdinalTraits<GO>::invalid ();

    ::Tpetra::Details::initializeKokkos ();

#ifdef HAVE_TPETRA_DEBUG
    // Global sum of numLocalElements over all processes.
    // Keep this for later debug checks.
    const GO debugGlobalSum =
      initialNonuniformDebugCheck (numGlobalElements, numLocalElements,
                                   indexBase, comm);
#endif // HAVE_TPETRA_DEBUG

    // Distribute the elements across the processes so that they are
    // - non-overlapping
    // - contiguous

    // This differs from the first Map constructor (that only takes a
    // global number of elements) in that the user has specified the
    // number of local elements, so that the elements are not
    // (necessarily) evenly distributed over the processes.

    // Compute my local offset.  This is an inclusive scan, so to get
    // the final offset, we subtract off the input.
    GO scanResult = 0;
    scan<int, GO> (*comm, REDUCE_SUM, numLocalElements, outArg (scanResult));
    const GO myOffset = scanResult - numLocalElements;

    if (numGlobalElements != GINV) {
      globalNumIndices_ = numGlobalElements; // Use the user's value.
    }
    else {
      // Inclusive scan means that the last process has the final sum.
      // Rather than doing a reduceAll to get the sum of
      // numLocalElements, we can just have the last process broadcast
      // its result.  That saves us a round of log(numProcs) messages.
      const int numProcs = comm->getSize ();
      GO globalSum = scanResult;
      if (numProcs > 1) {
        broadcast (*comm, numProcs - 1, outArg (globalSum));
      }
      globalNumIndices_ = globalSum;

#ifdef HAVE_TPETRA_DEBUG
      // No need for an all-reduce here; both come from collectives.
      TEUCHOS_TEST_FOR_EXCEPTION(
        globalSum != debugGlobalSum, std::logic_error,
        "Tpetra::Map constructor (contiguous nonuniform): "
        "globalSum = " << globalSum << " != debugGlobalSum = " << debugGlobalSum
        << ".  Please report this bug to the Tpetra developers.");
#endif // HAVE_TPETRA_DEBUG
    }
    myNumIndices_ = numLocalElements;
    indexBase_ = indexBase;
    minAllGID_ = (globalNumIndices_ == 0) ?
      std::numeric_limits<GO>::max () :
      indexBase;
    maxAllGID_ = (globalNumIndices_ == 0) ?
      std::numeric_limits<GO>::lowest () :
      indexBase + static_cast<GO> (globalNumIndices_) - static_cast<GO> (1);
    minMyGID_ = (myNumIndices_ == 0) ?
      std::numeric_limits<GO>::max () :
      indexBase + static_cast<GO> (myOffset);
    maxMyGID_ = (myNumIndices_ == 0) ?
      std::numeric_limits<GO>::lowest () :
      indexBase + myOffset + static_cast<GO> (numLocalElements) - static_cast<GO> (1);
    firstContiguousGID_ = minMyGID_;
    lastContiguousGID_ = maxMyGID_;
    contiguous_ = true;
    distributed_ = checkIsDist ();

    // Create the Directory on demand in getRemoteIndexList().
    //setupDirectory ();
  }

  Map::global_ordinal_type
  Map::
  initialNonuniformDebugCheck (const global_ordinal_type numGlobalElements,
                               const local_ordinal_type numLocalElements,
                               const global_ordinal_type indexBase,
                               const Teuchos::RCP<const Teuchos::Comm<int> >& comm) const
  {
#ifdef HAVE_TPETRA_DEBUG
    using Teuchos::broadcast;
    using Teuchos::outArg;
    using Teuchos::ptr;
    using Teuchos::REDUCE_MAX;
    using Teuchos::REDUCE_MIN;
    using Teuchos::REDUCE_SUM;
    using Teuchos::reduceAll;
    using GO = global_ordinal_type;
    const GO GINV = ::Tpetra::Details::OrdinalTraits<GO>::invalid ();

    // The user has specified the distribution of indices over the
    // processes.  The distribution is not necessarily contiguous or
    // equally shared over the processes.

    GO debugGlobalSum = 0; // Will be global sum of numLocalElements
    reduceAll<int, GO> (*comm, REDUCE_SUM, static_cast<GO> (numLocalElements),
                         outArg (debugGlobalSum));
    // In debug mode only, check whether numGlobalElements and
    // indexBase are the same over all processes in the communicator.
    {
      GO proc0NumGlobalElements = numGlobalElements;
      broadcast<int, GO> (*comm, 0, outArg (proc0NumGlobalElements));
      GO minNumGlobalElements = numGlobalElements;
      GO maxNumGlobalElements = numGlobalElements;
      reduceAll<int, GO> (*comm, REDUCE_MIN, numGlobalElements,
			  outArg (minNumGlobalElements));
      reduceAll<int, GO> (*comm, REDUCE_MAX, numGlobalElements,
			  outArg (maxNumGlobalElements));
      TEUCHOS_TEST_FOR_EXCEPTION
	(minNumGlobalElements != maxNumGlobalElements ||
	 numGlobalElements != minNumGlobalElements,
	 std::invalid_argument,
	 "Tpetra::Map constructor: All processes must provide the same number "
	 "of global elements.  This is true even if that argument is Teuchos::"
	 "OrdinalTraits<global_ordinal_t>::invalid() to signal that the Map "
	 "must compute the global number of elements.  Process 0 set "
	 "numGlobalElements = " << proc0NumGlobalElements << ".  The calling "
	 "process " << comm->getRank () << " set numGlobalElements = "
	 << numGlobalElements << ".  The min and max values over all processes "
	 "are " << minNumGlobalElements << " resp. " << maxNumGlobalElements
	 << ".");

      GO proc0IndexBase = indexBase;
      broadcast<int, GO> (*comm_, 0, outArg (proc0IndexBase));
      GO minIndexBase = indexBase;
      GO maxIndexBase = indexBase;
      reduceAll<int, GO> (*comm, REDUCE_MIN, indexBase, outArg (minIndexBase));
      reduceAll<int, GO> (*comm, REDUCE_MAX, indexBase, outArg (maxIndexBase));
      TEUCHOS_TEST_FOR_EXCEPTION(
        minIndexBase != maxIndexBase || indexBase != minIndexBase,
        std::invalid_argument,
        "Tpetra::Map constructor: "
        "All processes must provide the same indexBase argument.  "
        "Process 0 set indexBase = " << proc0IndexBase << ".  The calling "
        "process " << comm->getRank () << " set indexBase = " << indexBase
        << ".  The min and max values over all processes are "
        << minIndexBase << " resp. " << maxIndexBase << ".");

      // Make sure that the sum of numLocalElements over all processes
      // equals numGlobalElements.
      TEUCHOS_TEST_FOR_EXCEPTION
        (numGlobalElements != GINV && debugGlobalSum != numGlobalElements,
         std::invalid_argument, "Tpetra::Map constructor: The sum of each "
         "process' number of indices over all processes, " << debugGlobalSum
         << " != numGlobalElements = " << numGlobalElements << ".  If you "
         "would like this constructor to compute numGlobalElements for you, "
         "you may set numGlobalElements = "
         "Teuchos::OrdinalTraits<Map::global_ordinal_type>::invalid() on "
	 "input.");
    }

    return debugGlobalSum;
#else
    return global_ordinal_type (0);
#endif // HAVE_TPETRA_DEBUG
  }

  void
  Map::
  initWithNonownedHostIndexList (const global_ordinal_type numGlobalElements,
                                 const Kokkos::View<const global_ordinal_type*,
                                   Kokkos::LayoutLeft,
                                   Kokkos::HostSpace,
                                   Kokkos::MemoryUnmanaged>& entryList_host,
                                 const global_ordinal_type indexBase,
                                 const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
  {
    using Kokkos::LayoutLeft;
    using Kokkos::subview;
    using Kokkos::View;
    using Teuchos::as;
    using Teuchos::broadcast;
    using Teuchos::outArg;
    using Teuchos::ptr;
    using Teuchos::REDUCE_MAX;
    using Teuchos::REDUCE_MIN;
    using Teuchos::REDUCE_SUM;
    using Teuchos::reduceAll;
    using LO = local_ordinal_type;
    using GO = global_ordinal_type;
    const GO GINV = ::Tpetra::Details::OrdinalTraits<GO>::invalid ();

    // Make sure that Kokkos has been initialized (Github Issue #513).
    TEUCHOS_TEST_FOR_EXCEPTION
      (! Kokkos::is_initialized (), std::runtime_error,
       "Tpetra::Map constructor: The Kokkos execution space "
       << Teuchos::TypeNameTraits<execution_space>::name ()
       << " has not been initialized.  "
       "Please initialize it before creating a Map.")

    // The user has specified the distribution of indices over the
    // processes, via the input array of global indices on each
    // process.  The distribution is not necessarily contiguous or
    // equally shared over the processes.

    // The length of the input array on this process is the number of
    // local indices to associate with this process, even though the
    // input array contains global indices.
    const LO numLocalElements = static_cast<LO> (entryList_host.size ());

    initialNonuniformDebugCheck (numGlobalElements, numLocalElements,
                                 indexBase, comm);

    // NOTE (mfh 20 Feb 2013, 10 Oct 2016) In some sense, this global
    // reduction is redundant, since the directory Map will have to do
    // the same thing.  Thus, we could do the scan and broadcast for
    // the directory Map here, and give the computed offsets to the
    // directory Map's constructor.  However, a reduction costs less
    // than a scan and broadcast, so this still saves time if users of
    // this Map don't ever need the Directory (i.e., if they never
    // call getRemoteIndexList on this Map).
    if (numGlobalElements != GINV) {
      globalNumIndices_ = numGlobalElements; // Use the user's value.
    }
    else { // The user wants us to compute the sum.
      reduceAll<int, GO> (*comm, REDUCE_SUM,
			  static_cast<GO> (numLocalElements),
			  outArg (globalNumIndices_));
    }

    // mfh 20 Feb 2013: We've never quite done the right thing for
    // duplicate GIDs here.  Duplicate GIDs have always been counted
    // distinctly in myNumIndices_, and thus should get a
    // different LID.  However, we've always used std::map or a hash
    // table for the GID -> LID lookup table, so distinct GIDs always
    // map to the same LID.  Furthermore, the order of the input GID
    // list matters, so it's not desirable to sort for determining
    // uniqueness.
    //
    // I've chosen for now to write this code as if the input GID list
    // contains no duplicates.  If this is not desired, we could use
    // the lookup table itself to determine uniqueness: If we haven't
    // seen the GID before, it gets a new LID and it's added to the
    // LID -> GID and GID -> LID tables.  If we have seen the GID
    // before, it doesn't get added to either table.  I would
    // implement this, but it would cost more to do the double lookups
    // in the table (one to check, and one to insert).
    //
    // More importantly, since we build the GID -> LID table in (a
    // thread-) parallel (way), the order in which duplicate GIDs may
    // get inserted is not defined.  This would make the assignment of
    // LID to GID nondeterministic.

    myNumIndices_ = numLocalElements;
    indexBase_ = indexBase;

    minMyGID_ = indexBase_;
    maxMyGID_ = indexBase_;

    // NOTE (mfh 27 May 2015): While finding the initial contiguous
    // GID range requires looking at all the GIDs in the range,
    // dismissing an interval of GIDs only requires looking at the
    // first and last GIDs.  Thus, we could do binary search backwards
    // from the end in order to catch the common case of a contiguous
    // interval followed by noncontiguous entries.  On the other hand,
    // we could just expose this case explicitly as yet another Map
    // constructor, and avoid the trouble of detecting it.
    if (myNumIndices_ > 0) {
      // Find contiguous GID range, with the restriction that the
      // beginning of the range starts with the first entry.  While
      // doing so, fill in the LID -> GID table.
      View<GO*, LayoutLeft, device_type> lgMap ("lgMap", myNumIndices_);
      auto lgMap_host = Kokkos::create_mirror_view (lgMap);

      // The input array entryList_host is already on host, so we
      // don't need to take a host view of it.
      // auto entryList_host = Kokkos::create_mirror_view (entryList);
      // Kokkos::deep_copy (entryList_host, entryList);

      firstContiguousGID_ = entryList_host[0];
      lastContiguousGID_ = firstContiguousGID_+1;

      // FIXME (mfh 23 Sep 2015) We need to copy the input GIDs
      // anyway, so we have to look at them all.  The logical way to
      // find the first noncontiguous entry would thus be to "reduce,"
      // where the local reduction result is whether entryList[i] + 1
      // == entryList[i+1].

      lgMap_host[0] = firstContiguousGID_;
      LO i = 1;
      for ( ; i < myNumIndices_; ++i) {
        const GO curGid = entryList_host[i];
        const LO curLid = i;

        if (lastContiguousGID_ != curGid) break;

        // Add the entry to the LID->GID table only after we know that
        // the current GID is in the initial contiguous sequence, so
        // that we don't repeat adding it in the first iteration of
        // the loop below over the remaining noncontiguous GIDs.
        lgMap_host[curLid] = curGid;
        ++lastContiguousGID_;
      }
      --lastContiguousGID_;

      // [firstContiguousGID_, lastContigousGID_] is the initial
      // sequence of contiguous GIDs.  We can start the min and max
      // GID using this range.
      minMyGID_ = firstContiguousGID_;
      maxMyGID_ = lastContiguousGID_;

      // Compute the GID -> LID lookup table, _not_ including the
      // initial sequence of contiguous GIDs.
      {
        const std::pair<LO, LO> ncRange (i, entryList_host.extent (0));
        auto nonContigGids_host = subview (entryList_host, ncRange);
        TEUCHOS_TEST_FOR_EXCEPTION
          (static_cast<LO> (nonContigGids_host.extent (0)) !=
           static_cast<LO> (entryList_host.extent (0) - i),
           std::logic_error, "Tpetra::Map noncontiguous constructor: "
           "nonContigGids_host.extent(0) = "
           << nonContigGids_host.extent (0)
           << " != entryList_host.extent(0) - i = "
           << (entryList_host.extent (0) - i) << " = "
           << entryList_host.extent (0) << " - " << i
           << ".  Please report this bug to the Tpetra developers.");

        // FixedHashTable's constructor expects an owned device View,
        // so we must deep-copy the subview of the input indices.
        View<GO*, LayoutLeft, device_type>
          nonContigGids ("nonContigGids", nonContigGids_host.size ());
        Kokkos::deep_copy (nonContigGids, nonContigGids_host);

        glMap_ = global_to_local_table_type (nonContigGids,
                                             firstContiguousGID_,
                                             lastContiguousGID_,
                                             static_cast<LO> (i));
      }

      // FIXME (mfh 10 Oct 2016) When we construct the global-to-local
      // table above, we have to look at all the (noncontiguous) input
      // indices anyway.  Thus, why not have the constructor compute
      // and return the min and max?

      for ( ; i < myNumIndices_; ++i) {
        const GO curGid = entryList_host[i];
        const LO curLid = i;
        lgMap_host[curLid] = curGid; // LID -> GID table

        // While iterating through entryList, we compute its
        // (process-local) min and max elements.
        if (curGid < minMyGID_) {
          minMyGID_ = curGid;
        }
        if (curGid > maxMyGID_) {
          maxMyGID_ = curGid;
        }
      }

      // We filled lgMap on host above; now sync back to device.
      Kokkos::deep_copy (lgMap, lgMap_host);

      // "Commit" the local-to-global lookup table we filled in above.
      lgMap_ = lgMap;
      // We've already created this, so use it.
      lgMapHost_ = lgMap_host;
    }
    else {
      minMyGID_ = std::numeric_limits<global_ordinal_type>::max();
      maxMyGID_ = std::numeric_limits<global_ordinal_type>::lowest();
      // This insures tests for GIDs in the range
      // [firstContiguousGID_, lastContiguousGID_] fail for processes
      // with no local elements.
      firstContiguousGID_ = indexBase_+1;
      lastContiguousGID_ = indexBase_;
      // glMap_ was default constructed, so it's already empty.
    }

    // Compute the min and max of all processes' GIDs.  If
    // myNumIndices_ == 0 on this process, minMyGID_ and maxMyGID_
    // are both indexBase_.  This is wrong, but fixing it would
    // require either a fancy sparse all-reduce, or a custom reduction
    // operator that ignores invalid values ("invalid" means
    // Tpetra::Details::OrdinalTraits<GO>::invalid()).
    //
    // Also, while we're at it, use the same all-reduce to figure out
    // if the Map is distributed.  "Distributed" means that there is
    // at least one process with a number of local elements less than
    // the number of global elements.
    //
    // We're computing the min and max of all processes' GIDs using a
    // single MAX all-reduce, because min(x,y) = -max(-x,-y) (when x
    // and y are signed).  (This lets us combine the min and max into
    // a single all-reduce.)  If each process sets localDist=1 if its
    // number of local elements is strictly less than the number of
    // global elements, and localDist=0 otherwise, then a MAX
    // all-reduce on localDist tells us if the Map is distributed (1
    // if yes, 0 if no).  Thus, we can append localDist onto the end
    // of the data and get the global result from the all-reduce.
    if (std::numeric_limits<GO>::is_signed) {
      // Does my process NOT own all the elements?
      const GO localDist =
        (static_cast<GO> (myNumIndices_) < globalNumIndices_) ? 1 : 0;

      GO minMaxInput[3];
      minMaxInput[0] = -minMyGID_;
      minMaxInput[1] = maxMyGID_;
      minMaxInput[2] = localDist;

      GO minMaxOutput[3];
      minMaxOutput[0] = 0;
      minMaxOutput[1] = 0;
      minMaxOutput[2] = 0;
      reduceAll<int, GO> (*comm, REDUCE_MAX, 3, minMaxInput, minMaxOutput);
      minAllGID_ = -minMaxOutput[0];
      maxAllGID_ = minMaxOutput[1];
      const GO globalDist = minMaxOutput[2];
      distributed_ = (comm_->getSize () > 1 && globalDist == 1);
    }
    else { // unsigned; use two reductions
      // This is always correct, no matter the signedness of GO.
      reduceAll<int, GO> (*comm_, REDUCE_MIN, minMyGID_, outArg (minAllGID_));
      reduceAll<int, GO> (*comm_, REDUCE_MAX, maxMyGID_, outArg (maxAllGID_));
      distributed_ = checkIsDist ();
    }

    contiguous_  = false; // "Contiguous" is conservative.

    TEUCHOS_TEST_FOR_EXCEPTION(
      minAllGID_ < indexBase_,
      std::invalid_argument,
      "Tpetra::Map constructor (noncontiguous): "
      "Minimum global ID = " << minAllGID_ << " over all process(es) is "
      "less than the given indexBase = " << indexBase_ << ".");

    // Create the Directory on demand in getRemoteIndexList().
    //setupDirectory ();
  }

  namespace { // (anonymous)
    template<class Input, class Output,
	     const bool same = std::is_same<Input, Output>::value>
    struct Select {};
    
    template<class Input, class Output>
    struct Select<Input, Output, true> {
      static Output get (const Input& x, const Output& /* y */) { return x; }
    };

    template<class Input, class Output>
    struct Select<Input, Output, false> {
      static Output get (const Input& /* x */, const Output& y) { return y; }
    };

    template<class Input, class Output,
	     const bool same = std::is_same<Input, Output>::value>
    Output select (const Input& x, const Output& y) {
      return Select<Input, Output, same>::get (x, y);
    }
    
    template<class InputIndexType, class OutputIndexType, class DeviceType>
    Kokkos::View<const OutputIndexType*, DeviceType>
    convertIndexViews (const Kokkos::View<const InputIndexType*, DeviceType>& inputInds)
    {
      Kokkos::View<const OutputIndexType*, DeviceType> outputInds;
      if (std::is_same<InputIndexType, OutputIndexType>::value) {
	Kokkos::View<OutputIndexType*, DeviceType> outputInds_nc
	  (Kokkos::view_alloc (inputInds.label (), Kokkos::WithoutInitializing),
	   inputInds.extent (0));
	try {
	  ::Tpetra::Details::copyOffsets (outputInds_nc, inputInds);
	}
	catch (std::exception& e) {
	  TEUCHOS_TEST_FOR_EXCEPTION
	    (true, std::runtime_error, "Tpetra::Map constructor: "
	     "One or more input indices overflowed global_ordinal_type.");
	}
	outputInds = outputInds_nc;
      }
      return select (inputInds, outputInds);
    }

    // This one needs to return ArrayRCP, so that it can allocate if needed.
    template<class InputIndexType, class OutputIndexType>
    Kokkos::View<const OutputIndexType*, Kokkos::LayoutLeft,
		 Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace> >
    convertIndexArrayViews (const Teuchos::ArrayView<const InputIndexType>& inputInds)
    {
      using host_device_type = Kokkos::Device<Kokkos::DefaultHostExecutionSpace,
					      Kokkos::HostSpace>;
      Kokkos::View<const InputIndexType*, host_device_type> inputInds_k
	(inputInds.getRawPtr (), inputInds.size ());
      Kokkos::View<const OutputIndexType*, host_device_type> outputInds_k;
      
      if (std::is_same<InputIndexType, OutputIndexType>::value) {
	Kokkos::View<OutputIndexType*, host_device_type> outputInds_nc
	  (Kokkos::view_alloc (std::string ("lgMap"), Kokkos::WithoutInitializing),
	   inputInds.size ());
	try {
	  ::Tpetra::Details::copyOffsets (outputInds_nc, inputInds_k);
	}
	catch (std::exception& e) {
	  TEUCHOS_TEST_FOR_EXCEPTION
	    (true, std::runtime_error, "Tpetra::Map constructor: "
	     "One or more input indices overflowed global_ordinal_type.");
	}
	outputInds_k = outputInds_nc;
      }
      return select (inputInds_k, outputInds_k);
    }
  } // namespace (anonymous)

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Teuchos::ArrayView<const int>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    ::Tpetra::Details::initializeKokkos ();

    const local_ordinal_type numLclInds =
      static_cast<local_ordinal_type> (entryList.size ());
    auto inputInds_k = convertIndexArrayViews<int, global_ordinal_type> (entryList);
    // Not quite sure if I trust both ArrayView and View to behave
    // correctly if the pointer is nonnull but the array length is
    // nonzero, so I'll make sure it's null if the length is zero.
    initWithNonownedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const int indexList[],
       const local_ordinal_type indexListSize,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    Map (numGlobalElements,
	 Teuchos::ArrayView<const int> (indexList, indexListSize),
	 indexBase,
	 comm)
  {}

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Teuchos::ArrayView<const unsigned int>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    ::Tpetra::Details::initializeKokkos ();

    const local_ordinal_type numLclInds =
      static_cast<local_ordinal_type> (entryList.size ());
    auto inputInds_k = convertIndexArrayViews<unsigned int, global_ordinal_type> (entryList);
    // Not quite sure if I trust both ArrayView and View to behave
    // correctly if the pointer is nonnull but the array length is
    // nonzero, so I'll make sure it's null if the length is zero.
    initWithNonownedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const unsigned int indexList[],
       const local_ordinal_type indexListSize,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    Map (numGlobalElements,
	 Teuchos::ArrayView<const unsigned int> (indexList, indexListSize),
	 indexBase,
	 comm)
  {}

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Teuchos::ArrayView<const long>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    ::Tpetra::Details::initializeKokkos ();

    const local_ordinal_type numLclInds =
      static_cast<local_ordinal_type> (entryList.size ());
    auto inputInds_k = convertIndexArrayViews<long, global_ordinal_type> (entryList);
    // Not quite sure if I trust both ArrayView and View to behave
    // correctly if the pointer is nonnull but the array length is
    // nonzero, so I'll make sure it's null if the length is zero.
    initWithNonownedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const long indexList[],
       const local_ordinal_type indexListSize,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    Map (numGlobalElements,
	 Teuchos::ArrayView<const long> (indexList, indexListSize),
	 indexBase,
	 comm)
  {}

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Teuchos::ArrayView<const unsigned long>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    ::Tpetra::Details::initializeKokkos ();

    const local_ordinal_type numLclInds =
      static_cast<local_ordinal_type> (entryList.size ());
    auto inputInds_k = convertIndexArrayViews<unsigned long, global_ordinal_type> (entryList);
    // Not quite sure if I trust both ArrayView and View to behave
    // correctly if the pointer is nonnull but the array length is
    // nonzero, so I'll make sure it's null if the length is zero.
    initWithNonownedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const unsigned long indexList[],
       const local_ordinal_type indexListSize,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    Map (numGlobalElements,
	 Teuchos::ArrayView<const unsigned long> (indexList, indexListSize),
	 indexBase,
	 comm)
  {}

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Teuchos::ArrayView<const long long>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    ::Tpetra::Details::initializeKokkos ();

    const local_ordinal_type numLclInds =
      static_cast<local_ordinal_type> (entryList.size ());
    auto inputInds_k = convertIndexArrayViews<long long, global_ordinal_type> (entryList);
    // Not quite sure if I trust both ArrayView and View to behave
    // correctly if the pointer is nonnull but the array length is
    // nonzero, so I'll make sure it's null if the length is zero.
    initWithNonownedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const long long indexList[],
       const local_ordinal_type indexListSize,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    Map (numGlobalElements,
	 Teuchos::ArrayView<const long long> (indexList, indexListSize),
	 indexBase,
	 comm)
  {}

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Teuchos::ArrayView<const unsigned long long>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    ::Tpetra::Details::initializeKokkos ();

    const local_ordinal_type numLclInds =
      static_cast<local_ordinal_type> (entryList.size ());
    auto inputInds_k = convertIndexArrayViews<unsigned long long, global_ordinal_type> (entryList);
    // Not quite sure if I trust both ArrayView and View to behave
    // correctly if the pointer is nonnull but the array length is
    // nonzero, so I'll make sure it's null if the length is zero.
    initWithNonownedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const unsigned long long indexList[],
       const local_ordinal_type indexListSize,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    Map (numGlobalElements,
	 Teuchos::ArrayView<const unsigned long long> (indexList, indexListSize),
	 indexBase,
	 comm)
  {}

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Kokkos::View<const long long*, device_type>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    auto inputInds_k = convertIndexViews<long long, global_ordinal_type> (entryList);
    initWithOwnedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Kokkos::View<const unsigned long long*, device_type>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    auto inputInds_k = convertIndexViews<unsigned long long, global_ordinal_type> (entryList);
    initWithOwnedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Kokkos::View<const long*, device_type>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    auto inputInds_k = convertIndexViews<long, global_ordinal_type> (entryList);
    initWithOwnedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Kokkos::View<const unsigned long*, device_type>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    auto inputInds_k = convertIndexViews<unsigned long, global_ordinal_type> (entryList);
    initWithOwnedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Kokkos::View<const int*, device_type>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    auto inputInds_k = convertIndexViews<int, global_ordinal_type> (entryList);
    initWithOwnedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }

  Map::
  Map (const global_ordinal_type numGlobalElements,
       const Kokkos::View<const unsigned int*, device_type>& entryList,
       const global_ordinal_type indexBase,
       const Teuchos::RCP<const Teuchos::Comm<int> >& comm) :
    comm_ (comm),
    uniform_ (false),
    directory_ (new Directory ())
  {
    auto inputInds_k = convertIndexViews<unsigned int, global_ordinal_type> (entryList);
    initWithOwnedHostIndexList (numGlobalElements, inputInds_k, indexBase, comm);
  }
  
  void
  Map::
  initWithOwnedHostIndexList (const global_ordinal_type numGlobalElements,
			      const Kokkos::View<const global_ordinal_type*, device_type>& entryList,
			      const global_ordinal_type indexBase,
			      const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
  {
    using Kokkos::LayoutLeft;
    using Kokkos::subview;
    using Kokkos::View;
    using Teuchos::arcp;
    using Teuchos::ArrayView;
    using Teuchos::as;
    using Teuchos::broadcast;
    using Teuchos::outArg;
    using Teuchos::ptr;
    using Teuchos::REDUCE_MAX;
    using Teuchos::REDUCE_MIN;
    using Teuchos::REDUCE_SUM;
    using Teuchos::reduceAll;
    using Teuchos::typeName;
    using LO = local_ordinal_type;
    using GO = global_ordinal_type;
    const GO GINV = ::Tpetra::Details::OrdinalTraits<GO>::invalid ();

    ::Tpetra::Details::initializeKokkos ();

    // The user has specified the distribution of indices over the
    // processes, via the input array of global indices on each
    // process.  The distribution is not necessarily contiguous or
    // equally shared over the processes.

    // The length of the input array on this process is the number of
    // local indices to associate with this process, even though the
    // input array contains global indices.
    const LO numLocalElements = static_cast<LO> (entryList.size ());

    initialNonuniformDebugCheck (numGlobalElements, numLocalElements,
                                 indexBase, comm);

    // NOTE (mfh 20 Feb 2013, 10 Oct 2016) In some sense, this global
    // reduction is redundant, since the directory Map will have to do
    // the same thing.  Thus, we could do the scan and broadcast for
    // the directory Map here, and give the computed offsets to the
    // directory Map's constructor.  However, a reduction costs less
    // than a scan and broadcast, so this still saves time if users of
    // this Map don't ever need the Directory (i.e., if they never
    // call getRemoteIndexList on this Map).
    if (numGlobalElements != GINV) {
      globalNumIndices_ = numGlobalElements; // Use the user's value.
    }
    else { // The user wants us to compute the sum.
      reduceAll<int, GO> (*comm, REDUCE_SUM,
			  static_cast<GO> (numLocalElements),
			  outArg (globalNumIndices_));
    }

    // mfh 20 Feb 2013: We've never quite done the right thing for
    // duplicate GIDs here.  Duplicate GIDs have always been counted
    // distinctly in myNumIndices_, and thus should get a
    // different LID.  However, we've always used std::map or a hash
    // table for the GID -> LID lookup table, so distinct GIDs always
    // map to the same LID.  Furthermore, the order of the input GID
    // list matters, so it's not desirable to sort for determining
    // uniqueness.
    //
    // I've chosen for now to write this code as if the input GID list
    // contains no duplicates.  If this is not desired, we could use
    // the lookup table itself to determine uniqueness: If we haven't
    // seen the GID before, it gets a new LID and it's added to the
    // LID -> GID and GID -> LID tables.  If we have seen the GID
    // before, it doesn't get added to either table.  I would
    // implement this, but it would cost more to do the double lookups
    // in the table (one to check, and one to insert).
    //
    // More importantly, since we build the GID -> LID table in (a
    // thread-) parallel (way), the order in which duplicate GIDs may
    // get inserted is not defined.  This would make the assignment of
    // LID to GID nondeterministic.

    myNumIndices_ = numLocalElements;
    indexBase_ = indexBase;

    minMyGID_ = indexBase_;
    maxMyGID_ = indexBase_;

    // NOTE (mfh 27 May 2015): While finding the initial contiguous
    // GID range requires looking at all the GIDs in the range,
    // dismissing an interval of GIDs only requires looking at the
    // first and last GIDs.  Thus, we could do binary search backwards
    // from the end in order to catch the common case of a contiguous
    // interval followed by noncontiguous entries.  On the other hand,
    // we could just expose this case explicitly as yet another Map
    // constructor, and avoid the trouble of detecting it.
    if (myNumIndices_ > 0) {
      // Find contiguous GID range, with the restriction that the
      // beginning of the range starts with the first entry.  While
      // doing so, fill in the LID -> GID table.
      View<GO*, LayoutLeft, device_type> lgMap ("lgMap", myNumIndices_);
      auto lgMap_host = Kokkos::create_mirror_view (lgMap);

      // Creating the mirror view is trivial, and the deep_copy is a
      // no-op, if entryList is on host already.
      auto entryList_host = Kokkos::create_mirror_view (entryList);
      Kokkos::deep_copy (entryList_host, entryList);

      firstContiguousGID_ = entryList_host[0];
      lastContiguousGID_ = firstContiguousGID_+1;

      // FIXME (mfh 23 Sep 2015) We need to copy the input GIDs
      // anyway, so we have to look at them all.  The logical way to
      // find the first noncontiguous entry would thus be to "reduce,"
      // where the local reduction result is whether entryList[i] + 1
      // == entryList[i+1].

      lgMap_host[0] = firstContiguousGID_;
      LO i = 1;
      for ( ; i < myNumIndices_; ++i) {
        const GO curGid = entryList_host[i];
        const LO curLid = i;

        if (lastContiguousGID_ != curGid) break;

        // Add the entry to the LID->GID table only after we know that
        // the current GID is in the initial contiguous sequence, so
        // that we don't repeat adding it in the first iteration of
        // the loop below over the remaining noncontiguous GIDs.
        lgMap_host[curLid] = curGid;
        ++lastContiguousGID_;
      }
      --lastContiguousGID_;

      // [firstContiguousGID_, lastContigousGID_] is the initial
      // sequence of contiguous GIDs.  We can start the min and max
      // GID using this range.
      minMyGID_ = firstContiguousGID_;
      maxMyGID_ = lastContiguousGID_;

      // Compute the GID -> LID lookup table, _not_ including the
      // initial sequence of contiguous GIDs.
      {
        const std::pair<LO, LO> ncRange (i, entryList.extent (0));
        auto nonContigGids = subview (entryList, ncRange);
        TEUCHOS_TEST_FOR_EXCEPTION
          (static_cast<LO> (nonContigGids.extent (0)) !=
           static_cast<LO> (entryList.extent (0) - i),
           std::logic_error, "Tpetra::Map noncontiguous constructor: "
           "nonContigGids.extent(0) = "
           << nonContigGids.extent (0)
           << " != entryList.extent(0) - i = "
           << (entryList.extent (0) - i) << " = "
           << entryList.extent (0) << " - " << i
           << ".  Please report this bug to the Tpetra developers.");

        glMap_ = global_to_local_table_type (nonContigGids,
                                             firstContiguousGID_,
                                             lastContiguousGID_,
                                             i);
      }

      // FIXME (mfh 10 Oct 2016) When we construct the global-to-local
      // table above, we have to look at all the (noncontiguous) input
      // indices anyway.  Thus, why not have the constructor compute
      // and return the min and max?

      for ( ; i < myNumIndices_; ++i) {
        const GO curGid = entryList_host[i];
        const LO curLid = i;
        lgMap_host[curLid] = curGid; // LID -> GID table

        // While iterating through entryList, we compute its
        // (process-local) min and max elements.
        if (curGid < minMyGID_) {
          minMyGID_ = curGid;
        }
        if (curGid > maxMyGID_) {
          maxMyGID_ = curGid;
        }
      }

      // We filled lgMap on host above; now sync back to device.
      Kokkos::deep_copy (lgMap, lgMap_host);

      // "Commit" the local-to-global lookup table we filled in above.
      lgMap_ = lgMap;
      // We've already created this, so use it.
      lgMapHost_ = lgMap_host;
    }
    else {
      minMyGID_ = std::numeric_limits<global_ordinal_type>::max();
      maxMyGID_ = std::numeric_limits<global_ordinal_type>::lowest();
      // This insures tests for GIDs in the range
      // [firstContiguousGID_, lastContiguousGID_] fail for processes
      // with no local elements.
      firstContiguousGID_ = indexBase_+1;
      lastContiguousGID_ = indexBase_;
      // glMap_ was default constructed, so it's already empty.
    }

    // Compute the min and max of all processes' GIDs.  If
    // myNumIndices_ == 0 on this process, minMyGID_ and maxMyGID_
    // are both indexBase_.  This is wrong, but fixing it would
    // require either a fancy sparse all-reduce, or a custom reduction
    // operator that ignores invalid values ("invalid" means
    // ::Tpetra::Details::OrdinalTraits<GO>::invalid()).
    //
    // Also, while we're at it, use the same all-reduce to figure out
    // if the Map is distributed.  "Distributed" means that there is
    // at least one process with a number of local elements less than
    // the number of global elements.
    //
    // We're computing the min and max of all processes' GIDs using a
    // single MAX all-reduce, because min(x,y) = -max(-x,-y) (when x
    // and y are signed).  (This lets us combine the min and max into
    // a single all-reduce.)  If each process sets localDist=1 if its
    // number of local elements is strictly less than the number of
    // global elements, and localDist=0 otherwise, then a MAX
    // all-reduce on localDist tells us if the Map is distributed (1
    // if yes, 0 if no).  Thus, we can append localDist onto the end
    // of the data and get the global result from the all-reduce.
    if (std::numeric_limits<GO>::is_signed) {
      // Does my process NOT own all the elements?
      const GO localDist =
        (static_cast<GO> (myNumIndices_) < globalNumIndices_) ? 1 : 0;

      GO minMaxInput[3];
      minMaxInput[0] = -minMyGID_;
      minMaxInput[1] = maxMyGID_;
      minMaxInput[2] = localDist;

      GO minMaxOutput[3];
      minMaxOutput[0] = 0;
      minMaxOutput[1] = 0;
      minMaxOutput[2] = 0;
      reduceAll<int, GO> (*comm, REDUCE_MAX, 3, minMaxInput, minMaxOutput);
      minAllGID_ = -minMaxOutput[0];
      maxAllGID_ = minMaxOutput[1];
      const GO globalDist = minMaxOutput[2];
      distributed_ = (comm_->getSize () > 1 && globalDist == 1);
    }
    else { // unsigned; use two reductions
      // This is always correct, no matter the signedness of GO.
      reduceAll<int, GO> (*comm_, REDUCE_MIN, minMyGID_, outArg (minAllGID_));
      reduceAll<int, GO> (*comm_, REDUCE_MAX, maxMyGID_, outArg (maxAllGID_));
      distributed_ = checkIsDist ();
    }

    contiguous_  = false; // "Contiguous" is conservative.

    TEUCHOS_TEST_FOR_EXCEPTION(
      minAllGID_ < indexBase_,
      std::invalid_argument,
      "Tpetra::Map constructor (noncontiguous): "
      "Minimum global ID = " << minAllGID_ << " over all process(es) is "
      "less than the given indexBase = " << indexBase_ << ".");

    // Create the Directory on demand in getRemoteIndexList().
    //setupDirectory ();
  }


  Map::~Map ()
  {
    if (! Kokkos::is_initialized ()) {
      std::ostringstream os;
      os << "WARNING: Tpetra::Map destructor (~Map()) is being called after "
        "Kokkos::finalize() has been called.  This is user error!  There are "
        "two likely causes: " << std::endl <<
        "  1. You have a static Tpetra::Map (or RCP or shared_ptr of a Map)"
         << std::endl <<
        "  2. You declare and construct a Tpetra::Map (or RCP or shared_ptr "
        "of a Tpetra::Map) at the same scope in main() as Kokkos::finalize() "
        "or Tpetra::finalize()." << std::endl << std::endl <<
        "Don't do either of these!  Please refer to GitHib Issue #2372."
         << std::endl;
      ::Tpetra::Details::printOnce (std::cerr, os.str (),
                                    this->getComm ().getRawPtr ());
    }
    else {
      using ::Tpetra::Details::mpiIsInitialized;
      using ::Tpetra::Details::mpiIsFinalized;
      using ::Tpetra::Details::teuchosCommIsAnMpiComm;

      Teuchos::RCP<const Teuchos::Comm<int> > comm = this->getComm ();
      if (! comm.is_null () && teuchosCommIsAnMpiComm (*comm) &&
          mpiIsInitialized () && mpiIsFinalized ()) {
        // Tpetra itself does not require MPI, even if building with
        // MPI.  It is legal to create Tpetra objects that do not use
        // MPI, even in an MPI program.  However, calling Tpetra stuff
        // after MPI_Finalize() has been called is a bad idea, since
        // some Tpetra defaults may use MPI if available.
        std::ostringstream os;
        os << "WARNING: Tpetra::Map destructor (~Map()) is being called after "
          "MPI_Finalize() has been called.  This is user error!  There are "
          "two likely causes: " << std::endl <<
          "  1. You have a static Tpetra::Map (or RCP or shared_ptr of a Map)"
           << std::endl <<
          "  2. You declare and construct a Tpetra::Map (or RCP or shared_ptr "
          "of a Tpetra::Map) at the same scope in main() as MPI_finalize() or "
          "Tpetra::finalize()." << std::endl << std::endl <<
          "Don't do either of these!  Please refer to GitHib Issue #2372."
           << std::endl;
        ::Tpetra::Details::printOnce (std::cerr, os.str (), comm.getRawPtr ());
      }
    }
    // mfh 20 Mar 2018: We can't check Tpetra::isInitialized() yet,
    // because Tpetra does not yet require Tpetra::initialize /
    // Tpetra::finalize.
  }


  bool
  Map::isOneToOne () const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(
      getComm ().is_null (), std::logic_error, "Tpetra::Map::isOneToOne: "
      "getComm() returns null.  Please report this bug to the Tpetra "
      "developers.");

    // This is a collective operation, if it hasn't been called before.
    setupDirectory ();
    return directory_->isOneToOne (*this);
  }

  Map::local_ordinal_type
  Map::getMaxLocalIndex () const
  {
    const local_ordinal_type myNumInds = this->getMyNumIndices ();
    if (myNumInds == 0) {
      return ::Tpetra::Details::OrdinalTraits<local_ordinal_type>::invalid ();
    }
    else { // Local indices are always zero-based.
      return local_ordinal_type (myNumInds - 1);
    }
  }

  Map::local_ordinal_type
  Map::getLocalIndex (const global_ordinal_type globalIndex) const
  {
    if (isContiguous ()) {
      if (globalIndex < getMinGlobalIndex () ||
          globalIndex > getMaxGlobalIndex ()) {
        return ::Tpetra::Details::OrdinalTraits<local_ordinal_type>::invalid ();
      }
      return static_cast<local_ordinal_type> (globalIndex - getMinGlobalIndex ());
    }
    else if (globalIndex >= firstContiguousGID_ &&
             globalIndex <= lastContiguousGID_) {
      return static_cast<local_ordinal_type> (globalIndex - firstContiguousGID_);
    }
    else {
      // If the given global index is not in the table, this returns
      // the same value as OrdinalTraits<local_ordinal_type>::invalid().
      return glMap_.get (globalIndex);
    }
  }

  Map::local_ordinal_type
  Map::getLocalElement (const global_ordinal_type globalIndex) const
  {
    return getLocalIndex (globalIndex);
  }
  
  Map::global_ordinal_type
  Map::getGlobalIndex (const local_ordinal_type localIndex) const
  {
    if (localIndex < getMinLocalIndex () || localIndex > getMaxLocalIndex ()) {
      return ::Tpetra::Details::OrdinalTraits<global_ordinal_type>::invalid ();
    }
    if (isContiguous ()) {
      return getMinGlobalIndex () + localIndex;
    }
    else {
      // This is a host Kokkos::View access, with no RCP or ArrayRCP
      // involvement.  As a result, it is thread safe.
      //
      // lgMapHost_ is a host pointer; this does NOT assume UVM.
      return lgMapHost_[localIndex];
    }
  }

  Map::global_ordinal_type
  Map::getGlobalElement (const local_ordinal_type localIndex) const
  {
    return getGlobalIndex (localIndex);
  }
  
  bool
  Map::isMyLocalIndex (local_ordinal_type localIndex) const
  {
    if (localIndex < getMinLocalIndex () || localIndex > getMaxLocalIndex ()) {
      return false;
    } else {
      return true;
    }
  }

  bool
  Map::isNodeLocalElement (const local_ordinal_type localIndex) const
  {
    return isMyLocalIndex (localIndex);
  }

  bool
  Map::isMyGlobalIndex (const global_ordinal_type globalIndex) const {
    return this->getLocalIndex (globalIndex) !=
      ::Tpetra::Details::OrdinalTraits<local_ordinal_type>::invalid ();
  }

  bool
  Map::isNodeGlobalElement (const global_ordinal_type globalIndex) const {
    return isMyGlobalIndex (globalIndex);
  }

  bool Map::isUniform () const {
    return uniform_;
  }

  bool Map::isContiguous () const {
    return contiguous_;
  }

  typename Map::local_map_type
  Map::
  getLocalMap () const
  {
    return local_map_type (glMap_, lgMap_, getIndexBase (),
                           getMinGlobalIndex (), getMaxGlobalIndex (),
                           firstContiguousGID_, lastContiguousGID_,
                           getMyNumIndices (), isContiguous ());
  }

  bool
  Map::
  isCompatible (const Map& map) const
  {
    using Teuchos::outArg;
    using Teuchos::REDUCE_MIN;
    using Teuchos::reduceAll;
    //
    // Tests that avoid the Boolean all-reduce below by using
    // globally consistent quantities.
    //
    if (this == &map) {
      // Pointer equality on one process always implies pointer
      // equality on all processes, since Map is immutable.
      return true;
    }
    else if (getComm ()->getSize () != map.getComm ()->getSize ()) {
      // The two communicators have different numbers of processes.
      // It's not correct to call isCompatible() in that case.  This
      // may result in the all-reduce hanging below.
      return false;
    }
    else if (getGlobalNumIndices () != map.getGlobalNumIndices ()) {
      // Two Maps are definitely NOT compatible if they have different
      // global numbers of indices.
      return false;
    }
    else if (isContiguous () && isUniform () &&
             map.isContiguous () && map.isUniform ()) {
      // Contiguous uniform Maps with the same number of processes in
      // their communicators, and with the same global numbers of
      // indices, are always compatible.
      return true;
    }
    else if (! isContiguous () && ! map.isContiguous () &&
             lgMap_.extent (0) != 0 && map.lgMap_.extent (0) != 0 &&
             lgMap_.data () == map.lgMap_.data ()) {
      // Noncontiguous Maps whose global index lists are nonempty and
      // have the same pointer must be the same (and therefore
      // contiguous).
      //
      // Nonempty is important.  For example, consider a communicator
      // with two processes, and two Maps that share this
      // communicator, with zero global indices on the first process,
      // and different nonzero numbers of global indices on the second
      // process.  In that case, on the first process, the pointers
      // would both be NULL.
      return true;
    }

    TEUCHOS_TEST_FOR_EXCEPTION(
      getGlobalNumIndices () != map.getGlobalNumIndices (), std::logic_error,
      "Tpetra::Map::isCompatible: There's a bug in this method.  We've already "
      "checked that this condition is true above, but it's false here.  "
      "Please report this bug to the Tpetra developers.");

    // Do both Maps have the same number of indices on each process?
    const int locallyCompat =
      (getMyNumIndices () == map.getMyNumIndices ()) ? 1 : 0;

    int globallyCompat = 0;
    reduceAll<int, int> (*comm_, REDUCE_MIN, locallyCompat, outArg (globallyCompat));
    return (globallyCompat == 1);
  }

  bool
  Map::
  locallySameAs (const Map& map) const
  {
    using Teuchos::ArrayView;
    typedef global_ordinal_type GO;
    typedef typename ArrayView<const GO>::size_type size_type;

    // If both Maps are contiguous, we can compare their GID ranges
    // easily by looking at the min and max GID on this process.
    // Otherwise, we'll compare their GID lists.  If only one Map is
    // contiguous, then we only have to call getNodeElementList() on
    // the noncontiguous Map.  (It's best to avoid calling it on a
    // contiguous Map, since it results in unnecessary storage that
    // persists for the lifetime of the Map.)

    if (this == &map) {
      // Pointer equality on one process always implies pointer
      // equality on all processes, since Map is immutable.
      return true;
    }
    else if (getMyNumIndices () != map.getMyNumIndices ()) {
      return false;
    }
    else if (getMinGlobalIndex () != map.getMinGlobalIndex () ||
             getMaxGlobalIndex () != map.getMaxGlobalIndex ()) {
      return false;
    }
    else {
      if (isContiguous ()) {
        if (map.isContiguous ()) {
          return true; // min and max match, so the ranges match.
        }
        else { // *this is contiguous, but map is not contiguous
          TEUCHOS_TEST_FOR_EXCEPTION(
            ! this->isContiguous () || map.isContiguous (), std::logic_error,
            "Tpetra::Map::locallySameAs: BUG");
          ArrayView<const GO> rhsElts = map.getNodeElementList ();
          const GO minLhsGid = this->getMinGlobalIndex ();
          const size_type numRhsElts = rhsElts.size ();
          for (size_type k = 0; k < numRhsElts; ++k) {
            const GO curLhsGid = minLhsGid + static_cast<GO> (k);
            if (curLhsGid != rhsElts[k]) {
              return false; // stop on first mismatch
            }
          }
          return true;
        }
      }
      else if (map.isContiguous ()) { // *this is not contiguous, but map is
        TEUCHOS_TEST_FOR_EXCEPTION(
          this->isContiguous () || ! map.isContiguous (), std::logic_error,
          "Tpetra::Map::locallySameAs: BUG");
        ArrayView<const GO> lhsElts = this->getNodeElementList ();
        const GO minRhsGid = map.getMinGlobalIndex ();
        const size_type numLhsElts = lhsElts.size ();
        for (size_type k = 0; k < numLhsElts; ++k) {
          const GO curRhsGid = minRhsGid + static_cast<GO> (k);
          if (curRhsGid != lhsElts[k]) {
            return false; // stop on first mismatch
          }
        }
        return true;
      }
      else if (this->lgMap_.data () == map.lgMap_.data ()) {
        // Pointers to LID->GID "map" (actually just an array) are the
        // same, and the number of GIDs are the same.
        return this->getMyNumIndices () == map.getMyNumIndices ();
      }
      else { // we actually have to compare the GIDs
        if (this->getMyNumIndices () != map.getMyNumIndices ()) {
          return false; // We already checked above, but check just in case
        }
        else {
          ArrayView<const GO> lhsElts =     getNodeElementList ();
          ArrayView<const GO> rhsElts = map.getNodeElementList ();

          // std::equal requires that the latter range is as large as
          // the former.  We know the ranges have equal length, because
          // they have the same number of local entries.
          return std::equal (lhsElts.begin (), lhsElts.end (), rhsElts.begin ());
        }
      }
    }
  }

  bool
  Map::
  isLocallyFitted (const Map& map) const
  {
    if (this == &map)
      return true;

    // We are going to check if lmap1 is fitted into lmap2
    auto lmap1 = map.getLocalMap();
    auto lmap2 = this->getLocalMap();

    auto numLocalElements1 = lmap1.getMyNumIndices();
    auto numLocalElements2 = lmap2.getMyNumIndices();

    if (numLocalElements1 > numLocalElements2) {
      // There are more indices in the first map on this process than in second map.
      return false;
    }

    if (lmap1.isContiguous () && lmap2.isContiguous ()) {
      // When both Maps are contiguous, just check the interval inclusion.
      return ((lmap1.getMinGlobalIndex () == lmap2.getMinGlobalIndex ()) &&
              (lmap1.getMaxGlobalIndex () <= lmap2.getMaxGlobalIndex ()));
    }

    if (lmap1.getMinGlobalIndex () < lmap2.getMinGlobalIndex () ||
        lmap1.getMaxGlobalIndex () > lmap2.getMaxGlobalIndex ()) {
      // The second map does not include the first map bounds, and thus some of
      // the first map global indices are not in the second map.
      return false;
    }

    using range_type = Kokkos::RangePolicy<local_ordinal_type, execution_space>;

    // Check all elements.
    local_ordinal_type numDiff = 0;
    Kokkos::parallel_reduce("isLocallyFitted", range_type(0, numLocalElements1),
      KOKKOS_LAMBDA(const local_ordinal_type i, local_ordinal_type& diff) {
        diff += (lmap1.getGlobalIndex(i) != lmap2.getGlobalIndex(i));
      }, numDiff);

    return (numDiff == 0);
  }

  bool
  Map::
  isSameAs (const Map& map) const
  {
    using Teuchos::outArg;
    using Teuchos::REDUCE_MIN;
    using Teuchos::reduceAll;
    //
    // Tests that avoid the Boolean all-reduce below by using
    // globally consistent quantities.
    //
    if (this == &map) {
      // Pointer equality on one process always implies pointer
      // equality on all processes, since Map is immutable.
      return true;
    }
    else if (getComm ()->getSize () != map.getComm ()->getSize ()) {
      // The two communicators have different numbers of processes.
      // It's not correct to call isSameAs() in that case.  This
      // may result in the all-reduce hanging below.
      return false;
    }
    else if (getGlobalNumIndices () != map.getGlobalNumIndices ()) {
      // Two Maps are definitely NOT the same if they have different
      // global numbers of indices.
      return false;
    }
    else if (getMinAllGlobalIndex () != map.getMinAllGlobalIndex () ||
             getMaxAllGlobalIndex () != map.getMaxAllGlobalIndex () ||
             getIndexBase () != map.getIndexBase ()) {
      // If the global min or max global index doesn't match, or if
      // the index base doesn't match, then the Maps aren't the same.
      return false;
    }
    else if (isDistributed () != map.isDistributed ()) {
      // One Map is distributed and the other is not, which means that
      // the Maps aren't the same.
      return false;
    }
    else if (isContiguous () && isUniform () &&
             map.isContiguous () && map.isUniform ()) {
      // Contiguous uniform Maps with the same number of processes in
      // their communicators, with the same global numbers of indices,
      // and with matching index bases and ranges, must be the same.
      return true;
    }

    // The two communicators must have the same number of processes,
    // with process ranks occurring in the same order.  This uses
    // MPI_COMM_COMPARE.  The MPI 3.1 standard (Section 6.4) says:
    // "Operations that access communicators are local and their
    // execution does not require interprocess communication."
    // However, just to be sure, I'll put this call after the above
    // tests that don't communicate.
    if (! ::Tpetra::Details::congruent (*comm_, * (map.getComm ()))) {
      return false;
    }

    // If we get this far, we need to check local properties and then
    // communicate local sameness across all processes.
    const int isSame_lcl = locallySameAs (map) ? 1 : 0;

    // Return true if and only if all processes report local sameness.
    int isSame_gbl = 0;
    reduceAll<int, int> (*comm_, REDUCE_MIN, isSame_lcl, outArg (isSame_gbl));
    return isSame_gbl == 1;
  }

  namespace { // (anonymous)
    template <class LO, class GO, class DT>
    class FillLgMap {
    public:
      FillLgMap (const Kokkos::View<GO*, DT>& lgMap,
                 const GO startGid) :
        lgMap_ (lgMap), startGid_ (startGid)
      {
        Kokkos::RangePolicy<LO, typename DT::execution_space>
          range (static_cast<LO> (0), static_cast<LO> (lgMap.size ()));
        Kokkos::parallel_for (range, *this);
      }

      KOKKOS_INLINE_FUNCTION void operator () (const LO& lid) const {
        lgMap_(lid) = startGid_ + static_cast<GO> (lid);
      }

    private:
      const Kokkos::View<GO*, DT> lgMap_;
      const GO startGid_;
    };

  } // namespace (anonymous)


  typename Map::global_indices_array_type
  Map::getMyGlobalIndices () const
  {
    typedef local_ordinal_type LO;
    typedef global_ordinal_type GO;
    typedef device_type DT;

    typedef decltype (lgMap_) const_lg_view_type;
    typedef typename const_lg_view_type::non_const_type lg_view_type;

    // If the local-to-global mapping doesn't exist yet, and if we
    // have local entries, then create and fill the local-to-global
    // mapping.
    const bool needToCreateLocalToGlobalMapping =
      lgMap_.extent (0) == 0 && myNumIndices_ > 0;

    if (needToCreateLocalToGlobalMapping) {
#ifdef HAVE_TEUCHOS_DEBUG
      // The local-to-global mapping should have been set up already
      // for a noncontiguous map.
      TEUCHOS_TEST_FOR_EXCEPTION( ! isContiguous(), std::logic_error,
        "Tpetra::Map::getNodeElementList: The local-to-global mapping (lgMap_) "
        "should have been set up already for a noncontiguous Map.  Please report"
        " this bug to the Tpetra team.");
#endif // HAVE_TEUCHOS_DEBUG

      const LO numElts = static_cast<LO> (getMyNumIndices ());

      lg_view_type lgMap ("lgMap", numElts);
      FillLgMap<LO, GO, DT> fillIt (lgMap, minMyGID_);

      auto lgMapHost = Kokkos::create_mirror_view (lgMap);
      Kokkos::deep_copy (lgMapHost, lgMap);

      // "Commit" the local-to-global lookup table we filled in above.
      lgMap_ = lgMap;
      lgMapHost_ = lgMapHost;

      // lgMapHost_ may be a UVM View, so insert a fence to ensure
      // coherent host access.  We only need to do this once, because
      // lgMapHost_ is immutable after initialization.
      execution_space::fence ();
    }

    return lgMap_;
  }

  Teuchos::ArrayView<const Map::global_ordinal_type>
  Map::getNodeElementList () const
  {
    using GO = global_ordinal_type; // convenient abbreviation

    // If the local-to-global mapping doesn't exist yet, and if we
    // have local entries, then create and fill the local-to-global
    // mapping.
    (void) this->getMyGlobalIndices ();

    // This does NOT assume UVM; lgMapHost_ is a host pointer.
    const GO* lgMapHostRawPtr = lgMapHost_.data ();
    // The third argument forces ArrayView not to try to track memory
    // in a debug build.  We have to use it because the memory does
    // not belong to a Teuchos memory management class.
    return Teuchos::ArrayView<const GO> (lgMapHostRawPtr,
                                         lgMapHost_.extent (0),
                                         Teuchos::RCP_DISABLE_NODE_LOOKUP);
  }

  bool Map::isDistributed() const {
    return distributed_;
  }

  std::string Map::description() const {
    using Teuchos::TypeNameTraits;
    std::ostringstream os;

    os << "Tpetra::Map: {"
       << "local_ordinal_type: " << TypeNameTraits<local_ordinal_type>::name ()
       << ", global_ordinal_type: " << TypeNameTraits<global_ordinal_type>::name ()
       << ", execution_space: " << TypeNameTraits<execution_space>::name ()
       << ", memory_space: " << TypeNameTraits<memory_space>::name ();
    if (this->getObjectLabel () != "") {
      os << ", Label: \"" << this->getObjectLabel () << "\"";
    }
    os << ", Global number of entries: " << getGlobalNumIndices ()
       << ", Number of processes: " << getComm ()->getSize ()
       << ", Uniform: " << (isUniform () ? "true" : "false")
       << ", Contiguous: " << (isContiguous () ? "true" : "false")
       << ", Distributed: " << (isDistributed () ? "true" : "false")
       << "}";
    return os.str ();
  }

  /// \brief Print the calling process' verbose describe() information
  ///   to the given output string.
  ///
  /// This is an implementation detail of describe().
  std::string
  Map::
  localDescribeToString (const Teuchos::EVerbosityLevel vl) const
  {
    using LO = local_ordinal_type;
    using std::endl;

    // This preserves current behavior of Map.
    if (vl < Teuchos::VERB_HIGH) {
      return std::string ();
    }
    auto outStringP = Teuchos::rcp (new std::ostringstream ());
    Teuchos::RCP<Teuchos::FancyOStream> outp =
      Teuchos::getFancyOStream (outStringP);
    Teuchos::FancyOStream& out = *outp;

    auto comm = this->getComm ();
    const int myRank = comm->getRank ();
    const int numProcs = comm->getSize ();
    out << "Process " << myRank << " of " << numProcs << ":" << endl;
    Teuchos::OSTab tab1 (out);

    const LO numEnt = static_cast<LO> (this->getMyNumIndices ());
    out << "My number of entries: " << numEnt << endl
        << "My minimum global index: " << this->getMinGlobalIndex () << endl
        << "My maximum global index: " << this->getMaxGlobalIndex () << endl;

    if (vl == Teuchos::VERB_EXTREME) {
      out << "My global indices: [";
      const LO minLclInd = this->getMinLocalIndex ();
      for (LO k = 0; k < numEnt; ++k) {
        out << minLclInd + this->getGlobalIndex (k);
        if (k + 1 < numEnt) {
          out << ", ";
        }
      }
      out << "]" << endl;
    }

    out.flush (); // make sure the ostringstream got everything
    return outStringP->str ();
  }

  void
  Map::
  describe (Teuchos::FancyOStream &out,
            const Teuchos::EVerbosityLevel verbLevel) const
  {
    using Teuchos::TypeNameTraits;
    using Teuchos::VERB_DEFAULT;
    using Teuchos::VERB_NONE;
    using Teuchos::VERB_LOW;
    using Teuchos::VERB_HIGH;
    using std::endl;
    using LO = local_ordinal_type;
    using GO = global_ordinal_type;
    const Teuchos::EVerbosityLevel vl =
      (verbLevel == VERB_DEFAULT) ? VERB_LOW : verbLevel;

    if (vl == VERB_NONE) {
      return; // don't print anything
    }
    // If this Map's Comm is null, then the Map does not participate
    // in collective operations with the other processes.  In that
    // case, it is not even legal to call this method.  The reasonable
    // thing to do in that case is nothing.
    auto comm = this->getComm ();
    if (comm.is_null ()) {
      return;
    }
    const int myRank = comm->getRank ();
    const int numProcs = comm->getSize ();

    // Only Process 0 should touch the output stream, but this method
    // in general may need to do communication.  Thus, we may need to
    // preserve the current tab level across multiple "if (myRank ==
    // 0) { ... }" inner scopes.  This is why we sometimes create
    // OSTab instances by pointer, instead of by value.  We only need
    // to create them by pointer if the tab level must persist through
    // multiple inner scopes.
    Teuchos::RCP<Teuchos::OSTab> tab0, tab1;

    if (myRank == 0) {
      // At every verbosity level but VERB_NONE, Process 0 prints.
      // By convention, describe() always begins with a tab before
      // printing.
      tab0 = Teuchos::rcp (new Teuchos::OSTab (out));
      out << "\"Tpetra::Map\":" << endl;
      tab1 = Teuchos::rcp (new Teuchos::OSTab (out));
      {
        out << "Template parameters:" << endl;
        Teuchos::OSTab tab2 (out);
        out << "local_ordinal_type: " << TypeNameTraits<LO>::name () << endl
            << "global_ordinal_type: " << TypeNameTraits<GO>::name () << endl
            << "execution_space: " << TypeNameTraits<execution_space>::name () << endl
	    << "memory_space: " << TypeNameTraits<memory_space>::name () << endl;
      }
      const std::string label = this->getObjectLabel ();
      if (label != "") {
        out << "Label: \"" << label << "\"" << endl;
      }
      out << "Global number of entries: " << getGlobalNumIndices () << endl
          << "Minimum global index: " << getMinAllGlobalIndex () << endl
          << "Maximum global index: " << getMaxAllGlobalIndex () << endl
          << "Index base: " << getIndexBase () << endl
          << "Number of processes: " << numProcs << endl
          << "Uniform: " << (isUniform () ? "true" : "false") << endl
          << "Contiguous: " << (isContiguous () ? "true" : "false") << endl
          << "Distributed: " << (isDistributed () ? "true" : "false") << endl;
    }

    // This is collective over the Map's communicator.
    if (vl >= VERB_HIGH) { // VERB_HIGH or VERB_EXTREME
      const std::string lclStr = this->localDescribeToString (vl);
      ::Tpetra::Details::gathervPrint (out, lclStr, *comm);
    }
  }

  Teuchos::RCP<const Map>
  Map::replaceCommWithSubset (const Teuchos::RCP<const Teuchos::Comm<int> >& newComm) const
  {
    using Teuchos::RCP;
    using Teuchos::rcp;
    using LO = local_ordinal_type;
    using GO = global_ordinal_type;

    // mfh 26 Mar 2013: The lazy way to do this is simply to recreate
    // the Map by calling its ordinary public constructor, using the
    // original Map's data.  This only involves O(1) all-reduces over
    // the new communicator, which in the common case only includes a
    // small number of processes.

    // Create the Map to return.
    if (newComm.is_null () || newComm->getSize () < 1) {
      return Teuchos::null; // my process does not participate in the new Map
    }
    else if (newComm->getSize () == 1) {
      // The case where the new communicator has only one process is
      // easy.  We don't have to communicate to get all the
      // information we need.  Use the default comm to create the new
      // Map, then fill in all the fields directly.
      RCP<Map> newMap (new Map ());

      newMap->comm_ = newComm;
      // mfh 07 Oct 2016: Preserve original behavior, even though the
      // original index base may no longer be the globally min global
      // index.  See #616 for why this doesn't matter so much anymore.
      newMap->indexBase_ = this->indexBase_;
      newMap->globalNumIndices_ = this->myNumIndices_;
      newMap->myNumIndices_ = this->myNumIndices_;
      newMap->minMyGID_ = this->minMyGID_;
      newMap->maxMyGID_ = this->maxMyGID_;
      newMap->minAllGID_ = this->minMyGID_;
      newMap->maxAllGID_ = this->maxMyGID_;
      newMap->firstContiguousGID_ = this->firstContiguousGID_;
      newMap->lastContiguousGID_ = this->lastContiguousGID_;
      // Since the new communicator has only one process, neither
      // uniformity nor contiguity have changed.
      newMap->uniform_ = this->uniform_;
      newMap->contiguous_ = this->contiguous_;
      // The new communicator only has one process, so the new Map is
      // not distributed.
      newMap->distributed_ = false;
      newMap->lgMap_ = this->lgMap_;
      newMap->lgMapHost_ = this->lgMapHost_;
      newMap->glMap_ = this->glMap_;
      // It's OK not to initialize the new Map's Directory.
      // This is initialized lazily, on first call to getRemoteIndexList.

      return newMap;
    }
    else { // newComm->getSize() != 1
      // Even if the original Map is contiguous, the new Map might not
      // be, especially if the excluded processes have ranks != 0 or
      // newComm->getSize()-1.  The common case for this method is to
      // exclude many (possibly even all but one) processes, so it
      // likely doesn't pay to do the global communication (over the
      // original communicator) to figure out whether we can optimize
      // the result Map.  Thus, we just set up the result Map as
      // noncontiguous.
      //
      // TODO (mfh 07 Oct 2016) We don't actually need to reconstruct
      // the global-to-local table, etc.  Optimize this code path to
      // avoid unnecessary local work.

      // Make Map (re)compute the global number of elements.
      const GO RECOMPUTE = ::Tpetra::Details::OrdinalTraits<GO>::invalid ();
      // TODO (mfh 07 Oct 2016) If we use any Map constructor, we have
      // to use the noncontiguous Map constructor, since the new Map
      // might not be contiguous.  Even if the old Map was contiguous,
      // some process in the "middle" might have been excluded.  If we
      // want to avoid local work, we either have to do the setup by
      // hand, or write a new Map constructor.

      Kokkos::View<const global_ordinal_type*, device_type> lgMap =
	this->getMyGlobalIndices ();
      typedef typename std::decay<decltype (lgMap.extent (0)) >::type size_type;
      const size_type lclNumInds =
        static_cast<size_type> (this->getMyNumIndices ());
      using Teuchos::TypeNameTraits;
      TEUCHOS_TEST_FOR_EXCEPTION
        (lgMap.extent (0) != lclNumInds, std::logic_error,
         "Tpetra::Map::replaceCommWithSubset: Result of getMyGlobalIndices() "
         "has length " << lgMap.extent (0) << " (of type " <<
         TypeNameTraits<size_type>::name () << ") != this->getMyNumIndices()"
         " = " << this->getMyNumIndices () << ".  The latter, upon being "
         "cast to size_type = " << TypeNameTraits<size_type>::name () << ", "
         "becomes " << lclNumInds << ".  Please report this bug to the Tpetra "
         "developers.");

      const GO indexBase = this->getIndexBase ();
      return rcp (new Map (RECOMPUTE, lgMap, indexBase, newComm));
    }
  }

  Teuchos::RCP<const Map>
  Map::removeEmptyProcesses () const
  {
    using Teuchos::Comm;
    using Teuchos::null;
    using Teuchos::outArg;
    using Teuchos::RCP;
    using Teuchos::rcp;
    using Teuchos::REDUCE_MIN;
    using Teuchos::reduceAll;

    // Create the new communicator.  split() returns a valid
    // communicator on all processes.  On processes where color == 0,
    // ignore the result.  Passing key == 0 tells MPI to order the
    // processes in the new communicator by their rank in the old
    // communicator.
    const int color = (myNumIndices_ == 0) ? 0 : 1;
    // MPI_Comm_split must be called collectively over the original
    // communicator.  We can't just call it on processes with color
    // one, even though we will ignore its result on processes with
    // color zero.
    RCP<const Comm<int> > newComm = comm_->split (color, 0);
    if (color == 0) {
      newComm = null;
    }

    // Create the Map to return.
    if (newComm.is_null ()) {
      return null; // my process does not participate in the new Map
    } else {
      RCP<Map> map            = rcp (new Map ());

      map->comm_              = newComm;
      map->indexBase_         = indexBase_;
      map->globalNumIndices_ = globalNumIndices_;
      map->myNumIndices_  = myNumIndices_;
      map->minMyGID_          = minMyGID_;
      map->maxMyGID_          = maxMyGID_;
      map->minAllGID_         = minAllGID_;
      map->maxAllGID_         = maxAllGID_;
      map->firstContiguousGID_= firstContiguousGID_;
      map->lastContiguousGID_ = lastContiguousGID_;

      // Uniformity and contiguity have not changed.  The directory
      // has changed, but we've taken care of that above.
      map->uniform_    = uniform_;
      map->contiguous_ = contiguous_;

      // If the original Map was NOT distributed, then the new Map
      // cannot be distributed.
      //
      // If the number of processes in the new communicator is 1, then
      // the new Map is not distributed.
      //
      // Otherwise, we have to check the new Map using an all-reduce
      // (over the new communicator).  For example, the original Map
      // may have had some processes with zero elements, and all other
      // processes with the same number of elements as in the whole
      // Map.  That Map is technically distributed, because of the
      // processes with zero elements.  Removing those processes would
      // make the new Map locally replicated.
      if (! distributed_ || newComm->getSize () == 1) {
        map->distributed_ = false;
      } else {
        const int iOwnAllGids = (myNumIndices_ == globalNumIndices_) ? 1 : 0;
        int allProcsOwnAllGids = 0;
        reduceAll<int, int> (*newComm, REDUCE_MIN, iOwnAllGids, outArg (allProcsOwnAllGids));
        map->distributed_ = (allProcsOwnAllGids == 1) ? false : true;
      }

      map->lgMap_ = lgMap_;
      map->lgMapHost_ = lgMapHost_;
      map->glMap_ = glMap_;

      // Map's default constructor creates an uninitialized Directory.
      // The Directory will be initialized on demand in
      // getRemoteIndexList().
      //
      // FIXME (mfh 26 Mar 2013) It should be possible to "filter" the
      // directory more efficiently than just recreating it.  If
      // directory recreation proves a bottleneck, we can always
      // revisit this.  On the other hand, Directory creation is only
      // collective over the new, presumably much smaller
      // communicator, so it may not be worth the effort to optimize.

      return map;
    }
  }

  void
  Map::setupDirectory () const
  {
    TEUCHOS_TEST_FOR_EXCEPTION(
      directory_.is_null (), std::logic_error, "Tpetra::Map::setupDirectory: "
      "The Directory is null.  "
      "Please report this bug to the Tpetra developers.");

    // Only create the Directory if it hasn't been created yet.
    // This is a collective operation.
    if (! directory_->initialized ()) {
      directory_->initialize (*this);
    }
  }

  ::Tpetra::LookupStatus
  Map::
  getRemoteIndexList (const Teuchos::ArrayView<const global_ordinal_type>& GIDs,
                      const Teuchos::ArrayView<int>& PIDs,
                      const Teuchos::ArrayView<local_ordinal_type>& LIDs) const
  {
    using ::Tpetra::Details::OrdinalTraits;
    typedef Teuchos::ArrayView<int>::size_type size_type;

    // Empty Maps (i.e., containing no indices on any processes in the
    // Map's communicator) are perfectly valid.  In that case, if the
    // input GID list is nonempty, we fill the output arrays with
    // invalid values, and return IDNotPresent to notify the caller.
    // It's perfectly valid to give getRemoteIndexList GIDs that the
    // Map doesn't own.  SubmapImport test 2 needs this functionality.
    if (getGlobalNumIndices () == 0) {
      if (GIDs.size () == 0) {
        return ::Tpetra::AllIDsPresent; // trivially
      }
      else {
        for (size_type k = 0; k < PIDs.size (); ++k) {
          PIDs[k] = OrdinalTraits<int>::invalid ();
        }
        for (size_type k = 0; k < LIDs.size (); ++k) {
          LIDs[k] = OrdinalTraits<local_ordinal_type>::invalid ();
        }
        return ::Tpetra::IDNotPresent;
      }
    }

    // getRemoteIndexList must be called collectively, and Directory
    // initialization is collective too, so it's OK to initialize the
    // Directory on demand.
    setupDirectory ();
    return directory_->getDirectoryEntries (*this, GIDs, PIDs, LIDs);
  }

  ::Tpetra::LookupStatus
  Map::
  getRemoteIndexList (const Teuchos::ArrayView<const global_ordinal_type> & GIDs,
                      const Teuchos::ArrayView<int> & PIDs) const
  {
    if (getGlobalNumIndices () == 0) {
      if (GIDs.size () == 0) {
        return ::Tpetra::AllIDsPresent; // trivially
      } else {
        // The Map contains no indices, so all output PIDs are invalid.
        for (Teuchos::ArrayView<int>::size_type k = 0; k < PIDs.size (); ++k) {
          PIDs[k] = ::Tpetra::Details::OrdinalTraits<int>::invalid ();
        }
        return ::Tpetra::IDNotPresent;
      }
    }

    // getRemoteIndexList must be called collectively, and Directory
    // initialization is collective too, so it's OK to initialize the
    // Directory on demand.
    setupDirectory ();
    return directory_->getDirectoryEntries (*this, GIDs, PIDs);
  }

  Teuchos::RCP<const Teuchos::Comm<int> >
  Map::getComm () const {
    return comm_;
  }

  Teuchos::RCP<Map::node_type>
  Map::getNode () const
  {
    // Node instances don't do anything any more, but sometimes it
    // helps for them to be nonnull.
    using node_type =
      Kokkos::Compat::KokkosDeviceWrapperNode<execution_space, memory_space>;
    return Teuchos::rcp (new node_type);
  }

  bool
  Map::checkIsDist() const
  {
    using Teuchos::as;
    using Teuchos::outArg;
    using Teuchos::REDUCE_MIN;
    using Teuchos::reduceAll;
    using GO = global_ordinal_type;

    bool global = false;
    if (comm_->getSize () > 1) {
      // The communicator has more than one process, but that doesn't
      // necessarily mean the Map is distributed.
      int localRep = 0;
      if (globalNumIndices_ == GO (myNumIndices_)) {
        // The number of local elements on this process equals the
        // number of global elements.
        //
        // NOTE (mfh 22 Nov 2011) Does this still work if there were
        // duplicates in the global ID list on input (the third Map
        // constructor), so that the number of local elements (which
        // are not duplicated) on this process could be less than the
        // number of global elements, even if this process owns all
        // the elements?
        localRep = 1;
      }
      int allLocalRep;
      reduceAll<int, int> (*comm_, REDUCE_MIN, localRep, outArg (allLocalRep));
      if (allLocalRep != 1) {
        // At least one process does not own all the elements.
        // This makes the Map a distributed Map.
        global = true;
      }
    }
    // If the communicator has only one process, then the Map is not
    // distributed.
    return global;
  }

Teuchos::RCP<const Map>
createLocalMap (const Map::global_ordinal_type lclNumInds,
		const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  const Map::global_ordinal_type indexBase (0);
  return Teuchos::rcp (new Map (lclNumInds, indexBase, comm,
				::Tpetra::LocallyReplicated));
}

Teuchos::RCP<const Map>
createLocalMapWithNode (const Map::local_ordinal_type lclNumInds,
			const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  return createLocalMap (lclNumInds, comm);
}

Teuchos::RCP<const Map>
createUniformContigMap (const Map::global_ordinal_type gblNumInds,
			const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  const Map::global_ordinal_type indexBase (0);
  return Teuchos::rcp (new Map (gblNumInds, indexBase, comm,
				::Tpetra::GloballyDistributed));
}

Teuchos::RCP<const Map>
createUniformContigMapWithNode (const Map::global_ordinal_type gblNumInds,
				const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  return createUniformContigMap (gblNumInds, comm);
}

Teuchos::RCP<const Map>
createContigMap (const Map::global_ordinal_type gblNumInds,
		 const Map::local_ordinal_type lclNumInds,
		 const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  using map_type = Map;
  const map_type::global_ordinal_type indexBase (0);
  return Teuchos::rcp (new map_type (gblNumInds, lclNumInds, indexBase, comm));
}

Teuchos::RCP<const Map>
createContigMapWithNode (const Map::global_ordinal_type gblNumInds,
			 const Map::local_ordinal_type lclNumInds,
			 const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  return createContigMap (gblNumInds, lclNumInds, comm);
}

Teuchos::RCP<const Map>
createNonContigMap (const Teuchos::ArrayView<const Map::global_ordinal_type>& myGblInds,
		    const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  using GO = Map::global_ordinal_type;
  const GO INV = ::Tpetra::Details::OrdinalTraits<GO>::invalid ();
  // FIXME (mfh 22 Jul 2016) This is what I found here, but maybe this
  // shouldn't be zero, given that the index base is supposed to equal
  // the globally min global index?
  const GO indexBase = 0;
  return Teuchos::rcp (new Map (INV, myGblInds, indexBase, comm));
}

Teuchos::RCP<const Map>
createNonContigMapWithNode (const Teuchos::ArrayView<const Map::global_ordinal_type>& myGblInds,
			    const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  return createNonContigMap (myGblInds, comm);
}

Teuchos::RCP<const Map>
createWeightedContigMap (const int myWeight,
			 const Map::global_ordinal_type numElements,
			 const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  using map_type = Map;
  
  Teuchos::RCP<map_type> map;
  int sumOfWeights, elemsLeft, localNumElements;
  const int numImages = comm->getSize();
  const int myImageID = comm->getRank();
  Teuchos::reduceAll<int>(*comm,Teuchos::REDUCE_SUM,myWeight,Teuchos::outArg(sumOfWeights));
  const double myShare = ((double)myWeight) / ((double)sumOfWeights);
  localNumElements = (int)std::floor( myShare * ((double)numElements) );
  // std::cout << "numElements: " << numElements << "  myWeight: " << myWeight << "  sumOfWeights: " << sumOfWeights << "  myShare: " << myShare << std::endl;
  Teuchos::reduceAll<int>(*comm,Teuchos::REDUCE_SUM,localNumElements,Teuchos::outArg(elemsLeft));
  elemsLeft = numElements - elemsLeft;
  // std::cout << "(before) localNumElements: " << localNumElements << "  elemsLeft: " << elemsLeft << std::endl;
  // i think this is true. just test it for now.
  TEUCHOS_TEST_FOR_EXCEPT(elemsLeft < -numImages || numImages < elemsLeft);
  if (elemsLeft < 0) {
    // last elemsLeft nodes lose an element
    if (myImageID >= numImages-elemsLeft) --localNumElements;
  }
  else if (elemsLeft > 0) {
    // first elemsLeft nodes gain an element
    if (myImageID < elemsLeft) ++localNumElements;
  }
  // std::cout << "(after) localNumElements: " << localNumElements << std::endl;
  return createContigMap (numElements, localNumElements, comm);
}

Teuchos::RCP<const Map>
createWeightedContigMapWithNode (const int myWeight,
				 const Map::global_ordinal_type numElements,
				 const Teuchos::RCP<const Teuchos::Comm<int> >& comm)
{
  return createWeightedContigMap (myWeight, numElements, comm);
}

Teuchos::RCP<const Map>
createOneToOne (const Teuchos::RCP<const Map>& M)
{
  using Teuchos::Array;
  using Teuchos::ArrayView;
  using Teuchos::rcp;
  using map_type = Map;
  using LO = map_type::local_ordinal_type;
  using GO = map_type::global_ordinal_type;  
  const GO GINV = ::Tpetra::Details::OrdinalTraits<GO>::invalid ();
  const int myRank = M->getComm ()->getRank ();

  // Bypasses for special cases where either M is known to be
  // one-to-one, or the one-to-one version of M is easy to compute.
  // This is why we take M as an RCP, not as a const reference -- so
  // that we can return M itself if it is 1-to-1.
  if (! M->isDistributed ()) {
    // For a locally replicated Map, we assume that users want to push
    // all the GIDs to Process 0.

    // mfh 05 Nov 2013: getGlobalNumIndices() does indeed return what
    // you think it should return, in this special case of a locally
    // replicated contiguous Map.
    const GO numGlobalEntries = M->getGlobalNumIndices ();
    if (M->isContiguous ()) {
      const LO numLocalEntries =
        (myRank == 0) ? static_cast<LO> (numGlobalEntries) : LO (0);
      return rcp (new map_type (numGlobalEntries, numLocalEntries,
				M->getIndexBase (), M->getComm ()));
    }
    else {
      ArrayView<const GO> myGids =
        (myRank == 0) ? M->getNodeElementList () : Teuchos::null;
      return rcp (new map_type (GINV, myGids (), M->getIndexBase (),
				M->getComm ()));
    }
  }
  else if (M->isContiguous ()) {
    // Contiguous, distributed Maps are one-to-one by construction.
    // (Locally replicated Maps can be contiguous.)
    return M;
  }
  else {
    Directory directory;
    const LO numMyElems = M->getMyNumIndices ();
    ArrayView<const GO> myElems = M->getNodeElementList ();
    Array<int> owner_procs_vec (numMyElems);

    directory.getDirectoryEntries (*M, myElems, owner_procs_vec ());

    Array<GO> myOwned_vec (numMyElems);
    LO numMyOwnedElems = 0;
    for (LO i = 0; i < numMyElems; ++i) {
      const GO GID = myElems[i];
      const int owner = owner_procs_vec[i];

      if (myRank == owner) {
        myOwned_vec[numMyOwnedElems++] = GID;
      }
    }
    myOwned_vec.resize (numMyOwnedElems);

    return rcp (new map_type (GINV, myOwned_vec (),
			      M->getIndexBase (),
			      M->getComm ()));
  }
}

Teuchos::RCP<const Map>
createOneToOne (const Teuchos::RCP<const Map>& M,
		const ::Tpetra::Details::TieBreak<
		  Map::local_ordinal_type,
  		  Map::global_ordinal_type>& tie_break)
{
  using Teuchos::Array;
  using Teuchos::ArrayView;
  using Teuchos::rcp;
  using map_type = Map;
  using LO = map_type::local_ordinal_type;
  using GO = map_type::global_ordinal_type;
  int myID = M->getComm()->getRank();

  // FIXME (mfh 20 Feb 2013) We should have a bypass for contiguous
  // Maps (which are 1-to-1 by construction).

  //Based off Epetra's one to one.

  Directory directory;
  directory.initialize (*M, tie_break);
  LO numMyElems = M->getMyNumIndices ();
  ArrayView<const GO> myElems = M->getNodeElementList ();
  Array<int> owner_procs_vec (numMyElems);

  directory.getDirectoryEntries (*M, myElems, owner_procs_vec ());

  Array<GO> myOwned_vec (numMyElems);
  LO numMyOwnedElems = 0;
  for (LO i = 0; i < numMyElems; ++i) {
    GO GID = myElems[i];
    int owner = owner_procs_vec[i];

    if (myID == owner) {
      myOwned_vec[numMyOwnedElems++] = GID;
    }
  }
  myOwned_vec.resize (numMyOwnedElems);

  // FIXME (mfh 08 May 2014) The above Directory should be perfectly
  // valid for the new Map.  Why can't we reuse it?
  const auto GINV = ::Tpetra::Details::OrdinalTraits<GO>::invalid ();
  return rcp (new map_type (GINV, myOwned_vec (), M->getIndexBase (),
			    M->getComm ()));
}

} // namespace TpetraNew

bool
operator== (const TpetraNew::Map& map1,
	    const TpetraNew::Map& map2)
{
  return map1.isSameAs (map2);
}

bool
operator!= (const TpetraNew::Map& map1,
	    const TpetraNew::Map& map2)
{
  return ! map1.isSameAs (map2);
}
