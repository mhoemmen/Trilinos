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

#ifndef TPETRA_DETAILS_SORTINGDISTRIBUTOR_HPP
#define TPETRA_DETAILS_SORTINGDISTRIBUTOR_HPP

#include "Tpetra_Distributor.hpp"
#include <memory>

namespace Tpetra {
namespace Details {

class SortingDistributor {
public:
  //! Destructor (virtual for memory safety).
  virtual ~SortingDistributor () = default;

  //! Swap the contents of rhs with those of *this.
  void swap (SortingDistributor& rhs);

  // create from sends
  SortingDistributor (size_t& numImports, // out
                      const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                      const Teuchos::ArrayView<const int>& exportPIDs); // in

  // create from receives
  template <class OrdinalType>
  SortingDistributor (Teuchos::Array<OrdinalType>& exportIDs,
                      Teuchos::Array<int>& exportPIDs,
                      const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                      const Teuchos::ArrayView<const OrdinalType>& remoteIDs,
                      const Teuchos::ArrayView<const int>& remotePIDs);

  struct reverse_tag {};

  // create reverse
  SortingDistributor (const SortingDistributor& d, reverse_tag);

  /// \brief Execute the (forward) communication plan.
  ///
  /// Call this version of the method when you have the same number
  /// of Packets for each LID (local ID) to send or receive.
  ///
  /// \tparam Packet The type of data to send and receive.
  ///
  /// \param exports [in] Contains the values to be sent by this
  ///   process.  On exit from this method, it's OK to modify the
  ///   entries of this buffer.
  ///
  /// \param numPackets [in] The number of Packets per export /
  ///   import.  This version of the routine assumes that each LID
  ///   has the same number of Packets associated with it.  (\c
  ///   MultiVector is an example of a DistObject subclass
  ///   satisfying this property.)
  ///
  /// \param imports [out] On entry, buffer must be large enough to
  ///   accomodate the data exported (sent) to us.  On exit,
  ///   contains the values exported to us.
  template <class Packet>
  void
  doPostsAndWaits (const Teuchos::ArrayView<const Packet>& exports,
                   const size_t numPackets,
                   const Teuchos::ArrayView<Packet>& imports);

  /// \brief Execute the (forward) communication plan.
  ///
  /// Call this version of the method when you have possibly
  /// different numbers of Packets for each LID (local ID) to send
  /// or receive.
  ///
  /// \tparam Packet The type of data to send and receive.
  ///
  /// \param exports [in] Contains the values to be sent by this
  ///   process.  On exit from this method, it's OK to modify the
  ///   entries of this buffer.
  ///
  /// \param numExportPacketsPerLID [in] The number of packets for
  ///   each export LID (i.e., each LID to be sent).
  ///
  /// \param imports [out] On entry, buffer must be large enough to
  ///   accomodate the data exported (sent) to us.  On exit,
  ///   contains the values exported to us.
  ///
  /// \param numImportPacketsPerLID [in] The number of packets for
  ///   each import LID (i.e., each LID to be received).
  template <class Packet>
  void
  doPostsAndWaits (const Teuchos::ArrayView<const Packet>& exports,
                   const Teuchos::ArrayView<const size_t>& numExportPacketsPerLID,
                   const Teuchos::ArrayView<Packet>& imports,
                   const Teuchos::ArrayView<const size_t>& numImportPacketsPerLID);

  /// \brief Execute the reverse communication plan.
  ///
  /// This method takes the same arguments as the three-argument
  /// version of doPostsAndWaits().
  template <class Packet>
  void
  doReversePostsAndWaits (const Teuchos::ArrayView<const Packet> &exports,
                          const size_t numPackets,
                          const Teuchos::ArrayView<Packet> &imports);

  /// \brief Execute the reverse communication plan.
  ///
  /// This method takes the same arguments as the four-argument
  /// version of doPostsAndWaits().
  template <class Packet>
  void
  doReversePostsAndWaits (const Teuchos::ArrayView<const Packet> &exports,
                          const Teuchos::ArrayView<const size_t>& numExportPacketsPerLID,
                          const Teuchos::ArrayView<Packet> &imports,
                          const Teuchos::ArrayView<const size_t>& numImportPacketsPerLID);

  /// \brief Execute the (forward) communication plan.
  ///
  /// Call this version of the method when you have the same number
  /// of Packets for each LID (local ID) to send or receive.
  ///
  /// \tparam Packet The type of data to send and receive.
  ///
  /// \param exports [in] Contains the values to be sent by this
  ///   process.  On exit from this method, it's OK to modify the
  ///   entries of this buffer.
  ///
  /// \param numPackets [in] The number of Packets per export /
  ///   import.  This version of the routine assumes that each LID
  ///   has the same number of Packets associated with it.  (\c
  ///   MultiVector is an example of a DistObject subclass
  ///   satisfying this property.)
  ///
  /// \param imports [out] On entry, buffer must be large enough to
  ///   accomodate the data exported (sent) to us.  On exit,
  ///   contains the values exported to us.
  template <class ExpView, class ImpView>
  typename std::enable_if<(Kokkos::Impl::is_view<ExpView>::value && Kokkos::Impl::is_view<ImpView>::value)>::type
  doPostsAndWaits (const ExpView& exports,
                   const size_t numPackets,
                   const ImpView& imports);

  /// \brief Execute the (forward) communication plan.
  ///
  /// Call this version of the method when you have possibly
  /// different numbers of Packets for each LID (local ID) to send
  /// or receive.
  ///
  /// \tparam Packet The type of data to send and receive.
  ///
  /// \param exports [in] Contains the values to be sent by this
  ///   process.  On exit from this method, it's OK to modify the
  ///   entries of this buffer.
  ///
  /// \param numExportPacketsPerLID [in] The number of packets for
  ///   each export LID (i.e., each LID to be sent).
  ///
  /// \param imports [out] On entry, buffer must be large enough to
  ///   accomodate the data exported (sent) to us.  On exit,
  ///   contains the values exported to us.
  ///
  /// \param numImportPacketsPerLID [in] The number of packets for
  ///   each import LID (i.e., each LID to be received).
  template <class ExpView, class ImpView>
  typename std::enable_if<(Kokkos::Impl::is_view<ExpView>::value && Kokkos::Impl::is_view<ImpView>::value)>::type
  doPostsAndWaits (const ExpView &exports,
                   const Teuchos::ArrayView<const size_t>& numExportPacketsPerLID,
                   const ImpView &imports,
                   const Teuchos::ArrayView<const size_t>& numImportPacketsPerLID);

private:
  Teuchos::RCP< ::Tpetra::Distributor> distributor_;
  size_t numSends_ = 0; // original number on createToSends
  std::shared_ptr<size_t[]> sendPerm_; // permute sorted to unsorted
  std::shared_ptr<size_t[]> sendInvPerm_; // permute unsorted to sorted
  size_t numRecvs_ = 0; // only with reverse ?
  std::shared_ptr<size_t[]> recvPerm_;
  std::shared_ptr<size_t[]> recvInvPerm_;
  std::unique_ptr<SortingDistributor> reverseDistributor_;
};

template <class OrdinalType>
SortingDistributor::
SortingDistributor (Teuchos::Array<OrdinalType>& exportIDs,
                    Teuchos::Array<int>& exportPIDs,
                    const Teuchos::RCP<const Teuchos::Comm<int> >& comm,
                    const Teuchos::ArrayView<const OrdinalType>& remoteIDs,
                    const Teuchos::ArrayView<const int>& remotePIDs) :
  distributor_ (std::unique_ptr< ::Tpetra::Distributor> (new ::Tpetra::Distributor (comm)))
{
  distributor_->template createFromRecvs<OrdinalType> (remoteIDs, remotePIDs, exportIDs, exportPIDs);
}

template <class Packet>
void
SortedDistributor::
doPostsAndWaits (const Teuchos::ArrayView<const Packet>& exports,
                 const size_t numPackets,
                 const Teuchos::ArrayView<Packet>& imports)
{
  const bool must_unsort_receives = (numRecvs_ != 0);
  std::unique_ptr<Packet[]> sortedImports;
  Teuchos::ArrayView<Packet> sortedImports_av;
  if (must_unsort_receives) {
    const size_t numImports = static_cast<size_t> (imports.size ());
    TEUCHOS_ASSERT( numPackets * numRecvs_ == numImports );
    sortedImports = std::unique_ptr<Packet[]> (new Packet[numPackets * numRecvs_]);
    sortedImports_av = Teuchos::ArrayView<Packet> (sortedImports.get (), numPackets * numRecvs_);
  }
  else {
    sortedImports_av = imports;
  }

  if (numSends_ == 0) {
    distributor_->template doPostsAndWaits<Packet> (exports, numPackets, sortedImports_av);
  }
  else {
    const size_t numExports = static_cast<size_t> (exports.size ());
    TEUCHOS_ASSERT( numPackets * numSends_ == numExports );
    Teuchos::Array<Packet> sortedExports (numExports);

    // Permute from unsorted to sorted, i.e., apply inverse permutation.
    for (size_t k = 0; k < numSends_; ++k) {
      const size_t sortedStart = sendInvPerm_[k];
      const size_t unsortedStart = k;
      for (size_t packetInd = 0; packetInd < numPackets; ++packetInd) {
        sortedExports[sortedStart + packetInd] = exports[unsortedStart + packedInd];
      }
    }
    distributor_->template doPostsAndWaits (sortedExports, numPackets, sortedImports_av);
  }

  if (must_unsort_receives) {
    // Permute from sorted to unsorted, i.e., apply forward permutation.
    for (size_t k = 0; k < numRecvs_; ++k) {
      const size_t sortedStart = k;
      const size_t unsortedStart = recvPerm_[k];
      for (size_t packetInd = 0; packetInd < numPackets; ++packetInd) {
        imports[unsortedStart + packetInd] = sortedImports[unsortedStart + packedInd];
      }
    }
  }
}

template <class Packet>
void
SortedDistributor::
doPostsAndWaits (const Teuchos::ArrayView<const Packet>& exports,
                 const Teuchos::ArrayView<const size_t>& numExportPacketsPerLID,
                 const Teuchos::ArrayView<Packet>& imports,
                 const Teuchos::ArrayView<const size_t>& numImportPacketsPerLID)
{
  TEUCHOS_ASSERT( false );
}

template <class Packet>
void
SortedDistributor::
doReversePostsAndWaits (const Teuchos::ArrayView<const Packet>& exports,
                        const size_t numPackets,
                        const Teuchos::ArrayView<Packet>& imports)
{
  if (reverseDistributor_.get () != nullptr) {
    reverseDistributor_ = std::unique_ptr<SortingDistributor> (new SortingDistributor (*this, reverse_tag {}));
  }
  reverseDistributor_->template doPostsAndWaits (exports, numPackets, imports);
}

template <class Packet>
void
SortedDistributor::
doReversePostsAndWaits (const Teuchos::ArrayView<const Packet>& exports,
                        const Teuchos::ArrayView<const size_t>& numExportPacketsPerLID,
                        const Teuchos::ArrayView<Packet>& imports,
                        const Teuchos::ArrayView<const size_t>& numImportPacketsPerLID)
{
  if (reverseDistributor_.get () != nullptr) {
    reverseDistributor_ = std::unique_ptr<SortingDistributor> (new SortingDistributor (*this, reverse_tag {}));
  }
  reverseDistributor_->template doPostsAndWaits (exports, numExportPacketsPerLID, imports, numImportPacketsPerLID);
}

template <class ExpView, class ImpView>
typename std::enable_if<(Kokkos::Impl::is_view<ExpView>::value && Kokkos::Impl::is_view<ImpView>::value)>::type
SortedDistributor::
doPostsAndWaits (const ExpView& exports,
                 const size_t numPackets,
                 const ImpView& imports)
{
  TEUCHOS_ASSERT( false );
}

template <class ExpView, class ImpView>
typename std::enable_if<(Kokkos::Impl::is_view<ExpView>::value && Kokkos::Impl::is_view<ImpView>::value)>::type
SortedDistributor::
doPostsAndWaits (const ExpView& exports,
                 const Teuchos::ArrayView<const size_t>& numExportPacketsPerLID,
                 const ImpView& imports,
                 const Teuchos::ArrayView<const size_t>& numImportPacketsPerLID)
{
  TEUCHOS_ASSERT( false );
}

template <class ExpView, class ImpView>
typename std::enable_if<(Kokkos::Impl::is_view<ExpView>::value && Kokkos::Impl::is_view<ImpView>::value)>::type
SortedDistributor::
doReversePostsAndWaits (const ExpView& exports,
                        const size_t numPackets,
                        const ImpView& imports)
{
  if (reverseDistributor_.get () != nullptr) {
    reverseDistributor_ = std::unique_ptr<SortingDistributor> (new SortingDistributor (*this, reverse_tag {}));
  }
  reverseDistributor_->template doPostsAndWaits (exports, numPackets, imports);
}

template <class ExpView, class ImpView>
typename std::enable_if<(Kokkos::Impl::is_view<ExpView>::value && Kokkos::Impl::is_view<ImpView>::value)>::type
SortedDistributor::
doReversePostsAndWaits (const ExpView& exports,
                        const Teuchos::ArrayView<const size_t>& numExportPacketsPerLID,
                        const ImpView& imports,
                        const Teuchos::ArrayView<const size_t>& numImportPacketsPerLID)
{
  if (reverseDistributor_.get () != nullptr) {
    reverseDistributor_ = std::unique_ptr<SortingDistributor> (new SortingDistributor (*this, reverse_tag {}));
  }
  reverseDistributor_->template doPostsAndWaits (exports, numExportPacketsPerLID, imports, numImportPacketsPerLID);
}

} // namespace Details
} // namespace Tpetra

#endif // TPETRA_DETAILS_SORTINGDISTRIBUTOR_HPP
