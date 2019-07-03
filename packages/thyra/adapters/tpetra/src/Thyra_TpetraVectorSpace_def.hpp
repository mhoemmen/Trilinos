// @HEADER
// ***********************************************************************
//
//    Thyra: Interfaces and Support for Abstract Numerical Algorithms
//                 Copyright (2004) Sandia Corporation
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
// Questions? Contact Roscoe A. Bartlett (bartlettra@ornl.gov)
//
// ***********************************************************************
// @HEADER


#ifndef THYRA_TPETRA_VECTOR_SPACE_HPP
#define THYRA_TPETRA_VECTOR_SPACE_HPP


#include "Thyra_TpetraVectorSpace_decl.hpp"
#include "Thyra_TpetraThyraWrappers.hpp"
#include "Thyra_TpetraVector.hpp"
#include "Thyra_TpetraMultiVector.hpp"
#include "Thyra_TpetraEuclideanScalarProd.hpp"
#include "Tpetra_Details_StaticView.hpp"

namespace Thyra {


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::create()
{
  const RCP<this_t> vs(new this_t);
  vs->weakSelfPtr_ = vs.create_weak();
  return vs;
}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
void TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::initialize(
  const RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> > &tpetraMap
  )
{
  comm_ = convertTpetraToThyraComm(tpetraMap->getComm());
  tpetraMap_ = tpetraMap;
  this->updateState(tpetraMap->getGlobalNumElements(),
    !tpetraMap->isDistributed());
  this->setScalarProd(tpetraEuclideanScalarProd<Scalar,LocalOrdinal,GlobalOrdinal,Node>());
}


// Overridden from VectorSpace


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<VectorBase<Scalar> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createMember() const
{
  return tpetraVector<Scalar>(
    weakSelfPtr_.create_strong().getConst(),
    Teuchos::rcp(
      new Tpetra::Vector<Scalar,LocalOrdinal,GlobalOrdinal,Node>(tpetraMap_, false)
      )
    );
}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< MultiVectorBase<Scalar> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createMembers(int numMembers) const
{
  return tpetraMultiVector<Scalar>(
    weakSelfPtr_.create_strong().getConst(),
    tpetraVectorSpace<Scalar>(
      Tpetra::createLocalMapWithNode<LocalOrdinal, GlobalOrdinal, Node>(
        numMembers, tpetraMap_->getComm()
        )
      ),
    Teuchos::rcp(
      new Tpetra::MultiVector<Scalar,LocalOrdinal,GlobalOrdinal,Node>(
        tpetraMap_, numMembers, false)
      )
    );
  // ToDo: Create wrapper function to create locally replicated vector space
  // and use it.
}


namespace { // (anonymous)

// FIXME (mfh 03 Jul 2019) It's not clear to me why this depends on Tpetra.
// Also, there might already be some generic way to do this in Thyra.
template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
class CopyMultiVectorViewBack {
private:
  using TMV = TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>;

public:
  CopyMultiVectorViewBack(MultiVectorBase<Scalar>& mv,
    const RTOpPack::SubMultiVectorView<Scalar>& raw_mv)
    : mv_(dynamic_cast<TMV&>(mv)), raw_mv_(raw_mv)
  {}

  ~CopyMultiVectorViewBack() {
    RTOpPack::ConstSubMultiVectorView<Scalar> smv;
    mv_.acquireDetachedView(Range1D(), Range1D(), &smv);
    RTOpPack::assign_entries<Scalar>(Teuchos::outArg(raw_mv_), smv);
    mv_.releaseDetachedView(&smv);
  }

private:
  TpetraMultiVector<Scalar, LocalOrdinal, GlobalOrdinal, Node>& mv_;
  const RTOpPack::SubMultiVectorView<Scalar> raw_mv_;
};

} // namespace (anonymous)

template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< MultiVectorBase<Scalar> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createMembersView(
  const RTOpPack::SubMultiVectorView<Scalar> &raw_mv ) const
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_TEST_FOR_EXCEPT( raw_mv.subDim() != this->dim() );
#endif
  using LO = LocalOrdinal;
  using GO = GlobalOrdinal;
  using MV = Tpetra::MultiVector<Scalar, LO, GO, Node>;

  RCP<MultiVectorBase<Scalar>> mv;
  RCP<MV> tpetraMV;

  if (! tpetraMap_->isDistributed ()) {
    using Tpetra::Details::getStatic2dDualView;
    using IST = typename MV::impl_scalar_type;
    using DT = typename MV::device_type;

    // TODO Check whether the cached static storage is currently in
    // use by another non-distributed Tpetra::MultiVector.
    auto dv = getStatic2dDualView<IST, DT> (tpetraMap_->getGlobalNumElements(), raw_mv.numSubCols());
    tpetraMV = Teuchos::rcp (new MV (tpetraMap_, dv));

    if (tpetraDomainSpace_.is_null() || raw_mv.numSubCols() != tpetraDomainSpace_->localSubDim()) {
      using Tpetra::createLocalMapWithNode;
      auto lclMap = createLocalMapWithNode<LO, GO, Node>(raw_mv.numSubCols(), tpetraMap_->getComm());
      tpetraDomainSpace_ = tpetraVectorSpace<Scalar>(lclMap);
    }
    mv = tpetraMultiVector<Scalar>(weakSelfPtr_.create_strong().getConst(), tpetraDomainSpace_, tpetraMV);
  }
  else {
    using Teuchos::rcp_dynamic_cast;
    using TMV = TpetraMultiVector<Scalar, LO, GO, Node>;

    mv = this->createMembers(raw_mv.numSubCols());
    tpetraMV = rcp_dynamic_cast<TMV>(mv, true)->getTpetraMultiVector();
  }
  // Copy initial values in raw_mv into multi-vector
  RTOpPack::SubMultiVectorView<Scalar> smv;
  mv->acquireDetachedView(Range1D(),Range1D(),&smv);
  RTOpPack::assign_entries<Scalar>(
    Ptr<const RTOpPack::SubMultiVectorView<Scalar> >(Teuchos::outArg(smv)),
    raw_mv
    );
  mv->commitDetachedView(&smv);
  // Just before MultiVector is destroyed, copy "view"'s data back.
  Teuchos::set_extra_data(
    Teuchos::rcp(new CopyMultiVectorViewBack<Scalar, LO, GO, Node>(*mv, raw_mv)),
    "CopyMultiVectorViewBack",
    Teuchos::outArg(mv),
    Teuchos::PRE_DESTROY
    );
  return mv;
}


template<class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const MultiVectorBase<Scalar> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::createMembersView(
  const RTOpPack::ConstSubMultiVectorView<Scalar> &raw_mv ) const
{
#ifdef TEUCHOS_DEBUG
  TEUCHOS_TEST_FOR_EXCEPT( raw_mv.subDim() != this->dim() );
#endif
  using LO = LocalOrdinal;
  using GO = GlobalOrdinal;
  using MV = Tpetra::MultiVector<Scalar, LO, GO, Node>;

  // Create a multi-vector
  RCP< MultiVectorBase<Scalar> > mv;
  RCP<MV> tpetraMV;

  if (! tpetraMap_->isDistributed()) {
    using Tpetra::Details::getStatic2dDualView;
    using IST = typename MV::impl_scalar_type;
    using DT = typename MV::device_type;

    auto dv = getStatic2dDualView<IST, DT> (tpetraMap_->getGlobalNumElements(),
                                            raw_mv.numSubCols());
    tpetraMV = Teuchos::rcp (new MV (tpetraMap_, dv));

    if (tpetraDomainSpace_.is_null() ||
        raw_mv.numSubCols() != tpetraDomainSpace_->localSubDim()) {
      using Tpetra::createLocalMapWithNode;
      auto lclMap = createLocalMapWithNode<LO, GO, Node>(raw_mv.numSubCols(),
                                                         tpetraMap_->getComm());
      tpetraDomainSpace_ = tpetraVectorSpace<Scalar>(lclMap);
    }
    mv = tpetraMultiVector<Scalar>(weakSelfPtr_.create_strong().getConst(),
                                   tpetraDomainSpace_, tpetraMV);
  }
  else {
    using Teuchos::rcp_dynamic_cast;
    using TMV = TpetraMultiVector<Scalar, LO, GO, Node>;

    mv = this->createMembers(raw_mv.numSubCols());
    tpetraMV = rcp_dynamic_cast<TMV> (mv, true)->getTpetraMultiVector();
  }
  // Copy values in raw_mv into multi-vector
  RTOpPack::SubMultiVectorView<Scalar> smv;
  mv->acquireDetachedView(Range1D(),Range1D(),&smv);
  RTOpPack::assign_entries<Scalar>(
    Ptr<const RTOpPack::SubMultiVectorView<Scalar> >(Teuchos::outArg(smv)),
    raw_mv );
  mv->commitDetachedView(&smv);
  return mv;
}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
bool TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::hasInCoreView(
  const Range1D& rng_in, const EViewType viewType, const EStrideType strideType
  ) const
{
  const Range1D rng = full_range(rng_in,0,this->dim()-1);
  const Ordinal l_localOffset = this->localOffset();

  const Ordinal myLocalSubDim = tpetraMap_.is_null () ?
    static_cast<Ordinal> (0) : tpetraMap_->getNodeNumElements ();

  return ( l_localOffset<=rng.lbound() && rng.ubound()<l_localOffset+myLocalSubDim );
}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP< const VectorSpaceBase<Scalar> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::clone() const
{
  return tpetraVectorSpace<Scalar>(tpetraMap_);
}

template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Tpetra::Map<LocalOrdinal,GlobalOrdinal,Node> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getTpetraMap() const
{
  return tpetraMap_;
}

// Overridden from SpmdVectorSpaceDefaultBase


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
RCP<const Teuchos::Comm<Ordinal> >
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::getComm() const
{
  return comm_;
}


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
Ordinal TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::localSubDim() const
{
  return tpetraMap_.is_null () ? static_cast<Ordinal> (0) :
    static_cast<Ordinal> (tpetraMap_->getNodeNumElements ());
}

// private


template <class Scalar, class LocalOrdinal, class GlobalOrdinal, class Node>
TpetraVectorSpace<Scalar,LocalOrdinal,GlobalOrdinal,Node>::TpetraVectorSpace()
{
  // The base classes should automatically default initialize to a safe
  // uninitialized state.
}


} // end namespace Thyra


#endif // THYRA_TPETRA_VECTOR_SPACE_HPP
