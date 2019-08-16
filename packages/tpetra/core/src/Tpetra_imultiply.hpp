#include "Tpetra_MultiVector.hpp"
#include "Tpetra_iallreduce.hpp"
#include "Tpetra_Details_Behavior.hpp"
#include "Tpetra_Details_Profiling.hpp"

// Compute A^T * B (for real numbers) or A^H * B (for complex
// numbers), where A and B are both distributed over the same Map.
template <class ST, class LO, class GO, class NT>
std::shared_ptr< ::Tpetra::Details::CommRequest>
imultidot
(Kokkos::View<
   typename MultiVector<ST, LO, GO, NT>::dot_type,
   typename MultiVector<ST, LO, GO, NT>::device_type> out,
 MultiVector<ST, LO, GO, NT>& A,
 MultiVector<ST, LO, GO, NT>& B)
{
  using ::Tpetra::Details::ProfilingRegion;
  using Teuchos::RCP;
  using Teuchos::rcp;
  using Teuchos::rcpFromRef;
  using std::endl;
  using MV = MultiVector<ST, LO, GO, NT>;
  using IST = typename MV::impl_scalar_type;
  using ATS = Kokkos::ArithTraits<impl_scalar_type>;
  using STS = Teuchos::ScalarTraits<ST>;
  const char tfecfFuncName[] = "imultidot: ";
  ProfilingRegion region ("Tpetra::imultidot");

  TEUCHOS_TEST_FOR_EXCEPTION
    (! A.isConstantStride () || ! B.isConstantStride (),
     std::invalid_argument, "Tpetra::imultidot: Both A and B "
     "must have constant stride.")

  // In debug mode, check compatibility of local dimensions.  We only
  // do this in debug mode, since it requires an all-reduce to ensure
  // correctness on all processses.  It's entirely possible that only
  // some processes may have incompatible local dimensions.  Throwing
  // an exception only on those processes could cause this method to
  // hang.
  const bool debug = ::Tpetra::Details::Behavior::debug ();
  if (debug) {
    using std::endl;

    const bool mapsSame = A.getMap ()->isSameAs (* (B.getMap ()));
    int lclGood = mapsSame ? 1 : 0;

    const size_t A_nrows = A.getLocalLength ();
    const size_t A_ncols = A.getNumVectors ();
    const size_t B_nrows = A.getLocalLength ();
    const size_t B_ncols = A.getNumVectors ();

    auto comm = myMap->getComm ();
    const int myRank = comm->getRank ();
    std::ostringstream errStrm;
    if (A_nrows != B_nrows) {
      lclGood = 0;
      errStrm << "Proc " << myRank << ": A.getLocalLength()="
              << A_nrows << " != B_nrows=" << B_nrows << endl;
    }
    if (A_ncols != size_t (out.extent (0))) {
      lclGood = 0;
      errStrm << "Proc " << myRank << ": A.getNumVectors()="
              << A_ncols << " != out.extent(0)=" << out.extent (0)
              << endl;
    }
    if (B_ncols != size_t (out.extent (1))) {
      lclGood = 0;
      errStrm << "Proc " << myRank << ": B.getNumVectors()="
              << B_ncols << " != out.extent(1)=" << out.extent (1)
              << endl;
    }

    using Teuchos::REDUCE_MIN;
    using Teuchos::reduceAll;
    using Teuchos::outArg;
    int gblGood = 0;
    reduceAll<int, int> (*comm, REDUCE_MIN, lclGood,
                         outArg (gblGood));
    if (gblGood != 1) {
      std::ostringstream os;
      ::Tpetra::Details::gathervPrint (os, errStrm.str (), *comm);

      TEUCHOS_TEST_FOR_EXCEPTION
        (true, std::runtime_error, "Tpetra::imultidot: Dimensions "
         "don't match on at least one process." << endl << "Global "
         "dimensions: out: " << out.extent (0) << " x " <<
         out.extent (1) << ", A: " << A.getGlobalLength () << " x "
         << A.getNumVectors () << ", B: " << B.getGlobalLength ()
         << " x " << B.getNumVectors () << endl << os.str ());
    }
  }

  // Case 2: C(local) = A^T(distr) * B  (distr)

  using const_lcl_mv_type = Kokkos::View<const IST**,
    Kokkos::LayoutLeft,
    typename multivec_type::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using Tpetra::readOnly;
  using Tpetra::withLocalAccess;
  {
    ProfilingRegion regionAccess ("Tpetra::imultidot: access");
    withLocalAccess
    ([&] (const const_lcl_mv_type& A_lcl,
          const const_lcl_mv_type& B_lcl) {
       ProfilingRegion regionGemm ("Tpetra::imultidot: gemm");
       const IST alpha = Kokkos::ArithTraits<IST>::one ();       
       const IST beta = Kokkos::ArithTraits<IST>::zero ();
       const bool isComplex = STS::isComplex;
       const char transA = isComplex ? "C" : "T";
       KokkosBlas::gemm (&transA, "N", alpha, A_lcl, B_lcl,
                         beta, out);
     },

    }

    if (! isConstantStride ()) {
      ::Tpetra::deep_copy (*this, *C_tmp); // Copy the result back into *this.
    }

    // Dispose of (possibly) extra copies of A and B.
    A_tmp = Teuchos::null;
    B_tmp = Teuchos::null;

    // If Case 2 then sum up *this and distribute it to all processes.
    if (Case2) {
      this->reduce ();
    }
  }
