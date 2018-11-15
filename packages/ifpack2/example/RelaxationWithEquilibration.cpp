#include "Ifpack2_Factory.hpp"
#include "Ifpack2_Details_CanChangeMatrix.hpp"
#include "BelosTpetraAdapter.hpp"
#include "BelosSolverFactory.hpp"
#include "MatrixMarket_Tpetra.hpp"
#include "Tpetra_ComputeGatherMap.hpp"
#include "Tpetra_computeRowAndColumnOneNorms.hpp"
#include "Tpetra_CrsMatrix.hpp"
#include "Tpetra_MultiVector.hpp"
#include "Tpetra_Core.hpp"
#include "Tpetra_leftAndOrRightScaleCrsMatrix.hpp"
#include "Teuchos_CommandLineProcessor.hpp"
#include "Teuchos_CommHelpers.hpp"
#include "Teuchos_ParameterList.hpp"
#include "Teuchos_LAPACK.hpp"
//#include "Teuchos_ParameterXMLFileReader.hpp"
#include "KokkosBlas1_abs.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosBlas3_gemm.hpp"

#include <algorithm> // std::transform
#include <cctype> // std::toupper
#include <sstream>
#include <functional>

namespace { // (anonymous)

template<class MV>
auto norm (const MV& X) -> decltype (X.getVector (0)->norm2 ())
{
  return X.getVector (0)->norm2 ();
}

template<class MV, class OP>
void residual (MV& R, const MV& B, const OP& A, const MV& X)
{
  using STS = Teuchos::ScalarTraits<typename MV::scalar_type>;
  
  Tpetra::deep_copy (R, B);
  A.apply (X, R, Teuchos::NO_TRANS, -STS::one (), STS::one ());
}

template<class MV>
typename MV::dot_type dot (const MV& X, const MV& Y)
{
  Teuchos::Array<typename MV::dot_type> dots (X.getNumVectors ());
  Y.dot (X, dots ());
  return dots[0];
}

template<class MV, class OP>
std::tuple<typename MV::mag_type, int, bool>
bicgstab (MV& x,
	  const OP& A,
	  const OP* const M,
	  const MV& b,
	  const int max_it,
	  const typename MV::mag_type tol)
{
  using STS = Teuchos::ScalarTraits<typename MV::scalar_type>;
  using STM = Teuchos::ScalarTraits<typename MV::mag_type>;
  using dot_type = typename MV::dot_type;
  using mag_type = typename MV::mag_type;  
  
  int iter = 0;

  mag_type bnrm2 = norm (b);
  if (bnrm2 == STM::zero ()) {
    x.putScalar (STS::zero ());
    bnrm2 = STM::one ();
    return {bnrm2, 0, true};
  }

  MV r (b.getMap (), b.getNumVectors ());

  residual (r, b, A, x); // r = b - A*x;
  mag_type error = norm (r) / bnrm2;
  if (error < tol) {
    return {error, 0, true};
  }
  dot_type omega = STS::one ();
  dot_type alpha;
  dot_type rho;
  dot_type rho_1;  
  dot_type beta;
  mag_type resid;

  MV r_tld (r.getMap (), r.getNumVectors ());
  Tpetra::deep_copy (r_tld, r);

  MV p (r.getMap (), r.getNumVectors ());
  MV p_hat (x.getMap (), r.getNumVectors ());
  MV v (r.getMap (), r.getNumVectors ());
  MV s (r.getMap (), r.getNumVectors ());
  MV s_hat (r.getMap (), r.getNumVectors ());
  MV t (r.getMap (), r.getNumVectors ());    

  for (iter = 1; iter <= max_it; ++iter) {
    std::cerr << ">>> Hand-rolled BiCGSTAB: iter = " << (iter - 1) << std::endl;
    
    rho = dot (r_tld, r); // ( r_tld'*r );
    std::cerr << "  rho = " << rho << std::endl;

    if (rho == 0.0) {
      return {error, iter, false};
    }

    if (iter > 1) {
      beta = (rho / rho_1) * (alpha / omega);
      std::cerr << "  beta = " << beta << std::endl;

      p.update (-omega, v, 0.0); // p = p - omega*v
      p.update (1.0, r, beta, p, 0.0); // p = r + beta*( p - omega*v );
    }
    else {
      p = r;
    }
    
#if 0
    if (iter > 1) {
      beta = (rho / rho_1) * (alpha / omega);
      std::cerr << "  beta = " << beta << std::endl;

      p.update (-omega, v, 1.0); // p = p - omega*v
      p.update (1.0, r, beta, p, 0.0); // p = r + beta*( p - omega*v );
    }
    else {
      Tpetra::deep_copy (p, r);
    }
#endif // 0

#if 0
    if (iter > 1) {
      beta = (rho / rho_1) * (alpha / omega);
      std::cerr << "  beta = " << beta << std::endl;
      if (false) { // NOTE (mfh 13 Nov 2018) weirdly, this branch works
	//p.update (-omega, v, 0.0); // p = p - omega*v
	p.update (-omega, v, 0.0); // p = p - omega*v
	p.update (1.0, r, beta, p, 0.0); // p = r + beta*( p - omega*v );
      }
      else if (false) { // NOTE (mfh 13 Nov 2018) does NOT work
	MV p_tmp (p, Teuchos::Copy);
	p_tmp.update (-omega, v, 0.0);
	p.update (1.0, r, beta, p_tmp, 0.0);
      }
      else if (false) { // NOTE (mfh 13 Nov 2018) does NOT work
	MV p_tmp (p, Teuchos::Copy);
	p_tmp.update (-omega, v, 1.0);
	p.update (1.0, r, beta, p_tmp, 0.0);
      }
      else { // NOTE (mfh 13 Nov 2018) does NOT work
	MV p_tmp (p, Teuchos::Copy);
	p_tmp.update (-omega, v, 0.0);
	p.putScalar (0.0);
	p.update (1.0, r, beta, p_tmp, 0.0);
      }
    }
    else {
      p = r; // oddly enough, this is actually what we want when there
	     // is no preconditioner.  If we do deep_copy instead,
	     // then the solver fails to converge.  I'm not sure how
	     // this works, given the formulae above.
      //Tpetra::deep_copy (p, r);
    }
#endif // 0

    std::cerr << "  ||P_0||_2 = " << norm (p) << std::endl;

    if (M != nullptr) {
      M->apply (p, p_hat);
    }
    else {
      Tpetra::deep_copy (p_hat, p);
    }

    A.apply (p_hat, v);

    std::cerr << "  ||Y_0||_2 = " << norm (p_hat) << std::endl;
    std::cerr << "  ||V_0||_2 = " << norm (v) << std::endl;
    std::cerr << "  ||Rhat_0||_2 = " << norm (r_tld) << std::endl;        
    std::cerr << "  rhatV = " << dot (r_tld, v) << std::endl;        

    alpha = rho / dot (r_tld, v);
    std::cerr << "  alpha = " << alpha << std::endl;    
    s.update (1.0, r, -alpha, v, 0.0); // s = r - alpha*v;
    if (norm(s) < tol) {
      // x = x + alpha*p_hat;
      resid = norm (s) / bnrm2;
      break;
    }

    if (M != nullptr) {
      M->apply (s, s_hat);
    }
    else {
      Tpetra::deep_copy (s_hat, s);
    }

    A.apply (s_hat, t);
    omega = dot (t, s) / dot (t, t);
    std::cerr << "  omega = " << omega << std::endl;

    // x = x + alpha*p_hat + omega*s_hat;
    x.update (alpha, p_hat, omega, s_hat, 1.0); 
    r.update (1.0, s, -omega, t, 0.0); // r = s - omega*t

    error = norm (r) / bnrm2;
    if (error <= tol) {
      break;
    }
    if (omega == 0.0) {
      break;
    }
    rho_1 = rho;
  }

  if (error <= tol) {
    return {error, iter, true};
  }
  else {
    return {error, iter, false};
  }
}

template<class MV>
bool
AZ_breakdown_f (MV& v, MV& w, const typename MV::dot_type v_dot_w)
{
  using STS = Teuchos::ScalarTraits<typename MV::scalar_type>;
  
  const auto v_norm = norm (v);
  const auto w_norm = norm (w);
  return STS::magnitude (v_dot_w) <= 100.0 * v_norm * w_norm * STS::eps ();
}

template<class MV, class OP>
std::tuple<typename MV::mag_type, int, bool>
bicgstab_aztecoo (MV& x,
		  const OP& A,
		  const OP* const M,
		  const MV& b,
		  const int max_it,
		  const typename MV::mag_type tol)
{
  using STS = Teuchos::ScalarTraits<typename MV::scalar_type>;
  using STM = Teuchos::ScalarTraits<typename MV::mag_type>;
  using dot_type = typename MV::dot_type;
  using mag_type = typename MV::mag_type;  
  
  int iter = 0;

  bool brkdown_will_occur = false;
  dot_type alpha = STS::one ();
  dot_type beta = STS::zero ();
  mag_type true_scaled_r = STM::zero ();
  dot_type omega = STS::one ();
  dot_type rhonm1 = STS::one ();
  dot_type rhon = STS::zero ();
  dot_type sigma = STS::zero ();
  mag_type brkdown_tol = STS::eps ();
  mag_type scaled_r_norm = -STM::one ();
  mag_type actual_residual = -STM::one ();
  mag_type rec_residual = -STM::one ();
  dot_type dtemp = STS::zero ();

  MV phat (b.getMap (), b.getNumVectors (), false);  
  MV p (b.getMap (), b.getNumVectors (), false);
  MV shat (b.getMap (), b.getNumVectors (), false);
  MV s (b.getMap (), b.getNumVectors (), false);
  MV r (b.getMap (), b.getNumVectors (), false);
  MV r_tld (b.getMap (), b.getNumVectors (), false);
  MV v (b.getMap (), b.getNumVectors (), false);

  residual (r, b, A, x); // r = b - A*x;

  // "v, p <- 0"
  v.putScalar (STS::zero ());
  p.putScalar (STS::zero ());

  // "set rtilda" [sic]
  constexpr int AZ_aux_vec = 0; // AZ_resid = 0; AZ_rand (?) = 1
  constexpr int AZ_resid = 0;
  if (AZ_aux_vec == AZ_resid) {
    Tpetra::deep_copy (r_tld, r);
  }
  else {
    r_tld.randomize ();
  }

  // AZ_compute_global_scalars does all this, neatly bundled into a
  // single all-reduce.
  const mag_type b_norm = norm (b);
  actual_residual = norm (r);  
  rec_residual = actual_residual;
  scaled_r_norm = rec_residual / b_norm;
  rhon = dot (r_tld, r);
  if (scaled_r_norm <= tol) {
    return {scaled_r_norm, 0, true};
  }
  
  for (iter = 1; iter <= max_it; ++iter) {
    std::cerr << ">>> AztecOO-ish BiCGSTAB: iter = " << (iter - 1) << std::endl;
    if (brkdown_will_occur) {
      residual (v, b, A, x); // v = b - A*x
      actual_residual = norm (v);
      scaled_r_norm = actual_residual / b_norm;
      std::cerr << "Uh oh, breakdown" << std::endl;
      return {scaled_r_norm, iter, false};
    }
    
    beta = (rhon / rhonm1) * (alpha / omega);

    if (STS::magnitude (rhon) < brkdown_tol) {
      if (AZ_breakdown_f(r, r_tld, rhon)) {
	brkdown_will_occur = true;
      }
      else {
	brkdown_tol = 0.1 * STS::magnitude (rhon);
      }
    }      

    rhonm1 = rhon;

    /* p    = r + beta*(p - omega*v)       */
    /* phat = M^-1 p                       */
    /* v    = A phat                       */
    
    dtemp = beta * omega;
    p.update (STS::one (), r, -dtemp, v, beta);
    Tpetra::deep_copy (phat, p);

    if (M != nullptr) {
      M->apply (p, phat);
    }
    A.apply (phat, v);
    sigma = dot (r_tld, v);

    if (STS::magnitude (sigma) < brkdown_tol) {
      if (AZ_breakdown_f(r_tld, v, sigma)) { // actual break down
	residual (v, b, A, x); // v = b - A*x;
	actual_residual = norm (v);
	scaled_r_norm = actual_residual / b_norm;
	return {scaled_r_norm, iter, false};
      }
      else {
	brkdown_tol = 0.1 * STS::magnitude (sigma);
      }
    }

    alpha = rhon / sigma;

    s.update (STS::one (), r, -alpha, v, STS::zero ());
    Tpetra::deep_copy (shat, s);

    if (M != nullptr) {
      M->apply (s, shat);
    }
    A.apply (shat, r);

    /* omega = (t,s)/(t,t) with r = t */

    const auto dot_vec_0 = dot (r, s);
    const auto dot_vec_1 = dot (r, r);
    if (STM::magnitude (dot_vec_1) < tol) {
      omega = STS::zero ();
      brkdown_will_occur = true;
    }
    else {
      omega = dot_vec_0 / dot_vec_1;
    }

    /* x = x + alpha*phat + omega*shat */
    /* r = s - omega*r */

    // DAXPY_F77(&N, &alpha, phat, &one, x, &one);
    // DAXPY_F77(&N, &omega, shat, &one, x, &one);
    x.update (alpha, phat, omega, shat, STS::one ());
    
    // for (i = 0; i < N; i++) r[i] = s[i] - omega * r[i];
    r.update (STS::one (), s, -omega);

    rec_residual = norm (r);
    scaled_r_norm = rec_residual / b_norm;
    rhon = dot (r, r_tld);
    if (scaled_r_norm <= tol) {
      return {scaled_r_norm, 0, true};
    }
  }

  return {scaled_r_norm, iter, scaled_r_norm <= tol};
}

template<class SC, class LO, class GO, class NT>
std::pair<Teuchos::RCP<Tpetra::CrsMatrix<SC, LO, GO, NT> >,
	  Teuchos::RCP<Tpetra::MultiVector<SC, LO, GO, NT> > >
gatherCrsMatrixAndMultiVector (LO& errCode,
			       const Tpetra::CrsMatrix<SC, LO, GO, NT>& A,
			       const Tpetra::MultiVector<SC, LO, GO, NT>& B)
{
  using Tpetra::Details::computeGatherMap;
  using crs_matrix_type = Tpetra::CrsMatrix<SC, LO, GO, NT>;
  using export_type = Tpetra::Export<LO, GO, NT>;
  using mv_type = Tpetra::MultiVector<SC, LO, GO, NT>;
  
  auto rowMap_gathered = computeGatherMap (A.getRowMap (), Teuchos::null);
  export_type exp (A.getRowMap (), rowMap_gathered);
  auto A_gathered =
    Teuchos::rcp (new crs_matrix_type (rowMap_gathered,
				       A.getGlobalMaxNumRowEntries (),
				       Tpetra::StaticProfile));
  A_gathered->doExport (A, exp, Tpetra::INSERT);
  auto domainMap_gathered = computeGatherMap (A.getDomainMap (), Teuchos::null);
  auto rangeMap_gathered = computeGatherMap (A.getRangeMap (), Teuchos::null);        
  A_gathered->fillComplete (domainMap_gathered, rangeMap_gathered);

  auto B_gathered =
    Teuchos::rcp (new mv_type (rangeMap_gathered, B.getNumVectors ()));
  B_gathered->doExport (B, exp, Tpetra::ADD);

  return std::make_pair (A_gathered, B_gathered);
}
  
using host_device_type = Kokkos::Device<Kokkos::DefaultHostExecutionSpace, Kokkos::HostSpace>;

template<class SC = Tpetra::MultiVector<>::scalar_type,
	 class LO = Tpetra::MultiVector<>::local_ordinal_type,
	 class GO = Tpetra::MultiVector<>::global_ordinal_type,
	 class NT = Tpetra::MultiVector<>::node_type>
using HostDenseMatrix =
  Kokkos::View<typename Tpetra::MultiVector<SC, LO, GO, NT>::impl_scalar_type**,
	       Kokkos::LayoutLeft,
	       host_device_type>;

template<class SC, class LO, class GO, class NT>
HostDenseMatrix<SC, LO, GO, NT>
densifyGatheredCrsMatrix (LO& errCode,
			  const Tpetra::CrsMatrix<SC, LO, GO, NT>& A,
			  const std::string& label)
{
  const LO numRows = LO (A.getRangeMap ()->getNodeNumElements ());
  const LO numCols = LO (A.getDomainMap ()->getNodeNumElements ());
  
  using dense_matrix_type = HostDenseMatrix<SC, LO, GO, NT>;
  dense_matrix_type A_dense (label, numRows, numCols);

  for (LO lclRow = 0; lclRow < numRows; ++lclRow) {
    LO numEnt = 0;	  
    const LO* lclColInds = nullptr;
    const SC* vals = nullptr;
    const LO curErrCode = A.getLocalRowView (lclRow, numEnt, vals, lclColInds);
    if (errCode != 0) {
      errCode = curErrCode;
    }
    else {
      for (LO k = 0; k < numEnt; ++k) {
	const LO lclCol = lclColInds[k];
	using impl_scalar_type =
	  typename Tpetra::CrsMatrix<SC, LO, GO, NT>::impl_scalar_type;
	A_dense(lclRow, lclCol) += impl_scalar_type (vals[k]);
      }
    }
  }

  return A_dense;
}

template<class SC, class LO, class GO, class NT>
HostDenseMatrix<SC, LO, GO, NT>
copyGatheredMultiVector (Tpetra::MultiVector<SC, LO, GO, NT>& X,
			 const std::string& label)
{
  using dense_matrix_type = HostDenseMatrix<SC, LO, GO, NT>;
  using dev_memory_space = typename Tpetra::MultiVector<SC, LO, GO, NT>::device_type::memory_space;

  X.template sync<Kokkos::HostSpace> ();
  auto X_lcl = X.template getLocalView<Kokkos::HostSpace> ();
  dense_matrix_type X_copy (label, X.getLocalLength (), X.getNumVectors ());
  Kokkos::deep_copy (X_copy, X_lcl);

  X.template sync<dev_memory_space> ();
  return X_copy;
}

template<class SC, class LO, class GO, class NT>
LO
gatherAndDensify (HostDenseMatrix<SC, LO, GO, NT>& A_dense,
		  HostDenseMatrix<SC, LO, GO, NT>& B_dense,
		  const Tpetra::CrsMatrix<SC, LO, GO, NT>& A,
		  Tpetra::MultiVector<SC, LO, GO, NT>& B)
{
  LO errCode = 0;
  auto A_and_B_gathered = gatherCrsMatrixAndMultiVector (errCode, A, B);

  if (errCode == 0) {
    A_dense = densifyGatheredCrsMatrix (errCode, * (A_and_B_gathered.first), "A_dense");
    B_dense = copyGatheredMultiVector (B, "B_dense");
  }
  return errCode;
}

template<class DenseMatrixType>
void
solveLeastSquaresProblemAndReport (DenseMatrixType A,
				   DenseMatrixType B,
				   const double RCOND = -1.0 /* negative means use machine precision */)
{
  const int numRows = int (A.extent (0));
  const int numCols = int (A.extent (1));
  const int NRHS = int (B.extent (1));

  DenseMatrixType A_copy ("A_copy", numRows, numCols);
  Kokkos::deep_copy (A_copy, A);
  
  DenseMatrixType B_copy ("B_copy", numRows, NRHS);
  Kokkos::deep_copy (B_copy, B);  
  //DenseMatrixType X ("X", numCols, NRHS);

  const int LDA = (numRows == 0) ? 1 : int (A_copy.stride (1));
  const int LDB = (numRows == 0) ? 1 : int (B_copy.stride (1));

  std::vector<double> S (std::min (numRows, numCols));
  int RANK = 0;
  int LWORK = -1; // workspace query
  int INFO = 0;
  Teuchos::LAPACK<int, double> lapack;

  using std::cerr;
  using std::cout;
  using std::endl;
  cout << "Solver:" << endl
       << "  Solver type: LAPACK's DGELSS" << endl;

  std::vector<double> WORK (1);
  lapack.GELSS (numRows, numCols, NRHS, A_copy.data (), LDA,
		B_copy.data (), LDB,
		S.data (), RCOND, &RANK, WORK.data (), LWORK, &INFO);
  if (INFO != 0) {
    cerr << "DGELSS returned INFO = " << INFO << " != 0." << endl;
    return;
  }
  LWORK = int (WORK[0]);
  if (LWORK < 0) {
    cerr << "DGELSS reported LWORK = " << LWORK << " < 0." << endl;
    return;
  }
  WORK.resize (LWORK);
  lapack.GELSS (numRows, numCols, NRHS, A_copy.data (), LDA,
		B_copy.data (), LDB,
		S.data (), RCOND, &RANK, WORK.data (), LWORK, &INFO);

  cout << "Results:" << endl
       << "  INFO: " << INFO << endl
       << "  RCOND: " << RCOND << endl
       << "  Singular values: ";
  for (double sigma : S) {
    cout << sigma << " ";
  }
  cout << endl;

  if (numRows == numCols) {
    std::vector<double> B_norms (NRHS);
    for (int k = 0; k < NRHS; ++k) {
      B_norms[k] =
	KokkosBlas::nrm2 (Kokkos::subview (B, Kokkos::ALL (), k));
    }
    
    auto X = B_copy;
    DenseMatrixType R ("R", numRows, NRHS);
    Kokkos::deep_copy (R, B);
    KokkosBlas::gemm ("N", "N", -1.0, A, X, +1.0, R);

    std::vector<double> explicitResidualNorms (NRHS);
    for (int k = 0; k < NRHS; ++k) {
      explicitResidualNorms[k] =
	KokkosBlas::nrm2 (Kokkos::subview (R, Kokkos::ALL (), k));
    }

    for (int j = 0; j < NRHS; ++j) {
      cout << "  For right-hand side " << j
	   << ": ||B-A*X||_2 = "
	   << explicitResidualNorms[j]
	   << ", ||B||_2 = " << B_norms[j] << endl;
    }
    cout << endl;
  }
}


template<class DenseMatrixType>
void
findEigenvaluesAndReport (DenseMatrixType A)
{
  using std::cerr;
  using std::cout;
  using std::endl;
  
  const int numRows = int (A.extent (0));
  const int numCols = int (A.extent (1));
  const int N = std::min (numRows, numCols);

  DenseMatrixType A_copy ("A_copy", numRows, numCols);
  Kokkos::deep_copy (A_copy, A);
  
  const int LDA = (numRows == 0) ? 1 : int (A_copy.stride (1));

  std::vector<double> realParts (N);
  std::vector<double> imagParts (N);
  std::vector<double> WORK (1);  

  int INFO = 0;
  int LWORK = -1; // workspace query
  Teuchos::LAPACK<int, double> lapack;  
  lapack.GEEV ('N', 'N', N, A_copy.data (), LDA, realParts.data (),
	       imagParts.data (), nullptr, 1, nullptr, 1, WORK.data (),
	       LWORK, nullptr, &INFO);
  if (INFO != 0) {
    cerr << "DGELSS returned INFO = " << INFO << " != 0." << endl;
    return;
  }
  LWORK = int (WORK[0]);
  if (LWORK < 0) {
    cerr << "DGEEV reported LWORK = " << LWORK << " < 0." << endl;
    return;
  }
  WORK.resize (LWORK);

  cout << "Solver:" << endl
       << "  Solver type: LAPACK's DGEEV" << endl;
  lapack.GEEV ('N', 'N', N, A_copy.data (), LDA, realParts.data (),
	       imagParts.data (), nullptr, 1, nullptr, 1, WORK.data (),
	       LWORK, nullptr, &INFO);

  cout << "Results:" << endl
       << "  INFO: " << INFO << endl
       << "  Eigenvalues: ";
  for (int k = 0; k < N; ++k) {
    cout << "(" << realParts[k] << "," << imagParts[k] << ")";
    if (k + 1 < N) {
      cout << ", ";
    }
  }
  cout << endl << endl;
}
  

template<class SC, class LO, class GO, class NT>
Teuchos::RCP<Tpetra::CrsMatrix<SC, LO, GO, NT> >
deepCopyFillCompleteCrsMatrix (const Tpetra::CrsMatrix<SC, LO, GO, NT>& A)
{
  using Teuchos::RCP;
  using crs_matrix_type = Tpetra::CrsMatrix<SC, LO, GO, NT>;

  TEUCHOS_TEST_FOR_EXCEPTION
    (! A.isFillComplete (), std::invalid_argument,
     "deepCopyFillCompleteCrsMatrix: Input matrix A must be fillComplete.");
  RCP<crs_matrix_type> A_copy (new crs_matrix_type (A.getCrsGraph ()));
  auto A_copy_lcl = A_copy->getLocalMatrix ();
  auto A_lcl = A.getLocalMatrix ();
  Kokkos::deep_copy (A_copy_lcl.values, A_lcl.values);
  A_copy->fillComplete (A.getDomainMap (), A.getRangeMap ());
  return A_copy;
}

template<class ViewType1,
         class ViewType2,
         class IndexType,
         const bool takeSquareRootsOfScalingFactors,
         const bool takeAbsoluteValueOfScalingFactors =
           ! std::is_same<
               typename Kokkos::ArithTraits<
                 typename ViewType1::non_const_value_type
               >::mag_type,
               typename ViewType2::non_const_value_type
             >::value,
         const int rank = ViewType1::Rank>
class ElementWiseMultiply {};

template<class ViewType1,
         class ViewType2,
         class IndexType,
         const bool takeSquareRootsOfScalingFactors,
         const bool takeAbsoluteValueOfScalingFactors>
class ElementWiseMultiply<ViewType1,
                          ViewType2,
                          IndexType,
                          takeSquareRootsOfScalingFactors,
                          takeAbsoluteValueOfScalingFactors,
                          1> {
public:
  static_assert (ViewType1::Rank == 1, "ViewType1 must be a rank-1 "
                 "Kokkos::View in order to use this specialization.");

  ElementWiseMultiply (const ViewType1& X,
                       const ViewType2& scalingFactors) :
    X_ (X),
    scalingFactors_ (scalingFactors)
  {}

  KOKKOS_INLINE_FUNCTION void operator () (const IndexType i) const {
    using val_type = typename ViewType2::non_const_value_type;
    using KAT = Kokkos::ArithTraits<val_type>;
    using mag_type = typename KAT::mag_type;
    using KAM = Kokkos::ArithTraits<mag_type>;

    if (takeAbsoluteValueOfScalingFactors) {
      const mag_type scalFactAbs = KAT::abs (scalingFactors_(i));
      const mag_type scalFinalVal = takeSquareRootsOfScalingFactors ?
        KAM::sqrt (scalFactAbs) : scalFactAbs;
      X_(i) = X_(i) * scalFinalVal;
    }
    else {
      const val_type scalFact = scalingFactors_(i);
      const val_type scalFinalVal = takeSquareRootsOfScalingFactors ?
        KAT::sqrt (scalFact) : scalFact;
      X_(i) = X_(i) * scalFinalVal;
    }
  }

private:
  ViewType1 X_;
  typename ViewType2::const_type scalingFactors_;
};

template<class ViewType1,
         class ViewType2,
         class IndexType,
         const bool takeSquareRootsOfScalingFactors,
         const bool takeAbsoluteValueOfScalingFactors>
class ElementWiseMultiply<ViewType1,
                          ViewType2,
                          IndexType,
                          takeSquareRootsOfScalingFactors,
                          takeAbsoluteValueOfScalingFactors,
                          2> {
public:
  static_assert (ViewType1::Rank == 2, "ViewType1 must be a rank-2 "
                 "Kokkos::View in order to use this specialization.");

  ElementWiseMultiply (const ViewType1& X,
                       const ViewType2& scalingFactors) :
    X_ (X),
    scalingFactors_ (scalingFactors)
  {}

  KOKKOS_INLINE_FUNCTION void operator () (const IndexType i) const {
    using val_type = typename ViewType2::non_const_value_type;
    using KAT = Kokkos::ArithTraits<val_type>;
    using mag_type = typename KAT::mag_type;
    using KAM = Kokkos::ArithTraits<mag_type>;

    for (IndexType j = 0; j < static_cast<IndexType> (X_.extent (1)); ++j) {
      if (takeAbsoluteValueOfScalingFactors) {
        const mag_type scalFactAbs = KAT::abs (scalingFactors_(i));
        const mag_type scalFinalVal = takeSquareRootsOfScalingFactors ?
          KAM::sqrt (scalFactAbs) : scalFactAbs;
        X_(i,j) = X_(i,j) * scalFinalVal;
      }
      else {
        const val_type scalFact = scalingFactors_(i);
        const val_type scalFinalVal = takeSquareRootsOfScalingFactors ?
          KAT::sqrt (scalFact) : scalFact;
        X_(i,j) = X_(i,j) * scalFinalVal;
      }
    }
  }

private:
  ViewType1 X_;
  typename ViewType2::const_type scalingFactors_;
};

template<class MultiVectorViewType,
         class ScalingFactorsViewType,
         class IndexType>
void
elementWiseMultiply (const MultiVectorViewType& X,
                     const ScalingFactorsViewType& scalingFactors,
                     const IndexType numRows,
                     const bool takeSquareRootsOfScalingFactors,
                     const bool takeAbsoluteValueOfScalingFactors =
                       ! std::is_same<
                           typename Kokkos::ArithTraits<
                             typename MultiVectorViewType::non_const_value_type
                           >::mag_type,
                           typename ScalingFactorsViewType::non_const_value_type
                         >::value)
{
  using execution_space = typename MultiVectorViewType::device_type::execution_space;
  using range_type = Kokkos::RangePolicy<execution_space, IndexType>;

  if (takeAbsoluteValueOfScalingFactors) {
    constexpr bool takeAbsVal = true;
    if (takeSquareRootsOfScalingFactors) {
      constexpr bool takeSquareRoots = true;
      using functor_type = ElementWiseMultiply<MultiVectorViewType,
        ScalingFactorsViewType, IndexType, takeSquareRoots, takeAbsVal>;
      Kokkos::parallel_for ("elementWiseMultiply",
                            range_type (0, numRows),
                            functor_type (X, scalingFactors));
    }
    else {
      constexpr bool takeSquareRoots = false;
      using functor_type = ElementWiseMultiply<MultiVectorViewType,
        ScalingFactorsViewType, IndexType, takeSquareRoots, takeAbsVal>;
      Kokkos::parallel_for ("elementWiseMultiply",
                            range_type (0, numRows),
                            functor_type (X, scalingFactors));
    }
  }
  else {
    constexpr bool takeAbsVal = false;
    if (takeSquareRootsOfScalingFactors) {
      constexpr bool takeSquareRoots = true;
      using functor_type = ElementWiseMultiply<MultiVectorViewType,
        ScalingFactorsViewType, IndexType, takeSquareRoots, takeAbsVal>;
      Kokkos::parallel_for ("elementWiseMultiply",
                            range_type (0, numRows),
                            functor_type (X, scalingFactors));
    }
    else {
      constexpr bool takeSquareRoots = false;
      using functor_type = ElementWiseMultiply<MultiVectorViewType,
        ScalingFactorsViewType, IndexType, takeSquareRoots, takeAbsVal>;
      Kokkos::parallel_for ("elementWiseMultiply",
                            range_type (0, numRows),
                            functor_type (X, scalingFactors));
    }
  }
}

template<class MultiVectorType, class ScalingFactorsViewType>
void
elementWiseMultiplyMultiVector (MultiVectorType& X,
                                const ScalingFactorsViewType& scalingFactors,
                                const bool takeSquareRootsOfScalingFactors,
                                const bool takeAbsoluteValueOfScalingFactors =
                                  ! std::is_same<
                                      typename Kokkos::ArithTraits<
                                        typename MultiVectorType::scalar_type
                                      >::mag_type,
                                      typename ScalingFactorsViewType::non_const_value_type
                                    >::value)
{
  using device_type = typename MultiVectorType::device_type;
  using dev_memory_space = typename device_type::memory_space;
  using index_type = typename MultiVectorType::local_ordinal_type;

  const index_type lclNumRows = static_cast<index_type> (X.getLocalLength ());

  if (X.template need_sync<dev_memory_space> ()) {
    X.template sync<dev_memory_space> ();
  }
  X.template modify<dev_memory_space> ();

  auto X_lcl = X.template getLocalView<dev_memory_space> ();
  if (static_cast<std::size_t> (X.getNumVectors ()) == std::size_t (1)) {
    using pair_type = Kokkos::pair<index_type, index_type>;
    auto X_lcl_1d = Kokkos::subview (X_lcl, pair_type (0, lclNumRows), 0);
    elementWiseMultiply (X_lcl_1d, scalingFactors, lclNumRows,
                         takeSquareRootsOfScalingFactors,
                         takeAbsoluteValueOfScalingFactors);
  }
  else {
    elementWiseMultiply (X_lcl, scalingFactors, lclNumRows,
                         takeSquareRootsOfScalingFactors,
                         takeAbsoluteValueOfScalingFactors);
  }
}

template<class ViewType1,
         class ViewType2,
         class IndexType,
         const bool takeSquareRootsOfScalingFactors,
         const bool takeAbsoluteValueOfScalingFactors =
           ! std::is_same<
               typename Kokkos::ArithTraits<
                 typename ViewType1::non_const_value_type
               >::mag_type,
               typename ViewType2::non_const_value_type
             >::value,
         const int rank = ViewType1::Rank>
class ElementWiseDivide {};

template<class ViewType1,
         class ViewType2,
         class IndexType,
         const bool takeSquareRootsOfScalingFactors,
         const bool takeAbsoluteValueOfScalingFactors>
class ElementWiseDivide<ViewType1,
                        ViewType2,
                        IndexType,
                        takeSquareRootsOfScalingFactors,
                        takeAbsoluteValueOfScalingFactors,
                        1> {
public:
  static_assert (ViewType1::Rank == 1, "ViewType1 must be a rank-1 "
                 "Kokkos::View in order to use this specialization.");

  ElementWiseDivide (const ViewType1& X,
                     const ViewType2& scalingFactors) :
    X_ (X),
    scalingFactors_ (scalingFactors)
  {}

  KOKKOS_INLINE_FUNCTION void operator () (const IndexType i) const {
    using val_type = typename ViewType2::non_const_value_type;
    using KAT = Kokkos::ArithTraits<val_type>;
    using mag_type = typename KAT::mag_type;
    using KAM = Kokkos::ArithTraits<mag_type>;

    if (takeAbsoluteValueOfScalingFactors) {
      const mag_type scalFactAbs = KAT::abs (scalingFactors_(i));
      const mag_type scalFinalVal = takeSquareRootsOfScalingFactors ?
        KAM::sqrt (scalFactAbs) : scalFactAbs;
      X_(i) = X_(i) / scalFinalVal;
    }
    else {
      const val_type scalFact = scalingFactors_(i);
      const val_type scalFinalVal = takeSquareRootsOfScalingFactors ?
        KAT::sqrt (scalFact) : scalFact;
      X_(i) = X_(i) / scalFinalVal;
    }
  }

private:
  ViewType1 X_;
  typename ViewType2::const_type scalingFactors_;
};

template<class ViewType1,
         class ViewType2,
         class IndexType,
         const bool takeSquareRootsOfScalingFactors,
         const bool takeAbsoluteValueOfScalingFactors>
class ElementWiseDivide<ViewType1,
                        ViewType2,
                        IndexType,
                        takeSquareRootsOfScalingFactors,
                        takeAbsoluteValueOfScalingFactors,
                        2> {
public:
  static_assert (ViewType1::Rank == 2, "ViewType1 must be a rank-2 "
                 "Kokkos::View in order to use this specialization.");

  ElementWiseDivide (const ViewType1& X,
                     const ViewType2& scalingFactors) :
    X_ (X),
    scalingFactors_ (scalingFactors)
  {}

  KOKKOS_INLINE_FUNCTION void operator () (const IndexType i) const {
    using val_type = typename ViewType2::non_const_value_type;
    using KAT = Kokkos::ArithTraits<val_type>;
    using mag_type = typename KAT::mag_type;
    using KAM = Kokkos::ArithTraits<mag_type>;

    for (IndexType j = 0; j < static_cast<IndexType> (X_.extent (1)); ++j) {
      if (takeAbsoluteValueOfScalingFactors) {
        const mag_type scalFactAbs = KAT::abs (scalingFactors_(i));
        const mag_type scalFinalVal = takeSquareRootsOfScalingFactors ?
          KAM::sqrt (scalFactAbs) : scalFactAbs;
        X_(i,j) = X_(i,j) / scalFinalVal;
      }
      else {
        const val_type scalFact = scalingFactors_(i);
        const val_type scalFinalVal = takeSquareRootsOfScalingFactors ?
          KAT::sqrt (scalFact) : scalFact;
        X_(i,j) = X_(i,j) / scalFinalVal;
      }
    }
  }

private:
  ViewType1 X_;
  typename ViewType2::const_type scalingFactors_;
};

template<class MultiVectorViewType,
         class ScalingFactorsViewType,
         class IndexType>
void
elementWiseDivide (const MultiVectorViewType& X,
                   const ScalingFactorsViewType& scalingFactors,
                   const IndexType numRows,
                   const bool takeSquareRootsOfScalingFactors,
                   const bool takeAbsoluteValueOfScalingFactors =
                     ! std::is_same<
                         typename Kokkos::ArithTraits<
                           typename MultiVectorViewType::non_const_value_type
                         >::mag_type,
                         typename ScalingFactorsViewType::non_const_value_type
                       >::value)
{
  using execution_space = typename MultiVectorViewType::device_type::execution_space;
  using range_type = Kokkos::RangePolicy<execution_space, IndexType>;

  if (takeAbsoluteValueOfScalingFactors) {
    constexpr bool takeAbsVal = true;
    if (takeSquareRootsOfScalingFactors) {
      constexpr bool takeSquareRoots = true;
      using functor_type = ElementWiseDivide<MultiVectorViewType,
        ScalingFactorsViewType, IndexType, takeSquareRoots, takeAbsVal>;
      Kokkos::parallel_for ("elementWiseDivide",
                            range_type (0, numRows),
                            functor_type (X, scalingFactors));
    }
    else {
      constexpr bool takeSquareRoots = false;
      using functor_type = ElementWiseDivide<MultiVectorViewType,
        ScalingFactorsViewType, IndexType, takeSquareRoots, takeAbsVal>;
      Kokkos::parallel_for ("elementWiseDivide",
                            range_type (0, numRows),
                            functor_type (X, scalingFactors));
    }
  }
  else {
    constexpr bool takeAbsVal = false;
    if (takeSquareRootsOfScalingFactors) {
      constexpr bool takeSquareRoots = true;
      using functor_type = ElementWiseDivide<MultiVectorViewType,
        ScalingFactorsViewType, IndexType, takeSquareRoots, takeAbsVal>;
      Kokkos::parallel_for ("elementWiseDivide",
                            range_type (0, numRows),
                            functor_type (X, scalingFactors));
    }
    else {
      constexpr bool takeSquareRoots = false;
      using functor_type = ElementWiseDivide<MultiVectorViewType,
        ScalingFactorsViewType, IndexType, takeSquareRoots, takeAbsVal>;
      Kokkos::parallel_for ("elementWiseDivide",
                            range_type (0, numRows),
                            functor_type (X, scalingFactors));
    }
  }
}

template<class MultiVectorType, class ScalingFactorsViewType>
void
elementWiseDivideMultiVector (MultiVectorType& X,
                              const ScalingFactorsViewType& scalingFactors,
                              const bool takeSquareRootsOfScalingFactors,
                              const bool takeAbsoluteValueOfScalingFactors =
                                ! std::is_same<
                                    typename Kokkos::ArithTraits<
                                      typename MultiVectorType::scalar_type
                                    >::mag_type,
                                    typename ScalingFactorsViewType::non_const_value_type
                                  >::value)
{
  using device_type = typename MultiVectorType::device_type;
  using dev_memory_space = typename device_type::memory_space;
  using index_type = typename MultiVectorType::local_ordinal_type;

  const index_type lclNumRows = static_cast<index_type> (X.getLocalLength ());

  if (X.template need_sync<dev_memory_space> ()) {
    X.template sync<dev_memory_space> ();
  }
  X.template modify<dev_memory_space> ();

  auto X_lcl = X.template getLocalView<dev_memory_space> ();
  if (static_cast<std::size_t> (X.getNumVectors ()) == std::size_t (1)) {
    using pair_type = Kokkos::pair<index_type, index_type>;
    auto X_lcl_1d = Kokkos::subview (X_lcl, pair_type (0, lclNumRows), 0);
    elementWiseDivide (X_lcl_1d, scalingFactors, lclNumRows,
                       takeSquareRootsOfScalingFactors,
                       takeAbsoluteValueOfScalingFactors);
  }
  else {
    elementWiseDivide (X_lcl, scalingFactors, lclNumRows,
                       takeSquareRootsOfScalingFactors,
                       takeAbsoluteValueOfScalingFactors);
  }
}

// See example here:
//
// http://en.cppreference.com/w/cpp/string/byte/toupper
std::string stringToUpper (std::string s)
{
  std::transform (s.begin (), s.end (), s.begin (),
                  [] (unsigned char c) { return std::toupper (c); });
  return s;
}

std::vector<std::string>
splitIntoStrings (const std::string& s,
                  const char sep = ',')
{
  using size_type = std::string::size_type;

  size_type cur_pos;
  size_type last_pos = 0;
  size_type length = s.length ();

  std::vector<std::string> strings;
  while (last_pos < length + size_type (1)) {
    cur_pos = s.find_first_of(sep, last_pos);
    if (cur_pos == std::string::npos) {
      cur_pos = length;
    }
    if (cur_pos != last_pos) {
      auto token = std::string (s.data () + last_pos,
                                static_cast<size_type> (cur_pos - last_pos));
      strings.push_back (stringToUpper (token));
    }
    last_pos = cur_pos + size_type (1);
  }
  return strings;
}

template<class T>
std::vector<T>
splitIntoValues (const std::string& s,
                 const char sep = ',')
{
  using size_type = std::string::size_type;

  size_type cur_pos;
  size_type last_pos = 0;
  size_type length = s.length ();

  std::vector<T> values;
  while (last_pos < length + size_type (1)) {
    cur_pos = s.find_first_of(sep, last_pos);
    if (cur_pos == std::string::npos) {
      cur_pos = length;
    }
    if (cur_pos != last_pos) {
      auto token = std::string (s.data () + last_pos,
                                static_cast<size_type> (cur_pos - last_pos));
      T val {};
      std::istringstream is (token);
      is >> val;
      if (is) {
        values.push_back (val);
      }
    }
    last_pos = cur_pos + size_type (1);
  }
  return values;
}

// Values of command-line arguments.
struct CmdLineArgs {
  std::string matrixFilename;
  std::string rhsFilename;
  std::string solverTypes = "GMRES";
  std::string orthogonalizationMethod = "ICGS";
  std::string convergenceToleranceValues = "1.0e-2";
  std::string maxIterValues = "100";
  std::string restartLengthValues = "20";
  std::string preconditionerTypes = "RELAXATION";
  bool solverVerbose = false;
  bool equilibrate = false;
  bool assumeSymmetric = false;
  bool assumeZeroInitialGuess = true;
  bool useDiagonalToEquilibrate = false;
  bool useLapack = false;
  bool useCustomBicgstab = false;
};

// Read in values of command-line arguments.
bool
getCmdLineArgs (CmdLineArgs& args, int argc, char* argv[])
{
  Teuchos::CommandLineProcessor cmdp (false, true);
  cmdp.setOption ("matrixFilename", &args.matrixFilename, "Name of Matrix "
                  "Market file with the sparse matrix A");
  cmdp.setOption ("rhsFilename", &args.rhsFilename, "Name of Matrix Market "
                  "file with the right-hand side vector(s) B");
  cmdp.setOption ("solverTypes", &args.solverTypes,
                  "One or more Belos solver types, "
                  "separated by commas");
  cmdp.setOption ("convergenceTolerances", &args.convergenceToleranceValues,
                  "One or more doubles, separated by commas; each value "
                  "is a convergence tolerance to try");
  cmdp.setOption ("orthogonalizationMethod", &args.orthogonalizationMethod,
                  "Orthogonalization method (for GMRES solver only; "
                  "ignored otherwise)");
  cmdp.setOption ("maxIters", &args.maxIterValues,
                  "One or more integers, separated by commas; each value "
                  "is a maximum number of solver iterations to try");
  cmdp.setOption ("restartLengths", &args.restartLengthValues,
                  "One or more integers, separated by commas; each value "
                  "is a maximum restart length to try (for GMRES solver only; "
                  "ignored otherwise)");
  cmdp.setOption ("preconditionerTypes", &args.preconditionerTypes,
                  "One or more Ifpack2 preconditioner types, "
                  "separated by commas");
  cmdp.setOption ("solverVerbose", "solverQuiet", &args.solverVerbose,
                  "Whether the Belos solver should print verbose output");
  cmdp.setOption ("equilibrate", "no-equilibrate", &args.equilibrate,
                  "Whether to equilibrate the linear system before solving it");
  cmdp.setOption ("assumeSymmetric", "no-assumeSymmetric",
                  &args.assumeSymmetric, "Whether equilibration should assume "
                  "that the matrix is symmetric");
  cmdp.setOption ("assumeZeroInitialGuess", "assumeNonzeroInitialGuess",
                  &args.assumeZeroInitialGuess, "Whether equilibration should "
                  "assume that the initial guess (vector) is zero");
  cmdp.setOption ("useDiagonalToEquilibrate", "useOneNorms",
                  &args.useDiagonalToEquilibrate,
                  "Whether equilibration should use the matrix's diagonal; "
                  "default is to use row and column one norms");
  cmdp.setOption ("lapack", "no-lapack",
                  &args.useLapack,
                  "Whether to compare against LAPACK's LU factorization "
		  "(expert driver).  If --equilibration, then use "
		  "equilibration in LAPACK too.");
  cmdp.setOption ("custom-bicgstab", "no-custom-bicgstab",
                  &args.useCustomBicgstab,
                  "Whether to compare against a hand-rolled BiCGSTAB "
		  "implementation.");

  auto result = cmdp.parse (argc, argv);
  return result == Teuchos::CommandLineProcessor::PARSE_SUCCESSFUL;
}

template<class ScalarType>
struct BelosSolverResult {
  typename Teuchos::ScalarTraits<ScalarType>::magnitudeType achievedTolerance;
  int numIters;
  bool converged;
  bool lossOfAccuracy;
};

template<class CrsMatrixType>
class BelosIfpack2Solver {
private:
  using scalar_type = typename CrsMatrixType::scalar_type;
  using local_ordinal_type = typename CrsMatrixType::local_ordinal_type;
  using global_ordinal_type = typename CrsMatrixType::global_ordinal_type;
  using node_type = typename CrsMatrixType::node_type;
  using row_matrix_type =
    Tpetra::RowMatrix<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using MV = Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using OP = Tpetra::Operator<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;
  using problem_type = Belos::LinearProblem<scalar_type, MV, OP>;
  using solver_type = Belos::SolverManager<scalar_type, MV, OP>;
  using preconditioner_type =
    Ifpack2::Preconditioner<scalar_type, local_ordinal_type, global_ordinal_type, node_type>;

  void createPreconditioner ()
  {
    rightPrec_ = Teuchos::null; // destroy old one first, to reduce peak memory use
    if (precType_ != "NONE") {
      rightPrec_ =
        Ifpack2::Factory::create<row_matrix_type> (precType_, A_);
      rightPrec_->setParameters (* (precParams_));
    }
  }

  void createSolver ()
  {
    Belos::SolverFactory<scalar_type, MV, OP> belosFactory;
    solver_ = Teuchos::null; // destroy old one first, to reduce peak memory use
    solver_ = belosFactory.create (solverType_, solverParams_);
  }

  void setPreconditionerMatrix (const Teuchos::RCP<const row_matrix_type>& A)
  {
    if (precType_ != "NONE" && rightPrec_.get () != nullptr) {
      // Not all Ifpack2 preconditioners have a setMatrix method.  Use
      // it if it exists; otherwise, start over with a new instance.
      using can_change_matrix = Ifpack2::Details::CanChangeMatrix<row_matrix_type>;
      can_change_matrix* rightPrec = dynamic_cast<can_change_matrix*> (rightPrec_.get ());
      if (rightPrec != nullptr) {
        rightPrec_->setMatrix (A);
      }
      else {
        rightPrec_ = Teuchos::null; // blow it away; make a new one only on demand
      }
    }
  }

  void initializePreconditioner ()
  {
    if (precType_ != "NONE") {
      if (rightPrec_.get () == nullptr) {
        createPreconditioner ();
      }
      rightPrec_->initialize ();
    }
  }

  void computePreconditioner ()
  {
    if (precType_ != "NONE") {
      if (rightPrec_.get () == nullptr) {
        createPreconditioner ();
        rightPrec_->initialize ();
      }
      rightPrec_->compute ();
    }
  }

  void equilibrateMatrix ()
  {
    TEUCHOS_TEST_FOR_EXCEPTION
      (A_.get () == nullptr, std::runtime_error, "Solver: You must call "
       "setMatrix with a nonnull matrix before you may call compute.");
    if (equilibrate_) {
      using Tpetra::computeRowAndColumnOneNorms;
      using Tpetra::leftAndOrRightScaleCrsMatrix;

      equibResult_ = computeRowAndColumnOneNorms (*A_, assumeSymmetric_);
      if (useDiagonalToEquilibrate_) {
        using device_type = typename node_type::device_type;
        using mag_type = typename Kokkos::ArithTraits<scalar_type>::mag_type;
        using view_type = Kokkos::View<mag_type*, device_type>;

        view_type rowDiagAbsVals ("rowDiagAbsVals",
                                  equibResult_.rowDiagonalEntries.extent (0));
        KokkosBlas::abs (rowDiagAbsVals, equibResult_.rowDiagonalEntries);
        view_type colDiagAbsVals ("colDiagAbsVals",
                                  equibResult_.colDiagonalEntries.extent (0));
        KokkosBlas::abs (colDiagAbsVals, equibResult_.colDiagonalEntries);

        leftAndOrRightScaleCrsMatrix (*A_, rowDiagAbsVals, colDiagAbsVals,
                                      true, true, equibResult_.assumeSymmetric,
                                      Tpetra::SCALING_DIVIDE);
      }
      else {
        auto colScalingFactors = equibResult_.assumeSymmetric ?
          equibResult_.colNorms :
          equibResult_.rowScaledColNorms;
        leftAndOrRightScaleCrsMatrix (*A_, equibResult_.rowNorms,
                                      colScalingFactors, true, true,
                                      equibResult_.assumeSymmetric,
                                      Tpetra::SCALING_DIVIDE);
      }
    } // if equilibrate_
  }

  void
  preScaleRightHandSides (Tpetra::MultiVector<scalar_type, local_ordinal_type,
                            global_ordinal_type, node_type>& B) const
  {
    if (equilibrate_) {
      if (useDiagonalToEquilibrate_) {
        const bool takeSquareRootsOfScalingFactors = false; // just use the diagonal entries
        elementWiseDivideMultiVector (B, equibResult_.rowDiagonalEntries,
                                      takeSquareRootsOfScalingFactors);
      }
      else {
        const bool takeSquareRootsOfScalingFactors = equibResult_.assumeSymmetric;
        elementWiseDivideMultiVector (B, equibResult_.rowNorms,
                                      takeSquareRootsOfScalingFactors);
      }
    }
  }

  void
  preScaleInitialGuesses (Tpetra::MultiVector<scalar_type, local_ordinal_type,
                            global_ordinal_type, node_type>& X) const
  {
    if (equilibrate_ && ! assumeZeroInitialGuess_) {
      if (useDiagonalToEquilibrate_) {
        const bool takeSquareRootsOfScalingFactors = false; // just use the diagonal entries
        elementWiseMultiplyMultiVector (X, equibResult_.colDiagonalEntries,
                                        takeSquareRootsOfScalingFactors);
      }
      else {
        auto colScalingFactors = equibResult_.assumeSymmetric ?
          equibResult_.colNorms :
          equibResult_.rowScaledColNorms;
        const bool takeSquareRootsOfScalingFactors =
          equibResult_.assumeSymmetric;
        elementWiseMultiplyMultiVector (X, colScalingFactors,
                                        takeSquareRootsOfScalingFactors);
      }
    }
  }

  void
  postScaleSolutionVectors (Tpetra::MultiVector<scalar_type, local_ordinal_type,
                              global_ordinal_type, node_type>& X) const
  {
    if (equilibrate_) {
      if (useDiagonalToEquilibrate_) {
        const bool takeSquareRootsOfScalingFactors = false; // just use the diagonal entries
        elementWiseDivideMultiVector (X, equibResult_.colDiagonalEntries,
                                      takeSquareRootsOfScalingFactors);
      }
      else {
        auto colScalingFactors = equibResult_.assumeSymmetric ?
          equibResult_.colNorms :
          equibResult_.rowScaledColNorms;
        const bool takeSquareRootsOfScalingFactors =
          equibResult_.assumeSymmetric;
        elementWiseDivideMultiVector (X, colScalingFactors,
                                      takeSquareRootsOfScalingFactors);
      }
    }
  }

public:
  BelosIfpack2Solver () = default;

  BelosIfpack2Solver (const Teuchos::RCP<CrsMatrixType>& A,
                      const std::string& solverType = "GMRES",
                      const std::string& precType = "NONE") :
    A_ (A),
    solverType_ (solverType),
    precType_ (precType),
    equilibrate_ (false),
    useDiagonalToEquilibrate_ (false)
  {}

  void setMatrix (const Teuchos::RCP<const CrsMatrixType>& A)
  {
    if (A_.get () != A.get ()) {
      setPreconditionerMatrix (A);
      // Belos solvers don't deal well with a complete change of the matrix.
      solver_ = Teuchos::null;
    }
    A_ = A;
  }

  void
  setPreconditionerTypeAndParameters (const std::string& precType,
                                      const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
    precType_ = precType;
    precParams_ = params;
    if (rightPrec_.get () != nullptr) {
      if (precType_ != precType) {
        rightPrec_ = Teuchos::null; // blow it away; make a new one only on demand
      }
      else {
        rightPrec_->setParameters (*params);
      }
    }
  }

  void
  setSolverTypeAndParameters (const std::string& solverType,
                              const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
    solverType_ = solverType;
    solverParams_ = params;
    if (solver_.get () != nullptr) {
      if (solverType_ != solverType) {
        solver_ = Teuchos::null; // blow it away; make a new one only on demand
      }
      else {
        solver_->setParameters (params);
      }
    }
  }

  void
  setEquilibrationParameters (const Teuchos::RCP<Teuchos::ParameterList>& params)
  {
    if (params.get () != nullptr) {
      equilibrate_ = params->get ("Equilibrate", equilibrate_);
      assumeSymmetric_ = params->get ("Assume symmetric", assumeSymmetric_);
      assumeZeroInitialGuess_ = params->get ("Assume zero initial guess",
                                             assumeZeroInitialGuess_);
      useDiagonalToEquilibrate_ = params->get ("Use diagonal to equilibrate",
                                               useDiagonalToEquilibrate_);
    }
  }

  void initialize ()
  {
    TEUCHOS_TEST_FOR_EXCEPTION
      (A_.get () == nullptr, std::runtime_error, "Solver: You must call "
       "setMatrix with a nonnull matrix before you may call initialize.");
    // Calling this implies that the matrix's graph has changed.
    // Belos' solvers don't handle that very well, so best practice is
    // to recreate them in this case.
    solver_ = Teuchos::null;
    initializePreconditioner ();
  }

  void compute ()
  {
    TEUCHOS_TEST_FOR_EXCEPTION
      (A_.get () == nullptr, std::runtime_error, "Solver: You must call "
       "setMatrix with a nonnull matrix before you may call compute.");
    equilibrateMatrix ();
    // equilibration changes the matrix, so don't compute the
    // preconditioner until after doing that.
    computePreconditioner ();
  }

  BelosSolverResult<scalar_type>
  solve (Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>& X,
         Tpetra::MultiVector<scalar_type, local_ordinal_type, global_ordinal_type, node_type>& B)
  {
    using Teuchos::RCP;
    using Teuchos::rcpFromRef;

    if (solver_.get () == nullptr) {
      createSolver ();
    }
    if (rightPrec_.get () == nullptr) {
      createPreconditioner ();
      initializePreconditioner ();
      computePreconditioner ();
    }

    preScaleRightHandSides (B);
    preScaleInitialGuesses (X);

    RCP<problem_type> problem (new problem_type (A_, rcpFromRef (X), rcpFromRef (B)));
    if (rightPrec_.get () != nullptr) {
      problem->setRightPrec (rightPrec_);
    }
    problem->setProblem ();
    solver_->setProblem (problem);
    const Belos::ReturnType solveResult = solver_->solve ();

    postScaleSolutionVectors (X);

    typename Teuchos::ScalarTraits<scalar_type>::magnitudeType tol {-1.0};
    try {
      tol = solver_->achievedTol ();
    }
    catch (...) {}
    const int numIters = solver_->getNumIters ();
    const bool converged = (solveResult == Belos::Converged);
    const bool lossOfAccuracy = solver_->isLOADetected ();
    return {tol, numIters, converged, lossOfAccuracy};
  }

private:
  Teuchos::RCP<CrsMatrixType> A_;
  Teuchos::RCP<solver_type> solver_;
  Teuchos::RCP<preconditioner_type> rightPrec_;
  using equilibration_result_type = decltype (Tpetra::computeRowAndColumnOneNorms (*A_, false));
  equilibration_result_type equibResult_;

  std::string solverType_;
  Teuchos::RCP<Teuchos::ParameterList> solverParams_;
  std::string precType_;
  Teuchos::RCP<Teuchos::ParameterList> precParams_;

  bool equilibrate_;
  bool assumeSymmetric_;
  bool assumeZeroInitialGuess_;
  bool useDiagonalToEquilibrate_;
};

class TpetraInstance {
public:
  TpetraInstance (int* argc, char*** argv) {
    Tpetra::initialize (argc, argv);
  }

  ~TpetraInstance () {
    Tpetra::finalize ();
  }
};

template<class CrsMatrixType, class MultiVectorType>
void
solveAndReport (BelosIfpack2Solver<CrsMatrixType>& solver,
                const CrsMatrixType& A_original, // before scaling
                MultiVectorType& X,
                MultiVectorType& B,
                const std::string& solverType,
                const std::string& precType,
                const typename MultiVectorType::mag_type convergenceTolerance,
                const int maxIters,
                const int restartLength,
                const CmdLineArgs& args)
{
  using Teuchos::ParameterList;
  using Teuchos::RCP;

  X.putScalar (0.0);

  RCP<ParameterList> solverParams (new ParameterList ("Belos"));
  if (args.solverVerbose) {
    solverParams->set ("Verbosity",
                       Belos::IterationDetails |
                       Belos::FinalSummary |
                       Belos::StatusTestDetails);
  }
  solverParams->set ("Convergence Tolerance", convergenceTolerance);
  solverParams->set ("Maximum Iterations", maxIters);
  if (solverType == "GMRES") {
    solverParams->set ("Num Blocks", restartLength);
    solverParams->set ("Maximum Restarts", restartLength * maxIters);
    solverParams->set ("Orthogonalization", args.orthogonalizationMethod);
  }

  RCP<ParameterList> precParams (new ParameterList ("Ifpack2"));
  if (precType == "RELAXATION") {
    precParams->set ("relaxation: type", "Symmetric Gauss-Seidel");
  }

  RCP<ParameterList> equibParams (new ParameterList ("Equilibration"));
  equibParams->set ("Equilibrate", args.equilibrate);
  equibParams->set ("Assume symmetric", args.assumeSymmetric);
  equibParams->set ("Assume zero initial guess",
                    args.assumeZeroInitialGuess);
  equibParams->set ("Use diagonal to equilibrate",
                    args.useDiagonalToEquilibrate);

  solver.setSolverTypeAndParameters (solverType, solverParams);
  solver.setPreconditionerTypeAndParameters (precType, precParams);
  solver.setEquilibrationParameters (equibParams);

  solver.initialize ();
  solver.compute ();

  // Keep this around for later computation of the explicit residual
  // norm.  If the solver equilibrates, it will modify the original B.
  MultiVectorType R (B, Teuchos::Copy);

  // Compute ||B||_2.
  using mag_type = typename MultiVectorType::mag_type;
  Teuchos::Array<mag_type> norms (R.getNumVectors ());
  R.norm2 (norms ());
  mag_type B_norm2_max = Kokkos::ArithTraits<mag_type>::zero ();
  for (std::size_t j = 0; j < B.getNumVectors (); ++j) {
    // Any NaN will persist (since the first test will fail);
    // this is what we want
    B_norm2_max = norms[j] < B_norm2_max ? B_norm2_max : norms[j];
  }

  // Solve the linear system AX=B.
  auto result = solver.solve (X, B);

  // Compute the actual residual norm ||B - A*X||_2.
  using scalar_type = typename MultiVectorType::scalar_type;
  const scalar_type ONE = Teuchos::ScalarTraits<scalar_type>::one ();
  A_original.apply (X, R, Teuchos::NO_TRANS, -ONE, ONE); // R := -A*X + B
  R.norm2 (norms ());

  mag_type R_norm2_max = Kokkos::ArithTraits<mag_type>::zero ();
  for (std::size_t j = 0; j < R.getNumVectors (); ++j) {
    // Any NaN will persist (since the first test will fail);
    // this is what we want
    R_norm2_max = norms[j] < R_norm2_max ? R_norm2_max : norms[j];
  }

  X.norm2 (norms ());
  mag_type X_norm2_max = Kokkos::ArithTraits<mag_type>::zero ();
  for (std::size_t j = 0; j < R.getNumVectors (); ++j) {
    // Any NaN will persist (since the first test will fail);
    // this is what we want
    X_norm2_max = norms[j] < X_norm2_max ? X_norm2_max : norms[j];
  }

  const int myRank = X.getMap ()->getComm ()->getRank ();
  if (myRank == 0) {
    using std::cout;
    using std::endl;
    cout << "Solver:" << endl
         << "  Solver type: " << solverType << endl
         << "  Preconditioner type: " << precType << endl
         << "  Convergence tolerance: " << convergenceTolerance << endl
         << "  Maximum number of iterations: " << maxIters << endl;
    if (solverType == "GMRES") {
      cout << "  Restart length: " << restartLength << endl
           << "  Orthogonalization method: " << args.orthogonalizationMethod << endl;
    }
    cout << "Results:" << endl
         << "  Converged: " << (result.converged ? "true" : "false") << endl
         << "  Number of iterations: " << result.numIters << endl
         << "  Achieved tolerance: " << result.achievedTolerance << endl
         << "  Loss of accuracy: " << result.lossOfAccuracy << endl
         << "  ||B-A*X||_2: " << R_norm2_max << endl
         << "  ||B||_2: " << B_norm2_max << endl
         << "  ||X||_2: " << X_norm2_max << endl;
    if (B_norm2_max != Kokkos::ArithTraits<mag_type>::zero ()) {
      cout << "  ||B-A*X||_2 / ||B||_2: " << (R_norm2_max / B_norm2_max)
           << endl;
    }
    cout << endl;
  }
}

} // namespace (anonymous)

int
main (int argc, char* argv[])
{
  using Teuchos::ParameterList;
  using Teuchos::RCP;
  using std::cerr;
  using std::endl;
  using crs_matrix_type = Tpetra::CrsMatrix<>;
  using MV = Tpetra::MultiVector<>;
  // using mag_type = MV::mag_type;
  using reader_type = Tpetra::MatrixMarket::Reader<crs_matrix_type>;

  TpetraInstance tpetraInstance (&argc, &argv);
  auto comm = Tpetra::getDefaultComm ();

  // Get command-line arguments.
  CmdLineArgs args;
  const bool gotCmdLineArgs = getCmdLineArgs (args, argc, argv);
  if (! gotCmdLineArgs) {
    if (comm->getRank () == 0) {
      cerr << "Failed to get command-line arguments!" << endl;
    }
    return EXIT_FAILURE;
  }

  if (args.matrixFilename == "") {
    if (comm->getRank () == 0) {
      cerr << "Must specify sparse matrix filename!" << endl;
    }
    return EXIT_FAILURE;
  }

  std::vector<std::string> solverTypes;
  if (args.solverTypes == "") {
    solverTypes = {"GMRES", "TFQMR", "BICGSTAB"};
  }
  else {
    solverTypes = splitIntoStrings (args.solverTypes);
  }

  std::vector<std::string> preconditionerTypes;
  if (args.preconditionerTypes == "") {
    preconditionerTypes = {"RELAXATION"};
  }
  else {
    preconditionerTypes = splitIntoStrings (args.preconditionerTypes);
  }

  std::vector<int> maxIterValues;
  if (args.maxIterValues == "") {
    maxIterValues = {100};
  }
  else {
    maxIterValues = splitIntoValues<int> (args.maxIterValues);
  }

  std::vector<int> restartLengthValues;
  if (args.restartLengthValues == "") {
    restartLengthValues = {20};
  }
  else {
    restartLengthValues = splitIntoValues<int> (args.restartLengthValues);
  }

  std::vector<double> convergenceToleranceValues;
  if (args.convergenceToleranceValues == "") {
    convergenceToleranceValues = {20};
  }
  else {
    convergenceToleranceValues =
      splitIntoValues<double> (args.convergenceToleranceValues);
  }

  // Read sparse matrix A from Matrix Market file.
  RCP<crs_matrix_type> A =
    reader_type::readSparseFile (args.matrixFilename, comm);
  if (A.get () == nullptr) {
    if (comm->getRank () == 0) {
      cerr << "Failed to load sparse matrix A from file "
        "\"" << args.matrixFilename << "\"!" << endl;
    }
    return EXIT_FAILURE;
  }

  // Read right-hand side vector(s) B from Matrix Market file, or
  // generate B if file not specified.
  RCP<MV> B;
  RCP<MV> X;
  if (args.rhsFilename == "") {
    B = Teuchos::rcp (new MV (A->getRangeMap (), 1));
    X = Teuchos::rcp (new MV (A->getDomainMap (), 1));
    X->putScalar (1.0);
    A->apply (*X, *B);
    X->putScalar (0.0);

    double norm {0.0};
    using host_device_type = Kokkos::Device<Kokkos::DefaultExecutionSpace, Kokkos::HostSpace>;
    Kokkos::View<double*, host_device_type> normView (&norm, B->getNumVectors ());
    B->norm2 (normView);
    if (norm != 0.0) {
      B->scale (1.0 / norm);
    }
  }
  else {
    auto map = A->getRangeMap ();
    B = reader_type::readDenseFile (args.rhsFilename, comm, map);
    if (B.get () == nullptr) {
      if (comm->getRank () == 0) {
        cerr << "Failed to load right-hand side vector(s) from file \""
             << args.rhsFilename << "\"!" << endl;
      }
      return EXIT_FAILURE;
    }
    X = Teuchos::rcp (new MV (A->getDomainMap (), B->getNumVectors ()));
  }

  auto A_original = deepCopyFillCompleteCrsMatrix (*A);

  if (args.useLapack) {
    using std::cout;

    using lapack_matrix_type = HostDenseMatrix<>;
    lapack_matrix_type A_lapack;
    lapack_matrix_type B_lapack;
    int errCode = gatherAndDensify (A_lapack, B_lapack, *A, *B);
    
    int lapackSolveOk = (errCode == 0) ? 1 : 0;
    Teuchos::broadcast (*comm, 0, Teuchos::inOutArg (lapackSolveOk));
    if (lapackSolveOk != 1) {
      if (comm->getRank () == 0) {
	cerr << "Gathering and densification for LAPACK solve FAILED!" << endl;
      }
    }
    
    if (lapackSolveOk && comm->getRank () == 0) {
      const int numRows = int (A_lapack.extent (0));
      const int numCols = int (A_lapack.extent (1));
      const int LDA = (numRows == 0) ? 1 : A_lapack.stride (1);
      const int NRHS = int (B_lapack.extent (1));
      const int LDB = (numRows == 0) ? 1 : B_lapack.stride (1);

      cout << "The matrix A, in dense format:" << endl;
      for (int i = 0; i < numRows; ++i) {
	for (int j = 0; j < numCols; ++j) { 	    
	  cout << A_lapack(i,j);
	  if (j + 1 < numCols) {
	    cout << " ";
	  }
	}
	cout << endl;
      }
      cout << endl;

      cout << "The right-hand side(s) B, in dense format:" << endl;
      for (int i = 0; i < numRows; ++i) {
	for (int j = 0; j < NRHS; ++j) { 
	  cout << B_lapack(i,j);
	  if (j + 1 < NRHS) {
	    cout << " ";
	  }
	}
	cout << endl;
      }
      cout << endl;
	
      if (numRows != numCols) {
	cerr << "Matrix is not square, so we can't use LAPACK's LU factorization on it!" << endl;
	lapackSolveOk = 0;
      }
      else {
	findEigenvaluesAndReport (A_lapack);	
	solveLeastSquaresProblemAndReport (A_lapack, B_lapack);
	
	Teuchos::LAPACK<int, double> lapack;

	int INFO = 0;

	lapack_matrix_type AF_lapack ("AF", numRows, numCols);
	lapack_matrix_type X_lapack ("X_lapack", numCols, NRHS);
	const int LDAF = (numRows == 0) ? 1 : AF_lapack.stride (1);
	const int LDX = (numRows == 0) ? 1 : X_lapack.stride (1);

	// Save norms of columns of B, since _GESVX may modify B.
	std::vector<double> B_norms (NRHS);
	for (int k = 0; k < NRHS; ++k) {
	  B_norms[k] =
	    KokkosBlas::nrm2 (Kokkos::subview (B_lapack, Kokkos::ALL (), k));
	}

	// Save a copy of A for later.
	lapack_matrix_type A_copy ("A_copy", numRows, numCols);
	Kokkos::deep_copy (A_copy, A_lapack);
	
	// Save a copy of B for later.
	lapack_matrix_type R_lapack ("R_lapack", numRows, NRHS);
	Kokkos::deep_copy (R_lapack, B_lapack);	  

	std::vector<int> IPIV (numCols);
	std::vector<double> R (numRows);
	std::vector<double> C (numCols);
	std::vector<double> FERR (NRHS);
	std::vector<double> BERR (NRHS);
	std::vector<double> WORK (4*numRows);
	std::vector<int> IWORK (numRows);
	INFO = 0;

	const char FACT ('E');
	const char TRANS ('N');	
	char EQUED[1];
	EQUED[0] = args.equilibrate ? 'B' : 'N';
	double RCOND = 1.0;

	cout << "Solver:" << endl
	     << "  Solver type: LAPACK's _GESVX" << endl
	     << "  Equilibrate: " << (args.equilibrate ? "YES" : "NO") << endl;
	lapack.GESVX (FACT, TRANS, numRows, NRHS, A_lapack.data (), LDA,
		      AF_lapack.data (), LDAF, IPIV.data (), EQUED, R.data (),
		      C.data (), B_lapack.data (), LDB, X_lapack.data (), LDX,
		      &RCOND, FERR.data (), BERR.data (), WORK.data (),
		      IWORK.data (), &INFO);

	cout << "Results:" << endl
	     << "  INFO: " << INFO << endl
	     << "  RCOND: " << RCOND << endl;
	if (NRHS > 0) {
	  cout << "  Pivot growth factor: " << WORK[0] << endl;
	}
	for (int j = 0; j < NRHS; ++j) {
	  cout << "  For right-hand side " << j
	       << ": forward error is " << FERR[j]
	       << ", backward error is " << BERR[j] << endl;
	}

	// Compute the explicit residual norm(s).
	KokkosBlas::gemm ("N", "N", -1.0, A_copy, X_lapack, +1.0, R_lapack);
	std::vector<double> explicitResidualNorms (NRHS);
	for (int k = 0; k < NRHS; ++k) {
	  explicitResidualNorms[k] =
	    KokkosBlas::nrm2 (Kokkos::subview (R_lapack, Kokkos::ALL (), k));
	}

	for (int j = 0; j < NRHS; ++j) {
	  cout << "  For right-hand side " << j
	       << ": ||B-A*X||_2 = "
	       << explicitResidualNorms[j]
	       << ", ||B||_2 = " << B_norms[j] << endl;
	}
	cout << endl;
      }
    }
    
    Teuchos::broadcast (*comm, 0, Teuchos::inOutArg (lapackSolveOk));
    if (lapackSolveOk != 1) {
      if (comm->getRank () == 0) {
	cerr << "LAPACK solve FAILED!" << endl;
      }
    }
  }

  if (args.useCustomBicgstab) {
    for (double convTol : convergenceToleranceValues) {
      for (int maxIters : maxIterValues) {
	MV X_copy (*X, Teuchos::Copy);
	const Tpetra::Operator<>* M = nullptr;
	const Tpetra::Operator<>& A_ref = static_cast<const Tpetra::Operator<>& > (*A);
	auto result = bicgstab (X_copy, A_ref, M, *B, maxIters, convTol);

	using std::cout;
	using std::endl;
	cout << "Solver:" << endl
	     << "  Solver type: Custom BiCGSTAB" << endl
	     << "  Preconditioner type: NONE" << endl
	     << "  Convergence tolerance: " << convTol << endl
	     << "  Maximum number of iterations: " << maxIters << endl
	     << "Results:" << endl
	     << "  Converged: " << std::get<2> (result) << endl
	     << "  Number of iterations: " << std::get<1> (result) << endl
	     << "  Achieved tolerance: " << std::get<0> (result) << endl
	     << endl;
      }
    }
  }

  if (args.useCustomBicgstab) {
    for (double convTol : convergenceToleranceValues) {
      for (int maxIters : maxIterValues) {
	MV X_copy (*X, Teuchos::Copy);
	const Tpetra::Operator<>* M = nullptr;
	const Tpetra::Operator<>& A_ref = static_cast<const Tpetra::Operator<>& > (*A);
	auto result = bicgstab_aztecoo (X_copy, A_ref, M, *B, maxIters, convTol);

	using std::cout;
	using std::endl;
	cout << "Solver:" << endl
	     << "  Solver type: AztecOO-imitating custom BiCGSTAB" << endl
	     << "  Preconditioner type: NONE" << endl
	     << "  Convergence tolerance: " << convTol << endl
	     << "  Maximum number of iterations: " << maxIters << endl
	     << "Results:" << endl
	     << "  Converged: " << std::get<2> (result) << endl
	     << "  Number of iterations: " << std::get<1> (result) << endl
	     << "  Achieved tolerance: " << std::get<0> (result) << endl
	     << endl;
      }
    }
  }
  
  // Create the solver.
  BelosIfpack2Solver<crs_matrix_type> solver (A);

  // Solve the linear system using various solvers and preconditioners.
  for (std::string solverType : solverTypes) {
    for (std::string precType : preconditionerTypes) {
      for (int maxIters : maxIterValues) {
        for (int restartLength : restartLengthValues) {
          for (double convTol : convergenceToleranceValues) {
            solveAndReport (solver, *A_original, *X, *B,
                            solverType,
                            precType,
                            convTol,
                            maxIters,
                            restartLength,
                            args);
          }
        }
      }
    }
  }

  return EXIT_SUCCESS;
}
