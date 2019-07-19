#ifndef TPETRA_DETAILS_LOCALEXPLICITTRANSPOSE_HPP
#define TPETRA_DETAILS_LOCALEXPLICITTRANSPOSE_HPP

#include "Tpetra_Details_computeOffsets.hpp"
#include "Tpetra_Details_shortSort.hpp"

namespace Tpetra {
namespace Details {

template<class LocalCrsMatrixType>
static LocalCrsMatrixType
localExplicitTranspose (const LocalCrsMatrixType& A,
                        const bool conjugate,
                        const bool sort = true)
{
  using matrix_type = LocalCrsMatrixType;
  using LO = typename matrix_type::ordinal_type;

  const LO lclNumCols (A.numCols ());
  const LO lclNumRows (A.numRows ());
  const size_t nnz (A.nnz ());

  using graph_type = typename matrix_type::staticcrsgraph_type;
  graph_type G = A.graph;

  // Determine how many nonzeros there are per row in the transpose.
  using DT = typename matrix_type::device_type;
  Kokkos::View<LO*, DT> t_counts ("transpose row counts", lclNumCols);

  using execution_space = typename matrix_type::execution_space;
  using range_type = Kokkos::RangePolicy<LO, execution_space>;
  Kokkos::parallel_for
    ("Compute row counts of local transpose",
     range_type (0, lclNumRows),
     KOKKOS_LAMBDA (const LO row) {
      auto rowView = G.rowConst(row);
      const LO length  = rowView.length;

      for (LO colID = 0; colID < length; ++colID) {
        const LO col = rowView(colID);
        Kokkos::atomic_fetch_add (&t_counts[col], LO (1));
      }
    });

  using Kokkos::view_alloc;
  using Kokkos::WithoutInitializing;
  typename matrix_type::row_map_type::non_const_type t_offsets
    (view_alloc ("transpose ptr", WithoutInitializing),
     lclNumCols + 1);

  // TODO (mfh 10 Jul 2019, mfh 18 Jul 2019): This returns the sum
  // of all counts, which could be useful for checking nnz.
  using Tpetra::Details::computeOffsetsFromCounts;
  (void) computeOffsetsFromCounts (t_offsets, t_counts);

  typename matrix_type::index_type::non_const_type t_cols
    (view_alloc ("transpose lcl ind", WithoutInitializing), nnz);
  using values_type = typename matrix_type::values_type::non_const_type;
  values_type t_vals
    (view_alloc ("transpose val", WithoutInitializing), nnz);

  using IST = typename matrix_type::value_type;
  using offset_type = typename graph_type::size_type;
  Kokkos::parallel_for
    ("Compute local transpose",
     range_type (0, lclNumRows),
     KOKKOS_LAMBDA (const LO row) {
      auto rowView = A.rowConst(row);
      const LO length (rowView.length);

      for (LO colID = 0; colID < length; ++colID) {
        const LO col = rowView.colidx(colID);
        const offset_type beg = t_offsets[col];
        const LO old_count =
          Kokkos::atomic_fetch_sub (&t_counts[col], LO (1));
        const LO len (t_offsets[col+1] - beg);
        const offset_type insert_pos = beg + (len - old_count);
        t_cols[insert_pos] = row;
        t_vals[insert_pos] = conjugate ?
          Kokkos::ArithTraits<IST>::conj (rowView.value(colID)) :
          rowView.value(colID);
      }
    });

  // Invariant: At this point, all entries of t_counts are zero.
  // This means we can use it to store new post-merge counts.

  // NOTE (mfh 11 Jul 2019, 18 Jul 2019) Merging is unnecessary: above
  // parallel_for visits each row of the original matrix once, so
  // there can be no duplicate column indices in the transpose.
  if (sort) {
    using Tpetra::Details::shellSortKeysAndValues;
    Kokkos::parallel_for
      ("Sort rows of local transpose",
       range_type (0, lclNumCols),
       KOKKOS_LAMBDA (const LO lclCol) {
        const offset_type beg = t_offsets[lclCol];
        const LO len (t_offsets[lclCol+1] - t_offsets[lclCol]);

        LO* cols_beg = t_cols.data () + beg;
        IST* vals_beg = t_vals.data () + beg;
        shellSortKeysAndValues (cols_beg, vals_beg, len);
      });
  }
  return matrix_type ("transpose", lclNumCols, lclNumRows, nnz,
                      t_vals, t_offsets, t_cols);
}

} // namespace Details
} // namespace Tpetra

#endif // TPETRA_DETAILS_LOCALEXPLICITTRANSPOSE_HPP
