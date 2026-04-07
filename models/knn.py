"""
Three-Level Implementation
--------------------------
Level 1  : sklearn.neighbors.NearestNeighbors baseline  → KNNBaseline
Level 2  : Pure-NumPy hand-coded engine                 → KNNRetrievalEngine
Level 3  : Vectorisation proofs + full math docstrings   → (same class, annotated)
"""

import numpy as np
import time
from typing import Tuple


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 1 — sklearn Baseline  (for validation / mock integration only)
# ══════════════════════════════════════════════════════════════════════════════
class KNNBaseline:
    """
    Level 1 Baseline — wraps sklearn.neighbors.NearestNeighbors.

    Used ONLY to validate that the retrieval logic is correct before we
    drop the library entirely.  The public API is intentionally identical
    to KNNRetrievalEngine so the two are plug-and-play interchangeable.

    Parameters
    ----------
    None — metric is chosen per-call, mirroring Level 2's interface.
    """

    def __init__(self):
        # Lazy import: keeps Level 2/3 usable even without sklearn installed.
        from sklearn.neighbors import NearestNeighbors  # noqa: F401
        self._NearestNeighbors = NearestNeighbors
        self._X: np.ndarray | None = None

    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray) -> "KNNBaseline":
        """Store the training matrix.  Shape: (n_samples, n_features)."""
        self._X = np.array(X, dtype=np.float64)
        return self

    # ------------------------------------------------------------------
    def retrieve_by_radius(
        self,
        query: np.ndarray,
        radius: float,
        metric: str = "euclidean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return all points within *radius* of *query* (Merchant Mode).

        Returns
        -------
        indices   : 1-D int array — row indices in X
        distances : 1-D float array — corresponding distances
        """
        nn = self._NearestNeighbors(metric=metric)
        nn.fit(self._X)
        dists, idxs = nn.radius_neighbors(
            query.reshape(1, -1), radius=radius, return_distance=True
        )
        return idxs[0], dists[0]

    # ------------------------------------------------------------------
    def retrieve_top_k(
        self,
        query: np.ndarray,
        k: int,
        metric: str = "cosine",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return the *k* most similar points to *query* (Tourist Mode).

        Returns
        -------
        indices     : 1-D int array — row indices in X (best → worst)
        similarities: 1-D float array — cosine similarity scores
        """
        nn = self._NearestNeighbors(n_neighbors=k, metric=metric)
        nn.fit(self._X)
        dists, idxs = nn.kneighbors(query.reshape(1, -1))
        # sklearn returns cosine *distance* (1 - similarity); convert back.
        similarities = 1.0 - dists[0]
        return idxs[0], similarities


# ══════════════════════════════════════════════════════════════════════════════
# LEVEL 2 + 3 — Pure-NumPy Hand-Coded Engine  (the real deliverable)
# ══════════════════════════════════════════════════════════════════════════════
class KNNRetrievalEngine:
    """
    Universal k-Nearest Neighbors Retrieval Engine — pure NumPy, zero sklearn.

    Supports two distance / similarity metrics:

    ┌─────────────────────┬───────────────────────────────────────────┐
    │ Metric              │ Use-case                                  │
    ├─────────────────────┼───────────────────────────────────────────┤
    │ Euclidean distance  │ Merchant Mode — geo / spatial radius      │
    │ Cosine similarity   │ Tourist Mode  — NLP / semantic Top-K      │
    └─────────────────────┴───────────────────────────────────────────┘

    Design Principles (from the spec)
    ----------------------------------
    * Dimension-agnostic: accepts any (n_samples, n_features) matrix.
      No hard-coded feature count anywhere in this file.
    * No Python for-loops for distance computation.  All inner products
      and norms are computed with NumPy broadcasting / matrix ops so the
      engine stays fast on datasets with tens of thousands of rows.
    * sklearn-compatible API: .fit(), .retrieve_by_radius(), .retrieve_top_k()

    Performance target
    ------------------
    On a 5 000-row dataset the full retrieve call must complete < 3 s.
    (Typically well under 50 ms with NumPy broadcasting.)
    """

    def __init__(self):
        self._X: np.ndarray | None = None          # (n_samples, n_features)
        self._X_norms: np.ndarray | None = None    # (n_samples,) L2 norms — cached

    # ──────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────

    def fit(self, X: np.ndarray) -> "KNNRetrievalEngine":
        """
        Store the reference dataset and pre-compute per-row L2 norms.

        Pre-computing norms is a classic optimisation: Cosine similarity
        requires dividing by ‖xᵢ‖ for every sample xᵢ.  Doing this once
        here (O(n·d)) is far cheaper than repeating it inside every query.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The corpus to search against.  Any numeric dtype; cast internally
            to float64 for precision.

        Returns
        -------
        self — enables method chaining:  engine.fit(X).retrieve_top_k(q, 5)
        """
        self._X = np.array(X, dtype=np.float64)          # defensive copy
        # ‖xᵢ‖₂  for each row — shape (n_samples,)
        # np.linalg.norm with axis=1 is vectorised C code, no Python loop.
        self._X_norms = np.linalg.norm(self._X, axis=1)  # (n_samples,)
        return self

    # ------------------------------------------------------------------
    def retrieve_by_radius(
        self,
        query: np.ndarray,
        radius: float,
        metric: str = "euclidean",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Merchant Mode — return every point within *radius* of *query*.

        Math (Euclidean distance)
        -------------------------
        Given query vector q ∈ ℝᵈ and corpus X ∈ ℝⁿˣᵈ:

            diff   = X - q           shape (n, d)   [broadcasting: q → (1,d)]
            dist²  = Σ diff²  axis=1  shape (n,)
            dist   = √dist²           shape (n,)

        Broadcasting replaces the loop  ``for i in range(n): dist[i] = ‖X[i]-q‖``
        which would be O(n) Python iterations.  Instead NumPy executes a
        single C-level subtraction on the entire matrix.

        Parameters
        ----------
        query  : 1-D array, shape (n_features,)
        radius : float  — distance threshold (same unit as your features,
                          e.g. degrees lat/lon or normalised embedding distance)
        metric : str    — currently only 'euclidean' is supported here;
                          the param is kept for API compatibility.

        Returns
        -------
        indices   : 1-D int array — indices of matching rows in X
        distances : 1-D float array — Euclidean distance to each match
        """
        self._check_fitted()
        q = self._validate_query(query)

        distances = self._euclidean_distances(q)      # shape (n_samples,)
        mask = distances <= radius                    # boolean filter
        indices = np.where(mask)[0]                  # positions that qualify

        # Sort by ascending distance (closest first)
        order = np.argsort(distances[indices])
        indices = indices[order]
        return indices, distances[indices]

    # ------------------------------------------------------------------
    def retrieve_top_k(
        self,
        query: np.ndarray,
        k: int,
        metric: str = "cosine",
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Tourist Mode — return the *k* most semantically similar points.

        Math (Cosine Similarity)
        ------------------------
        Given query vector q ∈ ℝᵈ (after L2-normalisation) and corpus X:

            dot_products = X @ q          shape (n,)
                         = Σⱼ Xᵢⱼ · qⱼ   (matrix-vector product — BLAS level)

            cos_sim[i] = dot_products[i] / (‖X[i]‖ · ‖q‖)

        Because we cached ‖X[i]‖ in .fit() and compute ‖q‖ once, the
        denominator is a single element-wise divide — still no Python loop.

        np.argpartition is used for Top-K selection instead of a full sort:
            • Full argsort  : O(n log n)
            • argpartition  : O(n)  — finds k smallest without sorting all n.
        This matters when n is large (5 000 + rows).

        Parameters
        ----------
        query  : 1-D array, shape (n_features,)
        k      : int — number of top results to return
        metric : str — currently 'cosine'; kept for API compatibility.

        Returns
        -------
        indices     : 1-D int array — top-k row indices (best → worst)
        similarities: 1-D float array — cosine similarity ∈ [-1, 1]
        """
        self._check_fitted()
        q = self._validate_query(query)
        k = min(k, len(self._X))                      # can't return more than n

        similarities = self._cosine_similarities(q)   # shape (n_samples,)

        # argpartition gives us the k LARGEST without a full sort.
        # We negate similarities so "largest" maps to "smallest negative".
        top_k_rough = np.argpartition(-similarities, k)[:k]   # unordered top-k
        # Final sort only over k elements (cheap)
        order = np.argsort(-similarities[top_k_rough])
        indices = top_k_rough[order]
        return indices, similarities[indices]

    # ──────────────────────────────────────────────────────────────────
    # Private helpers — pure NumPy, no loops
    # ──────────────────────────────────────────────────────────────────

    def _euclidean_distances(self, q: np.ndarray) -> np.ndarray:
        """
        Compute Euclidean distance from *q* to every row in self._X.

        Vectorised via broadcasting
        ---------------------------
            diff = self._X - q       # (n, d) - (d,)  →  NumPy broadcasts q
            dist = ‖diff‖₂  per row

        Equivalent scalar formula (NOT what we do):
            for i in range(n):
                dist[i] = sqrt(sum((X[i,j] - q[j])**2 for j in range(d)))

        Returns
        -------
        distances : shape (n_samples,), dtype float64
        """
        diff = self._X - q                         # broadcasting: (n,d)-(d,)
        return np.sqrt(np.einsum("ij,ij->i", diff, diff))
        # np.einsum("ij,ij->i", A, A)  ==  (A**2).sum(axis=1)
        # einsum fuses multiply+sum in one pass → fewer temporaries in memory.

    def _cosine_similarities(self, q: np.ndarray) -> np.ndarray:
        """
        Compute Cosine similarity from *q* to every row in self._X.

        Formula
        -------
            cos_sim(xᵢ, q) = (xᵢ · q) / (‖xᵢ‖ · ‖q‖)

        Implementation
        --------------
            dot_products = self._X @ q          # BLAS dgemv — (n,d)@(d,)→(n,)
            q_norm       = ‖q‖₂                 # scalar
            denominator  = self._X_norms * q_norm  # (n,) * scalar → (n,)
            similarities = dot_products / denominator

        Edge case: if ‖xᵢ‖ == 0 or ‖q‖ == 0, similarity is undefined → 0.0.

        Returns
        -------
        similarities : shape (n_samples,), dtype float64, values ∈ [-1, 1]
        """
        q_norm = np.linalg.norm(q)
        if q_norm == 0.0:
            return np.zeros(len(self._X), dtype=np.float64)

        dot_products = self._X @ q                         # (n,)
        denominator  = self._X_norms * q_norm              # (n,)

        # Safe divide: where denominator is 0, output 0 (avoid NaN / inf)
        with np.errstate(invalid="ignore", divide="ignore"):
            sims = np.where(denominator != 0, dot_products / denominator, 0.0)
        return sims

    # ──────────────────────────────────────────────────────────────────
    # Guard helpers
    # ──────────────────────────────────────────────────────────────────

    def _check_fitted(self):
        if self._X is None:
            raise RuntimeError("Call .fit(X) before retrieval.")

    def _validate_query(self, query: np.ndarray) -> np.ndarray:
        q = np.array(query, dtype=np.float64).ravel()    # ensure 1-D
        if q.shape[0] != self._X.shape[1]:
            raise ValueError(
                f"Query has {q.shape[0]} features but X has {self._X.shape[1]}."
            )
        return q


# ══════════════════════════════════════════════════════════════════════════════
# QUICK SELF-TEST  (run: python knn_retrieval_engine.py)
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # ── Generate synthetic dataset (5 000 rows, 128-dim — NLP-scale) ──────────
    N_SAMPLES  = 5_000
    N_FEATURES = 128          # dimension-agnostic: change this freely
    X = rng.standard_normal((N_SAMPLES, N_FEATURES))

    # ── Build engine ──────────────────────────────────────────────────────────
    engine = KNNRetrievalEngine()
    engine.fit(X)
    print(f"  fit()  — corpus shape: {X.shape}")

    # ── Tourist Mode: Top-K cosine ────────────────────────────────────────────
    query_tourist = rng.standard_normal(N_FEATURES)
    t0 = time.perf_counter()
    top_idx, top_sim = engine.retrieve_top_k(query_tourist, k=5)
    elapsed_topk = (time.perf_counter() - t0) * 1000

    print(f"\n  Tourist Mode — Top-5 (cosine similarity)")
    for rank, (i, s) in enumerate(zip(top_idx, top_sim), 1):
        print(f"   Rank {rank}: sample #{i:>4d}  similarity = {s:.6f}")
    print(f"   ⏱  {elapsed_topk:.3f} ms")

    # ── Merchant Mode: radius Euclidean ───────────────────────────────────────
    query_merchant = rng.standard_normal(N_FEATURES)
    RADIUS = 14.0        # Euclidean radius (tune to your feature scale)
    t0 = time.perf_counter()
    rad_idx, rad_dist = engine.retrieve_by_radius(query_merchant, radius=RADIUS)
    elapsed_radius = (time.perf_counter() - t0) * 1000

    print(f"\n  Merchant Mode — radius={RADIUS} (Euclidean)")
    print(f"   Found {len(rad_idx)} matches")
    for i, d in zip(rad_idx[:5], rad_dist[:5]):
        print(f"   Sample #{i:>4d}  dist = {d:.4f}")
    print(f"   ⏱  {elapsed_radius:.3f} ms")

    # ── Performance gate (DoD: < 3 000 ms) ───────────────────────────────────
    total = elapsed_topk + elapsed_radius
    status = "  PASS" if total < 3000 else "❌  FAIL"
    print(f"\n{status} — total retrieval time: {total:.3f} ms  (limit 3 000 ms)")

    # ── Level 1 baseline cross-check (only if sklearn is installed) ───────────
    try:
        baseline = KNNBaseline()
        baseline.fit(X)
        b_idx, b_sim = baseline.retrieve_top_k(query_tourist, k=5)
        print("\n🔬  Level-1 baseline Top-5 indices :", b_idx)
        print("    Level-2 engine   Top-5 indices :", top_idx)
        match = np.array_equal(np.sort(b_idx), np.sort(top_idx))
        print(f"    Indices match: {'✅  YES' if match else '⚠️  check tolerance'}")
    except ImportError:
        print("\n  sklearn not installed — baseline cross-check skipped.")