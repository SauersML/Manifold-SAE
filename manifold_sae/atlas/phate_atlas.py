"""PHATE / diffusion-map atlas of SAE atoms.

Public API
----------
extract_atom_directions(state_dict) -> (F, D) numpy array
    Architecture-agnostic atom-direction extractor. Handles the three baseline
    checkpoint flavours found in `runs/sae_comparison/`:
      - TopK / L1: row of `W_d` (F, D).
      - Manifold:  D_k is (F, K_basis, D); we collapse the basis dimension by
        taking the mean across basis (curve average) AND optionally the first
        PC of each curve. Mean is robust and matches "atom direction" semantics
        used elsewhere in the codebase.
    Additional architectures (DAS-SAE, WassersteinSAE, equivariant, crosscoder)
    are matched on common key patterns: `W_dec`, `decoder.weight`, `D_atoms`,
    `atoms`, etc.

atom_atlas(model_path, n_components=2, hue_labels=None) -> dict
    Loads a checkpoint, extracts atom directions, ℓ2-normalizes them on the
    unit hypersphere (atom DIRECTION is what matters geometrically — the
    magnitude is a learned gain), runs PHATE (if installed) else a 80-LOC
    spectral-embedding fallback on a kNN-graph diffusion operator. Returns
    embedding + diffusion-time spectrum + extra diagnostics.

persistent_h1(atom_atlas_out, max_dim=1) -> list[(birth, death)]
    Vietoris-Rips persistence on the 2D embedding. Uses ripser if available,
    otherwise the inline 30-LOC sub-level-set H0/H1 estimator (alpha-complex-
    free approximation: 1-skeleton + spanning-tree barcode + cycle-basis).
    The dominant-cycle test (max persistence vs runner-up) is what falsifies
    the H1-ring hypothesis.

mapper_atlas(atom_atlas_out, filter_values=None, n_bins=8, overlap=0.3,
             cluster_k=2) -> dict
    Mapper graph: 1-D cover on a scalar filter (default: hue correlation),
    DBSCAN-style single-linkage clustering within each bin, edges between
    nodes that share atoms across overlapping bins.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Atom direction extractor
# ---------------------------------------------------------------------------

# (key, orientation) where orientation says how to interpret the 2-D tensor:
#   "rows"  -> rows are atoms (F, D); use as-is.
#   "cols"  -> cols are atoms (D, F); transpose.
#   "auto"  -> infer by comparing dims to encoder width.
_DECODER_KEY_CANDIDATES: tuple[tuple[str, str], ...] = (
    ("W_d", "auto"),            # this repo: (F, D) ; some forks: (D, F)
    ("W_dec", "auto"),
    ("D_atoms", "rows"),
    ("atoms", "rows"),
    ("decoder.weight", "cols"),  # nn.Linear weight: (out_features, in_features)
    ("dec.weight", "cols"),
    ("W_decoder", "auto"),
    ("D", "auto"),
    ("decoder", "auto"),
    ("W_out", "auto"),
)


def extract_atom_directions(state_dict: dict[str, torch.Tensor]) -> np.ndarray:
    """Return (F, D) array of per-atom decoder directions.

    Strategy: try `D_k` first (manifold-style curve basis, (F, K, D) ->
    mean over K), then standard 2-D decoder matrices in known orientations.
    """
    # Unwrap common checkpoint wrappers: {"state_dict": ..., "config": ...} or
    # {"model": ...}.
    for wrap in ("state_dict", "model", "module", "sae"):
        if wrap in state_dict and isinstance(state_dict[wrap], dict):
            inner = state_dict[wrap]
            try:
                return extract_atom_directions(inner)
            except KeyError:
                continue
    # Manifold curve basis: shape (F, K_basis, D)
    if "D_k" in state_dict:
        D_k = state_dict["D_k"].detach().cpu().float().numpy()
        if D_k.ndim == 3:
            # Mean over basis = curve mean direction; equivalent to evaluating
            # the curve at a uniform point distribution on [0,1] under the
            # symmetric Duchon basis.
            return D_k.mean(axis=1)
        return D_k

    # Standard SAE decoder lookup
    for key, orient in _DECODER_KEY_CANDIDATES:
        if key in state_dict:
            W = state_dict[key].detach().cpu().float().numpy()
            if W.ndim != 2:
                continue
            if orient == "rows":
                return W
            if orient == "cols":
                return W.T
            # auto: figure out D from b_dec / b_d if present, else from
            # the encoder weight shape, else assume rows-as-atoms.
            F_, D_ = W.shape
            d_hint: int | None = None
            for bkey in ("b_dec", "b_d", "decoder.bias", "b_out"):
                if bkey in state_dict and hasattr(state_dict[bkey], "shape"):
                    bs = tuple(state_dict[bkey].shape)
                    if len(bs) == 1:
                        d_hint = int(bs[0])
                        break
            if d_hint is None:
                enc_dims: list[int] = []
                for k2, v2 in state_dict.items():
                    if "enc" in k2.lower() and hasattr(v2, "ndim") and v2.ndim == 2:
                        enc_dims.extend(int(x) for x in v2.shape)
                if enc_dims:
                    # Take the larger of encoder dims as D.
                    d_hint = max(enc_dims)
            if d_hint is not None:
                if D_ == d_hint and F_ != d_hint:
                    return W
                if F_ == d_hint and D_ != d_hint:
                    return W.T
            # Last-resort default: rows-as-atoms (this repo's convention).
            return W

    # Fall-through: look for any 2-D tensor that pairs an encoder dim
    encoder_widths = set()
    for k, v in state_dict.items():
        if "enc" in k.lower() and hasattr(v, "ndim") and v.ndim == 2:
            encoder_widths.update(v.shape)
    for k, v in state_dict.items():
        if hasattr(v, "ndim") and v.ndim == 2 and "dec" in k.lower():
            W = v.detach().cpu().float().numpy()
            F_, D_ = W.shape
            if F_ in encoder_widths and F_ <= D_:
                return W
            if D_ in encoder_widths and D_ <= F_:
                return W.T

    raise KeyError(
        f"Could not locate decoder atoms in state_dict. Keys: {list(state_dict.keys())[:20]}"
    )


def _normalize_rows(X: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, eps)


# ---------------------------------------------------------------------------
# PHATE / spectral diffusion-map fallback
# ---------------------------------------------------------------------------

def _knn_affinity(X: np.ndarray, k: int = 15, alpha: float = 2.0) -> np.ndarray:
    """alpha-decaying kNN affinity (PHATE-style)."""
    n = X.shape[0]
    # cosine distance since atoms live on the unit sphere
    Xn = _normalize_rows(X)
    sim = Xn @ Xn.T
    sim = np.clip(sim, -1.0, 1.0)
    dist = np.arccos(sim)  # angular distance on the sphere
    np.fill_diagonal(dist, 0.0)
    # adaptive bandwidth = distance to k-th NN
    part = np.partition(dist, k, axis=1)[:, k]
    eps = np.maximum(part, 1e-8)
    W = np.exp(-((dist / eps[:, None]) ** alpha))
    W = 0.5 * (W + W.T)
    np.fill_diagonal(W, 0.0)
    return W


def _diffusion_embed(
    W: np.ndarray, n_components: int = 2, t: int = 8
) -> tuple[np.ndarray, np.ndarray]:
    """Diffusion-map embedding from affinity W; returns (Y, eigenvalues)."""
    d = W.sum(axis=1)
    d = np.maximum(d, 1e-12)
    # symmetric normalized graph Laplacian's diffusion operator
    P_sym = W / np.sqrt(d[:, None] * d[None, :])
    # eigendecompose (small, F ~ 512)
    vals, vecs = np.linalg.eigh(P_sym)
    order = np.argsort(-vals)
    vals = vals[order]
    vecs = vecs[:, order]
    # drop trivial eigenvector
    vals_nontriv = vals[1 : n_components + 1]
    vecs_nontriv = vecs[:, 1 : n_components + 1]
    # PHATE potential coordinates: |log(eigenvalue)| weighting prevents
    # collapse when eigenvalues fall off fast. Use diffusion-coords if
    # eigenvalues are well-separated, else potential-coords.
    Y = vecs_nontriv.astype(np.float64) * (np.maximum(vals_nontriv, 1e-12) ** max(t, 1))
    # un-symmetrize: divide by sqrt(degree) to get right eigenvectors of P
    Y = Y / np.sqrt(d[:, None])
    # If diffusion coords collapsed (numerical underflow at large t),
    # fall back to the unweighted Laplacian eigenvectors.
    col_std = Y.std(axis=0)
    if np.any(col_std < 1e-10):
        Y = vecs_nontriv.astype(np.float64) / np.sqrt(d[:, None])
        col_std = Y.std(axis=0)
    # rescale columns to unit variance for plotting / downstream geometry
    col_std = np.maximum(col_std, 1e-12)
    Y = Y / col_std
    return Y, vals


def atom_atlas(
    model_path: str | Path,
    n_components: int = 2,
    hue_labels: np.ndarray | None = None,
    knn: int = 15,
    diffusion_t: int = 8,
) -> dict[str, Any]:
    """Load any SAE, extract atoms, embed via PHATE (or spectral fallback)."""
    model_path = Path(model_path)
    sd = torch.load(model_path, map_location="cpu", weights_only=False)
    if hasattr(sd, "state_dict"):
        sd = sd.state_dict()
    D_atoms = extract_atom_directions(sd)
    # remove dead atoms (zero rows)
    norms = np.linalg.norm(D_atoms, axis=1)
    alive = norms > 1e-8
    D_alive = D_atoms[alive]

    backend = "spectral_fallback"
    embedding: np.ndarray
    diffusion_spectrum: np.ndarray
    try:
        import phate  # type: ignore

        op = phate.PHATE(
            n_components=n_components,
            knn=knn,
            decay=40,
            t=diffusion_t,
            verbose=False,
            n_jobs=1,
        )
        embedding = op.fit_transform(D_alive)
        # PHATE doesn't expose eigenvalues; approximate from its diff op
        try:
            P = op.diff_op  # type: ignore[attr-defined]
            if hasattr(P, "toarray"):
                P = P.toarray()
            vals = np.linalg.eigvalsh(0.5 * (P + P.T))
            diffusion_spectrum = np.sort(vals)[::-1]
        except Exception:
            diffusion_spectrum = np.array([])
        backend = "phate"
    except Exception:
        W = _knn_affinity(D_alive, k=knn)
        embedding, diffusion_spectrum = _diffusion_embed(
            W, n_components=n_components, t=diffusion_t
        )

    out: dict[str, Any] = {
        "model_path": str(model_path),
        "n_atoms_total": int(D_atoms.shape[0]),
        "n_atoms_alive": int(alive.sum()),
        "alive_mask": alive,
        "atoms": D_atoms,
        "atoms_alive": D_alive,
        "embedding": embedding,
        "diffusion_spectrum": diffusion_spectrum,
        "backend": backend,
    }
    if hue_labels is not None:
        hl = np.asarray(hue_labels)
        if hl.shape[0] == D_atoms.shape[0]:
            out["hue_labels"] = hl[alive]
        else:
            out["hue_labels"] = hl
    return out


# ---------------------------------------------------------------------------
# Persistent H1 (Vietoris–Rips on the 2-D embedding)
# ---------------------------------------------------------------------------

def _rips_h1_inline(points: np.ndarray, max_edge: float | None = None) -> list[tuple[float, float]]:
    """Tiny VR-H1 estimator: enumerates short edges, builds a filtration,
    tracks 1-cycles by their birth (longest edge of the smallest containing
    triangle) and death (longest edge needed to fill them in). Approximate
    but adequate for the dominant-cycle test on ≤512 points in 2-D.

    Returns list of (birth, death) bars for H1. Persistence = death - birth.
    """
    n = points.shape[0]
    # pairwise distances
    diff = points[:, None, :] - points[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    if max_edge is None:
        max_edge = float(np.median(D[D > 0]) * 3.0)

    # Sort edges by length
    iu, ju = np.triu_indices(n, k=1)
    el = D[iu, ju]
    order = np.argsort(el)
    iu, ju, el = iu[order], ju[order], el[order]
    mask = el <= max_edge
    iu, ju, el = iu[mask], ju[mask], el[mask]

    # Union-find for H0 (components); edges that DON'T merge components are
    # candidates for creating an H1 class (loop closure).
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    # Track adjacency to detect when an H1 class is killed by triangle
    # completion. We approximate death as the longest edge that, together
    # with two earlier edges, first closes a triangle through this cycle.
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    h1_open: list[float] = []  # birth radii of currently-alive cycles
    h1_bars: list[tuple[float, float]] = []

    for i, j, r in zip(iu, ju, el):
        ri, rj = find(int(i)), find(int(j))
        if ri != rj:
            parent[ri] = rj
            adj[int(i)].add(int(j))
            adj[int(j)].add(int(i))
        else:
            # cycle birth at radius r
            h1_open.append(float(r))
            adj[int(i)].add(int(j))
            adj[int(j)].add(int(i))
            # check whether this new edge completes any triangle => kills a
            # previously-born cycle. Use: any common neighbor between i and j.
            common = adj[int(i)] & adj[int(j)] - {int(i), int(j)}
            if common and h1_open:
                # Approximate: the OLDEST open cycle dies here.
                birth = h1_open.pop(0)
                h1_bars.append((birth, float(r)))

    # Any cycles still open at the end get death = +inf (or max_edge).
    for b in h1_open:
        h1_bars.append((b, float(max_edge)))
    return h1_bars


def persistent_h1(
    atom_atlas_out: dict[str, Any],
    max_dim: int = 1,
    use_embedding: bool = True,
) -> list[tuple[float, float]]:
    """Vietoris-Rips persistence diagram (H0 + H1)."""
    pts = atom_atlas_out["embedding"] if use_embedding else atom_atlas_out["atoms_alive"]
    try:
        from ripser import ripser  # type: ignore

        res = ripser(np.asarray(pts, dtype=np.float64), maxdim=max_dim)
        dgms = res["dgms"]
        if len(dgms) > 1:
            bars = [(float(b), float(d)) for b, d in dgms[1] if math.isfinite(d)]
            # also include infinite bars with synthetic death at max finite + max range
            infs = [(float(b), float("inf")) for b, d in dgms[1] if not math.isfinite(d)]
            return bars + infs
        return []
    except Exception:
        return _rips_h1_inline(np.asarray(pts, dtype=np.float64))


# ---------------------------------------------------------------------------
# Mapper graph
# ---------------------------------------------------------------------------

def _single_linkage_cluster(X: np.ndarray, eps: float) -> np.ndarray:
    """Single-linkage clustering at radius eps. Returns integer labels."""
    n = X.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    diff = X[:, None, :] - X[None, :, :]
    D = np.sqrt((diff * diff).sum(-1))
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] <= eps:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def mapper_atlas(
    atom_atlas_out: dict[str, Any],
    filter_values: np.ndarray | None = None,
    n_bins: int = 8,
    overlap: float = 0.3,
    cluster_eps: float | None = None,
) -> dict[str, Any]:
    """Mapper graph on the atom set.

    Cover: 1-D filter (default = first PHATE coordinate, or `hue_labels` if
    present in the atlas dict).
    Cluster: single-linkage at `cluster_eps` within each bin (default = median
    pairwise distance in the embedding / 5).
    Edges: nodes from overlapping bins that share ≥1 atom.
    """
    emb = atom_atlas_out["embedding"]
    n = emb.shape[0]
    if filter_values is None:
        filter_values = atom_atlas_out.get("hue_labels")
    if filter_values is None:
        filter_values = emb[:, 0]
    filter_values = np.asarray(filter_values, dtype=np.float64)
    assert filter_values.shape[0] == n, f"{filter_values.shape} vs {n}"

    if cluster_eps is None:
        diff = emb[:, None, :] - emb[None, :, :]
        D = np.sqrt((diff * diff).sum(-1))
        upper = D[np.triu_indices(n, k=1)]
        cluster_eps = float(np.median(upper)) / 5.0

    fmin, fmax = float(filter_values.min()), float(filter_values.max())
    if fmax - fmin < 1e-12:
        fmax = fmin + 1.0
    bin_w = (fmax - fmin) / n_bins
    pad = overlap * bin_w

    bin_centers = [fmin + (b + 0.5) * bin_w for b in range(n_bins)]
    bin_intervals = [(c - bin_w / 2 - pad, c + bin_w / 2 + pad) for c in bin_centers]

    nodes: list[dict[str, Any]] = []
    node_members: list[set[int]] = []
    bin_of_node: list[int] = []
    for b, (lo, hi) in enumerate(bin_intervals):
        idx = np.where((filter_values >= lo) & (filter_values <= hi))[0]
        if idx.size == 0:
            continue
        labels = _single_linkage_cluster(emb[idx], cluster_eps)
        for c in np.unique(labels):
            members = idx[labels == c]
            nodes.append({
                "id": len(nodes),
                "bin": int(b),
                "filter_center": float(bin_centers[b]),
                "size": int(members.size),
                "members": members.tolist(),
            })
            node_members.append(set(members.tolist()))
            bin_of_node.append(int(b))

    edges: list[tuple[int, int, int]] = []  # (u, v, weight)
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if abs(bin_of_node[i] - bin_of_node[j]) > 1:
                continue
            shared = len(node_members[i] & node_members[j])
            if shared > 0:
                edges.append((i, j, shared))

    return {
        "nodes": nodes,
        "edges": edges,
        "n_bins": n_bins,
        "overlap": overlap,
        "cluster_eps": float(cluster_eps),
        "filter_range": (fmin, fmax),
    }


# ---------------------------------------------------------------------------
# Convenience: graphviz .dot for the mapper graph
# ---------------------------------------------------------------------------

def mapper_to_dot(mapper_out: dict[str, Any], title: str = "mapper") -> str:
    lines = [f'graph "{title}" {{', "  rankdir=LR;", "  node [shape=circle, style=filled];"]
    max_sz = max((n["size"] for n in mapper_out["nodes"]), default=1)
    for n in mapper_out["nodes"]:
        size = 0.3 + 1.5 * (n["size"] / max_sz)
        # rainbow color from filter center
        f = n["filter_center"]
        fmin, fmax = mapper_out["filter_range"]
        frac = (f - fmin) / max(fmax - fmin, 1e-9)
        r = int(255 * max(0.0, math.cos(frac * 2 * math.pi)))
        g = int(255 * max(0.0, math.cos((frac - 1 / 3) * 2 * math.pi)))
        b = int(255 * max(0.0, math.cos((frac - 2 / 3) * 2 * math.pi)))
        color = f"#{r:02x}{g:02x}{b:02x}"
        lines.append(
            f'  n{n["id"]} [width={size:.2f}, fillcolor="{color}", '
            f'label="b{n["bin"]}\\n{n["size"]}"];'
        )
    for (u, v, w) in mapper_out["edges"]:
        lines.append(f'  n{u} -- n{v} [penwidth={0.5 + 0.5 * w:.2f}];')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers used by the CLI / experiment to compute hue labels per atom
# ---------------------------------------------------------------------------

def hue_label_from_color_centroids(
    D_atoms: np.ndarray,
    color_centroids: np.ndarray,
    color_hues: np.ndarray,
) -> np.ndarray:
    """For each atom, find the most-aligned color centroid (cosine sim) and
    return its hue ∈ [0, 1). `color_centroids` is (N_COLORS, D), `color_hues`
    is (N_COLORS,) in [0, 1).
    """
    Dn = _normalize_rows(D_atoms)
    Cn = _normalize_rows(color_centroids)
    sim = Dn @ Cn.T  # (F, N_COLORS)
    best = np.argmax(sim, axis=1)
    return color_hues[best]
