#!/usr/bin/env python3
"""Public implementation; private development notes omitted."""

from __future__ import annotations

import numpy as np


def mesh_edges(faces: np.ndarray) -> set[tuple[int, int]]:
    """Undirected edge set of a triangle mesh."""
    edges: set[tuple[int, int]] = set()
    for a, b, c in faces:
        for u, v in ((a, b), (b, c), (c, a)):
            edges.add((min(u, v), max(u, v)))
    return edges


def graph_laplacian(faces: np.ndarray, n_verts: int) -> np.ndarray:
    """Combinatorial Laplacian L = D - A of the mesh graph.

    Chosen over the cotangent Laplacian deliberately: cotangent weights depend on
    the geometry being fitted, which would make the basis a function of the
    current estimate. The combinatorial one depends on connectivity alone, so the
    subspace is fixed before any data is seen and cannot drift with the fit.
    """
    lap = np.zeros((n_verts, n_verts))
    for u, v in mesh_edges(faces):
        if u >= n_verts or v >= n_verts:
            continue
        lap[u, v] -= 1.0
        lap[v, u] -= 1.0
        lap[u, u] += 1.0
        lap[v, v] += 1.0
    return lap


def spectral_basis(faces: np.ndarray, n_verts: int, k: int) -> np.ndarray:
    """The k smoothest deformation modes, as an (n_verts, k) orthonormal basis.

    Mode 0 (the constant) is dropped: it is a global translation, which the fit's
    per-frame offset already absorbs. Keeping it would make the system singular.
    """
    lap = graph_laplacian(faces, n_verts)
    vals, vecs = np.linalg.eigh(lap)
    order = np.argsort(vals)
    keep = order[1 : k + 1]  # skip the constant mode
    basis = vecs[:, keep]
    # orthonormal columns (eigh already gives this, but be explicit)
    return basis / np.linalg.norm(basis, axis=0, keepdims=True)


def solve_coefficients(
    obs: np.ndarray,
    canonical: np.ndarray,
    basis: np.ndarray,
    rots: np.ndarray,
    ts: np.ndarray,
    scales: np.ndarray,
    w: np.ndarray,
    ridge: float = 1e-8,
) -> np.ndarray:
    """Least-squares deformation coefficients C (k,3) given all frame cameras.

    Model:  shape = canonical + basis @ C
    So      x_ij ~= s_i * R_i[:2] @ (canonical_j + basis_j @ C) + t_i

    which is linear in C. Solved as one system in 3k unknowns rather than 1404.
    """
    n_f, n_p, _ = obs.shape
    k = basis.shape[1]
    a_rows = []
    b_rows = []
    for i in range(n_f):
        m = scales[i] * rots[i][:2]  # (2,3)
        base = canonical @ m.T + ts[i]  # (P,2) prediction from canonical alone
        resid = obs[i] - base  # (P,2) what the deformation must explain
        sw = np.sqrt(w[i])[:, None]
        # d(pred)/dC[a,e] = basis[j,a] * m[:,e]
        # -> block for axis e is outer(basis[:,a], m[:,e]) over points
        blk = np.einsum("pa,de->pdae", basis, m).reshape(n_p * 2, k * 3)
        a_rows.append(blk * np.repeat(sw, 2, axis=0))
        b_rows.append((resid * sw).reshape(-1))
    a = np.vstack(a_rows)
    b = np.concatenate(b_rows)
    ata = a.T @ a + ridge * np.trace(a.T @ a) / (3 * k) * np.eye(3 * k)
    coeff = np.linalg.solve(ata, a.T @ b)
    return coeff.reshape(k, 3)


def coefficient_variance(
    obs: np.ndarray,
    canonical: np.ndarray,
    basis: np.ndarray,
    rots: np.ndarray,
    ts: np.ndarray,
    scales: np.ndarray,
    w: np.ndarray,
    sigma_px: float,
) -> np.ndarray:
    """Per-landmark positional variance implied by the coefficient covariance.

    The noise floor still has to be computable, because the asymmetry bracket
    depends on it. With a basis the uncertainty lives in the coefficients, so it
    is propagated back to landmarks through the basis: var_j = sum_a B_ja^2 *
    var(C_a), summed over the three axes.
    """
    n_f, n_p, _ = obs.shape
    k = basis.shape[1]
    a_rows = []
    for i in range(n_f):
        m = scales[i] * rots[i][:2]
        sw = np.sqrt(w[i])[:, None]
        blk = np.einsum("pa,de->pdae", basis, m).reshape(n_p * 2, k * 3)
        a_rows.append(blk * np.repeat(sw, 2, axis=0))
    a = np.vstack(a_rows)
    ata = a.T @ a
    cov = np.linalg.pinv(ata) * (sigma_px**2)
    var_c = np.diag(cov).reshape(k, 3)  # per coefficient, per axis
    return (basis**2) @ var_c.sum(axis=1)
