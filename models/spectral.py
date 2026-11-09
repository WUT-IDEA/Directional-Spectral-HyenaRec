"""Finite-difference operators and deterministic compact spectral bases."""

import torch


def finite_difference_matrix(length, difference_order, dtype=torch.float64):
    """Return the first- or second-order difference matrix on a lag chain."""
    if difference_order not in (1, 2):
        raise ValueError("difference_order must be 1 or 2")
    rows = length - difference_order
    if rows < 1:
        raise ValueError("sequence length is too short for the difference order")
    matrix = torch.zeros(rows, length, dtype=dtype)
    if difference_order == 1:
        indices = torch.arange(rows)
        matrix[indices, indices] = -1.0
        matrix[indices, indices + 1] = 1.0
    else:
        indices = torch.arange(rows)
        matrix[indices, indices] = 1.0
        matrix[indices, indices + 1] = -2.0
        matrix[indices, indices + 2] = 1.0
    return matrix


def exposure_spectral_basis(
    exposure_profile,
    rank,
    difference_order=2,
    alpha=1.0,
    epsilon=1e-3,
):
    """Build deterministic low modes of an exposure-weighted lag smoother.

    The null space is fixed explicitly (constant for D1; constant and linear
    for D2) so repeated zero eigenvalues cannot rotate across eigensolvers.
    Returns basis rows, eigenvalues, the roughness matrix, and local weights.
    """
    exposure = torch.as_tensor(exposure_profile, dtype=torch.float64).flatten()
    length = exposure.numel()
    if length < 2:
        raise ValueError("exposure_profile must contain at least two lags")
    if not torch.isfinite(exposure).all() or (exposure < 0).any():
        raise ValueError("exposure_profile must be finite and non-negative")
    if exposure.max() <= 0:
        raise ValueError("exposure_profile must contain positive exposure")
    if not 1 <= rank <= length:
        raise ValueError(f"rank must be in [1, {length}]")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    difference = finite_difference_matrix(length, difference_order)
    normalized = exposure / exposure.max()
    if difference_order == 1:
        local_exposure = 0.5 * (normalized[:-1] + normalized[1:])
    else:
        local_exposure = normalized[1:-1]
    local_weights = (local_exposure + epsilon).pow(-alpha)
    local_weights = local_weights / local_weights.mean()
    roughness = difference.T @ (local_weights[:, None] * difference)
    roughness = 0.5 * (roughness + roughness.T)

    constant = torch.ones(length, dtype=torch.float64)
    constant = constant / torch.linalg.vector_norm(constant)
    null_vectors = [constant]
    if difference_order == 2:
        linear = torch.arange(length, dtype=torch.float64)
        linear = linear - linear.mean()
        linear = linear / torch.linalg.vector_norm(linear)
        null_vectors.append(linear)
    null_basis = torch.stack(null_vectors, dim=1)
    complete_basis, _ = torch.linalg.qr(null_basis, mode="complete")
    complement = complete_basis[:, len(null_vectors):]
    reduced = complement.T @ roughness @ complement
    positive_values, positive_vectors = torch.linalg.eigh(reduced)
    positive_modes = complement @ positive_vectors
    modes = torch.cat([null_basis, positive_modes], dim=1)
    eigenvalues = torch.cat([
        torch.zeros(len(null_vectors), dtype=torch.float64),
        positive_values.clamp_min(0.0),
    ])

    # Resolve the remaining per-vector sign ambiguity deterministically.
    for column in range(modes.shape[1]):
        pivot = modes[:, column].abs().argmax()
        if modes[pivot, column] < 0:
            modes[:, column].neg_()
    return modes[:, :rank].T, eigenvalues[:rank], roughness, local_weights


