"""
model.py — Shallow neural network model Φ^N_θ.

Implements the model from Definition 1 of the paper:
    Φ^N_θ(x) = (1/N) Σᵢ σ*(x, θᵢ)

where θ = (θ₁, ..., θ_N) ∈ Z^N are the N "particles" (each a 2x2 matrix),
and σ*(x, z) = σ(z · x) is the jointly equivariant activation from spaces.py.
"""

import jax
import jax.numpy as jnp
from src.spaces import sigma_star


def forward(x, particles):
    """Forward pass of the shallow NN for a single input x.

    Φ^N_θ(x) = (1/N) Σᵢ σ*(x, θᵢ)

    We use jax.vmap to vectorize σ* over the N particles without a Python loop.

    jax.vmap(sigma_star, in_axes=(None, 0)) means:
        - None  → don't batch over x (same x for all particles)
        - 0     → batch over axis 0 of particles (i.e. iterate over each θᵢ)

    So if particles has shape (N, 2, 2), the vmapped function returns
    an array of shape (N, 2) — one output per particle — and we take the mean.

    Args:
        x:         (2,) input vector in X = R²
        particles: (N, 2, 2) array of N parameter matrices in Z = R^{2x2}

    Returns:
        (2,) output vector in Y = R²
    """
    # Apply σ* to each particle: (N, 2, 2) → (N, 2)
    contributions = jax.vmap(sigma_star, in_axes=(None, 0))(x, particles)

    # Average over the N particles: (N, 2) → (2,)
    return jnp.mean(contributions, axis=0)


def forward_batch(x_batch, particles):
    """Forward pass for a batch of M inputs.

    Vectorizes `forward` over the first argument (the batch of inputs).
    If x_batch has shape (M, 2), returns an array of shape (M, 2).

    Args:
        x_batch:   (M, 2) batch of input vectors
        particles: (N, 2, 2) array of N parameter matrices

    Returns:
        (M, 2) batch of output vectors
    """
    return jax.vmap(forward, in_axes=(0, None))(x_batch, particles)
