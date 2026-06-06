"""
model.py — Shallow neural network model Φ^N_θ.

Implements the model from Definition 1 of the paper:
    Φ^N_θ(x) = (1/N) Σᵢ σ*(x, θᵢ)

where θ = (θ₁, ..., θ_N) ∈ Z^N are the N "particles" (each a 2x2 matrix),
and σ*(x, z) = σ(z · x) is the jointly equivariant activation from spaces.py.
"""

from functools import partial

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


def forward_resnet(x, particles, alpha_arch=1):
    """Forward pass of the ResNet for a single input x.

    h_0 = x
    h_l = h_{l-1} + (alpha_arch / (L * M)) Σᵢ σ*(h_{l-1}, z^{i,l})

    The 1/M factor is handled implicitly by forward() which averages over particles.
    jax.lax.scan iterates over the L layers (axis 0 of particles) without
    unrolling the computation graph, keeping JIT compilation efficient.

    Args:
        x:          (2,) input vector in X = R²
        particles:  (L, M, 2, 2) array of L layers of M parameter matrices
        alpha_arch: architectural scaling constant (default 1)

    Returns:
        (2,) output vector h_L ∈ R²
    """

    L = particles.shape[0]

    def step(h, l_param):
        return (
            h + (alpha_arch / L) * forward(h, l_param),
            None,
        )  # The 1/M factor is already accounted in the forward function.

    # Carries x accross the loop applying step for each l in L (axis 0)
    h_final, _ = jax.lax.scan(step, x, particles)
    return h_final


@partial(jax.jit, static_argnames=["alpha_arch"])
def forward_batch_resnet(x_batch, particles, alpha_arch=1):
    """Forward pass of the ResNet for a batch of inputs.

    Vectorizes `forward_resnet` over the batch dimension.

    JIT-compiled: this forward is called EAGERLY in the training loop (to sample
    the teacher's minibatch via sample_data_resnet, and to evaluate the loss at
    snapshots). Without jit, its lax.scan runs interpreted op-by-op (~100 ms per
    call), which dominated ResNet training time; compiled it is ~1000x faster.
    Nesting this inside the already-jitted loss/step functions is fine (XLA
    inlines it), and jit does not change the numerics.

    Args:
        x_batch:    (B, 2) batch of input vectors
        particles:  (L, M, 2, 2) array of L layers of M parameter matrices
        alpha_arch: architectural scaling constant (default 1)

    Returns:
        (B, 2) batch of output vectors
    """
    return jax.vmap(forward_resnet, in_axes=(0, None, None))(
        x_batch, particles, alpha_arch
    )
