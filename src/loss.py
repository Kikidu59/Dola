"""
loss.py — Loss function and population risk estimation.

Implements:
    - Quadratic loss: ℓ(ŷ, y) = ||ŷ - y||²
    - Regularization: r(z) = ||z||² (Frobenius norm squared for matrices)
    - Regularized population risk: R_{τ,β}(θ) estimated by Monte Carlo

From Equation (1) and Section 4 of the paper. The regularized risk is:
    R_{τ,β}(µ) = R(µ) + τ ∫ r dµ + β H_λ(µ)

In practice, the entropy term β H_λ(µ) is handled by the noise in SGD
(the SGLD term), not by an explicit computation. So here we only need
the first two terms: the data loss and the L2 regularization.
"""

import jax
import jax.numpy as jnp
from src.model import forward, forward_batch


def quadratic_loss(y_pred, y_true):
    """Quadratic loss for a single pair: ℓ(ŷ, y) = ||ŷ - y||².

    Args:
        y_pred: (2,) predicted output
        y_true: (2,) true label

    Returns:
        scalar loss value
    """
    diff = y_pred - y_true
    return jnp.sum(diff**2)


def quadratic_loss_batch(y_pred_batch, y_true_batch):
    """Average quadratic loss over a batch of M pairs.

    (1/M) Σₖ ||ŷₖ - yₖ||²

    Args:
        y_pred_batch: (M, 2) predicted outputs
        y_true_batch: (M, 2) true labels

    Returns:
        scalar — mean loss over the batch
    """
    return jnp.mean(jax.vmap(quadratic_loss)(y_pred_batch, y_true_batch))


def regularization(particles):
    """L2 regularization: (1/N) Σᵢ ||θᵢ||² (Frobenius norm squared).

    This is the empirical version of τ ∫ r dµ with r(z) = ||z||².

    Args:
        particles: (N, 2, 2) array of parameter matrices

    Returns:
        scalar — mean squared Frobenius norm
    """
    # ||θᵢ||²_F = sum of squares of all entries, for each particle
    norms_sq = jnp.sum(particles**2, axis=(1, 2))  # (N,)
    return jnp.mean(norms_sq)


def regularization_resnet(particles):
    """(1/(L*M)) Σ_l Σ_i ||z^{i,l}||²_F"""
    return regularization(particles.reshape(-1, 2, 2))


def population_risk(particles, x_batch, y_batch, tau=1e-4):
    """Estimate the regularized population risk by Monte Carlo.

    R̂_{τ}(θ) = (1/M) Σₖ ||Φ^N_θ(xₖ) - yₖ||²  +  τ · (1/N) Σᵢ ||θᵢ||²

    The entropy term (β) is not included here — it's handled implicitly
    by the Gaussian noise in the SGD update (SGLD).

    Args:
        particles: (N, 2, 2) student particles
        x_batch:   (M, 2) input samples
        y_batch:   (M, 2) corresponding true labels
        tau:       regularization strength (default 1e-4, as in the paper)

    Returns:
        scalar — estimated regularized risk
    """
    y_pred = forward_batch(x_batch, particles)
    data_loss = quadratic_loss_batch(y_pred, y_batch)
    reg_loss = regularization(particles)
    return data_loss + tau * reg_loss
