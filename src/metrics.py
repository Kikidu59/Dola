"""
metrics.py — Evaluation metrics for comparing training schemes.

Implements the metrics used in Sections 4 and Appendix F:

    - W₂² (squared Wasserstein-2 distance) between empirical particle
      distributions, computed via the POT library.
    - RMD (Relative Measure Distance): a normalized version of W₂,
      defined as RMD²(µ, ν) = W₂²(µ, ν) / (M₂(µ) + M₂(ν))
      where M₂(µ) = 2·E[||Z||²] for Z ~ µ.
    - L² distance between models, estimated by Monte Carlo.

Install POT with: pip install pot
"""

import jax
import jax.numpy as jnp
import numpy as np

from src.model import forward_batch
from src.spaces import project_EG, action_on_z, GROUP_ELEMENTS, action_on_x


# =============================================================================
# Wasserstein-2 distance (via POT)
# =============================================================================


def _particles_to_vectors(particles):
    """Flatten (N, 2, 2) particles to (N, 4) vectors for OT computation.

    POT works with point clouds in R^d, so we flatten each 2x2 matrix
    into a 4-dimensional vector.

    Args:
        particles: (N, 2, 2) JAX array

    Returns:
        (N, 4) numpy array (POT requires numpy, not JAX arrays)
    """
    N = particles.shape[0]
    return np.array(particles.reshape(N, 4))


def w2_squared(particles_a, particles_b):
    """Squared Wasserstein-2 distance between two empirical distributions.

    Uses POT's emd2 (Earth Mover's Distance) with squared Euclidean cost.

    W₂²(µ, ν) = min_γ Σᵢⱼ γᵢⱼ ||θᵢ^a - θⱼ^b||²

    where γ ranges over couplings with uniform marginals.

    Args:
        particles_a: (N_a, 2, 2) first set of particles
        particles_b: (N_b, 2, 2) second set of particles

    Returns:
        scalar — W₂² value
    """
    import ot  # import here to make the dependency optional

    X = _particles_to_vectors(particles_a)
    Y = _particles_to_vectors(particles_b)

    N_a, N_b = len(X), len(Y)

    # Uniform weights (empirical distributions)
    a = np.ones(N_a) / N_a
    b = np.ones(N_b) / N_b

    # Squared Euclidean cost matrix: C[i,j] = ||X[i] - Y[j]||²
    M = ot.dist(X, Y, metric="sqeuclidean")

    # Exact OT (linear program)
    return float(ot.emd2(a, b, M))


# =============================================================================
# RMD (Relative Measure Distance)
# =============================================================================


def second_moment(particles):
    """Second moment M₂(µ) = 2·E[||Z||²] for the empirical distribution.

    For particles θ₁, ..., θ_N:  M₂ = (2/N) Σᵢ ||θᵢ||²_F

    Args:
        particles: (N, 2, 2)

    Returns:
        scalar
    """
    norms_sq = jnp.sum(particles**2, axis=(1, 2))  # (N,)
    return float(2.0 * jnp.mean(norms_sq))


def rmd_squared(particles_a, particles_b):
    """Squared RMD (Relative Measure Distance).

    This normalization ensures 0 ≤ RMD ≤ 1 and allows comparisons
    across different values of N.

    Args:
        particles_a: (N_a, 2, 2)
        particles_b: (N_b, 2, 2)

    Returns:
        scalar — RMD² value
    """
    w2sq = w2_squared(particles_a, particles_b)
    m2_a = second_moment(particles_a)
    m2_b = second_moment(particles_b)

    denom = m2_a + m2_b
    if denom < 1e-12:
        return 0.0

    return w2sq / denom


def rmd(particles_a, particles_b):
    """RMD (not squared).

    Args:
        particles_a, particles_b: (N, 2, 2) each

    Returns:
        scalar — RMD value (in [0, 1])
    """
    return float(jnp.sqrt(rmd_squared(particles_a, particles_b)))


# =============================================================================
# Projection-based RMD: distance to E^G and to symmetrized version
# =============================================================================


def project_particles_EG(particles):
    """Project all particles onto E^G.

    Args:
        particles: (N, 2, 2)

    Returns:
        (N, 2, 2) — projected particles
    """
    return jax.vmap(project_EG)(particles)


def symmetrize_particles(particles):
    """Symmetrize the empirical distribution: (ν)^G.

    For G = C_2, this means adding the G-image of each particle.
    Result has 2N particles.

    Args:
        particles: (N, 2, 2)

    Returns:
        (2N, 2, 2) — symmetrized particles
    """
    g = GROUP_ELEMENTS[1]
    transformed = jax.vmap(lambda z: action_on_z(g, z))(particles)
    return jnp.concatenate([particles, transformed], axis=0)


def rmd_to_projected(particles):
    """RMD²(ν, ν^{E^G}): how far the distribution is from E^G.

    Args:
        particles: (N, 2, 2)

    Returns:
        scalar — RMD² value
    """
    projected = project_particles_EG(particles)
    return rmd_squared(particles, projected)


def rmd_to_symmetrized(particles):
    """RMD²(ν, ν^G): how far the distribution is from being WI.

    Args:
        particles: (N, 2, 2)

    Returns:
        scalar — RMD² value
    """
    symmetrized = symmetrize_particles(particles)
    return rmd_squared(particles, symmetrized)


# =============================================================================
# L² distance between models
# =============================================================================


def l2_distance(particles_a, particles_b, key, n_samples=100):
    """Estimated L² distance between two shallow models.

    ||Φ^a - Φ^b||²_{L²} ≈ (1/M) Σₖ ||Φ^a(xₖ) - Φ^b(xₖ)||²

    where xₖ ~ N(0, σ²_π · Id₂) with σ_π = 4.

    Args:
        particles_a: (N_a, 2, 2) first model's particles
        particles_b: (N_b, 2, 2) second model's particles
        key:         JAX random key
        n_samples:   number of Monte Carlo samples (default 100)

    Returns:
        scalar — estimated L² distance (not squared)
    """
    from src.teacher import SIGMA_PI

    x_batch = SIGMA_PI * jax.random.normal(key, shape=(n_samples, 2))
    y_a = forward_batch(x_batch, particles_a)
    y_b = forward_batch(x_batch, particles_b)

    # Mean squared difference
    msd = jnp.mean(jnp.sum((y_a - y_b) ** 2, axis=1))
    return float(jnp.sqrt(msd))
