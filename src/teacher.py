"""
teacher.py — Teacher models for the teacher-student experiments.

Implements the three teacher variants from Appendix F (page 49):
    - Arbitrary: 5 fixed particles in Z = R^{2x2}, chosen by the authors
    - WI (Weakly Invariant): 10 particles = the 5 arbitrary + their G-images
    - SI (Strongly Invariant): 5 particles living in E^G, specified by
      2D coordinates in the orthonormal basis of E^G

All teachers share the same activation σ* and use the shallow model forward pass.
The data distribution is X ~ N(0, σ²_π · Id₂) with σ_π = 4, and Y = f*(X).
"""

import jax
import jax.numpy as jnp
from src.spaces import GROUP_ELEMENTS, action_on_z, get_EG_basis
from src.model import forward, forward_batch


SCALE = 0.5


def make_arbitrary_particles():
    """Create the 5 arbitrary teacher particles from Appendix F."""
    flat = SCALE * jnp.array(
        [
            [-1.0, 0.0, 0.0, 0.5],
            [0.5, 1.0, 0.0, 1.0],
            [-0.5, 0.3, 1.0, 0.0],
            [0.0, -1.0, -0.5, 1.0],
            [0.7, -0.7, 0.5, 0.7],
        ]
    )
    return flat.reshape(5, 2, 2)


def make_wi_particles():
    """Create the WI teacher: 5 arbitrary + their 5 G-images = 10 particles."""
    arbitrary = make_arbitrary_particles()
    g = GROUP_ELEMENTS[1]
    transformed = jax.vmap(lambda z: action_on_z(g, z))(arbitrary)
    return jnp.concatenate([arbitrary, transformed], axis=0)


def make_si_particles():
    """Create the SI teacher: 5 particles in E^G from 2D coordinates."""
    coeffs = SCALE * jnp.array(
        [
            [1.0, 0.0],
            [0.5, 1.0],
            [-0.5, 0.3],
            [0.0, -1.0],
            [0.7, 0.7],
        ]
    )

    e1, e2 = get_EG_basis()

    particles = (
        coeffs[:, 0:1, None] * e1[None, :, :] + coeffs[:, 1:2, None] * e2[None, :, :]
    )
    return particles


def make_si_particles_uv():
    """SI teacher adapted for uv activation. (Non zero coefs)"""
    coeffs = SCALE * jnp.array(
        [
            [0.8, 0.6],
            [-0.5, 0.9],
            [0.7, -0.8],
            [-0.3, 0.5],
            [0.6, 0.7],
        ]
    )
    e1, e2 = get_EG_basis()
    return coeffs[:, 0:1, None] * e1[None, :, :] + coeffs[:, 1:2, None] * e2[None, :, :]


SIGMA_PI = 4.0


def sample_data(key, teacher_particles, n_samples):
    """Sample (X, Y) pairs: X ~ N(0, σ²_π·Id₂), Y = f*(X)."""
    x_batch = SIGMA_PI * jax.random.normal(key, shape=(n_samples, 2))
    y_batch = forward_batch(x_batch, teacher_particles)
    return x_batch, y_batch
