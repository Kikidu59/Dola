"""
test_metrics.py — Tests for the evaluation metrics.

Checks that:
    1. RMD between identical distributions is 0
    2. RMD between different distributions is > 0
    3. RMD lies in [0, 1]
    4. W₂² is non-negative
    5. Projected particles have RMD²_to_EG ≈ 0 (tested for both setups)
    6. The L² distance from a model to itself is 0
    7. The L² distance between different models is > 0

Note: requires the `pot` library (pip install pot).
"""

import jax
import jax.numpy as jnp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.teacher import make_arbitrary_particles, make_wi_particles
from src.training import init_particles_wi
from src.metrics import (
    w2_squared,
    rmd_squared,
    rmd,
    l2_distance,
    rmd_to_projected,
    project_particles_EG,
)

TOL = 1e-5


# =============================================================================
# W₂² and RMD — setup-independent (OT on the particle vectors)
# =============================================================================


def test_rmd_identical():
    """RMD between identical distributions is 0."""
    particles = make_arbitrary_particles()
    r = rmd(particles, particles)
    assert r < TOL, f"RMD(µ, µ) = {r}, expected 0"


def test_rmd_different():
    """RMD between different distributions is > 0."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    a = init_particles_wi(k1, N=50)
    b = init_particles_wi(k2, N=50)
    r = rmd(a, b)
    assert r > 0.01, f"RMD should be > 0 for different distributions, got {r}"


def test_rmd_bounded():
    """RMD lies in [0, 1]."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    a = init_particles_wi(k1, N=50)
    b = init_particles_wi(k2, N=50)
    r = rmd(a, b)
    assert 0 <= r <= 1.0, f"RMD = {r}, expected in [0, 1]"


def test_w2_nonnegative():
    """W₂² is non-negative."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    a = init_particles_wi(k1, N=30)
    b = init_particles_wi(k2, N=30)
    assert w2_squared(a, b) >= 0


# =============================================================================
# rmd_to_projected — setup-dependent (project_EG depends on SETUP)
# =============================================================================


def test_projected_closer_to_EG(any_setup):
    """Particles projected onto E^G have RMD²_to_EG ≈ 0.

    Tested for both setups because project_EG is setup-dependent."""
    particles = init_particles_wi(jax.random.PRNGKey(3), N=50)

    rmd_before = rmd_to_projected(particles)
    projected = project_particles_EG(particles)
    rmd_after = rmd_to_projected(projected)

    assert rmd_after < TOL, (
        f"Setup {any_setup}: RMD² of projected particles to E^G = {rmd_after}"
    )
    assert rmd_before > rmd_after, (
        f"Setup {any_setup}: projection should reduce the distance to E^G"
    )


# =============================================================================
# L² distance — setup-independent (uses forward_batch)
# =============================================================================


def test_l2_distance_self():
    """The L² distance from a model to itself is 0."""
    particles = make_arbitrary_particles()
    d = l2_distance(particles, particles, jax.random.PRNGKey(4), n_samples=200)
    assert d < TOL, f"L²(f, f) = {d}, expected 0"


def test_l2_distance_positive():
    """The L² distance between different models is > 0."""
    teacher = make_wi_particles()
    student = init_particles_wi(jax.random.PRNGKey(6), N=20)
    d = l2_distance(student, teacher, jax.random.PRNGKey(5), n_samples=200)
    assert d > 0.01, f"L² distance should be > 0, got {d}"
