"""
test_metrics.py — Tests pour les métriques d'évaluation.

Vérifie :
    1. RMD entre distributions identiques est 0
    2. RMD entre distributions différentes est > 0
    3. RMD est dans [0, 1]
    4. W₂² est non-négatif
    5. Les particules projetées ont RMD²_to_EG ≈ 0 (testé pour les deux setups)
    6. La distance L² d'un modèle à lui-même est 0
    7. La distance L² entre modèles différents est > 0

Note : nécessite la bibliothèque `pot` (pip install pot).
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
# Tests W₂² et RMD — indépendants du setup (OT sur les vecteurs de particules)
# =============================================================================


def test_rmd_identical():
    """RMD entre distributions identiques est 0."""
    particles = make_arbitrary_particles()
    r = rmd(particles, particles)
    assert r < TOL, f"RMD(µ, µ) = {r}, attendu 0"


def test_rmd_different():
    """RMD entre distributions différentes est > 0."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    a = init_particles_wi(k1, N=50)
    b = init_particles_wi(k2, N=50)
    r = rmd(a, b)
    assert r > 0.01, f"RMD devrait être > 0 pour des distributions différentes, obtenu {r}"


def test_rmd_bounded():
    """RMD est dans [0, 1]."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(1))
    a = init_particles_wi(k1, N=50)
    b = init_particles_wi(k2, N=50)
    r = rmd(a, b)
    assert 0 <= r <= 1.0, f"RMD = {r}, attendu dans [0, 1]"


def test_w2_nonnegative():
    """W₂² est non-négatif."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(2))
    a = init_particles_wi(k1, N=30)
    b = init_particles_wi(k2, N=30)
    assert w2_squared(a, b) >= 0


# =============================================================================
# rmd_to_projected — dépend du setup (project_EG dépend de SETUP)
# =============================================================================


def test_projected_closer_to_EG(any_setup):
    """Les particules projetées sur E^G ont RMD²_to_EG ≈ 0.

    Testé pour les deux setups car project_EG est setup-dépendant."""
    particles = init_particles_wi(jax.random.PRNGKey(3), N=50)

    rmd_before = rmd_to_projected(particles)
    projected = project_particles_EG(particles)
    rmd_after = rmd_to_projected(projected)

    assert rmd_after < TOL, (
        f"Setup {any_setup} : RMD² des particules projetées vers E^G = {rmd_after}"
    )
    assert rmd_before > rmd_after, (
        f"Setup {any_setup} : la projection devrait réduire la distance à E^G"
    )


# =============================================================================
# Distance L² — indépendante du setup (utilise forward_batch)
# =============================================================================


def test_l2_distance_self():
    """Distance L² d'un modèle à lui-même est 0."""
    particles = make_arbitrary_particles()
    d = l2_distance(particles, particles, jax.random.PRNGKey(4), n_samples=200)
    assert d < TOL, f"L²(f, f) = {d}, attendu 0"


def test_l2_distance_positive():
    """Distance L² entre modèles différents est > 0."""
    teacher = make_wi_particles()
    student = init_particles_wi(jax.random.PRNGKey(6), N=20)
    d = l2_distance(student, teacher, jax.random.PRNGKey(5), n_samples=200)
    assert d > 0.01, f"Distance L² devrait être > 0, obtenu {d}"
