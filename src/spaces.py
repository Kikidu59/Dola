"""
spaces.py — Mathematical building blocks for the mean-field symmetry experiments.

Configurable setup: swap between different activations by changing SETUP.

Setup "matrix": σ*(x, z) = σ(z·x),        z ∈ R^{2x2}, action M_g·z = g·z·gᵀ
Setup "uv":     σ*(x, z) = v·σ(uᵀx / d),  z = [u; v] stored as (2,2), action M_g·(u,v) = (g·u, g·v)

Usage (e.g. in a notebook):
    import src.spaces as spaces
    spaces.SETUP = spaces.make_setup_uv()    # switch to uv activation
    spaces.SETUP = spaces.make_setup_matrix() # switch back
"""

import jax
import jax.numpy as jnp
import jax.nn


# =============================================================================
# Group G = C_2 (these NEVER change regardless of setup)
# =============================================================================

GROUP_ELEMENTS = [
    jnp.eye(2),
    jnp.array([[0.0, 1.0], [1.0, 0.0]]),
]


def action_on_x(g, x):
    """Action of g ∈ G on x ∈ X = R² (also used for Y = R²)."""
    return g @ x


# =============================================================================
# Setup: matrix activation — σ*(x, z) = σ(z · x)
# =============================================================================


def make_setup_matrix():
    P = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    def sigma_star(x, z):
        return jax.nn.sigmoid(z @ x)

    def act_on_z(g, z):
        return g @ z @ g.T

    def proj_EG(z):
        return 0.5 * (z + P @ z @ P)

    basis = [
        (1.0 / jnp.sqrt(2.0)) * jnp.eye(2),
        (1.0 / jnp.sqrt(2.0)) * P,
    ]

    return {
        "name": "matrix",
        "sigma_star": sigma_star,
        "action_on_z": act_on_z,
        "project_EG": proj_EG,
        "EG_BASIS": basis,
    }


# =============================================================================
# Setup: uv activation — σ*(x, z) = v · σ(uᵀx / d), with d = 2
# z stored as (2,2): row 0 = u, row 1 = v
# =============================================================================


def make_setup_uv():
    P = jnp.array([[0.0, 1.0], [1.0, 0.0]])

    def sigma_star(x, z):
        u, v = z[0], z[1]
        return v * jax.nn.sigmoid(jnp.dot(u, x) / u.shape[0])

    def act_on_z(g, z):
        # M_g·(u, v) = (g·u, g·v)
        return jnp.stack([g @ z[0], g @ z[1]])

    def proj_EG(z):
        # Average over G: (1/2)(z + M_g·z)
        gz = jnp.stack([P @ z[0], P @ z[1]])
        return 0.5 * (z + gz)

    # E^G = {(u,v) : P·u = u, P·v = v}
    # Fixed points of P are vectors proportional to (1,1).
    # So E^G = {([[a,a],[b,b]]) : a,b ∈ R}, dimension 2.
    # Orthonormal basis (Frobenius norm):
    basis = [
        (1.0 / jnp.sqrt(2.0)) * jnp.array([[1.0, 1.0], [0.0, 0.0]]),
        (1.0 / jnp.sqrt(2.0)) * jnp.array([[0.0, 0.0], [1.0, 1.0]]),
    ]

    return {
        "name": "uv",
        "sigma_star": sigma_star,
        "action_on_z": act_on_z,
        "project_EG": proj_EG,
        "EG_BASIS": basis,
    }


# =============================================================================
# Active setup (default: uv — set it explicitly before running anything)
# =============================================================================

SETUP = make_setup_uv()


# =============================================================================
# Public API — these read from SETUP; the rest of the codebase only uses these
# =============================================================================


def sigma_star(x, z):
    return SETUP["sigma_star"](x, z)


def action_on_z(g, z):
    return SETUP["action_on_z"](g, z)


def project_EG(z):
    return SETUP["project_EG"](z)


def get_EG_basis():
    return SETUP["EG_BASIS"]
