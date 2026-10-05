"""
test_spaces.py — Tests for the mathematical building blocks in spaces.py.

Checks the key algebraic and geometric properties for both setups
(matrix and UV):
    1. The group action is an involution (g² = e in C_2)
    2. The projection onto E^G is idempotent
    3. The projection yields a fixed point of the group action
    4. The projection leaves elements already in E^G unchanged
    5. The basis of E^G is orthonormal (Frobenius inner product)
    6. The basis spans E^G (decomposition / reconstruction)
    7. σ* is jointly equivariant
    8. σ* has the right output shape

Setup-specific tests (explicit values):
    - test_action_on_z_explicit_matrix / _uv
    - test_sigma_star_explicit_matrix / _uv
"""

import jax
import jax.numpy as jnp
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spaces import (
    GROUP_ELEMENTS,
    action_on_x,
    action_on_z,
    project_EG,
    sigma_star,
    get_EG_basis,
)

TOL = 1e-6


# =============================================================================
# Algebraic properties — valid for both setups (any_setup)
# =============================================================================


def test_group_involution_on_x(any_setup):
    """Applying g twice to x gives back x (g² = e in C_2)."""
    key = jax.random.PRNGKey(0)
    g = GROUP_ELEMENTS[1]
    for _ in range(10):
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape=(2,))
        assert jnp.allclose(action_on_x(g, action_on_x(g, x)), x, atol=TOL)


def test_group_involution_on_z(any_setup):
    """Applying M_g twice to z gives back z."""
    key = jax.random.PRNGKey(1)
    g = GROUP_ELEMENTS[1]
    for _ in range(10):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, shape=(2, 2))
        assert jnp.allclose(action_on_z(g, action_on_z(g, z)), z, atol=TOL)


def test_identity_action(any_setup):
    """The identity element e leaves x and z unchanged."""
    key = jax.random.PRNGKey(2)
    e = GROUP_ELEMENTS[0]
    for _ in range(10):
        key, k1, k2 = jax.random.split(key, 3)
        x = jax.random.normal(k1, shape=(2,))
        z = jax.random.normal(k2, shape=(2, 2))
        assert jnp.allclose(action_on_x(e, x), x, atol=TOL)
        assert jnp.allclose(action_on_z(e, z), z, atol=TOL)


def test_projection_idempotent(any_setup):
    """P_{E^G}(P_{E^G}(z)) = P_{E^G}(z) for every z."""
    key = jax.random.PRNGKey(3)
    for _ in range(10):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, shape=(2, 2))
        pz = project_EG(z)
        assert jnp.allclose(project_EG(pz), pz, atol=TOL)


def test_projection_invariance(any_setup):
    """The projection yields a fixed point of the action: M_g(P(z)) = P(z)."""
    key = jax.random.PRNGKey(4)
    g = GROUP_ELEMENTS[1]
    for _ in range(10):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, shape=(2, 2))
        pz = project_EG(z)
        assert jnp.allclose(action_on_z(g, pz), pz, atol=TOL)


def test_projection_fixes_EG_elements(any_setup):
    """The projection does not move an element already in E^G."""
    e1, e2 = get_EG_basis()
    z_in_EG = 0.7 * e1 + (-0.3) * e2  # in E^G whatever the setup
    assert jnp.allclose(project_EG(z_in_EG), z_in_EG, atol=TOL)


def test_basis_orthonormality(any_setup):
    """The basis vectors e1, e2 of E^G are orthonormal (Frobenius)."""
    e1, e2 = get_EG_basis()
    norm_e1 = jnp.sqrt(jnp.trace(e1.T @ e1))
    norm_e2 = jnp.sqrt(jnp.trace(e2.T @ e2))
    inner = jnp.trace(e1.T @ e2)
    assert jnp.allclose(norm_e1, 1.0, atol=TOL), f"||e1|| = {norm_e1}"
    assert jnp.allclose(norm_e2, 1.0, atol=TOL), f"||e2|| = {norm_e2}"
    assert jnp.allclose(inner, 0.0, atol=TOL), f"<e1,e2> = {inner}"


def test_basis_spans_EG(any_setup):
    """Any linear combination of the basis lies in E^G, and decomposing an
    element of E^G on the basis reconstructs it exactly."""
    e1, e2 = get_EG_basis()
    g = GROUP_ELEMENTS[1]

    # An arbitrary linear combination lies in E^G
    z = 1.5 * e1 + (-0.8) * e2
    assert jnp.allclose(action_on_z(g, z), z, atol=TOL)

    # Decomposition of an element of E^G on the basis
    z_eg = 0.7 * e1 + (-0.3) * e2
    c1 = jnp.trace(e1.T @ z_eg)
    c2 = jnp.trace(e2.T @ z_eg)
    assert jnp.allclose(c1 * e1 + c2 * e2, z_eg, atol=TOL)


def test_sigma_star_joint_equivariance(any_setup):
    """σ*(ρ_g·x, M_g·z) = ρ̂_g·σ*(x, z) for every g, x, z.

    This is THE key property of the model."""
    key = jax.random.PRNGKey(5)
    for g in GROUP_ELEMENTS:
        for _ in range(20):
            key, k1, k2 = jax.random.split(key, 3)
            x = jax.random.normal(k1, shape=(2,))
            z = jax.random.normal(k2, shape=(2, 2))
            lhs = sigma_star(action_on_x(g, x), action_on_z(g, z))
            rhs = action_on_x(g, sigma_star(x, z))
            assert jnp.allclose(
                lhs, rhs, atol=TOL
            ), f"Equivariance failed for g={g}: LHS={lhs}, RHS={rhs}"


def test_sigma_star_output_shape(any_setup):
    """σ*(x, z): R² × R^{2×2} → R²."""
    x = jnp.array([1.0, -0.5])
    z = jnp.array([[0.3, 0.1], [-0.2, 0.4]])
    assert sigma_star(x, z).shape == (2,)


# =============================================================================
# Explicit tests specific to the matrix setup
# =============================================================================


def test_action_on_z_explicit_matrix(matrix_setup):
    """Matrix setup: M_g([[1,2],[3,4]]) = P@z@P = [[4,3],[2,1]]."""
    g = GROUP_ELEMENTS[1]
    z = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = action_on_z(g, z)
    expected = jnp.array([[4.0, 3.0], [2.0, 1.0]])
    assert jnp.allclose(result, expected, atol=TOL), f"Got: {result}"


def test_sigma_star_explicit_matrix(matrix_setup):
    """Matrix setup: σ*(x, z) = sigmoid(z@x).
    x=[1,0], z=[[2,0],[0,-1]] → sigmoid([2,0])."""
    x = jnp.array([1.0, 0.0])
    z = jnp.array([[2.0, 0.0], [0.0, -1.0]])
    expected = jax.nn.sigmoid(jnp.array([2.0, 0.0]))
    assert jnp.allclose(sigma_star(x, z), expected, atol=TOL)


def test_projection_matrix_explicit(matrix_setup):
    """Matrix setup: E^G = matrices of the form [[a,b],[b,a]] (fixed by z ↦ P@z@P).
    [[0.7,-0.3],[-0.3,0.7]] already lies in E^G."""
    z_sym = jnp.array([[0.7, -0.3], [-0.3, 0.7]])
    assert jnp.allclose(project_EG(z_sym), z_sym, atol=TOL)


def test_projection_matrix_non_symmetric(matrix_setup):
    """Matrix setup: projecting a matrix outside E^G averages it with its image under G."""
    z = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    pz = project_EG(z)
    # P(z) = (z + P@z@P) / 2
    # P@z@P = [[4,3],[2,1]], hence P(z) = [[2.5, 2.5],[2.5, 2.5]]
    expected = jnp.array([[2.5, 2.5], [2.5, 2.5]])
    assert jnp.allclose(pz, expected, atol=TOL), f"Got: {pz}"


# =============================================================================
# Explicit tests specific to the UV setup
# =============================================================================


def test_action_on_z_explicit_uv(uv_setup):
    """UV setup: M_g([[1,2],[3,4]]) = [[P@[1,2]], [P@[3,4]]] = [[2,1],[4,3]]."""
    g = GROUP_ELEMENTS[1]
    z = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = action_on_z(g, z)
    expected = jnp.array([[2.0, 1.0], [4.0, 3.0]])
    assert jnp.allclose(result, expected, atol=TOL), f"Got: {result}"


def test_sigma_star_explicit_uv(uv_setup):
    """UV setup: σ*(x, z) = v * sigmoid(uᵀx / d), with d = 2.
    x=[1,0], z=[[2,0],[0,-1]] → u=[2,0], v=[0,-1]
    → sigmoid((2*1 + 0*0) / 2) = sigmoid(1) ≈ 0.7311
    → output = [0, -sigmoid(1)]."""
    x = jnp.array([1.0, 0.0])
    z = jnp.array([[2.0, 0.0], [0.0, -1.0]])
    s = float(jax.nn.sigmoid(jnp.array(1.0)))
    expected = jnp.array([0.0, -s])
    assert jnp.allclose(sigma_star(x, z), expected, atol=TOL)


def test_EG_uv_structure(uv_setup):
    """UV setup: E^G = matrices with equal columns [[a,a],[b,b]].
    The projection averages the two columns."""
    z = jnp.array([[1.0, 3.0], [-2.0, 4.0]])
    pz = project_EG(z)
    # Averaged columns: each row becomes [(1+3)/2, ...] and [(-2+4)/2, ...]
    expected = jnp.array([[2.0, 2.0], [1.0, 1.0]])
    assert jnp.allclose(pz, expected, atol=TOL), f"Got: {pz}"


def test_projection_uv_fixes_equal_columns(uv_setup):
    """UV setup: a matrix with equal columns is fixed by the projection."""
    z_in_EG = jnp.array([[0.7, 0.7], [-0.3, -0.3]])
    assert jnp.allclose(project_EG(z_in_EG), z_in_EG, atol=TOL)
