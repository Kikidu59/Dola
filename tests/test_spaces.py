"""
test_spaces.py — Tests pour les briques mathématiques de spaces.py.

Vérifie les propriétés algébriques et géométriques clés pour les deux setups
(matrix et UV) :
    1. L'action de groupe est une involution (g² = e dans C_2)
    2. La projection sur E^G est idempotente
    3. La projection donne un point fixe de l'action de groupe
    4. La projection fixe les éléments déjà dans E^G
    5. La base de E^G est orthonormale (produit interne de Frobenius)
    6. La base engendre E^G (décomposition / recomposition)
    7. σ* est conjointement équivariante
    8. σ* a la bonne forme de sortie

Tests spécifiques à chaque setup (valeurs explicites) :
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
# Propriétés algébriques — valides pour les deux setups (any_setup)
# =============================================================================


def test_group_involution_on_x(any_setup):
    """Appliquer g deux fois sur x redonne x (g² = e dans C_2)."""
    key = jax.random.PRNGKey(0)
    g = GROUP_ELEMENTS[1]
    for _ in range(10):
        key, subkey = jax.random.split(key)
        x = jax.random.normal(subkey, shape=(2,))
        assert jnp.allclose(action_on_x(g, action_on_x(g, x)), x, atol=TOL)


def test_group_involution_on_z(any_setup):
    """Appliquer M_g deux fois sur z redonne z."""
    key = jax.random.PRNGKey(1)
    g = GROUP_ELEMENTS[1]
    for _ in range(10):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, shape=(2, 2))
        assert jnp.allclose(action_on_z(g, action_on_z(g, z)), z, atol=TOL)


def test_identity_action(any_setup):
    """L'élément identité e laisse x et z inchangés."""
    key = jax.random.PRNGKey(2)
    e = GROUP_ELEMENTS[0]
    for _ in range(10):
        key, k1, k2 = jax.random.split(key, 3)
        x = jax.random.normal(k1, shape=(2,))
        z = jax.random.normal(k2, shape=(2, 2))
        assert jnp.allclose(action_on_x(e, x), x, atol=TOL)
        assert jnp.allclose(action_on_z(e, z), z, atol=TOL)


def test_projection_idempotent(any_setup):
    """P_{E^G}(P_{E^G}(z)) = P_{E^G}(z) pour tout z."""
    key = jax.random.PRNGKey(3)
    for _ in range(10):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, shape=(2, 2))
        pz = project_EG(z)
        assert jnp.allclose(project_EG(pz), pz, atol=TOL)


def test_projection_invariance(any_setup):
    """La projection donne un point fixe de l'action : M_g(P(z)) = P(z)."""
    key = jax.random.PRNGKey(4)
    g = GROUP_ELEMENTS[1]
    for _ in range(10):
        key, subkey = jax.random.split(key)
        z = jax.random.normal(subkey, shape=(2, 2))
        pz = project_EG(z)
        assert jnp.allclose(action_on_z(g, pz), pz, atol=TOL)


def test_projection_fixes_EG_elements(any_setup):
    """La projection ne bouge pas un élément déjà dans E^G."""
    e1, e2 = get_EG_basis()
    z_in_EG = 0.7 * e1 + (-0.3) * e2  # toujours dans E^G quel que soit le setup
    assert jnp.allclose(project_EG(z_in_EG), z_in_EG, atol=TOL)


def test_basis_orthonormality(any_setup):
    """Les vecteurs de base e1, e2 de E^G sont orthonormaux (Frobenius)."""
    e1, e2 = get_EG_basis()
    norm_e1 = jnp.sqrt(jnp.trace(e1.T @ e1))
    norm_e2 = jnp.sqrt(jnp.trace(e2.T @ e2))
    inner = jnp.trace(e1.T @ e2)
    assert jnp.allclose(norm_e1, 1.0, atol=TOL), f"||e1|| = {norm_e1}"
    assert jnp.allclose(norm_e2, 1.0, atol=TOL), f"||e2|| = {norm_e2}"
    assert jnp.allclose(inner, 0.0, atol=TOL), f"<e1,e2> = {inner}"


def test_basis_spans_EG(any_setup):
    """Toute combinaison linéaire de la base est dans E^G, et la décomposition
    d'un élément de E^G via la base le reconstruit exactement."""
    e1, e2 = get_EG_basis()
    g = GROUP_ELEMENTS[1]

    # Une combinaison linéaire quelconque est dans E^G
    z = 1.5 * e1 + (-0.8) * e2
    assert jnp.allclose(action_on_z(g, z), z, atol=TOL)

    # Décomposition d'un élément de E^G dans la base
    z_eg = 0.7 * e1 + (-0.3) * e2
    c1 = jnp.trace(e1.T @ z_eg)
    c2 = jnp.trace(e2.T @ z_eg)
    assert jnp.allclose(c1 * e1 + c2 * e2, z_eg, atol=TOL)


def test_sigma_star_joint_equivariance(any_setup):
    """σ*(ρ_g·x, M_g·z) = ρ̂_g·σ*(x, z) pour tout g, x, z.

    C'est LA propriété clé du modèle."""
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
            ), f"Équivariance échouée pour g={g}: LHS={lhs}, RHS={rhs}"


def test_sigma_star_output_shape(any_setup):
    """σ*(x, z) : R² × R^{2×2} → R²."""
    x = jnp.array([1.0, -0.5])
    z = jnp.array([[0.3, 0.1], [-0.2, 0.4]])
    assert sigma_star(x, z).shape == (2,)


# =============================================================================
# Tests explicites spécifiques au setup matrix
# =============================================================================


def test_action_on_z_explicit_matrix(matrix_setup):
    """Setup matrix : M_g([[1,2],[3,4]]) = P@z@P = [[4,3],[2,1]]."""
    g = GROUP_ELEMENTS[1]
    z = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = action_on_z(g, z)
    expected = jnp.array([[4.0, 3.0], [2.0, 1.0]])
    assert jnp.allclose(result, expected, atol=TOL), f"Obtenu : {result}"


def test_sigma_star_explicit_matrix(matrix_setup):
    """Setup matrix : σ*(x, z) = sigmoid(z@x).
    x=[1,0], z=[[2,0],[0,-1]] → sigmoid([2,0])."""
    x = jnp.array([1.0, 0.0])
    z = jnp.array([[2.0, 0.0], [0.0, -1.0]])
    expected = jax.nn.sigmoid(jnp.array([2.0, 0.0]))
    assert jnp.allclose(sigma_star(x, z), expected, atol=TOL)


def test_projection_matrix_explicit(matrix_setup):
    """Setup matrix : E^G = matrices symétriques.
    [[0.7,-0.3],[-0.3,0.7]] est dans E^G (déjà symétrique)."""
    z_sym = jnp.array([[0.7, -0.3], [-0.3, 0.7]])
    assert jnp.allclose(project_EG(z_sym), z_sym, atol=TOL)


def test_projection_matrix_non_symmetric(matrix_setup):
    """Setup matrix : la projection d'une matrice non symétrique donne sa partie symétrique."""
    z = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    pz = project_EG(z)
    # E^G = sym : P(z) = (z + z^T) / 2 en termes des entrées...
    # En réalité P(z) = (z + P@z@P) / 2
    # P@z@P = [[4,3],[2,1]], donc P(z) = [[2.5, 2.5],[2.5, 2.5]]
    expected = jnp.array([[2.5, 2.5], [2.5, 2.5]])
    assert jnp.allclose(pz, expected, atol=TOL), f"Obtenu : {pz}"


# =============================================================================
# Tests explicites spécifiques au setup UV
# =============================================================================


def test_action_on_z_explicit_uv(uv_setup):
    """Setup UV : M_g([[1,2],[3,4]]) = [[P@[1,2]], [P@[3,4]]] = [[2,1],[4,3]]."""
    g = GROUP_ELEMENTS[1]
    z = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    result = action_on_z(g, z)
    expected = jnp.array([[2.0, 1.0], [4.0, 3.0]])
    assert jnp.allclose(result, expected, atol=TOL), f"Obtenu : {result}"


def test_sigma_star_explicit_uv(uv_setup):
    """Setup UV : σ*(x, z) = v * sigmoid(u^T x).
    x=[1,0], z=[[2,0],[0,-1]] → u=[2,0], v=[0,-1]
    → sigmoid(2*1 + 0*0) = sigmoid(2) ≈ 0.8808
    → sortie = [0, -sigmoid(2)]."""
    x = jnp.array([1.0, 0.0])
    z = jnp.array([[2.0, 0.0], [0.0, -1.0]])
    s = float(jax.nn.sigmoid(jnp.array(1.0)))
    expected = jnp.array([0.0, -s])
    assert jnp.allclose(sigma_star(x, z), expected, atol=TOL)


def test_EG_uv_structure(uv_setup):
    """Setup UV : E^G = matrices à colonnes égales [[a,a],[b,b]].
    La projection moyenne les deux colonnes."""
    z = jnp.array([[1.0, 3.0], [-2.0, 4.0]])
    pz = project_EG(z)
    # Colonnes moyennées : col = [(1+3)/2, (-2+4)/2] pour chaque ligne
    expected = jnp.array([[2.0, 2.0], [1.0, 1.0]])
    assert jnp.allclose(pz, expected, atol=TOL), f"Obtenu : {pz}"


def test_projection_uv_fixes_equal_columns(uv_setup):
    """Setup UV : une matrice à colonnes égales est fixée par la projection."""
    z_in_EG = jnp.array([[0.7, 0.7], [-0.3, -0.3]])
    assert jnp.allclose(project_EG(z_in_EG), z_in_EG, atol=TOL)
