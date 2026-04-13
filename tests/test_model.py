"""
test_model.py — Tests pour le modèle shallow, les teachers et la loss.

Vérifie pour les deux setups (matrix et UV) :
    1. Les shapes sont corrects à chaque étape
    2. Teacher avec ses propres particules comme student → loss ≈ 0
    3. Le teacher WI produit une fonction équivariante : f*(P·x) = P·f*(x)
    4. Le teacher SI a ses particules dans E^G
    5. La loss et la régularisation sont non-négatives
    6. L'échantillonnage de données produit les bonnes shapes
    7. make_si_particles_uv() produit des particules dans E^G (UV uniquement)
"""

import jax
import jax.numpy as jnp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spaces import GROUP_ELEMENTS, action_on_x, action_on_z, project_EG
from src.model import forward, forward_batch
from src.teacher import (
    make_arbitrary_particles,
    make_wi_particles,
    make_si_particles,
    make_si_particles_uv,
    sample_data,
)
from src.loss import quadratic_loss, quadratic_loss_batch, regularization, population_risk

TOL = 1e-5


# =============================================================================
# Tests de shape — indépendants du setup
# =============================================================================


def test_particle_shapes():
    """Chaque variant de teacher produit la bonne shape."""
    assert make_arbitrary_particles().shape == (5, 2, 2)
    assert make_wi_particles().shape == (10, 2, 2)


def test_forward_shape():
    """forward(x, particles) → (2,)."""
    particles = make_arbitrary_particles()
    x = jnp.array([1.0, -0.5])
    assert forward(x, particles).shape == (2,)


def test_forward_batch_shape():
    """forward_batch(x_batch, particles) → (M, 2)."""
    particles = make_arbitrary_particles()
    x_batch = jax.random.normal(jax.random.PRNGKey(0), shape=(20, 2))
    assert forward_batch(x_batch, particles).shape == (20, 2)


def test_sample_data_shape():
    """sample_data retourne des batches de shapes cohérentes."""
    particles = make_arbitrary_particles()
    x, y = sample_data(jax.random.PRNGKey(0), particles, n_samples=50)
    assert x.shape == (50, 2)
    assert y.shape == (50, 2)


# =============================================================================
# Tests de loss — indépendants du setup
# =============================================================================


def test_loss_shapes():
    """Les fonctions de loss renvoient des scalaires."""
    y_pred = jnp.array([0.5, 0.3])
    y_true = jnp.array([0.4, 0.6])
    assert quadratic_loss(y_pred, y_true).shape == ()
    assert quadratic_loss_batch(jnp.ones((10, 2)), jnp.zeros((10, 2))).shape == ()
    assert regularization(make_arbitrary_particles()).shape == ()


def test_loss_nonnegative():
    """La loss et la régularisation sont toujours >= 0."""
    key = jax.random.PRNGKey(10)
    k1, k2, k3 = jax.random.split(key, 3)
    assert quadratic_loss(jax.random.normal(k1, (2,)), jax.random.normal(k2, (2,))) >= 0
    assert regularization(jax.random.normal(k3, (10, 2, 2))) >= 0


def test_loss_zero_when_equal():
    """La loss est 0 quand prédiction = cible."""
    y = jnp.array([0.42, -0.73])
    assert jnp.allclose(quadratic_loss(y, y), 0.0, atol=1e-8)


def test_loss_explicit():
    """ŷ = (1,0), y = (0,1) → ||ŷ-y||² = 2."""
    loss = quadratic_loss(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    assert jnp.allclose(loss, 2.0, atol=1e-8)


# =============================================================================
# Auto-prédiction — testés pour les deux setups
# =============================================================================


def test_self_prediction_loss_arbitrary(any_setup):
    """Quand le student utilise les particules du teacher arbitrary, loss ≈ 0."""
    key = jax.random.PRNGKey(42)
    particles = make_arbitrary_particles()
    x, y = sample_data(key, particles, n_samples=100)
    loss = quadratic_loss_batch(forward_batch(x, particles), y)
    assert loss < TOL, f"Loss auto-prédiction (arbitrary) = {loss}"


def test_self_prediction_loss_wi(any_setup):
    """Idem pour le teacher WI."""
    key = jax.random.PRNGKey(42)
    particles = make_wi_particles()
    x, y = sample_data(key, particles, n_samples=100)
    loss = quadratic_loss_batch(forward_batch(x, particles), y)
    assert loss < TOL, f"Loss auto-prédiction (WI) = {loss}"


def test_self_prediction_loss_si(any_setup):
    """Idem pour le teacher SI (dépend du setup via make_si_particles)."""
    key = jax.random.PRNGKey(42)
    particles = make_si_particles()
    x, y = sample_data(key, particles, n_samples=100)
    loss = quadratic_loss_batch(forward_batch(x, particles), y)
    assert loss < TOL, f"Loss auto-prédiction (SI) = {loss}"


def test_population_risk_with_self(any_setup):
    """Le risque populationnel avec student = teacher ≈ τ·reg (data loss ≈ 0)."""
    key = jax.random.PRNGKey(11)
    tau = 1e-4
    particles = make_arbitrary_particles()
    x, y = sample_data(key, particles, n_samples=200)
    risk = population_risk(particles, x, y, tau=tau)
    expected_reg = tau * regularization(particles)
    assert jnp.allclose(risk, expected_reg, atol=TOL)


# =============================================================================
# Équivariance — testés pour les deux setups
# =============================================================================


def test_wi_teacher_equivariance(any_setup):
    """Le teacher WI est équivariant : f*(P·x) = P·f*(x)."""
    key = jax.random.PRNGKey(7)
    particles = make_wi_particles()
    g = GROUP_ELEMENTS[1]
    for _ in range(20):
        key, subkey = jax.random.split(key)
        x = 4.0 * jax.random.normal(subkey, shape=(2,))
        lhs = forward(action_on_x(g, x), particles)
        rhs = action_on_x(g, forward(x, particles))
        assert jnp.allclose(lhs, rhs, atol=TOL), f"f*(Px)={lhs}, Pf*(x)={rhs}"


def test_si_teacher_equivariance(any_setup):
    """Le teacher SI est aussi équivariant (SI ⊂ WI fonctionnellement)."""
    key = jax.random.PRNGKey(8)
    particles = make_si_particles()
    g = GROUP_ELEMENTS[1]
    for _ in range(20):
        key, subkey = jax.random.split(key)
        x = 4.0 * jax.random.normal(subkey, shape=(2,))
        lhs = forward(action_on_x(g, x), particles)
        rhs = action_on_x(g, forward(x, particles))
        assert jnp.allclose(lhs, rhs, atol=TOL)


def test_arbitrary_teacher_not_equivariant(any_setup):
    """Le teacher arbitrary n'est généralement PAS équivariant (sanity check)."""
    key = jax.random.PRNGKey(9)
    particles = make_arbitrary_particles()
    g = GROUP_ELEMENTS[1]
    found_difference = False
    for _ in range(20):
        key, subkey = jax.random.split(key)
        x = 4.0 * jax.random.normal(subkey, shape=(2,))
        lhs = forward(action_on_x(g, x), particles)
        rhs = action_on_x(g, forward(x, particles))
        if not jnp.allclose(lhs, rhs, atol=1e-3):
            found_difference = True
            break
    assert found_difference, "Le teacher arbitrary semble équivariant — anormal !"


# =============================================================================
# Particules SI dans E^G — testés pour les deux setups
# =============================================================================


def test_si_particles_in_EG(any_setup):
    """Chaque particule SI vérifie M_g·θ = θ et project_EG(θ) = θ."""
    particles = make_si_particles()
    g = GROUP_ELEMENTS[1]
    for i in range(particles.shape[0]):
        z = particles[i]
        assert jnp.allclose(action_on_z(g, z), z, atol=TOL), (
            f"Particule SI {i} non fixée par G"
        )
        assert jnp.allclose(project_EG(z), z, atol=TOL), (
            f"Particule SI {i} déplacée par la projection"
        )


# =============================================================================
# Tests spécifiques au setup UV
# =============================================================================


def test_make_si_particles_uv_in_EG(uv_setup):
    """make_si_particles_uv() produit des particules dans E^G du setup UV.
    E^G(UV) = matrices à colonnes égales [[a,a],[b,b]]."""
    particles = make_si_particles_uv()
    assert particles.shape == (5, 2, 2)
    g = GROUP_ELEMENTS[1]
    for i in range(particles.shape[0]):
        z = particles[i]
        assert jnp.allclose(action_on_z(g, z), z, atol=TOL), (
            f"make_si_particles_uv : particule {i} non dans E^G(UV)"
        )
        # Structure concrète : colonnes égales
        assert jnp.allclose(z[:, 0], z[:, 1], atol=TOL), (
            f"Particule {i} n'a pas de colonnes égales : {z}"
        )


def test_si_particles_uv_structure(uv_setup):
    """make_si_particles() en setup UV produit des matrices à colonnes égales."""
    particles = make_si_particles()
    for i in range(particles.shape[0]):
        z = particles[i]
        assert jnp.allclose(z[:, 0], z[:, 1], atol=TOL), (
            f"Particule SI {i} n'a pas de colonnes égales en setup UV : {z}"
        )
