"""
test_training.py — Tests pour la boucle d'entraînement.

Vérifie :
    1. L'initialisation produit les bonnes shapes et propriétés
    2. Un step SGD met à jour les particules et conserve la shape
    3. L'initialisation SI place les particules dans E^G
    4. Le bruit projeté reste dans E^G
    5. La loss diminue sur un court entraînement
    6. La boucle train retourne un historique bien formé
"""

import jax
import jax.numpy as jnp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spaces import GROUP_ELEMENTS, action_on_z, project_EG
from src.teacher import make_wi_particles, make_arbitrary_particles
from src.training import (
    DEFAULT_CONFIG,
    loss_fn,
    sgd_step,
    init_particles_wi,
    init_particles_si,
    train,
)

TOL = 1e-5


# =============================================================================
# Tests d'initialisation
# =============================================================================


def test_init_wi_shape():
    """L'initialisation WI produit (N, 2, 2) particules."""
    particles = init_particles_wi(jax.random.PRNGKey(0), N=50)
    assert particles.shape == (50, 2, 2)


def test_init_si_shape():
    """L'initialisation SI produit (N, 2, 2) particules."""
    particles = init_particles_si(jax.random.PRNGKey(0), N=50)
    assert particles.shape == (50, 2, 2)


def test_init_si_in_EG(any_setup):
    """Chaque particule SI est dans E^G : M_g·θ = θ et project_EG(θ) = θ."""
    particles = init_particles_si(jax.random.PRNGKey(1), N=50)
    g = GROUP_ELEMENTS[1]
    for i in range(particles.shape[0]):
        z = particles[i]
        assert jnp.allclose(action_on_z(g, z), z, atol=TOL), (
            f"Particule SI {i} non fixée par G"
        )
        assert jnp.allclose(project_EG(z), z, atol=TOL), (
            f"Particule SI {i} déplacée par la projection"
        )


def test_init_wi_statistics():
    """L'initialisation WI est N(0, 1/16) : variance ≈ 1/16 = 0.0625."""
    particles = init_particles_wi(jax.random.PRNGKey(2), N=5000)
    var = jnp.var(particles)
    assert jnp.abs(var - 1.0 / 16.0) < 0.01, f"Variance = {var}"


# =============================================================================
# Tests d'un step SGD
# =============================================================================


def test_sgd_step_shape(any_setup):
    """sgd_step sans bruit projeté conserve la shape."""
    key = jax.random.PRNGKey(3)
    k1, k2, k3 = jax.random.split(key, 3)
    N = 20
    particles = init_particles_wi(k1, N)
    x_batch = 4.0 * jax.random.normal(k2, shape=(20, 2))
    y_batch = jax.random.normal(k2, shape=(20, 2))
    new_particles = sgd_step(particles, x_batch, y_batch, k3, 50.0, 1e-4, 1e-6, loss_fn, False)
    assert new_particles.shape == (N, 2, 2)


def test_sgd_step_projected_shape(any_setup):
    """sgd_step avec bruit projeté conserve la shape."""
    key = jax.random.PRNGKey(5)
    k1, k2, k3 = jax.random.split(key, 3)
    N = 20
    particles = init_particles_si(k1, N)
    x_batch = 4.0 * jax.random.normal(k2, shape=(20, 2))
    y_batch = jax.random.normal(k2, shape=(20, 2))
    new_particles = sgd_step(particles, x_batch, y_batch, k3, 50.0, 1e-4, 1e-6, loss_fn, True)
    assert new_particles.shape == (N, 2, 2)


def test_sgd_step_actually_updates(any_setup):
    """Après un step SGD, les particules doivent avoir changé."""
    key = jax.random.PRNGKey(4)
    k1, k2, k3 = jax.random.split(key, 3)
    particles = init_particles_wi(k1, N=20)
    x_batch, y_batch = jax.random.normal(k2, shape=(2, 20, 2))
    new_particles = sgd_step(particles, x_batch, y_batch, k3, 50.0, 1e-4, 1e-6, loss_fn, False)
    assert jnp.max(jnp.abs(new_particles - particles)) > 1e-8


def test_projected_noise_in_EG(any_setup):
    """Le bruit projeté via project_EG est bien dans E^G."""
    key = jax.random.PRNGKey(6)
    noise = jax.random.normal(key, shape=(50, 2, 2))
    noise_proj = jax.vmap(project_EG)(noise)
    g = GROUP_ELEMENTS[1]
    for i in range(50):
        assert jnp.allclose(action_on_z(g, noise_proj[i]), noise_proj[i], atol=TOL)


# =============================================================================
# Tests de la boucle train — setup matrix pour stabilité numérique
# =============================================================================


def test_loss_decreases(matrix_setup):
    """Sur un court entraînement, la loss doit diminuer."""
    key = jax.random.PRNGKey(7)
    teacher = make_wi_particles()
    config = {**DEFAULT_CONFIG, "T": 5.0, "gr": 5}
    history = train(teacher_particles=teacher, N=50, key=key, config=config, init_type="wi")
    losses = history["losses"]
    assert len(losses) >= 2
    assert losses[-1] < losses[0], (
        f"La loss n'a pas diminué : {losses[0]:.6f} → {losses[-1]:.6f}"
    )


def test_train_history_structure(matrix_setup):
    """train retourne un dict historique bien formé."""
    key = jax.random.PRNGKey(8)
    teacher = make_arbitrary_particles()
    config = {**DEFAULT_CONFIG, "T": 2.0, "gr": 2}
    N = 10
    history = train(teacher_particles=teacher, N=N, key=key, config=config, init_type="si")

    assert "particles" in history
    assert "losses" in history
    assert "steps" in history
    assert history["steps"][0] == 0
    for p in history["particles"]:
        assert p.shape == (N, 2, 2)
    assert len(history["losses"]) == len(history["steps"]) - 1


def test_train_si_init_stays_near_EG(matrix_setup):
    """Avec init SI et teacher WI (vanilla), les particules restent proches de E^G
    (propriété large-N, threshold généreux)."""
    key = jax.random.PRNGKey(9)
    teacher = make_wi_particles()
    config = {**DEFAULT_CONFIG, "beta": 0.0, "T": 2.0, "gr": 2}
    history = train(teacher_particles=teacher, N=100, key=key, config=config, init_type="si")
    final_particles = history["particles"][-1]
    g = GROUP_ELEMENTS[1]
    deviations = jax.vmap(lambda z: jnp.max(jnp.abs(action_on_z(g, z) - z)))(final_particles)
    assert jnp.mean(deviations) < 0.5, (
        f"Particules SI trop loin de E^G : déviation moyenne = {jnp.mean(deviations):.4f}"
    )


# =============================================================================
# Test setup UV — teacher arbitrary pour éviter l'instabilité UV+WI
# =============================================================================


def test_train_uv_arbitrary_teacher(uv_setup):
    """Entraînement court avec setup UV + teacher arbitrary : la loss doit diminuer."""
    key = jax.random.PRNGKey(42)
    teacher = make_arbitrary_particles()
    # T petit et N petit pour éviter l'explosion de v documentée avec WI+UV
    config = {**DEFAULT_CONFIG, "T": 2.0, "gr": 2}
    history = train(teacher_particles=teacher, N=20, key=key, config=config, init_type="si")
    losses = history["losses"]
    assert len(losses) >= 2
    assert losses[-1] < losses[0], (
        f"Loss UV (arbitrary) : {losses[0]:.6f} → {losses[-1]:.6f}"
    )
