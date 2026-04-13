"""
test_sl_techniques.py — Tests pour les techniques de Symmetry Leveraging (DA, FA, EA).

Vérifie :
    1. sgd_step avec chaque loss_fn (da/fa/ea) conserve la shape des particules
    2. Chaque schéma met effectivement à jour les particules
    3. FA produit un modèle équivariant (propriété mathématique, tout setup)
    4. EA garde les particules dans E^G (avec init SI et bruit projeté)
    5. Tous les schémas réduisent la loss sur un court entraînement
"""

import jax
import jax.numpy as jnp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spaces import GROUP_ELEMENTS, action_on_x, action_on_z, project_EG
from src.model import forward
from src.teacher import make_wi_particles, make_arbitrary_particles, sample_data
from src.training import (
    DEFAULT_CONFIG,
    init_particles_wi,
    init_particles_si,
    sgd_step,
    train,
    loss_fn,
    loss_fn_da,
    loss_fn_fa,
    loss_fn_ea,
)

TOL = 1e-5


# =============================================================================
# Shape et fonctionnement de base
# =============================================================================


def test_da_step_shape(any_setup):
    """sgd_step avec loss_fn_da conserve la shape."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    particles = init_particles_wi(k1, N=20)
    x, y = sample_data(k2, make_wi_particles(), 20)
    new_particles = sgd_step(particles, x, y, k3, 50.0, 1e-4, 1e-6, loss_fn_da, False)
    assert new_particles.shape == (20, 2, 2)


def test_fa_step_shape(any_setup):
    """sgd_step avec loss_fn_fa conserve la shape."""
    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    particles = init_particles_wi(k1, N=20)
    x, y = sample_data(k2, make_wi_particles(), 20)
    new_particles = sgd_step(particles, x, y, k3, 50.0, 1e-4, 1e-6, loss_fn_fa, False)
    assert new_particles.shape == (20, 2, 2)


def test_ea_step_shape(any_setup):
    """sgd_step avec loss_fn_ea et bruit projeté conserve la shape."""
    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)
    particles = init_particles_si(k1, N=20)
    x, y = sample_data(k2, make_wi_particles(), 20)
    new_particles = sgd_step(particles, x, y, k3, 50.0, 1e-4, 1e-6, loss_fn_ea, True)
    assert new_particles.shape == (20, 2, 2)


def test_all_loss_fns_update_particles(any_setup):
    """Chaque loss_fn produit une mise à jour non nulle des particules."""
    key = jax.random.PRNGKey(3)
    teacher = make_wi_particles()
    schemes = [
        ("DA", loss_fn_da, init_particles_wi, False),
        ("FA", loss_fn_fa, init_particles_wi, False),
        ("EA", loss_fn_ea, init_particles_si, True),
    ]
    for name, used_loss_fn, init_fn, project_noise in schemes:
        k1, k2, k3, key = jax.random.split(key, 4)
        particles = init_fn(k1, N=20)
        x, y = sample_data(k2, teacher, 20)
        new_particles = sgd_step(
            particles, x, y, k3, 50.0, 1e-4, 1e-6, used_loss_fn, project_noise
        )
        diff = jnp.max(jnp.abs(new_particles - particles))
        assert diff > 1e-8, f"{name} : les particules n'ont pas été mises à jour"


# =============================================================================
# FA produit un modèle équivariant — testé pour les deux setups
# =============================================================================


def test_fa_model_equivariant(any_setup):
    """Le modèle symétrisé Q_G·Φ^N_θ est équivariant :
    (Q_G·Φ)(P·x) = P·(Q_G·Φ)(x).

    On vérifie la propriété mathématique directement sur les forward pass,
    sans dépendre de l'entraînement."""
    key = jax.random.PRNGKey(4)
    particles = init_particles_wi(jax.random.PRNGKey(4), N=50)
    g = GROUP_ELEMENTS[1]

    key = jax.random.PRNGKey(5)
    for _ in range(20):
        key, subkey = jax.random.split(key)
        x = 4.0 * jax.random.normal(subkey, shape=(2,))
        x_perm = action_on_x(g, x)

        # Q_G·Φ(x) = 0.5 * (Φ(x) + P·Φ(P·x))
        y_x = forward(x, particles)
        y_perm_back = action_on_x(g, forward(x_perm, particles))
        qg_at_x = 0.5 * (y_x + y_perm_back)

        # Q_G·Φ(P·x) = 0.5 * (Φ(P·x) + P·Φ(P·P·x)) = 0.5 * (Φ(P·x) + P·Φ(x))
        y_px = forward(x_perm, particles)
        y_back_px = action_on_x(g, forward(x, particles))
        qg_at_px = 0.5 * (y_px + y_back_px)

        expected = action_on_x(g, qg_at_x)
        assert jnp.allclose(
            qg_at_px, expected, atol=TOL
        ), f"FA équivariance : QΦ(Px)={qg_at_px}, P·QΦ(x)={expected}"


# =============================================================================
# EA garde les particules dans E^G — setup matrix (stabilité)
# =============================================================================


def test_ea_particles_stay_in_EG(matrix_setup):
    """Avec loss_fn_ea, init SI et β=0, les particules restent dans E^G
    (le gradient de loss_fn_ea est dans E^G via la projection)."""
    key = jax.random.PRNGKey(6)
    teacher = make_wi_particles()
    config = {**DEFAULT_CONFIG, "beta": 0.0, "T": 3.0, "gr": 3}

    history = train(
        teacher_particles=teacher,
        N=50,
        key=key,
        config=config,
        init_type="si",
        used_loss_fn=loss_fn_ea,
    )

    final_particles = history["particles"][-1]
    g = GROUP_ELEMENTS[1]
    deviations = jax.vmap(lambda z: jnp.max(jnp.abs(action_on_z(g, z) - z)))(
        final_particles
    )
    assert (
        jnp.max(deviations) < 0.01
    ), f"EA : particules hors de E^G, déviation max = {jnp.max(deviations):.6f}"


def test_da_stays_near_EG_with_equivariant_data(matrix_setup):
    """Avec DA, teacher WI et init SI, les particules restent proches de E^G
    (threshold généreux, phénomène large-N)."""
    key = jax.random.PRNGKey(7)
    teacher = make_wi_particles()
    config = {**DEFAULT_CONFIG, "beta": 0.0, "T": 3.0, "gr": 3}

    history = train(
        teacher_particles=teacher,
        N=100,
        key=key,
        config=config,
        init_type="si",
        used_loss_fn=loss_fn_da,
    )

    final_particles = history["particles"][-1]
    deviations = jax.vmap(lambda z: jnp.max(jnp.abs(project_EG(z) - z)))(
        final_particles
    )
    assert (
        jnp.mean(deviations) < 0.5
    ), f"DA trop loin de E^G : déviation moyenne = {jnp.mean(deviations):.4f}"


# =============================================================================
# Tous les schémas réduisent la loss — setup matrix + teacher arbitrary
# =============================================================================


def test_all_schemes_reduce_loss(matrix_setup):
    """DA, FA et EA réduisent la loss sur un court entraînement."""
    teacher = make_wi_particles()
    config = {**DEFAULT_CONFIG, "T": 5.0, "gr": 5}

    schemes = [
        ("DA", loss_fn_da, "wi"),
        ("FA", loss_fn_fa, "wi"),
        ("EA", loss_fn_ea, "si"),
    ]

    for name, used_loss_fn, init_type in schemes:
        key = jax.random.PRNGKey(hash(name) % 2**31)
        history = train(
            teacher_particles=teacher,
            N=50,
            key=key,
            config=config,
            init_type=init_type,
            used_loss_fn=used_loss_fn,
        )
        losses = history["losses"]
        assert len(losses) >= 2, f"{name} : pas assez d'enregistrements de loss"
        min_later = min(losses[len(losses) // 2 :])
        assert (
            min_later < losses[0]
        ), f"{name} : loss n'a pas diminué ({losses[0]:.6f} → {min_later:.6f})"


# =============================================================================
# Test UV : DA/FA/EA avec teacher arbitrary (stable en UV)
# =============================================================================


def test_all_schemes_uv_arbitrary_teacher(uv_setup):
    """DA, FA et EA fonctionnent en setup UV avec le teacher arbitrary (T court)."""
    teacher = make_arbitrary_particles()
    # T=2 et N=20 pour éviter l'explosion connue UV+WI+T long
    config = {**DEFAULT_CONFIG, "T": 2.0, "gr": 2}

    schemes = [
        ("DA", loss_fn_da, "wi"),
        ("FA", loss_fn_fa, "wi"),
        ("EA", loss_fn_ea, "si"),
    ]

    for name, used_loss_fn, init_type in schemes:
        key = jax.random.PRNGKey(hash(name) % 2**31)
        history = train(
            teacher_particles=teacher,
            N=20,
            key=key,
            config=config,
            init_type=init_type,
            used_loss_fn=used_loss_fn,
        )
        assert len(history["losses"]) >= 1, f"{name} UV : historique vide"
        # Vérifie qu'il n'y a pas de NaN/Inf dans les particules finales
        final = history["particles"][-1]
        assert jnp.all(jnp.isfinite(final)), f"{name} UV : particules non finies"
