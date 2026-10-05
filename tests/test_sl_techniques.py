"""
test_sl_techniques.py — Tests for the Symmetry Leveraging techniques (DA, FA, EA).

Checks that:
    1. sgd_step with each loss_fn (da/fa/ea) preserves the particle shape
    2. Each scheme actually updates the particles
    3. FA yields an equivariant model (mathematical property, any setup)
    4. EA keeps the particles in E^G (with SI init and projected noise)
    5. Every scheme reduces the loss over a short training run
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
# Shapes and basic behaviour
# =============================================================================


def test_da_step_shape(any_setup):
    """sgd_step with loss_fn_da preserves the shape."""
    key = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(key, 3)
    particles = init_particles_wi(k1, N=20)
    x, y = sample_data(k2, make_wi_particles(), 20)
    new_particles = sgd_step(particles, x, y, k3, 50.0, 1e-4, 1e-6, loss_fn_da, False)
    assert new_particles.shape == (20, 2, 2)


def test_fa_step_shape(any_setup):
    """sgd_step with loss_fn_fa preserves the shape."""
    key = jax.random.PRNGKey(1)
    k1, k2, k3 = jax.random.split(key, 3)
    particles = init_particles_wi(k1, N=20)
    x, y = sample_data(k2, make_wi_particles(), 20)
    new_particles = sgd_step(particles, x, y, k3, 50.0, 1e-4, 1e-6, loss_fn_fa, False)
    assert new_particles.shape == (20, 2, 2)


def test_ea_step_shape(any_setup):
    """sgd_step with loss_fn_ea and projected noise preserves the shape."""
    key = jax.random.PRNGKey(2)
    k1, k2, k3 = jax.random.split(key, 3)
    particles = init_particles_si(k1, N=20)
    x, y = sample_data(k2, make_wi_particles(), 20)
    new_particles = sgd_step(particles, x, y, k3, 50.0, 1e-4, 1e-6, loss_fn_ea, True)
    assert new_particles.shape == (20, 2, 2)


def test_all_loss_fns_update_particles(any_setup):
    """Each loss_fn produces a non-zero particle update."""
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
        assert diff > 1e-8, f"{name}: particles were not updated"


# =============================================================================
# FA yields an equivariant model — tested for both setups
# =============================================================================


def test_fa_model_equivariant(any_setup):
    """The symmetrized model Q_G·Φ^N_θ is equivariant:
    (Q_G·Φ)(P·x) = P·(Q_G·Φ)(x).

    The mathematical property is checked directly on forward passes,
    independently of training."""
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
        ), f"FA equivariance: QΦ(Px)={qg_at_px}, P·QΦ(x)={expected}"


# =============================================================================
# EA keeps the particles in E^G — matrix setup (stability)
# =============================================================================


def test_ea_particles_stay_in_EG(matrix_setup):
    """With loss_fn_ea, SI init and β=0, the particles stay in E^G
    (the gradient of loss_fn_ea lies in E^G thanks to the projection)."""
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
    ), f"EA: particles left E^G, max deviation = {jnp.max(deviations):.6f}"


def test_da_stays_near_EG_with_equivariant_data(matrix_setup):
    """With DA, a WI teacher and SI init, the particles stay close to E^G
    (generous threshold: this is a large-N phenomenon)."""
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
    ), f"DA too far from E^G: mean deviation = {jnp.mean(deviations):.4f}"


# =============================================================================
# Every scheme reduces the loss — matrix setup, WI teacher
# =============================================================================


def test_all_schemes_reduce_loss(matrix_setup):
    """DA, FA and EA reduce the loss over a short training run."""
    teacher = make_wi_particles()
    config = {**DEFAULT_CONFIG, "T": 5.0, "gr": 5}

    schemes = [
        ("DA", loss_fn_da, "wi"),
        ("FA", loss_fn_fa, "wi"),
        ("EA", loss_fn_ea, "si"),
    ]

    # Fixed per-scheme seeds (hash(name) is salted per process and not reproducible)
    for seed, (name, used_loss_fn, init_type) in enumerate(schemes):
        key = jax.random.PRNGKey(seed)
        history = train(
            teacher_particles=teacher,
            N=50,
            key=key,
            config=config,
            init_type=init_type,
            used_loss_fn=used_loss_fn,
        )
        losses = history["losses"]
        assert len(losses) >= 2, f"{name}: not enough loss records"
        min_later = min(losses[len(losses) // 2 :])
        assert (
            min_later < losses[0]
        ), f"{name}: loss did not decrease ({losses[0]:.6f} → {min_later:.6f})"


# =============================================================================
# UV setup: DA/FA/EA with the arbitrary teacher (stable in UV)
# =============================================================================


def test_all_schemes_uv_arbitrary_teacher(uv_setup):
    """DA, FA and EA run in the UV setup with the arbitrary teacher (short T)."""
    teacher = make_arbitrary_particles()
    # T=2 and N=20 to avoid the known blow-up with UV + WI + long T
    config = {**DEFAULT_CONFIG, "T": 2.0, "gr": 2}

    schemes = [
        ("DA", loss_fn_da, "wi"),
        ("FA", loss_fn_fa, "wi"),
        ("EA", loss_fn_ea, "si"),
    ]

    # Fixed per-scheme seeds (hash(name) is salted per process and not reproducible)
    for seed, (name, used_loss_fn, init_type) in enumerate(schemes):
        key = jax.random.PRNGKey(seed)
        history = train(
            teacher_particles=teacher,
            N=20,
            key=key,
            config=config,
            init_type=init_type,
            used_loss_fn=used_loss_fn,
        )
        assert len(history["losses"]) >= 1, f"{name} UV: empty history"
        # Check there is no NaN/Inf in the final particles
        final = history["particles"][-1]
        assert jnp.all(jnp.isfinite(final)), f"{name} UV: non-finite particles"
