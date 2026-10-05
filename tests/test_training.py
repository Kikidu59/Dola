"""
test_training.py — Tests for the training loop.

Checks that:
    1. Initialization produces the right shapes and properties
    2. One SGD step updates the particles and preserves the shape
    3. SI initialization places the particles in E^G
    4. Projected noise stays in E^G
    5. The loss decreases over a short training run
    6. The train loop returns a well-formed history
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
    DEFAULT_CONFIG_RESNET,
    loss_fn,
    sgd_step,
    init_particles_wi,
    init_particles_si,
    train,
    train_resnet,
)

TOL = 1e-5


# =============================================================================
# Initialization tests
# =============================================================================


def test_init_wi_shape():
    """WI initialization produces (N, 2, 2) particles."""
    particles = init_particles_wi(jax.random.PRNGKey(0), N=50)
    assert particles.shape == (50, 2, 2)


def test_init_si_shape():
    """SI initialization produces (N, 2, 2) particles."""
    particles = init_particles_si(jax.random.PRNGKey(0), N=50)
    assert particles.shape == (50, 2, 2)


def test_init_si_in_EG(any_setup):
    """Every SI particle lies in E^G: M_g·θ = θ and project_EG(θ) = θ."""
    particles = init_particles_si(jax.random.PRNGKey(1), N=50)
    g = GROUP_ELEMENTS[1]
    for i in range(particles.shape[0]):
        z = particles[i]
        assert jnp.allclose(action_on_z(g, z), z, atol=TOL), (
            f"SI particle {i} is not fixed by G"
        )
        assert jnp.allclose(project_EG(z), z, atol=TOL), (
            f"SI particle {i} is moved by the projection"
        )


def test_init_wi_statistics():
    """WI initialization is N(0, 1/16): variance ≈ 1/16 = 0.0625."""
    particles = init_particles_wi(jax.random.PRNGKey(2), N=5000)
    var = jnp.var(particles)
    assert jnp.abs(var - 1.0 / 16.0) < 0.01, f"Variance = {var}"


# =============================================================================
# Single SGD step tests
# =============================================================================


def test_sgd_step_shape(any_setup):
    """sgd_step without projected noise preserves the shape."""
    key = jax.random.PRNGKey(3)
    k1, k2, k3 = jax.random.split(key, 3)
    N = 20
    particles = init_particles_wi(k1, N)
    x_batch = 4.0 * jax.random.normal(k2, shape=(20, 2))
    y_batch = jax.random.normal(k2, shape=(20, 2))
    new_particles = sgd_step(particles, x_batch, y_batch, k3, 50.0, 1e-4, 1e-6, loss_fn, False)
    assert new_particles.shape == (N, 2, 2)


def test_sgd_step_projected_shape(any_setup):
    """sgd_step with projected noise preserves the shape."""
    key = jax.random.PRNGKey(5)
    k1, k2, k3 = jax.random.split(key, 3)
    N = 20
    particles = init_particles_si(k1, N)
    x_batch = 4.0 * jax.random.normal(k2, shape=(20, 2))
    y_batch = jax.random.normal(k2, shape=(20, 2))
    new_particles = sgd_step(particles, x_batch, y_batch, k3, 50.0, 1e-4, 1e-6, loss_fn, True)
    assert new_particles.shape == (N, 2, 2)


def test_sgd_step_actually_updates(any_setup):
    """After one SGD step, the particles must have changed."""
    key = jax.random.PRNGKey(4)
    k1, k2, k3 = jax.random.split(key, 3)
    particles = init_particles_wi(k1, N=20)
    x_batch, y_batch = jax.random.normal(k2, shape=(2, 20, 2))
    new_particles = sgd_step(particles, x_batch, y_batch, k3, 50.0, 1e-4, 1e-6, loss_fn, False)
    assert jnp.max(jnp.abs(new_particles - particles)) > 1e-8


def test_projected_noise_in_EG(any_setup):
    """Noise projected with project_EG lies in E^G."""
    key = jax.random.PRNGKey(6)
    noise = jax.random.normal(key, shape=(50, 2, 2))
    noise_proj = jax.vmap(project_EG)(noise)
    g = GROUP_ELEMENTS[1]
    for i in range(50):
        assert jnp.allclose(action_on_z(g, noise_proj[i]), noise_proj[i], atol=TOL)


# =============================================================================
# Training loop tests — matrix setup for numerical stability
# =============================================================================


def test_loss_decreases(matrix_setup):
    """The loss must decrease over a short training run."""
    key = jax.random.PRNGKey(7)
    teacher = make_wi_particles()
    config = {**DEFAULT_CONFIG, "T": 5.0, "gr": 5}
    history = train(teacher_particles=teacher, N=50, key=key, config=config, init_type="wi")
    losses = history["losses"]
    assert len(losses) >= 2
    assert losses[-1] < losses[0], (
        f"Loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
    )


def test_train_history_structure(matrix_setup):
    """train returns a well-formed history dict."""
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
    """With SI init and a WI teacher (vanilla), the particles stay close to E^G
    (large-N property, generous threshold)."""
    key = jax.random.PRNGKey(9)
    teacher = make_wi_particles()
    config = {**DEFAULT_CONFIG, "beta": 0.0, "T": 2.0, "gr": 2}
    history = train(teacher_particles=teacher, N=100, key=key, config=config, init_type="si")
    final_particles = history["particles"][-1]
    g = GROUP_ELEMENTS[1]
    deviations = jax.vmap(lambda z: jnp.max(jnp.abs(action_on_z(g, z) - z)))(final_particles)
    assert jnp.mean(deviations) < 0.5, (
        f"SI particles too far from E^G: mean deviation = {jnp.mean(deviations):.4f}"
    )


# =============================================================================
# UV setup — arbitrary teacher, to avoid the UV + WI instability
# =============================================================================


def test_train_uv_arbitrary_teacher(uv_setup):
    """Short training run with UV setup + arbitrary teacher: the loss must decrease."""
    key = jax.random.PRNGKey(42)
    teacher = make_arbitrary_particles()
    # Small T and small N to avoid the documented blow-up of v with WI + UV
    config = {**DEFAULT_CONFIG, "T": 2.0, "gr": 2}
    history = train(teacher_particles=teacher, N=20, key=key, config=config, init_type="si")
    losses = history["losses"]
    assert len(losses) >= 2
    assert losses[-1] < losses[0], (
        f"UV loss (arbitrary): {losses[0]:.6f} → {losses[-1]:.6f}"
    )


# =============================================================================
# ResNet smoke tests
# =============================================================================


def test_train_resnet_history_structure(matrix_setup):
    """train_resnet returns a well-formed history, without crashing."""
    key = jax.random.PRNGKey(10)
    teacher = make_arbitrary_particles()
    L, M = 3, 10
    config = {**DEFAULT_CONFIG_RESNET, "T": 1.0, "gr": 2}
    history = train_resnet(teacher_particles=teacher, M=M, L=L, key=key, config=config, init_type="wi")

    assert "particles" in history
    assert "losses" in history
    assert "steps" in history
    assert history["steps"][0] == 0
    for p in history["particles"]:
        assert p.shape == (L, M, 2, 2)
    assert len(history["losses"]) == len(history["steps"]) - 1
    assert all(jnp.isfinite(jnp.array(history["losses"]))), "Non-finite loss detected"


def test_train_resnet_loss_decreases(matrix_setup):
    """The loss must decrease over a short ResNet training run."""
    key = jax.random.PRNGKey(11)
    teacher = make_wi_particles()
    config = {**DEFAULT_CONFIG_RESNET, "T": 3.0, "gr": 3}
    history = train_resnet(teacher_particles=teacher, M=20, L=3, key=key, config=config, init_type="wi")
    losses = history["losses"]
    assert losses[-1] < losses[0], (
        f"ResNet loss did not decrease: {losses[0]:.6f} → {losses[-1]:.6f}"
    )
