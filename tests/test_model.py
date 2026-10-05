"""
test_model.py — Tests for the shallow model, the teachers and the loss.

Checks, for both setups (matrix and UV), that:
    1. Shapes are correct at every stage
    2. A student using the teacher's own particles reaches loss ≈ 0
    3. The WI teacher yields an equivariant function: f*(P·x) = P·f*(x)
    4. The SI teacher's particles lie in E^G
    5. The loss and the regularization are non-negative
    6. Data sampling produces the right shapes
    7. make_si_particles_uv() produces particles in E^G (UV only)
"""

import jax
import jax.numpy as jnp
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.spaces import GROUP_ELEMENTS, action_on_x, action_on_z, project_EG
from src.model import forward, forward_batch, forward_resnet
from src.teacher import (
    make_arbitrary_particles,
    make_wi_particles,
    make_si_particles,
    make_si_particles_uv,
    sample_data,
)
from src.loss import (
    quadratic_loss,
    quadratic_loss_batch,
    regularization,
    population_risk,
)
from src.training import init_particles_wi_resnet

TOL = 1e-5


# =============================================================================
# Shape tests — setup-independent
# =============================================================================


def test_particle_shapes():
    """Each teacher variant has the right shape."""
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
    """sample_data returns batches with consistent shapes."""
    particles = make_arbitrary_particles()
    x, y = sample_data(jax.random.PRNGKey(0), particles, n_samples=50)
    assert x.shape == (50, 2)
    assert y.shape == (50, 2)


# =============================================================================
# Loss tests — setup-independent
# =============================================================================


def test_loss_shapes():
    """The loss functions return scalars."""
    y_pred = jnp.array([0.5, 0.3])
    y_true = jnp.array([0.4, 0.6])
    assert quadratic_loss(y_pred, y_true).shape == ()
    assert quadratic_loss_batch(jnp.ones((10, 2)), jnp.zeros((10, 2))).shape == ()
    assert regularization(make_arbitrary_particles()).shape == ()


def test_loss_nonnegative():
    """The loss and the regularization are always >= 0."""
    key = jax.random.PRNGKey(10)
    k1, k2, k3 = jax.random.split(key, 3)
    assert quadratic_loss(jax.random.normal(k1, (2,)), jax.random.normal(k2, (2,))) >= 0
    assert regularization(jax.random.normal(k3, (10, 2, 2))) >= 0


def test_loss_zero_when_equal():
    """The loss is 0 when prediction = target."""
    y = jnp.array([0.42, -0.73])
    assert jnp.allclose(quadratic_loss(y, y), 0.0, atol=1e-8)


def test_loss_explicit():
    """ŷ = (1,0), y = (0,1) → ||ŷ-y||² = 2."""
    loss = quadratic_loss(jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0]))
    assert jnp.allclose(loss, 2.0, atol=1e-8)


# =============================================================================
# Self-prediction — tested for both setups
# =============================================================================


def test_self_prediction_loss_arbitrary(any_setup):
    """When the student uses the arbitrary teacher's particles, loss ≈ 0."""
    key = jax.random.PRNGKey(42)
    particles = make_arbitrary_particles()
    x, y = sample_data(key, particles, n_samples=100)
    loss = quadratic_loss_batch(forward_batch(x, particles), y)
    assert loss < TOL, f"Self-prediction loss (arbitrary) = {loss}"


def test_self_prediction_loss_wi(any_setup):
    """Same for the WI teacher."""
    key = jax.random.PRNGKey(42)
    particles = make_wi_particles()
    x, y = sample_data(key, particles, n_samples=100)
    loss = quadratic_loss_batch(forward_batch(x, particles), y)
    assert loss < TOL, f"Self-prediction loss (WI) = {loss}"


def test_self_prediction_loss_si(any_setup):
    """Same for the SI teacher (setup-dependent through make_si_particles)."""
    key = jax.random.PRNGKey(42)
    particles = make_si_particles()
    x, y = sample_data(key, particles, n_samples=100)
    loss = quadratic_loss_batch(forward_batch(x, particles), y)
    assert loss < TOL, f"Self-prediction loss (SI) = {loss}"


def test_population_risk_with_self(any_setup):
    """The population risk with student = teacher is ≈ τ·reg (data loss ≈ 0)."""
    key = jax.random.PRNGKey(11)
    tau = 1e-4
    particles = make_arbitrary_particles()
    x, y = sample_data(key, particles, n_samples=200)
    risk = population_risk(particles, x, y, tau=tau)
    expected_reg = tau * regularization(particles)
    assert jnp.allclose(risk, expected_reg, atol=TOL)


# =============================================================================
# Equivariance — tested for both setups
# =============================================================================


def test_wi_teacher_equivariance(any_setup):
    """The WI teacher is equivariant: f*(P·x) = P·f*(x)."""
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
    """The SI teacher is equivariant too (SI ⊂ WI as functions)."""
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
    """The arbitrary teacher is generally NOT equivariant (sanity check)."""
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
    assert found_difference, "The arbitrary teacher looks equivariant — unexpected!"


# =============================================================================
# SI particles lie in E^G — tested for both setups
# =============================================================================


def test_si_particles_in_EG(any_setup):
    """Every SI particle satisfies M_g·θ = θ and project_EG(θ) = θ."""
    particles = make_si_particles()
    g = GROUP_ELEMENTS[1]
    for i in range(particles.shape[0]):
        z = particles[i]
        assert jnp.allclose(
            action_on_z(g, z), z, atol=TOL
        ), f"SI particle {i} is not fixed by G"
        assert jnp.allclose(
            project_EG(z), z, atol=TOL
        ), f"SI particle {i} is moved by the projection"


# =============================================================================
# UV-setup-specific tests
# =============================================================================


def test_make_si_particles_uv_in_EG(uv_setup):
    """make_si_particles_uv() produces particles in E^G of the UV setup.
    E^G(UV) = matrices with equal columns [[a,a],[b,b]]."""
    particles = make_si_particles_uv()
    assert particles.shape == (5, 2, 2)
    g = GROUP_ELEMENTS[1]
    for i in range(particles.shape[0]):
        z = particles[i]
        assert jnp.allclose(
            action_on_z(g, z), z, atol=TOL
        ), f"make_si_particles_uv: particle {i} is not in E^G(UV)"
        # Concrete structure: equal columns
        assert jnp.allclose(
            z[:, 0], z[:, 1], atol=TOL
        ), f"Particle {i} does not have equal columns: {z}"


def test_si_particles_uv_structure(uv_setup):
    """make_si_particles() in the UV setup produces matrices with equal columns."""
    particles = make_si_particles()
    for i in range(particles.shape[0]):
        z = particles[i]
        assert jnp.allclose(
            z[:, 0], z[:, 1], atol=TOL
        ), f"SI particle {i} does not have equal columns in the UV setup: {z}"


def test_forward_resnet_output_shape_matrix(matrix_setup):
    """forward_resnet maps a (2,) input to a (2,) output."""
    particles = init_particles_wi_resnet(jax.random.PRNGKey(42), 3, 5)
    x = jnp.array([1.0, 2.0])
    assert forward_resnet(x, particles).shape == x.shape
