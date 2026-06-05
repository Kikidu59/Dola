"""
training.py — SGD training for the shallow NN.

Core building blocks:
    - sgd_step:           one SGD/SGLD update, works with any loss function
    - init_particles_wi:  WI initialization (Gaussian)
    - init_particles_si:  SI initialization (Gaussian projected onto E^G)
    - train:              full training loop, accepts any loss_fn and
                          optional data transform (for DA)

The SL technique is NOT baked into the step function. Instead:
    - Vanilla: use loss_fn (default)
    - DA:      pass augment_fn to train (transforms data before gradient)
    - FA:      pass loss_fn_fa as loss_fn
    - EA:      pass loss_fn_ea as loss_fn
"""

import jax
import jax.numpy as jnp
from src.spaces import project_EG, GROUP_ELEMENTS, action_on_x
from src.model import forward_batch, forward_batch_resnet
from src.loss import quadratic_loss_batch, regularization, regularization_resnet
from src.teacher import sample_data, sample_data_resnet

# =============================================================================
# Default hyperparameters (from Appendix F, page 49)
# =============================================================================

DEFAULT_CONFIG = {
    "alpha": 50.0,
    "tau": 1e-4,
    "beta": 1e-6,
    "batch_size": 20,
    "T": 20.0,
    "gr": 5,
}

DEFAULT_CONFIG_RESNET = {
    "alpha": 5.0,
    "tau": 1e-4,
    "beta": 0.0,
    "alpha_arch": 1,
    "batch_size": 20,
    "T": 5.0,
    "gr": 5,
}


# =============================================================================
# Loss functions (vanilla, FA, EA)
# =============================================================================


def loss_fn(particles, x_batch, y_batch, tau):
    """Vanilla loss: ℓ(Φ^N_θ(x), y) + τ·reg."""
    y_pred = forward_batch(x_batch, particles)
    return quadratic_loss_batch(y_pred, y_batch) + tau * regularization(particles)


def loss_fn_fa(particles, x_batch, y_batch, tau):
    """FA loss: ℓ(Q_G · Φ^N_θ(x), y) + τ·reg.

    Q_G · Φ(x) = (1/2)[Φ(x) + P · Φ(P · x)]  for G = C_2.
    """
    g = GROUP_ELEMENTS[1]

    y_orig = forward_batch(x_batch, particles)
    x_perm = jax.vmap(lambda x: action_on_x(g, x))(x_batch)
    y_perm = forward_batch(x_perm, particles)
    y_perm_back = jax.vmap(lambda y: action_on_x(g, y))(y_perm)

    y_pred_sym = 0.5 * (y_orig + y_perm_back)
    return quadratic_loss_batch(y_pred_sym, y_batch) + tau * regularization(particles)


def loss_fn_da(particles, x_batch, y_batch, tau):
    """DA loss: average over all group elements (not sampled).

    L_DA = (1/|G|) Σ_{g∈G} L(g·x, g·y)
    For C_2: L_DA = (1/2)[L(x,y) + L(P·x, P·y)]
    """

    g = GROUP_ELEMENTS[1]

    # Loss on original data
    y_pred_orig = forward_batch(x_batch, particles)
    loss_orig = quadratic_loss_batch(y_pred_orig, y_batch)

    # Loss on transformed data
    x_perm = jax.vmap(lambda x: action_on_x(g, x))(x_batch)
    y_perm = jax.vmap(lambda y: action_on_x(g, y))(y_batch)
    y_pred_perm = forward_batch(x_perm, particles)
    loss_perm = quadratic_loss_batch(y_pred_perm, y_perm)

    return 0.5 * (loss_orig + loss_perm) + tau * regularization(particles)


def loss_fn_ea(particles, x_batch, y_batch, tau):
    """EA loss: ℓ(Φ^{N,EA}_θ(x), y) + τ·reg.

    Projects particles onto E^G before the forward pass.
    """
    particles_proj = jax.vmap(project_EG)(particles)
    y_pred = forward_batch(x_batch, particles_proj)
    return quadratic_loss_batch(y_pred, y_batch) + tau * regularization(particles_proj)


# ResNet version of the loss functions.


def loss_fn_resnet(particles, x_batch, y_batch, alpha_arch, tau):
    """Vanilla ResNet loss: ℓ(h_L(x), y). No regularization."""
    y_pred = forward_batch_resnet(x_batch, particles, alpha_arch)
    return quadratic_loss_batch(y_pred, y_batch) + tau * regularization_resnet(
        particles
    )


def loss_fn_fa_resnet(particles, x_batch, y_batch, alpha_arch, tau):
    """FA ResNet loss: symmetrizes the output via Q_G · h_L(x)."""
    g = GROUP_ELEMENTS[1]

    y_orig = forward_batch_resnet(x_batch, particles, alpha_arch)
    x_perm = jax.vmap(lambda x: action_on_x(g, x))(x_batch)
    y_perm = forward_batch_resnet(x_perm, particles, alpha_arch)
    y_perm_back = jax.vmap(lambda y: action_on_x(g, y))(y_perm)

    y_pred_sym = 0.5 * (y_orig + y_perm_back)
    return quadratic_loss_batch(y_pred_sym, y_batch) + tau * regularization_resnet(
        particles
    )


def loss_fn_da_resnet(particles, x_batch, y_batch, alpha_arch, tau):
    """DA ResNet loss: averages the loss over all G-transforms of the data."""
    g = GROUP_ELEMENTS[1]

    # Loss on original data
    y_pred_orig = forward_batch_resnet(x_batch, particles, alpha_arch)
    loss_orig = quadratic_loss_batch(y_pred_orig, y_batch)

    # Loss on transformed data
    x_perm = jax.vmap(lambda x: action_on_x(g, x))(x_batch)
    y_perm = jax.vmap(lambda y: action_on_x(g, y))(y_batch)
    y_pred_perm = forward_batch_resnet(x_perm, particles, alpha_arch)
    loss_perm = quadratic_loss_batch(y_pred_perm, y_perm)

    return 0.5 * (loss_orig + loss_perm) + tau * regularization_resnet(particles)


def loss_fn_ea_resnet(particles, x_batch, y_batch, alpha_arch, tau):
    """EA ResNet loss: projects all particles onto E^G before the forward pass.

    Requires double-vmap since particles has shape (L, M, 2, 2):
    outer vmap over L layers, inner vmap over M particles per layer.
    """
    particles_proj = jax.vmap(jax.vmap(project_EG))(particles)
    y_pred = forward_batch_resnet(x_batch, particles_proj, alpha_arch)
    return quadratic_loss_batch(y_pred, y_batch) + tau * regularization_resnet(
        particles_proj
    )


# def augment_data(x_batch, y_batch, key):
#     """DA: apply a random g ∈ G independently to EACH sample."""
#     from src.spaces import GROUP_ELEMENTS, action_on_x

#     B = x_batch.shape[0]
#     keys = jax.random.split(key, B)
#     g_indices = jax.vmap(lambda k: jax.random.randint(k, shape=(), minval=0, maxval=2))(
#         keys
#     )

#     def transform_one(x, y, g_idx):
#         g = jax.lax.cond(
#             g_idx == 1, lambda: GROUP_ELEMENTS[1], lambda: GROUP_ELEMENTS[0]
#         )
#         return action_on_x(g, x), action_on_x(g, y)

#     x_aug, y_aug = jax.vmap(transform_one)(x_batch, y_batch, g_indices)
#     return x_aug, y_aug


# =============================================================================
# Single SGD step
# =============================================================================


@jax.jit(static_argnames=["used_loss_fn", "project_noise"])
def sgd_step(
    particles, x_batch, y_batch, key, alpha, tau, beta, used_loss_fn, project_noise
):
    """One SGD/SGLD step. Works for ANY scheme.

    θ ← θ - α · ∇L + √(2β·α/N) · noise

    Args:
        particles:     (N, 2, 2) current parameters
        x_batch:       (B, 2) inputs (possibly augmented for DA)
        y_batch:       (B, 2) labels (possibly augmented for DA)
        key:           JAX random key for noise
        alpha:         learning rate
        tau:           regularization strength
        beta:          noise intensity
        used_loss_fn:  which loss to differentiate (loss_fn, loss_fn_fa, loss_fn_ea)
        project_noise: if True, project noise onto E^G (for SI init)

    Returns:
        (N, 2, 2) updated particles
    """
    N = particles.shape[0]
    grad = jax.grad(used_loss_fn, argnums=0)(particles, x_batch, y_batch, tau)

    noise = jax.random.normal(key, shape=particles.shape)
    noise = jax.lax.cond(
        project_noise,
        lambda n: jax.vmap(project_EG)(n),
        lambda n: n,
        noise,
    )
    noise_scale = jnp.sqrt(2 * beta * alpha / N)

    return particles - alpha * grad + noise_scale * noise


@jax.jit(
    static_argnames=[
        "used_loss_fn",
        "project_noise",
        "alpha",
        "tau",
        "alpha_arch",
        "beta",
    ]
)
def sgd_step_resnet(
    particles,
    x_batch,
    y_batch,
    key,
    alpha,
    tau,
    beta,
    alpha_arch,
    used_loss_fn,
    project_noise,
):
    """One SGLD step for the ResNet. No regularization.

    θ ← θ - α · ∇L(θ, alpha_arch) + √(2βα/M) · noise

    Args:
        particles:     (L, M, 2, 2) current parameters
        x_batch:       (B, 2) inputs
        y_batch:       (B, 2) labels
        key:           JAX random key
        alpha:         learning rate
        beta:          noise intensity
        tau:           regularization strenght
        alpha_arch:    architectural scaling constant
        used_loss_fn:  ResNet loss function to differentiate
        project_noise: if True, project noise onto E^G (for SI init)

    Returns:
        (L, M, 2, 2) updated particles
    """
    M = particles.shape[1]
    L = particles.shape[0]
    grad = jax.grad(used_loss_fn, argnums=0)(
        particles, x_batch, y_batch, alpha_arch, tau
    )
    if beta == 0.0:
        return particles - alpha * grad

    noise = jax.random.normal(key, shape=particles.shape)
    noise = jax.lax.cond(
        project_noise,
        lambda n: jax.vmap(jax.vmap(project_EG))(n),
        lambda n: n,
        noise,
    )
    noise_scale = jnp.sqrt(2 * alpha * beta / (M * L))

    return particles - alpha * grad + noise_scale * noise


# =============================================================================
# Student initialization
# =============================================================================


def init_particles_wi(key, N):
    """WI init: Z ~ N(0, 1/16), each particle is a 2x2 matrix."""
    return (1.0 / 4.0) * jax.random.normal(key, shape=(N, 2, 2))


def init_particles_si(key, N):
    """SI init: project N(0, 1/16) onto E^G."""
    raw = (1.0 / 4.0) * jax.random.normal(key, shape=(N, 2, 2))
    return jax.vmap(project_EG)(raw)


def init_particles_wi_resnet(key, L, M):
    """WI init for ResNet: Z ~ N(0, 1/16), shape (L, M, 2, 2)."""
    return (1.0 / 4.0) * jax.random.normal(key, shape=(L, M, 2, 2))


def init_particles_si_resnet(key, L, M):
    """SI init for ResNet: project N(0, 1/16) onto E^G, shape (L, M, 2, 2).

    Double-vmap: outer over L layers, inner over M particles per layer.
    """
    raw = (1.0 / 4.0) * jax.random.normal(key, shape=(L, M, 2, 2))
    return jax.vmap(jax.vmap(project_EG))(raw)


# =============================================================================
# Training loop
# =============================================================================


def train(teacher_particles, N, key, config=None, init_type="si", used_loss_fn=None):
    """Full training loop.

    Args:
        teacher_particles: (N*, 2, 2)
        N:                 number of student particles
        key:               JAX random key
        config:            hyperparameters (defaults to DEFAULT_CONFIG)
        init_type:         "si" or "wi"
        used_loss_fn:      loss function to use (default: vanilla loss_fn)
        augment_fn:        optional data transform, e.g. augment_data for DA.
                           Signature: augment_fn(x, y, key) → (x_aug, y_aug)

    Returns:
        dict with "particles", "losses", "steps"
    """
    if config is None:
        config = DEFAULT_CONFIG
    if used_loss_fn is None:
        used_loss_fn = loss_fn

    alpha = config["alpha"]
    tau = config["tau"]
    beta = config["beta"]
    B = config["batch_size"]
    T = config["T"]
    gr = config["gr"]

    Ne = int(N * T)
    save_every = max(1, Ne // gr)

    # Init
    key, init_key = jax.random.split(key)
    if init_type == "si":
        particles = init_particles_si(init_key, N)
    else:
        particles = init_particles_wi(init_key, N)

    # SI init → projected noise (Equation 5), WI → normal noise (Equation 1)
    project_noise = init_type == "si"

    history = {"particles": [particles], "losses": [], "steps": [0]}

    for k in range(Ne):
        key, data_key, noise_key = jax.random.split(key, 3)

        # Sample fresh minibatch
        x_batch, y_batch = sample_data(data_key, teacher_particles, B)

        # Optional data augmentation (for DA)
        # if augment_fn is not None:
        #     x_batch, y_batch = augment_fn(x_batch, y_batch, aug_key)

        # SGD step
        particles = sgd_step(
            particles,
            x_batch,
            y_batch,
            noise_key,
            alpha,
            tau,
            beta,
            used_loss_fn,
            project_noise,
        )

        # Save snapshots
        if (k + 1) % save_every == 0 or k == Ne - 1:
            key, eval_key = jax.random.split(key)
            x_eval, y_eval = sample_data(eval_key, teacher_particles, 100)
            loss_val = loss_fn(particles, x_eval, y_eval, tau)
            history["particles"].append(particles)
            history["losses"].append(float(loss_val))
            history["steps"].append(k + 1)

    return history


def train_resnet(
    teacher_particles, M, L, key, config=None, init_type="si", used_loss_fn=None
):
    """Full training loop for the ResNet.

    Args:
        teacher_particles: (N*, 2, 2) teacher particles
        M:                 number of particles per layer
        L:                 number of layers (depth)
        key:               JAX random key
        config:            hyperparameters (defaults to DEFAULT_CONFIG_RESNET)
        init_type:         "si" or "wi"
        used_loss_fn:      ResNet loss function (default: loss_fn_resnet)

    Returns:
        dict with "particles", "losses", "steps"
    """

    if config is None:
        config = DEFAULT_CONFIG_RESNET
    if used_loss_fn is None:
        used_loss_fn = loss_fn_resnet

    alpha = config["alpha"]
    alpha_arch = config["alpha_arch"]
    beta = config["beta"]
    tau = config["tau"]
    B = config["batch_size"]
    T = config["T"]
    gr = config["gr"]

    Ne = int(M * L * T)
    save_every = max(1, Ne // gr)

    # Init
    key, init_key = jax.random.split(key)
    if init_type == "si":
        particles = init_particles_si_resnet(init_key, L, M)
    else:
        particles = init_particles_wi_resnet(init_key, L, M)

    # SI init → projected noise (Equation 5), WI → normal noise (Equation 1)
    project_noise = init_type == "si"

    history = {"particles": [particles], "losses": [], "steps": [0]}

    for k in range(Ne):
        key, data_key, noise_key = jax.random.split(key, 3)

        # Sample fresh minibatch
        x_batch, y_batch = sample_data_resnet(
            data_key, teacher_particles, B, alpha_arch
        )

        # Optional data augmentation (for DA)
        # if augment_fn is not None:
        #     x_batch, y_batch = augment_fn(x_batch, y_batch, aug_key)

        # SGD step
        particles = sgd_step_resnet(
            particles,
            x_batch,
            y_batch,
            noise_key,
            alpha,
            tau,
            beta,
            alpha_arch,
            used_loss_fn,
            project_noise,
        )

        # Save snapshots
        if (k + 1) % save_every == 0 or k == Ne - 1:
            key, eval_key = jax.random.split(key)
            x_eval, y_eval = sample_data_resnet(
                eval_key, teacher_particles, 100, alpha_arch
            )
            loss_val = loss_fn_resnet(particles, x_eval, y_eval, alpha_arch, tau)
            history["particles"].append(particles)
            history["losses"].append(float(loss_val))
            history["steps"].append(k + 1)

    return history
