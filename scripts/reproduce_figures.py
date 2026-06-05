"""
reproduce_figures.py — Reproducible regeneration of the project figures.

Regenerates, with fixed random seeds and N_r repetitions for error bars, the two
families of figures used in the report (all under SI initialization):

  (1) RMD^2 to E^G at the end of training, as a function of network width/depth,
      for every architecture (NN, ResNet), both activation setups (matrix, uv),
      both teachers (WI, arbitrary), and every SL technique (vanilla, DA, FA, EA).
        - NN:     swept over N = [5, 10, 50, 100, 500, 1000]
        - ResNet: swept over L = [1, 10, 50, 100, 500, 1000] with width M = 5
      For the ResNet, the plotted quantity is the RMD^2 to E^G averaged over the
      L layers, i.e. mean_l RMD^2(nu_l, (nu_l)^{E^G}).

  (2) Training loss vs SGD step, for the same configurations, at
        - NN:     N = 500 particles
        - ResNet: (M = 5, L = 500)

All hyperparameters come straight from the configs defined in src/training.py.
The config is chosen per (architecture, setup):
  - NN / matrix : DEFAULT_CONFIG        (alpha = 50)
  - NN / uv     : DEFAULT_CONFIG_UV     (alpha = 5, tau = 0) — the (u,v) unit is
                  numerically unstable with the matrix-tuned alpha = 50.
  - ResNet      : DEFAULT_CONFIG_RESNET (both setups).
The SGD horizon is pinned to T = 20 (NN) and T = 5 (ResNet). Figures are saved as
PDF (English labels).

Seeds: for repetition `rep` at width/depth `size`, the base key is
PRNGKey(rep * 1000 + size); each technique then uses fold_in(base, technique_idx).
This makes every figure deterministic and reproducible.

Usage:
    python scripts/reproduce_figures.py                  # full run (heavy)
    python scripts/reproduce_figures.py --reps 10        # explicit N_r
    python scripts/reproduce_figures.py --outdir figures # output directory
    python scripts/reproduce_figures.py --quick          # tiny smoke test
    python scripts/reproduce_figures.py --skip-loss      # only RMD figures
    python scripts/reproduce_figures.py --skip-rmd       # only loss figures

Note on runtime: the full sweep trains
    2 setups x 2 architectures x 2 teachers x 4 techniques x 6 sizes x N_r
runs for the RMD figures (plus the loss-vs-step runs). With N or L up to 1000 and
T = 20 / T = 5, this is a heavy computation and is meant to be run on a machine
with enough time/compute, not interactively.
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")  # headless backend: write PDFs without a display

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

# Make `src` importable when the script is run from anywhere.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import src.spaces as spaces
from src.training import (
    train,
    train_resnet,
    loss_fn,
    loss_fn_da,
    loss_fn_fa,
    loss_fn_ea,
    loss_fn_resnet,
    loss_fn_da_resnet,
    loss_fn_fa_resnet,
    loss_fn_ea_resnet,
    sgd_step,
    sgd_step_resnet,
    init_particles_si,
    init_particles_wi,
    init_particles_si_resnet,
    init_particles_wi_resnet,
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_UV,
    DEFAULT_CONFIG_RESNET,
)
from src.teacher import (
    make_wi_particles,
    make_arbitrary_particles,
    make_wi_particles_resnet,
    make_arbitrary_particles_resnet,
    sample_data,
    sample_data_resnet,
)
from src.metrics import rmd_to_projected
from src.utils import plot_rmd_curves, plot_training_curves

# =============================================================================
# Experiment specification
# =============================================================================

NN_SIZES = [10, 50, 100, 500, 1000]  # NN: number of particles N
RESNET_DEPTHS = [10, 50, 100, 500, 1000]  # ResNet: depth L
RESNET_WIDTH_M = 5  # ResNet: particles per layer M

NN_LOSS_N = 500  # loss-vs-step: NN width
RESNET_LOSS_L = 500  # loss-vs-step: ResNet depth

# Number of points sampled on each loss-vs-step curve. The loss is logged on a
# fine grid that starts at the very first SGD step (so the early decay is
# visible), unlike train()/train_resnet() whose default granularity only records
# the loss every Ne/gr steps.
LOSS_CURVE_POINTS = 200

N_REPS = 10  # repetitions (error bars)

SETUPS = ["matrix", "uv"]
TEACHERS = ["WI", "arbitrary"]

# Technique name -> (NN loss fn, ResNet loss fn, fold-in index for the RNG).
# The fold-in index keeps the per-technique seeds distinct but reproducible.
TECHNIQUES = {
    "vanilla": (loss_fn, loss_fn_resnet, 0),
    "DA": (loss_fn_da, loss_fn_da_resnet, 1),
    "FA": (loss_fn_fa, loss_fn_fa_resnet, 2),
    "EA": (loss_fn_ea, loss_fn_ea_resnet, 3),
}

TEACHER_LABEL = {"WI": "WI teacher", "arbitrary": "arbitrary teacher"}


# =============================================================================
# Configs (taken as-is from src/training.py; only T is pinned explicitly)
# =============================================================================


def config_for(arch, setup_name, T):
    """Hyperparameter config for a given architecture AND activation setup.

    The right config depends on the setup, not just the architecture:
      - NN / matrix : DEFAULT_CONFIG       (alpha = 50)
      - NN / uv     : DEFAULT_CONFIG_UV    (alpha = 5, tau = 0) — the (u,v) unit
                      is unstable with the matrix-tuned alpha = 50.
      - ResNet      : DEFAULT_CONFIG_RESNET for BOTH setups (its residual blocks
                      already use a small learning rate; the uv ResNet notebook
                      did not override alpha/tau either).
    The SGD horizon T is pinned explicitly.
    """
    if arch == "NN":
        base = DEFAULT_CONFIG_UV if setup_name == "uv" else DEFAULT_CONFIG
    else:  # ResNet — same config for matrix and uv
        base = DEFAULT_CONFIG_RESNET
    return {**base, "T": T}


# =============================================================================
# Setup handling
# =============================================================================


def set_setup(name):
    """Activate the requested activation setup ('matrix' or 'uv').

    The single-SGD-step functions in training.py are jitted and bake in the
    active activation at trace time, but their compilation cache key does NOT
    include the setup. After changing spaces.SETUP we therefore clear the JAX
    caches so they re-trace under the new activation. We only ever switch setups
    at a block boundary (never inside a setup), so this is both correct and cheap.
    """
    if name == "matrix":
        spaces.SETUP = spaces.make_setup_matrix()
    elif name == "uv":
        spaces.SETUP = spaces.make_setup_uv()
    else:
        raise ValueError(f"unknown setup: {name!r}")
    if hasattr(jax, "clear_caches"):
        jax.clear_caches()


def make_teacher(arch, kind):
    """Build the teacher particles for the given architecture and teacher kind.

    Following the figure notebooks, the ResNet teacher is a single-layer
    (depth-1) teacher whose target function the variable-depth student must fit.
    Teachers must be built AFTER the setup is active: the WI teacher uses the
    group action, which depends on the active setup.
    """
    if arch == "NN":
        return make_wi_particles() if kind == "WI" else make_arbitrary_particles()
    # ResNet
    return (
        make_wi_particles_resnet(1)
        if kind == "WI"
        else make_arbitrary_particles_resnet(1)
    )


# =============================================================================
# Metrics / training wrappers
# =============================================================================


def rmd_value(arch, final_particles):
    """RMD^2 to E^G of the final-snapshot particles.

    - NN:     RMD^2(nu_N, (nu_N)^{E^G}) over the N particles.
    - ResNet: mean over the L layers of RMD^2(nu_l, (nu_l)^{E^G}).
    """
    if arch == "NN":
        return float(rmd_to_projected(final_particles))
    return float(
        jnp.mean(jnp.array([rmd_to_projected(layer) for layer in final_particles]))
    )


def train_one(arch, teacher, size, key, config, nn_lf, rn_lf):
    """Run a single training and return its full history dict."""
    if arch == "NN":
        return train(teacher, size, key, config, "si", used_loss_fn=nn_lf)
    return train_resnet(
        teacher, RESNET_WIDTH_M, size, key, config, "si", used_loss_fn=rn_lf
    )


# =============================================================================
# Sweeps
# =============================================================================


def rmd_sweep(arch, setup_name, kind, sizes, config, n_reps):
    """RMD^2 to E^G at end of training vs width/depth, for all techniques.

    Returns a dict technique -> list (per size) of list (per rep) of RMD values,
    which is exactly the structure plot_rmd_curves expects.
    """
    teacher = make_teacher(arch, kind)
    results = {t: [] for t in TECHNIQUES}
    for size in sizes:
        for t in TECHNIQUES:
            results[t].append([])
        for rep in range(n_reps):
            base = jax.random.PRNGKey(rep * 1000 + size)
            for t, (nn_lf, rn_lf, idx) in TECHNIQUES.items():
                h = train_one(
                    arch,
                    teacher,
                    size,
                    jax.random.fold_in(base, idx),
                    config,
                    nn_lf,
                    rn_lf,
                )
                results[t][-1].append(rmd_value(arch, h["particles"][-1]))
        print(f"    [{arch}/{setup_name}/{kind}] size={size} done")
    return results


def loss_history_run(arch, teacher, size, key, config, used_loss_fn, n_points):
    """Train exactly like train()/train_resnet() but log the loss on a fine grid
    that STARTS at the first SGD step.

    This mirrors the training loop of src/training.py (same init, same per-step
    SGLD update, same SI-projected noise, same vanilla loss used for evaluation),
    with two differences tailored to the loss-vs-step figures:
      - the loss is recorded at the very first step (k = 0) and then on an evenly
        spaced grid of about ``n_points`` steps, instead of only every Ne/gr steps;
      - particle snapshots are NOT stored, so logging densely stays cheap.

    Returns {"steps": [0, s_1, ..., Ne], "losses": [...]}, compatible with
    plot_training_curves (which drops the leading step 0 and aligns the rest with
    the recorded losses).
    """
    alpha = config["alpha"]
    tau = config["tau"]
    beta = config["beta"]
    B = config["batch_size"]
    T = config["T"]
    is_resnet = arch == "ResNet"
    alpha_arch = config["alpha_arch"] if is_resnet else None

    if is_resnet:
        Ne = int(RESNET_WIDTH_M * size * T)
    else:
        Ne = int(size * T)
    stride = max(1, Ne // n_points)

    key, init_key = jax.random.split(key)
    if is_resnet:
        particles = init_particles_si_resnet(init_key, size, RESNET_WIDTH_M)
    else:
        particles = init_particles_si(init_key, size)
    project_noise = True  # SI init throughout these experiments

    history = {"steps": [0], "losses": []}
    for k in range(Ne):
        key, data_key, noise_key = jax.random.split(key, 3)
        if is_resnet:
            x_batch, y_batch = sample_data_resnet(data_key, teacher, B, alpha_arch)
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
        else:
            x_batch, y_batch = sample_data(data_key, teacher, B)
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

        if k == 0 or (k + 1) % stride == 0 or k == Ne - 1:
            key, eval_key = jax.random.split(key)
            if is_resnet:
                x_eval, y_eval = sample_data_resnet(eval_key, teacher, 100, alpha_arch)
                loss_val = loss_fn_resnet(particles, x_eval, y_eval, alpha_arch, tau)
            else:
                x_eval, y_eval = sample_data(eval_key, teacher, 100)
                loss_val = loss_fn(particles, x_eval, y_eval, tau)
            history["steps"].append(k + 1)
            history["losses"].append(float(loss_val))

    return history


def loss_sweep(arch, setup_name, kind, size, config, n_reps):
    """Collect loss-vs-step histories (logged from the first SGD step) for all
    techniques.

    Returns a dict technique -> list of N_r history dicts, the structure
    plot_training_curves expects.
    """
    teacher = make_teacher(arch, kind)
    histories = {t: [] for t in TECHNIQUES}
    for rep in range(n_reps):
        base = jax.random.PRNGKey(rep * 1000 + size)
        for t, (nn_lf, rn_lf, idx) in TECHNIQUES.items():
            used_loss_fn = rn_lf if arch == "ResNet" else nn_lf
            h = loss_history_run(
                arch,
                teacher,
                size,
                jax.random.fold_in(base, idx),
                config,
                used_loss_fn,
                LOSS_CURVE_POINTS,
            )
            histories[t].append(h)
    print(f"    [{arch}/{setup_name}/{kind}] loss curves (size={size}) done")
    return histories


# =============================================================================
# Main driver
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reps", type=int, default=N_REPS, help="number of repetitions (N_r)"
    )
    parser.add_argument(
        "--outdir",
        default=os.path.join(os.path.dirname(__file__), "..", "figures"),
        help="output directory for the PDF figures",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny smoke test (reduced sizes/reps/T) — for validation only",
    )
    parser.add_argument(
        "--skip-loss", action="store_true", help="skip the loss-vs-step figures"
    )
    parser.add_argument(
        "--skip-rmd", action="store_true", help="skip the RMD-vs-size figures"
    )
    # Subset selection — lets a SLURM job array run one (setup, arch, teacher)
    # block per task. Each block produces its own self-contained PDF(s), so the
    # work is embarrassingly parallel. Defaults = run everything.
    parser.add_argument(
        "--setup", choices=SETUPS, default=None, help="run only this setup"
    )
    parser.add_argument(
        "--arch", choices=["NN", "ResNet"], default=None, help="run only this architecture"
    )
    parser.add_argument(
        "--teacher", choices=TEACHERS, default=None, help="run only this teacher"
    )
    args = parser.parse_args()

    setups = [args.setup] if args.setup else SETUPS
    teachers = [args.teacher] if args.teacher else TEACHERS

    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    if args.quick:
        nn_sizes, resnet_depths = [5, 10], [1, 5]
        nn_loss_n, resnet_loss_l = 10, 5
        reps = min(args.reps, 2)
        nn_T, rn_T = 1.0, 0.5
    else:
        nn_sizes, resnet_depths = NN_SIZES, RESNET_DEPTHS
        nn_loss_n, resnet_loss_l = NN_LOSS_N, RESNET_LOSS_L
        reps = args.reps
        nn_T, rn_T = 20.0, 5.0

    # (arch, sweep sizes, x-axis label, loss-curve size, horizon T)
    # The config itself is resolved per (arch, setup) inside the loop below.
    archs = [
        ("NN", nn_sizes, "Number of particles (N)", nn_loss_n, nn_T),
        ("ResNet", resnet_depths, "ResNet depth (L)", resnet_loss_l, rn_T),
    ]
    if args.arch:
        archs = [a for a in archs if a[0] == args.arch]

    for setup_name in setups:
        set_setup(setup_name)
        print(f"\n=== SETUP: {setup_name} ===")
        for arch, sizes, xlabel, loss_size, T in archs:
            config = config_for(arch, setup_name, T)
            for kind in teachers:
                tlab = TEACHER_LABEL[kind]

                if not args.skip_rmd:
                    results = rmd_sweep(arch, setup_name, kind, sizes, config, reps)
                    title = (
                        f"RMD² to E^G at end of training — {arch}, "
                        f"{setup_name} setup, {tlab} (SI init)"
                    )
                    fname = os.path.join(outdir, f"rmd_{arch}_{setup_name}_{kind}.pdf")
                    plot_rmd_curves(
                        sizes, results, xlabel, title, save_pdf=True, filename=fname
                    )
                    plt.close("all")
                    print(f"      saved {os.path.basename(fname)}")

                if not args.skip_loss:
                    histories = loss_sweep(
                        arch, setup_name, kind, loss_size, config, reps
                    )
                    if arch == "NN":
                        sz_txt = f"N={loss_size}"
                    else:
                        sz_txt = f"M={RESNET_WIDTH_M}, L={loss_size}"
                    title = (
                        f"Training loss vs SGD step — {arch}, {setup_name} setup, "
                        f"{tlab} (SI init, {sz_txt})"
                    )
                    fname = os.path.join(outdir, f"loss_{arch}_{setup_name}_{kind}.pdf")
                    plot_training_curves(
                        histories, title, save_pdf=True, filename=fname
                    )
                    plt.close("all")
                    print(f"      saved {os.path.basename(fname)}")

    print(f"\nDone. PDFs written to: {outdir}")


if __name__ == "__main__":
    main()
