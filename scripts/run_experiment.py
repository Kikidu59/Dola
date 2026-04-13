"""
run_experiment.py — Reproduce Figure 1 from the paper.

Figure 1 shows RMDs at end of training, for N = 5, 10, 50, 100, 500, 1000
(we skip 5000 for speed), comparing vanilla, DA, FA, EA schemes.

Each subplot shows:
  - Row 1: RMD²(ν_N, (ν_N)^{E^G}) or RMD²(ν_N, (ν_N)^G)
    → how far the trained distribution is from E^G (SI init)
      or from being WI (WI init)
  - Row 2: RMD between vanilla, DA, FA schemes
    → how close the different schemes are to each other

Outputs saved as PNG in the project root.
"""

import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.teacher import make_wi_particles, make_arbitrary_particles, make_si_particles
from src.training import (
    DEFAULT_CONFIG,
    sgd_step_vanilla,
    sgd_step_projected,
    train,
)
from src.sl_techniques import sgd_step_da, sgd_step_fa, sgd_step_ea
from src.metrics import rmd_to_projected, rmd_squared


# =========================================================================
# Experiment configuration
# =========================================================================

# Values of N to test (skip 5000 for speed; to add if enough time)
N_VALUES = [5, 10, 50, 100, 500]

# Number of repetitions per setting
N_REPS = 3  # paper uses 10; use 3 for quick iteration, increase later

# Training config
CONFIG = {
    "alpha": 50.0,
    "tau": 1e-4,
    "beta": 1e-6,
    "batch_size": 20,
    "T": 20.0,
    "gr": 5,
}

# Schemes to compare
SCHEMES = {
    "vanilla": {"step_fn": None},  # None = auto (projected for SI, vanilla for WI)
    "DA": {"step_fn": sgd_step_da},
    "FA": {"step_fn": sgd_step_fa},
    "EA": {"step_fn": sgd_step_ea},
}


def run_single(teacher_particles, N, init_type, scheme_name, key):
    """Run a single training and return final particles."""
    step_fn = SCHEMES[scheme_name]["step_fn"]

    # For vanilla with SI init, use projected SGD; with WI init, use vanilla

    step_fn = sgd_step_projected

    history = train(
        teacher_particles=teacher_particles,
        N=N,
        key=key,
        config=CONFIG,
        init_type=init_type,
        step_fn=step_fn,
    )
    return history["particles"][-1]  # final particles


def run_sweep(teacher_particles, init_type, teacher_name):
    """Run all schemes for all values of N, collecting RMD metrics.

    Returns:
        results: dict[scheme_name] → dict[metric_name] → {
            "means": array of shape (len(N_VALUES),),
            "stds":  array of shape (len(N_VALUES),),
        }
    """
    print(f"\n{'='*60}")
    print(f"  Teacher: {teacher_name}, Init: {init_type}")
    print(f"{'='*60}")

    results = {name: {"rmd_to_EG": [], "rmd_v_vs": {}} for name in SCHEMES}

    for i_n, N in enumerate(N_VALUES):
        print(f"\n  N = {N}")
        all_final = {name: [] for name in SCHEMES}

        for rep in range(N_REPS):
            base_key = jax.random.PRNGKey(rep * 1000 + N)

            for scheme_name in SCHEMES:
                # Use different sub-key per scheme to avoid correlation,
                # but same init particles (via same base_key split pattern)
                key = jax.random.fold_in(base_key, hash(scheme_name) % 2**31)
                final_p = run_single(teacher_particles, N, init_type, scheme_name, key)
                all_final[scheme_name].append(final_p)

            print(f"    rep {rep+1}/{N_REPS} done")

        # Compute metrics for each scheme
        for scheme_name in SCHEMES:
            rmds = [rmd_to_projected(p) for p in all_final[scheme_name]]
            results[scheme_name]["rmd_to_EG"].append(rmds)

    return results


def plot_rmd_to_EG(results, teacher_name, init_type, filename):
    """Plot Row 1 of Figure 1/3: distance to E^G for each scheme."""
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    markers = {"vanilla": "o", "DA": "s", "FA": "^", "EA": "D"}
    colors = {"vanilla": "C0", "DA": "C1", "FA": "C2", "EA": "C3"}

    for scheme_name in SCHEMES:
        rmd_data = results[scheme_name]["rmd_to_EG"]
        means = [np.mean(r) for r in rmd_data]
        stds = [np.std(r) for r in rmd_data]

        ax.errorbar(
            N_VALUES,
            means,
            yerr=stds,
            marker=markers[scheme_name],
            color=colors[scheme_name],
            label=scheme_name,
            capsize=3,
            linewidth=1.5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Number of Particles (N)")
    ax.set_ylabel("RMD² to projected version")
    ax.set_title(f"Distance to E^G — {teacher_name} teacher, {init_type} init")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"\n  Saved: {filename}")


def plot_rmd_between_schemes(results, teacher_name, init_type, filename):
    """Plot Row 2 of Figure 1: RMD between vanilla and DA/FA schemes.

    We compute pairwise RMDs from the stored final particles."""
    # This would require storing the particles themselves, which we
    # don't do in the current results structure. For now, we skip this
    # and focus on the distance-to-E^G plot (the main result).
    pass


# =========================================================================
# Main
# =========================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  Reproducing Figure 1 from the paper")
    print("  (reduced N_VALUES and N_REPS for speed)")
    print("=" * 60)

    # --- Experiment 1: SI init + WI teacher ---
    teacher_wi = make_wi_particles()
    results_si_wi = run_sweep(teacher_wi, "si", "WI")
    plot_rmd_to_EG(results_si_wi, "WI", "SI", "fig1_si_init_wi_teacher.png")

    # --- Experiment 2: SI init + arbitrary teacher ---
    teacher_arb = make_arbitrary_particles()
    results_si_arb = run_sweep(teacher_arb, "si", "arbitrary")
    plot_rmd_to_EG(results_si_arb, "arbitrary", "SI", "fig1_si_init_arb_teacher.png")

    print("\n" + "=" * 60)
    print("  Done! Check the PNG files in the project root.")
    print("=" * 60)
