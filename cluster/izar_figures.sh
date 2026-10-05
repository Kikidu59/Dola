#!/bin/bash
# ===================================================================
# Regenerate ALL report figures (RMD-vs-size + loss-vs-step) for the
# DOLA project on the Izar GPU cluster.
#
# NOTE ON GPU USEFULNESS: this workload is made of tiny 2x2-matrix
# particle updates (M=5, N<=1000) driven by a long Python loop of SGD
# steps. It is latency-bound, and the ResNet forward/backward is a
# sequential scan over the L layers. A V100 therefore gives at best a
# modest speedup on the deep-ResNet runs and little-to-nothing on the
# NN / small-size runs. The more effective lever is a SLURM ARRAY that
# parallelises the independent (setup x arch x teacher) blocks — see
# cluster/jed_figures.sh for that variant.
#
# Submit from the repository root:   sbatch cluster/izar_figures.sh
# ===================================================================
#SBATCH --job-name=dola_figures
#SBATCH --account=math-454
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=dola_figures_%j.out

set -euo pipefail

# --- modules -------------------------------------------------------
module purge
module load gcc cuda python

# --- run from the repository root (where sbatch is called) ---------
cd "$SLURM_SUBMIT_DIR"

# --- environment (idempotent) --------------------------------------
# A dedicated venv with the CUDA build of JAX. Built on first run and
# reused afterwards. TIP: if the compute nodes have no internet, create
# it ONCE on the login node by running the block below interactively,
# then resubmit — the job will just reuse .venv_izar.
if [ ! -d ".venv_izar" ]; then
    echo "=== creating .venv_izar (first run only) ==="
    python -m venv .venv_izar
    source .venv_izar/bin/activate
    pip install --upgrade pip
    # GPU build of JAX (CUDA 12) + the libraries the script imports.
    pip install "jax[cuda12]" matplotlib plotly numpy pot
else
    source .venv_izar/bin/activate
fi

# --- sanity check: is the GPU visible to JAX? ----------------------
python -c "import jax; print('JAX backend:', jax.default_backend(), '| devices:', jax.devices())"

# --- generate the figures ------------------------------------------
OUTDIR="$SLURM_SUBMIT_DIR/figures"
mkdir -p "$OUTDIR"

echo "=== generating figures -> $OUTDIR ==="
srun python scripts/reproduce_figures.py --reps 10 --outdir "$OUTDIR"

echo "=== done ==="
ls -la "$OUTDIR"
