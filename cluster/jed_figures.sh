#!/bin/bash
# ===================================================================
# Regenerate ALL report figures (RMD-vs-size + loss-vs-step) for the
# DOLA project on the Jed CPU cluster, as a SLURM JOB ARRAY.
#
# The work is embarrassingly parallel: each of the 8 array tasks runs
# one (setup x architecture x teacher) block and writes its own PDFs
# (rmd_*.pdf and loss_*.pdf), all into a shared figures/ directory with
# distinct filenames — so the tasks never collide. Wall-clock is then
# set by the single heaviest block (a deep-ResNet one), not by the sum.
#
# ---------------------------------------------------------------------
# ONE-TIME SETUP (run ONCE on the LOGIN node before sbatch, so the 8
# array tasks do NOT race to build the same venv):
#
#     cd ~/path/to/dola
#     module load gcc python
#     python -m venv .venv_jed
#     source .venv_jed/bin/activate
#     pip install --upgrade pip
#     pip install "jax==0.9.1" "jaxlib==0.9.1" matplotlib plotly numpy pot
#     # ^ PIN JAX 0.9.1 (the version in uv.lock). Newer 0.10.x DEADLOCKS the
#     #   lax.scan used by the ResNet on XLA-CPU (job sits at 0% CPU forever);
#     #   the NN path has no scan and is unaffected.
#
# Then submit from the repository root:   sbatch cluster/jed_figures.sh
# ===================================================================
#SBATCH --job-name=dola_figures
#SBATCH --account=math-454
#SBATCH --partition=academic
#SBATCH --array=0-7
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=dola_figures_%A_%a.out

set -euo pipefail

# --- map the array index -> one (setup, arch, teacher) block --------
SETUPS=(  matrix matrix matrix    matrix    uv uv uv       uv       )
ARCHS=(   NN     NN     ResNet    ResNet    NN NN ResNet   ResNet   )
TEACHERS=(WI     arbitrary WI     arbitrary WI arbitrary WI arbitrary)

i=$SLURM_ARRAY_TASK_ID
SETUP=${SETUPS[$i]}
ARCH=${ARCHS[$i]}
TEACHER=${TEACHERS[$i]}
echo "=== task $i : setup=$SETUP arch=$ARCH teacher=$TEACHER ==="

# --- environment ---------------------------------------------------
module purge
module load gcc python
cd "$SLURM_SUBMIT_DIR"

if [ ! -d ".venv_jed" ]; then
    echo "ERROR: .venv_jed not found. Build it ONCE on the login node first" >&2
    echo "       (see the ONE-TIME SETUP block at the top of this script)." >&2
    exit 1
fi
source .venv_jed/bin/activate

# Thread control. Do NOT disable XLA's Eigen threadpool
# (--xla_cpu_multi_thread_eigen=false): on this cluster's JAX it DEADLOCKS the
# lax.scan used by the ResNet forward/backward (the job then sits at 0% CPU).
# Instead we simply rely on SLURM's cgroup: with cpus-per-task=2, XLA-CPU sees
# only 2 cores, so there is no node-wide oversubscription. We still pin the
# numpy/BLAS/POT thread pools, which are independent of XLA.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1   # numpy / POT (OT cost matrix)
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Unbuffered stdout so the `size=... done` progress lines show up in the .out
# file LIVE instead of being held in Python's block buffer until the job ends.
export PYTHONUNBUFFERED=1

# --- run this one block --------------------------------------------
OUTDIR="$SLURM_SUBMIT_DIR/figures"
mkdir -p "$OUTDIR"

srun python -u scripts/reproduce_figures.py \
    --reps 10 \
    --setup "$SETUP" \
    --arch "$ARCH" \
    --teacher "$TEACHER" \
    --outdir "$OUTDIR"

echo "=== task $i done ==="
