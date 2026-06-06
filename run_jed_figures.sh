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
#     cd ~/path/to/DOLA
#     module load gcc python
#     python -m venv .venv_jed
#     source .venv_jed/bin/activate
#     pip install --upgrade pip
#     pip install jax matplotlib plotly numpy pot     # CPU build of JAX
#
# Then submit:   sbatch run_jed_figures.sh
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

# CRITICAL: force single-threaded math. The model is made of tiny 2x2 ops, so
# spreading each op over many threads only adds synchronisation overhead — and
# by default XLA-CPU grabs ALL cores of the (many-core) node, so several array
# tasks sharing a node oversubscribe it badly (this is what blew past the time
# limit). One thread per task is both faster here and collision-free.
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1   # numpy / POT (OT cost matrix)
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# --- run this one block --------------------------------------------
OUTDIR="$SLURM_SUBMIT_DIR/figures"
mkdir -p "$OUTDIR"

srun python scripts/reproduce_figures.py \
    --reps 10 \
    --setup "$SETUP" \
    --arch "$ARCH" \
    --teacher "$TEACHER" \
    --outdir "$OUTDIR"

echo "=== task $i done ==="
