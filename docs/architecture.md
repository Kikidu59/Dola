# Code architecture

This page describes how the code in `src/` is organised and how the pieces fit together.
For the scientific context and how to run things, see the [README](../README.md).

## Setups (`src/spaces.py`)

The group is G = C₂ = {Id, P}, where P swaps the two coordinates of ℝ².
Two interchangeable "setups" define the unit activation σ\*(x, z) and the action of G on a particle z ∈ ℝ^{2×2}:

| Setup    | Activation σ\*(x, z)            | Action M_g · z           | Invariant subspace E^G           |
|----------|---------------------------------|--------------------------|----------------------------------|
| `matrix` | σ(z · x)                        | g · z · gᵀ               | matrices `[[a, b], [b, a]]`      |
| `uv`     | v · σ(uᵀx / d), z = [u; v]      | (g · u, g · v)           | matrices `[[a, a], [b, b]]`      |

σ is the logistic sigmoid and d = 2. In both cases σ\* is *jointly equivariant*: σ\*(g·x, M_g·z) = g·σ\*(x, z).

The active setup is a module-level global, switched with:

```python
import src.spaces as spaces
spaces.SETUP = spaces.make_setup_matrix()   # or spaces.make_setup_uv()
jax.clear_caches()                          # drop jitted functions traced with the old setup
```

The rest of the code only goes through the public API (`sigma_star`, `action_on_z`, `project_EG`, `get_EG_basis`) and is therefore setup-agnostic.

## Models (`src/model.py`)

**Shallow network** (mean-field scaling), particles of shape `(N, 2, 2)`:

    Φ^N_θ(x) = (1/N) Σᵢ σ*(x, θᵢ)

**ResNet** (mean-ODE scaling), particles of shape `(L, M, 2, 2)` — L layers of M particles:

    h_0 = x
    h_l = h_{l-1} + (α / (L·M)) Σᵢ σ*(h_{l-1}, z^{i,l})

`α` (`alpha_arch`) is an architectural constant (default 1), distinct from the learning rate.
The depth recursion is a `jax.lax.scan`, so the graph is not unrolled over L, and the batched forward is jitted.

## Teachers and data (`src/teacher.py`)

Teacher–student setting with inputs X ~ N(0, σ_π² Id₂), σ_π = 4, and labels Y = f\*(X). Three teachers:

- **arbitrary** — 5 fixed particles, no symmetry;
- **WI** (weakly invariant) — the 5 arbitrary particles plus their G-images, so f\* is equivariant;
- **SI** (strongly invariant) — 5 particles living in E^G.

ResNet teachers repeat the same particles at every layer.

## Symmetry-leveraging schemes (`src/training.py`)

The scheme is *not* baked into the SGD step: each one is just a different loss function passed to `train` / `train_resnet`.

| Scheme  | Loss                     | Mechanism                                                     |
|---------|--------------------------|---------------------------------------------------------------|
| Vanilla | `loss_fn`                | plain quadratic loss                                          |
| DA      | `loss_fn_da`             | loss averaged over all G-transforms of the data               |
| FA      | `loss_fn_fa`             | output symmetrised: Q_G·Φ(x) = ½ [Φ(x) + P·Φ(P·x)]            |
| EA      | `loss_fn_ea`             | particles projected onto E^G before the forward pass          |

Each has a `*_resnet` counterpart. The update is SGD with optional Langevin noise (SGLD):

    θ ← θ − α ∇L(θ) + √(2αβ / N) · ξ

When `project_noise=True` (SI initialisation), ξ is projected onto E^G so that it cannot push particles out of the invariant subspace.

Initialisations: **WI** — i.i.d. N(0, 1/16) entries; **SI** — the same Gaussian projected onto E^G.

Default hyperparameters (`DEFAULT_CONFIG`, `DEFAULT_CONFIG_UV`, `DEFAULT_CONFIG_RESNET`) follow Appendix F of the paper. The UV setup uses a smaller learning rate (α = 5 instead of 50) because the matrix-tuned value makes its dynamics unstable.

## Metrics (`src/metrics.py`)

- **W₂²** between empirical particle distributions, via optimal transport (POT);
- **RMD²(µ, ν) = W₂²(µ, ν) / (M₂(µ) + M₂(ν))**, a normalised Wasserstein distance in [0, 1];
- `rmd_to_projected` — RMD² between ν and its projection onto E^G (distance to strong invariance);
- `rmd_to_symmetrized` — RMD² between ν and its G-symmetrisation (distance to weak invariance);
- `l2_distance` — Monte-Carlo L² distance between two shallow models.

For the ResNet, the reported distance to E^G is the RMD² averaged over the L layers.

## Plotting (`src/utils.py`)

Matplotlib / Plotly helpers used by the notebooks and by `scripts/reproduce_figures.py`. They are imported lazily by `src/__init__.py`, so the numerical core does not depend on them.
