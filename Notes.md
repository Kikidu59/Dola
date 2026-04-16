# DOLA — Notes de développement

## Architecture générale

### Configurations (`src/spaces.py`)
Deux setups interchangeables via `spaces.SETUP` :
- **matrix** : `σ*(x, z) = σ(z·x)`, action `M_g·z = g·z·gᵀ`
- **uv** : `σ*(x, z) = v·σ(uᵀx)`, action `M_g·(u,v) = (g·u, g·v)`

Le reste du code est agnostique au setup — il suffit de changer `SETUP` avant l'appel.

### Modèle shallow (`src/model.py`)
`Φ^N_θ(x) = (1/N) Σᵢ σ*(x, θᵢ)`, particules de shape `(N, 2, 2)`.

### Schémas de symmetrisation (`src/training.py`)
| Scheme | Mécanisme |
|--------|-----------|
| Vanilla | loss standard |
| FA | symmétrisation de l'output : `Q_G · Φ(x)` |
| DA | moyenne de la loss sur les transformées de G |
| EA | projection des particules sur `E^G` avant le forward |

Initialisation : **WI** (Gaussienne) ou **SI** (Gaussienne projetée sur `E^G`).

---

## ResNet — En cours

### Définition
`h_0 = x`
`h_l = h_{l-1} + (α/LM) Σᵢ σ*(h_{l-1}, z^{i,l})`

Particules de shape `(L, M, 2, 2)` — L couches, M particules par couche.  
`α` (`alpha_arch`) est une constante architecturale (défaut 1), distincte du learning rate.

### Todo list

- [x] `forward_resnet(x, particles, alpha_arch)` — `src/model.py`
- [x] `forward_batch_resnet(x_batch, particles, alpha_arch)` — `src/model.py`
- [x] `init_particles_wi_resnet(key, L, M)` — `src/training.py`
- [x] `init_particles_si_resnet(key, L, M)` — `src/training.py`
- [x] `loss_fn_resnet` — `src/training.py`
- [x] `loss_fn_fa_resnet` — `src/training.py`
- [x] `loss_fn_ea_resnet` — `src/training.py`
- [x] `loss_fn_da_resnet` — `src/training.py`
- [x] `train_resnet(teacher_particles, L, M, key, config, init_type, used_loss_fn)` — `src/training.py`
- [x] Tests `test_model.py` — shape `forward_resnet`
- [x] Tests `test_training.py` — smoke test `train_resnet`
