"""
src — Mean-field symmetries in overparametrized neural networks.

"The paper" in the docstrings of this package refers to:
    J. Maass Martínez & J. Fontbona, "Symmetries in Overparametrized Neural
    Networks: A Mean-Field View", NeurIPS 2024 (arXiv:2405.19995).

The plotting helpers (src.utils) are imported lazily, so the numerical core
can be used without matplotlib / plotly installed.
"""

from src.spaces import (
    GROUP_ELEMENTS,
    action_on_x,
    action_on_z,
    project_EG,
    sigma_star,
    get_EG_basis,
    make_setup_matrix,
    make_setup_uv,
)

from src.model import (
    forward,
    forward_batch,
    forward_resnet,
    forward_batch_resnet,
)

from src.teacher import (
    make_arbitrary_particles,
    make_wi_particles,
    make_si_particles,
    make_si_particles_uv,
    make_arbitrary_particles_resnet,
    make_wi_particles_resnet,
    make_si_particles_resnet,
    make_si_particles_uv_resnet,
    sample_data,
    sample_data_resnet,
    SCALE,
    SIGMA_PI,
)

from src.loss import (
    quadratic_loss,
    quadratic_loss_batch,
    regularization,
    regularization_resnet,
    population_risk,
)

from src.training import (
    DEFAULT_CONFIG,
    DEFAULT_CONFIG_UV,
    DEFAULT_CONFIG_RESNET,
    loss_fn,
    loss_fn_fa,
    loss_fn_ea,
    loss_fn_da,
    loss_fn_resnet,
    loss_fn_fa_resnet,
    loss_fn_ea_resnet,
    loss_fn_da_resnet,
    sgd_step,
    sgd_step_resnet,
    init_particles_wi,
    init_particles_si,
    init_particles_wi_resnet,
    init_particles_si_resnet,
    train,
    train_resnet,
)


from src.metrics import (
    w2_squared,
    rmd_squared,
    rmd,
    rmd_to_projected,
    rmd_to_symmetrized,
    l2_distance,
    project_particles_EG,
    symmetrize_particles,
)

# Plotting helpers are resolved lazily (PEP 562) so that importing the
# numerical core does not pull in matplotlib / plotly.
_PLOTTING_NAMES = {
    "DEFAULT_MARKERS",
    "DEFAULT_COLORS",
    "plot_rmd_curves",
    "plot_loss_curves",
    "plot_training_curves",
    "plot_particles_3d",
    "plot_uv_equivariance",
    "plot_uv_equivariance_resnet",
    "plot_particles_3d_resnet",
}


def __getattr__(name):
    if name in _PLOTTING_NAMES:
        from src import utils

        return getattr(utils, name)
    raise AttributeError(f"module 'src' has no attribute {name!r}")
