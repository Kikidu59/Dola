"""
src — Mean-field symmetries in overparametrized neural networks.
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
)

from src.teacher import (
    make_arbitrary_particles,
    make_wi_particles,
    make_si_particles,
    sample_data,
    SCALE,
    SIGMA_PI,
)

from src.loss import (
    quadratic_loss,
    quadratic_loss_batch,
    regularization,
    population_risk,
)

from src.training import (
    DEFAULT_CONFIG,
    loss_fn,
    loss_fn_fa,
    loss_fn_ea,
    loss_fn_da,
    sgd_step,
    init_particles_wi,
    init_particles_si,
    train,
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
