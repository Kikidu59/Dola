"""Plotting helpers used by the figure notebooks."""

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


DEFAULT_MARKERS = {"vanilla": "o", "DA": "s", "FA": "^", "EA": "D"}

_DOWNLOAD_CONFIG = {
    "toImageButtonOptions": {
        "format": "png",
        "scale": 4,
    }
}


def plot_rmd_curves(x_values, results, xlabel, title, markers=None, exclude=None):
    if markers is None:
        markers = DEFAULT_MARKERS
    skip = set(exclude) if exclude else set()
    for name in results:
        if name in skip:
            continue
        means = [jnp.mean(jnp.array(r)) for r in results[name]]
        stds = [jnp.std(jnp.array(r)) for r in results[name]]
        plt.errorbar(
            x_values,
            means,
            yerr=stds,
            marker=markers[name],
            label=name,
            capsize=3,
        )
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel(xlabel)
    plt.ylabel("RMD² to projected version")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def plot_particles_3d(
    student_particles, teacher_particles, title, show_invariant_plane=False
):
    """3D scatter: axes = z1, z2, z3, color = z4.

    If ``show_invariant_plane`` is True (matrix setup only), overlay the
    translucent plane z2 = z3 colored by z4 = z1 — the visible slice of
    E^G = {z : z = P z P} = {[[a,b],[b,a]]}.
    """
    S = jnp.array(student_particles).reshape(-1, 4)
    T = jnp.array(teacher_particles).reshape(-1, 4)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=S[:, 0],
            y=S[:, 1],
            z=S[:, 2],
            mode="markers",
            marker=dict(
                size=2,
                color=S[:, 3],
                colorscale="Viridis",
                colorbar=dict(title="z4"),
                cmin=-0.6,
                cmax=0.6,
            ),
            name="Student",
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=T[:, 0],
            y=T[:, 1],
            z=T[:, 2],
            mode="markers",
            marker=dict(
                size=6,
                color=T[:, 3],
                colorscale="Viridis",
                symbol="diamond",
                cmin=-0.6,
                cmax=0.6,
                line=dict(width=1, color="black"),
            ),
            name="Teacher",
        )
    )

    if show_invariant_plane:
        lo, hi = -0.6, 0.6
        fig.add_trace(
            go.Mesh3d(
                x=[lo, hi, hi, lo],
                y=[lo, lo, hi, hi],
                z=[lo, lo, hi, hi],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                intensity=[lo, hi, hi, lo],
                colorscale="Viridis",
                cmin=-0.6,
                cmax=0.6,
                showscale=False,
                opacity=0.35,
                name="E^G (z2 = z3)",
                showlegend=True,
                hoverinfo="name",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="z1",
            yaxis_title="z2",
            zaxis_title="z3",
            xaxis=dict(range=[-0.6, 0.6]),
            yaxis=dict(range=[-0.6, 0.6]),
            zaxis=dict(range=[-0.6, 0.6]),
        ),
        width=700,
        height=600,
    )
    fig.show(config=_DOWNLOAD_CONFIG)
    return fig


def plot_uv_equivariance(particles, teacher_particles=None, title=None):
    """Side-by-side scatter of u = (u_x, u_y) and v = (v_x, v_y) for the UV setup.

    A dashed diagonal y = x marks the G-invariant subspace
    (fixed points of P = [[0,1],[1,0]], i.e. vectors proportional to (1, 1)).
    Styled for Beamer slides: large fonts, equal aspect, clean grid.
    """
    parts = np.array(particles)
    teach = np.array(teacher_particles) if teacher_particles is not None else None

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    labels = (r"$u_x$", r"$u_y$", r"$v_x$", r"$v_y$")
    subtitles = (r"Layer 1 weights $u$", r"Layer 2 weights $v$")

    for k, ax in enumerate(axes):
        xs, ys = parts[:, k, 0], parts[:, k, 1]
        lim = float(np.max(np.abs(np.concatenate([xs, ys])))) * 1.15 + 1e-6
        if teach is not None:
            lim = max(lim, float(np.max(np.abs(teach[:, k]))) * 1.15)

        ax.axhline(0, color="0.8", lw=0.8, zorder=0)
        ax.axvline(0, color="0.8", lw=0.8, zorder=0)
        ax.plot(
            [-lim, lim],
            [-lim, lim],
            linestyle="--",
            color="crimson",
            lw=1.6,
            label=r"$E^G$ (invariant subspace)",
            zorder=1,
        )

        ax.scatter(
            xs,
            ys,
            s=28,
            alpha=0.75,
            color="#1f77b4",
            edgecolors="white",
            linewidths=0.4,
            label="Student",
            zorder=3,
        )
        if teach is not None:
            ax.scatter(
                teach[:, k, 0],
                teach[:, k, 1],
                marker="D",
                s=70,
                color="#ff7f0e",
                edgecolors="black",
                linewidths=0.8,
                label="Teacher",
                zorder=4,
            )

        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel(labels[2 * k], fontsize=13)
        ax.set_ylabel(labels[2 * k + 1], fontsize=13)
        ax.set_title(subtitles[k], fontsize=13)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.tick_params(labelsize=10)

    axes[0].legend(loc="best", fontsize=10, framealpha=0.9)

    if title is not None:
        fig.suptitle(title, fontsize=14, y=1.02)
    fig.tight_layout()
    plt.show()


def plot_uv_equivariance_resnet(particles, teacher_particles=None, title=None):
    """Two side-by-side 3D scenes of ResNet UV particles across depth L.

    Left scene:  axes = (u_x, u_y, layer l)
    Right scene: axes = (v_x, v_y, layer l)

    A translucent red plane marks the G-invariant subspace (u_x = u_y across
    all layers for the left scene, v_x = v_y for the right). Student markers
    are colored by layer index (Viridis). Teacher particles are shown as
    large orange diamonds.

    Args:
        particles:         array of shape (L, M, 2, 2) — final-snapshot weights.
        teacher_particles: (L, M*, 2, 2) or (M*, 2, 2); optional.
        title:             figure title.
    """
    parts = np.array(particles)
    if parts.ndim == 3:
        parts = parts[None, ...]
    L, M = parts.shape[0], parts.shape[1]
    layer_idx_student = np.repeat(np.arange(L), M)

    teach = None
    teach_layer_idx = None
    if teacher_particles is not None:
        teach = np.array(teacher_particles)
        if teach.ndim == 3:
            teach = teach[None, ...]
        teach_layer_idx = np.repeat(np.arange(teach.shape[0]), teach.shape[1])

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "scene"}]],
        subplot_titles=("Poids u = (u_x, u_y)", "Poids v = (v_x, v_y)"),
        horizontal_spacing=0.05,
    )

    for k, col in enumerate([1, 2]):
        xs = parts[:, :, k, 0].ravel()
        ys = parts[:, :, k, 1].ravel()

        lim = float(np.max(np.abs(np.concatenate([xs, ys])))) * 1.15 + 1e-6
        if teach is not None:
            lim = max(lim, float(np.max(np.abs(teach[:, :, k]))) * 1.15)

        # E^G: the diagonal plane u_x = u_y extended across the layer axis.
        fig.add_trace(
            go.Mesh3d(
                x=[-lim, lim, lim, -lim],
                y=[-lim, lim, lim, -lim],
                z=[0, 0, L - 1, L - 1] if L > 1 else [-0.5, -0.5, 0.5, 0.5],
                i=[0, 0],
                j=[1, 2],
                k=[2, 3],
                color="crimson",
                opacity=0.25,
                name="E^G (invariant subspace)",
                showlegend=(k == 0),
                hoverinfo="name",
            ),
            row=1,
            col=col,
        )

        fig.add_trace(
            go.Scatter3d(
                x=xs,
                y=ys,
                z=layer_idx_student,
                mode="markers",
                marker=dict(
                    size=3,
                    color=layer_idx_student,
                    colorscale="Viridis",
                    cmin=0,
                    cmax=max(L - 1, 1),
                    colorbar=dict(title="layer l", x=1.02) if k == 1 else None,
                    showscale=(k == 1),
                ),
                name="Student",
                showlegend=(k == 0),
            ),
            row=1,
            col=col,
        )

        if teach is not None:
            fig.add_trace(
                go.Scatter3d(
                    x=teach[:, :, k, 0].ravel(),
                    y=teach[:, :, k, 1].ravel(),
                    z=teach_layer_idx,
                    mode="markers",
                    marker=dict(
                        size=6,
                        symbol="diamond",
                        color="#ff7f0e",
                        line=dict(width=1, color="black"),
                    ),
                    name="Teacher",
                    showlegend=(k == 0),
                ),
                row=1,
                col=col,
            )

    axis_tmpl = lambda xt, yt: dict(
        xaxis=dict(title=xt, range=[-lim, lim]),
        yaxis=dict(title=yt, range=[-lim, lim]),
        zaxis=dict(title="layer l"),
        aspectmode="cube",
    )
    fig.update_layout(
        title=title,
        scene=axis_tmpl("u_x", "u_y"),
        scene2=axis_tmpl("v_x", "v_y"),
        width=1100,
        height=550,
        margin=dict(l=10, r=10, t=60, b=10),
        legend=dict(x=0.0, y=1.0),
    )
    fig.show(config=_DOWNLOAD_CONFIG)
    return fig


def plot_particles_3d_resnet(history, teacher_particles, title):
    """3D scatter with a slider to browse training snapshots.

    A translucent plane shows (the visible part of) the G-invariant subspace E^G.
    In the matrix setup, E^G = {z : z = P z P} = {[[a,b],[b,a]]}, i.e. z2 = z3
    AND z1 = z4. The plane is drawn at z2 = z3 and its surface is colored by
    z4 = z1 (the value z4 must take for a point on the plane to actually be
    in E^G), using the same Viridis colormap as the student markers.
    """
    snapshots = history["particles"]
    steps = history["steps"]

    T = jnp.array(teacher_particles).reshape(-1, 4)

    lo, hi = -0.6, 0.6
    plane = go.Mesh3d(
        x=[lo, hi, hi, lo],
        y=[lo, lo, hi, hi],
        z=[lo, lo, hi, hi],
        i=[0, 0],
        j=[1, 2],
        k=[2, 3],
        intensity=[lo, hi, hi, lo],
        colorscale="Viridis",
        cmin=-0.6,
        cmax=0.6,
        showscale=False,
        opacity=0.35,
        name="E^G (z2 = z3)",
        showlegend=True,
        hoverinfo="name",
    )

    frames = []
    for snap, step in zip(snapshots, steps):
        sp = jnp.array(snap)
        if sp.ndim == 4:
            sp = sp.reshape(-1, 2, 2)
        S = sp.reshape(-1, 4)

        frames.append(
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=S[:, 0],
                        y=S[:, 1],
                        z=S[:, 2],
                        mode="markers",
                        marker=dict(
                            size=2,
                            color=S[:, 3],
                            colorscale="Viridis",
                            colorbar=dict(title="z4"),
                            cmin=-0.6,
                            cmax=0.6,
                        ),
                        name="Student",
                    ),
                    go.Scatter3d(
                        x=T[:, 0],
                        y=T[:, 1],
                        z=T[:, 2],
                        mode="markers",
                        marker=dict(
                            size=6,
                            color=T[:, 3],
                            colorscale="Viridis",
                            symbol="diamond",
                            cmin=-0.6,
                            cmax=0.6,
                            line=dict(width=1, color="black"),
                        ),
                        name="Teacher",
                    ),
                    plane,
                ],
                name=str(step),
            )
        )

    fig = go.Figure(data=frames[0].data, frames=frames)

    slider_steps = [
        dict(
            method="animate",
            args=[
                [str(step)],
                dict(
                    mode="immediate",
                    frame=dict(duration=0, redraw=True),
                    transition=dict(duration=0),
                ),
            ],
            label=str(step),
        )
        for step in steps
    ]

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="z1",
            yaxis_title="z2",
            zaxis_title="z3",
            xaxis=dict(range=[-0.6, 0.6]),
            yaxis=dict(range=[-0.6, 0.6]),
            zaxis=dict(range=[-0.6, 0.6]),
        ),
        sliders=[
            dict(
                active=0,
                currentvalue=dict(prefix="Step: "),
                pad=dict(t=50),
                steps=slider_steps,
            )
        ],
        width=700,
        height=600,
    )
    fig.show(config=_DOWNLOAD_CONFIG)
    return fig
