"""
visualise_3d.py
---------------
Interactive 3D visualisation of real fly-neuron skeletons with a time-resolved
spike animation.

FlyBrainViewer3D loads the **real anatomical skeletons** of a set of
interconnected neurons (as navis TreeNeurons with physical voxel coordinates),
renders them in an interactive Plotly 3D chart, and maps a LIF simulation's
spike raster onto the skeletons: a neuron that fires at time step t flashes
bright yellow, others stay dim grey. The result is saved as a standalone
interactive HTML file you can open in any browser and rotate / zoom / drag.

The skeletons come from NeuPrint/MaleCNS via :mod:`brain_client`; coordinates
are the dataset's raw voxel units (large integer values). We keep the same
units on every axis so shape is preserved, and expose ``scale_to_um`` so the
caller can convert to physical microns when the voxel size is known.
"""

from __future__ import annotations

import numpy as np
from plotly import graph_objects as go

DIM_COLOR = "rgb(180,180,185)"   # resting neuron, grey
FIRE_COLOR = "#FFE033"           # firing neuron, bright yellow
BACKGROUND = "rgb(20,22,25)"

# Downsample point count per neuron (keeps full shape, keeps HTML light).
MAX_NODES_PER_NEURON = 1500


def _thin_indices(n: int, max_nodes: int) -> np.ndarray:
    """Uniformly thin a node list of length n to <= max_nodes, always keeping
    the first and last index so the arbor's endpoints survive."""
    if n <= max_nodes:
        return np.arange(n)
    step = (n - 1) / (max_nodes - 1)
    return np.unique(np.round(np.arange(max_nodes) * step).astype(int))


class FlyBrainViewer3D:
    """Render interconnected neuron skeletons + a LIF spike raster as a
    time-animated interactive Plotly 3D figure."""

    def __init__(
        self,
        neurons,                    # navis.NeuronList
        body_ids: list[int],
        raster: np.ndarray,         # (T, N) 0/1, time x neurons
        *,
        scale_to_um: float = 1.0,   # multiply all coords by this (voxel->µm)
        dim_color: str = DIM_COLOR,
        fire_color: str = FIRE_COLOR,
        max_nodes_per_neuron: int = MAX_NODES_PER_NEURON,
        title: str = "Fly brain - LIF spike activity on real 3D skeletons",
    ) -> None:
        raster = np.asarray(raster, dtype=bool)
        if raster.ndim != 2:
            raise ValueError("raster must be 2D (time x neurons)")
        T, N = raster.shape
        if N != len(body_ids):
            raise ValueError(
                f"raster has {N} neuron columns but {len(body_ids)} bodyIds given"
            )
        if N != len(neurons):
            raise ValueError(
                f"{len(body_ids)} bodyIds but got {len(neurons)} skeletons"
            )
        self.body_ids = list(body_ids)
        self.raster = raster
        self.T, self.n = T, N
        self.neurons = list(neurons)
        self.scale_to_um = float(scale_to_um)
        self.dim_color = dim_color
        self.fire_color = fire_color
        self.max_nodes = max_nodes_per_neuron
        self.title = title

        self._frames = []        # per-neuron thinned (M,3) coordinate array
        self._prepare()

    # ------------------------------------------------------------------#

    def _prepare(self) -> None:
        """Extract thinned skeleton point clouds for each neuron."""
        for neu in self.neurons:
            nodes = neu.nodes
            if "x" not in nodes or "y" not in nodes or "z" not in nodes:
                raise ValueError(
                    f"Skeleton for body {neu.id} is missing x/y/z columns"
                )
            xyz = nodes[["x", "y", "z"]].to_numpy(dtype=np.float64)
            xyz = xyz * self.scale_to_um
            if len(xyz) == 0:
                raise ValueError(f"Skeleton for body {neu.id} has no nodes")
            keep = _thin_indices(len(xyz), self.max_nodes)
            self._frames.append(xyz[keep])

    @property
    def total_nodes(self) -> int:
        return sum(len(p) for p in self._frames)

    # ---------------------------------------------------------------------#

    def _build_fig(self) -> go.Figure:
        T, n = self.T, self.n
        fig = go.Figure()

        # Base traces (visible at step 0): one grey marker cloud per neuron.
        for i, pts in enumerate(self._frames):
            fig.add_trace(
                go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode="markers",
                    marker=dict(size=1.7, color=self.dim_color, opacity=1.0),
                    name=f"body {self.body_ids[i]}",
                    showlegend=True,
                )
            )

        # Frames: one per time step. Each frame restyles every neuron's markers
        # to bright yellow if it fired at that step, else stays grey.
        frames = []
        for t in range(T):
            fdata = []
            for i, pts in enumerate(self._frames):
                fired = bool(self.raster[t, i])
                fdata.append(
                    go.Scatter3d(
                        x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                        mode="markers",
                        marker=dict(
                            size=3.2 if fired else 1.6,
                            color=self.fire_color if fired else self.dim_color,
                            opacity=1.0,
                        ),
                        name=f"neuron {self.body_ids[i]}",
                        showlegend=False,
                    )
                )
            frames.append(
                go.Frame(
                    name=f"t{t}",
                    data=fdata,
                    layout=dict(title_text=f"{self.title}  -  step {t}"),
                )
            )
        fig.frames = frames

        # Slider + play controls.
        steps = [
            dict(
                method="animate",
                args=[[f"t{k}"],
                      {"frame": {"duration": 250, "redraw": True},
                       "transition": {"duration": 0}}],
                label=str(k),
            )
            for k in range(T)
        ]
        sliders = [dict(
            active=0, currentvalue=dict(prefix="time step: ", font=dict(size=15)),
            pad=dict(b=12, t=0), len=0.9, steps=steps,
        )]
        play_pause = [
            dict(
                type="buttons",
                buttons=[
                    dict(label="Play", method="animate",
                         args=[None,
                                    {"frame": {"duration": 260, "redraw": True},
                                     "fromcurrent": True,
                                     "transition": {"duration": 100}}]),
                    dict(label="Pause", method="animate",
                         args=[[None],
                                    {"frame": {"duration": 260, "redraw": True},
                                     "mode": "immediate"}]),
                    dict(label="Reset", method="animate",
                         args=[[None], {"frame": {"duration": 0, "redraw": True},
                                         "fromcurrent": True, "mode": "immediate"}]),
                ],
                pad=dict(r=10, t=10), showactive=False, x=0.05, xanchor="left",
                y=0.05, yanchor="bottom",
            )
        ]

        fig.update_layout(
            template="plotly_dark",
            title=self.title,
            scene=dict(
                xaxis=dict(title="x (vox)", showbackground=False, showgrid=True,
                           gridcolor="rgb(60,60,66)", zeroline=False),
                yaxis=dict(title="y (vox)", showbackground=False, showgrid=True,
                           gridcolor="rgb(60,60,66)", zeroline=False),
                zaxis=dict(title="z (vox)", showbackground=False, showgrid=True,
                            gridcolor="rgb(60,60,66)", zeroline=False),
                aspectmode="data",
                bgcolor=BACKGROUND,
            ),
            sliders=sliders,
            updatemenus=play_pause,
            paper_bgcolor=BACKGROUND,
            font=dict(color="rgb(230,230,230)"),
        )
        return fig

    def save(self, path: str = "brain_sim_3d.html") -> str:
        """Write the animated interactive figure to `path` and return the path."""
        fig = self._build_fig()
        fig.write_html(path, full_html=True, include_plotlyjs="cdn")
        return path