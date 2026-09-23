"""
simulator.py
------------
A simple Leaky Integrate-and-Fire (LIF) neural-network dynamics simulator.

LIFSimulator takes a connectivity matrix W and evolves membrane potentials
over discrete time steps, emitting discrete spikes whenever a neuron's
membrane potential crosses a threshold.
"""

from __future__ import annotations

import numpy as np


class LIFSimulator:
    """Leaky Integrate-and-Fire network driven by the connectivity matrix W."""

    def __init__(
        self,
        weight_matrix: np.ndarray,
        *,
        dt: float = 1.0,
        tau: float = 20.0,           # membrane time constant (ms)
        reset_potential: float = -65.0,  # mV (rest == reset here)
        threshold: float = -55.0,       # mV
        resistance: float = 1.0,        # dimensionless scale on total current
    ) -> None:
        W = np.asarray(weight_matrix, dtype=np.float64)
        if W.ndim != 2 or W.shape[0] != W.shape[1]:
            raise ValueError(
                f"Weight matrix must be 2D square, got shape {W.shape}"
            )
        self.W = W
        self.n = W.shape[0]

        self.dt = dt
        self.tau = tau
        self.v_rest = reset_potential
        self.v_reset = reset_potential
        self.threshold = threshold
        self.R = resistance

        # Leak factor for the discrete update: exp(-dt/tau) in [0, 1].
        self.decay = np.exp(-self.dt / self.tau)

        self.v = np.full(self.n, reset_potential, dtype=np.float64)
        self.step_count = 0
        self._last_spikes = np.zeros(self.n, dtype=np.float64)

    def reset(self) -> None:
        """Return every neuron to its reset potential and zero the clock."""
        self.v[:] = self.v_reset
        self.step_count = 0
        self._last_spikes[:] = 0.0

    def step(self, input_currents: np.ndarray) -> np.ndarray:
        """Advance the network by `dt`, returning the spike vector for this step.

        The returned array holds 1 for neurons that spiked this step, else 0.

        Membrane update (per-neuron basis):
            v  <-  v * decay + R * I_total
        where I_total combines the external input current and the synaptic
        current contributed by neurons that spiked in the *previous* step
        (W @ spikes_prev). A neuron at or above threshold fires and is reset.
        """
        I = np.asarray(input_currents, dtype=np.float64)
        if I.shape != (self.n,):
            raise ValueError(
                f"input_currents must have shape ({self.n},), got {I.shape}"
            )

        # Synaptic current from the previous step's spikes routed through W.
        synaptic = self.W @ self._last_spikes

        # Standard LIF membrane update.
        self.v = self.v * self.decay + self.R * (I + synaptic)

        # Threshold crossing -> spike. Reset only the neurons that fired.
        spikes = (self.v >= self.threshold).astype(np.float64)
        self.v[spikes > 0] = self.v_reset

        self._last_spikes = spikes
        self.step_count += 1
        return spikes

    def run(self, input_sequence: np.ndarray) -> dict:
        """
        Run several steps with an array of shape (T, N) of input currents.

        Returns a dict:
            raster        : (T, N) array of 0/1 spikes
            firing_rates  : (N,) mean spike rate per neuron over T steps (0..1)
            spike_counts  : (N,) total spikes per neuron
            time_steps    : (T,) integer step indices
        The raster (with its time axis) is what you need for time-resolved
        3D animation: row t says which neurons spiked at step t.
        """
        arr = np.asarray(input_sequence, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != self.n:
            raise ValueError(
                f"input_sequence must have shape (T, {self.n}), got {arr.shape}"
            )
        T = arr.shape[0]
        raster = np.zeros((T, self.n), dtype=np.int64)
        for t in range(T):
            raster[t] = self.step(arr[t])
        return {
            "raster": raster,
            "firing_rates": raster.sum(axis=0) / T,
            "spike_counts": raster.sum(axis=0),
            "time_steps": np.arange(T, dtype=int),
        }