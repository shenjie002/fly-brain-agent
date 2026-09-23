"""
motor_map.py
------------
The bridge between the fly-brain-agent (LIF neural dynamics) and the FlyGym
NeuroMechFly motor system.

Pipeline:
  Laya decision (FLAP / HOLD / TURN_LEFT / TURN_RIGHT / WALK)
    -> activates a *motor-neuron pool* (a set of neurons in the connectome
       sub-network, or a hand-authored wiring if NeuPrint is offline)
    -> LIFSimulator steps fire those neurons -> spike trains
    -> spike trains are decoded into motor commands:
         * wing motor neurons (b1,b2 / DVM / DLM)  -> wing flap amplitude
         * leg motor neurons (six legs CPG)        -> tripod gait phase offsets
    -> these motor commands drive the FlyGym NeuroMechFly model

Two operating modes:
  ONLINE : uses real MaleCNS connectome weights via BrainClient (NeuPrint).
  OFFLINE: uses a hand-authored motor-neuron wiring (no network needed) so the
           pipeline runs end-to-end on a laptop.
"""
from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ── Motor primitives that Laya can choose ────────────────────────────────── #
PRIMITIVES = ("FLAP", "HOLD", "TURN_LEFT", "TURN_RIGHT", "WALK")


# ── Hand-authored motor-neuron pools (OFFLINE mode) ──────────────────────── #
# A small recurrent network whose connectivity encodes the known Drosophila
# flight / walking motor architecture. Indices:
#   0  wing depressor L  (b1/DVM-like)
#   1  wing elevator L   (DLM-like)
#   2  wing depressor R
#   3  wing elevator R
#   4  haltere L (rate sensor)
#   5  haltere R
#   6  leg CPG front-L
#   7  leg CPG mid-L
#   8  leg CPG hind-L
#   9  leg CPG front-R
#  10  leg CPG mid-R
#  11  leg CPG hind-R
#  12  descending command neuron (receives Laya drive)
WING_DL, WING_EL = 0, 1
WING_DR, WING_ER = 2, 3
HALT_L, HALT_R = 4, 5
LEG_FL, LEG_ML, LEG_HL = 6, 7, 8
LEG_FR, LEG_MR, LEG_HR = 9, 10, 11
CMD = 12
N_NEURONS = 13


def _offline_weights() -> np.ndarray:
    """Hand-authored 13x13 connectivity (pre -> post). Positive = excitatory."""
    W = np.zeros((N_NEURONS, N_NEURONS), dtype=np.float64)

    # wing power stroke / recovery: antagonist alternation (central oscillator)
    W[WING_DL, WING_EL] = -1.2      # depressor firing inhibits elevator
    W[WING_EL, WING_DL] = -1.2
    W[WING_DR, WING_ER] = -1.2
    W[WING_ER, WING_DR] = -1.2
    # cross-coupling keeps L/R wings in phase
    W[WING_DL, WING_DR] = 0.6
    W[WING_DR, WING_DL] = 0.6
    W[WING_EL, WING_ER] = 0.6
    W[WING_ER, WING_EL] = 0.6
    # halteres couple to wing motor (phase stabiliser)
    W[HALT_L, WING_DL] = 0.5; W[HALT_R, WING_DR] = 0.5

    # six-leg CPG: tripod coordination.
    # tripod A = FL, MR, HL ; tripod B = FR, ML, HR ; A/B alternate.
    tripodA = [LEG_FL, LEG_MR, LEG_HL]
    tripodB = [LEG_FR, LEG_ML, LEG_HR]
    for a in tripodA:
        for b in tripodB:
            W[a, b] = -0.9          # A inhibits B and vice-versa -> alternation
            W[b, a] = -0.9
    # intra-tripod excitation keeps each tripod synchronised
    for grp in (tripodA, tripodB):
        for i in grp:
            for j in grp:
                if i != j:
                    W[i, j] = 0.7

    # descending command neuron drives everything
    for i in (WING_DL, WING_EL, WING_DR, WING_ER, HALT_L, HALT_R,
              *tripodA, *tripodB):
        W[CMD, i] = 0.4             # tonic drive from the brain command neuron

    return W


def _offline_stimulus(primitive: str, confidence: float) -> np.ndarray:
    """External current injected into each neuron for a given Laya primitive.

    Returns a vector of length N_NEURONS. The command neuron (CMD) is always
    driven proportionally to confidence; the primitive selects *which* motor
    pools get extra tonic drive. HOLD suppresses wings; WALK suppresses wings
    and drives legs; FLAP drives wings hard and suppresses legs.
    """
    I = np.zeros(N_NEURONS, dtype=np.float64)
    # choice 决定运动池 (主导); confidence 只在舒适区间内调制强度。
    # Laya 的 confidence 常偏低 (0.1~0.7), 不能直接当增益, 否则动作退化。
    drive = 6.0 * (0.65 + 0.35 * min(1.0, max(0.0, confidence)))  # 3.9 .. 6.0

    I[CMD] = drive                   # command neuron always gets the decision
    if primitive == "FLAP":
        I[WING_DL] = I[WING_EL] = I[WING_DR] = I[WING_ER] = drive * 1.8
        I[HALT_L] = I[HALT_R] = drive * 0.9
        # suppress legs during flight
        I[LEG_FL] = I[LEG_ML] = I[LEG_HL] = I[LEG_FR] = I[LEG_MR] = I[LEG_HR] = 0.0
    elif primitive == "WALK":
        I[LEG_FL] = I[LEG_ML] = I[LEG_HL] = I[LEG_FR] = I[LEG_MR] = I[LEG_HR] = drive * 1.5
        # suppress wings during walking
        I[WING_DL] = I[WING_EL] = I[WING_DR] = I[WING_ER] = 0.0
        I[HALT_L] = I[HALT_R] = 0.0
    elif primitive == "TURN_LEFT":
        I[LEG_FL] = I[LEG_HL] = drive * 2.0    # inner legs step harder
        I[LEG_FR] = I[LEG_HR] = drive * 0.3    # outer legs slow
        I[WING_DL] = I[WING_EL] = drive * 0.5  # slight asymmetric wing bias
    elif primitive == "TURN_RIGHT":
        I[LEG_FR] = I[LEG_HR] = drive * 2.0
        I[LEG_FL] = I[LEG_HL] = drive * 0.3
        I[WING_DR] = I[WING_ER] = drive * 0.5
    # HOLD: only CMD drive, nothing extra -> wings glide, legs stand
    return I


# ── Decoding: spike trains -> motor commands ─────────────────────────────── #
@dataclass
class MotorCommand:
    """The output of the neural decode stage, fed to FlyGym."""
    flap_amp: float = 0.0           # wing stroke amplitude (0..1) both wings
    flap_freq: float = 0.0          # wingbeat frequency (Hz)
    tripod_phase: float = 0.0       # 0..1 phase of the tripod gait cycle
    leg_gains: np.ndarray = field(
        default_factory=lambda: np.zeros(6, dtype=np.float64))
    raw_spikes: np.ndarray | None = None
    firing_rates: np.ndarray | None = None


def decode_spikes(spikes: np.ndarray, steps: int) -> MotorCommand:
    """Turn an (N x T) or (T x N) spike raster into a MotorCommand.

    spikes: shape (steps, N_NEURONS) of 0/1.
    """
    if spikes.ndim != 2:
        raise ValueError("spikes must be 2D (steps x neurons)")
    T = spikes.shape[0]
    if spikes.shape[1] != N_NEURONS:
        # transpose if given (N x T)
        if spikes.shape[0] == N_NEURONS:
            spikes = spikes.T
        else:
            raise ValueError(f"spikes last dim must be {N_NEURONS}")
    rates = spikes.sum(axis=0) / max(1, T)            # firing rate 0..1

    # wing amplitude: mean of the four wing motor firing rates
    wing_rates = rates[[WING_DL, WING_EL, WING_DR, WING_ER]]
    flap_amp = float(np.clip(wing_rates.mean(), 0.0, 1.0))
    # wingbeat frequency: alternation of depressor/elevator -> estimate from
    # the dominant oscillation of the wing pool summed spike count per step
    wing_pool = spikes[:, [WING_DL, WING_EL, WING_DR, WING_ER]].sum(axis=1)
    flap_freq = _estimate_freq(wing_pool)

    # legs: tripod phase from the six leg CPG rates
    leg_rates = rates[[LEG_FL, LEG_ML, LEG_HL, LEG_FR, LEG_MR, LEG_HR]]
    leg_gains = np.clip(leg_rates, 0.0, 1.0)
    leg_pool = spikes[:, [LEG_FL, LEG_ML, LEG_HL, LEG_FR, LEG_MR, LEG_HR]].sum(axis=1)
    tripod_phase = float((leg_pool.mean() * T) % 1.0)

    return MotorCommand(
        flap_amp=flap_amp, flap_freq=flap_freq,
        tripod_phase=tripod_phase, leg_gains=leg_gains,
        raw_spikes=spikes, firing_rates=rates,
    )


def _estimate_freq(sig: np.ndarray) -> float:
    """Rough dominant frequency of a 0/1 pulse train (normalised 0..1)."""
    if sig.sum() == 0:
        return 0.0
    # zero-crossing rate of the smoothed signal
    s = sig.astype(float)
    transitions = np.sum(np.abs(np.diff(s > s.mean())) > 0)
    return float(transitions) / (2.0 * len(s))        # cycles per step


# ── Public API: build the neural stage ───────────────────────────────────── #
def build_lif(primitive_weights: np.ndarray | None = None) -> "LIFSimulator":
    """Return a LIFSimulator wired with the motor-neuron connectivity."""
    from simulator import LIFSimulator
    W = primitive_weights if primitive_weights is not None else _offline_weights()
    return LIFSimulator(W, dt=1.0, tau=6.0, threshold=-50.0, reset_potential=-65.0)


def laya_to_motor(primitive: str, confidence: float, *, steps: int = 40,
                  weights: np.ndarray | None = None) -> tuple[MotorCommand, "LIFSimulator"]:
    """One-shot: Laya decision -> neural firing -> motor command.

    Returns (MotorCommand, simulator) so the caller can keep stepping / rendering.
    """
    sim = build_lif(weights)
    stim = _offline_stimulus(primitive, confidence)
    stim_seq = np.tile(stim, (steps, 1))            # (steps, N)
    res = sim.run(stim_seq)
    raster = res["raster"]                           # (steps, N)
    cmd = decode_spikes(raster, steps)
    return cmd, sim