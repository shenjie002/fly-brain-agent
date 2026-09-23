"""
main.py
-------
FastAPI service that wraps the fly-brain connectivity + LIF simulation engine.

Exposes:
    GET  /             -> service status
    POST /stimulate    -> feed an external stimulus (light/direction) and
                          return the firing behaviour of the simulated neurons,
                          emphasising the downstream (post-synaptic) readout.

Local scripts:
    python main.py run-server          -> run the FastAPI service (uvicorn)
    python main.py visualize           -> run a LIF simulation and write an
                                          interactive 3D brain_sim_3d.html
                                          (real skeletons + spike animation).
    python main.py visualize --body 12345 678 ...  --steps 20  (manual bodies)
"""

from __future__ import annotations

import os
import sys
import warnings

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()  # optional: load NEUPRINT_APPLICATION_TOKEN from a .env file

try:
    from .simulator import LIFSimulator
    from .brain_client import BrainClient, BrainClientError
except ImportError:  # allow `uvicorn main:app` / `python main.py` from project dir
    from simulator import LIFSimulator
    from brain_client import BrainClient, BrainClientError

app = FastAPI(
    title="Fly Brain Agent",
    description="LIF simulation of the Drosophila Male CNS (MaleCNS v1.0 via NeuPrint).",
    version="0.2.0",
)

# Default seed neuron for auto-selecting an interconnected network.
DEFAULT_SEED = 45882


class StimulusRequest(BaseModel):
    """An external stimulus applied to a set of bodyIds."""

    body_ids: list[int] | None = Field(
        None,
        description=(
            "Neuron bodyIds to load and simulate. If omitted, a small "
            "interconnected group is auto-selected around a random seed."
        ),
    )
    stimulus: list[float] = Field(
        ...,
        description=(
            "External input current, one entry per bodyId "
            "(e.g. simulated light or directional input)."
        ),
    )
    input_steps: int = Field(
        1, ge=1, le=1000, description="How many discrete LIF steps to run."
    )
    include_potentials: bool = Field(
        False, description="Also return final membrane potentials (verbose)."
    )


class StimulusResponse(BaseModel):
    body_ids: list[int]
    firing_rates: list[float]   # spikes per simulated step, per neuron (0..1)
    spike_counts: list[int]     # total spikes over all steps, per neuron
    downstream: list[float]     # firing rates restricted to post-synaptic neurons
    downstream_body_ids: list[int]
    final_potentials: list[float] | None = None


def _token() -> str:
    token = os.environ.get("NEUPRINT_APPLICATION_TOKEN")
    if not token:
        warnings.warn(
            "NEUPRINT_APPLICATION_TOKEN is not set; NeuPrint calls will fail.",
            RuntimeWarning,
        )
    return token


def _build_simulator(body_ids: list[int] | None) -> tuple[LIFSimulator, list[int]]:
    """Load connectivity and build a LIF network, resolving an auto-selected
    group when no bodyIds are supplied. Returns (simulator, resolved_body_ids)."""
    try:
        client = BrainClient(token=_token())
        if body_ids is None:
            seed = DEFAULT_SEED if os.environ.get("FLY_SEED") is None else int(os.environ["FLY_SEED"])
            body_ids = client.connected_subset([seed], k=3)
        W = client.connectivity_matrix(body_ids)
    except BrainClientError as exc:
        raise HTTPException(status_code=502, detail=f"NeuPrint error: {exc}")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {exc}")
    return LIFSimulator(W), list(body_ids)


@app.get("/")
async def index() -> dict:
    return {"service": "fly-brain-agent", "status": "ready"}


@app.post("/stimulate", response_model=StimulusResponse)
async def stimulate(request: StimulusRequest) -> dict:
    """Run the LIF dynamics over `input_steps` and return the neuron outputs."""
    sim, body_ids = _build_simulator(request.body_ids)
    N = len(body_ids)
    if len(request.stimulus) != N:
        raise HTTPException(
            status_code=422,
            detail=(
                f"stimulus must have {N} entries (one per bodyId), "
                f"got {len(request.stimulus)}."
            ),
        )

    stimulus_seq = np.tile(
        np.asarray(request.stimulus, dtype=np.float64), (request.input_steps, 1)
    )
    result = sim.run(stimulus_seq)  # dict with raster / firing_rates / ...
    raster = result["raster"]
    spike_counts = raster.sum(axis=0)
    firing_rates = (spike_counts / request.input_steps).tolist()

    # Downstream readout: neurons that receive at least one synapse from
    # another neuron in the loaded set (post-synaptic targets).
    post_recipients = np.flatnonzero(sim.W.sum(axis=0) > 0.0).tolist()
    downstream_body_ids = [body_ids[i] for i in post_recipients]
    downstream_rates = [firing_rates[i] for i in post_recipients]

    return StimulusResponse(
        body_ids=body_ids,
        firing_rates=firing_rates,
        spike_counts=spike_counts.astype(int).tolist(),
        downstream=downstream_rates,
        downstream_body_ids=downstream_body_ids,
        final_potentials=(sim.v.tolist() if request.include_potentials else None),
    )


# --------------------------------------------------------------------------- #
#  Local visualisation script entry:  python main.py visualize [--body-ids ...] [--steps N] [--out FILE]
# --------------------------------------------------------------------------- #

def _run_local(
    client: BrainClient,
    stimulus: np.ndarray,
    body_ids: list[int] | None,
    steps: int,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Resolve the neuron group, build connectivity, run LIF, return
    (raster, body_ids, weights_matrix)."""
    if body_ids is None:
        body_ids = client.connected_subset([DEFAULT_SEED], k=4)
    W = client.connectivity_matrix(body_ids)
    sim = LIFSimulator(W)

    stim = np.asarray(stimulus, dtype=np.float64).ravel()
    n = len(body_ids)
    if len(stim) > n:
        stim = stim[:n]
    pad = np.zeros(n, dtype=np.float64)
    pad[: len(stim)] = stim
    seq = np.tile(pad, (steps, 1))
    res = sim.run(seq)
    return res["raster"], list(body_ids), W


def visualise_cli(argv: list[str]) -> int:
    """CLI entry: run a LIF sim on a connected network and write the animated
    3D HTML of the real skeletons."""
    steps = 20
    stim = None
    body_ids = None
    scale = 1.0
    out = "brain_sim_3d.html"

    def parse_bodylist(s: str) -> list[int]:
        return [int(x) for x in s.replace(",", " ").split()]

    i = 0
    try:
        while i < len(argv):
            a = argv[i]
            if a == "--steps":
                steps = int(argv[i + 1]); i += 2
            elif a in ("--input", "--stimulus"):
                stim = [float(x) for x in argv[i + 1].split(",")]; i += 2
            elif a == "--body-ids":
                body_ids = parse_bodylist(argv[i + 1]); i += 2
            elif a == "--scale":
                scale = float(argv[i + 1]); i += 2
            elif a == "-o":
                out = argv[i + 1]; i += 2
            else:
                # unknown positional -> treat as comma list of bodyIds
                if argv[i].replace(",", "").isdigit():
                    body_ids = parse_bodylist(argv.pop(i))
                else:
                    print(f"Unknown option: {argv[i]}"); return 2
        # arg passed
    except (IndexError, ValueError) as exc:
        print(f"Bad argument: {exc}"); return 2

    try:
        client = BrainClient(token=_token())
        if body_ids is not None:
            n = len(body_ids)
            default_stim = np.zeros(n, dtype=np.float64)
            default_stim[0] = 25.0  # stimulate the seed strongly
            if stim is None:
                stim = default_stim
        else:
            if stim is None:
                stim = np.array([25.0], dtype=np.float64)
        stim = np.asarray(stim, dtype=np.float64)
        # If user gave fewer stimulus values than neurons, the sim will pad
        # with the same seed input on the first neuron and zeros elsewhere.
        raster, net, W = _run_local(client, stim, body_ids, steps)
    except BrainClientError as exc:
        print(f"[error] NeuPrint: {exc}")
        return 1

    if steps > raster.shape[0]:
        steps = raster.shape[0]
    # Use the full raster for time-resolved animation.
    raster = raster[:steps]

    # Build the viewer from real skeletons.
    try:
        from visualise_3d import FlyBrainViewer3D
        neurons = client.get_skeletons(net)
        viewer = FlyBrainViewer3D(neurons, net, raster, scale_to_um=scale)
        viewer.save(out)
        print(f"Saved animated 3D figure to: {os.path.abspath(out)}")
        print(f"  neurons: {len(net)}  skeletons nodes: {viewer.total_nodes}  steps: {steps}")
        return 0
    except Exception as exc:
        print(f"[error] visualisation failed: {exc}")
        return 1


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "visualize":
        sys.exit(visualise_cli(args[1:]))
    if args and args[0] == "run":
        import uvicorn

        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", "8000")),
            reload=False,
        )
    else:
        import uvicorn

        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=int(os.environ.get("PORT", "8000")),
            reload=True,
        )