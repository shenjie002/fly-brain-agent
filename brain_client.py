"""
brain_client.py
---------------
Connect to the MaleCNS v1.0 / NeuPrint Drosophila connectome and extract a
synaptic weight matrix W for a set of bodyIds.

Authentication token is read from the NEUPRINT_APPLICATION_TOKEN environment
variable (falling back to neuprint-python's own NEUPRINT_APPLICATION_CREDENTIALS
if the former is absent).
"""

from __future__ import annotations

import os
from typing import Sequence

import numpy as np
from neuprint import Client, fetch_adjacencies

# MaleCNS v1.0 — the full Drosophila male central nervous system connectome,
# served publicly on the Janelia neuPrint instance.
DEFAULT_ENDPOINT = "https://neuprint.janelia.org"
DEFAULT_DATASET = "male-cns:v1.0"


class BrainClientError(Exception):
    """Raised for any problem authenticating, connecting or querying NeuPrint."""


def _load_token() -> str:
    """Read the NeuPrint token from the environment, with a clear error."""
    token = os.environ.get("NEUPRINT_APPLICATION_TOKEN")
    if not token:
        # neuprint-python's own convention (a JSON credentials document).
        token = os.environ.get("NEUPRINT_APPLICATION_CREDENTIALS")
    if not token:
        raise BrainClientError(
            "No NeuPrint token found. Set the NEUPRINT_APPLICATION_TOKEN "
            "environment variable (or NEUPRINT_APPLICATION_CREDENTIALS)."
        )
    return token


class BrainClient:
    """Thin wrapper around neuprint-python used to build a connectivity matrix."""

    def __init__(
        self,
        dataset: str = DEFAULT_DATASET,
        endpoint: str = DEFAULT_ENDPOINT,
        token: str | None = None,
    ) -> None:
        token = token or _load_token()
        self.endpoint = endpoint
        self.dataset = dataset
        try:
            self.client = Client(endpoint, dataset=dataset, token=token)
        except RuntimeError as exc:  # endpoint / dataset / token problems
            raise BrainClientError(f"Failed to initialise NeuPrint client: {exc}") from exc

    def connectivity_matrix(self, body_ids: Sequence[int]) -> np.ndarray:
        """
        Return an N x N matrix W where W[i, j] is the total synapse count
        from neuron i (pre / upstream, W row) to neuron j (post / downstream,
        W column). Rows/columns follow the order of `body_ids`; a zero entry
        means no detected connection between that pair.
        """
        ids = list(dict.fromkeys(int(b) for b in body_ids))  # dedupe, keep order
        n = len(ids)
        index = {bid: i for i, bid in enumerate(ids)}
        W = np.zeros((n, n), dtype=np.float64)
        if n == 0:
            return W

        try:
            # Returns (neurons_df, roi_conn_df); roi_conn_df carries the
            # per-ROI rows: bodyId_pre, bodyId_post, roi, weight.
            _, roi_conn = fetch_adjacencies(
                sources=ids, targets=ids, client=self.client,
            )
        except RuntimeError as exc:  # network / query / CREDENTIALS failures
            raise BrainClientError(f"NeuPrint query failed: {exc}") from exc

        if roi_conn is not None and len(roi_conn):
            # Aggregate per-ROI weights into a single total strength per pair.
            total = (
                roi_conn.groupby(["bodyId_pre", "bodyId_post"], as_index=False)["weight"].sum()
            )
            for _, row in total.iterrows():
                pre, post = int(row["bodyId_pre"]), int(row["bodyId_post"])
                if pre in index and post in index:
                    W[index[pre], index[post]] = float(row["weight"])

        return W

    def connected_subset(
        self,
        seed_ids: Sequence[int],
        k: int = 3,
        *,
        include_upstream: bool = True,
        include_downstream: bool = True,
    ) -> list[int]:
        """
        Return a small group of bodyIds that are actually interconnected, so a
        downstream simulation always has real synaptic drive.

        The returned list starts with the seed(s), then adds up to `k` neurons
        downstream of the seed(s) and up to `k` upstream, then closes the loop
        by returning each newly-added neuron's downstream within the group.
        This produces a cyclic (or at least multiply-connected) subgraph with
        non-zero weight entries.

        Returns the resulting bodyId list (deduped, seeds first).
        """
        seeds = list(dict.fromkeys(int(b) for b in seed_ids))
        if not seeds:
            raise BrainClientError("at least one seed bodyId is required")
        group: list[int] = list(seeds)

        def _extend(criteria_sources, criteria_targets, label: str, k: int) -> None:
            """query one direction (upstream vs downstream) and add to `group`."""
            try:
                _, conn = fetch_adjacencies(
                    sources=criteria_sources,
                    targets=criteria_targets,
                    client=self.client,
                )
            except RuntimeError as exc:
                raise BrainClientError(f"NeuPrint query failed: {exc}") from exc
            if conn is None or not len(conn):
                return
            # iterate in strongest-first order
            col = "bodyId_post" if label == "downstream" else "bodyId_pre"
            other = conn.sort_values("weight", ascending=False)[col].tolist()
            added = 0
            for bid in other:
                bid = int(bid)
                if bid not in group:
                    group.append(bid)
                    added += 1
                    if added >= k:
                        break

        if include_downstream:
            _extend(seeds, None, "downstream", k)
        if include_upstream:
            _extend(None, seeds, "upstream", k)
        # close the loop: each returned neighbour's downstream *within* group
        _extend(group, group, "downstream", k)
        return group

    def get_skeletons(self, body_ids: Sequence[int]) -> "navis.NeuronList":
        """
        Fetch the **real 3D anatomy (skeletons)** of the given neurons as a
        navis.NeuronList (TreeNeurons with node x/y/z coordinates).

        Uses the same NeuPrint client + token as the connectivity queries, and
        navis' neuprint interface to download each neuron's SWC skeleton into a
        navigable 3D object. Nodes carry columns: node_id, parent_id, x, y, z,
        radius. The x/y/z coordinates are in the dataset's raw voxel space
        (large integer values, ~40k-60k for MaleCNS); scale to physical microns
        yourself if the voxel resolution is known.

        Raises
        ------
        BrainClientError
            If navis is not installed, or a neuron has no available skeleton.
        """
        try:
            from navis.interfaces import neuprint as nvn  # lazy heavy import
        except Exception as exc:  # pragma: no cover - environment
            raise BrainClientError(
                f"navis not installed ({exc}). Run: pip install navis"
            ) from exc

        ids = list(dict.fromkeys(int(b) for b in body_ids))
        if not ids:
            raise BrainClientError("at least one bodyId is required")
        try:
            neurons = nvn.fetch_skeletons(ids, client=self.client)
        except RuntimeError as exc:  # network / token / missing skeletons
            raise BrainClientError(f"Failed to fetch skeletons from NeuPrint: {exc}") from exc
        if neurons is None or len(neurons) == 0:
            raise BrainClientError(
                f"No skeleton data returned for bodyIds {ids}"
            )
        return neurons


def load_weights(
    body_ids: Sequence[int],
    *,
    dataset: str = DEFAULT_DATASET,
    endpoint: str = DEFAULT_ENDPOINT,
    token: str | None = None,
) -> np.ndarray:
    """Convenience helper: build a BrainClient and return its matrix."""
    client = BrainClient(dataset=dataset, endpoint=endpoint, token=token)
    return client.connectivity_matrix(body_ids)