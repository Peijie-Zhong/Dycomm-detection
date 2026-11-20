#!/usr/bin/env python3
"""Dynamic community detection on link streams with a TGAT encoder.

The TGAT encoder produces one embedding per observed (node, time) pair and is
trained with two self-supervised tasks. Both use a temporal random-walk
diffusion matrix, built around a sampled centre time, as their signal:

1. Local contrastive loss: embeddings of node-times in the same sampled
   sub-community are pulled together, weighted by the high-order modularity
   residual (diffusion co-occurrence minus its null-model expectation).
2. Pairwise-alignment loss: pairwise distances between embeddings are aligned
   with pairwise distances between the CountSketch of each node's diffusion row.

For each input link stream the script runs a short pilot over several temporal
kernel settings (raw_tau:cutoff_multiplier), fully trains the best finalists,
and keeps the checkpoint whose K-means partition has the highest AMI against
the ground-truth communities (AMI / ARI from evaluation.py).

Note: configurations and checkpoints are selected with the same ground-truth
labels that the reported AMI is computed on, so the reported AMI is an
optimistic (oracle-selected) estimate.

Input CSV columns: source, destination (or target), timestamp. Ground truth
(required): columns source_commu, destination_commu, either in the input file
itself or in a same-named file under --ground-truth-dir.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from evaluation import build_partition_from_csv, dynamic_ari, dynamic_mi
from module import TGAN


NodeTime = Tuple[int, int]

SOURCE_ALIASES = ("source", "model_author")
DESTINATION_ALIASES = ("destination", "target", "contrafactum_author")


# =============================================================================
# Data
# =============================================================================


@dataclass
class LinkStreamData:
    df: pd.DataFrame  # numeric link stream, sorted by (timestamp, source, destination)
    node_to_idx: Dict[int, int]  # node id -> TGAT index (0 is the padding index)
    src: np.ndarray
    dst: np.ndarray
    ts: np.ndarray  # timestamps rescaled to [0, 1]
    raw_ts: np.ndarray
    edge_idx: np.ndarray
    pairs: List[NodeTime]  # observed (node id, raw timestamp) pairs
    pair_to_idx: Dict[NodeTime, int]
    pair_ts: np.ndarray  # rescaled timestamp of each pair


def read_csv_any_encoding(path: Path) -> pd.DataFrame:
    last_error: UnicodeDecodeError | None = None
    for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(path, encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    raise last_error if last_error is not None else RuntimeError(f"Unable to read {path}")


def find_column(frame: pd.DataFrame, aliases: Sequence[str], path: Path) -> str:
    for name in aliases:
        if name in frame.columns:
            return name
    raise ValueError(f"{path}: none of the columns {list(aliases)} found")


def load_link_stream(path: Path) -> Tuple[pd.DataFrame, Dict[int, str]]:
    """Read a link stream and map node names to integer ids 1..N.

    Returns the numeric frame (source, destination, timestamp, source_name,
    destination_name, other columns) and the id -> original-name map.
    """
    raw = read_csv_any_encoding(path)
    source_col = find_column(raw, SOURCE_ALIASES, path)
    destination_col = find_column(raw, DESTINATION_ALIASES, path)
    if "timestamp" not in raw.columns:
        raise ValueError(f"{path}: missing column 'timestamp'")

    source_names = raw[source_col].astype(str)
    destination_names = raw[destination_col].astype(str)
    names = sorted(set(source_names) | set(destination_names))
    name_to_id = {name: index + 1 for index, name in enumerate(names)}
    id_to_name = {index: name for name, index in name_to_id.items()}

    frame = raw.drop(columns=[source_col, destination_col]).copy()
    frame["source_name"] = source_names
    frame["destination_name"] = destination_names
    frame["source"] = source_names.map(name_to_id).astype(int)
    frame["destination"] = destination_names.map(name_to_id).astype(int)
    frame["timestamp"] = frame["timestamp"].astype(int)
    frame = frame.sort_values(["timestamp", "source", "destination"]).reset_index(drop=True)
    first = ["source", "destination", "timestamp", "source_name", "destination_name"]
    frame = frame[first + [column for column in frame.columns if column not in first]]
    return frame, id_to_name


def rescale_times(raw_ts: np.ndarray) -> np.ndarray:
    span = float(raw_ts.max() - raw_ts.min())
    if span <= 0:
        return np.zeros_like(raw_ts, dtype=np.float32)
    return ((raw_ts - float(raw_ts.min())) / span).astype(np.float32)


def observed_node_times(df: pd.DataFrame) -> Tuple[List[NodeTime], Dict[NodeTime, int]]:
    pairs: List[NodeTime] = []
    pair_to_idx: Dict[NodeTime, int] = {}
    for source, destination, timestamp in df[["source", "destination", "timestamp"]].itertuples(index=False, name=None):
        for node in (int(source), int(destination)):
            key = (node, int(timestamp))
            if key not in pair_to_idx:
                pair_to_idx[key] = len(pairs)
                pairs.append(key)
    return pairs, pair_to_idx


def prepare_data(frame: pd.DataFrame) -> LinkStreamData:
    df = frame.sort_values(["timestamp", "source", "destination"]).reset_index(drop=True)
    nodes = sorted(set(df["source"].astype(int)) | set(df["destination"].astype(int)))
    node_to_idx = {node: idx + 1 for idx, node in enumerate(nodes)}
    src = df["source"].map(node_to_idx).to_numpy(np.int64)
    dst = df["destination"].map(node_to_idx).to_numpy(np.int64)
    raw_ts = df["timestamp"].to_numpy(np.float32)
    ts = rescale_times(raw_ts)
    edge_idx = np.arange(1, len(df) + 1, dtype=np.int64)
    pairs, pair_to_idx = observed_node_times(df)
    raw_to_internal = {int(raw): float(internal) for raw, internal in zip(raw_ts, ts)}
    pair_ts = np.asarray([raw_to_internal[int(timestamp)] for _node, timestamp in pairs], dtype=np.float32)
    return LinkStreamData(df, node_to_idx, src, dst, ts, raw_ts, edge_idx, pairs, pair_to_idx, pair_ts)


# =============================================================================
# TGAT encoder
# =============================================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class TemporalNeighborFinder:
    """Non-causal temporal neighbour lookup for TGAN (replaces graph.NeighborFinder).

    For a query (node u, time t) the candidates are all interactions of u,
    including those at exactly t. graph.NeighborFinder excludes same-time
    interactions and its find_before drops the closest past neighbour; both
    are fixed here.

    TGAN calls get_temporal_neighbor_bidirection when use_bidirection=True and
    get_temporal_neighbor otherwise. Both return (nodes, edge ids, times) of
    shape [B, num_neighbors], sorted by time and right-aligned, with node 0 as
    padding.
    """

    def __init__(self, data: LinkStreamData, uniform: bool):
        num_model_nodes = len(data.node_to_idx) + 1
        neighbors: List[List[Tuple[float, int, int]]] = [[] for _ in range(num_model_nodes)]
        for source, destination, edge_idx, timestamp in zip(data.src, data.dst, data.edge_idx, data.ts):
            neighbors[int(source)].append((float(timestamp), int(edge_idx), int(destination)))
            neighbors[int(destination)].append((float(timestamp), int(edge_idx), int(source)))
        self.ts: List[np.ndarray] = []
        self.eidx: List[np.ndarray] = []
        self.nodes: List[np.ndarray] = []
        for items in neighbors:
            items.sort()  # by time, then edge id
            self.ts.append(np.asarray([item[0] for item in items], dtype=np.float64))
            self.eidx.append(np.asarray([item[1] for item in items], dtype=np.int64))
            self.nodes.append(np.asarray([item[2] for item in items], dtype=np.int64))
        self.uniform = uniform

    def _select(self, node: int, cut_time: float, num_neighbors: int, include_future: bool) -> np.ndarray:
        """Positions (into the node's sorted lists) of the selected neighbours, in time order."""
        ts = self.ts[node]
        if len(ts) == 0:
            return np.empty(0, dtype=np.int64)
        end = len(ts) if include_future else int(np.searchsorted(ts, cut_time, side="right"))
        if end == 0:
            return np.empty(0, dtype=np.int64)
        if self.uniform:
            return np.sort(np.random.randint(0, end, num_neighbors))
        if not include_future:
            return np.arange(max(0, end - num_neighbors), end, dtype=np.int64)
        # The num_neighbors closest in |t' - t| lie within this window around the same-time block.
        left = int(np.searchsorted(ts, cut_time, side="left"))
        right = int(np.searchsorted(ts, cut_time, side="right"))
        start, stop = max(0, left - num_neighbors), min(len(ts), right + num_neighbors)
        window = np.arange(start, stop, dtype=np.int64)
        nearest = window[np.argsort(np.abs(ts[window] - cut_time), kind="stable")[:num_neighbors]]
        return np.sort(nearest)

    def _lookup(self, src_idx_l, cut_time_l, num_neighbors: int, include_future: bool):
        out_nodes = np.zeros((len(src_idx_l), num_neighbors), dtype=np.int32)
        out_eidx = np.zeros((len(src_idx_l), num_neighbors), dtype=np.int32)
        out_ts = np.zeros((len(src_idx_l), num_neighbors), dtype=np.float32)
        for row, (node, cut_time) in enumerate(zip(src_idx_l, cut_time_l)):
            positions = self._select(int(node), float(cut_time), num_neighbors, include_future)
            if len(positions) == 0:
                continue
            width = len(positions)
            out_nodes[row, num_neighbors - width:] = self.nodes[int(node)][positions]
            out_eidx[row, num_neighbors - width:] = self.eidx[int(node)][positions]
            out_ts[row, num_neighbors - width:] = self.ts[int(node)][positions]
        return out_nodes, out_eidx, out_ts

    def get_temporal_neighbor_bidirection(self, src_idx_l, cut_time_l, num_neighbors=20):
        """The num_neighbors interactions closest in time, before, at or after cut_time."""
        return self._lookup(src_idx_l, cut_time_l, num_neighbors, include_future=True)

    def get_temporal_neighbor(self, src_idx_l, cut_time_l, num_neighbors=20):
        """The latest num_neighbors interactions at or before cut_time."""
        return self._lookup(src_idx_l, cut_time_l, num_neighbors, include_future=False)


def build_neighbor_finder(data: LinkStreamData, uniform: bool) -> TemporalNeighborFinder:
    return TemporalNeighborFinder(data, uniform)


def build_raw_features(data: LinkStreamData, node_feat_dim: int, edge_feat_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    """Node features: random projection (node_feat_dim > 0) or one-hot ids.
    Edge features: a constant 1 and the rescaled edge time."""
    num_model_nodes = len(data.node_to_idx) + 1
    if 0 < node_feat_dim < num_model_nodes:
        rng = np.random.default_rng(7)
        node_features = rng.normal(size=(num_model_nodes, node_feat_dim)).astype(np.float32)
        node_features /= math.sqrt(float(node_feat_dim))
        node_features[0] = 0.0
    else:
        node_features = np.zeros((num_model_nodes, num_model_nodes), dtype=np.float32)
        node_features[1:, 1:] = np.eye(num_model_nodes - 1, dtype=np.float32)

    edge_features = np.zeros((len(data.df) + 1, edge_feat_dim), dtype=np.float32)
    edge_features[1:, 0] = 1.0
    if edge_feat_dim > 1:
        edge_features[1:, 1] = data.ts / max(float(data.ts.max()), 1.0)
    return node_features, edge_features


def build_encoder(args: argparse.Namespace, data: LinkStreamData) -> TGAN:
    node_features, edge_features = build_raw_features(data, args.raw_node_dim, args.raw_edge_dim)
    encoder = TGAN(
        build_neighbor_finder(data, args.uniform),
        node_features,
        edge_features,
        num_layers=args.n_layer,
        use_time="time",
        agg_method="attn",
        attn_mode="prod",
        seq_len=args.n_degree,
        n_head=args.n_head,
        drop_out=args.drop_out,
        node_dim=args.node_dim,
        time_dim=args.time_dim,
        use_bidirection=args.neighbor_mode == "bidirectional",
    )
    return encoder.to(args.device)


def encode_pairs(encoder: TGAN, data: LinkStreamData, args: argparse.Namespace) -> np.ndarray:
    encoder.eval()
    nodes = np.asarray([data.node_to_idx[node] for node, _ in data.pairs], dtype=np.int64)
    times = data.pair_ts.astype(np.float32)
    chunks: List[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, len(data.pairs), args.encode_batch_size):
            end = start + args.encode_batch_size
            emb = encoder.get_node_event_embeddings(nodes[start:end], times[start:end], args.n_degree)
            chunks.append(emb.detach().cpu().numpy())
    return np.vstack(chunks).astype(np.float32)


# =============================================================================
# Diffusion sub-community sampler (training signal for both losses)
# =============================================================================


@dataclass
class TemporalConfig:
    raw_tau: float
    cutoff_multiplier: float

    @property
    def name(self) -> str:
        return f"tau{self.raw_tau:g}_cutoff{self.cutoff_multiplier:g}"


@dataclass
class EventTransition:
    dst_nodes: List[np.ndarray]
    dst_pairs: List[np.ndarray]
    cumulative: List[np.ndarray]
    node_strength: np.ndarray


class DiffusionSubcommunitySampler:
    """Samples a batch of node-times around a centre time t.

    1. Build a time-weighted transition matrix P_t from edges within
       |edge_time - t| <= cutoff, with weights exp(-|edge_time - t| / tau).
    2. Diffusion row of node u: sum_k decay^(k-1) * (P_t^k)[u, :], k = 1..order.
    3. Pick root nodes, grow each into a sub-community from nodes whose
       diffusion score exceeds the null-model expectation.
    4. Lift the selected nodes to their nearest observed node-times, and return
       the high-order modularity residual (local contrastive signal) and the
       CountSketch of each diffusion row (pairwise-alignment target).
    """

    def __init__(self, data: LinkStreamData, args: argparse.Namespace, config: TemporalConfig):
        self.data = data
        self.args = args
        self.num_nodes = len(data.node_to_idx)
        self.pair_nodes = np.asarray([data.node_to_idx[node] for node, _ in data.pairs], dtype=np.int64)
        self.pair_times = data.pair_ts.astype(np.float32)

        span = float(data.raw_ts.max() - data.raw_ts.min())
        self.tau = float(config.raw_tau) / (span if span > 0 else 1.0)
        self.cutoff = self.tau * float(config.cutoff_multiplier) if config.cutoff_multiplier > 0 else None
        self.root_tau = self.tau * float(args.root_tau_multiplier)

        center_seed = args.center_seed if args.center_seed is not None else args.sampler_seed
        self.rng = np.random.default_rng(args.sampler_seed + 93011)
        sketch_rng = np.random.default_rng(args.sketch_seed + 93017)
        self.center_rng = np.random.default_rng(center_seed + 93023)
        self.node_buckets = sketch_rng.integers(0, args.sketch_dim, size=self.num_nodes + 1, dtype=np.int64)
        self.node_signs = sketch_rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=self.num_nodes + 1).astype(np.float32)
        self.node_signs[0] = 0.0

        self.unique_center_times, self.center_time_activity = np.unique(self.pair_times, return_counts=True)
        self._center_cycle = np.empty(0, dtype=np.float32)
        self._center_cycle_position = 0

        self.edge_pair_left = np.asarray(
            [data.pair_to_idx[(int(row.source), int(row.timestamp))] for row in data.df.itertuples(index=False)],
            dtype=np.int64,
        )
        self.edge_pair_right = np.asarray(
            [data.pair_to_idx[(int(row.destination), int(row.timestamp))] for row in data.df.itertuples(index=False)],
            dtype=np.int64,
        )

        self.node_pair_indices: List[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(self.num_nodes + 1)]
        self.node_pair_times: List[np.ndarray] = [np.empty(0, dtype=np.float32) for _ in range(self.num_nodes + 1)]
        for node in range(1, self.num_nodes + 1):
            indices = np.flatnonzero(self.pair_nodes == node)
            order = np.argsort(self.pair_times[indices], kind="mergesort")
            self.node_pair_indices[node] = indices[order].astype(np.int64)
            self.node_pair_times[node] = self.pair_times[indices][order].astype(np.float32)

        self._transition_cache: Dict[float, EventTransition] = {}
        self._prob_cache: Dict[float, List[Dict[int, float]]] = {}
        self._batch_diffusion_rows: Dict[Tuple[int, float], Dict[int, float]] = {}
        self.coefficient_sum = sum(
            float(args.diffusion_decay) ** order for order in range(max(1, int(args.diffusion_order)))
        )

    # ----- centre time ---------------------------------------------------------

    def _refresh_center_cycle(self) -> None:
        if self.args.center_sampling_mode == "unique_uniform":
            self._center_cycle = self.center_rng.permutation(self.unique_center_times)
        else:  # unique_activity
            probabilities = self.center_time_activity.astype(np.float64)
            probabilities /= probabilities.sum()
            positions = self.center_rng.choice(
                len(self.unique_center_times), size=len(self.unique_center_times), replace=False, p=probabilities
            )
            self._center_cycle = self.unique_center_times[positions]
        self._center_cycle_position = 0

    def sample_center_time(self) -> float:
        mode = self.args.center_sampling_mode
        if mode == "node_time":
            return float(self.pair_times[int(self.rng.integers(0, len(self.pair_times)))])
        if mode == "node_time_independent":
            return float(self.pair_times[int(self.center_rng.integers(0, len(self.pair_times)))])
        if self._center_cycle_position >= len(self._center_cycle):
            self._refresh_center_cycle()
        center_time = float(self._center_cycle[self._center_cycle_position])
        self._center_cycle_position += 1
        return center_time

    # ----- transition matrix and diffusion ------------------------------------

    def transition(self, center_time: float) -> EventTransition:
        key = float(center_time)
        if self.args.cache_transitions and key in self._transition_cache:
            return self._transition_cache[key]
        dst_nodes_tmp: List[List[int]] = [[] for _ in range(self.num_nodes + 1)]
        dst_pairs_tmp: List[List[int]] = [[] for _ in range(self.num_nodes + 1)]
        weights_tmp: List[List[float]] = [[] for _ in range(self.num_nodes + 1)]
        node_strength = np.zeros(self.num_nodes + 1, dtype=np.float64)
        if self.cutoff is None:
            edge_indices = range(len(self.data.src))
        else:
            edge_indices = np.flatnonzero(
                (self.data.ts >= center_time - self.cutoff) & (self.data.ts <= center_time + self.cutoff)
            )
        for edge_index in edge_indices:
            weight = math.exp(-abs(float(self.data.ts[edge_index]) - center_time) / max(self.tau, 1e-12))
            source = int(self.data.src[edge_index])
            destination = int(self.data.dst[edge_index])
            dst_nodes_tmp[source].append(destination)
            dst_pairs_tmp[source].append(int(self.edge_pair_right[edge_index]))
            weights_tmp[source].append(weight)
            dst_nodes_tmp[destination].append(source)
            dst_pairs_tmp[destination].append(int(self.edge_pair_left[edge_index]))
            weights_tmp[destination].append(weight)
            node_strength[source] += weight
            node_strength[destination] += weight
        dst_nodes: List[np.ndarray] = []
        dst_pairs: List[np.ndarray] = []
        cumulative: List[np.ndarray] = []
        for nodes, pairs, weights in zip(dst_nodes_tmp, dst_pairs_tmp, weights_tmp):
            if not weights:
                dst_nodes.append(np.empty(0, dtype=np.int64))
                dst_pairs.append(np.empty(0, dtype=np.int64))
                cumulative.append(np.empty(0, dtype=np.float32))
                continue
            weights_np = np.asarray(weights, dtype=np.float64)
            dst_nodes.append(np.asarray(nodes, dtype=np.int64))
            dst_pairs.append(np.asarray(pairs, dtype=np.int64))
            cumulative.append(np.cumsum(weights_np / np.clip(weights_np.sum(), 1e-12, None)).astype(np.float32))
        result = EventTransition(dst_nodes, dst_pairs, cumulative, node_strength)
        if self.args.cache_transitions:
            self._transition_cache[key] = result
        return result

    def node_prob_rows(self, center_time: float) -> List[Dict[int, float]]:
        key = float(center_time)
        cached = self._prob_cache.get(key)
        if cached is not None:
            return cached
        transition = self.transition(key)
        rows: List[Dict[int, float]] = []
        for node in range(self.num_nodes + 1):
            destinations = transition.dst_nodes[node]
            if len(destinations) == 0:
                rows.append({})
                continue
            cumulative = transition.cumulative[node]
            probabilities = np.diff(np.concatenate([np.asarray([0.0], dtype=np.float32), cumulative])).astype(np.float64)
            row: Dict[int, float] = {}
            for destination, probability in zip(destinations, probabilities):
                row[int(destination)] = row.get(int(destination), 0.0) + float(probability)
            rows.append(row)
        if self.args.cache_transitions:
            self._prob_cache[key] = rows
        return rows

    def diffusion_row(self, node: int, center_time: float) -> Dict[int, float]:
        """sum_k decay^(k-1) * (P_t^k)[node, :], with the node itself removed. Cached per batch."""
        cache_key = (int(node), float(center_time))
        cached = self._batch_diffusion_rows.get(cache_key)
        if cached is not None:
            return cached
        rows = self.node_prob_rows(center_time)
        current = {int(node): 1.0}
        output: Dict[int, float] = {}
        coefficient = 1.0
        for _order in range(max(1, int(self.args.diffusion_order))):
            nxt: Dict[int, float] = {}
            for source, source_probability in current.items():
                for destination, probability in rows[source].items():
                    nxt[destination] = nxt.get(destination, 0.0) + source_probability * probability
            if not nxt:
                break
            for destination, probability in nxt.items():
                output[destination] = output.get(destination, 0.0) + coefficient * probability
            current = nxt
            coefficient *= float(self.args.diffusion_decay)
        output.pop(int(node), None)
        self._batch_diffusion_rows[cache_key] = output
        return output

    # ----- sub-community construction -----------------------------------------

    def sample_pairs_for_node(self, node: int, center_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Lift a node to its observed node-time(s) closest to the centre time."""
        indices = self.node_pair_indices[int(node)]
        times = self.node_pair_times[int(node)]
        if len(indices) == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        if self.cutoff is None:
            left, right = 0, len(indices)
        else:
            left = int(np.searchsorted(times, center_time - self.cutoff, side="left"))
            right = int(np.searchsorted(times, center_time + self.cutoff, side="right"))
        if right <= left:
            pos = int(np.searchsorted(times, center_time, side="left"))
            if pos <= 0:
                return np.asarray([int(indices[0])], dtype=np.int64), np.ones(1, dtype=np.float32)
            if pos >= len(indices):
                return np.asarray([int(indices[-1])], dtype=np.int64), np.ones(1, dtype=np.float32)
            before_is_closer = abs(float(times[pos - 1] - center_time)) <= abs(float(times[pos] - center_time))
            nearest = int(indices[pos - 1] if before_is_closer else indices[pos])
            return np.asarray([nearest], dtype=np.int64), np.ones(1, dtype=np.float32)
        selected = indices[left:right]
        selected_times = times[left:right]
        if self.args.lifting_mode == "nearest" or int(self.args.lifting_top_q) <= 1:
            if self.args.event_time_selection == "nearest":
                pos = int(np.argmin(np.abs(selected_times - center_time)))
                return np.asarray([int(selected[pos])], dtype=np.int64), np.ones(1, dtype=np.float32)
            weights = np.exp(-np.abs(selected_times - center_time) / max(self.root_tau, 1e-12)).astype(np.float64)
            weights = weights / np.clip(weights.sum(), 1e-12, None)
            return np.asarray([int(self.rng.choice(selected, p=weights))], dtype=np.int64), np.ones(1, dtype=np.float32)
        take = min(int(self.args.lifting_top_q), len(selected))
        positions = np.argsort(np.abs(selected_times - center_time), kind="mergesort")[:take]
        selected = selected[positions].astype(np.int64)
        selected_times = selected_times[positions]
        tau = max(self.root_tau * float(self.args.lifting_tau_multiplier), 1e-12)
        weights = np.exp(-np.abs(selected_times - center_time) / tau).astype(np.float64)
        weights = weights / np.clip(weights.sum(), 1e-12, None)
        return selected, weights.astype(np.float32)

    def root_nodes(self, center_time: float, selected_nodes: set[int]) -> np.ndarray:
        transition = self.transition(center_time)
        available = np.asarray(
            [
                node
                for node in range(1, self.num_nodes + 1)
                if node not in selected_nodes and transition.node_strength[node] > 0
            ],
            dtype=np.int64,
        )
        if len(available) == 0:
            return np.empty(0, dtype=np.int64)
        size = min(int(self.args.num_subcommunities), len(available))
        if self.args.root_sample == "strength":
            weights = transition.node_strength[available].astype(np.float64)
            if float(weights.sum()) > 0:
                return self.rng.choice(available, size=size, replace=False, p=weights / weights.sum())
        return self.rng.choice(available, size=size, replace=False)

    def expand_subcommunity(self, root: int, center_time: float, activity: np.ndarray, selected_nodes: set[int]) -> List[int]:
        root_dist = self.diffusion_row(root, center_time)
        if not root_dist:
            return [int(root)]
        total_activity = float(activity.sum())
        root_activity = float(activity[int(root)])
        scores: Dict[int, float] = {}
        for node in np.flatnonzero(activity > 0):
            node = int(node)
            if node <= 0 or node in selected_nodes or node == int(root):
                continue
            high_order = root_activity * float(root_dist.get(node, 0.0))
            if self.args.expand_residual_mode == "symmetric":
                reverse_dist = self.diffusion_row(node, center_time)
                high_order = 0.5 * (high_order + float(activity[node]) * float(reverse_dist.get(int(root), 0.0)))
            expected = root_activity * float(activity[node]) * self.coefficient_sum / max(total_activity, 1e-12)
            residual = high_order - expected
            if residual > 0:
                scores[node] = residual
        sub_nodes = [int(root)]
        if not scores:
            return sub_nodes
        candidates = np.asarray(list(scores), dtype=np.int64)
        values = np.asarray([scores[int(node)] for node in candidates], dtype=np.float64)
        take = min(max(int(self.args.subcommunity_size) - 1, 0), len(candidates))
        if take <= 0:
            return sub_nodes
        if self.args.subcommunity_sample == "top":
            chosen = candidates[np.argsort(values)[::-1][:take]]
        else:
            chosen = self.rng.choice(candidates, size=take, replace=False, p=values / np.clip(values.sum(), 1e-12, None))
        sub_nodes.extend(int(node) for node in chosen)
        return sub_nodes

    def residual_and_targets(self, nodes: np.ndarray, center_time: float) -> Tuple[np.ndarray, np.ndarray]:
        """Modularity residual between the batch nodes, and CountSketch of each diffusion row."""
        distributions = [self.diffusion_row(int(node), center_time) for node in nodes]
        batch_size = len(nodes)
        directed_high_order = np.zeros((batch_size, batch_size), dtype=np.float64)
        targets = np.zeros((batch_size, self.args.sketch_dim), dtype=np.float32)
        global_activity = self.transition(center_time).node_strength.astype(np.float64)
        node_activity = global_activity[nodes].astype(np.float64)
        for row, (node, dist) in enumerate(zip(nodes, distributions)):
            total = float(sum(max(value, 0.0) for value in dist.values()))
            if total <= 0:
                continue
            for col, other_node in enumerate(nodes):
                if row != col:
                    directed_high_order[row, col] = max(float(dist.get(int(other_node), 0.0)), 0.0) * float(global_activity[int(node)])
            active_nodes = np.asarray([key for key, value in dist.items() if value > 0], dtype=np.int64)
            if len(active_nodes) > 0:
                values = np.asarray([dist[int(active_node)] for active_node in active_nodes], dtype=np.float64)
                values = values / np.clip(values.sum(), 1e-12, None)
                signed_values = values * self.node_signs[active_nodes]
                np.add.at(targets[row], self.node_buckets[active_nodes], signed_values.astype(np.float32))
        if self.args.batch_residual_mode == "directed":
            context = directed_high_order
        else:
            context = 0.5 * (directed_high_order + directed_high_order.T)
        np.fill_diagonal(context, 0.0)
        total_activity = float(global_activity.sum())
        expected = np.outer(node_activity, node_activity) * self.coefficient_sum / max(total_activity, 1e-12)
        residual = context - expected
        np.fill_diagonal(residual, 0.0)
        if self.args.batch_center_residual and batch_size > 1:
            residual = residual - residual.mean(axis=1, keepdims=True) - residual.mean(axis=0, keepdims=True) + residual.mean()
            np.fill_diagonal(residual, 0.0)
        return residual.astype(np.float32), targets.astype(np.float32)

    def lift_nodes_to_pairs(
        self, nodes: np.ndarray, center_time: float, node_residual: np.ndarray, node_targets: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pair_indices: List[int] = []
        pair_node_positions: List[int] = []
        pair_weights: List[float] = []
        for node_position, node in enumerate(nodes):
            pairs, weights = self.sample_pairs_for_node(int(node), center_time)
            for pair_index, weight in zip(pairs, weights):
                pair_indices.append(int(pair_index))
                pair_node_positions.append(int(node_position))
                pair_weights.append(float(weight))
        if len(pair_indices) == 0:
            return (
                np.empty(0, dtype=np.int64),
                np.zeros((0, 0), dtype=np.float32),
                np.zeros((0, self.args.sketch_dim), dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )
        positions = np.asarray(pair_node_positions, dtype=np.int64)
        weights = np.asarray(pair_weights, dtype=np.float32)
        lifted_residual = node_residual[np.ix_(positions, positions)].astype(np.float32)
        lifted_residual *= np.outer(weights, weights).astype(np.float32)
        np.fill_diagonal(lifted_residual, 0.0)
        lifted_targets = node_targets[positions].astype(np.float32)
        return np.asarray(pair_indices, dtype=np.int64), lifted_residual, lifted_targets, weights

    def fallback_batch(self) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        fallback = self.rng.choice(len(self.pair_times), size=min(self.num_nodes, len(self.pair_times)), replace=False)
        center_time = float(self.pair_times[int(fallback[0])])
        residual, targets = self.residual_and_targets(self.pair_nodes[fallback], center_time)
        return fallback.astype(np.int64), center_time, residual, targets, np.ones(len(fallback), dtype=np.float32)

    def sample_batch(self) -> Tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        """Returns (pair indices, centre time, modularity residual, sketch targets, lifting weights)."""
        self._batch_diffusion_rows.clear()
        for _attempt in range(20):
            center_time = self.sample_center_time()
            activity = self.transition(center_time).node_strength.astype(np.float64)
            selected_nodes: set[int] = set()
            while len(selected_nodes) < min(self.num_nodes, self.args.num_subcommunities * self.args.subcommunity_size):
                roots = self.root_nodes(center_time, selected_nodes)
                if len(roots) == 0:
                    break
                before = len(selected_nodes)
                for root in roots:
                    for node in self.expand_subcommunity(int(root), center_time, activity, selected_nodes):
                        if node > 0 and node not in selected_nodes:
                            selected_nodes.add(int(node))
                if len(selected_nodes) == before:
                    break
            if len(selected_nodes) < 2:
                continue
            kept_nodes = np.asarray(sorted(selected_nodes), dtype=np.int64)
            node_residual, node_targets = self.residual_and_targets(kept_nodes, center_time)
            pair_indices, lifted_residual, lifted_targets, lift_weights = self.lift_nodes_to_pairs(
                kept_nodes, center_time, node_residual, node_targets
            )
            if len(pair_indices) < 2:
                continue
            return pair_indices, center_time, lifted_residual, lifted_targets, lift_weights
        return self.fallback_batch()


# =============================================================================
# Training losses
# =============================================================================


def local_contrastive_loss(embeddings: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """-sum_ij residual_ij * cos(z_i, z_j) / sum_ij |residual_ij|."""
    if embeddings.shape[0] < 2:
        return embeddings.new_tensor(0.0)
    z = F.normalize(embeddings, p=2, dim=1, eps=1e-8)
    similarity = z @ z.t()
    similarity = similarity.masked_fill(torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device), 0.0)
    normalizer = torch.clamp(torch.sum(torch.abs(residual)), min=1e-8)
    return -torch.sum(residual * similarity) / normalizer


def pairwise_alignment_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
    weights: torch.Tensor,
    sample_size: int,
    scale_mode: str,
    metric: str,
) -> torch.Tensor:
    """Smooth-L1 between pairwise embedding distances and pairwise diffusion-sketch distances."""
    if predicted.shape[0] < 2:
        return predicted.new_tensor(0.0)
    if sample_size > 0 and predicted.shape[0] > sample_size:
        probabilities = weights / torch.clamp(weights.sum(), min=1e-8)
        sample_idx = torch.multinomial(probabilities, sample_size, replacement=False)
        predicted = predicted[sample_idx]
        target = target[sample_idx]
        weights = weights[sample_idx]
    pred_norm = F.normalize(predicted, p=2, dim=1, eps=1e-8)
    target_norm = F.normalize(target, p=2, dim=1, eps=1e-8)
    if metric == "l2":
        pred_pairwise = torch.sqrt(torch.clamp(2.0 - 2.0 * (pred_norm @ pred_norm.T), min=1e-12))
        target_pairwise = torch.sqrt(torch.clamp(2.0 - 2.0 * (target_norm @ target_norm.T), min=1e-12))
    else:  # inner_product
        pred_pairwise = pred_norm @ pred_norm.T
        target_pairwise = target_norm @ target_norm.T
    mask = ~torch.eye(pred_pairwise.shape[0], dtype=torch.bool, device=pred_pairwise.device)
    pred_values = pred_pairwise[mask]
    target_values = target_pairwise[mask]
    pair_weights = (weights.view(-1, 1) * weights.view(1, -1))[mask]
    pair_weights = pair_weights / torch.clamp(pair_weights.sum(), min=1e-8)
    if scale_mode == "standardized":
        pred_mean = torch.sum(pair_weights * pred_values)
        target_mean = torch.sum(pair_weights * target_values)
        pred_std = torch.sqrt(torch.clamp(torch.sum(pair_weights * (pred_values - pred_mean).pow(2)), min=1e-6))
        target_std = torch.sqrt(torch.clamp(torch.sum(pair_weights * (target_values - target_mean).pow(2)), min=1e-6))
        pred_values = (pred_values - pred_mean) / pred_std
        target_values = (target_values - target_mean) / target_std
    return torch.sum(pair_weights * F.smooth_l1_loss(pred_values, target_values, reduction="none"))


# =============================================================================
# Model selection: AMI / ARI against ground truth (evaluation.py)
# =============================================================================


def find_ground_truth(input_path: Path, raw_columns: Sequence[str], args: argparse.Namespace) -> Path:
    if args.ground_truth_dir is not None:
        candidate = args.ground_truth_dir / input_path.name
        if not candidate.exists():
            raise FileNotFoundError(f"No ground truth for {input_path.name}: {candidate} does not exist")
        return candidate
    if {"source_commu", "destination_commu"}.issubset(raw_columns):
        return input_path
    raise ValueError(
        f"{input_path} has no source_commu/destination_commu columns; pass --ground-truth-dir"
    )


class GroundTruthEvaluator:
    """Scores a node-time partition with AMI / ARI from evaluation.py."""

    def __init__(self, gt_path: Path, pairs: Sequence[NodeTime], id_to_name: Dict[int, str]):
        header = pd.DataFrame(columns=pd.read_csv(gt_path, nrows=0).columns)
        self.gt = build_partition_from_csv(
            str(gt_path),
            source_col=find_column(header, SOURCE_ALIASES, gt_path),
            destination_col=find_column(header, DESTINATION_ALIASES, gt_path),
            dtype_source=str,
            dtype_destination=str,
        )
        self.keys = sorted(self.gt.keys())
        # Predictions are keyed by original node name, as in the ground-truth file.
        self.pred_keys = [(id_to_name[int(node)], int(timestamp)) for node, timestamp in pairs]
        missing = set(self.keys) - set(self.pred_keys)
        if missing:
            example = sorted(missing)[:3]
            raise ValueError(f"{gt_path}: {len(missing)} ground-truth node-times not in the input, e.g. {example}")

    def score(self, labels: np.ndarray) -> Tuple[float, float]:
        pred = {key: int(label) for key, label in zip(self.pred_keys, labels)}
        ami = dynamic_mi(self.gt, pred, keys=self.keys, normalisation="ami")
        ari = dynamic_ari(self.gt, pred, keys=self.keys)
        return float(ami), float(ari)


def cluster_and_score(
    embeddings: np.ndarray,
    evaluator: GroundTruthEvaluator,
    args: argparse.Namespace,
    seed: int,
) -> Tuple[float, float, np.ndarray, float]:
    scaled = StandardScaler().fit_transform(embeddings)
    estimator = KMeans(n_clusters=int(args.n_clusters), n_init=int(args.kmeans_init), random_state=int(seed))
    labels = estimator.fit_predict(scaled).astype(np.int64)
    ami, ari = evaluator.score(labels)
    return ami, ari, labels, float(estimator.inertia_)


# =============================================================================
# Training
# =============================================================================


def clone_state_dict(model: torch.nn.Module) -> Dict[str, torch.Tensor]:
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def train_with_checkpoints(
    args: argparse.Namespace,
    data: LinkStreamData,
    evaluator: GroundTruthEvaluator,
    config: TemporalConfig,
    epochs: int,
    batches_per_epoch: int,
    eval_epochs: Sequence[int],
    stage: str,
) -> Tuple[List[dict], dict]:
    """Train one TGAT encoder; return per-checkpoint rows and the best-AMI checkpoint."""
    seed = int(args.model_seed)
    set_seed(seed)
    encoder = build_encoder(args, data)
    sampler = DiffusionSubcommunitySampler(data, args, config)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    target_memory = np.zeros((len(data.pairs), args.sketch_dim), dtype=np.float32)
    target_initialized = np.zeros(len(data.pairs), dtype=bool)
    sampled_center_counts: Dict[float, int] = {}
    sampled_pair_mask = np.zeros(len(data.pairs), dtype=bool)
    sampled_pair_exposures = 0
    all_center_times = np.unique(data.pair_ts)
    eval_epoch_set = {int(epoch) for epoch in eval_epochs}
    checkpoint_rows: List[dict] = []
    best = {"ami": -float("inf"), "state": None, "labels": None, "embeddings": None, "row": None}
    started = time.perf_counter()

    for epoch in range(1, int(epochs) + 1):
        losses: List[float] = []
        local_losses: List[float] = []
        pairwise_losses: List[float] = []
        batch_sizes: List[int] = []
        encoder.train()
        for _ in range(int(batches_per_epoch)):
            batch_pair_idx, center_time, residual_np, target_np, lift_weights_np = sampler.sample_batch()
            sampled_center_counts[float(center_time)] = sampled_center_counts.get(float(center_time), 0) + 1
            sampled_pair_mask[batch_pair_idx] = True
            sampled_pair_exposures += int(len(batch_pair_idx))

            # Smooth each node-time's sketch target over the batches it appears in.
            if args.target_ema_decay > 0:
                initialized = target_initialized[batch_pair_idx]
                if initialized.any():
                    old = batch_pair_idx[initialized]
                    target_memory[old] = (
                        args.target_ema_decay * target_memory[old] + (1.0 - args.target_ema_decay) * target_np[initialized]
                    )
                if (~initialized).any():
                    new = batch_pair_idx[~initialized]
                    target_memory[new] = target_np[~initialized]
                    target_initialized[new] = True
                target_np = target_memory[batch_pair_idx]

            nodes = sampler.pair_nodes[batch_pair_idx]
            times = sampler.pair_times[batch_pair_idx]
            residual = torch.from_numpy(residual_np).float().to(args.device)
            target = torch.from_numpy(target_np).float().to(args.device)
            lift_weights = torch.from_numpy(lift_weights_np).float().to(args.device)

            optimizer.zero_grad()
            embeddings = encoder.get_node_event_embeddings(nodes, times, args.n_degree)
            local_loss = local_contrastive_loss(embeddings, residual)
            loss = args.local_loss_weight * local_loss
            pairwise_loss = embeddings.new_tensor(0.0)
            if args.pairwise_loss_weight > 0:
                pairwise_loss = pairwise_alignment_loss(
                    embeddings,
                    target,
                    lift_weights,
                    args.pairwise_sample_size,
                    args.pairwise_scale_mode,
                    args.pairwise_metric,
                )
                loss = loss + args.pairwise_loss_weight * pairwise_loss
            loss.backward()
            optimizer.step()

            losses.append(float(loss.detach().cpu()))
            local_losses.append(float(local_loss.detach().cpu()))
            pairwise_losses.append(float(pairwise_loss.detach().cpu()))
            batch_sizes.append(len(batch_pair_idx))

        if args.verbose:
            print(
                f"[{stage}] {config.name} epoch={epoch:03d} loss={np.mean(losses):.6f} "
                f"local={np.mean(local_losses):.6f} pair={np.mean(pairwise_losses):.6f} "
                f"batch={np.mean(batch_sizes):.1f}",
                flush=True,
            )
        if epoch not in eval_epoch_set:
            continue

        observed_embeddings = encode_pairs(encoder, data, args)
        ami, ari, labels, inertia = cluster_and_score(observed_embeddings, evaluator, args, seed)
        elapsed = time.perf_counter() - started
        center_counts = np.asarray([sampled_center_counts.get(float(value), 0) for value in all_center_times], dtype=np.float64)
        row = {
            "stage": stage,
            "seed": seed,
            "raw_tau": float(config.raw_tau),
            "cutoff_multiplier": float(config.cutoff_multiplier),
            "epoch": int(epoch),
            "batches_per_epoch": int(batches_per_epoch),
            "n_clusters": int(args.n_clusters),
            "kmeans_inertia": inertia,
            "ami": ami,
            "ari": ari,
            "elapsed_sec": float(elapsed),
            "mean_train_loss": float(np.mean(losses)),
            "mean_local_loss": float(np.mean(local_losses)),
            "mean_pairwise_loss": float(np.mean(pairwise_losses)),
            "mean_batch_size": float(np.mean(batch_sizes)),
            "center_time_coverage": float(len(sampled_center_counts) / max(len(all_center_times), 1)),
            "max_draws_per_center_time": int(center_counts.max(initial=0)),
            "sampled_node_time_coverage": float(sampled_pair_mask.mean()),
            "mean_exposures_per_node_time": float(sampled_pair_exposures / max(len(data.pairs), 1)),
        }
        checkpoint_rows.append(row)
        print(
            f"[{stage}-eval] {config.name} epoch={epoch} K={args.n_clusters} AMI={ami:.6f} ARI={ari:.6f} "
            f"elapsed={elapsed:.1f}s",
            flush=True,
        )
        if ami > best["ami"]:
            best = {
                "ami": ami,
                "state": clone_state_dict(encoder),
                "labels": labels.copy(),
                "embeddings": observed_embeddings.copy(),
                "row": dict(row),
            }

    if best["state"] is None:
        raise RuntimeError("No checkpoint was evaluated")
    return checkpoint_rows, best


# =============================================================================
# Per-dataset pipeline
# =============================================================================


def parse_temporal_configs(text: str) -> List[TemporalConfig]:
    configs = []
    for item in text.split(","):
        item = item.strip()
        if item:
            raw_tau, cutoff_multiplier = item.split(":", maxsplit=1)
            configs.append(TemporalConfig(float(raw_tau), float(cutoff_multiplier)))
    if not configs:
        raise ValueError("At least one temporal configuration is required")
    return configs


def save_partitions(
    artifact_dir: Path, frame: pd.DataFrame, pairs: Sequence[NodeTime], labels: np.ndarray, id_to_name: Dict[int, str]
) -> Tuple[Path, Path]:
    """Write node-time communities and an edge partition in the evaluation.py format."""
    label_by_pair = {(int(node), int(timestamp)): int(label) for (node, timestamp), label in zip(pairs, labels)}
    node_time_path = artifact_dir / "node-time-communities.csv"
    pd.DataFrame(
        {
            "node": [id_to_name[int(node)] for node, _ in pairs],
            "timestamp": [int(timestamp) for _, timestamp in pairs],
            "community": labels.astype(np.int64),
        }
    ).sort_values(["node", "timestamp"], kind="stable").to_csv(node_time_path, index=False)

    edge_path = artifact_dir / "edge-partition.csv"
    pd.DataFrame(
        {
            "source": frame["source_name"],
            "destination": frame["destination_name"],
            "timestamp": frame["timestamp"].astype(int),
            "source_commu": [label_by_pair[(int(s), int(t))] for s, t in zip(frame["source"], frame["timestamp"])],
            "destination_commu": [label_by_pair[(int(d), int(t))] for d, t in zip(frame["destination"], frame["timestamp"])],
        }
    ).to_csv(edge_path, index=False)
    return node_time_path, edge_path


def run_dataset(input_path: Path, temporal_configs: Sequence[TemporalConfig], args: argparse.Namespace) -> dict:
    stem = input_path.stem
    artifact_dir = args.output_root / stem
    artifact_dir.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_dir / "summary.json"
    if args.resume and summary_path.exists():
        print(f"[skip] {stem}: {summary_path} exists (use --no-resume to retrain)", flush=True)
        return json.loads(summary_path.read_text(encoding="utf-8"))

    frame, id_to_name = load_link_stream(input_path)
    frame.to_csv(artifact_dir / "numeric-link-stream.csv", index=False)
    pd.DataFrame({"node_id": sorted(id_to_name), "node": [id_to_name[i] for i in sorted(id_to_name)]}).to_csv(
        artifact_dir / "node-id-map.csv", index=False
    )
    data = prepare_data(frame)
    gt_path = find_ground_truth(input_path, read_csv_any_encoding(input_path).columns, args)
    evaluator = GroundTruthEvaluator(gt_path, data.pairs, id_to_name)
    started = time.perf_counter()

    # Stage 1: short pilot run for every temporal kernel setting.
    pilot_rows: List[dict] = []
    pilot_best: List[dict] = []
    for config in temporal_configs:
        rows, best = train_with_checkpoints(
            args, data, evaluator, config,
            epochs=args.pilot_epochs,
            batches_per_epoch=args.pilot_batches_per_epoch,
            eval_epochs=[args.pilot_epochs],
            stage="pilot",
        )
        pilot_rows.extend(rows)
        pilot_best.append(best["row"])
        pd.DataFrame(pilot_rows).to_csv(artifact_dir / "pilot-search.csv", index=False)

    finalists = sorted(pilot_best, key=lambda row: row["ami"], reverse=True)[: args.num_finalists]
    pd.DataFrame(finalists).to_csv(artifact_dir / "pilot-finalists.csv", index=False)

    # Stage 2: full training of the best pilot settings.
    full_eval_epochs = list(range(args.full_eval_interval, args.full_epochs + 1, args.full_eval_interval))
    if args.full_epochs not in full_eval_epochs:
        full_eval_epochs.append(args.full_epochs)
    full_rows: List[dict] = []
    full_best: List[dict] = []
    for finalist in finalists:
        config = TemporalConfig(float(finalist["raw_tau"]), float(finalist["cutoff_multiplier"]))
        rows, best = train_with_checkpoints(
            args, data, evaluator, config,
            epochs=args.full_epochs,
            batches_per_epoch=args.full_batches_per_epoch,
            eval_epochs=full_eval_epochs,
            stage="full",
        )
        full_rows.extend(rows)
        full_best.append(best)
        pd.DataFrame(full_rows).to_csv(artifact_dir / "full-search.csv", index=False)

    selected = max(full_best, key=lambda result: result["ami"])
    selected_row = selected["row"]
    node_time_path, edge_path = save_partitions(artifact_dir, frame, data.pairs, selected["labels"], id_to_name)
    np.save(artifact_dir / "best-embeddings.npy", selected["embeddings"])
    torch.save(selected["state"], artifact_dir / "best-model.pt")
    elapsed = time.perf_counter() - started
    ami, ari = float(selected_row["ami"]), float(selected_row["ari"])

    summary = {
        "data_name": stem,
        "method": "TGAT + local contrastive + pairwise alignment (diffusion CountSketch target)",
        "selection": "configs and checkpoints selected by AMI on the ground truth (oracle selection)",
        "input_path": str(input_path),
        "node_time_communities": str(node_time_path),
        "edge_partition": str(edge_path),
        "ground_truth": str(gt_path),
        "ami": ami,
        "ari": ari,
        "n_clusters": int(args.n_clusters),
        "selected_raw_tau": float(selected_row["raw_tau"]),
        "selected_cutoff_multiplier": float(selected_row["cutoff_multiplier"]),
        "selected_epoch": int(selected_row["epoch"]),
        "local_loss_weight": float(args.local_loss_weight),
        "pairwise_loss_weight": float(args.pairwise_loss_weight),
        "nodes": int(len(id_to_name)),
        "interactions": int(len(data.df)),
        "observed_node_times": int(len(data.pairs)),
        "sampled_node_time_coverage": float(selected_row["sampled_node_time_coverage"]),
        "mean_local_loss": float(selected_row["mean_local_loss"]),
        "mean_pairwise_loss": float(selected_row["mean_pairwise_loss"]),
        "elapsed_sec": float(elapsed),
        "config": vars(args),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(
        f"[completed] {stem} AMI={ami:.6f} ARI={ari:.6f} tau={selected_row['raw_tau']:g} "
        f"cutoff={selected_row['cutoff_multiplier']:g} epoch={selected_row['epoch']} elapsed={elapsed:.1f}s",
        flush=True,
    )
    return summary


# =============================================================================
# Command line
# =============================================================================


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Dynamic community detection with a TGAT encoder trained by a local contrastive loss and a "
            "pairwise-alignment loss, both driven by a temporal random-walk diffusion matrix. "
            "Configurations and checkpoints are selected by AMI against the ground truth."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    group = parser.add_argument_group("input / output")
    group.add_argument("--inputs", type=Path, nargs="+", required=True,
                       help="link-stream CSV files (columns: source, destination|target, timestamp)")
    group.add_argument("--output-root", type=Path, default=Path("dycommu_tgat_outputs"),
                       help="one sub-directory per input file is written here")
    group.add_argument("--ground-truth-dir", type=Path, default=None,
                       help="directory holding ground-truth CSVs with the same file names as the inputs "
                            "(columns source_commu, destination_commu); if omitted, each input must contain "
                            "these columns itself")

    group = parser.add_argument_group("temporal-config search")
    group.add_argument("--temporal-configs", default="5:3,10:3,25:3,25:5",
                       help="comma-separated raw_tau:cutoff_multiplier pairs tried in the pilot stage "
                            "(raw_tau is in the input's timestamp units)")
    group.add_argument("--pilot-epochs", type=int, default=3)
    group.add_argument("--pilot-batches-per-epoch", type=int, default=8)
    group.add_argument("--num-finalists", type=int, default=2, help="pilot configs promoted to full training")
    group.add_argument("--full-epochs", type=int, default=10)
    group.add_argument("--full-batches-per-epoch", type=int, default=24)
    group.add_argument("--full-eval-interval", type=int, default=2,
                       help="evaluate AMI every N full-training epochs")

    group = parser.add_argument_group("training losses")
    group.add_argument("--local-loss-weight", type=float, default=1.0, help="weight of the local contrastive loss")
    group.add_argument("--pairwise-loss-weight", type=float, default=1.0, help="weight of the pairwise-alignment loss")
    group.add_argument("--pairwise-sample-size", type=int, default=256,
                       help="node-times sampled per batch for the pairwise loss (0 = all)")
    group.add_argument("--pairwise-scale-mode", choices=("standardized", "raw"), default="standardized")
    group.add_argument("--pairwise-metric", choices=("l2", "inner_product"), default="l2")
    group.add_argument("--target-ema-decay", type=float, default=0.9,
                       help="EMA decay of each node-time's sketch target across batches (0 = off)")
    group.add_argument("--lr", type=float, default=1e-3)
    group.add_argument("--weight-decay", type=float, default=1e-5)

    group = parser.add_argument_group("diffusion matrix and CountSketch target")
    group.add_argument("--diffusion-order", type=int, default=2, help="random-walk steps in the diffusion matrix")
    group.add_argument("--diffusion-decay", type=float, default=1.0, help="weight multiplier per extra step")
    group.add_argument("--sketch-dim", type=int, default=128, help="CountSketch width of the diffusion target")
    group.add_argument("--root-tau-multiplier", type=float, default=1.0,
                       help="time-decay scale for event-time selection, relative to tau")
    group.add_argument("--cache-transitions", action=argparse.BooleanOptionalAction, default=True,
                       help="cache transition matrices per centre time")

    group = parser.add_argument_group("batch sampling")
    group.add_argument("--center-sampling-mode", default="node_time",
                       choices=("node_time", "node_time_independent", "unique_uniform", "unique_activity"))
    group.add_argument("--num-subcommunities", type=int, default=8, help="root nodes drawn per batch")
    group.add_argument("--subcommunity-size", type=int, default=16, help="maximum nodes per sub-community")
    group.add_argument("--root-sample", choices=("strength", "uniform"), default="strength")
    group.add_argument("--subcommunity-sample", choices=("weighted", "top"), default="weighted")
    group.add_argument("--expand-residual-mode", choices=("directed", "symmetric"), default="directed")
    group.add_argument("--batch-residual-mode", choices=("directed", "symmetric"), default="directed")
    group.add_argument("--batch-center-residual", action=argparse.BooleanOptionalAction, default=True,
                       help="double-centre the batch modularity residual")
    group.add_argument("--lifting-mode", choices=("soft", "nearest"), default="soft",
                       help="lift each node to its top-q nearest node-times (soft) or to one node-time")
    group.add_argument("--lifting-top-q", type=int, default=2)
    group.add_argument("--lifting-tau-multiplier", type=float, default=1.0)
    group.add_argument("--event-time-selection", choices=("nearest", "sample"), default="nearest",
                       help="with --lifting-mode nearest: pick the nearest node-time or sample one by time decay")

    group = parser.add_argument_group("TGAT encoder")
    group.add_argument("--n-degree", type=int, default=20, help="temporal neighbours sampled per node")
    group.add_argument("--n-layer", type=int, default=2)
    group.add_argument("--n-head", type=int, default=2)
    group.add_argument("--node-dim", type=int, default=128)
    group.add_argument("--time-dim", type=int, default=32)
    group.add_argument("--drop-out", type=float, default=0.1)
    group.add_argument("--neighbor-mode", choices=("bidirectional", "past"), default="bidirectional",
                       help="neighbours closest in time on both sides of the query time, or only at/before it "
                            "(same-time interactions are included in both modes)")
    group.add_argument("--uniform", action="store_true",
                       help="sample temporal neighbours uniformly (with replacement) instead of the closest ones")
    group.add_argument("--raw-node-dim", type=int, default=0,
                       help="random node-feature width; 0 uses one-hot node ids")
    group.add_argument("--raw-edge-dim", type=int, default=2)
    group.add_argument("--encode-batch-size", type=int, default=255, help="batch size when embedding all node-times")

    group = parser.add_argument_group("clustering")
    group.add_argument("--n-clusters", type=int, default=13)
    group.add_argument("--kmeans-init", type=int, default=50)

    group = parser.add_argument_group("runtime")
    group.add_argument("--model-seed", type=int, default=33)
    group.add_argument("--sampler-seed", type=int, default=55)
    group.add_argument("--sketch-seed", type=int, default=33, help="seed of the CountSketch hash buckets and signs")
    group.add_argument("--center-seed", type=int, default=None, help="defaults to --sampler-seed")
    group.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    group.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True,
                       help="skip inputs whose summary.json already exists")
    group.add_argument("--verbose", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    missing = [path for path in args.inputs if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing input files: {missing}")
    temporal_configs = parse_temporal_configs(args.temporal_configs)
    args.output_root.mkdir(parents=True, exist_ok=True)

    summaries = []
    aggregate_path = args.output_root / "summary.csv"
    for input_path in args.inputs:
        summaries.append(run_dataset(input_path, temporal_configs, args))
        pd.DataFrame([{k: v for k, v in s.items() if k != "config"} for s in summaries]).to_csv(aggregate_path, index=False)

    aggregate = pd.DataFrame([{k: v for k, v in s.items() if k != "config"} for s in summaries]).sort_values("data_name")
    aggregate.to_csv(aggregate_path, index=False)
    run_summary = {
        "datasets": int(len(aggregate)),
        "mean_ami": float(aggregate["ami"].mean()),
        "mean_ari": float(aggregate["ari"].mean()),
        "total_elapsed_sec": float(aggregate["elapsed_sec"].sum()),
        "summary_file": str(aggregate_path),
    }
    (args.output_root / "run-summary.json").write_text(json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(run_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
