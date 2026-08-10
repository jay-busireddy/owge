#!/usr/bin/env python3
"""
OWGE Experiment 3: Mechanism-Adequacy Test
===========================================

Purpose
-------
This experiment follows two earlier OWGE studies:
  E1: exploratory laptop pilot.
  E2: 40-seed confirmatory test showing selective observation O3 > exhaustive O2,
      but simple replay O4R did not establish O4R > O1.

E3 asks a narrower and more faithful mechanistic question:
  Does an observer with weighted context + associative episodic memory + validated
  dream recombination + durable consolidation (O4D) outperform central-only O1
  on a task that genuinely requires compositional transfer across prior experiences?

The design intentionally does NOT alter E2 or tune a scalar to force O4D to win.
Primary inference is based on held-out compositional-transfer accuracy.

Observers
---------
  O1   central-only observation.
  O3   context-weighted observation (fixed rho).
  O4R  O3 + prioritized replay only (legacy/simple replay mechanism).
  O4M  O3 + associative key-value episodic memory, no dream.
  O4D  O3 + associative memory + validated cross-experience dream recombination.

Curriculum
----------
  Phase A CENTRAL: central cue is sufficient. O1 should be competitive.
  Phase B DELAYED: an early peripheral cue adds delayed predictive information.
  Phase C COMPOSITIONAL: training provides partial factor combinations in source
                         styles; held-out test requires novel combinations/styles.
                         O4D is allowed to recombine prior memory fragments into
                         validated synthetic dream episodes under the same total
                         gradient-update budget as the controls.

Mechanism diagnostics
---------------------
  * identical frozen base checkpoint and identical LoRA initialization.
  * matched real-batch schedule and matched exploration uniforms.
  * fixed optimizer-step budget per observer.
  * LoRA norm, singular spectrum, effective rank, changed-weight fraction.
  * neural active-unit counts and activation overlap.
  * associative graph nodes/edges, retrieved paths, path diversity, dream fraction.
  * causal ablations: no LoRA, no external memory, no short-term GRU memory.
  * legacy OWGE proxy is retained for continuity, but is NOT the primary endpoint.

Synthetic data are used so causal structure and dream validation are known exactly.
No internet images or pretrained large models are required.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

try:
    from scipy import stats
except Exception:
    stats = None


# -----------------------------------------------------------------------------
# Reproducibility / configuration
# -----------------------------------------------------------------------------


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class Config:
    # Visual sequence
    image_size: int = 40
    episode_len: int = 6
    cue_step: int = 1
    num_distractors: int = 5

    # Observer weighting (frozen before confirmatory run)
    rho_peripheral: float = 0.50
    distractor_weight: float = 0.06
    background_weight: float = 0.05

    # Model / LoRA
    embed_dim: int = 32
    hidden_dim: int = 32
    lora_rank: int = 4
    lora_alpha: float = 8.0
    lora_lr: float = 4e-3
    grad_clip: float = 1.0
    entropy_bonus: float = 0.002
    learning_objective: str = "ce"  # mechanism-adequacy study removes sampled-RL variance

    # Balanced base pretraining (not central-biased)
    base_pretrain_episodes: int = 1600
    base_pretrain_epochs: int = 6
    base_lr: float = 2e-3

    # Curriculum sizes
    phase_a_train: int = 420
    phase_b_train: int = 520
    phase_c_train: int = 240
    val_episodes: int = 220
    test_a_episodes: int = 400
    test_b_episodes: int = 500
    test_c_episodes: int = 800

    # Fixed epochs (no p-value-driven extension)
    phase_a_epochs: int = 5
    phase_b_epochs: int = 6
    phase_c_epochs: int = 10
    batch_size: int = 32

    # Compute budget: same total optimizer steps for every observer.
    common_real_fraction: float = 0.80

    # Associative episodic memory
    memory_max_records: int = 256
    memory_ingest_per_epoch: int = 72
    memory_top_k: int = 5
    memory_expand_k: int = 3
    memory_similarity_threshold: float = 0.68
    memory_edge_threshold: float = 0.72
    memory_vote_beta: float = 0.45
    memory_temperature: float = 0.18

    # Dream consolidation
    dream_min_memory: int = 32
    dream_cross_phase_prob: float = 0.70
    dream_cross_style_prob: float = 0.70
    dream_memory_weight: float = 0.65
    dream_duplicate_cosine: float = 0.985

    # Delayed regime information
    delayed_central_accuracy: float = 0.65
    delayed_peripheral_accuracy: float = 0.86

    # Probe / diagnostics
    activation_threshold: float = 0.25
    significant_sv_fraction: float = 0.05
    probe_episodes: int = 96

    # Primary confirmatory seeds: fresh and disjoint from E1/E2.
    seeds: List[int] = field(default_factory=lambda: [12001 + 53 * i for i in range(40)])

    # Program behavior
    preset: str = "confirmatory"
    device: str = "cpu"
    output: str = "owge_dream_confirmatory"
    save_checkpoints: bool = True
    resume: bool = False


def apply_preset(cfg: Config, preset: str) -> Config:
    cfg.preset = preset
    if preset == "smoke":
        cfg.base_pretrain_episodes = 80
        cfg.base_pretrain_epochs = 1
        cfg.phase_a_train = 36
        cfg.phase_b_train = 40
        cfg.phase_c_train = 48
        cfg.val_episodes = 24
        cfg.test_a_episodes = 32
        cfg.test_b_episodes = 32
        cfg.test_c_episodes = 48
        cfg.phase_a_epochs = 1
        cfg.phase_b_epochs = 1
        cfg.phase_c_epochs = 1
        cfg.memory_max_records = 48
        cfg.memory_ingest_per_epoch = 18
        cfg.dream_min_memory = 8
        cfg.probe_episodes = 16
        cfg.seeds = [12001]
    elif preset == "engineering":
        cfg.base_pretrain_episodes = 400
        cfg.base_pretrain_epochs = 2
        cfg.phase_a_train = 120
        cfg.phase_b_train = 140
        cfg.phase_c_train = 100
        cfg.val_episodes = 80
        cfg.test_a_episodes = 120
        cfg.test_b_episodes = 120
        cfg.test_c_episodes = 160
        cfg.phase_a_epochs = 2
        cfg.phase_b_epochs = 2
        cfg.phase_c_epochs = 4
        cfg.memory_max_records = 120
        cfg.memory_ingest_per_epoch = 36
        cfg.probe_episodes = 32
        # Engineering seeds are never used in the confirmatory p-value.
        cfg.seeds = [10001, 10067, 10139, 10211]
    elif preset == "confirmatory":
        pass
    else:
        raise ValueError(f"Unknown preset: {preset}")
    return cfg


# -----------------------------------------------------------------------------
# Scene generator / curriculum
# -----------------------------------------------------------------------------


SCENES: Dict[str, Dict[str, object]] = {
    "base": {
        "bg": (230, 230, 226), "central_shape": "circle", "periph_shape": "triangle",
        "true": (220, 70, 70), "false": (65, 120, 210), "accent1": (235, 170, 35), "accent0": (85, 85, 85),
    },
    "crossing": {
        "bg": (236, 236, 236), "central_shape": "circle", "periph_shape": "triangle",
        "true": (230, 65, 65), "false": (65, 110, 215), "accent1": (238, 185, 40), "accent0": (88, 88, 88),
    },
    "warehouse": {
        "bg": (225, 235, 228), "central_shape": "square", "periph_shape": "circle",
        "true": (205, 80, 65), "false": (55, 135, 185), "accent1": (238, 160, 45), "accent0": (92, 92, 92),
    },
    "factory": {
        "bg": (236, 229, 220), "central_shape": "diamond", "periph_shape": "square",
        "true": (190, 65, 55), "false": (65, 120, 190), "accent1": (225, 175, 30), "accent0": (80, 80, 80),
    },
    "harbor": {
        "bg": (220, 233, 240), "central_shape": "triangle", "periph_shape": "diamond",
        "true": (220, 72, 60), "false": (60, 130, 205), "accent1": (245, 175, 40), "accent0": (85, 85, 85),
    },
}

TRAIN_STYLES = ("crossing", "warehouse")
TEST_STYLES = ("factory", "harbor")


@dataclass
class Meta:
    phase: str
    label: int
    central_bit: int
    peripheral_bit: int
    style: str
    episode_seed: int
    source_group: int
    distractor_seed: int


def noisy_bit(rng: np.random.Generator, truth: int, accuracy: float) -> int:
    return int(truth if rng.random() < accuracy else 1 - truth)


def phase_rule(phase: str, central_bit: int, peripheral_bit: int) -> int:
    if phase == "A":
        return int(central_bit)
    if phase == "C":
        # Compositional transfer rule: hazard depends on the RELATION between
        # factors rather than either factor alone (XNOR / equality). Source
        # training omits (0,0), making that positive relation a true held-out
        # composition that dream recombination can synthesize and validate.
        return int(central_bit == peripheral_bit)
    if phase == "BASE":
        # Generic OR integration skill: both regions can matter, but this rule is
        # deliberately different from the Phase-C equality relation.
        return int(bool(central_bit) or bool(peripheral_bit))
    raise ValueError(f"phase_rule not defined for {phase}")


def make_meta(seed: int, cfg: Config, phase: str, split: str, styles: Sequence[str]) -> Meta:
    rng = np.random.default_rng(seed)
    style = str(styles[int(rng.integers(0, len(styles)))])
    source_group = int(rng.integers(0, 2))

    if phase == "BASE":
        y = int(rng.integers(0, 2))
        if y == 0:
            c, p = 0, 0
        else:
            c, p = [(0,1),(1,0),(1,1)][int(rng.integers(0,3))]
        y = phase_rule("BASE", c, p)
    elif phase == "A":
        c = int(rng.integers(0, 2))
        p = int(rng.integers(0, 2))  # independent distractor-like peripheral cue
        y = phase_rule("A", c, p)
    elif phase == "B":
        y = int(rng.integers(0, 2))
        c = noisy_bit(rng, y, cfg.delayed_central_accuracy)
        p = noisy_bit(rng, y, cfg.delayed_peripheral_accuracy)
    elif phase == "C":
        # Source training deliberately gives partial factor coverage.
        # Group 0: P=1 and C varies. Group 1: C=1 and P varies.
        # Validation/test use all four combinations uniformly.
        if split == "train":
            if source_group == 0:
                p = 1
                c = int(rng.integers(0, 2))
            else:
                c = 1
                p = int(rng.integers(0, 2))
        else:
            combo = int(rng.integers(0, 4))
            c, p = (combo >> 1) & 1, combo & 1
        y = phase_rule("C", c, p)
    else:
        raise ValueError(phase)

    return Meta(
        phase=phase,
        label=int(y),
        central_bit=int(c),
        peripheral_bit=int(p),
        style=style,
        episode_seed=int(seed),
        source_group=source_group,
        distractor_seed=int(seed * 1543 + 31),
    )


def semantic_boxes(size: int) -> Dict[str, Tuple[int, int, int, int]]:
    return {
        "central": (size // 2 - 8, size // 2 - 8, size // 2 + 8, size // 2 + 8),
        "peripheral": (3, 3, 13, 13),
        "marker": (size - 14, 4, size - 4, 12),
    }


def draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: Tuple[int, int, int, int], fill) -> None:
    x0, y0, x1, y1 = box
    if shape == "circle":
        draw.ellipse(box, fill=fill)
    elif shape == "square":
        draw.rectangle(box, fill=fill)
    elif shape == "diamond":
        cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
        draw.polygon([(cx, y0), (x1, cy), (cx, y1), (x0, cy)], fill=fill)
    elif shape == "triangle":
        draw.polygon([((x0 + x1) // 2, y0), (x1, y1), (x0, y1)], fill=fill)
    else:
        draw.rectangle(box, fill=fill)


def render_episode(meta: Meta, cfg: Config, num_distractors: Optional[int] = None) -> np.ndarray:
    nd = cfg.num_distractors if num_distractors is None else int(num_distractors)
    style = SCENES[meta.style]
    boxes = semantic_boxes(cfg.image_size)
    frames: List[np.ndarray] = []

    for t in range(cfg.episode_len):
        img = Image.new("RGB", (cfg.image_size, cfg.image_size), style["bg"])
        draw = ImageDraw.Draw(img)

        # Central factor remains visible throughout the episode.
        cfill = style["true"] if meta.central_bit else style["false"]
        draw_shape(draw, str(style["central_shape"]), boxes["central"], cfill)

        # Peripheral factor is transient: it exists only at the early cue step.
        if t == cfg.cue_step:
            pfill = style["accent1"] if meta.peripheral_bit else style["accent0"]
            draw_shape(draw, str(style["periph_shape"]), boxes["peripheral"], pfill)

        # Small phase-neutral marker gives visual diversity but no label information.
        x0, y0, x1, y1 = boxes["marker"]
        marker_fill = (130, 130, 130) if (meta.episode_seed + t) % 2 else (175, 175, 175)
        draw.rectangle((x0, y0, x1, y1), fill=marker_fill)

        drng = np.random.default_rng(meta.distractor_seed + t * 101)
        forbidden = list(boxes.values())
        for _ in range(nd):
            for _attempt in range(40):
                x = int(drng.integers(1, cfg.image_size - 5))
                y = int(drng.integers(1, cfg.image_size - 5))
                box = (x, y, x + 3, y + 3)
                if not any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in forbidden):
                    break
            color = tuple(int(v) for v in drng.integers(35, 220, size=3))
            if drng.random() < 0.5:
                draw.ellipse(box, fill=color)
            else:
                draw.rectangle(box, fill=color)
        frames.append(np.asarray(img, dtype=np.uint8))
    return np.stack(frames, axis=0)


class CurriculumDataset(Dataset):
    def __init__(self, n: int, seed: int, cfg: Config, phase: str, split: str, styles: Sequence[str]):
        self.n = int(n)
        self.seed = int(seed)
        self.cfg = cfg
        self.phase = phase
        self.split = split
        self.styles = tuple(styles)
        # Cache rendered uint8 episodes so repeated epochs/ablations do not re-render images.
        # This substantially reduces laptop runtime while preserving deterministic data.
        self._cache: Dict[int, Tuple[np.ndarray, int, Dict[str, object]]] = {}

    def __len__(self) -> int:
        return self.n

    def episode_seed(self, idx: int) -> int:
        return int(self.seed + idx * 7919)

    def get_meta(self, idx: int) -> Meta:
        return make_meta(self.episode_seed(idx), self.cfg, self.phase, self.split, self.styles)

    def __getitem__(self, idx: int):
        idx = int(idx)
        if idx not in self._cache:
            meta = self.get_meta(idx)
            frames = render_episode(meta, self.cfg)
            info = {
                "index": idx, "phase": meta.phase, "style": meta.style,
                "central_bit": int(meta.central_bit), "peripheral_bit": int(meta.peripheral_bit),
                "label": int(meta.label), "episode_seed": int(meta.episode_seed),
            }
            self._cache[idx] = (frames, int(meta.label), info)
        frames, label, info = self._cache[idx]
        x = torch.from_numpy(frames).permute(0, 3, 1, 2).float() / 255.0
        y = torch.tensor(label, dtype=torch.long)
        return x, y, dict(info)


# -----------------------------------------------------------------------------
# Information theory
# -----------------------------------------------------------------------------


def entropy_discrete(x: Sequence[int]) -> float:
    a = np.asarray(x, dtype=int)
    if len(a) == 0:
        return 0.0
    _, counts = np.unique(a, return_counts=True)
    p = counts / counts.sum()
    return float(-(p * np.log2(np.clip(p, 1e-12, 1.0))).sum())


def mutual_information(x: Sequence[int], y: Sequence[int]) -> float:
    x = np.asarray(x, int); y = np.asarray(y, int)
    if len(x) == 0:
        return 0.0
    out = 0.0
    for xv in np.unique(x):
        for yv in np.unique(y):
            pxy = np.mean((x == xv) & (y == yv))
            if pxy <= 0:
                continue
            px = np.mean(x == xv); py = np.mean(y == yv)
            out += pxy * math.log2(pxy / max(px * py, 1e-12))
    return float(out)


def conditional_mi(x: Sequence[int], y: Sequence[int], z: Sequence[int]) -> float:
    x = np.asarray(x, int); y = np.asarray(y, int); z = np.asarray(z, int)
    total = 0.0
    for zv in np.unique(z):
        m = z == zv
        if m.sum() > 1:
            total += float(m.mean()) * mutual_information(x[m], y[m])
    return float(total)


def dataset_information(ds: CurriculumDataset) -> Dict[str, float]:
    ys, cs, ps = [], [], []
    combos = []
    for i in range(len(ds)):
        m = ds.get_meta(i)
        ys.append(m.label); cs.append(m.central_bit); ps.append(m.peripheral_bit)
        combos.append(m.central_bit * 2 + m.peripheral_bit)
    return {
        "H_y": entropy_discrete(ys),
        "I_c_y": mutual_information(cs, ys),
        "I_p_y": mutual_information(ps, ys),
        "I_p_y_given_c": conditional_mi(ps, ys, cs),
        "H_combo": entropy_discrete(combos),
        "combo_coverage": float(len(set(combos)) / 4.0),
    }


# -----------------------------------------------------------------------------
# Observer policies / model
# -----------------------------------------------------------------------------


OBSERVERS = ("O1", "O3", "O4R", "O4M", "O4D")
MEMORY_OBSERVERS = ("O4M", "O4D")


def observer_weights(observer: str, cfg: Config) -> Dict[str, float]:
    if observer == "O1":
        return {"central": 1.0, "peripheral": 0.0, "background": 0.03}
    if observer in ("O3", "O4R", "O4M", "O4D"):
        return {"central": 1.0, "peripheral": float(cfg.rho_peripheral), "background": float(cfg.background_weight)}
    raise ValueError(observer)


def make_mask(observer: str, cfg: Config, device: torch.device) -> torch.Tensor:
    w = observer_weights(observer, cfg)
    s = cfg.image_size
    mask = torch.full((1, 1, s, s), float(w["background"]), device=device)
    boxes = semantic_boxes(s)
    x0, y0, x1, y1 = boxes["central"]
    mask[:, :, y0:y1+1, x0:x1+1] = float(w["central"])
    x0, y0, x1, y1 = boxes["peripheral"]
    mask[:, :, y0:y1+1, x0:x1+1] = float(w["peripheral"])
    return mask


class TinyCNN(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.conv3 = nn.Conv2d(16, 24, 3, padding=1)
        self.proj = nn.Linear(24*4, embed_dim)

    def forward(self, x: torch.Tensor, return_acts: bool = False):
        a1 = F.relu(self.conv1(x)); x = F.max_pool2d(a1, 2)
        a2 = F.relu(self.conv2(x)); x = F.max_pool2d(a2, 2)
        a3 = F.relu(self.conv3(x)); x = F.adaptive_avg_pool2d(a3, (2,2)).flatten(1)
        z = torch.tanh(self.proj(x))
        if return_acts:
            return z, (a1, a2, a3)
        return z


class BasePolicy(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = TinyCNN(cfg.embed_dim)
        self.gru = nn.GRUCell(cfg.embed_dim, cfg.hidden_dim)
        self.policy = nn.Linear(cfg.hidden_dim, 2)

    def encode(self, frames: torch.Tensor, return_acts: bool = False):
        b, t, c, h, w = frames.shape
        flat = frames.reshape(b*t, c, h, w)
        if return_acts:
            z, acts = self.encoder(flat, return_acts=True)
            return z.reshape(b, t, -1), acts
        z = self.encoder(flat)
        return z.reshape(b, t, -1)

    def hidden(self, emb: torch.Tensor, reset_each_step: bool = False) -> torch.Tensor:
        b, t, _ = emb.shape
        h = torch.zeros((b, self.gru.hidden_size), device=emb.device, dtype=emb.dtype)
        for step in range(t):
            if reset_each_step and step > 0:
                h = torch.zeros_like(h)
            h = self.gru(emb[:, step], h)
        return h

    def forward(self, frames: torch.Tensor, reset_each_step: bool = False):
        e = self.encode(frames)
        h = self.hidden(e, reset_each_step=reset_each_step)
        return self.policy(h)


class LoRAResidual(nn.Module):
    def __init__(self, dim: int, rank: int, alpha: float):
        super().__init__()
        self.scale = alpha / max(rank, 1)
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor):
        return x + self.scale * self.B(self.A(x))

    def delta_weight(self):
        return self.scale * (self.B.weight @ self.A.weight)


class LoRAPolicyDelta(nn.Module):
    def __init__(self, hidden: int, out: int, rank: int, alpha: float):
        super().__init__()
        self.scale = alpha / max(rank, 1)
        self.A = nn.Linear(hidden, rank, bias=False)
        self.B = nn.Linear(rank, out, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, h: torch.Tensor):
        return self.scale * self.B(self.A(h))

    def delta_weight(self):
        return self.scale * (self.B.weight @ self.A.weight)


@dataclass
class MemoryRecord:
    key: np.ndarray
    label: int
    style: str
    phase: str
    central_bit: int
    peripheral_bit: int
    frames: np.ndarray  # uint8 T,H,W,C
    is_dream: bool = False
    strength: float = 1.0


class AssociativeMemory:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.records: List[MemoryRecord] = []
        self.adj: Dict[int, List[Tuple[int, float]]] = {}
        self._key_matrix: Optional[np.ndarray] = None

    def __len__(self):
        return len(self.records)

    def clone(self):
        return copy.deepcopy(self)

    @staticmethod
    def _norm(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / max(n, 1e-12)

    def add(self, rec: MemoryRecord, rng: np.random.Generator) -> None:
        # Avoid nearly identical dream duplicates.
        if rec.is_dream and self.records:
            keys = self.key_matrix()
            q = self._norm(rec.key)
            if float(np.max(keys @ q)) >= self.cfg.dream_duplicate_cosine:
                return
        if len(self.records) < self.cfg.memory_max_records:
            self.records.append(rec)
        else:
            # Salience-aware consolidation: preferentially evict weak records.
            # This is generic (uses record strength/novelty), not a hand-coded combo.
            strengths = np.asarray([r.strength for r in self.records], dtype=float)
            min_s = float(strengths.min())
            weak = np.where(strengths <= min_s + 1e-8)[0]
            if rec.strength >= min_s:
                idx = int(rng.choice(weak))
            else:
                idx = int(rng.integers(0, len(self.records)))
            self.records[idx] = rec
        self._key_matrix = None

    def key_matrix(self) -> np.ndarray:
        if self._key_matrix is None:
            if not self.records:
                return np.zeros((0, self.cfg.embed_dim), dtype=np.float32)
            mat = np.stack([self._norm(r.key.astype(np.float32)) for r in self.records], axis=0)
            self._key_matrix = mat
        return self._key_matrix

    def rebuild_graph(self) -> None:
        self.adj = {i: [] for i in range(len(self.records))}
        k = self.key_matrix()
        if len(k) < 2:
            return
        sim = k @ k.T
        for i in range(len(self.records)):
            candidates = []
            for j in range(len(self.records)):
                if i == j:
                    continue
                s = float(sim[i, j])
                same_outcome = self.records[i].label == self.records[j].label
                threshold = self.cfg.memory_edge_threshold - (0.05 if same_outcome else 0.0)
                if s >= threshold:
                    score = s + (0.05 if same_outcome else 0.0)
                    candidates.append((j, score))
            candidates.sort(key=lambda x: x[1], reverse=True)
            self.adj[i] = candidates[:8]

    def retrieve(self, query_keys: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, float]], List[List[int]]]:
        b = len(query_keys)
        if len(self.records) == 0:
            return np.zeros((b, 2), dtype=np.float32), [self.empty_stats() for _ in range(b)], [[] for _ in range(b)]
        keys = self.key_matrix()
        biases, stats_rows, paths = [], [], []
        for q0 in query_keys:
            q = self._norm(q0.astype(np.float32))
            sims = keys @ q
            top = np.argsort(-sims)[:min(self.cfg.memory_top_k, len(sims))].tolist()
            expanded = list(top)
            edge_count = 0
            for idx in top[:2]:
                for nb, _score in self.adj.get(idx, [])[:self.cfg.memory_expand_k]:
                    edge_count += 1
                    if nb not in expanded:
                        expanded.append(nb)
            # Keep a bounded candidate set.
            expanded = expanded[:self.cfg.memory_top_k + 2*self.cfg.memory_expand_k]
            esims = np.asarray([max(float(sims[i]), self.cfg.memory_similarity_threshold - 0.15) for i in expanded])
            weights = np.exp(esims / max(self.cfg.memory_temperature, 1e-4))
            weights *= np.asarray([self.records[i].strength for i in expanded])
            weights /= max(weights.sum(), 1e-12)
            p1 = float(sum(w * self.records[i].label for w, i in zip(weights, expanded)))
            p1 = float(np.clip(p1, 0.02, 0.98))
            logit = math.log(p1 / (1.0 - p1)) * self.cfg.memory_vote_beta
            biases.append(np.asarray([-0.5*logit, 0.5*logit], dtype=np.float32))
            styles = {self.records[i].style for i in expanded}
            phases = {self.records[i].phase for i in expanded}
            dream_fraction = float(np.mean([self.records[i].is_dream for i in expanded])) if expanded else 0.0
            qmask = np.abs(q0) >= self.cfg.activation_threshold
            jaccards = []
            union_counts = []
            for i in expanded:
                rmask = np.abs(self.records[i].key) >= self.cfg.activation_threshold
                union = np.logical_or(qmask, rmask)
                inter = np.logical_and(qmask, rmask)
                jaccards.append(float(inter.sum()/max(union.sum(),1)))
                union_counts.append(float(union.sum()))
            stats_rows.append({
                "retrieved_nodes": float(len(expanded)),
                "retrieved_edges": float(edge_count),
                "path_style_diversity": float(len(styles)),
                "path_phase_diversity": float(len(phases)),
                "retrieved_dream_fraction": dream_fraction,
                "mean_similarity": float(np.mean([sims[i] for i in expanded])) if expanded else 0.0,
                "memory_p1": p1,
                "pathway_reuse_jaccard": float(np.mean(jaccards)) if jaccards else 0.0,
                "pathway_union_active_units": float(np.mean(union_counts)) if union_counts else 0.0,
            })
            paths.append(expanded)
        return np.stack(biases, axis=0), stats_rows, paths

    @staticmethod
    def empty_stats() -> Dict[str, float]:
        return {
            "retrieved_nodes": 0.0, "retrieved_edges": 0.0,
            "path_style_diversity": 0.0, "path_phase_diversity": 0.0,
            "retrieved_dream_fraction": 0.0, "mean_similarity": 0.0, "memory_p1": 0.5,
            "pathway_reuse_jaccard": 0.0, "pathway_union_active_units": 0.0,
        }

    def summary(self) -> Dict[str, float]:
        edges = int(sum(len(v) for v in self.adj.values()))
        dreams = int(sum(r.is_dream for r in self.records))
        return {
            "memory_records": float(len(self.records)),
            "memory_edges_directed": float(edges),
            "memory_dream_records": float(dreams),
            "memory_dream_fraction": float(dreams / max(len(self.records), 1)),
        }


class Agent(nn.Module):
    def __init__(self, base: BasePolicy, observer: str, cfg: Config, device: torch.device):
        super().__init__()
        self.observer = observer
        self.cfg = cfg
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad = False
        self.embed_adapter = LoRAResidual(cfg.embed_dim, cfg.lora_rank, cfg.lora_alpha)
        self.policy_delta = LoRAPolicyDelta(cfg.hidden_dim, 2, cfg.lora_rank, cfg.lora_alpha)
        self.register_buffer("obs_mask", make_mask(observer, cfg, device))
        self.to(device)

    @property
    def uses_external_memory(self) -> bool:
        return self.observer in MEMORY_OBSERVERS

    @property
    def uses_prioritized_replay(self) -> bool:
        return self.observer == "O4R"

    @property
    def uses_dream(self) -> bool:
        return self.observer == "O4D"

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.embed_adapter.parameters()
        yield from self.policy_delta.parameters()

    def apply_observer(self, frames: torch.Tensor) -> torch.Tensor:
        neutral = torch.full_like(frames, 0.5)
        mask = self.obs_mask.unsqueeze(1)
        return neutral + mask * (frames - neutral)

    def base_key(self, frames: torch.Tensor) -> torch.Tensor:
        observed = self.apply_observer(frames)
        with torch.no_grad():
            e = self.base.encode(observed)
        return e.mean(dim=1)

    def forward(
        self,
        frames: torch.Tensor,
        memory: Optional[AssociativeMemory] = None,
        *,
        disable_lora: bool = False,
        disable_external_memory: bool = False,
        force_no_short_memory: bool = False,
        return_trace: bool = False,
    ):
        observed = self.apply_observer(frames)
        with torch.no_grad():
            emb_base = self.base.encode(observed)
        query_key = emb_base.mean(dim=1)
        emb = emb_base if disable_lora else self.embed_adapter(emb_base)
        h = self.base.hidden(emb, reset_each_step=force_no_short_memory)
        logits = self.base.policy(h)
        if not disable_lora:
            logits = logits + self.policy_delta(h)

        mem_stats = [AssociativeMemory.empty_stats() for _ in range(frames.shape[0])]
        paths: List[List[int]] = [[] for _ in range(frames.shape[0])]
        if self.uses_external_memory and memory is not None and len(memory) and not disable_external_memory:
            bias_np, mem_stats, paths = memory.retrieve(query_key.detach().cpu().numpy())
            logits = logits + torch.from_numpy(bias_np).to(logits.device, dtype=logits.dtype)

        if return_trace:
            trace = {
                "query_key": query_key.detach(),
                "adapted_embedding": emb.mean(dim=1).detach(),
                "hidden": h.detach(),
                "memory_stats": mem_stats,
                "paths": paths,
            }
            return logits, trace
        return logits

    def lora_matrices(self) -> Dict[str, np.ndarray]:
        return {
            "embed": self.embed_adapter.delta_weight().detach().cpu().numpy(),
            "policy": self.policy_delta.delta_weight().detach().cpu().numpy(),
        }


# -----------------------------------------------------------------------------
# Model initialization / base pretraining
# -----------------------------------------------------------------------------


def count_params(m: nn.Module) -> Tuple[int, int]:
    return int(sum(p.numel() for p in m.parameters())), int(sum(p.numel() for p in m.parameters() if p.requires_grad))


def stack_batch(ds: CurriculumDataset, indices: Sequence[int], device: torch.device):
    items = [ds[int(i)] for i in indices]
    x = torch.stack([z[0] for z in items]).to(device)
    y = torch.stack([z[1] for z in items]).to(device)
    infos = [z[2] for z in items]
    return x, y, infos


def pretrain_base(cfg: Config, device: torch.device) -> BasePolicy:
    seed = 91001
    set_global_seed(seed)
    base = BasePolicy(cfg).to(device)
    ds = CurriculumDataset(cfg.base_pretrain_episodes, seed + 200, cfg, "BASE", "train",
                           ("base", "crossing", "warehouse", "factory", "harbor"))
    opt = torch.optim.Adam(base.parameters(), lr=cfg.base_lr)
    rng = np.random.default_rng(seed + 300)
    steps = max(1, math.ceil(len(ds) / cfg.batch_size))
    for epoch in range(cfg.base_pretrain_epochs):
        perm = rng.permutation(len(ds))
        losses, accs = [], []
        for step in range(steps):
            idx = perm[step*cfg.batch_size:(step+1)*cfg.batch_size]
            if len(idx) == 0:
                continue
            x, y, _ = stack_batch(ds, idx, device)
            logits = base(x)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
            accs.append(float((logits.argmax(1) == y).float().mean().item()))
        print(f"[base-balanced] epoch {epoch+1}/{cfg.base_pretrain_epochs} loss={np.mean(losses):.4f} acc={np.mean(accs):.3f}")
    return base.eval()


def make_shared_lora_state(base: BasePolicy, cfg: Config, device: torch.device, seed: int) -> Dict[str, torch.Tensor]:
    # All observers start from exactly the same A/B matrices.
    set_global_seed(seed)
    tmp = Agent(base, "O3", cfg, device)
    state = {}
    for prefix, module in (("embed", tmp.embed_adapter), ("policy", tmp.policy_delta)):
        for k, v in module.state_dict().items():
            state[f"{prefix}.{k}"] = v.detach().clone()
    return state


def load_lora_state(agent: Agent, state: Dict[str, torch.Tensor]) -> None:
    e = {k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("embed.")}
    p = {k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("policy.")}
    agent.embed_adapter.load_state_dict(e)
    agent.policy_delta.load_state_dict(p)


# -----------------------------------------------------------------------------
# Learning / memory / dream
# -----------------------------------------------------------------------------


def deterministic_batch_indices(n: int, batch_size: int, n_steps: int, seed: int) -> List[np.ndarray]:
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_steps):
        out.append(rng.integers(0, n, size=batch_size, endpoint=False))
    return out


def deterministic_uniforms(batch_size: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).random(batch_size).astype(np.float32)


def reinforce_loss(logits: torch.Tensor, labels: torch.Tensor, uniforms: np.ndarray, baseline: float, entropy_bonus: float):
    prob = logits.softmax(1)
    u = torch.from_numpy(uniforms).to(logits.device)
    actions = (u < prob[:, 1]).long()
    rewards = torch.where(actions == labels, torch.ones_like(actions, dtype=torch.float32), -torch.ones_like(actions, dtype=torch.float32))
    logp = torch.log(torch.gather(prob, 1, actions[:, None]).squeeze(1).clamp_min(1e-8))
    entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(1)
    advantage = rewards - float(baseline)
    loss = -(logp * advantage.detach()).mean() - entropy_bonus * entropy.mean()
    return loss, rewards.detach(), actions.detach()


def policy_update_loss(logits: torch.Tensor, labels: torch.Tensor, uniforms: np.ndarray, baseline: float, cfg: Config):
    if cfg.learning_objective == "reinforce":
        return reinforce_loss(logits, labels, uniforms, baseline, cfg.entropy_bonus)
    prob = logits.softmax(1)
    entropy = -(prob * torch.log(prob.clamp_min(1e-8))).sum(1).mean()
    loss = F.cross_entropy(logits, labels) - cfg.entropy_bonus * entropy
    actions = logits.argmax(1)
    rewards = torch.where(actions == labels, torch.ones_like(actions, dtype=torch.float32), -torch.ones_like(actions, dtype=torch.float32))
    return loss, rewards.detach(), actions.detach()


def record_from_item(agent: Agent, item, device: torch.device, *, is_dream: bool = False, strength: float = 1.0) -> MemoryRecord:
    x, y, info = item
    xb = x.unsqueeze(0).to(device)
    key = agent.base_key(xb).detach().cpu().numpy()[0]
    raw = (x.permute(0, 2, 3, 1).numpy() * 255.0).clip(0, 255).astype(np.uint8)
    return MemoryRecord(
        key=key.astype(np.float32), label=int(y.item()), style=str(info["style"]), phase=str(info["phase"]),
        central_bit=int(info["central_bit"]), peripheral_bit=int(info["peripheral_bit"]),
        frames=raw, is_dream=is_dream, strength=float(strength),
    )


def ingest_memory(agent: Agent, memory: AssociativeMemory, ds: CurriculumDataset, cfg: Config, device: torch.device, seed: int) -> None:
    if not agent.uses_external_memory:
        return
    rng = np.random.default_rng(seed)
    k = min(cfg.memory_ingest_per_epoch, len(ds))
    ids = rng.choice(len(ds), size=k, replace=False)
    for idx in ids:
        memory.add(record_from_item(agent, ds[int(idx)], device), rng)
    memory.rebuild_graph()


def hybrid_dream_record(agent: Agent, memory: AssociativeMemory, cfg: Config, device: torch.device, rng: np.random.Generator) -> Optional[MemoryRecord]:
    if len(memory.records) < cfg.dream_min_memory:
        return None

    # Central donor: favor earlier central-learning knowledge (A) or C.
    c_pool = [i for i, r in enumerate(memory.records) if r.phase in ("A", "C") and not r.is_dream]
    # Peripheral donor: favor delayed-context knowledge (B) or C.
    p_pool = [i for i, r in enumerate(memory.records) if r.phase in ("B", "C") and not r.is_dream]
    if not c_pool or not p_pool:
        return None

    ia = int(rng.choice(c_pool)); ib = int(rng.choice(p_pool))
    ra, rb = memory.records[ia], memory.records[ib]

    # Encourage cross-style/cross-phase recombination, but never require it so the
    # mechanism does not become brittle on small memories.
    if rng.random() < cfg.dream_cross_style_prob:
        alt = [i for i in p_pool if memory.records[i].style != ra.style]
        if alt:
            ib = int(rng.choice(alt)); rb = memory.records[ib]
    if rng.random() < cfg.dream_cross_phase_prob:
        alt = [i for i in p_pool if memory.records[i].phase != ra.phase]
        if alt:
            ib = int(rng.choice(alt)); rb = memory.records[ib]

    frames = ra.frames.copy()
    boxes = semantic_boxes(cfg.image_size)
    x0, y0, x1, y1 = boxes["peripheral"]
    # Replace the transient peripheral fragment with the donor fragment.
    frames[cfg.cue_step, y0:y1+1, x0:x1+1, :] = rb.frames[cfg.cue_step, y0:y1+1, x0:x1+1, :]

    # Synthetic causal validator: phase-C truth is known from the generator.
    cbit, pbit = int(ra.central_bit), int(rb.peripheral_bit)
    label = phase_rule("C", cbit, pbit)

    x = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
    info = {"style": f"dream:{ra.style}+{rb.style}", "phase": "C", "central_bit": cbit, "peripheral_bit": pbit}
    item = (x, torch.tensor(label, dtype=torch.long), info)
    rec = record_from_item(agent, item, device, is_dream=True, strength=cfg.dream_memory_weight)
    # Validated novel recombinations receive stronger consolidation. Novelty is
    # estimated from representation distance, not from the hidden true combo label.
    if len(memory.records):
        sims = memory.key_matrix() @ AssociativeMemory._norm(rec.key.astype(np.float32))
        novelty = float(np.clip(1.0 - np.max(sims), 0.0, 1.0))
    else:
        novelty = 1.0
    rec.strength = float(np.clip(cfg.dream_memory_weight + 0.9*novelty, cfg.dream_memory_weight, 1.35))
    return rec


def generate_dream_batch(agent: Agent, memory: AssociativeMemory, cfg: Config, device: torch.device, batch_size: int, seed: int):
    rng = np.random.default_rng(seed)
    records: List[MemoryRecord] = []
    attempts = 0
    while len(records) < batch_size and attempts < batch_size * 12:
        attempts += 1
        rec = hybrid_dream_record(agent, memory, cfg, device, rng)
        if rec is not None:
            records.append(rec)
    if not records:
        return None, {"dream_candidates": float(attempts), "dream_valid": 0.0,
                      "dream_00":0.0,"dream_01":0.0,"dream_10":0.0,"dream_11":0.0}
    xs = [torch.from_numpy(r.frames.copy()).permute(0, 3, 1, 2).float() / 255.0 for r in records]
    ys = [r.label for r in records]
    x = torch.stack(xs).to(device)
    y = torch.tensor(ys, dtype=torch.long, device=device)
    combo_counts = {"dream_00":0.0,"dream_01":0.0,"dream_10":0.0,"dream_11":0.0}
    for r in records:
        combo_counts[f"dream_{r.central_bit}{r.peripheral_bit}"] += 1.0
    return (x, y, records), {"dream_candidates": float(attempts), "dream_valid": float(len(records)), **combo_counts}


def lora_geometry(agent: Agent, cfg: Config) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for name, m in agent.lora_matrices().items():
        s = np.linalg.svd(m, compute_uv=False)
        norm = float(np.linalg.norm(m))
        if s.sum() > 1e-12:
            p = s / s.sum()
            eff_rank = float(np.exp(-(p * np.log(np.clip(p, 1e-12, 1.0))).sum()))
            sig_count = int(np.sum(s >= cfg.significant_sv_fraction * s[0])) if len(s) else 0
        else:
            eff_rank = 0.0; sig_count = 0
        threshold = max(1e-8, 0.05 * float(np.max(np.abs(m))) if m.size else 1e-8)
        changed_frac = float(np.mean(np.abs(m) >= threshold)) if m.size else 0.0
        out[f"{name}_lora_norm"] = norm
        out[f"{name}_effective_rank"] = eff_rank
        out[f"{name}_significant_sv"] = float(sig_count)
        out[f"{name}_changed_fraction"] = changed_frac
        out[f"{name}_positive_fraction"] = float(np.mean(m > threshold)) if m.size else 0.0
        out[f"{name}_negative_fraction"] = float(np.mean(m < -threshold)) if m.size else 0.0
    out["total_lora_norm"] = float(math.sqrt(out["embed_lora_norm"]**2 + out["policy_lora_norm"]**2))
    return out


@torch.no_grad()
def evaluate(agent: Agent, ds: CurriculumDataset, cfg: Config, device: torch.device, memory: Optional[AssociativeMemory] = None,
             *, disable_lora=False, disable_external_memory=False, force_no_short_memory=False, collect_trace=True):
    agent.eval()
    ys, preds, probs = [], [], []
    combo_correct: Dict[str, List[float]] = {"00":[],"01":[],"10":[],"11":[]}
    hidden_counts, embed_counts, mem_rows = [], [], []
    retrieval_hits = []
    batch_size = cfg.batch_size
    for start in range(0, len(ds), batch_size):
        idx = list(range(start, min(start+batch_size, len(ds))))
        x, y, infos = stack_batch(ds, idx, device)
        if collect_trace:
            logits, trace = agent(x, memory, disable_lora=disable_lora, disable_external_memory=disable_external_memory,
                                  force_no_short_memory=force_no_short_memory, return_trace=True)
            h = trace["hidden"].cpu().numpy(); e = trace["adapted_embedding"].cpu().numpy()
            hidden_counts.extend((np.abs(h) >= cfg.activation_threshold).sum(1).tolist())
            embed_counts.extend((np.abs(e) >= cfg.activation_threshold).sum(1).tolist())
            for info, ms, path in zip(infos, trace["memory_stats"], trace["paths"]):
                mem_rows.append(ms)
                if memory is not None and path:
                    target = int(info["label"])
                    retrieval_hits.append(float(np.mean([memory.records[i].label == target for i in path])))
        else:
            logits = agent(x, memory, disable_lora=disable_lora, disable_external_memory=disable_external_memory,
                           force_no_short_memory=force_no_short_memory)
        p = logits.softmax(1)[:, 1]
        pred = logits.argmax(1)
        ys.extend(y.cpu().tolist()); preds.extend(pred.cpu().tolist()); probs.extend(p.cpu().tolist())
        for info, pp, yy in zip(infos, pred.cpu().tolist(), y.cpu().tolist()):
            combo = f"{int(info['central_bit'])}{int(info['peripheral_bit'])}"
            combo_correct[combo].append(float(int(pp)==int(yy)))
    yv = np.asarray(ys, int); pv = np.asarray(preds, int)
    acc = float(np.mean(yv == pv)) if len(yv) else 0.0
    reward = float(np.mean(np.where(yv == pv, 1.0, -1.0))) if len(yv) else 0.0
    tp = int(np.sum((yv==1)&(pv==1))); tn = int(np.sum((yv==0)&(pv==0)))
    fp = int(np.sum((yv==0)&(pv==1))); fn = int(np.sum((yv==1)&(pv==0)))
    tpr = tp / max(tp+fn,1); tnr = tn / max(tn+fp,1)
    balanced_accuracy = 0.5*(tpr+tnr)
    f1_pos = 2*tp/max(2*tp+fp+fn,1); f1_neg = 2*tn/max(2*tn+fp+fn,1)
    macro_f1 = 0.5*(f1_pos+f1_neg)
    out = {
        "accuracy": acc, "balanced_accuracy": float(balanced_accuracy), "macro_f1": float(macro_f1), "reward": reward,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
        "active_hidden_units": float(np.mean(hidden_counts)) if hidden_counts else 0.0,
        "active_embedding_units": float(np.mean(embed_counts)) if embed_counts else 0.0,
        "retrieval_label_precision": float(np.mean(retrieval_hits)) if retrieval_hits else 0.0,
        "combo00_accuracy": float(np.mean(combo_correct["00"])) if combo_correct["00"] else np.nan,
        "combo01_accuracy": float(np.mean(combo_correct["01"])) if combo_correct["01"] else np.nan,
        "combo10_accuracy": float(np.mean(combo_correct["10"])) if combo_correct["10"] else np.nan,
        "combo11_accuracy": float(np.mean(combo_correct["11"])) if combo_correct["11"] else np.nan,
    }
    if mem_rows:
        for k in mem_rows[0]:
            out[k] = float(np.mean([r[k] for r in mem_rows]))
    else:
        out.update(AssociativeMemory.empty_stats())
    return out


def resource_cost(observer: str, cfg: Config, memory: Optional[AssociativeMemory], dream_stats: Dict[str, float], n_updates: int) -> float:
    w = observer_weights(observer, cfg)
    visual = 0.5*w["central"] + 0.3*w["peripheral"] + 0.2*w["background"]
    mem = (len(memory.records) / max(cfg.memory_max_records, 1)) if memory is not None else 0.0
    graph = (sum(len(v) for v in memory.adj.values()) / max(cfg.memory_max_records*8, 1)) if memory is not None else 0.0
    dream = dream_stats.get("dream_valid", 0.0) / max(n_updates * cfg.batch_size, 1)
    return float(0.45*visual + 0.25*mem + 0.15*graph + 0.15*dream)


def attention_proxy(observer: str, cfg: Config, info: Dict[str, float]) -> Dict[str, float]:
    w = observer_weights(observer, cfg)
    raw = np.asarray([w["central"], w["peripheral"]], float)
    a = raw / max(raw.sum(), 1e-12)
    r = np.asarray([max(info["I_c_y"], 1e-8), max(info["I_p_y_given_c"], 1e-8)], float)
    r = r / max(r.sum(), 1e-12)
    c_att = 1.0 - 0.5*float(np.sum((a-r)**2))
    return {"C_att": float(np.clip(c_att, 0, 1)), "D_cue": float(a[1]), "N_sup": float(1.0 - w["background"])}


def legacy_owge_proxy(observer: str, cfg: Config, info: Dict[str, float], full: Dict[str, float], ablations: Dict[str, Dict[str, float]],
                      base_acc: float, cost: float, replay_gain: float = 0.0) -> Dict[str, float]:
    att = attention_proxy(observer, cfg, info)
    acc = full["accuracy"]
    no_short = ablations["no_short"]["accuracy"]
    no_lora = ablations["no_lora"]["accuracy"]
    no_both = ablations["no_lora_no_short"]["accuracy"]
    a_util = float(np.clip((acc-no_short)/max(acc,1e-8), 0, 1))
    t_gain = float(np.clip((acc-base_acc)/max(1-base_acc,1e-8), 0, 1))
    i_ret = float(np.clip((no_short-no_both)/max(1-no_both,1e-8), 0, 1))
    r_ret = float(np.clip(0.5 + np.clip(acc-no_short, -0.5, 0.5), 0, 1))
    u_adapt = float(np.clip(acc, 0, 1))
    q = (att["C_att"] + att["D_cue"] + 0.7*att["N_sup"] + a_util + 0.8*max(replay_gain,0) + t_gain) / 5.5
    q = q / (1 + 0.7*cost)
    eps = 1e-6
    p_chain = float(((i_ret+eps)*(r_ret+eps)*(u_adapt+eps))**(1/3))
    return {
        **att, "A_util": a_util, "T_gain": t_gain, "I_ret_legacy": i_ret,
        "R_ret_legacy": r_ret, "U_adapt": u_adapt, "Q_proc_legacy": q,
        "P_chain_legacy": p_chain, "OWGE_legacy": float(q*p_chain),
    }


def internalization_metrics(full: Dict[str, float], ab: Dict[str, Dict[str, float]], observer: str) -> Dict[str, float]:
    def gain(a: float, b: float) -> float:
        return float(np.clip((a-b)/max(1-b,1e-8), 0, 1))
    i_param = gain(full["balanced_accuracy"], ab["no_lora"]["balanced_accuracy"])
    i_external = gain(full["balanced_accuracy"], ab["no_external"]["balanced_accuracy"]) if observer in MEMORY_OBSERVERS else 0.0
    i_short = gain(full["balanced_accuracy"], ab["no_short"]["balanced_accuracy"])
    # Durable after episode reset: LoRA + external memory may remain, GRU temporal state is removed.
    durable_acc = ab["no_short"]["balanced_accuracy"]
    stripped_acc = ab["no_lora_no_external_no_short"]["balanced_accuracy"]
    i_durable_joint = gain(durable_acc, stripped_acc)
    return {
        "I_param": i_param, "I_external_memory": i_external, "I_short_term": i_short,
        "durable_no_short_accuracy": durable_acc,
        "stripped_no_durable_accuracy": stripped_acc,
        "I_durable_joint": i_durable_joint,
    }


def train_phase(agent: Agent, observer: str, phase: str, train_ds: CurriculumDataset, val_ds: CurriculumDataset,
                cfg: Config, device: torch.device, seed: int, memory: Optional[AssociativeMemory]) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    epochs = {"A": cfg.phase_a_epochs, "B": cfg.phase_b_epochs, "C": cfg.phase_c_epochs}[phase]
    opt = torch.optim.Adam(list(agent.trainable_parameters()), lr=cfg.lora_lr)
    steps_per_epoch = max(4, math.ceil(len(train_ds)/cfg.batch_size))
    common_steps = max(1, int(round(steps_per_epoch * cfg.common_real_fraction)))
    extra_steps = max(1, steps_per_epoch - common_steps)
    running_baseline = 0.0
    learning: List[Dict[str, float]] = []
    priorities = np.ones(len(train_ds), dtype=np.float64)
    dream_stats_total = {"dream_candidates": 0.0, "dream_valid": 0.0,
                         "dream_00":0.0,"dream_01":0.0,"dream_10":0.0,"dream_11":0.0}

    for epoch in range(epochs):
        agent.train()
        common_schedule = deterministic_batch_indices(len(train_ds), cfg.batch_size, common_steps,
                                                      seed + 100000*ord(phase) + 1000*epoch + 17)
        control_schedule = deterministic_batch_indices(len(train_ds), cfg.batch_size, extra_steps,
                                                       seed + 100000*ord(phase) + 1000*epoch + 29)
        rewards_epoch = []
        update_idx = 0

        # Common real-data gradient steps, identical sample indices across observers.
        for idx in common_schedule:
            x, y, infos = stack_batch(train_ds, idx, device)
            logits = agent(x, memory)
            u = deterministic_uniforms(len(y), seed + 700000 + 10000*epoch + update_idx)
            loss, rewards, _ = policy_update_loss(logits, y, u, running_baseline, cfg)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(agent.trainable_parameters()), cfg.grad_clip); opt.step()
            running_baseline = 0.9*running_baseline + 0.1*float(rewards.mean().item())
            rewards_epoch.extend(rewards.cpu().tolist())
            # Update replay priorities from current errors/uncertainty.
            with torch.no_grad():
                p = logits.softmax(1).cpu().numpy(); yy = y.cpu().numpy()
                pr = (p.argmax(1)!=yy).astype(float) + 0.25*(1-np.abs(p[:,1]-0.5)*2)
                for ii, vv in zip(idx, pr): priorities[int(ii)] = 0.8*priorities[int(ii)] + 0.2*float(vv)
            update_idx += 1

        # Mechanism-specific extra updates under the SAME optimizer-step budget.
        for ex in range(extra_steps):
            is_dream_step = False
            pending_dream_records: List[MemoryRecord] = []
            if observer == "O4R":
                rng = np.random.default_rng(seed + 800000 + epoch*1000 + ex)
                prob = priorities / max(priorities.sum(), 1e-12)
                idx = rng.choice(len(train_ds), size=cfg.batch_size, replace=True, p=prob)
                x, y, infos = stack_batch(train_ds, idx, device)
            elif observer == "O4D" and phase == "C" and memory is not None:
                dream_batch, ds_stats = generate_dream_batch(agent, memory, cfg, device, cfg.batch_size,
                                                              seed + 900000 + epoch*1000 + ex)
                for k in dream_stats_total:
                    dream_stats_total[k] += float(ds_stats.get(k,0.0))
                if dream_batch is not None:
                    x, y, dream_records = dream_batch
                    infos = [{"index": -1} for _ in range(len(y))]
                    is_dream_step = True
                    pending_dream_records = list(dream_records)
                else:
                    idx = control_schedule[ex]
                    x, y, infos = stack_batch(train_ds, idx, device)
            else:
                idx = control_schedule[ex]
                x, y, infos = stack_batch(train_ds, idx, device)

            # Dream consolidation explicitly forces the recombined episode through
            # the durable LoRA path before storing it in external memory.
            logits = agent(x, memory, disable_external_memory=is_dream_step)
            u = deterministic_uniforms(len(y), seed + 700000 + 10000*epoch + update_idx)
            loss, rewards, _ = policy_update_loss(logits, y, u, running_baseline, cfg)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(list(agent.trainable_parameters()), cfg.grad_clip); opt.step()
            running_baseline = 0.9*running_baseline + 0.1*float(rewards.mean().item())
            rewards_epoch.extend(rewards.cpu().tolist())
            if is_dream_step and memory is not None and pending_dream_records:
                rrng = np.random.default_rng(seed + 910000 + epoch*1000 + ex)
                for r in pending_dream_records:
                    memory.add(r, rrng)
                memory.rebuild_graph()
            update_idx += 1

        # Gradual episodic memory internalization after each epoch.
        if memory is not None:
            ingest_memory(agent, memory, train_ds, cfg, device, seed + 950000 + epoch*997 + ord(phase))

        val = evaluate(agent, val_ds, cfg, device, memory, collect_trace=True)
        geo = lora_geometry(agent, cfg)
        ms = memory.summary() if memory is not None else {"memory_records":0.0,"memory_edges_directed":0.0,"memory_dream_records":0.0,"memory_dream_fraction":0.0}
        learning.append({
            "phase": phase, "epoch": epoch+1, "train_reward": float(np.mean(rewards_epoch)) if rewards_epoch else 0.0,
            "val_accuracy": val["accuracy"], "val_reward": val["reward"],
            "active_hidden_units": val["active_hidden_units"], "active_embedding_units": val["active_embedding_units"],
            "retrieval_label_precision": val["retrieval_label_precision"],
            **geo, **ms,
            "dream_candidates_cum": dream_stats_total["dream_candidates"], "dream_valid_cum": dream_stats_total["dream_valid"],
            "dream_00_cum": dream_stats_total["dream_00"], "dream_01_cum": dream_stats_total["dream_01"],
            "dream_10_cum": dream_stats_total["dream_10"], "dream_11_cum": dream_stats_total["dream_11"],
            "gradient_updates_this_epoch": float(update_idx),
        })
        print(f"[{observer} seed={seed} phase={phase}] epoch {epoch+1}/{epochs} valAcc={val['accuracy']:.3f} "
              f"mem={int(ms['memory_records'])} dream={int(dream_stats_total['dream_valid'])}")

    return learning, dream_stats_total


# -----------------------------------------------------------------------------
# Statistics / plots
# -----------------------------------------------------------------------------


def paired_test(df: pd.DataFrame, metric: str, a: str, b: str, phase: str, alternative: str, name: str) -> Dict[str, object]:
    x = df[(df.observer==a)&(df.phase==phase)][["seed",metric]].rename(columns={metric:"a"})
    y = df[(df.observer==b)&(df.phase==phase)][["seed",metric]].rename(columns={metric:"b"})
    m = x.merge(y, on="seed")
    d = m.a - m.b
    if len(m) >= 2 and stats is not None:
        t, p2 = stats.ttest_rel(m.a, m.b)
        if alternative == "greater": p = p2/2 if t>0 else 1-p2/2
        elif alternative == "less": p = p2/2 if t<0 else 1-p2/2
        else: p = p2
        sem = stats.sem(d)
        ci = stats.t.interval(0.95, len(d)-1, loc=float(d.mean()), scale=float(sem)) if np.isfinite(sem) else (np.nan,np.nan)
    else:
        t=p=np.nan; ci=(np.nan,np.nan)
    dz = float(d.mean()/d.std(ddof=1)) if len(d)>1 and d.std(ddof=1)>1e-12 else 0.0
    return {
        "hypothesis": name, "phase": phase, "metric": metric, "comparison": f"{a} {alternative} {b}",
        "n_pairs": int(len(m)), "mean_a": float(m.a.mean()) if len(m) else np.nan,
        "mean_b": float(m.b.mean()) if len(m) else np.nan, "mean_difference": float(d.mean()) if len(m) else np.nan,
        "ci95_low": float(ci[0]), "ci95_high": float(ci[1]), "paired_cohens_dz": dz,
        "t_stat": float(t), "one_sided_p": float(p), "reject_H0_0.05": bool(p<0.05) if np.isfinite(p) else False,
    }


def holm_adjust(pvals: Sequence[float]) -> List[float]:
    p = np.asarray(pvals, float)
    out = np.full(len(p), np.nan)
    finite = np.where(np.isfinite(p))[0]
    if not len(finite): return out.tolist()
    order = finite[np.argsort(p[finite])]
    running = 0.0
    m = len(order)
    for rank, idx in enumerate(order):
        val = min(1.0, (m-rank)*p[idx])
        running = max(running, val)
        out[idx] = running
    return out.tolist()


def build_hypotheses(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    # PRIMARY: specifically scoped to compositional transfer.
    rows.append(paired_test(metrics, "balanced_accuracy", "O4D", "O1", "C", "greater",
                            "PRIMARY H_A: O4D > O1 on held-out compositional transfer"))
    secondary = [
        paired_test(metrics, "balanced_accuracy", "O4D", "O4R", "C", "greater", "S1: associative dream O4D > replay-only O4R"),
        paired_test(metrics, "balanced_accuracy", "O4D", "O4M", "C", "greater", "S2: dream adds value beyond associative memory alone"),
        paired_test(metrics, "balanced_accuracy", "O4M", "O3", "C", "greater", "S3: associative episodic memory O4M > weighted observation O3"),
        paired_test(metrics, "balanced_accuracy", "O3", "O1", "B", "greater", "S4: O3 > O1 when delayed peripheral information matters"),
        paired_test(metrics, "I_durable_joint", "O4D", "O4R", "C", "greater", "S5: O4D has greater durable internalization than replay-only O4R"),
        paired_test(metrics, "retrieval_label_precision", "O4D", "O4M", "C", "greater", "S6: dream improves associative retrieval precision"),
        paired_test(metrics, "combo00_accuracy", "O4D", "O4M", "C", "greater", "S7: dream improves the specifically unseen (0,0) composition"),
    ]
    if secondary:
        adj = holm_adjust([r["one_sided_p"] for r in secondary])
        for r, a in zip(secondary, adj):
            r["holm_adjusted_p"] = a
            r["reject_H0_holm_0.05"] = bool(a<0.05) if np.isfinite(a) else False
    rows[0]["holm_adjusted_p"] = np.nan
    rows[0]["reject_H0_holm_0.05"] = rows[0]["reject_H0_0.05"]
    return pd.DataFrame(rows + secondary)


def mean_sem(x: Sequence[float]):
    a = np.asarray(x,float)
    return float(a.mean()), float(a.std(ddof=1)/math.sqrt(len(a))) if len(a)>1 else 0.0


def save_plot(path: Path):
    plt.tight_layout(); plt.savefig(path, dpi=170); plt.close()


def make_plots(out: Path, metrics: pd.DataFrame, learning: pd.DataFrame, hypotheses: pd.DataFrame, geometry: pd.DataFrame):
    pdir = out/"plots"; pdir.mkdir(exist_ok=True)

    # 1 phase accuracy
    piv = metrics.groupby(["phase","observer"]).accuracy.mean().unstack()
    piv.plot(kind="bar", figsize=(9,5)); plt.ylabel("Held-out accuracy"); plt.title("1. Accuracy across curriculum regimes")
    save_plot(pdir/"01_phase_accuracy.png")

    # 2 phase reward
    metrics.groupby(["phase","observer"]).reward.mean().unstack().plot(kind="bar", figsize=(9,5)); plt.ylabel("Reward"); plt.title("2. Reward across regimes")
    save_plot(pdir/"02_phase_reward.png")

    # 3 primary paired scatter
    a=metrics[(metrics.phase=="C")&(metrics.observer=="O1")][["seed","balanced_accuracy"]].rename(columns={"balanced_accuracy":"O1"})
    b=metrics[(metrics.phase=="C")&(metrics.observer=="O4D")][["seed","balanced_accuracy"]].rename(columns={"balanced_accuracy":"O4D"})
    m=a.merge(b,on="seed")
    plt.figure(figsize=(6,6)); plt.scatter(m.O1,m.O4D); lo=min(m.O1.min(),m.O4D.min()); hi=max(m.O1.max(),m.O4D.max()); plt.plot([lo,hi],[lo,hi],linestyle="--")
    plt.xlabel("O1 transfer balanced accuracy"); plt.ylabel("O4D transfer balanced accuracy"); plt.title("3. Primary paired comparison: O4D vs O1")
    save_plot(pdir/"03_primary_paired_o4d_o1.png")

    # 4 difference distribution
    plt.figure(figsize=(7,4)); plt.hist(m.O4D-m.O1,bins=min(12,max(5,len(m)//3))); plt.axvline(0,linestyle="--"); plt.xlabel("O4D - O1 balanced accuracy"); plt.ylabel("Seeds"); plt.title("4. Primary paired differences")
    save_plot(pdir/"04_primary_difference_hist.png")

    # 5 learning curves
    plt.figure(figsize=(9,5))
    for obs in OBSERVERS:
        sub=learning[learning.observer==obs]
        # global curriculum epoch
        if len(sub):
            g=sub.groupby("global_epoch").val_accuracy.mean(); plt.plot(g.index,g.values,label=obs)
    plt.xlabel("Curriculum epoch"); plt.ylabel("Validation accuracy"); plt.legend(); plt.title("5. Curriculum learning curves")
    save_plot(pdir/"05_learning_curves.png")

    # 6 O4 variants in C
    metrics[(metrics.phase=="C") & metrics.observer.isin(["O3","O4R","O4M","O4D"])].groupby("observer").balanced_accuracy.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("Transfer balanced accuracy"); plt.title("6. Mechanism decomposition in compositional transfer")
    save_plot(pdir/"06_mechanism_decomposition.png")

    # 7 retrieval precision
    metrics[metrics.phase=="C"].groupby("observer").retrieval_label_precision.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("Top-path label precision"); plt.title("7. Associative retrieval precision")
    save_plot(pdir/"07_retrieval_precision.png")

    # 8 retrieved nodes
    metrics[metrics.phase=="C"].groupby("observer").retrieved_nodes.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("Nodes per query"); plt.title("8. Retrieved associative pathways")
    save_plot(pdir/"08_retrieved_nodes.png")

    # 9 retrieved edges
    metrics[metrics.phase=="C"].groupby("observer").retrieved_edges.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("Traversed graph edges"); plt.title("9. Graph pathways traversed")
    save_plot(pdir/"09_retrieved_edges.png")

    # 10 pathway style diversity
    metrics[metrics.phase=="C"].groupby("observer").path_style_diversity.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("Distinct styles in retrieval path"); plt.title("10. Cross-style pathway diversity")
    save_plot(pdir/"10_path_style_diversity.png")

    # 11 dream fraction retrieved
    metrics[metrics.phase=="C"].groupby("observer").retrieved_dream_fraction.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("Dream fraction"); plt.title("11. Dream records used during transfer")
    save_plot(pdir/"11_retrieved_dream_fraction.png")


    # 11b pathway reuse Jaccard
    metrics[metrics.phase=="C"].groupby("observer").pathway_reuse_jaccard.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("Active-dimension Jaccard"); plt.title("11b. Representation pathway reuse during retrieval")
    save_plot(pdir/"11b_pathway_reuse_jaccard.png")

    # 12 active hidden units
    metrics.groupby(["phase","observer"]).active_hidden_units.mean().unstack().plot(kind="bar", figsize=(9,5)); plt.ylabel("Active GRU units"); plt.title("12. Neural pathway activation count")
    save_plot(pdir/"12_active_hidden_units.png")

    # 13 active embedding units
    metrics.groupby(["phase","observer"]).active_embedding_units.mean().unstack().plot(kind="bar", figsize=(9,5)); plt.ylabel("Active embedding units"); plt.title("13. Representation activation count")
    save_plot(pdir/"13_active_embedding_units.png")

    # 14 LoRA norm
    geometry[geometry.phase=="C"].groupby("observer").total_lora_norm.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("||Delta W||"); plt.title("14. Durable LoRA update magnitude")
    save_plot(pdir/"14_lora_norm.png")

    # 15 effective rank
    geometry[geometry.phase=="C"].groupby("observer")[["embed_effective_rank","policy_effective_rank"]].mean().plot(kind="bar", figsize=(8,4)); plt.ylabel("Effective rank"); plt.title("15. LoRA update effective rank")
    save_plot(pdir/"15_lora_effective_rank.png")

    # 16 changed weight fraction
    geometry[geometry.phase=="C"].groupby("observer")[["embed_changed_fraction","policy_changed_fraction"]].mean().plot(kind="bar", figsize=(8,4)); plt.ylabel("Fraction above 5% max magnitude"); plt.title("16. Fraction of materially changed LoRA pathways")
    save_plot(pdir/"16_changed_weight_fraction.png")

    # 17 internalization decomposition
    metrics[metrics.phase=="C"].groupby("observer")[["I_param","I_external_memory","I_short_term","I_durable_joint"]].mean().plot(kind="bar", figsize=(10,5)); plt.ylabel("Normalized causal gain"); plt.title("17. Internalization and memory ablations")
    save_plot(pdir/"17_internalization_decomposition.png")

    # 18 legacy OWGE
    metrics[metrics.phase=="C"].groupby("observer").OWGE_legacy.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("Legacy OWGE proxy"); plt.title("18. Legacy OWGE proxy (diagnostic only)")
    save_plot(pdir/"18_legacy_owge.png")

    # 19 resource cost
    metrics[metrics.phase=="C"].groupby("observer").resource_cost.mean().plot(kind="bar", figsize=(7,4)); plt.ylabel("Relative resource cost"); plt.title("19. Resource cost")
    save_plot(pdir/"19_resource_cost.png")

    # 20 performance-cost scatter
    g=metrics[metrics.phase=="C"].groupby("observer")[["balanced_accuracy","resource_cost"]].mean().reset_index()
    plt.figure(figsize=(7,5)); plt.scatter(g.resource_cost,g.balanced_accuracy)
    for _,r in g.iterrows(): plt.text(r.resource_cost,r.balanced_accuracy,r.observer)
    plt.xlabel("Resource cost"); plt.ylabel("Transfer balanced accuracy"); plt.title("20. Transfer-performance / resource trade-off")
    save_plot(pdir/"20_performance_cost.png")

    # 21 p-values
    plt.figure(figsize=(10,4)); vals=-np.log10(np.clip(hypotheses.one_sided_p.astype(float),1e-12,1)); plt.bar(range(len(vals)),vals); plt.axhline(-math.log10(0.05),linestyle="--"); plt.xticks(range(len(vals)),[f"H{i+1}" for i in range(len(vals))]); plt.ylabel("-log10(p)"); plt.title("21. Prespecified hypothesis tests")
    save_plot(pdir/"21_hypothesis_pvalues.png")

    # 22 memory records over curriculum
    plt.figure(figsize=(9,5))
    for obs in ["O4M","O4D"]:
        sub=learning[learning.observer==obs]; g=sub.groupby("global_epoch").memory_records.mean(); plt.plot(g.index,g.values,label=obs)
    plt.xlabel("Curriculum epoch"); plt.ylabel("Memory records"); plt.legend(); plt.title("22. Episodic memory growth")
    save_plot(pdir/"22_memory_growth.png")

    # 23 dream records over curriculum
    sub=learning[learning.observer=="O4D"]
    if len(sub):
        g=sub.groupby("global_epoch").memory_dream_records.mean(); plt.figure(figsize=(8,4)); plt.plot(g.index,g.values); plt.xlabel("Curriculum epoch"); plt.ylabel("Dream records"); plt.title("23. Validated dream consolidation growth"); save_plot(pdir/"23_dream_growth.png")


    # 23b dream composition coverage
    if len(sub):
        last=sub.sort_values("global_epoch").groupby("seed",as_index=False).tail(1)
        vals=[last["dream_00_cum"].mean(),last["dream_01_cum"].mean(),last["dream_10_cum"].mean(),last["dream_11_cum"].mean()]
        plt.figure(figsize=(7,4)); plt.bar(["00","01","10","11"],vals); plt.ylabel("Mean validated dream records generated"); plt.title("23b. Dream recombination coverage by factor combination"); save_plot(pdir/"23b_dream_combo_coverage.png")


    # 23c external visual-style transfer
    st=metrics[metrics.phase=="C_style_shift"].groupby("observer").balanced_accuracy.mean()
    if len(st):
        st.plot(kind="bar",figsize=(7,4)); plt.ylabel("Balanced accuracy"); plt.title("23c. Secondary held-out visual-style transfer"); save_plot(pdir/"23c_style_shift_transfer.png")

    # 24 retention after final phase
    ret=metrics[metrics.phase.isin(["A_after_C","B_after_C"])].groupby(["phase","observer"]).accuracy.mean().unstack()
    if len(ret):
        ret.plot(kind="bar",figsize=(9,5)); plt.ylabel("Accuracy after Phase C"); plt.title("24. Retention of prior knowledge after compositional learning"); save_plot(pdir/"24_retention_after_c.png")


def make_preview(out: Path, cfg: Config):
    d=out/"data_preview"; d.mkdir(exist_ok=True)
    rows=[]
    for j,(phase,split,styles) in enumerate([("A","test",TEST_STYLES),("B","test",TEST_STYLES),("C","train",TRAIN_STYLES),("C","test",TEST_STYLES)]):
        ds=CurriculumDataset(4, 33001+j*100, cfg, phase, split, styles)
        for i in range(4):
            x,y,info=ds[i]; epdir=d/f"{phase}_{split}_{i:02d}"; epdir.mkdir(exist_ok=True)
            for t in range(cfg.episode_len):
                arr=(x[t].permute(1,2,0).numpy()*255).astype(np.uint8); Image.fromarray(arr).save(epdir/f"frame_{t:02d}.png")
            rows.append({**info,"split":split,"folder":str(epdir.name)})
    pd.DataFrame(rows).to_csv(d/"test_manifest.csv",index=False)


# -----------------------------------------------------------------------------
# Main experiment
# -----------------------------------------------------------------------------


def datasets_for_seed(cfg: Config, seed: int):
    # Source train/val use crossing/warehouse. Held-out tests use factory/harbor.
    return {
        "A_train": CurriculumDataset(cfg.phase_a_train, seed+1000, cfg, "A", "train", TRAIN_STYLES),
        "A_val": CurriculumDataset(cfg.val_episodes, seed+1100, cfg, "A", "val", TRAIN_STYLES),
        "A_test": CurriculumDataset(cfg.test_a_episodes, seed+1200, cfg, "A", "test", TEST_STYLES),
        "B_train": CurriculumDataset(cfg.phase_b_train, seed+2000, cfg, "B", "train", TRAIN_STYLES),
        "B_val": CurriculumDataset(cfg.val_episodes, seed+2100, cfg, "B", "val", TRAIN_STYLES),
        "B_test": CurriculumDataset(cfg.test_b_episodes, seed+2200, cfg, "B", "test", TEST_STYLES),
        "C_train": CurriculumDataset(cfg.phase_c_train, seed+3000, cfg, "C", "train", TRAIN_STYLES),
        "C_val": CurriculumDataset(cfg.val_episodes, seed+3100, cfg, "C", "val", TRAIN_STYLES),
        "C_test": CurriculumDataset(cfg.test_c_episodes, seed+3200, cfg, "C", "test", TRAIN_STYLES),
        "C_style_test": CurriculumDataset(cfg.test_c_episodes, seed+3250, cfg, "C", "test", TEST_STYLES),
    }


def evaluate_with_ablations(agent: Agent, ds: CurriculumDataset, cfg: Config, device: torch.device, memory: Optional[AssociativeMemory]):
    full=evaluate(agent,ds,cfg,device,memory)
    ab={
        "no_lora": evaluate(agent,ds,cfg,device,memory,disable_lora=True),
        "no_external": evaluate(agent,ds,cfg,device,memory,disable_external_memory=True),
        "no_short": evaluate(agent,ds,cfg,device,memory,force_no_short_memory=True),
        "no_lora_no_short": evaluate(agent,ds,cfg,device,memory,disable_lora=True,force_no_short_memory=True),
        "no_lora_no_external_no_short": evaluate(agent,ds,cfg,device,memory,disable_lora=True,disable_external_memory=True,force_no_short_memory=True),
    }
    return full,ab


def run(cfg: Config, out: Path):
    out.mkdir(parents=True,exist_ok=True); (out/"results").mkdir(exist_ok=True); (out/"checkpoints").mkdir(exist_ok=True)
    with open(out/"run_config.json","w") as f: json.dump(asdict(cfg),f,indent=2)
    make_preview(out,cfg)

    device=torch.device("cuda" if cfg.device=="auto" and torch.cuda.is_available() else (cfg.device if cfg.device!="auto" else "cpu"))
    print(f"Device: {device}")
    base_path = out/"checkpoints"/"balanced_base.pt"
    if cfg.resume and base_path.exists():
        base = BasePolicy(cfg).to(device)
        base.load_state_dict(torch.load(base_path, map_location=device))
        base.eval()
        print("Loaded balanced base checkpoint for resume.")
    else:
        base=pretrain_base(cfg,device)
        torch.save(base.state_dict(), base_path)
    total,_=count_params(base); print(f"Base parameters: {total}")

    def load_rows(name: str):
        path = out/"results"/name
        return pd.read_csv(path).to_dict("records") if cfg.resume and path.exists() else []
    metrics_rows=load_rows("metrics.csv"); learning_rows=load_rows("learning_curves.csv"); geometry_rows=load_rows("lora_geometry.csv"); info_rows=load_rows("information_theory.csv")
    start_time=time.time()

    # Base reference accuracy uses O3 mask and zero LoRA; only for legacy diagnostic.
    for seed_index,seed in enumerate(cfg.seeds):
        print(f"\n=== Seed {seed_index+1}/{len(cfg.seeds)}: {seed} ===")
        ds=datasets_for_seed(cfg,seed)
        existing_info={(int(r["seed"]),str(r["phase"])) for r in info_rows if "seed" in r and "phase" in r}
        for phase in ("A","B","C"):
            if (seed, phase) not in existing_info:
                inf=dataset_information(ds[f"{phase}_test"]); info_rows.append({"seed":seed,"phase":phase,**inf})

        shared_state=make_shared_lora_state(base,cfg,device,seed+50000)
        completed={(int(r["seed"]),str(r["observer"])) for r in metrics_rows if str(r.get("phase"))=="C"}
        for observer in OBSERVERS:
            if cfg.resume and (seed, observer) in completed:
                print(f"[resume] skipping completed seed={seed} observer={observer}")
                continue
            # Same LoRA initialization for every observer in this seed.
            agent=Agent(base,observer,cfg,device); load_lora_state(agent,shared_state)
            memory=AssociativeMemory(cfg) if observer in MEMORY_OBSERVERS else None
            global_epoch=0
            dream_total={"dream_candidates":0.0,"dream_valid":0.0,"dream_00":0.0,"dream_01":0.0,"dream_10":0.0,"dream_11":0.0}

            for phase in ("A","B","C"):
                learning,dream_stats=train_phase(agent,observer,phase,ds[f"{phase}_train"],ds[f"{phase}_val"],cfg,device,seed,memory)
                for k in dream_total:
                    dream_total[k] += float(dream_stats.get(k,0.0))
                for row in learning:
                    global_epoch+=1
                    learning_rows.append({"seed":seed,"observer":observer,"global_epoch":global_epoch,**row})

                test_ds=ds[f"{phase}_test"]
                full,ab=evaluate_with_ablations(agent,test_ds,cfg,device,memory)
                internal=internalization_metrics(full,ab,observer)
                geo=lora_geometry(agent,cfg)
                cost=resource_cost(observer,cfg,memory,dream_total,global_epoch)

                # Base/zero-LoRA reference under the SAME observer mask.
                tmp=Agent(base,observer,cfg,device); load_lora_state(tmp,shared_state)
                base_acc=evaluate(tmp,test_ds,cfg,device,None if observer not in MEMORY_OBSERVERS else AssociativeMemory(cfg),disable_lora=True,disable_external_memory=True)["accuracy"]
                info=dataset_information(test_ds)
                legacy=legacy_owge_proxy(observer,cfg,info,full,ab,base_acc,cost)
                ms=memory.summary() if memory is not None else {"memory_records":0.0,"memory_edges_directed":0.0,"memory_dream_records":0.0,"memory_dream_fraction":0.0}

                metrics_rows.append({
                    "seed":seed,"observer":observer,"phase":phase,
                    **full, **internal, **legacy, **ms,
                    "resource_cost":cost,"base_zero_lora_accuracy":base_acc,
                    "dream_candidates":dream_total["dream_candidates"],"dream_valid":dream_total["dream_valid"],
                    "dream_00":dream_total["dream_00"],"dream_01":dream_total["dream_01"],
                    "dream_10":dream_total["dream_10"],"dream_11":dream_total["dream_11"],
                    "dream_validation_rate":dream_total["dream_valid"]/max(dream_total["dream_candidates"],1.0),
                })
                geometry_rows.append({"seed":seed,"observer":observer,"phase":phase,**geo})

            # Secondary external-validity probe: same compositional rule and
            # combinations, but with held-out visual styles. This is NOT the primary
            # hypothesis because it mixes cognition with raw perceptual OOD shift.
            style_eval = evaluate(agent, ds["C_style_test"], cfg, device, memory)
            metrics_rows.append({
                "seed":seed,"observer":observer,"phase":"C_style_shift", **style_eval,
                "I_param":np.nan,"I_external_memory":np.nan,"I_short_term":np.nan,"I_durable_joint":np.nan,
                "durable_no_short_accuracy":np.nan,"stripped_no_durable_accuracy":np.nan,
                "C_att":np.nan,"D_cue":np.nan,"N_sup":np.nan,"A_util":np.nan,"T_gain":np.nan,
                "I_ret_legacy":np.nan,"R_ret_legacy":np.nan,"U_adapt":np.nan,"Q_proc_legacy":np.nan,"P_chain_legacy":np.nan,"OWGE_legacy":np.nan,
                "memory_records":float(len(memory.records)) if memory is not None else 0.0,
                "memory_edges_directed":float(sum(len(v) for v in memory.adj.values())) if memory is not None else 0.0,
                "memory_dream_records":float(sum(r.is_dream for r in memory.records)) if memory is not None else 0.0,
                "memory_dream_fraction":float(np.mean([r.is_dream for r in memory.records])) if memory is not None and len(memory.records) else 0.0,
                "resource_cost":resource_cost(observer,cfg,memory,dream_total,global_epoch),
                "base_zero_lora_accuracy":np.nan,"dream_candidates":dream_total["dream_candidates"],"dream_valid":dream_total["dream_valid"],
                "dream_00":dream_total["dream_00"],"dream_01":dream_total["dream_01"],
                "dream_10":dream_total["dream_10"],"dream_11":dream_total["dream_11"],
                "dream_validation_rate":dream_total["dream_valid"]/max(dream_total["dream_candidates"],1.0),
            })

            # Retention probes after Phase C using final parameters/memory.
            for prior_phase in ("A","B"):
                final=evaluate(agent,ds[f"{prior_phase}_test"],cfg,device,memory)
                metrics_rows.append({
                    "seed":seed,"observer":observer,"phase":f"{prior_phase}_after_C", **final,
                    "I_param":np.nan,"I_external_memory":np.nan,"I_short_term":np.nan,"I_durable_joint":np.nan,
                    "durable_no_short_accuracy":np.nan,"stripped_no_durable_accuracy":np.nan,
                    "C_att":np.nan,"D_cue":np.nan,"N_sup":np.nan,"A_util":np.nan,"T_gain":np.nan,
                    "I_ret_legacy":np.nan,"R_ret_legacy":np.nan,"U_adapt":np.nan,"Q_proc_legacy":np.nan,"P_chain_legacy":np.nan,"OWGE_legacy":np.nan,
                    "memory_records":float(len(memory.records)) if memory is not None else 0.0,
                    "memory_edges_directed":float(sum(len(v) for v in memory.adj.values())) if memory is not None else 0.0,
                    "memory_dream_records":float(sum(r.is_dream for r in memory.records)) if memory is not None else 0.0,
                    "memory_dream_fraction":float(np.mean([r.is_dream for r in memory.records])) if memory is not None and len(memory.records) else 0.0,
                    "resource_cost":resource_cost(observer,cfg,memory,dream_total,global_epoch),
                    "base_zero_lora_accuracy":np.nan,"dream_candidates":dream_total["dream_candidates"],"dream_valid":dream_total["dream_valid"],
                    "dream_00":dream_total["dream_00"],"dream_01":dream_total["dream_01"],
                    "dream_10":dream_total["dream_10"],"dream_11":dream_total["dream_11"],
                    "dream_validation_rate":dream_total["dream_valid"]/max(dream_total["dream_candidates"],1.0),
                })

            if cfg.save_checkpoints:
                torch.save({
                    "seed":seed,"observer":observer,
                    "embed_adapter":agent.embed_adapter.state_dict(),"policy_delta":agent.policy_delta.state_dict(),
                    "memory_summary":memory.summary() if memory is not None else {},
                }, out/"checkpoints"/f"seed_{seed}_{observer}.pt")
            # Save after each completed observer so a long laptop run can resume without
            # repeating already-finished observers.
            pd.DataFrame(metrics_rows).to_csv(out/"results"/"metrics.csv",index=False)
            pd.DataFrame(learning_rows).to_csv(out/"results"/"learning_curves.csv",index=False)
            pd.DataFrame(geometry_rows).to_csv(out/"results"/"lora_geometry.csv",index=False)
            pd.DataFrame(info_rows).to_csv(out/"results"/"information_theory.csv",index=False)

        # Incremental save after every seed to protect long laptop runs.
        pd.DataFrame(metrics_rows).to_csv(out/"results"/"metrics.csv",index=False)
        pd.DataFrame(learning_rows).to_csv(out/"results"/"learning_curves.csv",index=False)
        pd.DataFrame(geometry_rows).to_csv(out/"results"/"lora_geometry.csv",index=False)
        pd.DataFrame(info_rows).to_csv(out/"results"/"information_theory.csv",index=False)
        print(f"Saved incremental results after seed {seed}.")

    metrics=pd.DataFrame(metrics_rows); learning=pd.DataFrame(learning_rows); geometry=pd.DataFrame(geometry_rows); info=pd.DataFrame(info_rows)
    hypotheses=build_hypotheses(metrics)
    hypotheses.to_csv(out/"results"/"hypothesis_tests.csv",index=False)
    metrics.to_csv(out/"results"/"metrics.csv",index=False); learning.to_csv(out/"results"/"learning_curves.csv",index=False)
    geometry.to_csv(out/"results"/"lora_geometry.csv",index=False); info.to_csv(out/"results"/"information_theory.csv",index=False)

    summary=metrics[metrics.phase.isin(["A","B","C"])].groupby(["phase","observer"]).agg(
        accuracy_mean=("accuracy","mean"),accuracy_sd=("accuracy","std"),
        balanced_accuracy_mean=("balanced_accuracy","mean"),macro_f1_mean=("macro_f1","mean"),reward_mean=("reward","mean"),
        I_durable_mean=("I_durable_joint","mean"),retrieval_precision=("retrieval_label_precision","mean"),
        resource_cost=("resource_cost","mean"),OWGE_legacy=("OWGE_legacy","mean")
    ).reset_index()
    summary.to_csv(out/"results"/"observer_summary.csv",index=False)
    make_plots(out,metrics,learning,hypotheses,geometry)

    with open(out/"results"/"SUMMARY.txt","w") as f:
        f.write("OWGE Experiment 3: Mechanism-Adequacy Confirmatory Test\n")
        f.write("=====================================================\n\n")
        f.write("PRIMARY NULL H0: mean Phase-C balanced accuracy(O4D) <= mean Phase-C balanced accuracy(O1).\n")
        f.write("PRIMARY EXPECTED HA: mean Phase-C balanced accuracy(O4D) > mean Phase-C balanced accuracy(O1).\n")
        f.write("Primary endpoint: held-out Phase-C compositional-transfer balanced accuracy.\n\n")
        f.write(summary.to_string(index=False)); f.write("\n\nHypothesis tests:\n")
        f.write(hypotheses.to_string(index=False)); f.write("\n\n")
        f.write(f"Elapsed seconds: {time.time()-start_time:.1f}\n")
        f.write("Important: p<0.05 rejects the stated null; p>=0.05 means fail to reject, not proof of H0.\n")

    print("\n=== Final hypothesis tests ===")
    print(hypotheses[["hypothesis","n_pairs","mean_a","mean_b","mean_difference","ci95_low","ci95_high","one_sided_p","reject_H0_0.05"]].to_string(index=False))
    print(f"\nResults written to: {out}")


def parse_args():
    p=argparse.ArgumentParser(description="OWGE Experiment 3: associative dream consolidation")
    p.add_argument("--preset",choices=["smoke","engineering","confirmatory"],default="confirmatory")
    p.add_argument("--output",default="owge_dream_confirmatory")
    p.add_argument("--device",default="auto",choices=["auto","cpu","cuda"])
    p.add_argument("--rho",type=float,default=None,help="Fixed peripheral reserve. Confirmatory default is 0.50.")
    p.add_argument("--seeds",type=int,default=None,help="Optional truncate seed count for debugging only; do not use for confirmatory inference.")
    p.add_argument("--resume",action="store_true",help="Resume a long run from completed observer checkpoints/CSV rows.")
    return p.parse_args()


def main():
    args=parse_args(); cfg=apply_preset(Config(),args.preset); cfg.output=args.output; cfg.device=args.device
    if args.rho is not None: cfg.rho_peripheral=float(args.rho)
    if args.seeds is not None: cfg.seeds=cfg.seeds[:int(args.seeds)]
    cfg.resume = bool(args.resume)
    out=Path(args.output)
    run(cfg,out)


if __name__=="__main__":
    main()
