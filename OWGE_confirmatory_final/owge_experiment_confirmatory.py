#!/usr/bin/env python3
"""
OWGE controlled validation harness
=================================
Laptop-first synthetic experiment for testing hypotheses from the published
Observation-to-Weighted Generalization Efficiency (OWGE) conceptual paper.

The program intentionally separates roles:
  * MDP/POMDP: controls whether useful state is fully or partially observable.
  * Information theory: measures/controls how much predictive information the
    central/peripheral cues contain about the hidden outcome.
  * O1..O5: fixed observation/memory policies under matched data and budgets.
  * Tiny neural base model: identical frozen starting checkpoint for all observers.
  * LoRA: low-rank long-term adaptation on top of the frozen base model.
  * REINFORCE: reward-based adaptation in the POMDP.
  * Statistics: paired tests across matched random seeds.

Default data are synthetic images because causal relevance is then known exactly,
pretraining leakage is impossible, and train/validation/test episodes can be made
strictly disjoint. Real images can be added later for external validity.

Run examples:
  python owge_experiment.py --preset smoke
  python owge_experiment.py --preset laptop
  python owge_experiment.py --preset paper --device cpu

Outputs are written under --output (default: ./owge_run):
  plots/                 >= 12 PNG plots
  results/metrics.csv
  results/learning_curves.csv
  results/hypothesis_tests.csv
  results/information_theory.csv
  results/lora_geometry.csv
  data_preview/          sample synthetic frames and test manifest
  run_config.json

This is research code, not a claim that the operationalization below is the only
valid implementation of OWGE. The sister paper should report every operational
choice explicitly and should revise/reject the metric if it fails predictive tests.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import statistics
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
from torch.distributions import Categorical
from torch.utils.data import DataLoader, Dataset

try:
    from scipy import stats
except Exception as exc:  # pragma: no cover
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
class ExperimentConfig:
    # image / POMDP
    image_size: int = 48
    episode_len: int = 6
    cue_step: int = 1
    central_accuracy: float = 0.65
    peripheral_accuracy: float = 0.82
    num_distractors: int = 5
    fully_observable: bool = False
    goal_switch_fraction: float = 0.0  # optional later-goal/billboard extension

    # observer policy
    rho_peripheral: float = 0.35
    weak_billboard_weight: float = 0.12
    distractor_weight: float = 0.08

    # base model
    embed_dim: int = 32
    hidden_dim: int = 32
    lora_rank: int = 4
    lora_alpha: float = 8.0

    # base pretraining
    base_pretrain_episodes: int = 1200
    base_pretrain_epochs: int = 5
    base_lr: float = 3e-3

    # observer adaptation
    train_episodes: int = 600
    val_episodes: int = 200
    test_episodes: int = 400
    train_epochs: int = 8
    batch_size: int = 32
    lora_lr: float = 5e-3
    entropy_bonus: float = 0.01
    grad_clip: float = 1.0

    # O4/O5 validated replay
    replay_buffer_size: int = 256
    replay_batches_per_epoch: int = 3
    replay_batch_size: int = 24
    replay_min_delta: float = 0.0

    # repeated matched trials
    seeds: List[int] = field(default_factory=lambda: [11, 22, 33, 44, 55])

    # light paper diagnostics/sweeps
    mi_sweep_levels: List[float] = field(default_factory=lambda: [0.50, 0.65, 0.80, 0.95])
    # Peripheral-reserve sweep. rho is NOT assumed correct a priori; it is treated as
    # an experimental variable and must be reported across this grid.
    rho_sweep_levels: List[float] = field(default_factory=lambda: [0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00])
    rho_sweep_seeds: List[int] = field(default_factory=lambda: [303, 404])
    distractor_sweep: List[int] = field(default_factory=lambda: [0, 2, 5, 10, 20])
    sweep_seeds: List[int] = field(default_factory=lambda: [101, 202])
    sweep_train_epochs: int = 4
    sweep_train_episodes: int = 300
    reversal_epochs: int = 4
    reversal_episodes: int = 300
    mdp_control_epochs: int = 3
    mdp_control_episodes: int = 240

    # OWGE experimental weights (declared before seeing final results)
    owge_wc: float = 1.0
    owge_wd: float = 1.0
    owge_wn: float = 0.7
    owge_wa: float = 1.0
    owge_wr: float = 0.8
    owge_wt: float = 1.0
    owge_lambda_cost: float = 0.7
    owge_eta_false_assoc: float = 0.5

    # modes
    preset: str = "laptop"
    device: str = "cpu"
    num_workers: int = 0


def apply_preset(cfg: ExperimentConfig, preset: str) -> ExperimentConfig:
    cfg.preset = preset
    if preset == "smoke":
        cfg.base_pretrain_episodes = 240
        cfg.base_pretrain_epochs = 2
        cfg.train_episodes = 120
        cfg.val_episodes = 60
        cfg.test_episodes = 100
        cfg.train_epochs = 2
        cfg.batch_size = 24
        cfg.seeds = [11, 22]
        cfg.mi_sweep_levels = [0.50, 0.80, 0.95]
        cfg.rho_sweep_levels = [0.00, 0.35, 0.75, 1.00]
        cfg.rho_sweep_seeds = [303]
        cfg.distractor_sweep = [0, 5, 15]
        cfg.sweep_seeds = [101]
        cfg.sweep_train_epochs = 1
        cfg.sweep_train_episodes = 100
        cfg.reversal_epochs = 1
        cfg.reversal_episodes = 100
        cfg.mdp_control_epochs = 1
        cfg.mdp_control_episodes = 80
    elif preset == "laptop":
        pass
    elif preset == "paper":
        cfg.base_pretrain_episodes = 4000
        cfg.base_pretrain_epochs = 10
        cfg.train_episodes = 1500
        cfg.val_episodes = 400
        cfg.test_episodes = 800
        cfg.train_epochs = 15
        cfg.batch_size = 48
        # Confirmatory seeds are intentionally disjoint from all exploratory pilot
        # seeds (main: 11,22,33,44,55; MI: 101,202; rho: 303,404).
        # Forty paired seeds were fixed before the confirmatory run based on the
        # pilot effect size only for power planning (not significance hunting).
        cfg.seeds = [2001 + i * 37 for i in range(40)]
        cfg.mi_sweep_levels = [0.50, 0.60, 0.70, 0.80, 0.90, 0.98]
        cfg.rho_sweep_levels = [0.00, 0.05, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00]
        cfg.rho_sweep_seeds = [7001, 7109, 7211, 7301, 7409]
        cfg.distractor_sweep = [0, 2, 5, 10, 20, 35]
        cfg.sweep_seeds = [8001, 8109, 8207, 8303, 8419]
        cfg.sweep_train_epochs = 7
        cfg.sweep_train_episodes = 700
        cfg.reversal_epochs = 7
        cfg.reversal_episodes = 700
        cfg.mdp_control_epochs = 7
        cfg.mdp_control_episodes = 600
    else:
        raise ValueError(f"Unknown preset: {preset}")
    return cfg


# -----------------------------------------------------------------------------
# User-defined scene families and synthetic POMDP
# -----------------------------------------------------------------------------

# The latent causal structure is identical across families. Surface geometry and
# palette change, giving a controlled transfer set without training-image overlap.
SCENES: Dict[str, Dict[str, object]] = {
    "crossing": {
        "bg": (235, 235, 235), "central_shape": "circle", "periph_shape": "triangle",
        "true": (235, 70, 70), "false": (70, 110, 220), "accent": (235, 185, 35),
    },
    "warehouse": {
        "bg": (225, 235, 228), "central_shape": "square", "periph_shape": "circle",
        "true": (210, 85, 65), "false": (55, 135, 185), "accent": (235, 160, 45),
    },
    "factory": {
        "bg": (236, 229, 220), "central_shape": "diamond", "periph_shape": "square",
        "true": (190, 70, 55), "false": (65, 120, 190), "accent": (225, 175, 30),
    },
    "harbor": {
        "bg": (220, 233, 240), "central_shape": "triangle", "periph_shape": "diamond",
        "true": (220, 75, 60), "false": (60, 130, 205), "accent": (245, 175, 40),
    },
    # base-only visual family; not used for final train/test comparisons
    "base": {
        "bg": (232, 232, 226), "central_shape": "square", "periph_shape": "triangle",
        "true": (225, 85, 75), "false": (75, 125, 205), "accent": (225, 180, 55),
    },
}

TRAIN_STYLES = ("crossing", "warehouse")
VAL_STYLES = ("crossing", "warehouse")
TEST_STYLES = ("factory", "harbor")
BASE_STYLES = ("base",)


@dataclass
class EpisodeMeta:
    hazard: int
    central_bits: List[int]
    peripheral_bit: int
    ticket_bit: int
    style: str
    episode_seed: int
    distractor_seed: int
    fully_observable: bool
    reversed_peripheral: bool


def noisy_binary(rng: np.random.Generator, truth: int, accuracy: float) -> int:
    return int(truth if rng.random() < accuracy else 1 - truth)


def make_episode_meta(
    seed: int,
    cfg: ExperimentConfig,
    styles: Sequence[str],
    *,
    peripheral_accuracy: Optional[float] = None,
    num_distractors: Optional[int] = None,
    fully_observable: Optional[bool] = None,
    reversed_peripheral: bool = False,
    distractor_variant: int = 0,
) -> Tuple[EpisodeMeta, int]:
    rng = np.random.default_rng(seed)
    hazard = int(rng.integers(0, 2))
    style = str(styles[int(rng.integers(0, len(styles)))])
    full = cfg.fully_observable if fully_observable is None else fully_observable
    cacc = 1.0 if full else cfg.central_accuracy
    pacc = cfg.peripheral_accuracy if peripheral_accuracy is None else peripheral_accuracy

    central_bits = [noisy_binary(rng, hazard, cacc) for _ in range(cfg.episode_len)]
    p_truth = 1 - hazard if reversed_peripheral else hazard
    peripheral_bit = noisy_binary(rng, p_truth, pacc)
    ticket_bit = int(rng.integers(0, 2))
    distractor_seed = int(seed * 1009 + 17 + distractor_variant * 7919)
    meta = EpisodeMeta(
        hazard=hazard,
        central_bits=central_bits,
        peripheral_bit=peripheral_bit,
        ticket_bit=ticket_bit,
        style=style,
        episode_seed=seed,
        distractor_seed=distractor_seed,
        fully_observable=full,
        reversed_peripheral=reversed_peripheral,
    )
    return meta, (cfg.num_distractors if num_distractors is None else num_distractors)


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


def semantic_boxes(size: int) -> Dict[str, Tuple[int, int, int, int]]:
    return {
        "central": (size // 2 - 10, size // 2 - 10, size // 2 + 10, size // 2 + 10),
        "peripheral": (3, 3, 15, 15),
        "billboard": (size - 17, 6, size - 3, 19),
        "state_beacon": (size // 2 - 4, size - 10, size // 2 + 4, size - 3),
    }


def render_frame(
    meta: EpisodeMeta,
    t: int,
    cfg: ExperimentConfig,
    num_distractors: int,
    *,
    distractor_variant: int = 0,
) -> np.ndarray:
    size = cfg.image_size
    style = SCENES[meta.style]
    img = Image.new("RGB", (size, size), style["bg"])
    draw = ImageDraw.Draw(img)
    boxes = semantic_boxes(size)

    # central cue is present every frame, but only probabilistically reports hidden state in POMDP mode
    cbit = meta.central_bits[t]
    cfill = style["true"] if cbit else style["false"]
    draw_shape(draw, str(style["central_shape"]), boxes["central"], cfill)

    # delayed peripheral cue appears only early in the episode
    if t == cfg.cue_step:
        pfill = style["accent"] if meta.peripheral_bit else (90, 90, 90)
        draw_shape(draw, str(style["periph_shape"]), boxes["peripheral"], pfill)

    # billboard is unrelated to hazard but can be queried in a later-goal extension
    if t == max(0, cfg.cue_step - 1):
        bfill = (155, 80, 185) if meta.ticket_bit else (70, 175, 175)
        draw.rectangle(boxes["billboard"], fill=bfill)
        # two stripes make the visual token easier for a tiny model than text OCR
        x0, y0, x1, y1 = boxes["billboard"]
        if meta.ticket_bit:
            draw.line((x0 + 2, y0 + 3, x1 - 2, y0 + 3), fill=(245, 245, 245), width=2)
            draw.line((x0 + 2, y1 - 3, x1 - 2, y1 - 3), fill=(245, 245, 245), width=2)

    # Fully-observable MDP control: an explicit state beacon makes s_t observable.
    if meta.fully_observable:
        beacon = (250, 30, 30) if meta.hazard else (30, 210, 80)
        draw.rectangle(boxes["state_beacon"], fill=beacon)

    # causal-free distractors; positions/colors depend only on distractor seed/variant, never on hazard
    drng = np.random.default_rng(meta.distractor_seed + t * 101 + distractor_variant * 7919)
    forbidden = list(boxes.values())
    for _ in range(num_distractors):
        for _attempt in range(30):
            x = int(drng.integers(1, size - 5))
            y = int(drng.integers(1, size - 5))
            box = (x, y, x + 4, y + 4)
            overlap = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3]) for b in forbidden)
            if not overlap:
                break
        color = tuple(int(v) for v in drng.integers(35, 220, size=3))
        if drng.random() < 0.5:
            draw.ellipse(box, fill=color)
        else:
            draw.rectangle(box, fill=color)

    return np.asarray(img, dtype=np.uint8)


def render_episode(
    meta: EpisodeMeta,
    cfg: ExperimentConfig,
    num_distractors: int,
    *,
    distractor_variant: int = 0,
) -> np.ndarray:
    return np.stack(
        [render_frame(meta, t, cfg, num_distractors, distractor_variant=distractor_variant)
         for t in range(cfg.episode_len)],
        axis=0,
    )


class SyntheticEpisodeDataset(Dataset):
    def __init__(
        self,
        n: int,
        seed: int,
        cfg: ExperimentConfig,
        styles: Sequence[str],
        *,
        peripheral_accuracy: Optional[float] = None,
        num_distractors: Optional[int] = None,
        fully_observable: Optional[bool] = None,
        reversed_peripheral: bool = False,
        distractor_variant: int = 0,
    ) -> None:
        self.n = int(n)
        self.seed = int(seed)
        self.cfg = cfg
        self.styles = tuple(styles)
        self.peripheral_accuracy = peripheral_accuracy
        self.num_distractors = num_distractors
        self.fully_observable = fully_observable
        self.reversed_peripheral = reversed_peripheral
        self.distractor_variant = distractor_variant

    def __len__(self) -> int:
        return self.n

    def episode_seed(self, idx: int) -> int:
        return self.seed + idx * 7919

    def get_meta(self, idx: int) -> Tuple[EpisodeMeta, int]:
        return make_episode_meta(
            self.episode_seed(idx), self.cfg, self.styles,
            peripheral_accuracy=self.peripheral_accuracy,
            num_distractors=self.num_distractors,
            fully_observable=self.fully_observable,
            reversed_peripheral=self.reversed_peripheral,
            distractor_variant=self.distractor_variant,
        )

    def __getitem__(self, idx: int):
        meta, nd = self.get_meta(idx)
        frames = render_episode(meta, self.cfg, nd, distractor_variant=self.distractor_variant)
        # T,C,H,W float32 in [0,1]
        x = torch.from_numpy(frames.copy()).permute(0, 3, 1, 2).float() / 255.0
        y = torch.tensor(meta.hazard, dtype=torch.long)
        # Metadata needed for info/diagnostics. Keep collatable primitive types.
        info = {
            "central_first": int(meta.central_bits[0]),
            "central_last": int(meta.central_bits[-1]),
            "peripheral": int(meta.peripheral_bit),
            "ticket": int(meta.ticket_bit),
            "style": meta.style,
            "seed": int(meta.episode_seed),
            "index": int(idx),
        }
        return x, y, info


# -----------------------------------------------------------------------------
# Information theory
# -----------------------------------------------------------------------------


def entropy_discrete(x: Sequence[int]) -> float:
    arr = np.asarray(x, dtype=int)
    if arr.size == 0:
        return 0.0
    vals, counts = np.unique(arr, return_counts=True)
    p = counts.astype(float) / counts.sum()
    return float(-(p * np.log2(np.clip(p, 1e-12, 1.0))).sum())


def mutual_information_discrete(x: Sequence[int], y: Sequence[int]) -> float:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    if len(x) != len(y) or len(x) == 0:
        return 0.0
    mi = 0.0
    for xv in np.unique(x):
        for yv in np.unique(y):
            pxy = np.mean((x == xv) & (y == yv))
            if pxy <= 0:
                continue
            px = np.mean(x == xv)
            py = np.mean(y == yv)
            mi += pxy * math.log2(pxy / max(px * py, 1e-12))
    return float(mi)


def conditional_mutual_information(x: Sequence[int], y: Sequence[int], z: Sequence[int]) -> float:
    x = np.asarray(x, dtype=int)
    y = np.asarray(y, dtype=int)
    z = np.asarray(z, dtype=int)
    total = 0.0
    for zv in np.unique(z):
        mask = z == zv
        pz = float(mask.mean())
        if mask.sum() > 1:
            total += pz * mutual_information_discrete(x[mask], y[mask])
    return float(total)


def dataset_information(dataset: SyntheticEpisodeDataset) -> Dict[str, float]:
    hazards, central, periph, tickets = [], [], [], []
    # Metadata only; no image rendering needed.
    for i in range(len(dataset)):
        meta, _ = dataset.get_meta(i)
        hazards.append(meta.hazard)
        central.append(meta.central_bits[-1])
        periph.append(meta.peripheral_bit)
        tickets.append(meta.ticket_bit)
    return {
        "H_hazard_bits": entropy_discrete(hazards),
        "I_central_hazard_bits": mutual_information_discrete(central, hazards),
        "I_peripheral_hazard_bits": mutual_information_discrete(periph, hazards),
        "I_peripheral_hazard_given_central_bits": conditional_mutual_information(periph, hazards, central),
        "I_ticket_hazard_bits": mutual_information_discrete(tickets, hazards),
    }


# -----------------------------------------------------------------------------
# Observer policies
# -----------------------------------------------------------------------------

OBSERVERS = ("O1", "O2", "O3", "O4", "O5R")


def observer_region_weights(observer: str, cfg: ExperimentConfig) -> Dict[str, float]:
    if observer == "O1":
        return {"central": 1.0, "peripheral": 0.0, "billboard": 0.0, "distractor": 0.0}
    if observer == "O2":
        return {"central": 1.0, "peripheral": 1.0, "billboard": 1.0, "distractor": 1.0}
    if observer in ("O3", "O4", "O5R"):
        return {
            "central": 1.0,
            "peripheral": float(cfg.rho_peripheral),
            "billboard": float(cfg.weak_billboard_weight),
            "distractor": float(cfg.distractor_weight),
        }
    raise ValueError(observer)


def observer_attention_distribution(observer: str, cfg: ExperimentConfig, num_distractors: int) -> np.ndarray:
    """Semantic attention over [central, peripheral, billboard, all distractors]."""
    w = observer_region_weights(observer, cfg)
    raw = np.array([
        w["central"],
        w["peripheral"],
        w["billboard"],
        w["distractor"] * max(1, num_distractors),
    ], dtype=float)
    if raw.sum() <= 0:
        raw[0] = 1.0
    return raw / raw.sum()


def make_observer_mask(observer: str, cfg: ExperimentConfig, device: torch.device) -> torch.Tensor:
    """Return 1,C,H,W mask. Background is weakly retained; semantic ROIs get policy weights."""
    s = cfg.image_size
    weights = observer_region_weights(observer, cfg)
    # background low-pass retention keeps basic geometry while suppressing irrelevant pixels
    if observer == "O2":
        mask = torch.ones((1, 1, s, s), device=device)
    else:
        bg = 0.05 if observer == "O1" else max(0.03, cfg.distractor_weight)
        mask = torch.full((1, 1, s, s), float(bg), device=device)
    boxes = semantic_boxes(s)
    for name in ("central", "peripheral", "billboard"):
        x0, y0, x1, y1 = boxes[name]
        mask[:, :, y0:y1+1, x0:x1+1] = float(weights[name])
    # MDP state beacon should be visible to every observer when present.
    x0, y0, x1, y1 = boxes["state_beacon"]
    mask[:, :, y0:y1+1, x0:x1+1] = 1.0
    return mask


# -----------------------------------------------------------------------------
# Tiny base model + LoRA
# -----------------------------------------------------------------------------


class TinyCNNEncoder(nn.Module):
    """Very small image encoder suitable for CPU experiments."""
    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Linear(24, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x).flatten(1)
        return torch.tanh(self.proj(z))


class BaseSequencePolicy(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.encoder = TinyCNNEncoder(cfg.embed_dim)
        self.gru = nn.GRUCell(cfg.embed_dim, cfg.hidden_dim)
        self.policy = nn.Linear(cfg.hidden_dim, 2)

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        # frames B,T,C,H,W -> B,T,E
        b, t, c, h, w = frames.shape
        emb = self.encoder(frames.reshape(b * t, c, h, w))
        return emb.reshape(b, t, -1)

    def sequence_hidden(self, emb: torch.Tensor, reset_each_step: bool = False) -> torch.Tensor:
        b, t, _ = emb.shape
        h = torch.zeros((b, self.gru.hidden_size), device=emb.device, dtype=emb.dtype)
        for step in range(t):
            if reset_each_step and step > 0:
                h = torch.zeros_like(h)
            h = self.gru(emb[:, step], h)
        return h

    def forward(self, frames: torch.Tensor, reset_each_step: bool = False) -> torch.Tensor:
        emb = self.encode_frames(frames)
        h = self.sequence_hidden(emb, reset_each_step=reset_each_step)
        return self.policy(h)


class LoRAResidual(nn.Module):
    """Low-rank residual update y = x + scale * B(A(x)); B starts at zero."""
    def __init__(self, dim: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        self.A = nn.Linear(dim, rank, bias=False)
        self.B = nn.Linear(rank, dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.scale * self.B(self.A(x))

    def delta_weight(self) -> torch.Tensor:
        return self.scale * (self.B.weight @ self.A.weight)


class LoRAPolicyDelta(nn.Module):
    """LoRA update for a frozen hidden->action policy matrix."""
    def __init__(self, hidden_dim: int, out_dim: int, rank: int, alpha: float):
        super().__init__()
        self.rank = rank
        self.scale = alpha / max(rank, 1)
        self.A = nn.Linear(hidden_dim, rank, bias=False)
        self.B = nn.Linear(rank, out_dim, bias=False)
        nn.init.kaiming_uniform_(self.A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.B.weight)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.scale * self.B(self.A(h))

    def delta_weight(self) -> torch.Tensor:
        return self.scale * (self.B.weight @ self.A.weight)


class ObserverAgent(nn.Module):
    def __init__(self, base: BaseSequencePolicy, observer: str, cfg: ExperimentConfig, device: torch.device):
        super().__init__()
        self.observer = observer
        self.cfg = cfg
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad = False
        self.embed_adapter = LoRAResidual(cfg.embed_dim, cfg.lora_rank, cfg.lora_alpha)
        self.policy_delta = LoRAPolicyDelta(cfg.hidden_dim, 2, cfg.lora_rank, cfg.lora_alpha)
        self.register_buffer("obs_mask", make_observer_mask(observer, cfg, device))
        self.to(device)

    @property
    def retrieval_enabled(self) -> bool:
        return self.observer != "O5R"

    @property
    def uses_replay(self) -> bool:
        return self.observer in ("O4", "O5R")

    def trainable_parameters(self) -> Iterable[nn.Parameter]:
        yield from self.embed_adapter.parameters()
        yield from self.policy_delta.parameters()

    def apply_observer(self, frames: torch.Tensor) -> torch.Tensor:
        # Attenuate low-priority regions toward neutral gray instead of setting all to zero.
        # This operationalizes graded access to visual evidence under a finite-processing policy.
        mask = self.obs_mask.unsqueeze(1)  # 1,1,1,H,W
        neutral = torch.full_like(frames, 0.5)
        return neutral + mask * (frames - neutral)

    def forward(
        self,
        frames: torch.Tensor,
        force_no_memory: bool = False,
        disable_lora: bool = False,
    ) -> torch.Tensor:
        frames = self.apply_observer(frames)
        with torch.no_grad():
            emb = self.base.encode_frames(frames)
        # LoRA is the experimental proxy for durable internalization across episodes.
        # disable_lora=True is a causal ablation: same observer/base/GRU, but remove
        # only the learned durable adapter update.
        if not disable_lora:
            emb = self.embed_adapter(emb)
        reset = force_no_memory or (not self.retrieval_enabled)
        # frozen GRU still receives gradients wrt adapted inputs; its weights stay frozen
        b, t, _ = emb.shape
        h = torch.zeros((b, self.cfg.hidden_dim), device=emb.device, dtype=emb.dtype)
        for step in range(t):
            if reset and step > 0:
                h = torch.zeros_like(h)
            h = self.base.gru(emb[:, step], h)
        logits = self.base.policy(h)
        if not disable_lora:
            logits = logits + self.policy_delta(h)
        return logits

    def lora_vector(self) -> np.ndarray:
        pieces = [
            self.embed_adapter.delta_weight().detach().cpu().flatten(),
            self.policy_delta.delta_weight().detach().cpu().flatten(),
        ]
        return torch.cat(pieces).numpy()

    def lora_norm(self) -> float:
        v = self.lora_vector()
        return float(np.linalg.norm(v))


# -----------------------------------------------------------------------------
# Training / evaluation
# -----------------------------------------------------------------------------


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return int(total), int(trainable)


def make_loader(dataset: Dataset, cfg: ExperimentConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset, batch_size=cfg.batch_size, shuffle=shuffle,
        num_workers=cfg.num_workers, pin_memory=False,
    )


def pretrain_base(cfg: ExperimentConfig, device: torch.device, seed: int = 9001) -> BaseSequencePolicy:
    set_global_seed(seed)
    base = BaseSequencePolicy(cfg).to(device)
    # Base pretraining uses a disjoint visual style and mostly central/fully observable structure.
    pre_cfg = copy.deepcopy(cfg)
    pre_cfg.central_accuracy = 0.95
    pre_cfg.peripheral_accuracy = 0.50
    ds = SyntheticEpisodeDataset(
        cfg.base_pretrain_episodes, seed + 1000, pre_cfg, BASE_STYLES,
        fully_observable=False, peripheral_accuracy=0.50, num_distractors=2,
    )
    loader = make_loader(ds, cfg, shuffle=True)
    opt = torch.optim.Adam(base.parameters(), lr=cfg.base_lr)
    base.train()
    for epoch in range(cfg.base_pretrain_epochs):
        losses = []
        correct = total = 0
        for frames, labels, _ in loader:
            frames, labels = frames.to(device), labels.to(device)
            logits = base(frames)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(base.parameters(), cfg.grad_clip)
            opt.step()
            losses.append(float(loss.item()))
            correct += int((logits.argmax(1) == labels).sum().item())
            total += labels.numel()
        print(f"[base] epoch {epoch+1}/{cfg.base_pretrain_epochs} loss={np.mean(losses):.4f} acc={correct/max(total,1):.3f}")
    return base.eval()


@torch.no_grad()
def evaluate_agent(
    agent: ObserverAgent,
    dataset: Dataset,
    cfg: ExperimentConfig,
    device: torch.device,
    *,
    force_no_memory: bool = False,
    disable_lora: bool = False,
) -> Dict[str, object]:
    loader = make_loader(dataset, cfg, shuffle=False)
    agent.eval()
    all_y, all_pred, all_prob = [], [], []
    for frames, labels, _ in loader:
        frames, labels = frames.to(device), labels.to(device)
        logits = agent(frames, force_no_memory=force_no_memory, disable_lora=disable_lora)
        prob = logits.softmax(1)[:, 1]
        pred = logits.argmax(1)
        all_y.extend(labels.cpu().tolist())
        all_pred.extend(pred.cpu().tolist())
        all_prob.extend(prob.cpu().tolist())
    y = np.asarray(all_y, dtype=int)
    pred = np.asarray(all_pred, dtype=int)
    prob = np.asarray(all_prob, dtype=float)
    acc = float((y == pred).mean()) if len(y) else 0.0
    reward = float(np.where(y == pred, 1.0, -1.0).mean()) if len(y) else 0.0
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return {
        "accuracy": acc,
        "reward": reward,
        "y": y,
        "pred": pred,
        "prob": prob,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


@torch.no_grad()
def evaluate_unadapted_base(
    base: BaseSequencePolicy,
    observer: str,
    dataset: Dataset,
    cfg: ExperimentConfig,
    device: torch.device,
) -> float:
    temp = ObserverAgent(base, observer, cfg, device)
    # LoRA B matrices start at zero => exact base behavior under observer masking.
    return float(evaluate_agent(temp, dataset, cfg, device)["accuracy"])


def reinforce_batch_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    baseline: float,
    entropy_bonus: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dist = Categorical(logits=logits)
    actions = dist.sample()
    rewards = torch.where(actions == labels, torch.ones_like(actions, dtype=torch.float32), -torch.ones_like(actions, dtype=torch.float32))
    advantage = rewards - float(baseline)
    loss = -(dist.log_prob(actions) * advantage.detach()).mean() - entropy_bonus * dist.entropy().mean()
    return loss, rewards.detach(), actions.detach()


def clone_lora_state(agent: ObserverAgent) -> Dict[str, torch.Tensor]:
    state = {}
    for prefix, module in (("embed", agent.embed_adapter), ("policy", agent.policy_delta)):
        for k, v in module.state_dict().items():
            state[f"{prefix}.{k}"] = v.detach().clone()
    return state


def load_lora_state(agent: ObserverAgent, state: Dict[str, torch.Tensor]) -> None:
    e = {k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("embed.")}
    p = {k.split(".", 1)[1]: v for k, v in state.items() if k.startswith("policy.")}
    agent.embed_adapter.load_state_dict(e)
    agent.policy_delta.load_state_dict(p)


def train_observer(
    base: BaseSequencePolicy,
    observer: str,
    train_ds: SyntheticEpisodeDataset,
    val_ds: SyntheticEpisodeDataset,
    cfg: ExperimentConfig,
    device: torch.device,
    seed: int,
    *,
    epochs_override: Optional[int] = None,
) -> Tuple[ObserverAgent, List[Dict[str, float]], Dict[str, float]]:
    observer_seed_offset = {"O1": 101, "O2": 202, "O3": 303, "O4": 404, "O5R": 505}[observer]
    set_global_seed(seed + observer_seed_offset)
    agent = ObserverAgent(base, observer, cfg, device)
    optimizer = torch.optim.Adam(list(agent.trainable_parameters()), lr=cfg.lora_lr)
    loader = make_loader(train_ds, cfg, shuffle=True)
    epochs = cfg.train_epochs if epochs_override is None else epochs_override
    running_baseline = 0.0
    replay_indices: List[Tuple[float, int]] = []
    learning_rows: List[Dict[str, float]] = []
    replay_accepted = 0
    replay_attempted = 0
    replay_gain_total = 0.0

    for epoch in range(epochs):
        agent.train()
        epoch_rewards = []
        sample_cursor = 0
        for frames, labels, info in loader:
            frames, labels = frames.to(device), labels.to(device)
            logits = agent(frames)
            loss, rewards, actions = reinforce_batch_loss(logits, labels, running_baseline, cfg.entropy_bonus)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(agent.trainable_parameters()), cfg.grad_clip)
            optimizer.step()
            batch_mean = float(rewards.mean().item())
            running_baseline = 0.9 * running_baseline + 0.1 * batch_mean
            epoch_rewards.extend(rewards.cpu().tolist())

            # Priority replay buffer: errors and uncertainty, using the actual dataset indices
            # carried through the DataLoader so replay is fully reproducible.
            probs = logits.softmax(1).detach().cpu().numpy()
            lab_np = labels.detach().cpu().numpy()
            uncertainty = 1.0 - np.abs(probs[:, 1] - 0.5) * 2.0
            wrong = (probs.argmax(1) != lab_np).astype(float)
            priorities = wrong + 0.25 * uncertainty
            batch_indices = info["index"].detach().cpu().numpy() if torch.is_tensor(info["index"]) else np.asarray(info["index"])
            for idx_value, pr in zip(batch_indices, priorities):
                replay_indices.append((float(pr), int(idx_value)))
            if len(replay_indices) > cfg.replay_buffer_size * 3:
                replay_indices = sorted(replay_indices, reverse=True)[:cfg.replay_buffer_size]

        # Validated replay: tentative updates are accepted only if validation reward does not degrade.
        val_before = float(evaluate_agent(agent, val_ds, cfg, device)["reward"])
        accepted_this_epoch = 0
        gain_this_epoch = 0.0
        if agent.uses_replay and cfg.replay_batches_per_epoch > 0 and replay_indices:
            replay_attempted += 1
            saved = clone_lora_state(agent)
            top = sorted(replay_indices, reverse=True)[:cfg.replay_buffer_size]
            candidate_ids = [idx for _, idx in top]
            agent.train()
            for _ in range(cfg.replay_batches_per_epoch):
                chosen = random.sample(candidate_ids, k=min(cfg.replay_batch_size, len(candidate_ids)))
                batch = [train_ds[i] for i in chosen]
                frames = torch.stack([b[0] for b in batch]).to(device)
                labels = torch.stack([b[1] for b in batch]).to(device)
                logits = agent(frames)
                loss, _, _ = reinforce_batch_loss(logits, labels, running_baseline, cfg.entropy_bonus)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(list(agent.trainable_parameters()), cfg.grad_clip)
                optimizer.step()
            val_after = float(evaluate_agent(agent, val_ds, cfg, device)["reward"])
            gain_this_epoch = val_after - val_before
            if gain_this_epoch >= cfg.replay_min_delta:
                replay_accepted += 1
                replay_gain_total += gain_this_epoch
                accepted_this_epoch = 1
            else:
                load_lora_state(agent, saved)
                val_after = val_before
        else:
            val_after = val_before

        train_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
        val_acc = float(evaluate_agent(agent, val_ds, cfg, device)["accuracy"])
        learning_rows.append({
            "epoch": epoch + 1,
            "train_reward": train_reward,
            "val_reward": val_after,
            "val_accuracy": val_acc,
            "replay_accepted": accepted_this_epoch,
            "replay_gain": gain_this_epoch,
        })
        print(f"[{observer}] epoch {epoch+1}/{epochs} trainR={train_reward:.3f} valR={val_after:.3f} valAcc={val_acc:.3f} replay={accepted_this_epoch}")

    replay_stats = {
        "replay_attempted": float(replay_attempted),
        "replay_accepted": float(replay_accepted),
        "replay_accept_rate": float(replay_accepted / replay_attempted) if replay_attempted else 0.0,
        "replay_gain_mean": float(replay_gain_total / replay_accepted) if replay_accepted else 0.0,
    }
    return agent.eval(), learning_rows, replay_stats


# -----------------------------------------------------------------------------
# OWGE experimental operationalization
# -----------------------------------------------------------------------------


def clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def attention_metrics(observer: str, cfg: ExperimentConfig, info: Dict[str, float], num_distractors: int) -> Dict[str, float]:
    a = observer_attention_distribution(observer, cfg, num_distractors)
    # Realized relevance proxy comes from causal information in the synthetic generator.
    r = np.array([
        max(info["I_central_hazard_bits"], 1e-8),
        max(info["I_peripheral_hazard_given_central_bits"], 1e-8),
        max(info["I_ticket_hazard_bits"], 0.0),
        0.0,
    ], dtype=float)
    if r.sum() <= 0:
        r[0] = 1.0
    r = r / r.sum()
    c_att = 1.0 - 0.5 * float(np.sum((a - r) ** 2))
    d_cue = float(a[1]) if info["I_peripheral_hazard_given_central_bits"] > 1e-6 else 0.0
    n_sup = 1.0 - float(a[3])
    return {"C_att": clip01(c_att), "D_cue": clip01(d_cue), "N_sup": clip01(n_sup)}


def resource_cost(observer: str, agent: ObserverAgent, cfg: ExperimentConfig, replay_stats: Dict[str, float]) -> float:
    w = observer_region_weights(observer, cfg)
    # Relative sensing/processing cost: semantic access breadth + trainable LoRA + replay.
    visual = np.mean([w["central"], w["peripheral"], w["billboard"], w["distractor"]])
    _, trainable = count_parameters(agent)
    lora_cost = trainable / 1000.0  # normalized thousands of parameters
    replay_cost = replay_stats.get("replay_attempted", 0.0) * cfg.replay_batches_per_epoch / max(cfg.train_epochs, 1)
    raw = 0.55 * visual + 0.15 * lora_cost + 0.30 * replay_cost
    return float(raw)


def compute_owge_metrics(
    observer: str,
    cfg: ExperimentConfig,
    info: Dict[str, float],
    agent: ObserverAgent,
    test_eval: Dict[str, object],
    no_memory_eval: Dict[str, object],
    lora_ablated_eval: Dict[str, object],
    lora_ablated_no_memory_eval: Dict[str, object],
    base_test_acc: float,
    distractor_high_acc: float,
    replay_stats: Dict[str, float],
) -> Dict[str, float]:
    att = attention_metrics(observer, cfg, info, cfg.num_distractors)
    test_acc = float(test_eval["accuracy"])
    no_mem_acc = float(no_memory_eval["accuracy"])
    lora_ablated_acc = float(lora_ablated_eval["accuracy"])
    lora_ablated_no_mem_acc = float(lora_ablated_no_memory_eval["accuracy"])
    # Utility proxies are all observable interventions/ablations.
    a_util = clip01((test_acc - no_mem_acc) / max(test_acc, 1e-8))
    r_gain = clip01(max(0.0, replay_stats.get("replay_gain_mean", 0.0)) / 2.0)  # rewards span [-1,+1]
    t_gain = clip01((test_acc - base_test_acc) / max(1.0 - base_test_acc, 1e-8))
    false_assoc = clip01((test_acc - distractor_high_acc) / max(test_acc, 1e-8))
    c_res = resource_cost(observer, agent, cfg, replay_stats)

    numerator = (
        cfg.owge_wc * att["C_att"] +
        cfg.owge_wd * att["D_cue"] +
        cfg.owge_wn * att["N_sup"] +
        cfg.owge_wa * a_util +
        cfg.owge_wr * r_gain +
        cfg.owge_wt * t_gain
    )
    denom_weight = cfg.owge_wc + cfg.owge_wd + cfg.owge_wn + cfg.owge_wa + cfg.owge_wr + cfg.owge_wt
    q_proc = (numerator / denom_weight) / (1.0 + cfg.owge_lambda_cost * c_res)

    # Bottleneck probes. Internalization is now measured by a direct LoRA ablation
    # rather than only by adapted-vs-base accuracy. The primary I_ret asks whether
    # durable LoRA learning still improves performance when within-episode retrieval
    # is disabled; this better separates durable internalization from GRU memory.
    i_adapter_gain = clip01((test_acc - lora_ablated_acc) / max(1.0 - lora_ablated_acc, 1e-8))
    i_durable = clip01((no_mem_acc - lora_ablated_no_mem_acc) / max(1.0 - lora_ablated_no_mem_acc, 1e-8))
    i_ret = i_durable
    r_ret = clip01(0.5 + max(-0.5, min(0.5, test_acc - no_mem_acc)))
    u_adapt = clip01(test_acc)
    eps = 1e-6
    p_chain = float(((i_ret + eps) * (r_ret + eps) * (u_adapt + eps)) ** (1.0 / 3.0))
    owge_plus = q_proc * p_chain - cfg.owge_eta_false_assoc * false_assoc
    return {
        **att,
        "A_util": a_util,
        "R_gain": r_gain,
        "T_gain": t_gain,
        "C_res": c_res,
        "F_assoc": false_assoc,
        "I_ret": i_ret,
        "I_adapter_gain": i_adapter_gain,
        "I_durable_no_memory_gain": i_durable,
        "lora_ablated_accuracy": lora_ablated_acc,
        "lora_ablated_no_memory_accuracy": lora_ablated_no_mem_acc,
        "R_ret": r_ret,
        "U_adapt": u_adapt,
        "P_chain": p_chain,
        "Q_proc": q_proc,
        "OWGE_plus": float(owge_plus),
    }


# -----------------------------------------------------------------------------
# Statistics
# -----------------------------------------------------------------------------


def cohens_d_paired(x: Sequence[float], y: Sequence[float]) -> float:
    d = np.asarray(x, float) - np.asarray(y, float)
    if len(d) < 2 or np.std(d, ddof=1) < 1e-12:
        return 0.0
    return float(np.mean(d) / np.std(d, ddof=1))


def one_sided_paired_test(x: Sequence[float], y: Sequence[float], alternative: str = "greater") -> Tuple[float, float]:
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    if len(x) != len(y) or len(x) < 2 or stats is None:
        return float("nan"), float("nan")
    t, p2 = stats.ttest_rel(x, y, nan_policy="omit")
    if np.isnan(t):
        return float(t), float(p2)
    if alternative == "greater":
        p1 = p2 / 2.0 if t > 0 else 1.0 - p2 / 2.0
    elif alternative == "less":
        p1 = p2 / 2.0 if t < 0 else 1.0 - p2 / 2.0
    else:
        p1 = p2
    return float(t), float(p1)


def build_hypothesis_tests(metrics: pd.DataFrame, reversal: pd.DataFrame, mi_sweep: pd.DataFrame, mdp_control: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def paired(metric: str, a: str, b: str, alt: str, hypothesis: str, condition: str = "main"):
        sub = metrics if condition == "main" else metrics[metrics["condition"] == condition]
        pa = sub[sub.observer == a].sort_values("seed")
        pb = sub[sub.observer == b].sort_values("seed")
        merged = pa[["seed", metric]].merge(pb[["seed", metric]], on="seed", suffixes=("_a", "_b"))
        t, p = one_sided_paired_test(merged[f"{metric}_a"], merged[f"{metric}_b"], alternative=alt)
        rows.append({
            "hypothesis": hypothesis,
            "metric": metric,
            "comparison": f"{a} {alt} {b}",
            "n_pairs": len(merged),
            "mean_a": merged[f"{metric}_a"].mean() if len(merged) else np.nan,
            "mean_b": merged[f"{metric}_b"].mean() if len(merged) else np.nan,
            "mean_difference": (merged[f"{metric}_a"] - merged[f"{metric}_b"]).mean() if len(merged) else np.nan,
            "paired_cohens_d": cohens_d_paired(merged[f"{metric}_a"], merged[f"{metric}_b"]) if len(merged) else np.nan,
            "t_stat": t,
            "one_sided_p": p,
            "reject_H0_at_0.05": bool(p < 0.05) if np.isfinite(p) else False,
        })

    # PRIMARY confirmatory hypothesis, fixed before the confirmatory run:
    # H0: mean held-out accuracy(O4) <= mean held-out accuracy(O1)
    # HA: mean held-out accuracy(O4) >  mean held-out accuracy(O1)
    paired("accuracy", "O4", "O1", "greater", "PRIMARY: O4 > O1 under delayed partial observability")

    # Secondary/mechanistic hypotheses. These explain where any O4-vs-O1
    # difference comes from; they are not substitutes for the primary test.
    paired("accuracy", "O3", "O1", "greater", "H3a: O3 > O1 when delayed peripheral information is non-zero")
    paired("accuracy", "O3", "O2", "greater", "H3b: O3 > O2 under finite resources and sparse useful context")
    paired("accuracy", "O4", "O3", "greater", "H4: O4 > O3 when validated replay finds stable reusable structure")
    paired("OWGE_plus", "O5R", "O4", "less", "H6: retrieval-disconnected O5R has lower end-to-end OWGE than O4")

    # H5 from reversal recovery data: compare final accuracy after reversal.
    if len(reversal):
        last = reversal.sort_values("reversal_epoch").groupby(["seed", "observer"], as_index=False).tail(1)
        a = last[last.observer == "O4"][["seed", "accuracy"]]
        b = last[last.observer == "O3"][["seed", "accuracy"]]
        m = a.merge(b, on="seed", suffixes=("_o4", "_o3"))
        t, p = one_sided_paired_test(m.accuracy_o4, m.accuracy_o3, alternative="less")
        rows.append({
            "hypothesis": "H5: O4 can recover more slowly than O3 after predictive reversal",
            "metric": "reversal_final_accuracy",
            "comparison": "O4 less O3",
            "n_pairs": len(m),
            "mean_a": m.accuracy_o4.mean() if len(m) else np.nan,
            "mean_b": m.accuracy_o3.mean() if len(m) else np.nan,
            "mean_difference": (m.accuracy_o4 - m.accuracy_o3).mean() if len(m) else np.nan,
            "paired_cohens_d": cohens_d_paired(m.accuracy_o4, m.accuracy_o3) if len(m) else np.nan,
            "t_stat": t, "one_sided_p": p,
            "reject_H0_at_0.05": bool(p < 0.05) if np.isfinite(p) else False,
        })

    # H2: O2 efficiency should decrease with distractor count. Fit simple slope if available in metrics.
    # H1/H2/H3 crossover evidence is additionally visualized in sweeps; report O3-O1 slope vs MI.
    if len(mi_sweep) >= 3:
        piv = mi_sweep.pivot_table(index=["seed", "peripheral_accuracy", "peripheral_cmi"], columns="observer", values="accuracy").reset_index()
        if "O3" in piv and "O1" in piv:
            piv["gap"] = piv["O3"] - piv["O1"]
            if stats is not None and len(piv) >= 3:
                slope, intercept, r, p, stderr = stats.linregress(piv["peripheral_cmi"], piv["gap"])
            else:
                slope = intercept = r = p = stderr = np.nan
            rows.append({
                "hypothesis": "H1/H3 regime test: O3-O1 advantage increases as conditional peripheral information increases",
                "metric": "slope gap vs conditional MI",
                "comparison": "slope > 0",
                "n_pairs": len(piv),
                "mean_a": float(slope), "mean_b": 0.0, "mean_difference": float(slope),
                "paired_cohens_d": np.nan, "t_stat": np.nan, "one_sided_p": float(p / 2.0) if np.isfinite(p) and slope > 0 else float(1 - p / 2.0) if np.isfinite(p) else np.nan,
                "reject_H0_at_0.05": bool((p / 2.0) < 0.05 and slope > 0) if np.isfinite(p) else False,
            })


    # H1 control: the O3-vs-O1 advantage should be larger in the partially observable
    # delayed-cue world than in the fully observable MDP control.
    if len(mdp_control):
        pom = metrics.pivot_table(index="seed", columns="observer", values="accuracy")
        mdp = mdp_control.pivot_table(index="seed", columns="observer", values="accuracy")
        common = pom.index.intersection(mdp.index)
        if len(common) >= 2 and all(o in pom.columns for o in ("O1", "O3")) and all(o in mdp.columns for o in ("O1", "O3")):
            gap_pom = pom.loc[common, "O3"] - pom.loc[common, "O1"]
            gap_mdp = mdp.loc[common, "O3"] - mdp.loc[common, "O1"]
            t, p = one_sided_paired_test(gap_pom, gap_mdp, alternative="greater")
            rows.append({
                "hypothesis": "H1 control: O3-O1 advantage is larger in POMDP than fully observable MDP",
                "metric": "difference_of_observer_gaps",
                "comparison": "(O3-O1)_POMDP greater (O3-O1)_MDP",
                "n_pairs": len(common),
                "mean_a": float(gap_pom.mean()), "mean_b": float(gap_mdp.mean()),
                "mean_difference": float((gap_pom-gap_mdp).mean()),
                "paired_cohens_d": cohens_d_paired(gap_pom, gap_mdp),
                "t_stat": t, "one_sided_p": p,
                "reject_H0_at_0.05": bool(p < 0.05) if np.isfinite(p) else False,
            })

    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plotting (>= 12 separate figures)
# -----------------------------------------------------------------------------


def save_bar(df: pd.DataFrame, value: str, title: str, ylabel: str, path: Path) -> None:
    order = [o for o in OBSERVERS if o in set(df.observer)]
    means = [df[df.observer == o][value].mean() for o in order]
    sems = [df[df.observer == o][value].sem() for o in order]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(order, means, yerr=sems, capsize=4)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Observer")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def generate_plots(
    metrics: pd.DataFrame,
    curves: pd.DataFrame,
    info_df: pd.DataFrame,
    mi_sweep: pd.DataFrame,
    rho_sweep: pd.DataFrame,
    distractor_df: pd.DataFrame,
    reversal_df: pd.DataFrame,
    lora_df: pd.DataFrame,
    mdp_control_df: pd.DataFrame,
    plot_dir: Path,
) -> None:
    plot_dir.mkdir(parents=True, exist_ok=True)

    save_bar(metrics, "accuracy", "Held-out transfer accuracy by observer", "Accuracy", plot_dir / "01_accuracy_by_observer.png")
    save_bar(metrics, "reward", "Held-out POMDP reward by observer", "Mean reward", plot_dir / "02_reward_by_observer.png")
    save_bar(metrics, "OWGE_plus", "Experimental OWGE+ by observer", "OWGE+", plot_dir / "03_owge_by_observer.png")
    save_bar(metrics, "C_att", "Attention calibration by observer", "C_att", plot_dir / "04_attention_calibration.png")
    save_bar(metrics, "D_cue", "Delayed-cue retention by observer", "D_cue", plot_dir / "05_delayed_cue_retention.png")
    save_bar(metrics, "C_res", "Estimated resource cost by observer", "Relative cost", plot_dir / "06_resource_cost.png")
    save_bar(metrics, "lora_norm", "LoRA update magnitude", "Frobenius/vector norm", plot_dir / "07_lora_update_norm.png")

    # 08 learning curves
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for o in OBSERVERS:
        d = curves[curves.observer == o]
        if len(d):
            g = d.groupby("epoch")["val_accuracy"].mean()
            ax.plot(g.index, g.values, marker="o", label=o)
    ax.set_title("Validation learning curves")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "08_learning_curves.png", dpi=170)
    plt.close(fig)

    # 09 OWGE vs transfer
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for o in OBSERVERS:
        d = metrics[metrics.observer == o]
        if len(d):
            ax.scatter(d["OWGE_plus"], d["accuracy"], label=o, s=42)
    if len(metrics) >= 2:
        z = np.polyfit(metrics["OWGE_plus"], metrics["accuracy"], 1)
        xs = np.linspace(metrics["OWGE_plus"].min(), metrics["OWGE_plus"].max(), 50)
        ax.plot(xs, z[0] * xs + z[1], linestyle="--")
    ax.set_title("Does OWGE predict held-out adaptive performance?")
    ax.set_xlabel("OWGE+")
    ax.set_ylabel("Held-out accuracy")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "09_owge_vs_transfer.png", dpi=170)
    plt.close(fig)

    # 10 information structure by peripheral accuracy
    if len(mi_sweep):
        fig, ax = plt.subplots(figsize=(7.0, 4.6))
        mi_line = mi_sweep.groupby("peripheral_accuracy")["peripheral_cmi"].mean()
        ax.plot(mi_line.index, mi_line.values, marker="o")
        ax.set_title("Controlled peripheral information in the POMDP")
        ax.set_xlabel("Peripheral cue accuracy")
        ax.set_ylabel("I(peripheral; hazard | central), bits")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "10_peripheral_information_control.png", dpi=170)
        plt.close(fig)

        # 11 performance vs conditional MI
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for o in sorted(mi_sweep.observer.unique()):
            d = mi_sweep[mi_sweep.observer == o].groupby("peripheral_accuracy").agg({"peripheral_cmi":"mean", "accuracy":"mean"}).reset_index()
            ax.plot(d["peripheral_cmi"], d["accuracy"], marker="o", label=o)
        ax.set_title("Observer performance vs peripheral conditional information")
        ax.set_xlabel("I(peripheral; hazard | central), bits")
        ax.set_ylabel("Held-out accuracy")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "11_accuracy_vs_peripheral_information.png", dpi=170)
        plt.close(fig)

        # 12 O3-O1 gap vs MI
        piv = mi_sweep.pivot_table(index=["seed", "peripheral_accuracy", "peripheral_cmi"], columns="observer", values="accuracy").reset_index()
        if "O3" in piv and "O1" in piv:
            piv["gap"] = piv["O3"] - piv["O1"]
            g = piv.groupby("peripheral_accuracy").agg({"peripheral_cmi":"mean", "gap":"mean"}).reset_index()
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            ax.axhline(0.0, linewidth=1)
            ax.plot(g["peripheral_cmi"], g["gap"], marker="o")
            ax.set_title("Crossover diagnostic: O3 minus O1")
            ax.set_xlabel("I(peripheral; hazard | central), bits")
            ax.set_ylabel("Accuracy(O3) - Accuracy(O1)")
            ax.grid(alpha=0.25)
            fig.tight_layout()
            fig.savefig(plot_dir / "12_o3_minus_o1_vs_information.png", dpi=170)
            plt.close(fig)

    # 13 distractor robustness
    if len(distractor_df):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for o in sorted(distractor_df.observer.unique()):
            d = distractor_df[distractor_df.observer == o].groupby("num_distractors")["accuracy"].mean()
            ax.plot(d.index, d.values, marker="o", label=o)
        ax.set_title("Distractor robustness")
        ax.set_xlabel("Number of causally irrelevant distractors")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "13_distractor_robustness.png", dpi=170)
        plt.close(fig)

        # 14 O2 relative efficiency decline
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        d = distractor_df[distractor_df.observer == "O2"].groupby("num_distractors").agg({"accuracy":"mean", "relative_efficiency":"mean"})
        ax.plot(d.index, d["relative_efficiency"], marker="o")
        ax.set_title("O2 exhaustive-observer efficiency as distractors grow")
        ax.set_xlabel("Number of distractors")
        ax.set_ylabel("Accuracy / relative processing cost")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "14_o2_efficiency_vs_distractors.png", dpi=170)
        plt.close(fig)

    # 15 reversal recovery
    if len(reversal_df):
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        for o in ("O3", "O4"):
            d = reversal_df[reversal_df.observer == o].groupby("reversal_epoch")["accuracy"].mean()
            ax.plot(d.index, d.values, marker="o", label=o)
        ax.set_title("Recovery after peripheral-cue reversal")
        ax.set_xlabel("Reversal adaptation epoch")
        ax.set_ylabel("Accuracy")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "15_reversal_recovery.png", dpi=170)
        plt.close(fig)

    # 16 LoRA cosine similarity heat map
    if len(lora_df):
        observers = sorted(lora_df.observer.unique())
        mat = np.eye(len(observers))
        for i, a in enumerate(observers):
            for j, b in enumerate(observers):
                pair = lora_df[((lora_df.observer_a == a) & (lora_df.observer_b == b))]
                if len(pair):
                    mat[i, j] = pair.cosine.mean()
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="coolwarm")
        ax.set_xticks(range(len(observers)), observers)
        ax.set_yticks(range(len(observers)), observers)
        ax.set_title("Pairwise LoRA update cosine similarity")
        fig.colorbar(im, ax=ax, label="Cosine similarity")
        fig.tight_layout()
        fig.savefig(plot_dir / "16_lora_cosine_similarity.png", dpi=170)
        plt.close(fig)

    # 17 confusion error rate by observer
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    d = metrics.groupby("observer").agg(fp=("fp", "mean"), fn=("fn", "mean"), n=("test_n", "mean")).reset_index()
    x = np.arange(len(d))
    ax.bar(x - 0.18, d.fp / d.n, width=0.36, label="False positive rate (count/N)")
    ax.bar(x + 0.18, d.fn / d.n, width=0.36, label="False negative rate (count/N)")
    ax.set_xticks(x, d.observer)
    ax.set_title("Held-out error composition")
    ax.set_ylabel("Error fraction")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "17_error_composition.png", dpi=170)
    plt.close(fig)

    # 18 matched POMDP vs fully observable MDP control
    if len(mdp_control_df):
        pom = metrics.groupby("observer")["accuracy"].mean()
        mdp = mdp_control_df.groupby("observer")["accuracy"].mean()
        order = [o for o in OBSERVERS if o in pom.index and o in mdp.index]
        x = np.arange(len(order))
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        ax.bar(x - 0.18, [pom[o] for o in order], width=0.36, label="POMDP")
        ax.bar(x + 0.18, [mdp[o] for o in order], width=0.36, label="MDP control")
        ax.set_xticks(x, order)
        ax.set_title("Partial observability control: POMDP vs MDP")
        ax.set_ylabel("Held-out accuracy")
        ax.set_xlabel("Observer")
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "18_pomdp_vs_mdp_control.png", dpi=170)
        plt.close(fig)

    # 19 direct durable-internalization intervention (LoRA ablation with GRU memory disabled)
    if "I_ret" in metrics.columns:
        save_bar(metrics, "I_ret", "Durable internalization from LoRA ablation", "I_ret", plot_dir / "19_internalization_lora_ablation.png")

    # 20-22 rho sweep: rho is treated as a measured experimental variable.
    if len(rho_sweep):
        g = rho_sweep.groupby("rho").agg(accuracy=("accuracy", "mean"), OWGE_plus=("OWGE_plus", "mean"), I_ret=("I_ret", "mean"), cost=("relative_processing_cost", "mean")).reset_index()
        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(g["rho"], g["accuracy"], marker="o")
        ax.set_title("O3 held-out accuracy across peripheral reserve rho")
        ax.set_xlabel("Peripheral reserve rho")
        ax.set_ylabel("Held-out accuracy")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "20_rho_sweep_accuracy.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        ax.plot(g["rho"], g["OWGE_plus"], marker="o", label="OWGE+")
        ax.plot(g["rho"], g["I_ret"], marker="o", label="I_ret")
        ax.set_title("OWGE and durable internalization across rho")
        ax.set_xlabel("Peripheral reserve rho")
        ax.set_ylabel("Score")
        ax.legend()
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "21_rho_sweep_owge_internalization.png", dpi=170)
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(7.2, 4.8))
        efficiency = g["accuracy"] / g["cost"].clip(lower=1e-8)
        ax.plot(g["rho"], efficiency, marker="o")
        ax.set_title("Resource-adjusted O3 performance across rho")
        ax.set_xlabel("Peripheral reserve rho")
        ax.set_ylabel("Accuracy / relative processing cost")
        ax.grid(alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_dir / "22_rho_sweep_resource_efficiency.png", dpi=170)
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main experiment / sweeps
# -----------------------------------------------------------------------------


def lora_pair_rows(seed: int, agents: Dict[str, ObserverAgent]) -> List[Dict[str, object]]:
    rows = []
    vecs = {o: a.lora_vector() for o, a in agents.items()}
    for a, va in vecs.items():
        for b, vb in vecs.items():
            denom = np.linalg.norm(va) * np.linalg.norm(vb)
            cosine = float(np.dot(va, vb) / denom) if denom > 1e-12 else 0.0
            rows.append({"seed": seed, "observer": a, "observer_a": a, "observer_b": b, "cosine": cosine})
    return rows


def main_condition_run(
    base: BaseSequencePolicy,
    cfg: ExperimentConfig,
    device: torch.device,
    seed: int,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], List[Dict[str, object]], Dict[str, ObserverAgent]]:
    train_ds = SyntheticEpisodeDataset(cfg.train_episodes, seed * 10000 + 100, cfg, TRAIN_STYLES)
    val_ds = SyntheticEpisodeDataset(cfg.val_episodes, seed * 10000 + 200, cfg, VAL_STYLES)
    test_ds = SyntheticEpisodeDataset(cfg.test_episodes, seed * 10000 + 300, cfg, TEST_STYLES)
    high_dist_ds = SyntheticEpisodeDataset(cfg.test_episodes, seed * 10000 + 300, cfg, TEST_STYLES, num_distractors=max(cfg.distractor_sweep))
    info = dataset_information(test_ds)

    metrics_rows, curve_rows = [], []
    agents: Dict[str, ObserverAgent] = {}
    for observer in OBSERVERS:
        print(f"\n=== seed={seed} observer={observer} ===")
        agent, curve, replay_stats = train_observer(base, observer, train_ds, val_ds, cfg, device, seed)
        agents[observer] = agent
        test_eval = evaluate_agent(agent, test_ds, cfg, device)
        no_mem_eval = evaluate_agent(agent, test_ds, cfg, device, force_no_memory=True)
        lora_ablated_eval = evaluate_agent(agent, test_ds, cfg, device, disable_lora=True)
        lora_ablated_no_mem_eval = evaluate_agent(agent, test_ds, cfg, device, force_no_memory=True, disable_lora=True)
        high_dist_eval = evaluate_agent(agent, high_dist_ds, cfg, device)
        base_acc = evaluate_unadapted_base(base, observer, test_ds, cfg, device)
        ow = compute_owge_metrics(
            observer, cfg, info, agent, test_eval, no_mem_eval,
            lora_ablated_eval, lora_ablated_no_mem_eval, base_acc,
            float(high_dist_eval["accuracy"]), replay_stats,
        )
        total_params, trainable_params = count_parameters(agent)
        row = {
            "condition": "main",
            "seed": seed,
            "observer": observer,
            "accuracy": float(test_eval["accuracy"]),
            "reward": float(test_eval["reward"]),
            "no_memory_accuracy": float(no_mem_eval["accuracy"]),
            "high_distractor_accuracy": float(high_dist_eval["accuracy"]),
            "base_accuracy": float(base_acc),
            "lora_ablated_accuracy": float(lora_ablated_eval["accuracy"]),
            "lora_ablated_no_memory_accuracy": float(lora_ablated_no_mem_eval["accuracy"]),
            "lora_norm": agent.lora_norm(),
            "total_params": total_params,
            "trainable_params": trainable_params,
            "test_n": len(test_ds),
            "tp": test_eval["tp"], "tn": test_eval["tn"], "fp": test_eval["fp"], "fn": test_eval["fn"],
            **info,
            **replay_stats,
            **ow,
        }
        metrics_rows.append(row)
        for cr in curve:
            curve_rows.append({"seed": seed, "observer": observer, **cr})

    return metrics_rows, curve_rows, lora_pair_rows(seed, agents), agents


def run_mdp_control(base: BaseSequencePolicy, cfg: ExperimentConfig, device: torch.device) -> pd.DataFrame:
    """Train matched observers in a fully observable MDP control world.

    This is stronger than merely evaluating POMDP-trained agents with extra state:
    each observer is adapted from the same frozen base under a world where the hidden
    hazard is explicitly visible. If peripheral-memory machinery is only valuable
    because of partial observability, observer gaps should shrink here.
    """
    rows = []
    local = copy.deepcopy(cfg)
    local.train_episodes = cfg.mdp_control_episodes
    local.val_episodes = max(60, cfg.mdp_control_episodes // 3)
    local.test_episodes = max(120, cfg.mdp_control_episodes // 2)
    local.train_epochs = cfg.mdp_control_epochs
    for seed in cfg.seeds:
        train_ds = SyntheticEpisodeDataset(local.train_episodes, seed * 10000 + 5100, local, TRAIN_STYLES, fully_observable=True)
        val_ds = SyntheticEpisodeDataset(local.val_episodes, seed * 10000 + 5200, local, VAL_STYLES, fully_observable=True)
        test_ds = SyntheticEpisodeDataset(local.test_episodes, seed * 10000 + 5300, local, TEST_STYLES, fully_observable=True)
        for observer in OBSERVERS:
            agent, _, replay_stats = train_observer(base, observer, train_ds, val_ds, local, device, seed + 7000, epochs_override=local.train_epochs)
            ev = evaluate_agent(agent, test_ds, local, device)
            rows.append({
                "seed": seed, "observer": observer, "mode": "MDP_control",
                "accuracy": ev["accuracy"], "reward": ev["reward"],
                "replay_accept_rate": replay_stats.get("replay_accept_rate", 0.0),
            })
    return pd.DataFrame(rows)


def run_mi_sweep(base: BaseSequencePolicy, cfg: ExperimentConfig, device: torch.device) -> pd.DataFrame:
    """Retrain only O1 and O3 at several peripheral information levels; keeps sweep laptop-feasible."""
    rows = []
    local = copy.deepcopy(cfg)
    local.train_episodes = cfg.sweep_train_episodes
    local.train_epochs = cfg.sweep_train_epochs
    local.val_episodes = max(60, cfg.sweep_train_episodes // 3)
    local.test_episodes = max(100, cfg.sweep_train_episodes // 2)
    for seed in cfg.sweep_seeds:
        for pacc in cfg.mi_sweep_levels:
            train_ds = SyntheticEpisodeDataset(local.train_episodes, seed * 10000 + int(pacc * 1000), local, TRAIN_STYLES, peripheral_accuracy=pacc)
            val_ds = SyntheticEpisodeDataset(local.val_episodes, seed * 10000 + 1000 + int(pacc * 1000), local, VAL_STYLES, peripheral_accuracy=pacc)
            test_ds = SyntheticEpisodeDataset(local.test_episodes, seed * 10000 + 2000 + int(pacc * 1000), local, TEST_STYLES, peripheral_accuracy=pacc)
            inf = dataset_information(test_ds)
            for observer in ("O1", "O3"):
                agent, _, _ = train_observer(base, observer, train_ds, val_ds, local, device, seed + int(pacc * 100), epochs_override=local.train_epochs)
                ev = evaluate_agent(agent, test_ds, local, device)
                rows.append({
                    "seed": seed, "peripheral_accuracy": pacc, "observer": observer,
                    "accuracy": ev["accuracy"], "reward": ev["reward"],
                    "peripheral_mi": inf["I_peripheral_hazard_bits"],
                    "peripheral_cmi": inf["I_peripheral_hazard_given_central_bits"],
                })
    return pd.DataFrame(rows)


def run_rho_sweep(base: BaseSequencePolicy, cfg: ExperimentConfig, device: torch.device) -> pd.DataFrame:
    """Systematically sweep the peripheral reserve rho for O3.

    rho is an experimental variable, not a claimed universal constant. Each rho
    receives matched synthetic train/validation/test episodes for each sweep seed.
    The sweep is diagnostic and uses a disjoint seed family from the main hypothesis
    tests, so final held-out main results are not used to choose rho.
    """
    rows = []
    local = copy.deepcopy(cfg)
    local.train_episodes = cfg.sweep_train_episodes
    local.train_epochs = cfg.sweep_train_epochs
    local.val_episodes = max(60, cfg.sweep_train_episodes // 3)
    local.test_episodes = max(100, cfg.sweep_train_episodes // 2)
    for seed in cfg.rho_sweep_seeds:
        # All rho values for a seed see exactly the same episodes.
        train_seed = seed * 10000 + 5100
        val_seed = seed * 10000 + 5200
        test_seed = seed * 10000 + 5300
        for rho in cfg.rho_sweep_levels:
            local.rho_peripheral = float(rho)
            train_ds = SyntheticEpisodeDataset(local.train_episodes, train_seed, local, TRAIN_STYLES)
            val_ds = SyntheticEpisodeDataset(local.val_episodes, val_seed, local, VAL_STYLES)
            test_ds = SyntheticEpisodeDataset(local.test_episodes, test_seed, local, TEST_STYLES)
            inf = dataset_information(test_ds)
            agent, _, replay_stats = train_observer(
                base, "O3", train_ds, val_ds, local, device,
                seed + int(round(rho * 1000)), epochs_override=local.train_epochs,
            )
            ev = evaluate_agent(agent, test_ds, local, device)
            no_mem = evaluate_agent(agent, test_ds, local, device, force_no_memory=True)
            ablated = evaluate_agent(agent, test_ds, local, device, disable_lora=True)
            ablated_no_mem = evaluate_agent(agent, test_ds, local, device, force_no_memory=True, disable_lora=True)
            base_acc = evaluate_unadapted_base(base, "O3", test_ds, local, device)
            high_dist_ds = SyntheticEpisodeDataset(local.test_episodes, test_seed, local, TEST_STYLES, num_distractors=max(local.distractor_sweep))
            high_dist = evaluate_agent(agent, high_dist_ds, local, device)
            ow = compute_owge_metrics(
                "O3", local, inf, agent, ev, no_mem, ablated, ablated_no_mem,
                base_acc, float(high_dist["accuracy"]), replay_stats,
            )
            weights = observer_region_weights("O3", local)
            proc_cost = weights["central"] + weights["peripheral"] + weights["billboard"] + local.num_distractors * weights["distractor"]
            rows.append({
                "seed": seed, "rho": float(rho), "observer": "O3",
                "accuracy": float(ev["accuracy"]), "reward": float(ev["reward"]),
                "OWGE_plus": float(ow["OWGE_plus"]), "I_ret": float(ow["I_ret"]),
                "I_adapter_gain": float(ow["I_adapter_gain"]),
                "lora_norm": agent.lora_norm(), "relative_processing_cost": float(proc_cost),
                "peripheral_cmi": inf["I_peripheral_hazard_given_central_bits"],
            })
    return pd.DataFrame(rows)


def run_distractor_sweep(agents_by_seed: Dict[int, Dict[str, ObserverAgent]], cfg: ExperimentConfig, device: torch.device) -> pd.DataFrame:
    rows = []
    for seed, agents in agents_by_seed.items():
        for nd in cfg.distractor_sweep:
            ds = SyntheticEpisodeDataset(cfg.test_episodes, seed * 10000 + 300, cfg, TEST_STYLES, num_distractors=nd)
            for o, agent in agents.items():
                ev = evaluate_agent(agent, ds, cfg, device)
                weights = observer_region_weights(o, cfg)
                proc_cost = max(0.05, weights["central"] + weights["peripheral"] + weights["billboard"] + nd * weights["distractor"])
                rows.append({
                    "seed": seed, "observer": o, "num_distractors": nd,
                    "accuracy": ev["accuracy"], "reward": ev["reward"],
                    "relative_processing_cost": proc_cost,
                    "relative_efficiency": float(ev["accuracy"]) / proc_cost,
                })
    return pd.DataFrame(rows)


def run_reversal(
    base: BaseSequencePolicy,
    trained_agents: Dict[int, Dict[str, ObserverAgent]],
    cfg: ExperimentConfig,
    device: torch.device,
) -> pd.DataFrame:
    """Continue O3/O4 on a world where the peripheral relationship reverses; old O4 replay can become stale."""
    rows = []
    for seed, agents in trained_agents.items():
        for observer in ("O3", "O4"):
            agent = copy.deepcopy(agents[observer]).to(device)
            opt = torch.optim.Adam(list(agent.trainable_parameters()), lr=cfg.lora_lr)
            rev_train = SyntheticEpisodeDataset(cfg.reversal_episodes, seed * 10000 + 8100, cfg, TRAIN_STYLES, reversed_peripheral=True)
            rev_test = SyntheticEpisodeDataset(max(120, cfg.test_episodes // 2), seed * 10000 + 8200, cfg, TEST_STYLES, reversed_peripheral=True)
            old_train = SyntheticEpisodeDataset(min(cfg.replay_buffer_size, cfg.reversal_episodes), seed * 10000 + 100, cfg, TRAIN_STYLES, reversed_peripheral=False)
            baseline = 0.0
            for epoch in range(1, cfg.reversal_epochs + 1):
                loader = make_loader(rev_train, cfg, shuffle=True)
                agent.train()
                for frames, labels, _ in loader:
                    frames, labels = frames.to(device), labels.to(device)
                    logits = agent(frames)
                    loss, rewards, _ = reinforce_batch_loss(logits, labels, baseline, cfg.entropy_bonus)
                    opt.zero_grad(); loss.backward(); opt.step()
                    baseline = 0.9 * baseline + 0.1 * float(rewards.mean().item())
                # O4 intentionally continues a small amount of replay from old stable world.
                # Validation gating in the main phase should limit this, but reversal tests whether stale
                # consolidation can slow adaptation when the causal relation changes.
                if observer == "O4":
                    ids = random.sample(range(len(old_train)), k=min(cfg.replay_batch_size, len(old_train)))
                    batch = [old_train[i] for i in ids]
                    frames = torch.stack([b[0] for b in batch]).to(device)
                    labels = torch.stack([b[1] for b in batch]).to(device)
                    logits = agent(frames)
                    loss, _, _ = reinforce_batch_loss(logits, labels, baseline, cfg.entropy_bonus)
                    opt.zero_grad(); loss.backward(); opt.step()
                ev = evaluate_agent(agent, rev_test, cfg, device)
                rows.append({"seed": seed, "observer": observer, "reversal_epoch": epoch, "accuracy": ev["accuracy"], "reward": ev["reward"]})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Preview/test-set export
# -----------------------------------------------------------------------------


def export_preview_dataset(cfg: ExperimentConfig, out_dir: Path, n_episodes: int = 12) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = SyntheticEpisodeDataset(n_episodes, 424242, cfg, TEST_STYLES)
    rows = []
    for i in range(n_episodes):
        meta, nd = ds.get_meta(i)
        frames = render_episode(meta, cfg, nd)
        ep_dir = out_dir / f"episode_{i:03d}"
        ep_dir.mkdir(exist_ok=True)
        for t, frame in enumerate(frames):
            Image.fromarray(frame).save(ep_dir / f"frame_{t:02d}.png")
        rows.append({
            "episode": i,
            "episode_seed": meta.episode_seed,
            "style": meta.style,
            "hazard_hidden_state": meta.hazard,
            "central_bits": "".join(str(x) for x in meta.central_bits),
            "peripheral_bit_at_delayed_cue": meta.peripheral_bit,
            "billboard_ticket_bit": meta.ticket_bit,
            "num_distractors": nd,
            "split": "TEST_PREVIEW_ONLY",
        })
    pd.DataFrame(rows).to_csv(out_dir / "test_manifest.csv", index=False)


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="OWGE synthetic POMDP/LoRA validation experiment")
    parser.add_argument("--preset", choices=["smoke", "laptop", "paper"], default="laptop")
    parser.add_argument("--output", default="owge_run")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, mps")
    parser.add_argument("--skip-sweeps", action="store_true")
    parser.add_argument("--seed-count", type=int, default=None, help="override number of matched main seeds")
    parser.add_argument("--rho", type=float, default=None, help="freeze main O3/O4/O5R peripheral reserve rho after tuning (0..1)")
    args = parser.parse_args()

    cfg = apply_preset(ExperimentConfig(), args.preset)
    if args.seed_count is not None:
        cfg.seeds = cfg.seeds[: max(1, args.seed_count)]
    if args.rho is not None:
        if not 0.0 <= args.rho <= 1.0:
            raise ValueError("--rho must be between 0 and 1")
        cfg.rho_peripheral = float(args.rho)
    if args.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    cfg.device = str(device)

    out = Path(args.output)
    result_dir = out / "results"
    plot_dir = out / "plots"
    preview_dir = out / "data_preview"
    result_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)
    export_preview_dataset(cfg, preview_dir)
    (out / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))

    print("\nOWGE validation harness")
    print(f"preset={cfg.preset} device={device} output={out.resolve()}")
    print("Generating one shared base model. Every observer starts from this exact checkpoint.\n")
    base = pretrain_base(cfg, device)
    base_total, base_trainable = count_parameters(base)
    print(f"BaseSequencePolicy parameters: total={base_total:,}, trainable before freeze={base_trainable:,}")

    metrics_rows: List[Dict[str, object]] = []
    curve_rows: List[Dict[str, object]] = []
    lora_rows: List[Dict[str, object]] = []
    agents_by_seed: Dict[int, Dict[str, ObserverAgent]] = {}

    for seed in cfg.seeds:
        m, c, l, agents = main_condition_run(base, cfg, device, seed)
        metrics_rows.extend(m); curve_rows.extend(c); lora_rows.extend(l)
        agents_by_seed[seed] = agents

    metrics = pd.DataFrame(metrics_rows)
    curves = pd.DataFrame(curve_rows)
    lora_df = pd.DataFrame(lora_rows)

    # Information-theory summary for main dataset and MDP control dataset.
    info_rows = []
    for seed in cfg.seeds:
        pomdp = SyntheticEpisodeDataset(cfg.test_episodes, seed * 10000 + 300, cfg, TEST_STYLES, fully_observable=False)
        mdp = SyntheticEpisodeDataset(cfg.test_episodes, seed * 10000 + 300, cfg, TEST_STYLES, fully_observable=True)
        for mode, ds in (("POMDP", pomdp), ("MDP_control", mdp)):
            row = {"seed": seed, "mode": mode, **dataset_information(ds)}
            info_rows.append(row)
    info_df = pd.DataFrame(info_rows)

    if args.skip_sweeps:
        mi_sweep = pd.DataFrame()
        rho_sweep = pd.DataFrame()
        distractor_df = run_distractor_sweep(agents_by_seed, cfg, device)
        reversal_df = pd.DataFrame()
    else:
        print("\n=== information-theory / peripheral-cue sweep ===")
        mi_sweep = run_mi_sweep(base, cfg, device)
        print("\n=== peripheral-reserve rho sweep ===")
        rho_sweep = run_rho_sweep(base, cfg, device)
        print("\n=== distractor sweep ===")
        distractor_df = run_distractor_sweep(agents_by_seed, cfg, device)
        print("\n=== reversal experiment ===")
        reversal_df = run_reversal(base, agents_by_seed, cfg, device)

    print("\n=== fully observable MDP control training ===")
    mdp_control_df = run_mdp_control(base, cfg, device)

    hypothesis_tests = build_hypothesis_tests(metrics, reversal_df, mi_sweep, mdp_control_df)

    # Save machine-readable results.
    metrics.to_csv(result_dir / "metrics.csv", index=False)
    curves.to_csv(result_dir / "learning_curves.csv", index=False)
    lora_df.to_csv(result_dir / "lora_geometry.csv", index=False)
    info_df.to_csv(result_dir / "information_theory.csv", index=False)
    mi_sweep.to_csv(result_dir / "mi_sweep.csv", index=False)
    rho_sweep.to_csv(result_dir / "rho_sweep.csv", index=False)
    distractor_df.to_csv(result_dir / "distractor_sweep.csv", index=False)
    reversal_df.to_csv(result_dir / "reversal.csv", index=False)
    mdp_control_df.to_csv(result_dir / "mdp_control_training.csv", index=False)
    hypothesis_tests.to_csv(result_dir / "hypothesis_tests.csv", index=False)

    generate_plots(metrics, curves, info_df, mi_sweep, rho_sweep, distractor_df, reversal_df, lora_df, mdp_control_df, plot_dir)

    # Compact human-readable summary.
    summary_lines = []
    summary_lines.append("OWGE VALIDATION SUMMARY")
    summary_lines.append("=======================")
    summary_lines.append(f"Base model parameters: {base_total:,}")
    summary_lines.append(f"Main matched seeds: {len(cfg.seeds)}")
    summary_lines.append("")
    for o in OBSERVERS:
        d = metrics[metrics.observer == o]
        summary_lines.append(
            f"{o}: accuracy={d.accuracy.mean():.3f} +/- {d.accuracy.std(ddof=1) if len(d)>1 else 0:.3f}, "
            f"OWGE+={d.OWGE_plus.mean():.3f}, LoRA norm={d.lora_norm.mean():.3f}"
        )
    summary_lines.append("")
    if len(rho_sweep):
        rho_mean = rho_sweep.groupby("rho")["accuracy"].mean()
        best_rho = float(rho_mean.idxmax())
        summary_lines.append(f"Exploratory rho sweep best mean held-out accuracy at rho={best_rho:.3f}; do not retroactively substitute this into the same test run.")
    summary_lines.append("")
    summary_lines.append("Hypothesis tests:")
    if len(hypothesis_tests):
        for _, r in hypothesis_tests.iterrows():
            summary_lines.append(
                f"- {r['hypothesis']}: p={r['one_sided_p']:.4g} "
                f"reject_H0={bool(r['reject_H0_at_0.05'])}"
            )
    else:
        summary_lines.append("- No statistical tests produced.")
    (result_dir / "SUMMARY.txt").write_text("\n".join(summary_lines))
    print("\n" + "\n".join(summary_lines))
    print(f"\nPlots written to {plot_dir.resolve()} (target >= 22 files with sweeps enabled).")
    print(f"Return the entire '{out}' folder (or zip it) for analysis and LaTeX paper generation.")


if __name__ == "__main__":
    main()
