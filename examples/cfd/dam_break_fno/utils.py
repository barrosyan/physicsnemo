# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Dataset, model construction and rollout shared by training and inference."""

from pathlib import Path

import h5py
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

from physicsnemo.models.fno import FNO


def load_split(data_dir: str, split: str) -> dict[str, torch.Tensor]:
    """Load one split written by generate_dataset.py."""
    path = Path(data_dir) / f"dam_break_{split}.h5"
    with h5py.File(path, "r") as f:
        return {
            "alpha": torch.from_numpy(f["alpha"][:]),  # (cases, times, H, W)
            "mask": torch.from_numpy(f["mask"][:]),  # (H, W)
            "params": torch.from_numpy(f["params"][:]),
            "times": torch.from_numpy(f["times"][:]),
        }


class DamBreakWindows(Dataset):
    """Windows of ``unroll_steps + 2`` consecutive frames from every case.

    A window holds the two input frames (t - dt, t) followed by the
    ``unroll_steps`` target frames. At t = 0 the fluid is at rest, so the frame
    before the first one is taken equal to it.
    """

    def __init__(self, alpha: torch.Tensor, unroll_steps: int):
        first = alpha[:, :1]
        self.alpha = torch.cat([first, alpha], dim=1)  # prepend the rest state
        self.unroll_steps = unroll_steps
        self.windows_per_case = self.alpha.shape[1] - unroll_steps - 1

    def __len__(self) -> int:
        return self.alpha.shape[0] * self.windows_per_case

    def __getitem__(self, index: int) -> torch.Tensor:
        case, start = divmod(index, self.windows_per_case)
        return self.alpha[case, start : start + self.unroll_steps + 2]


def build_model(cfg: DictConfig) -> FNO:
    """FNO that maps (alpha_{t-1}, alpha_t, mask) to the update alpha_{t+1} - alpha_t."""
    return FNO(
        in_channels=cfg.arch.in_channels,
        out_channels=cfg.arch.out_channels,
        decoder_layers=cfg.arch.decoder_layers,
        decoder_layer_size=cfg.arch.decoder_layer_size,
        dimension=2,
        latent_channels=cfg.arch.latent_channels,
        num_fno_layers=cfg.arch.fno_layers,
        num_fno_modes=cfg.arch.fno_modes,
        padding=cfg.arch.padding,
    )


def step(
    model: torch.nn.Module,
    previous: torch.Tensor,
    current: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Advance alpha by one write interval.

    ``previous`` and ``current`` have shape (batch, H, W) and ``mask`` (H, W).
    The prediction is clamped to [0, 1] and set to zero inside the obstacle.
    """
    mask_b = mask.expand_as(current)
    inputs = torch.stack([previous, current, mask_b], dim=1)
    update = model(inputs)[:, 0]
    return ((current + update).clamp(0.0, 1.0)) * (1.0 - mask_b)


def rollout(
    model: torch.nn.Module,
    initial: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int,
    previous: torch.Tensor | None = None,
) -> torch.Tensor:
    """Autoregressive rollout from ``initial`` (batch, H, W).

    Returns the predicted frames, shape (batch, num_steps, H, W).
    """
    previous = initial if previous is None else previous
    current = initial
    frames = []
    for _ in range(num_steps):
        previous, current = current, step(model, previous, current, mask)
        frames.append(current)
    return torch.stack(frames, dim=1)


def relative_l2(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Relative L2 error over the last two (spatial) dimensions."""
    diff = (prediction - target).flatten(-2).norm(dim=-1)
    return diff / target.flatten(-2).norm(dim=-1)
