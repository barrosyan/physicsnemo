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

"""Roll the trained FNO out on the test cases and report errors against interFoam."""

import json
import time

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402
from hydra.utils import to_absolute_path  # noqa: E402
from omegaconf import DictConfig  # noqa: E402

from physicsnemo.utils import load_checkpoint  # noqa: E402

from utils import build_model, load_split, relative_l2, rollout  # noqa: E402

DOMAIN = 0.584  # tank side, m


def wet_iou(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Intersection over union of the regions where alpha > 0.5."""
    p, t = prediction > 0.5, target > 0.5
    inter = (p & t).flatten(-2).sum(-1).float()
    union = (p | t).flatten(-2).sum(-1).float()
    return inter / union


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    test = load_split(to_absolute_path(cfg.data.dir), "test")
    alpha, mask, times = (
        test["alpha"].to(device),
        test["mask"].to(device),
        test["times"],
    )

    model = build_model(cfg).to(device)
    load_checkpoint("./checkpoints", models=model, device=device)
    model.eval()

    with torch.no_grad():
        rollout(model, alpha[:1, 0], mask, 1)  # warm-up
        start = time.perf_counter()
        pred = rollout(model, alpha[:, 0], mask, alpha.shape[1] - 1)
        if device == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
    target = alpha[:, 1:]

    error = relative_l2(pred, target)  # (cases, steps)
    persistence = relative_l2(alpha[:, :1].expand_as(target), target)
    iou = wet_iou(pred, target)
    volume = pred.sum((-2, -1)) / target.sum((-2, -1)) - 1.0
    metrics = {
        "test_cases": alpha.shape[0],
        "steps": target.shape[1],
        "rollout_relative_l2_mean": error.mean().item(),
        "rollout_relative_l2_final": error[:, -1].mean().item(),
        "persistence_relative_l2_mean": persistence.mean().item(),
        "wet_region_iou_mean": iou.mean().item(),
        "wet_region_iou_final": iou[:, -1].mean().item(),
        "water_volume_error_final_abs_mean": volume[:, -1].abs().mean().item(),
        "rollout_seconds_all_test_cases": elapsed,
        "device": device,
    }
    with open("metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # Error growth over the rollout.
    fig, ax = plt.subplots(figsize=(5, 3.2))
    t = times[1:].numpy()
    ax.plot(t, error.mean(0).cpu(), label="FNO rollout")
    ax.plot(t, persistence.mean(0).cpu(), "--", label="persistence (alpha at t=0)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("relative L2 error of alpha.water")
    ax.legend()
    fig.tight_layout()
    fig.savefig("error_vs_time.png", dpi=150)

    # Prediction against interFoam for the first test case.
    columns = [int(torch.argmin((times - s).abs())) for s in cfg.inference.plot_times]
    fig, axes = plt.subplots(2, len(columns), figsize=(2.6 * len(columns), 5.4))
    extent = (0, DOMAIN, 0, DOMAIN)
    for j, k in enumerate(columns):
        for i, (field, name) in enumerate(((target, "interFoam"), (pred, "FNO"))):
            ax = axes[i, j]
            ax.imshow(
                field[0, k - 1].cpu(),
                origin="lower",
                extent=extent,
                cmap="Blues",
                vmin=0,
                vmax=1,
            )
            ax.contourf(
                mask.cpu(),
                levels=[0.5, 1.5],
                colors="0.3",
                extent=extent,
                origin="lower",
            )
            ax.set_title(f"{name}, t = {times[k]:.2f} s", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.tight_layout()
    fig.savefig("rollout.png", dpi=150)


if __name__ == "__main__":
    main()
