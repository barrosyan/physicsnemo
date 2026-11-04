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

"""Train an autoregressive FNO surrogate of the interFoam dam break."""

import hydra
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.utils.logging import LaunchLogger, PythonLogger

from utils import DamBreakWindows, build_model, load_split, relative_l2, rollout


@hydra.main(version_base="1.3", config_path=".", config_name="config.yaml")
def main(cfg: DictConfig) -> None:
    DistributedManager.initialize()
    dist = DistributedManager()
    device = dist.device
    torch.manual_seed(cfg.training.seed)

    log = PythonLogger(name="dam_break_fno")
    log.file_logging()
    LaunchLogger.initialize()

    data_dir = to_absolute_path(cfg.data.dir)
    train = load_split(data_dir, "train")
    valid = load_split(data_dir, "validation")
    mask = train["mask"].to(device)
    unroll = cfg.training.unroll_steps

    loader = DataLoader(
        DamBreakWindows(train["alpha"], unroll),
        batch_size=cfg.training.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_model(cfg).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.initial_lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.max_epochs * len(loader)
    )
    ckpt_args = {
        "path": "./checkpoints",
        "models": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }
    start_epoch = load_checkpoint(device=device, **ckpt_args)
    log.info(
        f"{sum(p.numel() for p in model.parameters())} parameters, "
        f"{len(loader.dataset)} training windows of {unroll} steps"
    )

    for epoch in range(max(1, start_epoch + 1), cfg.training.max_epochs + 1):
        model.train()
        with LaunchLogger(
            "train", epoch=epoch, num_mini_batch=len(loader), epoch_alert_freq=1
        ) as logger:
            for window in loader:
                window = window.to(device)
                # Unrolled training: the model is fed its own predictions so
                # that it learns to correct the errors it makes in a rollout.
                pred = rollout(model, window[:, 1], mask, unroll, previous=window[:, 0])
                loss = torch.mean((pred - window[:, 2:]) ** 2)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                logger.log_minibatch({"loss": loss.detach()})
            logger.log_epoch({"learning rate": optimizer.param_groups[0]["lr"]})

        if epoch % cfg.training.validation_freq == 0:
            model.eval()
            with torch.no_grad(), LaunchLogger("valid", epoch=epoch) as logger:
                alpha = valid["alpha"].to(device)
                pred = rollout(model, alpha[:, 0], mask, alpha.shape[1] - 1)
                error = relative_l2(pred, alpha[:, 1:]).mean()
                logger.log_epoch({"rollout relative L2": error.item()})

        if epoch % cfg.training.checkpoint_freq == 0:
            save_checkpoint(**ckpt_args, epoch=epoch)

    save_checkpoint(**ckpt_args, epoch=cfg.training.max_epochs)
    log.success("Training completed")


if __name__ == "__main__":
    main()
