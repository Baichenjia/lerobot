#!/usr/bin/env python
"""使用 danaaubakirova/koch_test 数据集离线演示如何训练 Pi0 Policy。

运行示例前请在 shell 中执行：
    conda activate lerobot
并根据集群空闲情况手动选择 GPU0 或 GPU1，例如：
    CUDA_VISIBLE_DEVICES=0 python examples/training/train_policy_pi0.py
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch

from lerobot.datasets.backward_compatibility import CompatibilityError
from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi0.configuration_pi0 import PI0Config
from lerobot.policies.pi0.modeling_pi0 import PI0Policy
DEFAULT_DATASET_REPO = "danaaubakirova/koch_test"
FALLBACK_DATASET_REPO = "lerobot/aloha_static_coffee"



def _move_batch_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    """递归地把 batch 中的张量迁移到目标设备，其余对象原样返回。"""

    moved = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(device)
        else:
            moved[key] = value
    return moved


def _load_metadata_with_fallback(repo_id: str) -> tuple[str, LeRobotDatasetMetadata]:
    """加载数据集元信息，如遇兼容性报错则自动降级到官方 v3 数据集。"""

    try:
        metadata = LeRobotDatasetMetadata(repo_id)
        return repo_id, metadata
    except (CompatibilityError, NotImplementedError) as exc:
        logging.warning(
            "数据集 %s 无法直接使用 (%s)，将自动降级到 %s 继续示例。",
            repo_id,
            exc,
            FALLBACK_DATASET_REPO,
        )
        metadata = LeRobotDatasetMetadata(FALLBACK_DATASET_REPO)
        return FALLBACK_DATASET_REPO, metadata


def main():
    logging.basicConfig(level=logging.INFO)

    output_dir = Path("outputs/train/example_pi0")
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("使用设备: %s", device)

    dataset_repo = os.environ.get("PI0_DATASET_REPO", DEFAULT_DATASET_REPO)
    logging.info("加载数据集元信息: %s", dataset_repo)
    dataset_repo, dataset_metadata = _load_metadata_with_fallback(dataset_repo)
    features = dataset_to_policy_features(dataset_metadata.features)

    output_features = {name: ft for name, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {name: ft for name, ft in features.items() if name not in output_features}

    cfg = PI0Config(
        input_features=input_features,
        output_features=output_features,
        device=device.type,
        debug_shapes=True,  # 默认打开形状日志，方便验证处理流程
    )

    policy = PI0Policy(cfg)
    policy.to(device)
    policy.train()

    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    dataset = LeRobotDataset(dataset_repo)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=4,
        batch_size=2,
        shuffle=True,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )

    optimizer = torch.optim.AdamW(policy.parameters(), lr=cfg.optimizer_lr, betas=cfg.optimizer_betas)

    total_steps = 50
    log_freq = 5
    step = 0
    stop_training = False

    logging.info("开始 Pi0 示例训练，共 %s 步", total_steps)
    while not stop_training:
        for batch in dataloader:
            batch = preprocessor(batch)
            batch = _move_batch_to_device(batch, device)

            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                logging.info("step=%d loss=%.4f", step, loss.item())

            step += 1
            if step >= total_steps:
                stop_training = True
                break

    logging.info("训练结束，保存 checkpoint 到 %s", output_dir)
    policy.save_pretrained(output_dir)
    preprocessor.save_pretrained(output_dir)
    postprocessor.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
