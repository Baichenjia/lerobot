# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

"""This script demonstrates how to train Diffusion Policy on the PushT environment."""

import logging
from pathlib import Path

import einops
import torch

from lerobot.configs.types import FeatureType
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.datasets.utils import dataset_to_policy_features
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.factory import make_pre_post_processors

# 设置日志记录
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler("outputs/train/dimension_debug.log", mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def log_tensor_info(name: str, tensor: torch.Tensor):
    """记录张量的维度信息"""
    if isinstance(tensor, torch.Tensor):
        shape_str = " × ".join(map(str, tensor.shape))
        logger.info(f"[TENSOR] {name}: shape=({shape_str}), dtype={tensor.dtype}, device={tensor.device}")
        # 如果是标量或小张量，记录具体值
        if tensor.numel() <= 10:
            logger.info(f"  └─ values: {tensor.flatten().tolist()}")
    else:
        logger.info(f"[NON-TENSOR] {name}: {type(tensor)} = {tensor}")


def log_scalar_info(name: str, value):
    """记录标量信息"""
    logger.info(f"[SCALAR] {name}: {value} (type: {type(value).__name__})")


def main():
    # Create a directory to store the training checkpoint.
    output_directory = Path("outputs/train/example_pusht_diffusion")
    output_directory.mkdir(parents=True, exist_ok=True)

    # # Select your device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Number of offline training steps (we'll only do offline training for this example.)
    # Adjust as you prefer. 5000 steps are needed to get something worth evaluating.
    training_steps = 5
    log_freq = 1

    # When starting from scratch (i.e. not from a pretrained policy), we need to specify 2 things before
    # creating the policy:
    #   - input/output shapes: to properly size the policy
    #   - dataset stats: for normalization and denormalization of input/outputs
    logger.info("=" * 80)
    logger.info("Loading dataset metadata...")
    dataset_metadata = LeRobotDatasetMetadata("lerobot/pusht")
    log_scalar_info("dataset_metadata.fps", dataset_metadata.fps)
    log_scalar_info("dataset_metadata.total_frames", dataset_metadata.total_frames)
    
    features = dataset_to_policy_features(dataset_metadata.features)
    logger.info(f"Features: {features}")
    
    output_features = {key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION}
    input_features = {key: ft for key, ft in features.items() if key not in output_features}
    logger.info(f"Input features: {input_features}")
    logger.info(f"Output features: {output_features}")

    # Policies are initialized with a configuration class, in this case `DiffusionConfig`.
    logger.info("=" * 80)
    logger.info("Creating DiffusionConfig...")
    cfg = DiffusionConfig(input_features=input_features, output_features=output_features)
    
    # 记录配置参数
    log_scalar_info("config.n_obs_steps", cfg.n_obs_steps)
    log_scalar_info("config.horizon", cfg.horizon)
    log_scalar_info("config.n_action_steps", cfg.n_action_steps)
    log_scalar_info("config.vision_backbone", cfg.vision_backbone)
    log_scalar_info("config.crop_shape", cfg.crop_shape)
    log_scalar_info("config.spatial_softmax_num_keypoints", cfg.spatial_softmax_num_keypoints)
    log_scalar_info("config.down_dims", cfg.down_dims)
    log_scalar_info("config.diffusion_step_embed_dim", cfg.diffusion_step_embed_dim)
    log_scalar_info("config.num_train_timesteps", cfg.num_train_timesteps)
    log_scalar_info("config.prediction_type", cfg.prediction_type)
    log_scalar_info("config.global_cond_dim (computed)", 
                    cfg.robot_state_feature.shape[0] + len(cfg.image_features) * (cfg.spatial_softmax_num_keypoints * 2))
    
    # We can now instantiate our policy with this config and the dataset stats.
    logger.info("=" * 80)
    logger.info("Creating DiffusionPolicy...")
    policy = DiffusionPolicy(cfg)
    policy.train()
    policy.to(device)
    
    logger.info(f"Policy device: {next(policy.parameters()).device}")
    logger.info(f"Policy dtype: {next(policy.parameters()).dtype}")
    
    # 记录 policy 结构
    logger.info("=" * 80)
    logger.info("Policy structure:")
    for name, module in policy.named_modules():
        if name:  # 跳过根模块
            logger.info(f"  {name}: {module.__class__.__name__}")
    
    preprocessor, postprocessor = make_pre_post_processors(cfg, dataset_stats=dataset_metadata.stats)

    # Another policy-dataset interaction is with the delta_timestamps.
    delta_timestamps = {
        "observation.image": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "observation.state": [i / dataset_metadata.fps for i in cfg.observation_delta_indices],
        "action": [i / dataset_metadata.fps for i in cfg.action_delta_indices],
    }
    logger.info(f"Delta timestamps: {delta_timestamps}")

    # We can then instantiate the dataset with these delta_timestamps configuration.
    logger.info("=" * 80)
    logger.info("Creating dataset...")
    dataset = LeRobotDataset("lerobot/pusht", delta_timestamps=delta_timestamps)
    log_scalar_info("dataset.len", len(dataset))

    # Then we create our optimizer and dataloader for offline training.
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-4)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=0,  # 设置为 0 以便更好地调试
        batch_size=64,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Run training loop.
    logger.info("=" * 80)
    logger.info("Starting training loop...")
    step = 0
    done = False
    
    for batch in dataloader:
        logger.info("=" * 80)
        logger.info(f"BATCH {step}: Initial batch keys: {list(batch.keys())}")
        
        # 记录初始 batch 的维度
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                log_tensor_info(f"batch[{key}]", val)
        
        batch = preprocessor(batch)
        logger.info(f"BATCH {step}: After preprocessor keys: {list(batch.keys())}")
        
        # 记录预处理后的 batch 维度
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                log_tensor_info(f"batch[{key}]", val)
        
        # 前向传播，记录详细维度
        logger.info("-" * 80)
        logger.info(f"BATCH {step}: Calling policy.forward()...")
        
        # 手动执行 forward 以记录更多维度信息
        if cfg.image_features:
            batch_copy = dict(batch)  # shallow copy
            batch_copy["observation.images"] = torch.stack([batch_copy[key] for key in cfg.image_features], dim=-4)
            log_tensor_info(f"batch_copy[observation.images] (stacked)", batch_copy["observation.images"])
            batch = batch_copy
        
        # 获取 diffusion 模型的详细维度信息
        logger.info("-" * 80)
        logger.info(f"BATCH {step}: Diffusion model compute_loss details:")
        
        # 记录 compute_loss 中的关键变量
        obs_state = batch["observation.state"]
        action = batch["action"]
        log_tensor_info("observation.state", obs_state)
        log_tensor_info("action", action)
        
        n_obs_steps = obs_state.shape[1]
        horizon = action.shape[1]
        log_scalar_info("n_obs_steps (from batch)", n_obs_steps)
        log_scalar_info("horizon (from batch)", horizon)
        
        # 记录 global_cond 的制备过程
        logger.info("-" * 80)
        logger.info(f"BATCH {step}: _prepare_global_conditioning details:")
        
        batch_size, n_obs_steps_batch = obs_state.shape[:2]
        log_scalar_info("batch_size", batch_size)
        log_scalar_info("n_obs_steps_batch", n_obs_steps_batch)
        
        global_cond_feats = [obs_state]
        log_tensor_info("global_cond_feats[0] (observation.state)", global_cond_feats[0])
        
        if cfg.image_features:
            images = batch["observation.images"]
            log_tensor_info("observation.images (stacked)", images)
            
            if cfg.use_separate_rgb_encoder_per_camera:
                logger.info("Using separate RGB encoder per camera")
                images_per_camera = einops.rearrange(images, "b s n ... -> n (b s) ...")
                log_tensor_info("images_per_camera (rearranged)", images_per_camera)
            else:
                logger.info("Using shared RGB encoder")
                images_rearranged = einops.rearrange(images, "b s n ... -> (b s n) ...")
                log_tensor_info("images_rearranged (for shared encoder)", images_rearranged)
                img_features = policy.diffusion.rgb_encoder(images_rearranged)
                log_tensor_info("img_features (after rgb_encoder)", img_features)
                img_features = einops.rearrange(
                    img_features, "(b s n) ... -> b s (n ...)", b=batch_size, s=n_obs_steps
                )
                log_tensor_info("img_features (after rearrange back)", img_features)
                global_cond_feats.append(img_features)
        
        # 拼接并展平 global_cond_feats
        global_cond = torch.cat(global_cond_feats, dim=-1).flatten(start_dim=1)
        log_tensor_info("global_cond (after cat and flatten)", global_cond)
        
        # 记录扩散过程
        logger.info("-" * 80)
        logger.info(f"BATCH {step}: Forward diffusion details:")
        
        trajectory = batch["action"]
        log_tensor_info("trajectory (batch[ACTION])", trajectory)
        
        # 采样噪声
        eps = torch.randn(trajectory.shape, device=trajectory.device)
        log_tensor_info("eps (sampled noise)", eps)
        
        # 采样随机时间步
        timesteps = torch.randint(
            low=0,
            high=policy.diffusion.noise_scheduler.config.num_train_timesteps,
            size=(trajectory.shape[0],),
            device=trajectory.device,
        ).long()
        log_tensor_info("timesteps (random)", timesteps)
        if timesteps.numel() <= 10:
            logger.info(f"  └─ timestep values: {timesteps.tolist()}")
        
        # 添加噪声
        noisy_trajectory = policy.diffusion.noise_scheduler.add_noise(trajectory, eps, timesteps)
        log_tensor_info("noisy_trajectory (after add_noise)", noisy_trajectory)
        
        # 运行 UNet
        logger.info("-" * 80)
        logger.info(f"BATCH {step}: UNet forward details:")
        
        # 在 UNet 内部记录
        unet_input = noisy_trajectory
        unet_timesteps = timesteps
        
        # 记录 UNet 配置
        log_scalar_info("unet.global_cond_dim", global_cond.shape[-1])
        log_scalar_info("unet.input_dim", unet_input.shape[-1])
        
        pred = policy.diffusion.unet(unet_input, unet_timesteps, global_cond=global_cond)
        log_tensor_info("pred (UNet output)", pred)
        
        # 计算损失
        logger.info("-" * 80)
        logger.info(f"BATCH {step}: Loss computation:")
        
        if cfg.prediction_type == "epsilon":
            target = eps
            logger.info("Using epsilon prediction type")
        elif cfg.prediction_type == "sample":
            target = batch["action"]
            logger.info("Using sample prediction type")
        
        log_tensor_info("target", target)
        
        loss = torch.nn.functional.mse_loss(pred, target, reduction="none")
        log_tensor_info("loss (before mean)", loss)
        
        loss_mean = loss.mean()
        log_scalar_info("loss.mean()", loss_mean.item())
        
        # 调用正式的 forward
        loss_official, _ = policy.forward(batch)
        log_scalar_info("loss_official", loss_official.item())
        
        loss_official.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % log_freq == 0:
            logger.info(f"step: {step} loss: {loss_official.item():.3f}")
        step += 1
        if step >= training_steps:
            break

    logger.info("=" * 80)
    logger.info("Training completed!")
    
    # Save a policy checkpoint.
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)
    logger.info(f"Checkpoints saved to {output_directory}")


if __name__ == "__main__":
    main()
