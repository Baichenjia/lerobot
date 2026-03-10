#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
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
"""Action Chunking Transformer Policy

As per Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware (https://huggingface.co/papers/2304.13705).
The majority of changes here involve removing unused code, unifying naming, and adding helpful comments.
"""

import math
from collections import deque
from collections.abc import Callable
from itertools import chain

import einops
import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d

from lerobot.policies.act.configuration_act import ACTConfig
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.utils.constants import ACTION, OBS_ENV_STATE, OBS_IMAGES, OBS_STATE


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    
    维度说明（基于 lerobot/pusht 数据集）：
        - batch_size (B): 64 - 批次大小
        - chunk_size (S): 16 - 动作序列长度（每次预测的动作块大小）
        - action_dim: 2 - 动作维度（pusht 数据集为 2 维机械臂位置）
        - dim_model (D): 512 - 模型隐藏层维度
        - latent_dim: 32 - VAE 潜在空间维度
        - dim_feedforward: 2048 - 前馈网络中间层维度
        - n_heads: 8 - 多头注意力头数
        - 图像特征图：H=3, W=3 - ResNet18 骨干网络输出的特征图尺寸
    """

    config_class = ACTConfig
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
    ):
        """
        初始化 ACT 策略模型
        
        Args:
            config: 策略配置类实例，包含输入输出特征、模型超参数等
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)  # 创建 ACT 模型主体

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
        # TODO(aliberts, rcadene): As of now, lr_backbone == lr
        # Should we remove this and just `return self.parameters()`?
        return [
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if not n.startswith("model.backbone") and p.requires_grad
                ]
            },
            {
                "params": [
                    p
                    for n, p in self.named_parameters()
                    if n.startswith("model.backbone") and p.requires_grad
                ],
                "lr": self.config.optimizer_lr_backbone,
            },
        ]

    def reset(self):
        """This should be called whenever the environment is reset."""
        if self.config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler.reset()
        else:
            self._action_queue = deque([], maxlen=self.config.n_action_steps)

    @torch.no_grad()
    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Select a single action given environment observations.

        This method wraps `select_actions` in order to return one action at a time for execution in the
        environment. It works by managing the actions in a queue and only calling `select_actions` when the
        queue is empty.
        """
        self.eval()  # keeping the policy in eval mode as it could be set to train mode while queue is consumed

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        # Action queue logic for n_action_steps > 1. When the action_queue is depleted, populate it by
        # querying the policy.
        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]

            # `self.model.forward` returns a (batch_size, n_action_steps, action_dim) tensor, but the queue
            # effectively has shape (n_action_steps, batch_size, *), hence the transpose.
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)  # shallow copy so that adding a key doesn't modify the original
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """前向传播并计算损失，用于训练或验证
        
        Args:
            batch: 包含观测和动作的字典，例如：
                - observation.image: (B, C, H, W) = (64, 3, 96, 96) 图像观测
                - observation.state: (B, state_dim) = (64, 2) 机器人状态
                - action: (B, S, action_dim) = (64, 16, 2) 动作序列
                - action_is_pad: (B, S) = (64, 16) 动作填充掩码
        
        Returns:
            loss: 标量损失值
            loss_dict: 包含各分项损失的字典
        """
        if self.config.image_features:
            batch = dict(batch)  # 浅拷贝，避免修改原始字典
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        # 通过模型获取预测动作和 VAE 潜在分布参数
        # actions_hat: (B, S, action_dim) = (64, 16, 2) - 预测的动作序列
        # mu_hat: (B, latent_dim) = (64, 32) - 潜在分布均值
        # log_sigma_x2_hat: (B, latent_dim) = (64, 32) - 潜在分布对数方差 (2log(σ))
        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        # 计算 L1 损失：预测动作与真实动作的差异
        # batch[ACTION]: (B, S, action_dim) = (64, 16, 2)
        # actions_hat: (B, S, action_dim) = (64, 16, 2)
        # batch["action_is_pad"]: (B, S) = (64, 16)
        # ~batch["action_is_pad"].unsqueeze(-1): (B, S, 1) = (64, 16, 1) - 填充位置为 0，其他为 1
        # l1_loss: 标量 - 平均 L1 损失
        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
            # 计算 KL 散度：D_KL(q(z|x) || N(0,I))
            # 公式：-0.5 * (1 + log(σ²) - μ² - exp(log(σ²)))
            # mu_hat.pow(2): (B, latent_dim) = (64, 32) - μ²
            # (log_sigma_x2_hat).exp(): (B, latent_dim) = (64, 32) - exp(log(σ²)) = σ²
            # .sum(-1): (B,) = (64,) - 沿潜在维度求和
            # .mean(): 标量 - 批次平均
            mean_kld = (
                (-0.5 * (1 + log_sigma_x2_hat - mu_hat.pow(2) - (log_sigma_x2_hat).exp())).sum(-1).mean()
            )
            loss_dict["kld_loss"] = mean_kld.item()
            loss = l1_loss + mean_kld * self.config.kl_weight
        else:
            loss = l1_loss

        return loss, loss_dict


class ACTTemporalEnsembler:
    def __init__(self, temporal_ensemble_coeff: float, chunk_size: int) -> None:
        """Temporal ensembling as described in Algorithm 2 of https://huggingface.co/papers/2304.13705.

        The weights are calculated as wᵢ = exp(-temporal_ensemble_coeff * i) where w₀ is the oldest action.
        They are then normalized to sum to 1 by dividing by Σwᵢ. Here's some intuition around how the
        coefficient works:
            - Setting it to 0 uniformly weighs all actions.
            - Setting it positive gives more weight to older actions.
            - Setting it negative gives more weight to newer actions.
        NOTE: The default value for `temporal_ensemble_coeff` used by the original ACT work is 0.01. This
        results in older actions being weighed more highly than newer actions (the experiments documented in
        https://github.com/huggingface/lerobot/pull/319 hint at why highly weighing new actions might be
        detrimental: doing so aggressively may diminish the benefits of action chunking).

        Here we use an online method for computing the average rather than caching a history of actions in
        order to compute the average offline. For a simple 1D sequence it looks something like:

        ```
        import torch

        seq = torch.linspace(8, 8.5, 100)
        print(seq)

        m = 0.01
        exp_weights = torch.exp(-m * torch.arange(len(seq)))
        print(exp_weights)

        # Calculate offline
        avg = (exp_weights * seq).sum() / exp_weights.sum()
        print("offline", avg)

        # Calculate online
        for i, item in enumerate(seq):
            if i == 0:
                avg = item
                continue
            avg *= exp_weights[:i].sum()
            avg += item * exp_weights[i]
            avg /= exp_weights[: i + 1].sum()
        print("online", avg)
        ```
        """
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        # (chunk_size,) count of how many actions are in the ensemble for each time step in the sequence.
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            # Initializes `self._ensembled_action` to the sequence of actions predicted during the first
            # time step of the episode.
            self.ensembled_actions = actions.clone()
            # Note: The last dimension is unsqueeze to make sure we can broadcast properly for tensor
            # operations later.
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            # self.ensembled_actions will have shape (batch_size, chunk_size - 1, action_dim). Compute
            # the online update for those entries.
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            # The last action, which has no prior online average, needs to get concatenated onto the end.
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        # "Consume" the first action.
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """动作分块 Transformer：ACTPolicy 的底层神经网络

    注意：在本代码中我们使用术语 `vae_encoder`、'encoder'、`decoder`。含义如下：
        - `vae_encoder`：根据变分自编码器（VAE）文献，这是编码目标数据（动作序列）
          和条件（机器人关节空间）的模型部分。
        - 带有交叉注意力的 Transformer `encoder`（不是 VAE 编码器）和 `decoder`（不是 VAE 解码器）
          用作 VAE 解码器。对于这些术语，我们去掉 `vae_` 前缀，因为我们有一个选项可以不用
          变分目标训练此模型（在这种情况下我们完全去掉 `vae_encoder`，模型与 VAE 无关）。

                                 Transformer
                                 推理时单独使用
                                 （训练时作为 VAE 解码器）
                                ┌───────────────────────┐
                                │             Outputs   │
                                │                ▲      │
                                │     ┌─────►┌───────┐  │
                   ┌──────┐     │     │      │Transf.│  │
                   │      │     │     ├─────►│decoder│  │
              ┌────┴────┐ │     │     │      │       │  │
              │         │ │     │ ┌───┴───┬─►│       │  │
              │ VAE     │ │     │ │       │  └───────┘  │
              │ encoder │ │     │ │Transf.│             │
              │         │ │     │ │encoder│             │
              └───▲─────┘ │     │ │       │             │
                  │       │     │ └▲──▲─▲─┘             │
                  │       │     │  │  │ │               │
                inputs    └─────┼──┘  │ image emb.      │
                                │    state emb.         │
                                └───────────────────────┘

    维度说明（基于 lerobot/pusht 数据集）：
        - B (batch_size): 64 - 批次大小
        - S (chunk_size): 16 - 动作序列长度
        - D (dim_model): 512 - 模型隐藏层维度
        - latent_dim: 32 - VAE 潜在空间维度
        - action_dim: 2 - 动作维度
        - H, W: 3, 3 - ResNet18 骨干网络输出的图像特征图尺寸
    """

    def __init__(self, config: ACTConfig):
        """初始化 ACT 模型

        Args:
            config: 策略配置类实例
        """
        # BERT 风格 VAE 编码器，输入令牌 [cls, robot_state, *action_sequence]
        # cls 令牌形成潜在分布的参数（如 [*means, *log_variances]）
        super().__init__()
        self.config = config
        print("\nself.config:", self.config)
        # self.config: ACTConfig(n_obs_steps=1, input_features={'observation.image': PolicyFeature(type=<FeatureType.VISUAL: 'VISUAL'>, shape=(3, 96, 96)), 
        # 'observation.state': PolicyFeature(type=<FeatureType.STATE: 'STATE'>, shape=(2,))}, output_features={'action': PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>, shape=(2,))}, device='cuda', use_amp=False, push_to_hub=True, repo_id=None, private=None, tags=None, 
        # license=None, chunk_size=16, n_action_steps=16, horizon=16, normalization_mapping={'VISUAL': <NormalizationMode.MEAN_STD: 'MEAN_STD'>, 
        # 'STATE': <NormalizationMode.MEAN_STD: 'MEAN_STD'>, 'ACTION': <NormalizationMode.MEAN_STD: 'MEAN_STD'>}, vision_backbone='resnet18', pretrained_backbone_weights='ResNet18_Weights.IMAGENET1K_V1', 
        # replace_final_stride_with_dilation=False, pre_norm=False, dim_model=512, n_heads=8, dim_feedforward=3200, feedforward_activation='relu', n_encoder_layers=4, 
        # n_decoder_layers=1, use_vae=True, latent_dim=32, n_vae_encoder_layers=4, temporal_ensemble_coeff=None, dropout=0.1, kl_weight=10.0, optimizer_lr=1e-05, optimizer_weight_decay=0.0001, 
        # optimizer_lr_backbone=1e-05)
        if self.config.use_vae:
            # VAE 编码器（用于训练时编码动作序列）
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            # cls 令牌嵌入（用于提取潜在分布参数）
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            # 机器人状态投影层：state_dim -> dim_model
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            # 动作序列投影层：(chunk_size, action_dim) -> (chunk_size, dim_model)
            # action_feature: PolicyFeature(type=<FeatureType.ACTION: 'ACTION'>, shape=(2,)) (2,)
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            # VAE 编码器输出投影层：dim_model -> latent_dim * 2 (mu + log_sigma)
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            # VAE 编码器固定正弦位置编码
            # num_input_token_encoder = 1 (cls) + chunk_size (actions) [+ 1 (robot_state)]
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        # 图像特征提取骨干网络（ResNet18）
        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            # 注意：这里假设使用 ResNet 模型（layer4 是最终特征图）
            # forward 方法返回字典：{"feature_map": output}
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        # Transformer（训练时使用变分目标时作为 VAE 解码器）
        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        # Transformer 编码器输入投影层
        # 令牌结构：[latent, (robot_state), (env_state), (image_feature_map_pixels)]
        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        # 潜在变量投影层：latent_dim -> dim_model
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        # 图像特征投影层（1x1 卷积）：backbone_feature_dim -> dim_model
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        # Transformer 编码器位置编码
        n_1d_tokens = 1  # latent 令牌
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        # 图像特征 2D 位置编码
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        # Transformer 解码器
        # 可学习的位置编码（类似 DETR 的对象查询）
        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)

        # 动作回归头：dim_model -> action_dim
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier 均匀初始化 Transformer 参数（与原始代码一致）"""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """ACT 模型前向传播（包含可选的 VAE 编码器）

        维度说明（基于 lerobot/pusht 数据集）：
            - B (batch_size): 64 - 批次大小
            - S (chunk_size): 16 - 动作序列长度
            - D (dim_model): 512 - 模型隐藏层维度
            - latent_dim: 32 - VAE 潜在空间维度
            - action_dim: 2 - 动作维度
            - H, W: 3, 3 - ResNet18 骨干网络输出的图像特征图尺寸

        Args:
            batch: 包含观测和动作的字典，结构如下：
                {
                    "observation.state" (可选): (B, state_dim) 机器人状态
                    "observation.images.*": (B, n_cameras, C, H, W) 图像观测
                    "observation.env_state" (可选): (B, env_dim) 环境状态
                    "action" (训练时 VAE 模式): (B, S, action_dim) 动作序列
                    "action_is_pad": (B, S) 动作填充掩码
                }

        Returns:
            actions: (B, S, action_dim) = (64, 16, 2) 预测的动作序列
            (mu, log_sigma_x2): 潜在分布参数
                - mu: (B, latent_dim) = (64, 32) 潜在分布均值
                - log_sigma_x2: (B, latent_dim) = (64, 32) 潜在分布对数方差 (2log(σ))
        """
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]

        # ==================== VAE 编码器部分（仅训练时使用）====================
        if self.config.use_vae and ACTION in batch and self.training:
            # 准备 VAE 编码器输入：[cls, robot_state, action_sequence]
            # cls_embed: 类别令牌，用于提取潜在分布参数
            # (B, 1, D) = (64, 1, 512)
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )
            
            if self.config.robot_state_feature:
                # robot_state_embed: 机器人状态嵌入
                # batch[OBS_STATE]: (B, state_dim) = (64, 2)
                # robot_state_embed: (B, D) = (64, 512)
                # robot_state_embed.unsqueeze(1): (B, 1, D) = (64, 1, 512)
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)
            
            # action_embed: 动作序列嵌入
            # batch[ACTION]: (B, S, action_dim) = (64, 16, 2)
            # action_embed: (B, S, D) = (64, 16, 512)
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])

            # vae_encoder_input: VAE 编码器完整输入
            # 使用 robot_state 时：(B, S+2, D) = (64, 18, 512)，其中 2=cls+robot_state
            # 不使用 robot_state 时：(B, S+1, D) = (64, 17, 512)
            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)

            # pos_embed: 固定正弦位置编码
            # (1, S+2, D) = (1, 18, 512)
            # Note: detach() 理论上不需要，但保留以匹配原始代码
            pos_embed = self.vae_encoder_pos_enc.clone().detach()

            # key_padding_mask: 键填充掩码，用于屏蔽填充位置
            # cls_joint_is_pad: (B, 2) = (64, 2)，False 表示不是填充令牌
            # batch["action_is_pad"]: (B, S) = (64, 16)
            # key_padding_mask: (B, S+2) = (64, 18)
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )

            # VAE 编码器前向传播，获取潜在分布参数
            # vae_encoder_input.permute(1, 0, 2): (S+2, B, D) = (18, 64, 512) - Transformer 需要 (seq, batch, dim)
            # pos_embed.permute(1, 0, 2): (S+2, B, D) = (18, 64, 512)
            # cls_token_out: (B, D) = (64, 512) - 选择 cls 令牌输出
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]
            
            # latent_pdf_params: 潜在分布参数 (均值和对数方差)
            # (B, latent_dim*2) = (64, 64)
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            
            # mu: 潜在分布均值
            # (B, latent_dim) = (64, 32)
            mu = latent_pdf_params[:, : self.config.latent_dim]
            
            # log_sigma_x2: 对数方差 2log(σ)，这样实现是为了匹配原始代码
            # (B, latent_dim) = (64, 32)
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]

            # 使用重参数化技巧采样潜在变量：z = μ + σ * ε, 其中 ε ~ N(0, I)
            # log_sigma_x2.div(2).exp(): (B, latent_dim) = (64, 32) - σ = exp(log(σ²)/2)
            # latent_sample: (B, latent_dim) = (64, 32)
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
        else:
            # 不使用 VAE 编码器时，将潜在变量设为全零
            # (仅推理或训练时不使用 VAE)
            mu = log_sigma_x2 = None
            # TODO(rcadene, alexander-soare): remove call to `.to` to speedup forward ; precompute and use buffer
            # latent_sample: (B, latent_dim) = (64, 32)
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )

        # ==================== Transformer 编码器输入准备 ====================
        # encoder_in_tokens: 编码器输入令牌列表
        # encoder_latent_input_proj(latent_sample): (B, D) = (64, 512) - 潜在变量投影
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        
        # encoder_in_pos_embed: 编码器位置编码列表
        # self.encoder_1d_feature_pos_embed.weight: (n_1d_tokens, D)，n_1d_tokens=1(latent)+1(state)+1(env_state)
        # .unsqueeze(1): (n_1d_tokens, 1, D)
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        
        # Robot state token (如果使用)
        # batch[OBS_STATE]: (B, state_dim) = (64, 2)
        # encoder_robot_state_input_proj: (B, D) = (64, 512)
        if self.config.robot_state_feature:
            encoder_in_tokens.append(self.encoder_robot_state_input_proj(batch[OBS_STATE]))
        # Environment state token (如果使用)
        # batch[OBS_ENV_STATE]: (B, env_dim)
        # encoder_env_state_input_proj: (B, D) = (64, 512)
        if self.config.env_state_feature:
            encoder_in_tokens.append(self.encoder_env_state_input_proj(batch[OBS_ENV_STATE]))

        # ==================== 图像特征处理 ====================
        if self.config.image_features:
            # 对于图像列表，H 和 W 可能变化，但 H*W 是常数
            # NOTE: 如果修改此部分，请在 MPS 设备上验证梯度是否稳定（无爆炸或 NaN）
            for img in batch[OBS_IMAGES]:
                # cam_features: 骨干网络提取的图像特征
                # img: (B, C, H_in, W_in) = (64, 3, 96, 96)
                # backbone_feature_map: (B, C, H, W) = (64, 512, 3, 3) - ResNet18 layer4 输出
                cam_features = self.backbone(img)["feature_map"]
                
                # cam_pos_embed: 图像特征位置编码
                # (B, D, H, W) = (64, 512, 3, 3)
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                
                # 1x1 卷积将特征投影到模型维度
                # cam_features: (B, D, H, W) = (64, 512, 3, 3)
                cam_features = self.encoder_img_feat_input_proj(cam_features)

                # 重排特征为 Transformer 需要的格式 (sequence, batch, dim)
                # cam_features: (H*W, B, D) = (9, 64, 512)，其中 9=3*3
                # cam_pos_embed: (H*W, B, D) = (9, 64, 512)
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")

                # 立即扩展而不是累积后拼接
                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # ==================== 堆叠所有令牌 ====================
        # encoder_in_tokens: (N, B, D) - N 为总令牌数
        # N = 1(latent) + 1(state) + 1(env_state) + 9*cameras(image_tokens)
        # 对于 pusht (1 个相机，使用 state): N = 1 + 1 + 9 = 11
        # encoder_in_tokens: (11, 64, 512)
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        
        # encoder_in_pos_embed: (N, B, D) = (11, 64, 512)
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)

        # ==================== Transformer 编码器前向传播 ====================
        # encoder_out: 编码器输出
        # (N, B, D) = (11, 64, 512)
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        
        # ==================== Transformer 解码器部分 ====================
        # decoder_in: 解码器输入（全零令牌，类似 DETR 的对象查询）
        # (S, B, D) = (16, 64, 512)
        # TODO(rcadene, alexander-soare): remove call to `device` ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        
        # decoder_out: 解码器输出
        # decoder_pos_embed.weight: (S, D) = (16, 512) - 可学习的位置编码
        # .unsqueeze(1): (S, 1, D) = (16, 1, 512)
        # decoder_out: (S, B, D) = (16, 64, 512)
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )

        # 转置回 (B, S, D) 格式
        # decoder_out: (B, S, D) = (64, 16, 512)
        decoder_out = decoder_out.transpose(0, 1)

        # ==================== 动作回归头 ====================
        # actions: 最终预测的动作序列
        # action_head: Linear(D, action_dim) = Linear(512, 2)
        # actions: (B, S, action_dim) = (64, 16, 2)
        actions = self.action_head(decoder_out)

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """ACT 编码器：运行多个编码器层，后接归一化层

    维度说明：
        - N: 输入令牌序列长度（包括 latent、state、image tokens 等）
        - B: 批次大小 = 64
        - D: 模型隐藏层维度 = 512
    """

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        """
        Args:
            config: 策略配置类实例
            is_vae_encoder: 是否为 VAE 编码器（决定使用多少层）
        """
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        # 根据是否为 VAE 编码器选择层数
        # n_vae_encoder_layers: 4 (VAE 编码器)
        # n_encoder_layers: 4 (主编码器)
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        """ACT 编码器前向传播

        Args:
            x: 输入令牌序列
                - 形状：(N, B, D) = (11, 64, 512)
                - N: 令牌数量（1 latent + 1 state + 9 image tokens）
            pos_embed: 位置编码
                - 形状：(N, B, D) = (11, 64, 512)
            key_padding_mask: 键填充掩码（仅 VAE 编码器使用）
                - 形状：(B, N) = (64, 18) - 用于屏蔽填充位置

        Returns:
            x: 编码器输出
                - 形状：(N, B, D) = (11, 64, 512)
        """
        # 依次通过所有编码器层
        for layer in self.layers:
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
        # 应用层归一化
        x = self.norm(x)
        return x


class ACTEncoderLayer(nn.Module):
    """ACT 编码器层：包含自注意力和前馈网络

    维度说明：
        - N: 输入令牌序列长度
        - B: 批次大小 = 64
        - D: 模型隐藏层维度 = 512
        - dim_feedforward: 前馈网络中间层维度 = 2048
    """

    def __init__(self, config: ACTConfig):
        """
        Args:
            config: 策略配置类实例
        """
        super().__init__()
        # 多头自注意力层
        # dim_model: 512, n_heads: 8, 每头维度 = 512/8 = 64
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # 前馈网络：Linear(D, dim_feedforward) -> Dropout -> Activation -> Linear(dim_feedforward, D)
        # dim_feedforward: 2048
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # 层归一化
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm  # 是否使用 Pre-Norm 结构

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        """ACT 编码器层前向传播

        Args:
            x: 输入令牌序列
                - 形状：(N, B, D) = (11, 64, 512)
            pos_embed: 位置编码（可选）
                - 形状：(N, B, D) = (11, 64, 512)
            key_padding_mask: 键填充掩码（可选，用于屏蔽填充位置）
                - 形状：(B, N) = (64, 11)

        Returns:
            x: 编码器层输出
                - 形状：(N, B, D) = (11, 64, 512)
        """
        # 保存残差连接
        skip = x  # (N, B, D)
        
        if self.pre_norm:
            # Pre-Norm 结构：先归一化再计算注意力
            x = self.norm1(x)
        
        # 自注意力计算
        # q, k: 查询和键，添加位置编码
        # q, k, value: (N, B, D) = (11, 64, 512)
        q = k = x if pos_embed is None else x + pos_embed
        # self_attn 返回 (output, attention_weights)
        # output: (N, B, D) = (11, 64, 512)
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # 只选择输出，忽略注意力权重
        
        # 残差连接 + Dropout
        # x: (N, B, D) = (11, 64, 512)
        x = skip + self.dropout1(x)
        
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        
        # 前馈网络
        # linear1: (N, B, D) -> (N, B, dim_feedforward) = (11, 64, 2048)
        # linear2: (N, B, dim_feedforward) -> (N, B, D) = (11, 64, 512)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        # 残差连接 + Dropout
        x = skip + self.dropout2(x)
        
        if not self.pre_norm:
            x = self.norm2(x)
        
        return x


class ACTDecoder(nn.Module):
    """ACT 解码器：运行多个解码器层，后接归一化层

    维度说明：
        - S: 解码器序列长度（chunk_size）= 16
        - N: 编码器序列长度 = 11
        - B: 批次大小 = 64
        - D: 模型隐藏层维度 = 512
    """

    def __init__(self, config: ACTConfig):
        """
        Args:
            config: 策略配置类实例
        """
        super().__init__()
        # n_decoder_layers: 1 (解码器层数)
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """ACT 解码器前向传播

        Args:
            x: 解码器输入令牌（全零令牌，类似 DETR 的对象查询）
                - 形状：(S, B, D) = (16, 64, 512)
            encoder_out: 编码器输出特征
                - 形状：(N, B, D) = (11, 64, 512)
            decoder_pos_embed: 解码器位置编码
                - 形状：(S, 1, D) = (16, 1, 512)
            encoder_pos_embed: 编码器位置编码
                - 形状：(N, 1, D) = (11, 1, 512)

        Returns:
            x: 解码器输出
                - 形状：(S, B, D) = (16, 64, 512)
        """
        # 依次通过所有解码器层
        for layer in self.layers:
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
        # 应用层归一化
        if self.norm is not None:
            x = self.norm(x)
        return x


class ACTDecoderLayer(nn.Module):
    """ACT 解码器层：包含自注意力、交叉注意力和前馈网络

    维度说明：
        - S: 解码器序列长度（chunk_size）= 16
        - N: 编码器序列长度 = 11
        - B: 批次大小 = 64
        - D: 模型隐藏层维度 = 512
        - dim_feedforward: 前馈网络中间层维度 = 2048
    """

    def __init__(self, config: ACTConfig):
        """
        Args:
            config: 策略配置类实例
        """
        super().__init__()
        # 多头自注意力层（用于解码器内部令牌间注意力）
        # dim_model: 512, n_heads: 8
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        
        # 多头交叉注意力层（用于解码器对编码器输出的注意力）
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # 前馈网络：Linear(D, dim_feedforward) -> Dropout -> Activation -> Linear(dim_feedforward, D)
        # dim_feedforward: 2048
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        # 层归一化（3 个：自注意力后、交叉注意力后、前馈网络后）
        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm  # 是否使用 Pre-Norm 结构

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        """可选地添加位置编码到张量

        Args:
            tensor: 输入张量 (S, B, D) 或 (N, B, D)
            pos_embed: 位置编码（可选）

        Returns:
            tensor + pos_embed（如果 pos_embed 不为 None），否则返回 tensor
        """
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """ACT 解码器层前向传播

        Args:
            x: 解码器输入令牌
                - 形状：(S, B, D) = (16, 64, 512)
                - S: 解码器序列长度（chunk_size）
            encoder_out: 编码器输出特征
                - 形状：(N, B, D) = (11, 64, 512)
                - N: 编码器序列长度（1 latent + 1 state + 9 image tokens）
            decoder_pos_embed: 解码器位置编码（用于查询和键）
                - 形状：(S, 1, D) = (16, 1, 512)
            encoder_pos_embed: 编码器位置编码（用于交叉注意力的键）
                - 形状：(N, 1, D) = (11, 1, 512)

        Returns:
            x: 解码器层输出
                - 形状：(S, B, D) = (16, 64, 512)
        """
        # 保存残差连接
        skip = x  # (S, B, D)
        
        if self.pre_norm:
            # Pre-Norm 结构：先归一化再计算注意力
            x = self.norm1(x)
        
        # 自注意力计算（解码器内部令牌间注意力）
        # q, k: 添加解码器位置编码
        # q, k, value: (S, B, D) = (16, 64, 512)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        # self_attn 返回 (output, attention_weights)
        # output: (S, B, D) = (16, 64, 512)
        x = self.self_attn(q, k, value=x)[0]  # 只选择输出，忽略注意力权重
        
        # 残差连接 + Dropout
        # x: (S, B, D) = (16, 64, 512)
        x = skip + self.dropout1(x)
        
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        
        # 交叉注意力计算（解码器查询编码器输出）
        # query: 添加解码器位置编码 (S, B, D) = (16, 64, 512)
        # key, value: 添加编码器位置编码 (N, B, D) = (11, 64, 512)
        # output: (S, B, D) = (16, 64, 512)
        x = self.multihead_attn(
            query=self.maybe_add_pos_embed(x, decoder_pos_embed),
            key=self.maybe_add_pos_embed(encoder_out, encoder_pos_embed),
            value=encoder_out,
        )[0]  # 只选择输出，忽略注意力权重
        
        # 残差连接 + Dropout
        x = skip + self.dropout2(x)
        
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        
        # 前馈网络
        # linear1: (S, B, D) -> (S, B, dim_feedforward) = (16, 64, 2048)
        # linear2: (S, B, dim_feedforward) -> (S, B, D) = (16, 64, 512)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        
        # 残差连接 + Dropout
        x = skip + self.dropout3(x)
        
        if not self.pre_norm:
            x = self.norm3(x)
        
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D 正弦位置编码（与 Attention Is All You Need 中相同）

    Args:
        num_positions: 需要的令牌位置数量
            - 对于 VAE 编码器：S+2 = 18（1 cls + 1 robot_state + 16 actions）
        dimension: 位置编码的维度
            - dim_model = 512

    Returns:
        位置编码张量
            - 形状：(num_positions, dimension) = (18, 512)
    """

    def get_position_angle_vec(position):
        """计算单个位置的角度向量"""
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    # 生成正弦表
    # sinusoid_table: (num_positions, dimension)
    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    
    # 偶数维度应用 sin，奇数维度应用 cos
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    
    return torch.from_numpy(sinusoid_table).float()


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D 正弦位置编码（与 Attention Is All You Need 中类似）

    变体：位置索引归一化到 [0, 2π] 范围
    （严格来说下界是 1/H 对于垂直方向，1/W 对于水平方向）

    维度说明：
        - B: 批次大小 = 64
        - C: 通道数 = 512
        - H, W: 特征图尺寸 = 3, 3（ResNet18 layer4 输出）
    """

    def __init__(self, dimension: int):
        """
        Args:
            dimension: 期望的位置编码维度
                - dim_model // 2 = 256（因为 x 和 y 方向各占一半）
        """
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        # 正弦频率几何级数的逆"公比"
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """为 2D 特征图生成正弦位置编码

        Args:
            x: 2D 特征图批次
                - 形状：(B, C, H, W) = (64, 512, 3, 3)

        Returns:
            对应的正弦位置编码
                - 形状：(1, C, H, W) = (1, 512, 3, 3)
        """
        # not_mask: 全 1 掩码，用于生成位置索引
        # (1, H, W) = (1, 3, 3)
        not_mask = torch.ones_like(x[0, :1])
        
        # Note: 这些类似于 range(1, H+1) 和 range(1, W+1)，但在大多数实现中
        # 它们会是 range(0, H) 和 range(0, W)。保持原样以匹配原始代码。
        
        # y_range: 垂直方向位置索引
        # (1, H, W) = (1, 3, 3)
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        
        # x_range: 水平方向位置索引
        # (1, H, W) = (1, 3, 3)
        x_range = not_mask.cumsum(2, dtype=torch.float32)

        # "归一化"位置索引使其范围在 [0, 2π]
        # Note: 分母上加 epsilon 理论上不需要，因为 y_embed 和 x_range 的所有值
        # 通过构造都是非零的。这是原始代码的遗留。
        
        # y_range: (1, H, W) = (1, 3, 3)，范围 [2π/H, 2π]
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        
        # x_range: (1, H, W) = (1, 3, 3)，范围 [2π/W, 2π]
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi

        # inverse_frequency: 逆频率，用于生成不同频率的正弦/余弦
        # (dimension,) = (256,)
        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )

        # x_range: (1, H, W, dimension) = (1, 3, 3, 256)
        x_range = x_range.unsqueeze(-1) / inverse_frequency
        
        # y_range: (1, H, W, dimension) = (1, 3, 3, 256)
        y_range = y_range.unsqueeze(-1) / inverse_frequency

        # Note: 这个 stack 然后 flatten 操作生成交错的正弦和余弦项
        # pos_embed_x 和 pos_embed_y: (1, H, W, dimension // 2) = (1, 3, 3, 128)
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        
        # 拼接 x 和 y 方向的位置编码并调整维度顺序
        # pos_embed: (1, dimension*2, H, W) = (1, 512, 3, 3)
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """根据字符串返回激活函数

    Args:
        activation: 激活函数名称 ("relu", "gelu", 或 "glu")

    Returns:
        对应的激活函数
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
