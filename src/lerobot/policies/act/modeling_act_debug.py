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
"""Action Chunking Transformer Policy - 调试版本，用于记录所有关键变量的维度信息"""

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

# ============ 日志记录工具函数 ============
DEBUG_LOG_FILE = "/home/gyh/Workspace/lerobot/outputs/train/example_pusht_act/debug_dimensions.log"

def log_tensor(name: str, tensor: Tensor, description: str = ""):
    """记录张量的维度信息"""
    if tensor is None:
        return
    shape_str = " × ".join(str(d) for d in tensor.shape)
    with open(DEBUG_LOG_FILE, "a") as f:
        f.write(f"{name}: shape={tuple(tensor.shape)} ({shape_str}) - {description}\n")

def log_value(name: str, value, description: str = ""):
    """记录标量或其他值"""
    with open(DEBUG_LOG_FILE, "a") as f:
        f.write(f"{name}: {value} - {description}\n")

def clear_log_file():
    """清空日志文件"""
    with open(DEBUG_LOG_FILE, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("ACT Policy 维度调试日志\n")
        f.write("=" * 80 + "\n\n")


class ACTPolicy(PreTrainedPolicy):
    """
    Action Chunking Transformer Policy as per Learning Fine-Grained Bimanual Manipulation with Low-Cost
    Hardware (paper: https://huggingface.co/papers/2304.13705, code: https://github.com/tonyzhaozh/act)
    """

    config_class = ACTConfig
    name = "act"

    def __init__(
        self,
        config: ACTConfig,
    ):
        """
        Args:
            config: Policy configuration class instance or None, in which case the default instantiation of
                    the configuration class is used.
        """
        super().__init__(config)
        config.validate_features()
        self.config = config

        self.model = ACT(config)

        if config.temporal_ensemble_coeff is not None:
            self.temporal_ensembler = ACTTemporalEnsembler(config.temporal_ensemble_coeff, config.chunk_size)

        self.reset()

    def get_optim_params(self) -> dict:
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
        """Select a single action given environment observations."""
        self.eval()

        if self.config.temporal_ensemble_coeff is not None:
            actions = self.predict_action_chunk(batch)
            action = self.temporal_ensembler.update(actions)
            return action

        if len(self._action_queue) == 0:
            actions = self.predict_action_chunk(batch)[:, : self.config.n_action_steps]
            self._action_queue.extend(actions.transpose(0, 1))
        return self._action_queue.popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()

        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions = self.model(batch)[0]
        return actions

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict]:
        """Run the batch through the model and compute the loss for training or validation."""
        if self.config.image_features:
            batch = dict(batch)
            batch[OBS_IMAGES] = [batch[key] for key in self.config.image_features]

        actions_hat, (mu_hat, log_sigma_x2_hat) = self.model(batch)

        l1_loss = (
            F.l1_loss(batch[ACTION], actions_hat, reduction="none") * ~batch["action_is_pad"].unsqueeze(-1)
        ).mean()

        loss_dict = {"l1_loss": l1_loss.item()}
        if self.config.use_vae:
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
        self.chunk_size = chunk_size
        self.ensemble_weights = torch.exp(-temporal_ensemble_coeff * torch.arange(chunk_size))
        self.ensemble_weights_cumsum = torch.cumsum(self.ensemble_weights, dim=0)
        self.reset()

    def reset(self):
        """Resets the online computation variables."""
        self.ensembled_actions = None
        self.ensembled_actions_count = None

    def update(self, actions: Tensor) -> Tensor:
        """
        Takes a (batch, chunk_size, action_dim) sequence of actions, update the temporal ensemble for all
        time steps, and pop/return the next batch of actions in the sequence.
        """
        self.ensemble_weights = self.ensemble_weights.to(device=actions.device)
        self.ensemble_weights_cumsum = self.ensemble_weights_cumsum.to(device=actions.device)
        if self.ensembled_actions is None:
            self.ensembled_actions = actions.clone()
            self.ensembled_actions_count = torch.ones(
                (self.chunk_size, 1), dtype=torch.long, device=self.ensembled_actions.device
            )
        else:
            self.ensembled_actions *= self.ensemble_weights_cumsum[self.ensembled_actions_count - 1]
            self.ensembled_actions += actions[:, :-1] * self.ensemble_weights[self.ensembled_actions_count]
            self.ensembled_actions /= self.ensemble_weights_cumsum[self.ensembled_actions_count]
            self.ensembled_actions_count = torch.clamp(self.ensembled_actions_count + 1, max=self.chunk_size)
            self.ensembled_actions = torch.cat([self.ensembled_actions, actions[:, -1:]], dim=1)
            self.ensembled_actions_count = torch.cat(
                [self.ensembled_actions_count, torch.ones_like(self.ensembled_actions_count[-1:])]
            )
        action, self.ensembled_actions, self.ensembled_actions_count = (
            self.ensembled_actions[:, 0],
            self.ensembled_actions[:, 1:],
            self.ensembled_actions_count[1:],
        )
        return action


class ACT(nn.Module):
    """Action Chunking Transformer: The underlying neural network for ACTPolicy."""

    def __init__(self, config: ACTConfig):
        super().__init__()
        self.config = config

        if self.config.use_vae:
            self.vae_encoder = ACTEncoder(config, is_vae_encoder=True)
            self.vae_encoder_cls_embed = nn.Embedding(1, config.dim_model)
            if self.config.robot_state_feature:
                self.vae_encoder_robot_state_input_proj = nn.Linear(
                    self.config.robot_state_feature.shape[0], config.dim_model
                )
            self.vae_encoder_action_input_proj = nn.Linear(
                self.config.action_feature.shape[0],
                config.dim_model,
            )
            self.vae_encoder_latent_output_proj = nn.Linear(config.dim_model, config.latent_dim * 2)
            num_input_token_encoder = 1 + config.chunk_size
            if self.config.robot_state_feature:
                num_input_token_encoder += 1
            self.register_buffer(
                "vae_encoder_pos_enc",
                create_sinusoidal_pos_embedding(num_input_token_encoder, config.dim_model).unsqueeze(0),
            )

        if self.config.image_features:
            backbone_model = getattr(torchvision.models, config.vision_backbone)(
                replace_stride_with_dilation=[False, False, config.replace_final_stride_with_dilation],
                weights=config.pretrained_backbone_weights,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(backbone_model, return_layers={"layer4": "feature_map"})

        self.encoder = ACTEncoder(config)
        self.decoder = ACTDecoder(config)

        if self.config.robot_state_feature:
            self.encoder_robot_state_input_proj = nn.Linear(
                self.config.robot_state_feature.shape[0], config.dim_model
            )
        if self.config.env_state_feature:
            self.encoder_env_state_input_proj = nn.Linear(
                self.config.env_state_feature.shape[0], config.dim_model
            )
        self.encoder_latent_input_proj = nn.Linear(config.latent_dim, config.dim_model)
        if self.config.image_features:
            self.encoder_img_feat_input_proj = nn.Conv2d(
                backbone_model.fc.in_features, config.dim_model, kernel_size=1
            )
        n_1d_tokens = 1
        if self.config.robot_state_feature:
            n_1d_tokens += 1
        if self.config.env_state_feature:
            n_1d_tokens += 1
        self.encoder_1d_feature_pos_embed = nn.Embedding(n_1d_tokens, config.dim_model)
        if self.config.image_features:
            self.encoder_cam_feat_pos_embed = ACTSinusoidalPositionEmbedding2d(config.dim_model // 2)

        self.decoder_pos_embed = nn.Embedding(config.chunk_size, config.dim_model)
        self.action_head = nn.Linear(config.dim_model, self.config.action_feature.shape[0])

        self._reset_parameters()

    def _reset_parameters(self):
        """Xavier-uniform initialization of the transformer parameters as in the original code."""
        for p in chain(self.encoder.parameters(), self.decoder.parameters()):
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, batch: dict[str, Tensor]) -> tuple[Tensor, tuple[Tensor, Tensor] | tuple[None, None]]:
        """A forward pass through the Action Chunking Transformer (with optional VAE encoder)."""
        if self.config.use_vae and self.training:
            assert ACTION in batch, (
                "actions must be provided when using the variational objective in training mode."
            )

        if OBS_IMAGES in batch:
            batch_size = batch[OBS_IMAGES][0].shape[0]
        else:
            batch_size = batch[OBS_ENV_STATE].shape[0]

        log_value("batch_size", batch_size, "批次大小")

        # Prepare the latent for input to the transformer encoder.
        if self.config.use_vae and ACTION in batch and self.training:
            # Prepare the input to the VAE encoder: [cls, *joint_space_configuration, *action_sequence].
            cls_embed = einops.repeat(
                self.vae_encoder_cls_embed.weight, "1 d -> b 1 d", b=batch_size
            )  # (B, 1, D)
            log_tensor("cls_embed", cls_embed, "类别 token 嵌入 (批次大小 × 1 × 模型维度)")
            
            if self.config.robot_state_feature:
                robot_state_embed = self.vae_encoder_robot_state_input_proj(batch[OBS_STATE])
                robot_state_embed = robot_state_embed.unsqueeze(1)  # (B, 1, D)
                log_tensor("robot_state_embed", robot_state_embed, "机器人状态嵌入 (批次大小 × 1 × 模型维度)")
            
            action_embed = self.vae_encoder_action_input_proj(batch[ACTION])  # (B, S, D)
            log_tensor("action_embed", action_embed, "动作嵌入 (批次大小 × 序列长度 × 模型维度)")

            if self.config.robot_state_feature:
                vae_encoder_input = [cls_embed, robot_state_embed, action_embed]  # (B, S+2, D)
            else:
                vae_encoder_input = [cls_embed, action_embed]
            vae_encoder_input = torch.cat(vae_encoder_input, axis=1)
            log_tensor("vae_encoder_input", vae_encoder_input, "VAE 编码器输入 (批次大小 × 总序列长度 × 模型维度)")

            # Prepare fixed positional embedding.
            pos_embed = self.vae_encoder_pos_enc.clone().detach()  # (1, S+2, D)
            log_tensor("vae_encoder_pos_embed", pos_embed, "VAE 编码器位置嵌入 (1 × 总序列长度 × 模型维度)")

            # Prepare key padding mask for the transformer encoder.
            cls_joint_is_pad = torch.full(
                (batch_size, 2 if self.config.robot_state_feature else 1),
                False,
                device=batch[OBS_STATE].device,
            )
            log_tensor("cls_joint_is_pad", cls_joint_is_pad, "类别和关节填充掩码 (批次大小 × 额外 token 数)")
            
            key_padding_mask = torch.cat(
                [cls_joint_is_pad, batch["action_is_pad"]], axis=1
            )  # (bs, seq+1 or 2)
            log_tensor("key_padding_mask", key_padding_mask, "键填充掩码 (批次大小 × 总序列长度)")

            # Forward pass through VAE encoder to get the latent PDF parameters.
            cls_token_out = self.vae_encoder(
                vae_encoder_input.permute(1, 0, 2),
                pos_embed=pos_embed.permute(1, 0, 2),
                key_padding_mask=key_padding_mask,
            )[0]  # select the class token, with shape (B, D)
            log_tensor("cls_token_out", cls_token_out, "类别 token 输出 (批次大小 × 模型维度)")
            
            latent_pdf_params = self.vae_encoder_latent_output_proj(cls_token_out)
            log_tensor("latent_pdf_params", latent_pdf_params, "潜在分布参数 (批次大小 × 2×潜在维度)")
            
            mu = latent_pdf_params[:, : self.config.latent_dim]
            log_tensor("mu", mu, "潜在分布均值 (批次大小 × 潜在维度)")
            
            # This is 2log(sigma). Done this way to match the original implementation.
            log_sigma_x2 = latent_pdf_params[:, self.config.latent_dim :]
            log_tensor("log_sigma_x2", log_sigma_x2, "潜在分布对数方差 (批次大小 × 潜在维度)")

            # Sample the latent with the reparameterization trick.
            latent_sample = mu + log_sigma_x2.div(2).exp() * torch.randn_like(mu)
            log_tensor("latent_sample", latent_sample, "重参数化采样的潜在变量 (批次大小 × 潜在维度)")
        else:
            # When not using the VAE encoder, we set the latent to be all zeros.
            mu = log_sigma_x2 = None
            latent_sample = torch.zeros([batch_size, self.config.latent_dim], dtype=torch.float32).to(
                batch[OBS_STATE].device
            )
            log_tensor("latent_sample", latent_sample, "潜在变量 (全零，批次大小 × 潜在维度)")

        # Prepare transformer encoder inputs.
        encoder_in_tokens = [self.encoder_latent_input_proj(latent_sample)]
        log_tensor("encoder_latent_token", encoder_in_tokens[0], "编码器潜在 token (批次大小 × 模型维度)")
        
        encoder_in_pos_embed = list(self.encoder_1d_feature_pos_embed.weight.unsqueeze(1))
        
        # Robot state token.
        if self.config.robot_state_feature:
            robot_state_token = self.encoder_robot_state_input_proj(batch[OBS_STATE])
            encoder_in_tokens.append(robot_state_token)
            log_tensor("encoder_robot_state_token", robot_state_token, "编码器机器人状态 token (批次大小 × 模型维度)")
        
        # Environment state token.
        if self.config.env_state_feature:
            env_state_token = self.encoder_env_state_input_proj(batch[OBS_ENV_STATE])
            encoder_in_tokens.append(env_state_token)
            log_tensor("encoder_env_state_token", env_state_token, "编码器环境状态 token (批次大小 × 模型维度)")

        if self.config.image_features:
            for img in batch[OBS_IMAGES]:
                cam_features = self.backbone(img)["feature_map"]
                log_tensor("backbone_feature_map", cam_features, "骨干网络特征图 (批次大小 × 通道 × 高 × 宽)")
                
                cam_pos_embed = self.encoder_cam_feat_pos_embed(cam_features).to(dtype=cam_features.dtype)
                log_tensor("cam_pos_embed", cam_pos_embed, "相机位置嵌入 (批次大小 × 模型维度/2 × 高 × 宽)")
                
                cam_features = self.encoder_img_feat_input_proj(cam_features)
                log_tensor("cam_features_proj", cam_features, "投影后的相机特征 (批次大小 × 模型维度 × 高 × 宽)")

                # Rearrange features to (sequence, batch, dim).
                cam_features = einops.rearrange(cam_features, "b c h w -> (h w) b c")
                log_tensor("cam_features_rearranged", cam_features, "重排的相机特征 (序列长度 × 批次大小 × 模型维度)")
                
                cam_pos_embed = einops.rearrange(cam_pos_embed, "b c h w -> (h w) b c")
                log_tensor("cam_pos_embed_rearranged", cam_pos_embed, "重排的位置嵌入 (序列长度 × 批次大小 × 模型维度)")

                encoder_in_tokens.extend(list(cam_features))
                encoder_in_pos_embed.extend(list(cam_pos_embed))

        # Stack all tokens along the sequence dimension.
        encoder_in_tokens = torch.stack(encoder_in_tokens, axis=0)
        log_tensor("encoder_in_tokens", encoder_in_tokens, "编码器输入 token (总序列长度 × 批次大小 × 模型维度)")
        
        encoder_in_pos_embed = torch.stack(encoder_in_pos_embed, axis=0)
        log_tensor("encoder_in_pos_embed", encoder_in_pos_embed, "编码器位置嵌入 (总序列长度 × 1 × 模型维度)")

        # Forward pass through the transformer modules.
        encoder_out = self.encoder(encoder_in_tokens, pos_embed=encoder_in_pos_embed)
        log_tensor("encoder_out", encoder_out, "编码器输出 (总序列长度 × 批次大小 × 模型维度)")
        
        # TODO(rcadene, alexander-soare): remove call to `device` to speedup forward ; precompute and use buffer
        decoder_in = torch.zeros(
            (self.config.chunk_size, batch_size, self.config.dim_model),
            dtype=encoder_in_pos_embed.dtype,
            device=encoder_in_pos_embed.device,
        )
        log_tensor("decoder_in", decoder_in, "解码器输入 (动作序列长度 × 批次大小 × 模型维度)")
        
        decoder_out = self.decoder(
            decoder_in,
            encoder_out,
            encoder_pos_embed=encoder_in_pos_embed,
            decoder_pos_embed=self.decoder_pos_embed.weight.unsqueeze(1),
        )
        log_tensor("decoder_out", decoder_out, "解码器输出 (动作序列长度 × 批次大小 × 模型维度)")

        # Move back to (B, S, C).
        decoder_out = decoder_out.transpose(0, 1)
        log_tensor("decoder_out_transposed", decoder_out, "转置后的解码器输出 (批次大小 × 动作序列长度 × 模型维度)")

        actions = self.action_head(decoder_out)
        log_tensor("actions", actions, "预测的动作 (批次大小 × 动作序列长度 × 动作维度)")

        return actions, (mu, log_sigma_x2)


class ACTEncoder(nn.Module):
    """Convenience module for running multiple encoder layers, maybe followed by normalization."""

    def __init__(self, config: ACTConfig, is_vae_encoder: bool = False):
        super().__init__()
        self.is_vae_encoder = is_vae_encoder
        num_layers = config.n_vae_encoder_layers if self.is_vae_encoder else config.n_encoder_layers
        self.layers = nn.ModuleList([ACTEncoderLayer(config) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(config.dim_model) if config.pre_norm else nn.Identity()

    def forward(
        self, x: Tensor, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None
    ) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x, pos_embed=pos_embed, key_padding_mask=key_padding_mask)
            log_tensor(f"encoder_layer_{i}_out", x, f"编码器第{i}层输出 (序列长度 × 批次大小 × 模型维度)")
        x = self.norm(x)
        log_tensor("encoder_norm_out", x, "编码器归一化输出 (序列长度 × 批次大小 × 模型维度)")
        return x


class ACTEncoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def forward(self, x, pos_embed: Tensor | None = None, key_padding_mask: Tensor | None = None) -> Tensor:
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = x if pos_embed is None else x + pos_embed
        log_tensor("q", q, "查询张量 (序列长度 × 批次大小 × 模型维度)")
        log_tensor("k", k, "键张量 (序列长度 × 批次大小 × 模型维度)")
        
        x = self.self_attn(q, k, value=x, key_padding_mask=key_padding_mask)
        x = x[0]  # note: [0] to select just the output, not the attention weights
        log_tensor("self_attn_out", x, "自注意力输出 (序列长度 × 批次大小 × 模型维度)")
        
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        log_tensor("ffn_out", x, "前馈网络输出 (序列长度 × 批次大小 × 模型维度)")
        
        x = skip + self.dropout2(x)
        if not self.pre_norm:
            x = self.norm2(x)
        return x


class ACTDecoder(nn.Module):
    def __init__(self, config: ACTConfig):
        """Convenience module for running multiple decoder layers followed by normalization."""
        super().__init__()
        self.layers = nn.ModuleList([ACTDecoderLayer(config) for _ in range(config.n_decoder_layers)])
        self.norm = nn.LayerNorm(config.dim_model)

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(
                x, encoder_out, decoder_pos_embed=decoder_pos_embed, encoder_pos_embed=encoder_pos_embed
            )
            log_tensor(f"decoder_layer_{i}_out", x, f"解码器第{i}层输出 (动作序列长度 × 批次大小 × 模型维度)")
        if self.norm is not None:
            x = self.norm(x)
            log_tensor("decoder_norm_out", x, "解码器归一化输出 (动作序列长度 × 批次大小 × 模型维度)")
        return x


class ACTDecoderLayer(nn.Module):
    def __init__(self, config: ACTConfig):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)
        self.multihead_attn = nn.MultiheadAttention(config.dim_model, config.n_heads, dropout=config.dropout)

        # Feed forward layers.
        self.linear1 = nn.Linear(config.dim_model, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.dim_model)

        self.norm1 = nn.LayerNorm(config.dim_model)
        self.norm2 = nn.LayerNorm(config.dim_model)
        self.norm3 = nn.LayerNorm(config.dim_model)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        self.dropout3 = nn.Dropout(config.dropout)

        self.activation = get_activation_fn(config.feedforward_activation)
        self.pre_norm = config.pre_norm

    def maybe_add_pos_embed(self, tensor: Tensor, pos_embed: Tensor | None) -> Tensor:
        return tensor if pos_embed is None else tensor + pos_embed

    def forward(
        self,
        x: Tensor,
        encoder_out: Tensor,
        decoder_pos_embed: Tensor | None = None,
        encoder_pos_embed: Tensor | None = None,
    ) -> Tensor:
        """
        Args:
            x: (Decoder Sequence, Batch, Channel) tensor of input tokens.
            encoder_out: (Encoder Sequence, B, C) output features from the last layer of the encoder we are
                cross-attending with.
            decoder_pos_embed: (ES, 1, C) positional embedding for keys (from the encoder).
            encoder_pos_embed: (DS, 1, C) Positional_embedding for the queries (from the decoder).
        Returns:
            (DS, B, C) tensor of decoder output features.
        """
        skip = x
        if self.pre_norm:
            x = self.norm1(x)
        q = k = self.maybe_add_pos_embed(x, decoder_pos_embed)
        log_tensor("decoder_q", q, "解码器查询张量 (动作序列长度 × 批次大小 × 模型维度)")
        
        x = self.self_attn(q, k, value=x)[0]  # select just the output, not the attention weights
        log_tensor("decoder_self_attn_out", x, "解码器自注意力输出 (动作序列长度 × 批次大小 × 模型维度)")
        
        x = skip + self.dropout1(x)
        if self.pre_norm:
            skip = x
            x = self.norm2(x)
        else:
            x = self.norm1(x)
            skip = x
        
        query = self.maybe_add_pos_embed(x, decoder_pos_embed)
        key = self.maybe_add_pos_embed(encoder_out, encoder_pos_embed)
        log_tensor("cross_attn_query", query, "交叉注意力查询 (动作序列长度 × 批次大小 × 模型维度)")
        log_tensor("cross_attn_key", key, "交叉注意力键 (总序列长度 × 批次大小 × 模型维度)")
        log_tensor("cross_attn_value", encoder_out, "交叉注意力值 (总序列长度 × 批次大小 × 模型维度)")
        
        x = self.multihead_attn(
            query=query,
            key=key,
            value=encoder_out,
        )[0]  # select just the output, not the attention weights
        log_tensor("cross_attn_out", x, "交叉注意力输出 (动作序列长度 × 批次大小 × 模型维度)")
        
        x = skip + self.dropout2(x)
        if self.pre_norm:
            skip = x
            x = self.norm3(x)
        else:
            x = self.norm2(x)
            skip = x
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        log_tensor("decoder_ffn_out", x, "解码器前馈网络输出 (动作序列长度 × 批次大小 × 模型维度)")
        
        x = skip + self.dropout3(x)
        if not self.pre_norm:
            x = self.norm3(x)
        return x


def create_sinusoidal_pos_embedding(num_positions: int, dimension: int) -> Tensor:
    """1D sinusoidal positional embeddings as in Attention is All You Need."""

    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / dimension) for hid_j in range(dimension)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(num_positions)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    result = torch.from_numpy(sinusoid_table).float()
    log_tensor("sinusoidal_pos_embedding", result, f"正弦位置嵌入 ({num_positions} × {dimension})")
    return result


class ACTSinusoidalPositionEmbedding2d(nn.Module):
    """2D sinusoidal positional embeddings similar to what's presented in Attention Is All You Need."""

    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension
        self._two_pi = 2 * math.pi
        self._eps = 1e-6
        self._temperature = 10000

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: A (B, C, H, W) batch of 2D feature map to generate the embeddings for.
        Returns:
            A (1, C, H, W) batch of corresponding sinusoidal positional embeddings.
        """
        not_mask = torch.ones_like(x[0, :1])  # (1, H, W)
        log_tensor("not_mask", not_mask, "非掩码 (1 × 高 × 宽)")
        
        y_range = not_mask.cumsum(1, dtype=torch.float32)
        x_range = not_mask.cumsum(2, dtype=torch.float32)
        log_tensor("y_range", y_range, "Y 范围 (1 × 高 × 宽)")
        log_tensor("x_range", x_range, "X 范围 (1 × 高 × 宽)")

        # "Normalize" the position index such that it ranges in [0, 2π].
        y_range = y_range / (y_range[:, -1:, :] + self._eps) * self._two_pi
        x_range = x_range / (x_range[:, :, -1:] + self._eps) * self._two_pi
        log_tensor("y_range_normalized", y_range, "归一化 Y 范围 (1 × 高 × 宽)")
        log_tensor("x_range_normalized", x_range, "归一化 X 范围 (1 × 高 × 宽)")

        inverse_frequency = self._temperature ** (
            2 * (torch.arange(self.dimension, dtype=torch.float32, device=x.device) // 2) / self.dimension
        )
        log_tensor("inverse_frequency", inverse_frequency, "逆频率 (维度/2,)")

        x_range = x_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        y_range = y_range.unsqueeze(-1) / inverse_frequency  # (1, H, W, 1)
        log_tensor("x_range_with_freq", x_range, "X 范围与频率 (1 × 高 × 宽 × 维度/2)")
        log_tensor("y_range_with_freq", y_range, "Y 范围与频率 (1 × 高 × 宽 × 维度/2)")

        # Note: this stack then flatten operation results in interleaved sine and cosine terms.
        pos_embed_x = torch.stack((x_range[..., 0::2].sin(), x_range[..., 1::2].cos()), dim=-1).flatten(3)
        pos_embed_y = torch.stack((y_range[..., 0::2].sin(), y_range[..., 1::2].cos()), dim=-1).flatten(3)
        log_tensor("pos_embed_x", pos_embed_x, "X 位置嵌入 (1 × 高 × 宽 × 维度/2)")
        log_tensor("pos_embed_y", pos_embed_y, "Y 位置嵌入 (1 × 高 × 宽 × 维度/2)")
        
        pos_embed = torch.cat((pos_embed_y, pos_embed_x), dim=3).permute(0, 3, 1, 2)  # (1, C, H, W)
        log_tensor("pos_embed_2d", pos_embed, "2D 位置嵌入 (1 × 维度 × 高 × 宽)")

        return pos_embed


def get_activation_fn(activation: str) -> Callable:
    """Return an activation function given a string."""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu/glu, not {activation}.")
