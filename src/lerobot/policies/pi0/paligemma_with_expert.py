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

"""
PaliGemma + Gemma Expert 关键张量速查表（默认 B=8, L_prefix≈512, L_suffix=51）
| 名称 | 含义 | 典型形状 |
| --- | --- | --- |
| inputs_embeds[0] | 图像+语言前缀嵌入 | [B, L_prefix, 2048] |
| inputs_embeds[1] | 状态+动作+时间后缀嵌入 | [B, L_suffix, 1024] |
| attention_mask | 合并后的注意力掩码 | [B, L_total, L_total] |
| query/key/value_states | RoPE 前后的注意力输入 | [B, L_total, H, D] |
| att_output | 合并后的注意力输出 | [B, L_total, H*D] |
"""

import logging  # [REVISE] Added for tensor-shape diagnostics

import torch
import torch.version

try:  # pytest 仅用于类型提示，缺失时使用简易兜底，避免运行期依赖
    from pytest import Cache
except ModuleNotFoundError:  # pragma: no cover - 测试环境通常已安装 pytest
    class Cache(dict):  # [REVISE] Provide minimal stand-in when pytest is absent
        """轻量级占位缓存，满足 forward 中的字典接口需求。"""

        pass
from torch import nn
from transformers import (
    AutoConfig,
    GemmaForCausalLM,
    PaliGemmaForConditionalGeneration,
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.models.auto import CONFIG_MAPPING

from lerobot.policies.pi0.flex_attention import flex_attention_forward


logger = logging.getLogger(__name__)  # [REVISE] Central logging handle reused across helpers


def _shape_repr(value):
    """中文格式化工具：把张量尺寸转换成可读列表。"""

    if value is None:
        return "None"
    if isinstance(value, (list, tuple)):
        shapes = []
        for item in value:
            if item is None:
                shapes.append("None")
            elif hasattr(item, "shape"):
                shapes.append(tuple(item.shape))
            else:
                shapes.append(str(item))
        return shapes
    if hasattr(value, "shape"):
        return tuple(value.shape)
    return str(value)


def _log_shapes(enabled: bool, title: str, **named_tensors):  # [REVISE] Emit human-readable tensor shapes when debugging
    """当 debug_shapes=True 时打印关键张量尺寸。"""

    if not enabled or not logger.isEnabledFor(logging.DEBUG):
        return
    parts = [f"{name}={_shape_repr(tensor)}" for name, tensor in named_tensors.items()]
    logger.debug("[%s] %s", title, ", ".join(parts))


def apply_rope(x, positions, max_wavelength=10_000):
    """
    Applies RoPE positions [B, L] to x [B, L, H, D].
    """
    d_half = x.shape[-1] // 2
    device = x.device
    dtype = x.dtype
    x = x.to(torch.float32)

    freq_exponents = (2.0 / x.shape[-1]) * torch.arange(d_half, dtype=torch.float32, device=device)
    timescale = max_wavelength**freq_exponents
    radians = positions[..., None].to(torch.float32) / timescale[None, None, :].to(torch.float32)

    radians = radians[..., None, :]

    sin = torch.sin(radians)  # .to(dtype=dtype)
    cos = torch.cos(radians)  # .to(dtype=dtype)

    x1, x2 = x.split(d_half, dim=-1)
    res = torch.empty_like(x)
    res[..., :d_half] = x1 * cos - x2 * sin
    res[..., d_half:] = x2 * cos + x1 * sin

    return res.to(dtype)


class PaliGemmaWithExpertConfig(PretrainedConfig):
    model_type = "PaliGemmaWithExpertModel"
    sub_configs = {"paligemma_config": AutoConfig, "gemma_expert_config": AutoConfig}

    def __init__(
        self,
        paligemma_config: dict | None = None,
        gemma_expert_config: dict | None = None,
        freeze_vision_encoder: bool = True,
        train_expert_only: bool = True,
        attention_implementation: str = "eager",
        debug_shapes: bool = False,  # [REVISE] Allow upstream configs to toggle verbose tensor logging
        **kwargs,
    ):
        self.freeze_vision_encoder = freeze_vision_encoder
        self.train_expert_only = train_expert_only
        self.attention_implementation = attention_implementation
        # 调试标志：用于控制张量尺寸日志
        self.debug_shapes = debug_shapes  # [REVISE] Persist toggle for downstream model usage

        if paligemma_config is None:
            # Default config from Pi0
            self.paligemma_config = CONFIG_MAPPING["paligemma"](
                transformers_version="4.48.1",
                _vocab_size=257152,
                bos_token_id=2,
                eos_token_id=1,
                hidden_size=2048,
                image_token_index=257152,
                model_type="paligemma",
                pad_token_id=0,
                projection_dim=2048,
                text_config={
                    "hidden_activation": "gelu_pytorch_tanh",
                    "hidden_size": 2048,
                    "intermediate_size": 16384,
                    "model_type": "gemma",
                    "num_attention_heads": 8,
                    "num_hidden_layers": 18,
                    "num_image_tokens": 256,
                    "num_key_value_heads": 1,
                    "torch_dtype": "float32",
                    "vocab_size": 257152,
                },
                vision_config={
                    "hidden_size": 1152,
                    "intermediate_size": 4304,
                    "model_type": "siglip_vision_model",
                    "num_attention_heads": 16,
                    "num_hidden_layers": 27,
                    "num_image_tokens": 256,
                    "patch_size": 14,
                    "projection_dim": 2048,
                    "projector_hidden_act": "gelu_fast",
                    "torch_dtype": "float32",
                    "vision_use_head": False,
                },
            )
        elif isinstance(self.paligemma_config, dict):
            # Override Pi0 default config for PaliGemma
            if "model_type" not in gemma_expert_config:
                paligemma_config["model_type"] = "paligemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.paligemma_config = cfg_cls(**paligemma_config)

        if gemma_expert_config is None:
            # Default config from Pi0
            self.gemma_expert_config = CONFIG_MAPPING["gemma"](
                attention_bias=False,
                attention_dropout=0.0,
                bos_token_id=2,
                eos_token_id=1,
                head_dim=256,
                hidden_act="gelu_pytorch_tanh",
                hidden_activation="gelu_pytorch_tanh",
                hidden_size=1024,
                initializer_range=0.02,
                intermediate_size=4096,
                max_position_embeddings=8192,
                model_type="gemma",
                num_attention_heads=8,
                num_hidden_layers=18,
                num_key_value_heads=1,
                pad_token_id=0,
                rms_norm_eps=1e-06,
                rope_theta=10000.0,
                torch_dtype="float32",
                transformers_version="4.48.1",
                use_cache=True,
                vocab_size=257152,
            )
        elif isinstance(self.gemma_expert_config, dict):
            # Override Pi0 default config for Gemma Expert
            if "model_type" not in gemma_expert_config:
                gemma_expert_config["model_type"] = "gemma"

            cfg_cls = CONFIG_MAPPING[paligemma_config["model_type"]]
            self.gemma_expert_config = cfg_cls(**gemma_expert_config)

        super().__init__(**kwargs)

    def __post_init__(self):
        super().__post_init__()
        if self.train_expert_only and not self.freeze_vision_encoder:
            raise ValueError(
                "You set `freeze_vision_encoder=False` and `train_expert_only=True` which are not compatible."
            )

        if self.attention_implementation not in ["eager", "fa2", "flex"]:
            raise ValueError(
                f"Wrong value provided for `attention_implementation` ({self.attention_implementation}). Expected 'eager', 'fa2' or 'flex'."
            )


class PaliGemmaWithExpertModel(PreTrainedModel):
    config_class = PaliGemmaWithExpertConfig

    def __init__(self, config: PaliGemmaWithExpertConfig):
        super().__init__(config=config)
        self.config = config
        self.paligemma = PaliGemmaForConditionalGeneration(config=config.paligemma_config)
        self.gemma_expert = GemmaForCausalLM(config=config.gemma_expert_config)
        # Remove unused embed_tokens
        self.gemma_expert.model.embed_tokens = None

        self.to_bfloat16_like_physical_intelligence()
        self.set_requires_grad()

    def _vision_backbone(self):  # [REVISE] Normalize access to vision tower across transformers versions
        """兼容不同 transformers 版本下 PaliGemma 视觉塔的属性命名。"""

        if hasattr(self.paligemma, "vision_tower") and self.paligemma.vision_tower is not None:
            return self.paligemma.vision_tower
        if hasattr(self.paligemma, "model"):
            model = self.paligemma.model
            if hasattr(model, "vision_tower") and model.vision_tower is not None:
                return model.vision_tower
            if hasattr(model, "vision_model") and model.vision_model is not None:
                return model.vision_model
        raise AttributeError("PaliGemma 模型缺少 vision_tower/vision_model 属性，请检查 transformers 版本。")

    def set_requires_grad(self):
        if self.config.freeze_vision_encoder:
            vision_module = self._vision_backbone()  # [REVISE] Reuse helper so attribute changes do not break finetuning
            vision_module.eval()
            for params in vision_module.parameters():
                params.requires_grad = False

        if self.config.train_expert_only:
            self.paligemma.eval()
            for params in self.paligemma.parameters():
                params.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)

        if self.config.freeze_vision_encoder:
            self._vision_backbone().eval()  # [REVISE] Ensure frozen tower stays eval-mode when training other parts

        if self.config.train_expert_only:
            self.paligemma.eval()

    def to_bfloat16_like_physical_intelligence(self):
        self.paligemma = self.paligemma.to(dtype=torch.bfloat16)

        params_to_change_dtype = [  # [REVISE] Ensure precision alignment across both vision+language blocks
            "language_model.model.layers",
            "gemma_expert.model.layers",
            "vision_tower",
            "vision_model",
            "multi_modal",
        ]
        for name, param in self.named_parameters():
            if any(selector in name for selector in params_to_change_dtype):
                param.data = param.data.to(dtype=torch.bfloat16)

    def embed_image(self, image: torch.Tensor):
        # Handle different transformers versions
        if hasattr(self.paligemma, "get_image_features"):
            emb = self.paligemma.get_image_features(image)
        else:
            emb = self.paligemma.model.get_image_features(image)

        _log_shapes(self.config.debug_shapes, "paligemma.embed_image", image=image, image_emb=emb)  # [REVISE] Record SigLIP embedding stats for troubleshooting
        return emb

    def embed_language_tokens(self, tokens: torch.Tensor):
        emb = self.paligemma.language_model.embed_tokens(tokens)
        _log_shapes(self.config.debug_shapes, "paligemma.embed_language", tokens=tokens, lang_emb=emb)  # [REVISE] Confirm token embeddings before fusion
        return emb

    # TODO: break down this huge forward into modules or functions
    def forward(
        self,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | Cache | None = None,
        inputs_embeds: list[torch.FloatTensor] = None,
        use_cache: bool | None = None,
        fill_kv_cache: bool | None = None,
    ):
        models = [self.paligemma.language_model, self.gemma_expert.model]

        for hidden_states in inputs_embeds:
            # TODO this is very inefficient
            # dtype is always the same, batch size too (if > 1 len)
            # device could be trickier in multi gpu edge cases but that's it
            if hidden_states is None:
                continue
            batch_size = hidden_states.shape[0]

        _log_shapes(
            self.config.debug_shapes,
            "paligemma.forward.inputs",
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
        )  # [REVISE] Snapshot fused prefix/suffix inputs entering joint transformer

        # RMSNorm
        num_layers = self.paligemma.config.text_config.num_hidden_layers
        head_dim = self.paligemma.config.text_config.head_dim
        for layer_idx in range(num_layers):
            query_states = []
            key_states = []
            value_states = []
            for i, hidden_states in enumerate(inputs_embeds):
                if hidden_states is None:
                    continue
                layer = models[i].layers[layer_idx]
                # normalizer = torch.tensor(models[i].config.hidden_size**0.5, dtype=hidden_states.dtype)
                # hidden_states = hidden_states * normalizer
                hidden_states = layer.input_layernorm(hidden_states)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, layer.self_attn.head_dim)

                hidden_states = hidden_states.to(dtype=torch.bfloat16)
                query_state = layer.self_attn.q_proj(hidden_states).view(hidden_shape)
                key_state = layer.self_attn.k_proj(hidden_states).view(hidden_shape)
                value_state = layer.self_attn.v_proj(hidden_states).view(hidden_shape)

                query_states.append(query_state)
                key_states.append(key_state)
                value_states.append(value_state)

            # B,L,H,D with L sequence length, H number of heads, D head dim
            # concatenate on the number of embeddings/tokens
            query_states = torch.cat(query_states, dim=1)
            key_states = torch.cat(key_states, dim=1)
            value_states = torch.cat(value_states, dim=1)

            if layer_idx == 0:
                # 只在首层打印一次 QKV 尺寸，避免日志过长
                _log_shapes(
                    self.config.debug_shapes,
                    "paligemma.forward.qkv",
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                )  # [REVISE] Verify first-layer projections after dtype conversion

            query_states = apply_rope(query_states, position_ids)
            key_states = apply_rope(key_states, position_ids)

            if use_cache and past_key_values is None:
                past_key_values = {}

            if use_cache:
                if fill_kv_cache:
                    past_key_values[layer_idx] = {
                        "key_states": key_states,
                        "value_states": value_states,
                    }
                else:
                    # TODO here, some optimization can be done - similar to a `StaticCache` we can declare the `max_len` before.
                    # so we create an empty cache, with just one cuda malloc, and if (in autoregressive case) we reach
                    # the max len, then we (for instance) double the cache size. This implementation already exists
                    # in `transformers`. (molbap)
                    key_states = torch.cat([past_key_values[layer_idx]["key_states"], key_states], dim=1)
                    value_states = torch.cat(
                        [past_key_values[layer_idx]["value_states"], value_states], dim=1
                    )

            attention_interface = self.get_attention_interface()
            att_output = attention_interface(
                attention_mask, batch_size, head_dim, query_states, key_states, value_states
            )
            att_output = att_output.to(dtype=torch.bfloat16)

            if layer_idx == 0:
                _log_shapes(self.config.debug_shapes, "paligemma.forward.att_output", att_output=att_output)  # [REVISE] Confirm fused attention activations before residuals

            # first part of att_output is prefix (up to sequence length, [:, 0:prefix_seq_len])
            outputs_embeds = []
            start = 0
            for i, hidden_states in enumerate(inputs_embeds):
                layer = models[i].layers[layer_idx]

                if hidden_states is not None:
                    end = start + hidden_states.shape[1]

                    if att_output.dtype != layer.self_attn.o_proj.weight.dtype:
                        att_output = att_output.to(layer.self_attn.o_proj.weight.dtype)
                    out_emb = layer.self_attn.o_proj(att_output[:, start:end])

                    # TODO: first dropout (by default 0.0)

                    # first residual
                    out_emb += hidden_states
                    after_first_residual = out_emb.clone()

                    out_emb = layer.post_attention_layernorm(out_emb)
                    out_emb = layer.mlp(out_emb)

                    # TODO: second dropout (by default 0.0)

                    # second residual
                    out_emb += after_first_residual

                    outputs_embeds.append(out_emb)

                    start = end
                else:
                    outputs_embeds.append(None)

            inputs_embeds = outputs_embeds

        # final norm
        outputs_embeds = []
        for i, hidden_states in enumerate(inputs_embeds):
            if hidden_states is not None:
                out_emb = models[i].norm(hidden_states)
                outputs_embeds.append(out_emb)
            else:
                outputs_embeds.append(None)

        _log_shapes(self.config.debug_shapes, "paligemma.forward.outputs", outputs_embeds=outputs_embeds)  # [REVISE] Ensure normalized outputs respect dtype expectations
        return outputs_embeds, past_key_values

    def get_attention_interface(self):
        if self.config.attention_implementation == "fa2":
            attention_interface = self.flash_attention_forward
        elif self.config.attention_implementation == "flex":
            attention_interface = flex_attention_forward
        else:
            attention_interface = self.eager_attention_forward
        return attention_interface

    def flash_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        raise NotImplementedError("FA2 is not implemented (yet)")

    def eager_attention_forward(
        self, attention_mask, batch_size, head_dim, query_states, key_states, value_states
    ):
        num_att_heads = self.config.paligemma_config.text_config.num_attention_heads
        num_key_value_heads = self.config.paligemma_config.text_config.num_key_value_heads
        num_key_value_groups = num_att_heads // num_key_value_heads

        # query_states: batch_size, sequence_length, num_att_head, head_dim
        # key_states: batch_size, sequence_length, num_key_value_head, head_dim
        # value_states: batch_size, sequence_length, num_key_value_head, head_dim
        sequence_length = key_states.shape[1]

        key_states = key_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        key_states = key_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        value_states = value_states[:, :, :, None, :].expand(
            batch_size, sequence_length, num_key_value_heads, num_key_value_groups, head_dim
        )
        value_states = value_states.reshape(
            batch_size, sequence_length, num_key_value_heads * num_key_value_groups, head_dim
        )

        # Attention here is upcasted to float32 to match the original eager implementation.

        query_states = query_states.to(dtype=torch.float32)
        key_states = key_states.to(dtype=torch.float32)

        query_states = query_states.transpose(1, 2)
        key_states = key_states.transpose(1, 2)

        att_weights = torch.matmul(query_states, key_states.transpose(2, 3))
        att_weights *= head_dim**-0.5
        big_neg = -2.3819763e38  # See gemma/modules.py

        masked_att_weights = torch.where(attention_mask[:, None, :, :], att_weights, big_neg)

        probs = nn.functional.softmax(masked_att_weights, dim=-1)
        probs = probs.to(dtype=value_states.dtype)

        # probs: batch_size, num_key_value_head, num_att_head, sequence_length, sequence_length
        # value_states: batch_size, sequence_length, num_att_heads, head_dim

        att_output = torch.matmul(probs, value_states.permute(0, 2, 1, 3))

        att_output = att_output.permute(0, 2, 1, 3)
        # we use -1 because sequence length can change
        att_output = att_output.reshape(batch_size, -1, num_key_value_heads * num_key_value_groups * head_dim)

        _log_shapes(
            self.config.debug_shapes,
            "paligemma.eager_attention",
            masked_att_weights=masked_att_weights,
            probs=probs,
            att_output=att_output,
        )  # [REVISE] Trace attention numerics to debug FA fallback behavior

        return att_output
