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
Flex Attention 实现模块

维度符号说明:
    B    : Batch Size (批次大小)
    Q_LEN: Query 序列长度
    KV_LEN: Key/Value 序列长度
    H_q  : Query 的注意力头数 (num_att_heads), 例如 8
    H_kv : Key/Value 的注意力头数 (num_key_value_heads), 例如 1
    D    : 每个头的特征维度 (head_dim), 例如 64
    H    : 重排后的头数 (GQA 扩展后 H = H_q)

核心流程 (12 步):
    1.  GQA 扩展：将 KV 从 H_kv=1 头扩展到 H_q=8 头
    2.  张量重排：转为 [B, H, SEQ_LEN, D] 格式 (flex attention 要求)
    3.  精度转换：float16 → float32 (避免数值不稳定)
    4.  构建因果掩码：确保因果性/自回归
    5.  定义 Mask 工厂函数：创建 mask_mod 函数
    6.  序列长度对齐：向上舍入到 block_size=128 的倍数
    7.  Padding Q/KV 张量：与 block_mask 尺寸匹配
    8.  Padding 因果掩码：填充到对齐后的尺寸
    9.  创建 4D Mask：生成完整的 mask 张量
    10. 创建 Block Mask：将像素级 mask 转换为块级 mask
    11. 执行 Flex Attention：只计算需要的块，跳过无效计算
    12. 输出后处理：裁剪 padding → 转置 → 展平 → [B, Q_LEN, H_q*D]
"""

import torch
import torch.nn.functional as F  # noqa: N812
from packaging.version import Version

if Version(torch.__version__) > Version("2.5.0"):
    # Flex attention is only available from torch 2.5 onwards
    from torch.nn.attention.flex_attention import (
        _mask_mod_signature,
        _round_up_to_multiple,
        create_block_mask,
        create_mask,
        flex_attention,
    )


# @torch.compile(dynamic=False)
def flex_attention_forward(
    attention_mask: torch.Tensor,
    batch_size: int,
    head_dim: int,
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    scaling=None,
):
    """
    Flex Attention 前向传播
    
    参数:
        attention_mask: 因果掩码 [B, Q_LEN, KV_LEN] 或 [B, 1, Q_LEN, KV_LEN]
        batch_size: 批次大小 B
        head_dim: 每个头的特征维度 D
        query_states: Query 张量 [B, Q_LEN, H_q, D]
        key_states: Key 张量 [B, KV_LEN, H_kv, D]
        value_states: Value 张量 [B, KV_LEN, H_kv, D]
        scaling: 缩放因子，默认为 None (使用 1/sqrt(D))
    
    返回:
        attn_output: 注意力输出 [B, Q_LEN, H_q * D]
    
    维度变换总览:
        输入 Q: [B, Q_LEN, H_q, D]
        输入 K: [B, KV_LEN, H_kv, D]
        输入 V: [B, KV_LEN, H_kv, D]
        
        GQA 扩展后 K: [B, KV_LEN, H_q, D]  (H_kv=1 → H_q=8)
        GQA 扩展后 V: [B, KV_LEN, H_q, D]
        
        转置后 Q: [B, H_q, Q_LEN, D]
        转置后 K: [B, H_q, KV_LEN, D]
        转置后 V: [B, H_q, KV_LEN, D]
        
        Flex Attention 计算 (内部不改变维度)
        
        输出转置：[B, Q_LEN, H_q, D]
        输出展平：[B, Q_LEN, H_q * D]
    """
    original_dtype = query_states.dtype
    
    # ==================== 超参数定义 ====================
    num_att_heads = 8       # H_q: Query 的注意力头数
    num_key_value_heads = 1 # H_kv: Key/Value 的注意力头数 (GQA 设置)
    num_key_value_groups = num_att_heads // num_key_value_heads  # = 8, 每组 KV 头服务的 Q 头数
    
    # ==================== 步骤 1: GQA 扩展 ====================
    # 目的：将 KV 从 H_kv=1 头扩展到 H_q=8 头，供所有 Q 头使用
    # 原理：GQA 让多个 Q 头共享同一组 KV，节省显存和计算
    
    # --- Key 状态扩展 ---
    # 输入 key_states: [B, KV_LEN, H_kv, D] = [B, KV_LEN, 1, D]
    key_states = key_states[:, :, :, None, :]
    # 插入新维度后：[B, KV_LEN, H_kv, 1, D]
    
    key_states = key_states.expand(
        batch_size, key_states.shape[1], num_key_value_heads, num_key_value_groups, head_dim
    )
    # expand 后：[B, KV_LEN, H_kv, num_key_value_groups, D]
    #         = [B, KV_LEN, 1, 8, D]
    
    key_states = key_states.reshape(
        batch_size, key_states.shape[1], num_key_value_heads * num_key_value_groups, head_dim
    )
    # reshape 后 key_states: [B, KV_LEN, H_q, D] = [B, KV_LEN, 8, D]
    # 现在 K 有 8 个头，与 Q 的头数匹配
    
    # --- Value 状态扩展 (同 Key) ---
    # 输入 value_states: [B, KV_LEN, H_kv, D] = [B, KV_LEN, 1, D]
    value_states = value_states[:, :, :, None, :]
    # 插入新维度后：[B, KV_LEN, H_kv, 1, D]
    
    value_states = value_states.expand(
        batch_size, value_states.shape[1], num_key_value_heads, num_key_value_groups, head_dim
    )
    # expand 后：[B, KV_LEN, 1, 8, D]
    
    value_states = value_states.reshape(
        batch_size, value_states.shape[1], num_key_value_heads * num_key_value_groups, head_dim
    )
    # reshape 后 value_states: [B, KV_LEN, H_q, D] = [B, KV_LEN, 8, D]
    
    # ==================== 步骤 2: 张量重排 ====================
    # 目的：转为 flex attention 要求的 [B, H, SEQ_LEN, D] 格式
    # 原始：[B, SEQ_LEN, H, D] → 转置后：[B, H, SEQ_LEN, D]
    
    # query_states: [B, Q_LEN, H_q, D] → [B, H_q, Q_LEN, D]
    query_states = query_states.transpose(1, 2)
    # key_states:   [B, KV_LEN, H_q, D] → [B, H_q, KV_LEN, D]
    key_states = key_states.transpose(1, 2)
    # value_states: [B, KV_LEN, H_q, D] → [B, H_q, KV_LEN, D]
    value_states = value_states.transpose(1, 2)
    
    # ==================== 步骤 3: 精度转换 ====================
    # 目的：转为 float32 避免注意力计算中的数值不稳定 (exp/softmax 容易溢出)
    query_states = query_states.to(torch.float32)
    key_states = key_states.to(torch.float32)
    value_states = value_states.to(torch.float32)

    # ==================== 步骤 4: 构建因果掩码 ====================
    # 目的：确保每个位置只能 attend 到之前和当前位置 (causal/autoregressive)
    causal_mask = attention_mask
    if causal_mask is not None:
        # causal_mask 输入可能是 [B, Q_LEN, KV_LEN] 或 [B, 1, Q_LEN, KV_LEN]
        # 取 KV 维度到 key_states.shape[2] (即 KV_LEN)
        causal_mask = causal_mask[:, None, :, : key_states.shape[2]]
        # 处理后 causal_mask: [B, 1, Q_LEN, KV_LEN]

        # 如果 head 维度是 1 但 Q 有多个头，需要扩展到所有头
        if causal_mask.shape[1] == 1 and query_states.shape[1] > 1:
            causal_mask = causal_mask.expand(-1, query_states.shape[1], -1, -1)
        # expand 后 causal_mask: [B, H_q, Q_LEN, KV_LEN]
        # 所有头共享同一个因果掩码

    # ==================== 步骤 5: 定义 Mask 工厂函数 ====================
    # 目的：创建一个函数，给定 (b, h, q_idx, kv_idx) 返回是否应该计算该位置
    def precomputed_mask_factory(precomputed_mask: torch.Tensor) -> _mask_mod_signature:
        def mask_mod(b, h, q_idx, kv_idx):
            # Danger zone: if b,h,q_idx,kv_idx exceed the shape, device-side assert occurs.
            # 返回 True 表示需要计算，False 表示跳过
            return precomputed_mask[b][h][q_idx][kv_idx]
        return mask_mod

    # 获取 causal_mask 的维度
    b_mask, h_mask, q_len, kv_len = causal_mask.shape
    # b_mask = B, h_mask = H_q, q_len = Q_LEN, kv_len = KV_LEN

    # ==================== 步骤 6: 序列长度对齐到 block_size ====================
    # 目的：Flex Attention 将序列分块处理 (block_size=128), 需要对齐到块的倍数
    block_size = 128  # 块大小，硬件友好的选择
    q_len_rounded = _round_up_to_multiple(q_len, block_size)
    # q_len_rounded: 向上舍入到 128 的倍数，例如 100 → 128
    kv_len_rounded = _round_up_to_multiple(kv_len, block_size)
    # kv_len_rounded: 向上舍入到 128 的倍数

    # 计算需要 padding 的数量
    pad_q = q_len_rounded - q_len  # Q 需要填充的数量
    pad_k = kv_len_rounded - kv_len  # KV 需要填充的数量

    # ==================== 步骤 7: Padding Q/KV 张量 ====================
    # 目的：将 Q/KV 填充到 block_size 的倍数，与 block_mask 匹配
    # 注意：flex attention 要求输入张量长度与 block_mask 一致
    if pad_q > 0 or pad_k > 0:
        # query_states: [B, H_q, Q_LEN, D] → [B, H_q, q_len_rounded, D]
        query_states = F.pad(query_states, (0, 0, 0, pad_q), value=0.0)
        # key_states: [B, H_q, KV_LEN, D] → [B, H_q, kv_len_rounded, D]
        key_states = F.pad(key_states, (0, 0, 0, pad_k), value=0.0)
        # value_states: [B, H_q, KV_LEN, D] → [B, H_q, kv_len_rounded, D]
        value_states = F.pad(value_states, (0, 0, 0, pad_k), value=0.0)

    # ==================== 步骤 8: Padding 因果掩码 ====================
    # 目的：将 causal_mask 填充到对齐后的尺寸，避免 CUDA 索引越界
    # F.pad 的 (0, pad_k, 0, pad_q) 表示：
    #   - 最后维度 (KV_LEN) 填充 (0, pad_k): 前面 0，后面 pad_k
    #   - 倒数第二维度 (Q_LEN) 填充 (0, pad_q): 前面 0，后面 pad_q
    padded_causal_mask = F.pad(causal_mask, (0, pad_k, 0, pad_q), value=0.0)
    # padded_causal_mask: [B, H_q, q_len_rounded, kv_len_rounded]
    # 例如：[B, 8, 128, 128] (如果原始是 100x100)

    # 创建原始 mask_mod 函数
    mask_mod_fn_orig = precomputed_mask_factory(padded_causal_mask)

    # ==================== 步骤 9: 创建 4D Mask ====================
    # 目的：create_mask 生成完整的 4D mask 张量，用于后续 block_mask 创建
    mask_4d = create_mask(
        mod_fn=mask_mod_fn_orig,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        device=causal_mask.device,
    )
    # mask_4d: [B, H_q, q_len_rounded, kv_len_rounded]
    # 值域：True/False 或 1/0，表示每个位置是否需要计算

    # ==================== 步骤 10: 创建 Block Mask ====================
    # 目的：将像素级 mask 转换为块级 mask，告诉 flex attention 哪些 128x128 块需要计算
    # 这是 Flex Attention 的核心优化：跳过整个块比逐个位置判断更高效
    mask_mod_fn_padded = precomputed_mask_factory(mask_4d)
    block_mask = create_block_mask(
        mask_mod=mask_mod_fn_padded,
        B=b_mask,
        H=h_mask,
        Q_LEN=q_len_rounded,
        KV_LEN=kv_len_rounded,
        BLOCK_SIZE=block_size,
        device=causal_mask.device,
    )
    # block_mask: 稀疏块掩码，内部结构由 flex attention 定义
    # 对于因果掩码，上三角的块会被标记为跳过
    # block_mask 对应：[B, H_q, q_len_rounded, kv_len_rounded]

    # ==================== 步骤 11: 执行 Flex Attention ====================
    # 目的：调用 flex attention 内核，只计算需要的块
    # 内部计算:
    #   1. 将 Q/KV 分成 128x128 的块
    #   2. 根据 block_mask 跳过不需要的块 (如因果掩码的上三角)
    #   3. 对每个需要的块计算 attention: softmax(Q @ K^T / sqrt(D)) @ V
    #   4. 使用 LSE (log-sum-exp) 保证数值稳定性
    attn_output, attention_weights = flex_attention(
        query_states,          # [B, H_q, q_len_rounded, D]
        key_states,            # [B, H_q, kv_len_rounded, D]
        value_states,          # [B, H_q, kv_len_rounded, D]
        block_mask=block_mask,  # 块级掩码，定义哪些块需要计算
        enable_gqa=True,       # 启用 GQA 优化 (虽然我们已经手动扩展了 KV)
        scale=head_dim**-0.5 if scaling is None else scaling,  # 1/sqrt(D)
        return_lse=True,       # 返回 log-sum-exp，用于数值稳定
    )
    # attn_output: [B, H_q, q_len_rounded, D]
    # attention_weights: 注意力权重 (如果 return_lse=True 则包含 LSE 信息)

    # ==================== 步骤 12: 输出后处理 ====================
    # 转回原始精度
    attn_output = attn_output.to(dtype=original_dtype)

    # 裁剪 padding 部分，回到原始 Q_LEN
    attn_output = attn_output[:, :, :q_len, :]
    # attn_output: [B, H_q, Q_LEN, D]

    # 转置回 [B, Q_LEN, H_q, D] 格式
    attn_output = attn_output.transpose(1, 2).contiguous()
    # attn_output: [B, Q_LEN, H_q, D]

    # 展平最后两个维度 [H_q, D] → [H_q * D]
    attn_output = attn_output.reshape(
        batch_size,
        -1,  # Q_LEN
        attn_output.shape[2] * attn_output.shape[3],  # H_q * D
    )
    # 最终输出 attn_output: [B, Q_LEN, H_q * D]
    # 例如：[B, Q_LEN, 8 * 64] = [B, Q_LEN, 512]
    
    return attn_output


if __name__ == "__main__":
    """
    Flex Attention 测试函数
    
    随机生成输入张量，验证维度变换和输出正确性
    """
    print("=" * 60)
    print("Flex Attention 维度测试")
    print("=" * 60)
    
    # ==================== 测试参数设置 ====================
    B = 2              # 批次大小
    Q_LEN = 100        # Query 序列长度
    KV_LEN = 100       # Key/Value 序列长度 (通常等于 Q_LEN)
    H_q = 8            # Query 头数
    H_kv = 1           # Key/Value 头数 (GQA 设置)
    D = 64             # 每个头的维度 (head_dim)
    
    print(f"\n【测试参数】")
    print(f"  Batch Size (B)        : {B}")
    print(f"  Query Sequence Length : {Q_LEN}")
    print(f"  KV Sequence Length    : {KV_LEN}")
    print(f"  Query Heads (H_q)     : {H_q}")
    print(f"  KV Heads (H_kv)       : {H_kv}")
    print(f"  Head Dimension (D)    : {D}")
    print(f"  GQA Groups            : {H_q // H_kv}")
    
    # ==================== 随机生成输入张量 ====================
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n【设备】{device}")
    
    # 输入 Q/KV: [B, SEQ_LEN, H, D]
    query_states = torch.randn(B, Q_LEN, H_q, D, device=device, dtype=torch.float16)
    key_states = torch.randn(B, KV_LEN, H_kv, D, device=device, dtype=torch.float16)
    value_states = torch.randn(B, KV_LEN, H_kv, D, device=device, dtype=torch.float16)
    
    print(f"\n【输入维度】")
    print(f"  query_states : {tuple(query_states.shape)}  [B, Q_LEN, H_q, D]")
    print(f"  key_states   : {tuple(key_states.shape)}  [B, KV_LEN, H_kv, D]")
    print(f"  value_states : {tuple(value_states.shape)}  [B, KV_LEN, H_kv, D]")
    
    # ==================== 生成因果掩码 ====================
    # 因果掩码：下三角为 1 (允许 attend)，上三角为 0 (禁止 attend)
    # 形状：[B, Q_LEN, KV_LEN]
    causal_mask = torch.tril(
        torch.ones(Q_LEN, KV_LEN, dtype=torch.bool, device=device)
    ).unsqueeze(0).expand(B, -1, -1)
    
    print(f"\n【因果掩码】")
    print(f"  attention_mask : {tuple(causal_mask.shape)}  [B, Q_LEN, KV_LEN]")
    print(f"  下三角=1 (允许), 上三角=0 (禁止)")
    
    # ==================== 执行 Flex Attention ====================
    print(f"\n【执行 Flex Attention】")
    
    with torch.no_grad():
        output = flex_attention_forward(
            attention_mask=causal_mask,
            batch_size=B,
            head_dim=D,
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
        )
    
    print(f"\n【输出维度】")
    print(f"  attn_output : {tuple(output.shape)}  [B, Q_LEN, H_q * D]")
    print(f"  预期维度    : [B, {Q_LEN}, {H_q * D}]")
    
    # ==================== 维度验证 ====================
    print(f"\n【维度验证】")
    expected_shape = (B, Q_LEN, H_q * D)
    if output.shape == expected_shape:
        print(f"  ✓ 维度正确！{tuple(output.shape)} == {expected_shape}")
    else:
        print(f"  ✗ 维度错误！{tuple(output.shape)} != {expected_shape}")
    
    # ==================== 数值检查 ====================
    print(f"\n【数值检查】")
    print(f"  dtype: {output.dtype}")
    print(f"  mean: {output.mean().item():.6f}")
    print(f"  std:  {output.std().item():.6f}")
    print(f"  min:  {output.min().item():.6f}")
    print(f"  max:  {output.max().item():.6f}")
    print(f"  has_nan: {torch.isnan(output).any().item()}")
    print(f"  has_inf: {torch.isinf(output).any().item()}")
    
    # ==================== 因果性验证 ====================
    print(f"\n【因果性验证】(仅示意，需要更复杂的测试)")
    print(f"  因果掩码确保每个位置只能 attend 到之前和当前位置")
    print(f"  这通过 block_mask 跳过未来位置的计算实现")
    
    print("\n" + "=" * 60)
    print("测试完成!")
    print("=" * 60)
