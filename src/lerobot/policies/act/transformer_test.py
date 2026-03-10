# 新增：语言tokenizer（使用HuggingFace transformers）
# =============================
from transformers import AutoTokenizer
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 加载一个流行的预训练tokenizer（如bert-base-uncased）
LANG_TOKENIZER_NAME = 'bert-base-uncased'
lang_tokenizer = AutoTokenizer.from_pretrained(LANG_TOKENIZER_NAME)


# =============================
# 自定义实现多头自注意力机制（不使用nn.MultiheadAttention）
# =============================
class SimpleMultiHeadSelfAttention(nn.Module):
    def __init__(self, dim_model=512, n_heads=8, dropout=0.1):
        super().__init__()
        assert dim_model % n_heads == 0, 'dim_model必须能被n_heads整除'
        self.n_heads = n_heads
        self.head_dim = dim_model // n_heads
        self.dim_model = dim_model
        # Q, K, V的线性变换
        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        # 输出线性变换
        self.out_proj = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, mask_type=None):
        # x: [seq_len, batch_size, dim_model]
        seq_len, batch_size, dim_model = x.shape
        # 1. 线性变换得到Q, K, V
        Q = self.q_proj(x)  # [seq_len, batch_size, dim_model]
        K = self.k_proj(x)  # [seq_len, batch_size, dim_model]
        V = self.v_proj(x)  # [seq_len, batch_size, dim_model]
        # 2. 拆分为多头
        def split_heads(tensor):
            return tensor.view(seq_len, batch_size, self.n_heads, self.head_dim).permute(2, 1, 0, 3)
        Q = split_heads(Q)  # [n_heads, batch_size, seq_len, head_dim]
        K = split_heads(K)  # [n_heads, batch_size, seq_len, head_dim]
        V = split_heads(V)  # [n_heads, batch_size, seq_len, head_dim]
        # 3. 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [n_heads, batch_size, seq_len, seq_len]
        # 4. 应用mask
        if mask_type == 'causal':
            mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(mask, float('-inf'))
        elif mask_type == 'bidirectional':
            pass  # 不做mask
        elif mask_type == 'custom' and attn_mask is not None:
            scores = scores.masked_fill(attn_mask.bool(), float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)                      # [n_heads, batch_size, seq_len, head_dim]
        context = context.permute(2, 1, 0, 3).contiguous()   # [seq_len, batch_size, n_heads, head_dim]
        context = context.view(seq_len, batch_size, self.dim_model)  # [seq_len, batch_size, n_heads*head_dim]
        out = self.out_proj(context)      # [seq_len, batch_size, dim_model]
        return out


# 1. 定义TransformerBlock
class SimpleTransformerBlock(nn.Module):
    def __init__(self, dim_model=512, n_heads=8, dim_feedforward=3200, dropout=0.1, activation='relu', mask_type=None):
        super().__init__()
        self.self_attn = SimpleMultiHeadSelfAttention(dim_model, n_heads, dropout)
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(dim_model)
        self.norm2 = nn.LayerNorm(dim_model)
        self.activation = F.relu if activation == 'relu' else F.gelu
        self.mask_type = mask_type

    def forward(self, x, attn_mask=None):
        # x: [seq_len, batch_size, dim_model]
        # 1. 多头自注意力机制
        attn_output = self.self_attn(x, attn_mask=attn_mask, mask_type=self.mask_type)  # [seq_len, batch_size, dim_model]
        # 2. 残差连接+归一化
        x = self.norm1(x + self.dropout(attn_output))  # [seq_len, batch_size, dim_model]
        # 3. 前馈神经网络
        ff_output = self.linear2(self.activation(self.linear1(x)))  # [seq_len, batch_size, dim_model]
        # 4. 残差连接+归一化
        x = self.norm2(x + self.dropout(ff_output))  # [seq_len, batch_size, dim_model]
        return x

# 2. 定义一个简单的Transformer模型（堆叠多个block）
class SimpleTransformerModel(nn.Module):
    def __init__(self, dim_model=512, n_heads=8, dim_feedforward=3200, dropout=0.1, n_layers=4, seq_len=100, action_dim=14, pos_embed_type='learned', mask_type=None):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(dim_model, n_heads, dim_feedforward, dropout, mask_type=mask_type)
            for _ in range(n_layers)
        ])
        self.action_head = nn.Linear(dim_model, action_dim)
        self.pos_embed_type = pos_embed_type
        if pos_embed_type == 'learned':
            self.pos_embed = nn.Parameter(torch.randn(seq_len, dim_model))
        elif pos_embed_type == 'sinusoidal':
            self.register_buffer('pos_embed', get_sinusoidal_pos_embed(seq_len, dim_model))
        else:
            raise ValueError('pos_embed_type must be "learned" or "sinusoidal"')

    def forward(self, x, attn_mask=None):
        x = x + self.pos_embed.unsqueeze(1)
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        x = x.transpose(0, 1)
        actions = self.action_head(x)
        return actions

# 正弦位置编码函数
# 返回: [seq_len, dim_model]
def get_sinusoidal_pos_embed(seq_len, dim_model):
    pe = torch.zeros(seq_len, dim_model)
    position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe



# =============================
# 添加一个简易ViT模块用于图像token提取
# =============================
class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, dim_model=512, n_heads=8, n_layers=4, dropout=0.1):
        super().__init__()
        assert img_size % patch_size == 0, 'img_size必须能被patch_size整除'
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.dim_model = dim_model
        # patch embedding: 将每个patch展平后映射到dim_model维度
        self.patch_embed = nn.Linear(patch_size * patch_size * in_channels, dim_model)
        # 位置编码
        self.pos_embed = nn.Parameter(torch.randn(self.n_patches, dim_model))
        # transformer encoder
        self.transformer = nn.ModuleList([
            SimpleTransformerBlock(dim_model, n_heads, dim_feedforward=3200, dropout=dropout)
            for _ in range(n_layers)
        ])
        # 可选：cls token用于全局特征
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim_model))

    def forward(self, img):
        # img: [batch_size, 3, 224, 224]
        batch_size = img.shape[0]
        # 1. 划分patch. torch.Size([1, 3, 14, 14, 16, 16])
        patches = img.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches: [batch_size, 3, n_patches_h, n_patches_w, patch_size, patch_size]  ([1, 3, 196, 16, 16])
        patches = patches.contiguous().view(batch_size, 3, self.n_patches, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # [batch_size, n_patches, 3, patch_size, patch_size]  ([1, 196, 3, 16, 16])
        patches = patches.view(batch_size, self.n_patches, -1)  # [batch_size, n_patches, 3*patch_size*patch_size]   ([1, 196, 768])
        # 2. patch embedding
        tokens = self.patch_embed(patches)  # [batch_size, n_patches, dim_model] ([1, 196, 512]) 一个MLP将每个patch映射到512维的token空间 
        # 3. 加位置编码
        tokens = tokens + self.pos_embed.unsqueeze(0)  # [batch_size, n_patches, dim_model]
        # 4. 加cls token
        cls_token = self.cls_token.expand(-1, batch_size, -1).transpose(0,1)  # [batch_size, 1, dim_model]
        tokens = torch.cat([cls_token, tokens], dim=1)  # [batch_size, n_patches+1, dim_model]
        # 5. transformer encoder
        x = tokens.transpose(0,1)  # [seq_len, batch_size, dim_model]
        for layer in self.transformer:
            x = layer(x)
        x = x.transpose(0,1)  # [batch_size, seq_len, dim_model]
        return x  # 返回所有token


# =============================
# 新增：集成ViT与Transformer用于机器人视觉动作预测
# =============================

# =============================
# 新增：视觉-语言-动作模型
# =============================
class VisionLanguageTransformerPolicy(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, dim_model=512, n_heads=8, n_layers=4, action_dim=14, dropout=0.1, lang_tokenizer=None, max_lang_len=32):
        super().__init__()
        self.vit = SimpleViT(img_size, patch_size, in_channels, dim_model, n_heads, n_layers, dropout)
        self.lang_tokenizer = lang_tokenizer
        self.max_lang_len = max_lang_len
        # 语言token embedding
        self.lang_embed = nn.Embedding(self.lang_tokenizer.vocab_size, dim_model)
        self.lang_pos_embed = nn.Parameter(torch.randn(self.max_lang_len, dim_model))
        # 合并视觉和语言token后再用transformer block
        self.fusion_transformer = nn.ModuleList([
            SimpleTransformerBlock(dim_model, n_heads, dim_feedforward=3200, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.action_head = nn.Linear(dim_model, action_dim)

    def forward(self, img, lang_text):
        # img: [batch_size, 3, 224, 224]
        # lang_text: str or list[str]
        batch_size = img.shape[0]
        # 1. 图像token
        visual_tokens = self.vit(img)  # [batch_size, n_visual_tokens, dim_model]
        # 2. 语言token
        if isinstance(lang_text, str):
            lang_text = [lang_text] * batch_size
        lang_enc = self.lang_tokenizer(lang_text, padding='max_length', truncation=True, max_length=self.max_lang_len, return_tensors='pt')
        input_ids = torch.tensor(lang_enc['input_ids']).to(img.device)  # [batch_size, max_lang_len] token ids
        lang_tokens = self.lang_embed(input_ids)  # [batch_size, max_lang_len, dim_model]
        lang_tokens = lang_tokens + self.lang_pos_embed.unsqueeze(0)  # 加位置编码
        # 3. 合并视觉和语言token
        all_tokens = torch.cat([lang_tokens, visual_tokens], dim=1)  # [batch_size, max_lang_len+n_visual_tokens, dim_model]
        x = all_tokens.transpose(0,1)  # [seq_len, batch_size, dim_model]
        for layer in self.fusion_transformer:
            x = layer(x)
        x = x.transpose(0,1)  # [batch_size, seq_len, dim_model]
        # 取第一个token（cls或语言首token）作为动作预测
        cls_token = x[:, 0, :]  # [batch_size, dim_model]
        actions = self.action_head(cls_token)  # [batch_size, action_dim]
        return actions

# =============================
# 新增：离散动作tokenizer（简单bin-packing示例）
# =============================
class SimpleActionTokenizer:
    """
    假设每个连续动作维度分bin离散化，适用于bin-packing/FAST等思想。
    这里只做简单均匀分bin，实际可参考lerobot/FAST等更复杂实现。
    """
    def __init__(self, action_dim, n_bins=256, action_range=(-1, 1)):
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.action_range = action_range
        self.bin_edges = torch.linspace(action_range[0], action_range[1], n_bins+1)

    def encode(self, actions):
        # actions: [batch_size, action_dim], float
        # 返回: [batch_size, action_dim], 每个元素为bin编号
        actions = torch.clamp(actions, self.action_range[0], self.action_range[1])
        bin_idx = torch.bucketize(actions, self.bin_edges) - 1
        bin_idx = torch.clamp(bin_idx, 0, self.n_bins-1)
        return bin_idx

    def decode(self, bin_idx):
        # bin_idx: [batch_size, action_dim], int
        # 返回: [batch_size, action_dim], float
        centers = (self.bin_edges[:-1] + self.bin_edges[1:]) / 2
        return centers[bin_idx]


# =============================
# 新增：视觉-语言-离散动作预测模型
# =============================
class VisionLanguageDiscretePolicy(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, dim_model=512, n_heads=8, n_layers=4, action_dim=14, n_bins=256, dropout=0.1, lang_tokenizer=None, max_lang_len=32):
        super().__init__()
        self.vit = SimpleViT(img_size, patch_size, in_channels, dim_model, n_heads, n_layers, dropout)
        self.lang_tokenizer = lang_tokenizer
        self.max_lang_len = max_lang_len
        self.action_dim = action_dim
        self.n_bins = n_bins
        # 语言token embedding
        self.lang_embed = nn.Embedding(self.lang_tokenizer.vocab_size, dim_model)
        self.lang_pos_embed = nn.Parameter(torch.randn(self.max_lang_len, dim_model))
        # 融合transformer
        self.fusion_transformer = nn.ModuleList([
            SimpleTransformerBlock(dim_model, n_heads, dim_feedforward=3200, dropout=dropout)
            for _ in range(n_layers)
        ])
        # 输出为每个动作维度的n_bins分类概率
        self.action_head = nn.Linear(dim_model, action_dim * n_bins)
        # 动作tokenizer
        self.action_tokenizer = SimpleActionTokenizer(action_dim, n_bins)

    def forward(self, img, lang_text):
        # img: [batch_size, 3, 224, 224]
        # lang_text: str or list[str]
        batch_size = img.shape[0]
        # 1. 图像token
        visual_tokens = self.vit(img)  # [batch_size, n_visual_tokens, dim_model]
        # 2. 语言token
        if isinstance(lang_text, str):
            lang_text = [lang_text] * batch_size
        lang_enc = self.lang_tokenizer(lang_text, padding='max_length', truncation=True, max_length=self.max_lang_len, return_tensors='pt')
        input_ids = torch.tensor(lang_enc['input_ids']).to(img.device)
        lang_tokens = self.lang_embed(input_ids)  # [batch_size, max_lang_len, dim_model]
        lang_tokens = lang_tokens + self.lang_pos_embed.unsqueeze(0)
        # 3. 合并视觉和语言token
        all_tokens = torch.cat([lang_tokens, visual_tokens], dim=1)  # [batch_size, max_lang_len+n_visual_tokens, dim_model]
        x = all_tokens.transpose(0,1)  # [seq_len, batch_size, dim_model]
        for layer in self.fusion_transformer:
            x = layer(x)
        x = x.transpose(0,1)  # [batch_size, seq_len, dim_model]
        # 取第一个token（cls或语言首token）
        cls_token = x[:, 0, :]  # [batch_size, dim_model]
        logits = self.action_head(cls_token)  # [batch_size, action_dim * n_bins]
        logits = logits.view(batch_size, self.action_dim, self.n_bins)  # [batch_size, action_dim, n_bins]
        probs = torch.softmax(logits, dim=-1)  # [batch_size, action_dim, n_bins]
        return probs  # 每个动作维度的离散概率分布

    def compute_loss(self, img, lang_text, expert_actions):
        """
            img: [batch_size, 3, 224, 224]
            lang_text: str or list[str]
            expert_actions: [batch_size, action_dim] (float, continuous)
            返回: 离散动作交叉熵损失
        """
        # 1. 前向获得概率分布
        probs = self.forward(img, lang_text)  # [batch_size, action_dim, n_bins]
        # 2. 离散化expert动作
        expert_bins = self.action_tokenizer.encode(expert_actions)  # [batch_size, action_dim] (long)
        # 3. 计算交叉熵损失
        batch_size, action_dim, n_bins = probs.shape
        log_probs = torch.log(probs + 1e-8)
        log_probs = log_probs.view(-1, n_bins)
        expert_bins = expert_bins.view(-1)
        loss = nn.functional.nll_loss(log_probs, expert_bins, reduction='mean')
        return loss


# =============================
# 新增：Qwen风格RoPE与单向Cross-Attention融合的视觉-语言-离散动作模型
# =============================
def apply_rope(x, seq_dim=-2):
    """
    x: [seq_len, batch_size, dim] or [batch_size, seq_len, dim]
    seq_dim: 0 or 1, 位置维度
    返回: 加入RoPE后同shape张量
    """
    # 仅支持偶数维度
    # x: [..., dim], e.g. (2, 10, 20) if x.shape = (batch, seq_len, dim)
    dim = x.shape[-1]  # e.g. 20
    assert dim % 2 == 0
    half = dim // 2    # e.g. 10
    seq_len = x.shape[seq_dim]  # e.g. 10 if seq_dim=1
    pos = torch.arange(seq_len, dtype=x.dtype, device=x.device)  # (seq_len,)
    freq = torch.exp(-math.log(10000.0) * torch.arange(0, half, dtype=x.dtype, device=x.device) / half)  # (half,)
    sinusoid = torch.einsum('i,j->ij', pos, freq)  # (seq_len, half)
    sin = sinusoid.sin()  # (seq_len, half)
    cos = sinusoid.cos()  # (seq_len, half)
    # Expand sin/cos to match x shape at seq_dim
    # shape: [1, ..., seq_len, ..., 1, half] (seq_len at seq_dim, half at last dim)
    shape = [1] * x.ndim
    shape[seq_dim] = seq_len
    shape[-1] = half
    sin = sin.view(*shape)  # e.g. (1, 10, 1, 10) if x.ndim==3 and seq_dim=1
    cos = cos.view(*shape)
    # x1, x2: same shape as x but last dim is half, e.g. (2, 10, 10)
    x1, x2 = x.split(half, dim=-1)
    # x1 * cos - x2 * sin: (same as x1), x1 * sin + x2 * cos: (same as x2)
    x_rope = torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)  # (same as x)
    return x_rope

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim_model // n_heads
        self.q_proj = nn.Linear(dim_model, dim_model)
        self.k_proj = nn.Linear(dim_model, dim_model)
        self.v_proj = nn.Linear(dim_model, dim_model)
        self.out_proj = nn.Linear(dim_model, dim_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value):
        # query: [seq_len_q, batch, dim], key/value: [seq_len_kv, batch, dim]
        seq_len_q, batch, dim = query.shape
        seq_len_kv = key.shape[0]
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)
        # RoPE for Q, K
        Q = apply_rope(Q, seq_dim=0)
        K = apply_rope(K, seq_dim=0)
        # 多头
        def split_heads(x):
            return x.view(seq_len_q if x is Q else seq_len_kv, batch, self.n_heads, self.head_dim).permute(2, 1, 0, 3)
        Q = split_heads(Q)  # [n_heads, batch, seq_len_q, head_dim]
        K = split_heads(K)  # [n_heads, batch, seq_len_kv, head_dim]
        V = split_heads(V)  # [n_heads, batch, seq_len_kv, head_dim]
        # 注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, V)  # [n_heads, batch, seq_len_q, head_dim]
        context = context.permute(2, 1, 0, 3).contiguous().view(seq_len_q, batch, self.n_heads * self.head_dim)
        out = self.out_proj(context)
        return out

class VisionLanguageDiscreteQwenPolicy(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, dim_model=512, n_heads=8, n_layers=4, action_dim=14, n_bins=256, dropout=0.1, lang_tokenizer=None, max_lang_len=32):
        super().__init__()
        self.vit = SimpleViT(img_size, patch_size, in_channels, dim_model, n_heads, n_layers, dropout)
        self.lang_tokenizer = lang_tokenizer
        self.max_lang_len = max_lang_len
        self.action_dim = action_dim
        self.n_bins = n_bins
        self.lang_embed = nn.Embedding(self.lang_tokenizer.vocab_size, dim_model)
        self.lang_pos_embed = nn.Parameter(torch.randn(self.max_lang_len, dim_model))
        self.cross_attn = CrossAttentionBlock(dim_model, n_heads, dropout)
        self.action_head = nn.Linear(dim_model, action_dim * n_bins)
        self.action_tokenizer = SimpleActionTokenizer(action_dim, n_bins)

    def forward(self, img, lang_text):
        batch_size = img.shape[0]
        # 1. 图像token（加RoPE）
        visual_tokens = self.vit(img)  # [batch, n_visual_tokens+1, dim]
        visual_tokens = visual_tokens.transpose(0,1)  # [seq_v, batch, dim]
        visual_tokens = apply_rope(visual_tokens, seq_dim=0)
        # 2. 语言token（加RoPE）
        if isinstance(lang_text, str):
            lang_text = [lang_text] * batch_size
        lang_enc = self.lang_tokenizer(lang_text, padding='max_length', truncation=True, max_length=self.max_lang_len, return_tensors='pt')
        input_ids = torch.tensor(lang_enc['input_ids']).to(img.device)
        lang_tokens = self.lang_embed(input_ids) + self.lang_pos_embed.unsqueeze(0)  # [batch, seq_l, dim]
        lang_tokens = lang_tokens.transpose(0,1)  # [seq_l, batch, dim]
        lang_tokens = apply_rope(lang_tokens, seq_dim=0)
        # 3. Cross-Attention: 语言token为query，视觉token为key/value
        fused_lang = self.cross_attn(lang_tokens, visual_tokens, visual_tokens)  # [seq_l, batch, dim]
        # 4. 取第一个语言token（如cls）做动作预测
        cls_token = fused_lang[0]  # [batch, dim]
        logits = self.action_head(cls_token)  # [batch, action_dim * n_bins]
        logits = logits.view(batch_size, self.action_dim, self.n_bins)
        probs = torch.softmax(logits, dim=-1)
        return probs

    def compute_loss(self, img, lang_text, expert_actions):
        probs = self.forward(img, lang_text)
        expert_bins = self.action_tokenizer.encode(expert_actions)
        batch_size, action_dim, n_bins = probs.shape
        log_probs = torch.log(probs + 1e-8)
        log_probs = log_probs.view(-1, n_bins)
        expert_bins = expert_bins.view(-1)
        loss = nn.functional.nll_loss(log_probs, expert_bins, reduction='mean')
        return loss


# 3. 测试脚本：随机生成输入数据
if __name__ == '__main__':
    # 假设输入有图像特征、状态特征等，先用随机数据模拟
    batch_size = 2  # 批大小
    seq_len = 100   # 序列长度（chunk_size）
    dim_model = 512 # transformer隐藏层维度
    action_dim = 14 # 动作维度

    # 随机生成输入特征（如图像、状态等经过编码后的特征）
    # x的shape: [seq_len, batch_size, dim_model]
    x = torch.randn(seq_len, batch_size, dim_model)

    # 新增：测试ViT+Transformer视觉动作预测
    print('\nVisionTransformerPolicy:')
    img = torch.randn(batch_size, 3, 224, 224)  # 模拟机器人摄像头图像
    # 新增：测试视觉-语言-动作模型
    print('\nVisionLanguageTransformerPolicy:')
    vision_lang_policy = VisionLanguageTransformerPolicy(
        img_size=224,
        patch_size=16,
        in_channels=3,
        dim_model=dim_model,
        n_heads=8,
        n_layers=4,
        action_dim=action_dim,
        dropout=0.1,
        lang_tokenizer=lang_tokenizer,
        max_lang_len=32
    )
    lang_instruction = "please help me to pick the cube and place it in the box"
    actions_vl = vision_lang_policy(img, lang_instruction)
    print('actions_vl shape:', actions_vl.shape)
    print('actions_vl:', actions_vl)

    # 新增：测试视觉-语言-离散动作模型
    print('\nVisionLanguageDiscretePolicy:')
    vision_lang_discrete_policy = VisionLanguageDiscretePolicy(
        img_size=224,
        patch_size=16,
        in_channels=3,
        dim_model=dim_model,
        n_heads=8,
        n_layers=4,
        action_dim=action_dim,
        n_bins=256,
        dropout=0.1,
        lang_tokenizer=lang_tokenizer,
        max_lang_len=32
    )
    probs = vision_lang_discrete_policy(img, lang_instruction)
    print('probs shape:', probs.shape)  # [batch_size, action_dim, n_bins]
    print('probs[0,0]:', probs[0,0])  # 第一个样本第一个动作维度的概率分布

    # 新增：生成随机expert连续动作并计算loss（调用类内方法）
    expert_actions = torch.rand(batch_size, action_dim, device=probs.device) * 2 - 1  # 随机专家动作，范围[-1,1]
    loss = vision_lang_discrete_policy.compute_loss(img, lang_instruction, expert_actions)
    print('discrete action ce loss:', loss.item())

    # 新增：测试Qwen风格视觉-语言-离散动作模型
    print('\nVisionLanguageDiscreteQwenPolicy:')
    vision_lang_qwen_policy = VisionLanguageDiscreteQwenPolicy(
        img_size=224,
        patch_size=16,
        in_channels=3,
        dim_model=dim_model,
        n_heads=8,
        n_layers=4,
        action_dim=action_dim,
        n_bins=256,
        dropout=0.1,
        lang_tokenizer=lang_tokenizer,
        max_lang_len=32
    )
    probs_qwen = vision_lang_qwen_policy(img, lang_instruction)
    print('probs_qwen shape:', probs_qwen.shape)
    expert_actions_qwen = torch.rand(batch_size, action_dim, device=probs_qwen.device) * 2 - 1
    loss_qwen = vision_lang_qwen_policy.compute_loss(img, lang_instruction, expert_actions_qwen)
    print('discrete action ce loss (qwen):', loss_qwen.item())

    # 创建模型（可选择位置编码类型：'learned' 或 'sinusoidal'）
    model = SimpleTransformerModel(
        dim_model=dim_model,
        n_heads=8,
        dim_feedforward=3200,
        dropout=0.1,
        n_layers=4,
        seq_len=seq_len,
        action_dim=action_dim,
        pos_embed_type='sinusoidal'  # 可改为'learned'体验不同效果
    )

    # 前向传播
    actions = model(x)
    # actions的shape: [batch_size, seq_len, action_dim]
    print('actions shape:', actions.shape)
    # 输出示例
    print(actions[0, 0, :])  # 第一个batch第一个时间步的动作

    # 测试自定义注意力
    attn = SimpleMultiHeadSelfAttention(dim_model=dim_model, n_heads=8, dropout=0.1)
    attn_out = attn(x)  # [seq_len, batch_size, dim_model]
    print('attn_out shape:', attn_out.shape)

    # 示例：causal mask
    print('\nCausal Mask Transformer:')
    model_causal = SimpleTransformerModel(
        dim_model=dim_model,
        n_heads=8,
        dim_feedforward=3200,
        dropout=0.1,
        n_layers=4,
        seq_len=seq_len,
        action_dim=action_dim,
        pos_embed_type='sinusoidal',
        mask_type='causal'
    )
    actions_causal = model_causal(x)
    print('actions_causal shape:', actions_causal.shape)

    # 示例：自定义mask（如只允许关注偶数位置）
    print('\nCustom Mask Transformer:')
    custom_mask = torch.zeros(seq_len, seq_len).bool()
    custom_mask[1::2, :] = True  # 仅奇数行被mask
    model_custom = SimpleTransformerModel(
        dim_model=dim_model,
        n_heads=8,
        dim_feedforward=3200,
        dropout=0.1,
        n_layers=4,
        seq_len=seq_len,
        action_dim=action_dim,
        pos_embed_type='sinusoidal',
        mask_type='custom'
    )
    actions_custom = model_custom(x, attn_mask=custom_mask)
    print('actions_custom shape:', actions_custom.shape)

# =============================
# 主要变量维度说明：
# x: [seq_len, batch_size, dim_model]  # transformer输入
# actions: [batch_size, seq_len, action_dim]  # transformer输出
# =============================
# 详细中文注释已添加，便于理解每一步
# =============================
