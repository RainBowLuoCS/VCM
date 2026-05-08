#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from abc import ABC, abstractmethod
import torch.distributed as dist
import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape
import math
import torch
from torch import einsum, nn
import torch.nn.functional as F
import numpy as np
from transformers.modeling_attn_mask_utils import AttentionMaskConverter
from transformers.models.llama.modeling_llama import LlamaConfig,LlamaMLP,LlamaRMSNorm
from functools import partial

## extra dependency

import copy
from einops import rearrange, repeat
from einops_exts import rearrange_many

def get_abs_pos(abs_pos, tgt_size):
    # abs_pos: L, C
    # tgt_size: (H, W)
    # return: M, C
    src_size = int(math.sqrt(abs_pos.size(0)))
    # tgt_size = int(math.sqrt(tgt_size))
    dtype = abs_pos.dtype
    return F.interpolate(
        abs_pos.float().reshape(1, src_size, src_size, -1).permute(0, 3, 1, 2),
        size=(tgt_size[0], tgt_size[1]),
        mode="bicubic",
        align_corners=False,
    ).permute(0, 2, 3, 1).flatten(0, 2).to(dtype=dtype)


# https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/pos_embed.py#L20
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    

    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float32)
    omega /= embed_dim / 2.
    omega = 1. / 10000 ** omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def coefficient_slow_edges(l, p, a, b, k=5):
    """
    Compute a coefficient based on the input length l and maximum length p.
    The coefficient is designed to:
        - Return a value close to 'a' when l is close to 0.
        - Return a value close to 'b' when l is close to p.
        - Change quickly in the middle and slowly near the edges.
    
    Args:
        l (float or torch.Tensor): Input length (current length).
        p (float): Maximum length.
        a (float): Minimum value of the coefficient.
        b (float): Maximum value of the coefficient.
        k (float): Growth rate parameter (default=5).
    
    Returns:
        torch.Tensor: Coefficient value.
    """
    # Ensure l is a tensor for compatibility
    l = torch.tensor(l, dtype=torch.float32) if not isinstance(l, torch.Tensor) else l
    
    # Normalize l by p to ensure input is in [0, 1]
    l_normalized = l / p
    
    # Compute the smooth S-shaped function using tanh
    s = (1 + torch.tanh(k * (2 * l_normalized - 1))) / 2
    
    # Scale and shift to [a, b]
    coef = a + (b - a) * s
    
    return coef


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class LlamaFusionAttention(nn.Module):

    def __init__(self, config, layer_idx):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.attention_dropout = config.attention_dropout
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        kv_hidden_states,
        attention_mask=None,
    ):
        bsz, q_len, _ = hidden_states.size()

        bsz, kv_len, _ = kv_hidden_states.size()

        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(kv_hidden_states)
        value_states = self.v_proj(kv_hidden_states)

        # print(key_states.shape,value_states.shape,query_states.shape)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(bsz, kv_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        kv_seq_len = key_states.shape[-2]

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

        attn_output = self.o_proj(attn_output)

        return attn_output
    
class LlamaFusionLayer(nn.Module):

    def __init__(self, config: LlamaConfig, layer_idx: int):

        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = LlamaFusionAttention(config=config, layer_idx=layer_idx)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.kv_input_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = LlamaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states,
        kv_hidden_states,
        attention_mask= None,
    ):

        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        kv_hidden_states=self.kv_input_layernorm(kv_hidden_states)

        # Self Attention
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            kv_hidden_states=kv_hidden_states,
            attention_mask=attention_mask,
        )

        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        return outputs
    
def merge_segments(x, x_mask):
    B, S, D = x.size()

    masked_x = x * x_mask.unsqueeze(-1)
    expanded_x_mask = torch.cat([x_mask, torch.zeros((B, 1), dtype=x_mask.dtype,
                                                     device=x_mask.device)], dim=1)

    cumsum_mask = torch.cumsum(expanded_x_mask, dim=1)
    cumsum_x = torch.cumsum(masked_x, dim=1)

    segment_mask = x_mask * ((cumsum_mask[:, 1:] - cumsum_mask[:, :-1]) == 0)
    new_x = []

    for i in range(B):
        cumsum_x_current = cumsum_x[i]
        cumsum_mask_current = cumsum_mask[i][:-1]
        segment_mask_current = segment_mask[i].bool()

        segment_sums = cumsum_x_current[segment_mask_current, :]
        segment_counts = cumsum_mask_current[segment_mask_current]

        expand_segment_sums = torch.cat([torch.zeros((1, D), dtype=segment_sums.dtype,
                                                     device=segment_sums.device), segment_sums])
        expand_segment_counts = torch.cat([torch.zeros((1,), dtype=segment_counts.dtype,
                                                       device=segment_counts.device), segment_counts])

        diff_sums = expand_segment_sums[1:, :] - expand_segment_sums[:-1, :]
        diff_counts = expand_segment_counts[1:] - expand_segment_counts[:-1]

        diff_counts = torch.where(diff_counts == 0, torch.ones_like(diff_counts), diff_counts)

        mean_segments = diff_sums / diff_counts.unsqueeze(-1)

        new_x.append(mean_segments)

    return new_x

def merge_segments_with_scores(x, x_mask, scores):
    """
    按照分数加权平均将相邻 mask 为 1 的特征合并，mask 为 0 的特征丢弃。

    Args:
        x (torch.Tensor): 输入特征，形状为 [B, S, D]。
        x_mask (torch.Tensor): 二值 mask，形状为 [B, S]。
        cls_x (torch.Tensor): 每个样本的 cls 特征，形状为 [B, D]。
        scores (torch.Tensor): 每个特征的分数，形状为 [B, S]。

    Returns:
        new_x (list[torch.Tensor]): 合并后的特征，每个元素形状为 [N, D]。
    """
    B, S, D = x.size()

    # 将特征乘以 mask，屏蔽掉不需要的特征
    masked_x = x * x_mask.unsqueeze(-1)
    masked_scores = scores * x_mask  # 只保留 mask 为 1 的分数

    # 扩展 mask，用于计算累计和
    expanded_x_mask = torch.cat([x_mask, torch.zeros((B, 1), dtype=x_mask.dtype,
                                                     device=x_mask.device)], dim=1)
    expanded_scores = torch.cat([scores, torch.zeros((B, 1), dtype=scores.dtype,
                                                     device=scores.device)], dim=1)

    # 计算累计和
    cumsum_mask = torch.cumsum(expanded_x_mask, dim=1)
    cumsum_x = torch.cumsum(masked_x * scores.unsqueeze(-1), dim=1)  # 特征乘以分数后累计
    cumsum_scores = torch.cumsum(masked_scores, dim=1)  # 分数的累计和

    # 找到每个 segment 的边界
    segment_mask = x_mask * ((cumsum_mask[:, 1:] - cumsum_mask[:, :-1]) == 0)
    new_x = []

    for i in range(B):
        # 当前样本的累计和和 segment mask
        cumsum_x_current = cumsum_x[i]
        cumsum_scores_current = cumsum_scores[i]
        cumsum_mask_current = cumsum_mask[i][:-1]
        segment_mask_current = segment_mask[i].bool()

        # 获取每个 segment 的累计和
        segment_sums = cumsum_x_current[segment_mask_current, :]  # [N, D]
        segment_score_sums = cumsum_scores_current[segment_mask_current]  # [N]

        # 扩展累计和，用于计算每个 segment 的值
        expand_segment_sums = torch.cat([torch.zeros((1, D), dtype=segment_sums.dtype,
                                                     device=segment_sums.device), segment_sums])
        expand_segment_score_sums = torch.cat([torch.zeros((1,), dtype=segment_score_sums.dtype,
                                                           device=segment_score_sums.device), segment_score_sums])

        diff_sums = expand_segment_sums[1:, :] - expand_segment_sums[:-1, :]  # [N, D]
        diff_score_sums = expand_segment_score_sums[1:] - expand_segment_score_sums[:-1]  # [N]

        # 防止分母为 0
        diff_score_sums = torch.where(diff_score_sums == 0, torch.ones_like(diff_score_sums), diff_score_sums)

        # 按分数加权平均
        weighted_mean_segments = diff_sums / diff_score_sums.unsqueeze(-1)  # [N, D]

        # 将 cls 特征拼接到结果中
        new_x.append(weighted_mean_segments)

    return new_x

def lengths_to_padding_mask(lens):
    bsz, max_lens = lens.size(0), torch.max(lens).item()
    mask = torch.arange(max_lens).to(lens.device).view(1, max_lens)
    mask = mask.expand(bsz, -1) >= lens.view(bsz, 1).expand(-1, max_lens)
    return mask

import torch

def batch_cosine_similarity(x_point, y_point):
    """
    计算一组点拟合直线与 y=x 的余弦相似度。
    :param x_point: 张量，形状为 (B, N)，表示批次中的 x 坐标
    :param y_point: 张量，形状为 (B, N)，表示批次中的 y 坐标
    :return: 张量，形状为 (B,)，表示每批次拟合直线与 y=x 的余弦相似度
    """
    # 计算批次大小和点数量
    B, N = x_point.shape

    # 计算 x 和 y 的均值
    x_mean = x_point.mean(dim=1, keepdim=True)  # (B, 1)
    y_mean = y_point.mean(dim=1, keepdim=True)  # (B, 1)

    # 中心化 x 和 y
    x_centered = x_point - x_mean  # (B, N)
    y_centered = y_point - y_mean  # (B, N)

    # 计算最小二乘法斜率 k = Cov(x, y) / Var(x)
    # Cov(x, y) = sum((x - mean_x) * (y - mean_y))
    # Var(x) = sum((x - mean_x)^2)
    cov_xy = torch.sum(x_centered * y_centered, dim=1)  # (B,)
    var_x = torch.sum(x_centered ** 2, dim=1)  # (B,)
    k = cov_xy / (var_x + 1e-8)  # 防止除以零

    # 计算拟合直线方向向量 (1, k)，与 y=x 的方向向量 (1, 1) 的余弦相似度
    # 余弦相似度公式：cos_sim = (v1 · v2) / (||v1|| * ||v2||)
    v1 = torch.stack([torch.ones_like(k), k], dim=1)  # (B, 2)，拟合直线向量
    v2 = torch.tensor([1.0, 1.0], device=x_point.device).view(1, 2)  # (1, 2)，y=x 向量

    # 计算点积 v1 · v2
    dot_product = torch.sum(v1 * v2, dim=1)  # (B,)

    # 计算向量模长 ||v1|| 和 ||v2||
    v1_norm = torch.norm(v1, dim=1)  # (B,)
    v2_norm = torch.norm(v2, dim=1)  # 标量

    # 计算余弦相似度
    cos_sim = dot_product / (v1_norm * v2_norm + 1e-8)  # (B,)

    return cos_sim

import numpy as np

def generate_exponential_sequence(A, B, n, k=2):
    """
    生成从 B 到 A 的长度为 n 的整数序列，以指数方式递增。
    
    参数:
    A : int - 最大值
    B : int - 最小值
    n : int - 序列长度
    k : float - 指数参数 (默认值为 2)
    
    返回:
    List[int] - 长度为 n 的整数序列
    """
    # 生成指数递增的浮点数序列
    sequence = [B + (A - B) * (i / (n - 1))**k for i in range(n)]
    
    # 将序列四舍五入为整数
    return [int(round(x)) for x in sequence]

@torch.no_grad()
def sinkhorn(out, vocab_dist, sinkhorn_iterations=3, epsilon=0.005, temperature=0.05):
    Q = torch.exp(out / epsilon).t()
    B = Q.shape[1]
    K = Q.shape[0] 
    vocab_dist_prob = (vocab_dist / temperature).softmax(dim=0)
    for it in range(sinkhorn_iterations):
        Q *= vocab_dist_prob
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    Q *= B # the colomns must sum to 1 so that Q is an assignment Q: K x B
    return Q.t()    # B x K

def mask_mean(x,mask):
    return (x*mask[...,None]).sum(1) / mask[...,None].sum(1).clamp(min=1)

class VisionConceptModel(nn.Module):
    def __init__(
        self,
        config,
        depth=6,
    ):
        super().__init__()

        dim=config.hidden_size

        self.t_dim=dim

        self.v_dim=1024

        self.t_heads=16

        self.v_heads=16

        self.resampler=nn.MultiheadAttention(embed_dim=self.v_dim,
                                            num_heads=self.v_heads,
                                            batch_first=True)
        
        self.semantic=nn.MultiheadAttention(embed_dim=self.t_dim,
                                            num_heads=self.t_heads,
                                            batch_first=True)
        
        self.ln_vq = nn.LayerNorm(self.v_dim,eps=1e-6)
        self.ln_vk = nn.LayerNorm(self.v_dim,eps=1e-6)
        self.ln_vv = nn.LayerNorm(self.v_dim,eps=1e-6)

        self.ln_tq = nn.LayerNorm(self.t_dim,eps=1e-6)
        self.ln_tk = nn.LayerNorm(self.t_dim,eps=1e-6)
        self.ln_tv = nn.LayerNorm(self.t_dim,eps=1e-6)

        self.v2tproj = nn.Parameter((self.t_dim ** -0.5) * torch.randn(self.v_dim, self.t_dim))
        self.t2vproj = nn.Parameter((self.v_dim ** -0.5) * torch.randn(self.t_dim, self.v_dim))

        self.t2tproj = nn.Parameter((self.t_dim ** -0.5) * torch.randn(self.t_dim, self.t_dim))

        self.linear_head=nn.Parameter((self.t_dim ** -0.5) * torch.randn(self.t_dim, 2))

        self.ln_sle = nn.LayerNorm(self.t_dim,eps=1e-6)

        self.ln_text = nn.LayerNorm(self.t_dim,eps=1e-6)

        self.ln_text_query = nn.LayerNorm(self.v_dim,eps=1e-6)

        semantic_pior = torch.load("./LLaVA/vicuna_7b_semantic.pt")
        semantic_pior = torch.FloatTensor(semantic_pior).unsqueeze(1)
        semantic_pior = semantic_pior.clamp(min=1) / semantic_pior.sum(dim=0, keepdim=True)
        self.semantic_pior=semantic_pior
        self.image_len = 256

        self.image_query=nn.Parameter(torch.zeros(self.image_len,self.v_dim))
        # self.mask2_query=nn.Parameter(torch.zeros(1,self.t_dim)).requires_grad_(False)

        nn.init.trunc_normal_(self.image_query, std=.02)
        # nn.init.trunc_normal_(self.mask2_query, std=.02)

        self.thres=0.5
        self.scale=4/9
        self.mask_scheduler=None
        self.min_image_tokens=2
        grid_size = int(math.sqrt(self.image_len))
        self.pos_embed = nn.Parameter(
            torch.from_numpy(get_2d_sincos_pos_embed(self.v_dim, grid_size)).half()
        ).requires_grad_(False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward(self, all_text_reps, all_text_padding_mask,all_text_attention_mask,
                text_reps, image_features, image_exist_mask,
                pseudo_mask_ratios=None,
                lm_head=None,lm_model=None):
        
        loss=0

        stage='sft' # 'sft' or 'pt'
        
        image_lens = torch.LongTensor([len(rep) for rep in image_features]).to(image_features[0].device)

        image_reps = torch.nn.utils.rnn.pad_sequence(image_features, batch_first=True)

        image_padding_mask = lengths_to_padding_mask(image_lens)

        assert torch.sum(image_padding_mask).item()==0,"assume image features with the some length"

        text_reps=all_text_reps

        b,n,_=text_reps.shape

        if stage=='sft':
            with torch.no_grad():
                t_out = self.semantic(
                    self.ln_tq(text_reps),
                    self.ln_tk(text_reps),
                    self.ln_tv(text_reps), 
                    attn_mask=~all_text_attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.t_heads, n, 1).flatten(0,1))[0]
                
                t_out=self.ln_text(t_out@self.t2tproj)
        else:
            t_out = self.semantic(
                    self.ln_tq(text_reps),
                    self.ln_tk(text_reps),
                    self.ln_tv(text_reps), 
                    attn_mask=~all_text_attention_mask.unsqueeze(1).unsqueeze(1).repeat(1, self.t_heads, n, 1).flatten(0,1))[0]
                
            t_out=self.ln_text(t_out@self.t2tproj)

        global_text_reps= mask_mean(t_out,all_text_attention_mask)

        global_image_reps=mask_mean(image_reps,~image_padding_mask) @ self.v2tproj

        # print(t_out,global_image_reps)

        key_text_logit=torch.bmm(t_out,global_image_reps.unsqueeze(1).transpose(2,1))[:,:,0]

        # key_text_prob=F.softmax(key_text_logit+~all_text_attention_mask*-1000,dim=1)

        # mean_key_prob=torch.tensor([torch.mean(x[i]) for i,x in zip(all_text_attention_mask,key_text_prob)]).type_as(key_text_prob)

        # key_text_mask=key_text_prob>mean_key_prob[:,None]

        # print(torch.sum(key_text_mask,dim=-1))


        key_question_text_prob=F.softmax(key_text_logit+all_text_padding_mask*-1000,dim=1)

        key_answer_text_prob=F.softmax(key_text_logit+(~(all_text_attention_mask*all_text_padding_mask))*-1000,dim=1)
        # print(key_text_prob)

        mean_key_question_prob=torch.tensor([torch.mean(x[i]) for i,x in zip(~all_text_padding_mask*all_text_attention_mask,key_question_text_prob)]).type_as(key_question_text_prob)

        mean_key_answer_prob=torch.tensor([torch.mean(x[i]) for i,x in zip(all_text_padding_mask*all_text_attention_mask,key_answer_text_prob)]).type_as(key_answer_text_prob)

        key_question_text_mask=(key_question_text_prob>mean_key_question_prob[:,None])*(all_text_attention_mask)

        key_answer_text_mask=(key_answer_text_prob>mean_key_answer_prob[:,None])*(all_text_attention_mask)

        # print()

        if stage=='sft':
            S = torch.rand(b, 1)
        else:
            S = torch.zeros(b, 1)

        random_probs = torch.rand(b,n)

        vcm_mask = (random_probs > S).int().to(t_out.device)

        # t_out[(key_question_text_mask+key_answer_text_mask)*vcm_mask]=0

        # print(torch.sum(key_text_mask*vcm_mask))

        def kl_loss(text_reps,text_mask,image_reps=None,lm_head=None):

            text_reps=torch.stack([torch.mean(x[m],dim=0) for x,m in zip(text_reps,text_mask)])

            # image_reps=torch.mean(image_reps,dim=1)

            text_features = nn.functional.normalize(text_reps, dim=-1, p=2)
            image_features = nn.functional.normalize(image_reps, dim=-1, p=2)

            prototypes = nn.functional.normalize(lm_head.weight, dim=-1, p=2)

            code_v = torch.mm(image_features.float(), prototypes.float().detach().T)
            code_t = torch.mm(text_features.float(), prototypes.float().detach().T)

            vocab_dist = self.semantic_pior.to(code_t)
    
            q_t = sinkhorn(code_t, vocab_dist)
            
            kl_loss=-torch.mean(torch.sum(q_t * F.log_softmax(code_v.float() / 0.005, dim=-1), dim=-1))
            # print(kl_loss)
            return kl_loss*0.05
        
        if stage!='sft':
            outputs = lm_model(
            input_ids=None,
            attention_mask=all_text_attention_mask,
            inputs_embeds=text_reps,
            cache_position=None)
            hidden_states = outputs.last_hidden_state
            # print(hidden_states.shape)
            # assert not torch.any(torch.isnan(hidden_states))
            # assert not torch.any(torch.isnan(global_image_reps))
            loss+=kl_loss(hidden_states,all_text_attention_mask,image_reps=global_text_reps,lm_head=lm_head)
            # loss+=kl_loss(hidden_states,all_text_attention_mask,image_reps=global_image_reps,lm_head=lm_head)
            
        global_answer_reps= mask_mean(t_out,all_text_padding_mask*all_text_attention_mask)

        # print(t_out.shape,key_question_text_mask.shape,vcm_mask.shape)

        # t_out[key_question_text_mask*vcm_mask]=self.mask2_query @ self.v2tproj

        global_question_reps= mask_mean(t_out,~all_text_padding_mask*all_text_attention_mask)

        # global_question_reps= mask_mean(t_out,~all_text_padding_mask*all_text_attention_mask*~(key_question_text_mask*vcm_mask))

        # print(torch.sum(~all_text_padding_mask*all_text_attention_mask,dim=1),torch.sum(all_text_padding_mask*all_text_attention_mask,dim=1))

        # print(global_question_reps,global_answer_reps)

        text_query=(global_question_reps-global_answer_reps) @ self.t2vproj 

        # text_query=mask_mean(t_out,all_text_attention_mask) @ self.t2vproj 

        text_query=self.ln_text_query(text_query)

        t,s,_=image_reps.shape

        if t!=b:
            # used for mmmu and seedbench
            text_reps=torch.repeat_interleave(text_reps,t,dim=0)
            text_padding_mask=torch.repeat_interleave(text_padding_mask,t,dim=0)
            b,n,_=text_reps.shape

        pos_embed = get_abs_pos(self.pos_embed, (24,24))
#  
        # v_out = self.resampler(
        #     self.ln_vq((self.image_query.unsqueeze(0).repeat(b, 1, 1) + text_query.unsqueeze(1) + self.pos_embed.unsqueeze(0))),
        #     self.ln_vk(image_reps+pos_embed.unsqueeze(0).repeat(b, 1, 1)),
        #     self.ln_vv(image_reps), attn_mask=None)[0]

        v_out = self.resampler(
            self.ln_vq((self.image_query.unsqueeze(0).repeat(b, 1, 1) + self.pos_embed.unsqueeze(0))),
            self.ln_vk(pos_embed.unsqueeze(0).repeat(b, 1, 1)),
            self.ln_vv(image_reps), attn_mask=None)[0]

        x = v_out @ self.v2tproj

        # print(x[0,0],x[0,1],x[1,0])

        if stage!='sft':
            loss+=kl_loss(hidden_states,all_text_attention_mask,image_reps=torch.mean(x,dim=1),lm_head=lm_head)

        if stage=='sft':
            keywords=3*torch.sum(key_question_text_mask*~vcm_mask,dim=-1)-torch.sum(key_answer_text_mask,dim=-1)
            
            select_image_reps=[i[:(((torch.clamp(keyword,-30,30)+30)/60)*self.image_len).int()+self.min_image_tokens] for keyword,i in zip(keywords,x)]

            select_image_reps=[torch.cat([i,j],dim=0).to(i) for i,j in zip(torch.mean(x,dim=1,keepdim=True),select_image_reps)]
            
            import random
            matry_list = range(2, 258, 2)
            num=random.choice(matry_list)
            # num=256

            select_image_reps=x[:,:num]

            select_image_reps=[torch.cat([i,j[:num]],dim=0).to(i) for i,j in zip(torch.mean(x,dim=1,keepdim=True),x)]

            prob_logits=x @ self.linear_head

            prob=F.softmax(prob_logits.float(), dim=-1, dtype=torch.float32)

            select_mask=(prob[:,:,1]>self.thres).bool()

            select_image_reps=merge_segments(x,select_mask)

            select_image_reps=[torch.cat([i,j],dim=0).to(i) for i,j in zip(torch.mean(x,dim=1,keepdim=True),select_image_reps)]
            
        #     # print(prob)
        else:

            # select_image_reps=torch.cat([torch.mean(x,dim=1,keepdim=True),x],dim=1)
            select_image_reps=x
        
        print([(len(i)) for i in select_image_reps])

        

        if self.training and stage=='sft':
            # pass
            pseudo_labels=torch.zeros_like(x[:, :, 0])
            pseudo_lengths=[]

            for i in range(b):
                pseudo_lengths.append(int((1-S[i,0])*(self.image_len*self.scale-self.min_image_tokens))+self.min_image_tokens)
                pseudo_labels[i,:pseudo_lengths[i]]=1
            print(pseudo_lengths)

            lprobs=F.log_softmax(prob_logits.float(),dim=-1).permute(1,0,2)

            vcm_loss=F.ctc_loss(lprobs,
                            pseudo_labels,
                            torch.LongTensor([s]*b).to(lprobs.device),
                            torch.LongTensor(pseudo_lengths).to(lprobs.device),
                            blank=0,
                            reduction='none',
                            zero_infinity=True
                            )
            vcm_loss=torch.sum((vcm_loss)).item()/sum(pseudo_lengths)

            lprobs=F.log_softmax(prob_logits.float(),dim=-1)
            a1=-torch.mean(prob[:,:]*lprobs[:,:])
            a2=torch.mean((torch.sum(prob[:,:,1],dim=1)-torch.Tensor(pseudo_lengths).to(lprobs.device))**2)
            # print(f"vcm_a1: {a1} vcm_a2: {a2}")
            vcm_loss=a1+a2
            loss+=0.4*vcm_loss
            print(f"vcm: {0.4*vcm_loss}")
        if sum(image_exist_mask)==0:
            loss*=0.0
        
        # print(f"total loss: {loss}")

        return select_image_reps, None, loss


class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        self.vcm=VisionConceptModel(config, 2)
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config, delay_load=False)
            self.mm_projector = build_vision_projector(config)

            if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
                self.image_newline = nn.Parameter(
                    torch.empty(config.hidden_size, dtype=self.dtype)
                )

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def get_vcm(self):
        vcm = getattr(self, 'vcm', None)
        if type(vcm) is list:
            vcm = vcm[0]
        return vcm

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower

        if self.get_vision_tower() is None:
            vision_tower = build_vision_tower(model_args)

            if fsdp is not None and len(fsdp) > 0:
                self.vision_tower = [vision_tower]
            else:
                self.vision_tower = vision_tower
        else:
            if fsdp is not None and len(fsdp) > 0:
                vision_tower = self.vision_tower[0]
            else:
                vision_tower = self.vision_tower
            vision_tower.load_model()

        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear')
        self.config.mm_hidden_size = vision_tower.hidden_size
        self.config.mm_vision_select_layer = mm_vision_select_layer
        self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if getattr(self, 'mm_projector', None) is None:
            self.mm_projector = build_vision_projector(self.config)

            if 'unpad' in mm_patch_merge_type:
                embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                self.image_newline = nn.Parameter(
                    torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                )
        else:
            # In case it is frozen by LoRA
            for p in self.mm_projector.parameters():
                p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None:
            mm_projector_weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(mm_projector_weights, 'mm_projector'))


def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor

def rl(x,steps):
    x=torch.repeat_interleave(x,steps,dim=0) if x is not None else x
    return x

def rm_system(x,system_len,has_begin=True):
    if has_begin:
        return torch.cat([x[:,:1],x[:,1+system_len:]],dim=1)
    else:
        return torch.cat([x[:,:0],x[:,0+system_len:]],dim=1)

class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()

    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels,
        images, image_sizes=None
    ):
        # if self.training:
        #     steps=4
        #     input_ids=rl(input_ids,steps)
        #     position_ids=rl(position_ids,steps)
        #     attention_mask=rl(attention_mask,steps)
        #     past_key_values=rl(past_key_values,steps)
        #     labels=rl(labels,steps)
        #     images=rl(images,steps)
        #     if image_sizes is not None:
        #         temp=[]
        #         for i in image_sizes:
        #             temp+=[i for i in range(steps)]
        #         image_sizes=temp

        vision_tower = self.get_vision_tower()
        # print(input_ids.shape)

        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(concat_images)
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(images)

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()

        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        vc_loss=0
        vc_flag=True
        ####
        if vc_flag:

            all_text_input_embeds=self.get_model().embed_tokens(input_ids.clamp(min=0)).detach()

            if self.training:
                x=input_ids*(labels==IGNORE_INDEX).int()
            else:
                x=input_ids

            all_text_padding_mask=(x<=0)

            chunk_text_input_embeds=[]

            images_exist_mask=[]

            for batch_idx, cur_input_ids in enumerate(input_ids):
                # cur_input_ids=cur_input_ids[cur_input_ids!=0]
                # cur_input_ids=cur_input_ids[attention_mask[batch_idx]]
                # cur_labels = labels[batch_idx]
                # cur_labels=cur_labels[attention_mask[batch_idx]]
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                if num_images == 0:
                    # cur_input_embeds = self.get_model().embed_tokens(cur_input_ids)
                    # chunk_text_input_embeds.append(cur_input_embeds[cur_labels==IGNORE_INDEX][33:])
                    images_exist_mask.append(False)
                    continue
                images_exist_mask.append(True)
                # image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                # cur_input_ids_noim = []
                # cur_labels_noim = []
                # for i in range(len(image_token_indices) - 1):
                #     cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                #     cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                # cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                # cur_input_embeds = cur_input_embeds[torch.cat(cur_labels_noim)==IGNORE_INDEX]
                # chunk_text_input_embeds.append(cur_input_embeds[33:])

            lm_model=self.get_model()

            system_prompt_len=33

            all_text_input_embeds=rm_system(all_text_input_embeds,system_prompt_len,has_begin=True)
            
            # mask=1 for anwser and mask=0 for question
            all_text_padding_mask=rm_system(all_text_padding_mask,system_prompt_len,has_begin=True)

            all_text_attention_mask=rm_system(attention_mask,system_prompt_len,has_begin=True)

            vcm=lm_model.get_vcm()

            lm_head=self.get_lm_head()

            new_image_features, new_text_features, vc_loss=vcm(all_text_input_embeds,
                                                               all_text_padding_mask,
                                                               all_text_attention_mask,
                                                               chunk_text_input_embeds,
                                                               image_features,
                                                               images_exist_mask,
                                                               lm_head=lm_head,
                                                               lm_model=lm_model)
            
            # assert torch.sum(new_text_features[0][all_text_padding_mask[0]]-all_text_input_embeds[0][all_text_padding_mask[0]])==0, "right?"

            assert len(image_features)==len(new_image_features)

            image_features=new_image_features

            del new_image_features

            # for i in image_features:
            #     print(i.shape)

            # print(new_text_features.shape)

            # assert new_text_features.size(1)==input_ids.size(1)
        
        vc_flag=False
        ####
        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_input_embeds_no_im=[]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                if vc_flag:
                    cur_input_embeds_no_im.append(new_text_features[batch_idx][image_token_indices[i]+1:image_token_indices[i+1]])
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            if not vc_flag:
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, vc_loss

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
