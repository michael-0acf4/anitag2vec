from dataclasses import asdict, dataclass, field
import json
import os
from typing import List

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from at2v.dloader import ShallowHash, TagDataset
from at2v.tokenizer import TagBPETokenizer

import math

# Note: 
# assert expr are traced from computation graph
# trigger only when debuging
DEBUG = os.getenv("DEBUG", "0") == "1"

class SegmentRoPEMultiheadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len_for_cache_buffer: int = 2048):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.half_d_head = self.d_head // 2

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # RoPE cache static buffer
        # I don't expect indices to exceed 20 honestly
        # but to be safe we max it out to the input size upstream
        freqs = 1.0 / (10000 ** (torch.arange(0, self.half_d_head, 1).float() / self.half_d_head))
        t = torch.arange(max_seq_len_for_cache_buffer)
        angles = torch.outer(t, freqs)

        self.register_buffer("cos_cache", torch.cos(angles).unsqueeze(0).unsqueeze(0)) # (1, 1, I, half_d_head)
        self.register_buffer("sin_cache", torch.sin(angles).unsqueeze(0).unsqueeze(0)) # (1, 1, I, half_d_head)

    def apply_rope(self, x: torch.Tensor, pos: torch.Tensor):
        if DEBUG:
            _, _, _, Dh = x.shape
            assert Dh % 2 == 0

        # pos = pos.to(torch.long).clamp(0, self.cos_cache.size(0) - 1) # cannot be statically analyzed
        pos = pos.to(torch.long)
        cos = self.cos_cache[0, 0, pos]  # (B, I, H)
        sin = self.sin_cache[0, 0, pos]  # (B, I, H)
        cos = cos.unsqueeze(1) 
        sin = sin.unsqueeze(1)

        x1 = x[..., :self.half_d_head]
        x2 = x[..., self.half_d_head:]
        rx1 = x1 * cos - x2 * sin
        rx2 = x1 * sin + x2 * cos
        return torch.cat([rx1, rx2], dim=-1)
        # return torch.stack([rx1, rx2], dim=-1).view(x.shape)

    # def forward(self, x: torch.Tensor, pos: torch.Tensor, mask: torch.Tensor = None):
    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        B, I, D = x.shape
        q = self.Wq(x).view(B, I, self.n_heads, self.d_head).transpose(1, 2)
        k = self.Wk(x).view(B, I, self.n_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, I, self.n_heads, self.d_head).transpose(1, 2)

        # RoPE + attn
        q = self.apply_rope(q, pos)
        k = self.apply_rope(k, pos)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        # if mask is not None:
        #     scores = scores.masked_fill(mask == 0, float("-inf"))

        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(B, I, D)
        return self.Wo(out)

class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, max_seq_len_for_cache_buffer: int):
        super().__init__()
        self.attn = SegmentRoPEMultiheadAttention(d_model, n_heads, max_seq_len_for_cache_buffer)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x, pos):
        x = x + self.attn(self.norm1(x), pos)
        x = x + self.ffn(self.norm2(x))
        return x

class TransformerStack(nn.Module):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, max_seq_len_for_cache_buffer: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            Block(d_model, n_heads, max_seq_len_for_cache_buffer)
            for _ in range(n_layers)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor):
        for block in self.blocks:
            x = block(x, pos)
        return x

class AniTag2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len_cut: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        output_emb: int,
        segmented_rope: bool = None
    ):
        super().__init__()
        buff = 100
        self.max_len_cut = max_len_cut
        self.segmented_rope = segmented_rope
        self.emb = nn.Embedding(num_embeddings=vocab_size + buff, embedding_dim=d_model)
        if self.segmented_rope:
            buff = 100
            self.transformer_rope = TransformerStack(
                n_layers=n_layers,
                d_model=d_model,
                n_heads=n_heads,
                # longest sequence is 1, 2, .., 128
                # (max_len_cut + 1) is the minimum cache size
                max_seq_len_for_cache_buffer=max_len_cut + buff
            )
        else:
            self.transformer = nn.TransformerEncoder(
                encoder_layer=nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=n_heads, batch_first=True
                ),
                num_layers=n_layers,
                enable_nested_tensor=False,
            )
        self.linproj = nn.Linear(2 * d_model, output_emb)

    @staticmethod
    def from_config(config: "ModelConfig"):
        return AniTag2Vec(
            vocab_size=config.HYPERP_TAGTOK_VOCAB_SIZE,
            max_len_cut=config.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,
            d_model=config.HYPERP_TRANSFORMER_D_MODEL,
            n_heads=config.HYPERP_TRANSFORMER_N_HEADS,
            n_layers=config.HYPERP_TRANSFORMER_N_LAYERS,
            output_emb=config.HYPERP_OUTPUT_EMB,
            segmented_rope=config.HYPERP_ENABLE_SEGMENTED_ROPE,
        )

    def forward(self, x: torch.Tensor, pos: torch.Tensor = None):
        ix = x                            # (B, I)
        x = self.emb(ix)                  # (B, D, D)
        if self.segmented_rope and pos is not None:
            if DEBUG:
                assert x.shape[0] == pos.shape[0]
                assert x.shape[1] == pos.shape[1]
            x = self.transformer_rope(x, pos)  # (B, D, D)
        else:
            x = self.transformer(x)            # (B, D, D)
        x = torch.cat(
            [
                x.mean(dim=1),            # (B, D) context
                x.max(dim=1).values       # (B, D) highlights
            ],  
            dim=-1,
        )                                 # (B, 2D)
        ox = self.linproj(x)              # (B, O)
        return ox

class AniTag2VecRunner:
    def __init__(
        self,
        tagtok: TagBPETokenizer,
        model: AniTag2Vec
    ):
        self.tokenizer = tagtok
        self.model = model
        self.device = next(model.parameters()).device

    def to_dataloader(self, inputs: List[List[str]]):
        dataset = TagDataset(
            list_of_tags=inputs,
            max_len_cut=self.model.max_len_cut,
            tokenizer=self.tokenizer
        )
        return DataLoader(dataset, batch_size=len(inputs), shuffle=False)

    def run_inference(self, inputs: List[List[str]]) -> torch.Tensor:
        # with torch.no_grad():
        with torch.inference_mode():
            batches = self.to_dataloader(inputs)
            for batch in batches:
                batch = batch.to(self.device)
                if self.model.segmented_rope:
                    pos = self.tokenizer.get_chunked_positions_torch_for_training(batch)
                    return self.model(batch, pos)
                return self.model(batch, None)

    def run_inference_human(self, inputs: List[str]):
        def get_hashtags(text: str):
            return [word[1:] for word in text.split() if word.startswith("#")]
        tagss = [get_hashtags(text) for text in inputs]
        return self.run_inference(tagss)

    def rank_cosim_from_vector(
        self,
        query: torch.Tensor,
        items: List[List[str]],
        best: bool=True
    ):
        q = F.normalize(query, dim=1)                         # (1, O)
        xs = F.normalize(self.run_inference(items), dim=1)    # (N, O)
        scores = (q @ xs.T).squeeze(0)                        # (N,)
        indices = torch.argsort(scores, descending=best)
        ranked_items = [items[i] for i in indices.tolist()]
        return list(zip(scores[indices], ranked_items))

    def rank_cosim(
        self,
        query: List[str],
        items: List[List[str]],
        best: bool=True
    ):
        vec = self.run_inference([query])
        return self.rank_cosim_from_vector(vec, items, best)


@dataclass
class TrainingCfg(ShallowHash):
    TRAINING_EVAL_SPLIT: int
    TRAINING_TEST_SPLIT: int
    TRAINING_BATCH_SIZE: int = 256
    TRAINING_PERM_LIMIT: int = 8
    TRAINING_SUBARRAY_COUNT: int = 5
    TRAINING_SHUFFLE_SEED: int = None
    TRAINING_EPOCHS: int = 10
    TRAINING_LOGITS_TEMPERATURE: float = 0.07
    TRAINING_AUG_DROP_PROB: float = 0.3
    TRAINING_LEARNING_RATE: float = 1e-4
    TRAINING_WEIGHT_DECAY: float = None # 0.01, 0.02


@dataclass
class ModelConfig(ShallowHash):
    HYPERP_TAGTOK_MAX_TOKEN_CLAMP: int = 128
    HYPERP_TAGTOK_VOCAB_SIZE: int = 5000
    HYPERP_TAGTOK_MIN_FREQ: int = 3
    HYPERP_TRANSFORMER_D_MODEL: int = 128
    HYPERP_TRANSFORMER_N_HEADS: int = 8
    HYPERP_TRANSFORMER_N_LAYERS: int = 2
    HYPERP_OUTPUT_EMB: int = 128
    HYPERP_ENABLE_SEGMENTED_ROPE: bool = None

    @classmethod
    def load_from_file(cls, path: str) -> "ModelConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

@dataclass
class LossLogger(ShallowHash):
    training_epoch_losses: List[float] = field(default_factory=list)
    eval_epoch_losses: List[float] = field(default_factory=list)
    test_losses: List[float] = field(default_factory=list)
    training_config: TrainingCfg = field(default_factory=TrainingCfg)

    @classmethod
    def load_from_file(cls, path: str) -> "LossLogger":
        with open(path, "r") as f:
            data = json.load(f)
            if "training_config" in data:
                data["training_config"] = TrainingCfg(**data["training_config"])
        return cls(**data)

    def add_avg_training_loss(self, loss: float):
        self.training_epoch_losses.append(loss)

    def add_avg_eval_loss(self, loss: float):
        self.eval_epoch_losses.append(loss)

    def add_test_loss(self, loss: float):
        self.test_losses.append(loss)
