from dataclasses import asdict, dataclass, field
import hashlib
import json
import os
from typing import List
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from at2v.dloader import ShallowHash, TagDataset
from at2v.tokenizer import TagBPETokenizer


class AniTag2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_len_cut: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        output_emb: int,
    ):
        super().__init__()
        buff = 100
        self.max_len_cut = max_len_cut
        self.emb = nn.Embedding(num_embeddings=vocab_size + buff, embedding_dim=d_model)
        self.transformer = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, batch_first=True
            ),
            num_layers=n_layers,
            enable_nested_tensor=False,
        )
        self.linproj = nn.Linear(2 * d_model, output_emb)

    def forward(self, x: torch.Tensor):
        ix = x                            # (B, I)
        x = self.emb(ix)                  # (B, D, D)
        x = self.transformer(x)           # (B, D, D)
        x = torch.cat(
            [
                x.mean(dim=1),             # (B, D) context
                x.max(dim=1).values        # (B, D) highlights
            ],  
            dim=-1,
        )                                  # (B, 2D)
        ox = self.linproj(x)               # (B, O)
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
                return self.model(batch)

    def run_inference_human(self, inputs: List[str]):
        def get_hashtags(text: str):
            return [word[1:] for word in text.split() if word.startswith("#")]
        tagss = [get_hashtags(text) for text in inputs]
        return self.run_inference(tagss)

    def run_inference_human(self, inputs: List[str]):
        def get_hashtags(text: str) -> List[str]:
            return re.findall(r"#([A-Za-z0-9_]+)", text)
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


@dataclass
class ModelConfig(ShallowHash):
    HYPERP_TAGTOK_MAX_TOKEN_CLAMP: int = 128
    HYPERP_TAGTOK_VOCAB_SIZE: int = 5000
    HYPERP_TAGTOK_MIN_FREQ: int = 3
    HYPERP_TRANSFORMER_D_MODEL: int = 128
    HYPERP_TRANSFORMER_N_HEADS: int = 8
    HYPERP_TRANSFORMER_N_LAYERS: int = 2
    HYPERP_OUTPUT_EMB: int = 128

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

    def add_avg_training_loss(self, loss: float):
        self.training_epoch_losses.append(loss)

    def add_avg_eval_loss(self, loss: float):
        self.eval_epoch_losses.append(loss)

    def add_test_loss(self, loss: float):
        self.test_losses.append(loss)
