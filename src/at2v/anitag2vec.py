from dataclasses import asdict, dataclass
import json
import os
from typing import List
import re

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from at2v.dloader import MergeSet, TagDataset
from at2v.tokenizer import TagBPETokenizer


class AniTag2Vec(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        output_emb: int,
    ):
        super().__init__()
        buff = 100
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
    def __init__(self, tagtok: TagBPETokenizer, model: AniTag2Vec):
        self.tokenizer = tagtok
        self.model = model
        self.device = next(model.parameters()).device

    def to_dataloader(self, inputs: List[List[str]]):
        dataset = TagDataset(
            list_of_tags=inputs,
            max_len_cut=64,
            tokenizer=self.tokenizer
        )
        return DataLoader(dataset, batch_size=len(inputs), shuffle=False)

    def run_inference(self, inputs: List[List[str]]):
        # with torch.no_grad():
        with torch.inference_mode():
            batches = self.to_dataloader(inputs)
            for batch in batches:
                batch = batch.to(self.device)
                return self.model(batch)

    def run_inference_human(self, inputs: List[str]):
        def get_hashtags(text: str) -> List[str]:
            return re.findall(r"#([A-Za-z0-9_]+)", text)
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
        items: List[List[str]]
    ):
        q = F.normalize(query, dim=1)                         # (1, O)
        xs = F.normalize(self.run_inference(items), dim=1)    # (N, O)
        scores = (q @ xs.T).squeeze(0)                        # (N,)
        indices = torch.argsort(scores, descending=True)
        ranked_items = [items[i] for i in indices.tolist()]
        return list(zip(scores[indices], ranked_items))

    def rank_cosim(self, query: List[str], items: List[List[str]]):
        query = self.run_inference([query])
        return self.rank_cosim_from_vector(query, items)


@dataclass
class SetupConfig:
    TRAINING_TAKE_EXAMPLES: int = 25000
    TRAINING_BATCH_SIZE: int = 256

    HYPERP_TAGTOK_MAX_TOKEN_CLAMP: int = 64
    HYPERP_TAGTOK_VOCAB_SIZE: int = 5000
    HYPERP_TAGTOK_MIN_FREQ: int = 3

    HYPERP_TRANSFORMER_D_MODEL: int = 64
    HYPERP_TRANSFORMER_N_HEADS: int = 32
    HYPERP_TRANSFORMER_N_LAYERS: int = 2
    HYPERP_OUTPUT_EMB: int = 128
    HYPERP_EPOCHS: int = 15

    @classmethod
    def load_from_file(cls, path: str) -> "SetupConfig":
        if not os.path.exists(path):
            default_config = cls()
            with open(path, "w") as f:
                json.dump(asdict(default_config), f, indent=2)
            return default_config

        with open(path, "r") as f:
            data = json.load(f)

        return cls(**data)
    
    def dump_to_file(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=indent)


def get_setup(
    cfg: SetupConfig,
    device: torch.device,
    prefix_path="."
):
    data = MergeSet.from_file(f"{prefix_path}/data/output/merged_tags.json")
    train_data = data.extend_with_synthetic(perm_limit=5, sub_array_count=5)

    tagtok = TagBPETokenizer(vocab_size=cfg.HYPERP_TAGTOK_VOCAB_SIZE, min_frequency=cfg.HYPERP_TAGTOK_MIN_FREQ)
    tagtok_file = f"{prefix_path}/checkpoints/token_vocab_size_{tagtok.vocab_size}_freq_{tagtok.min_frequency}.json"
    try:
        print(f"Loading tokenizer from '{tagtok_file}'..")
        tagtok.load(tagtok_file)
    except:
        print("Training new tokenizer..")
        tagtok.train(train_data, tagtok_file)
    print("Done!")

    anitag2vec = AniTag2Vec(
        vocab_size=tagtok.vocab_size,
        d_model=cfg.HYPERP_TRANSFORMER_D_MODEL,
        n_heads=cfg.HYPERP_TRANSFORMER_N_HEADS,
        n_layers=cfg.HYPERP_TRANSFORMER_N_LAYERS,
        output_emb=cfg.HYPERP_OUTPUT_EMB,
    )
    anitag2vec.to(device)

    return data, tagtok, anitag2vec
