import torch
import torch.nn as nn


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
