from typing import List, Union
import numpy as np
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
import torch


class TagBPETokenizer:
    def __init__(self, tokenizer: Union[Tokenizer, None] = None):
        def gettok():
            tok = Tokenizer(models.BPE(unk_token="[UNK]"))
            # Note:
            # breaks decoding for chinese/japanese/korean
            # tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

            tok.pre_tokenizer = pre_tokenizers.Whitespace()
            tok.decoder = decoders.ByteLevel()
            return tok

        self.tokenizer = tokenizer or gettok()
        self.special_tokens = [
            "[PAD]",  # ! reserve token id 0 for tensor padding on Tag2Vec
            "[SEP]",
            "[UNK]",
        ]

    def sep_token_id(self):
        return self.tokenizer.token_to_id("[SEP]")

    def pad_token_id(self):
        return self.tokenizer.token_to_id("[PAD]")

    def train(
        self,
        list_of_tags: List[List[str]],
        vocab_size: int,
        min_frequency: int,
        save_path: str = None
    ):
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            special_tokens=self.special_tokens,
        )
        self.tokenizer.train_from_iterator(
            [" ".join(tags) for tags in list_of_tags], trainer=trainer
        )
        if save_path:
            self.tokenizer.save(save_path)
            print(f"Tokenizer saved to {save_path}")
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab_size()

    @classmethod
    def load_from_file(cls, path: str) -> "TagBPETokenizer":
        return TagBPETokenizer(Tokenizer.from_file(path))

    def encode(self, text: str):
        return self.tokenizer.encode(text).tokens

    def encode_ids(self, text: str):
        return self.tokenizer.encode(text).ids

    # FIXME: find a way to include into a proper static computation graph
    # cummax solution is not well supported
    # hence we are forced to move this calculation out of the model
    def get_chunked_positions_torch(self, tokens: torch.Tensor):
        B, I = tokens.shape
        pos = torch.zeros(B, I, dtype=torch.long, device=tokens.device)
        current = torch.ones(B, dtype=torch.long, device=tokens.device)
        for t in range(I):
            entry = tokens[:, t]
            is_sep = torch.logical_or(entry == self.sep_token_id(), entry == self.pad_token_id())
            pos[:, t] = torch.where(is_sep, 0, current)
            current = torch.where(is_sep, 1, current + 1)
        return pos

    # Doesn't care about ONNX
    def get_chunked_positions_torch_for_training(self, tokens: torch.Tensor):
        is_sep = torch.logical_or(tokens == self.sep_token_id(), tokens == self.pad_token_id())
        indices = torch.arange(tokens.size(1), device=tokens.device).unsqueeze(0)
        # 1-based to handle the start of the sequence
        last_sep = torch.cummax(torch.where(is_sep, indices, -1), dim=1).values
        pos = indices - last_sep
        return torch.where(is_sep, 0, pos)

    def get_chunked_positions_numpy(self, tokens: np.ndarray):
        B, I = tokens.shape
        pos = np.zeros((B, I), dtype=np.int64)
        for b in range(B):
            current = 1
            for t in range(I):
                if tokens[b, t] == self.sep_token_id() or tokens[b, t] == self.pad_token_id():
                    pos[b, t] = 0
                    current = 1
                else:
                    pos[b, t] = current
                    current += 1
        return pos
