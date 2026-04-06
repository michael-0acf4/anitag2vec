from typing import List, Union
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class TagBPETokenizer:
    def __init__(self, tokenizer: Union[Tokenizer, None] = None):
        def gettok():
            tok = Tokenizer(
                models.BPE(unk_token="[UNK]")  # placeholder for handling unknowns
            )
            tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
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
