from dloader import MergeSet
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders


class TagBPETokenizer:
    def __init__(self, vocab_size=10000, min_frequency=2):
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.tokenizer = Tokenizer(
            models.BPE(unk_token="[UNK]")  # placeholder for handling unknowns
        )
        self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        self.tokenizer.decoder = decoders.ByteLevel()

    def train(self, tags: list[str], save_path: str = None):
        trainer = trainers.BpeTrainer(
            vocab_size=self.vocab_size,
            min_frequency=self.min_frequency,
            special_tokens=["[UNK]"],
        )
        self.tokenizer.train_from_iterator(tags, trainer=trainer)

        if save_path:
            self.tokenizer.save(save_path)
            print(f"Tokenizer saved to {save_path}")

    def load(self, path: str):
        self.tokenizer = Tokenizer.from_file(path)

    def encode(self, text: str):
        return self.tokenizer.encode(text).tokens

    def encode_ids(self, text: str):
        return self.tokenizer.encode(text).ids
