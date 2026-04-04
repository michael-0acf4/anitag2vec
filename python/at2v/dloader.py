from dataclasses import asdict, dataclass
import hashlib
from itertools import permutations
from typing import List, Optional
import json
import torch
from torch.utils.data import Dataset
import random

from at2v.tokenizer import TagBPETokenizer

@dataclass
class ShallowHash:
    def build_hash(self):
        ser = json.dumps(sorted(asdict(self).items()))
        return hashlib.sha256(ser.encode()).hexdigest()[:16]

    def dump_to_file(self, path: str, indent: int = 2) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=indent)

@dataclass
class MergeSet(ShallowHash):
    tags: List[str]
    real_examples: List[List[str]]

    @staticmethod
    def from_file(p: str) -> "MergeSet":
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return MergeSet(tags=raw["tags"], real_examples=raw["real_examples"])
        raise FileNotFoundError

    def get_extend_with_synthetic_then_shuffle(
        self,
        perm_limit: int,
        sub_array_count: int,
        seed: Optional[int]=None
    ) -> List[List[str]]:
        random.seed(seed)
        extended = []
        for example in self.real_examples:
            extended.append(example)
            if len(example) > 2:
                perms = 0
                for p in permutations(example):
                    extended.append(list(p))
                    perms += 1
                    if perms >= perm_limit:
                        break
                for _ in range(sub_array_count):
                    start = random.randint(0, len(example) - 2)
                    length = random.randint(2, len(example) - start)
                    sub = example[start : start + length]
                    random.shuffle(sub)
                    extended.append(sub)

        random.shuffle(extended)
        return extended


class TagDataset(Dataset):
    def __init__(
        self,
        list_of_tags: List[List[str]],
        tokenizer: TagBPETokenizer,
        max_len_cut: int,
    ):
        self.list_of_tags = list_of_tags
        self.tokenizer = tokenizer
        self.max_len_cut = max_len_cut
        assert max_len_cut > 32

    def __len__(self):
        return len(self.list_of_tags)

    def __getitem__(self, idx: int):
        tag_list = self.list_of_tags[idx]
        ids_list = [self.tokenizer.encode_ids(text) for text in tag_list]
        ids = []
        sep_id = self.tokenizer.sep_token_id()
        for i, curr in enumerate(ids_list):
            ids.extend(curr)
            if i != len(ids_list) - 1:
                ids.append(sep_id)

        if len(ids) > self.max_len_cut:
            ids = ids[: self.max_len_cut]
        else:
            pad_id = self.tokenizer.pad_token_id()
            paddings =  [pad_id] * (self.max_len_cut - len(ids))
            ids = ids + paddings

        return torch.tensor(ids, dtype=torch.long)
