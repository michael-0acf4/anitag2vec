from dataclasses import dataclass
from itertools import permutations
from typing import List
import json
import torch
from torch.utils.data import Dataset
import random

from at2v.tokenizer import TagBPETokenizer


@dataclass
class MergeSet:
    tags: List[str]
    real_examples: List[List[str]]

    @staticmethod
    def from_file(p: str) -> "MergeSet":
        with open(p, "r", encoding="utf-8") as f:
            raw = json.load(f)
            return MergeSet(tags=raw["tags"], real_examples=raw["real_examples"])
        raise FileNotFoundError

    def extend_with_synthetic(self, perm_limit=5, sub_array_count=5) -> List[List[str]]:
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

        return extended


class TagDataset(Dataset):
    def __init__(self, list_of_tags: List[List[str]], tokenizer: TagBPETokenizer, max_len_cut=16):
        self.list_of_tags = list_of_tags
        self.tokenizer = tokenizer
        self.max_len_cut = max_len_cut

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
