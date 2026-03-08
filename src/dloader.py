from dataclasses import dataclass
from typing import List
import json


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
