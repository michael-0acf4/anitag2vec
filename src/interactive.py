import json
import re
from typing import Any, List, Tuple

import torch
from at2v.anitag2vec import get_setup, SetupConfig, AniTag2VecRunner
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = SetupConfig.load_from_file("setup_params.json")
data, tagtok, anitag2vec = get_setup(
    cfg,
    device=device,
    prefix_path= "."
)

anitag2vec.load_state_dict(torch.load("checkpoints/anitag2vec_i128_e20_s59748_b256_p1871744.pth"))
anitag2vec.to(device)
anitag2vec.eval()
runner = AniTag2VecRunner(tagtok, anitag2vec)

def embed_tags(tags: List[str]) -> torch.Tensor:
    return runner.run_inference([tags])

def full_scan(
    query_vec: torch.Tensor,
    database: List[List[str]],
    batch: int,
    best=True
) -> List[Tuple[torch.Tensor, Any]]:
    top = []
    for start in tqdm(range(0, len(database), batch), desc="Scanning"):
        end = start + batch
        items = database[start:end]
        ranked = runner.rank_cosim_from_vector(
            query_vec,
            items,
            best
        )
        if len(ranked) > 0:
            if len(database) < batch:
                top.extend(ranked)
            else:
                top.append(ranked[0])
    sign = -1 if best else 1
    top.sort(key=lambda x: sign * x[0])
    return top

def eval_expr(
    expr: str,
    database: List[List[str]],
    best: bool,
    max_items
):
    try:
        __ = lambda term: embed_tags([t.strip() for t in term.split(",") if t.strip()])
        to_eval = re.sub(r'"([^"]+)"', r'__("\1")', expr)
        top = full_scan(eval(to_eval), database, batch=1000, best=best)
        return top[:max_items]
    except Exception as e:
        print(f"Evaluation failed: {e}")
    return []

# database = data.real_examples

with open("data/mal_5a250b8b201ace01.json", "r", encoding="utf-8") as f:
    database = json.load(f)

top = 5
print('Try an expression like "Drama, Romance, Supernatural" - 2 * "Shounen, TV"')
print("You can also prefix the whole expression with ! to rank from worst")
while True:
    val = input(">> ").strip()
    try:
        best = True
        if val.startswith("!"):
            val = val.removeprefix("!")
            best = False
        if val == "":
            continue
        ret = eval_expr(val, database, best, top)
        if len(ret) > 0:
            print("\n".join([f"{v.item():.2}: {', '.join(k)}" for v, k in ret]))
    except Exception as e:
        raise
    print()
