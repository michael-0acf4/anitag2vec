import json
import re
from typing import Any, List, Tuple

import torch
from at2v.anitag2vec import AniTag2Vec, SetupConfig, AniTag2VecRunner
from at2v.tokenizer import TagBPETokenizer
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = SetupConfig.load_from_file("checkpoints/setup_params_8ea07c7d34b64b69_c7359727bcee4f8b.json")
tagtok = TagBPETokenizer(vocab_size=cfg.HYPERP_TAGTOK_VOCAB_SIZE, min_frequency=cfg.HYPERP_TAGTOK_MIN_FREQ)
tagtok.load("checkpoints/token_dataset_c7359727bcee4f8b_vocab_size_5000_freq_3.json")

anitag2vec = AniTag2Vec(
    vocab_size=tagtok.vocab_size,
    max_len_cut=cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,
    d_model=cfg.HYPERP_TRANSFORMER_D_MODEL,
    n_heads=cfg.HYPERP_TRANSFORMER_N_HEADS,
    n_layers=cfg.HYPERP_TRANSFORMER_N_LAYERS,
    output_emb=cfg.HYPERP_OUTPUT_EMB,
)
anitag2vec.to(device)
anitag2vec.load_state_dict(torch.load("checkpoints/anitag2vec_8ea07c7d34b64b69_c7359727bcee4f8b_i128_e20_s60203_b256_p1871744.pth"))
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
