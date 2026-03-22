import re
from typing import Dict, List

import torch
from at2v.anitag2vec import get_setup, SetupConfig, AniTag2VecRunner
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cfg = SetupConfig.load_from_file("setup_params.json")
data, tagtok, anitag2vec = get_setup(
    cfg,
    device=device,
    prefix_path= "."
)

anitag2vec.load_state_dict(torch.load("checkpoints/anitag2vec_e15_s50000_p1871744.pth"))
anitag2vec.to(device)
anitag2vec.eval()

runner = AniTag2VecRunner(tagtok, anitag2vec)

def normalize(v: torch.Tensor):
    return v / (v.norm(dim=-1, keepdim=True) + 1e-8)

def embed_tags(tags: List[str]) -> torch.Tensor:
    vec = runner.run_inference([tags])
    return vec

# used inside eval
def __(tag_expr: str) -> torch.Tensor:
    tags = [t.strip() for t in tag_expr.split(",") if t.strip()]
    return embed_tags(tags)


def find_closest_vectors(query_vec: torch.Tensor, database: List[List[str]], batch=1000, top_k=10):
    query_vec = normalize(query_vec)
    best = []
    for start in tqdm(range(0, len(database), batch), desc="Processing"):
        end = start + batch
        items = database[start:end]
        ranked = runner.rank_cosim_from_vector(
            query_vec,
            items
        )
        if len(ranked) > 0:
            best.extend(ranked[:top_k])
    best.sort(key=lambda x: -x[0])

    return best[:top_k]


def find_closest(query: List[str], database: List[List[str]], top_k=10):
    qvec = embed_tags(query)
    return find_closest_vectors(qvec, database, top_k=top_k)


def find_furthest(query: List[str], database: List[List[str]], top_k=10):
    qvec = embed_tags(query)
    qvec = normalize(qvec)
    worst = []
    for start in tqdm(range(0, len(database), 1000), desc="Processing"):
        end = start + 1000
        items = database[start:end]
        ranked = runner.rank_cosim_from_vector(qvec, items)
        if len(ranked) > 0:
            worst.extend(ranked[:top_k])
    worst.sort(key=lambda x: x[0])
    return worst[:top_k]

def eval_expr(expr: str, database: List[List[str]], max_items=10):
    to_eval = re.sub(r'"([^"]+)"', r'__("\1")', expr)
    vec = eval(to_eval)
    vec = normalize(vec)
    return find_closest_vectors(vec, database, top_k=max_items)

database = data.real_examples
while True:
    val = input(">> ").strip()
    try:
        if re.search(r'[+\-/*]', val) is None:
            if val.startswith("!"):
                ret = find_furthest(
                    [val.removeprefix("!")],
                    database,
                    10
                )
            else:
                ret = find_closest(
                    [val],
                    database,
                    10
                )
        else:
            ret = eval_expr(val, database, 10)

        print(", ".join([f"{k} {round(v, 3)}" for v, k in ret]))

    except Exception as e:
        raise

    print()