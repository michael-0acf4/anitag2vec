import json
from typing import Callable, List

import torch
from at2v.dloader import MergeSet, TagDataset
from at2v.anitag2vec import AniTag2Vec, SetupConfig
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import asciichartpy as acp

from at2v.tokenizer import TagBPETokenizer

def augment_tags(
    x: torch.Tensor,
    hide_id: int,
    drop_prob=0.3,
    shuffle=True
):
    device = x.device
    keep_mask = torch.bernoulli(
        torch.full(x.shape, 1 - drop_prob, device=device)
    ).bool()
    x_augmented = torch.where(keep_mask, x, torch.full_like(x, hide_id))
    if shuffle:
        for i in range(x_augmented.size(0)):
            perm = torch.randperm(x_augmented.size(1), device=device)
            x_augmented[i] = x_augmented[i][perm]
    return x_augmented

def compute_loss(
    model: torch.nn.Module,
    pad_id: int,
    batch_data: torch.Tensor,
    temperature=0.07
):
    aug1 = augment_tags(batch_data, pad_id)
    aug2 = augment_tags(batch_data, pad_id)
    o1 = model(aug1)                     # (B, O)
    o2 = model(aug2)                     # (B, O)
    o1 = F.normalize(o1, p=2, dim=1)
    o2 = F.normalize(o2, p=2, dim=1)
    logits = (o1 @ o2.T) / temperature   # (B, B) where diagonal is self-similarity
    loss = F.cross_entropy(
        logits,
        torch.arange(o1.size(0)).to(
            batch_data.device
        ),  # target is the diagonal (item 0 in view1 == item 0 in view2)
    )
    return loss

def save_checkpoint(
    anitag2vec: AniTag2Vec,
    errors: List[int],
    hashsum: str,
    epoch: int,
    model_path: Callable
):
    torch.save(anitag2vec.state_dict(), model_path(epoch))
    cfg.dump_to_file(f"checkpoints/setup_params_{hashsum}.json")
    with open(f"checkpoints/errors_{hashsum}.json", "w") as f:
        json.dump(errors, f)

def train(
    cfg: SetupConfig
):
    cfg_hash = cfg.build_hash()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = MergeSet.from_file(f"{prefix_path}/data/output/merged_tags.json")
    data = MergeSet.from_file(f"./data/output/merged_tags_v2.json")
    data_hash = data.build_hash()
    train_data = data.extend_with_synthetic(
        perm_limit=cfg.TRAINING_PERM_LIMIT,
        sub_array_count=cfg.TRAINING_SUBARRAY_COUNT
    )
    hashsum = f"{cfg_hash}_{data_hash}"
    print(f"Training hash {cfg_hash}")

    tagtok = TagBPETokenizer(
        vocab_size=cfg.HYPERP_TAGTOK_VOCAB_SIZE,
        min_frequency=cfg.HYPERP_TAGTOK_MIN_FREQ
    )
    tagtok_file = f"./checkpoints/token_dataset_{data_hash}_vocab_size_{tagtok.vocab_size}_freq_{tagtok.min_frequency}.json"
    try:
        print(f"Loading tokenizer from '{tagtok_file}'..")
        tagtok.load(tagtok_file)
    except:
        print("Training new tokenizer..")
        tagtok.train(train_data, tagtok_file)

    tags = data.tags[:cfg.TRAINING_TAKE_EXAMPLES]
    dataset = TagDataset(
        tags,
        tokenizer=tagtok,
        max_len_cut=cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP
    )
    print(f"Loaded {len(dataset)} training examples | Hash {data_hash}")
    dataloader = DataLoader(dataset, batch_size=cfg.TRAINING_BATCH_SIZE, shuffle=True)

    anitag2vec = AniTag2Vec(
        vocab_size=tagtok.vocab_size,
        max_len_cut=cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,
        d_model=cfg.HYPERP_TRANSFORMER_D_MODEL,
        n_heads=cfg.HYPERP_TRANSFORMER_N_HEADS,
        n_layers=cfg.HYPERP_TRANSFORMER_N_LAYERS,
        output_emb=cfg.HYPERP_OUTPUT_EMB,
    )
    anitag2vec.to(device)
    total_params = sum(p.numel() for p in anitag2vec.parameters())
    print(f"Cooking model with {total_params:,} parameters")

    optimizer = torch.optim.Adam(anitag2vec.parameters(), lr=1e-4)
    anitag2vec.train()
    hide_id = tagtok.pad_token_id()
    model_path = lambda epochs: f"checkpoints/anitag2vec_{hashsum}_i{cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP}_e{epochs}_s{len(tags)}_b{cfg.TRAINING_BATCH_SIZE}_p{total_params}.pth"
    errors = []
    p_epochs = tqdm(range(cfg.TRAINING_EPOCHS), desc="Epochs")
    for epoch in p_epochs:
        total_loss = 0
        p_batches = tqdm(dataloader, desc="Batches", leave=False)
        for batch in p_batches:
            batch = batch.to(device)
            optimizer.zero_grad()

            loss = compute_loss(anitag2vec, hide_id, batch_data=batch, temperature=0.07)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            p_batches.set_description(f"Epoch {epoch} | Loss: {loss}")

        batch_loss = total_loss / len(dataloader)
        errors.append(batch_loss)
        p_epochs.set_description(f"Mean total Loss: {batch_loss:.4f}")
        if epoch % 5 == 0 and epoch != cfg.TRAINING_EPOCHS and epoch > 0:
            save_checkpoint(anitag2vec, errors, hashsum, epoch, model_path)

    print(acp.plot(errors))
    save_checkpoint(anitag2vec, errors, hashsum, cfg.TRAINING_EPOCHS, model_path)
    print("Done!")


setups = [
    SetupConfig(
        TRAINING_TAKE_EXAMPLES=70000,
        TRAINING_EPOCHS=20, #
        TRAINING_BATCH_SIZE=256, #
        TRAINING_PERM_LIMIT=8, #
        TRAINING_SUBARRAY_COUNT=7, #
        HYPERP_TAGTOK_MAX_TOKEN_CLAMP=128,
        HYPERP_TAGTOK_VOCAB_SIZE=5000,
        HYPERP_TAGTOK_MIN_FREQ=3,
        HYPERP_TRANSFORMER_D_MODEL=128,
        HYPERP_TRANSFORMER_N_HEADS=8,
        HYPERP_TRANSFORMER_N_LAYERS=2,
        HYPERP_OUTPUT_EMB=128,
    ),
]

for cfg in setups:
    train(cfg)

# hpfile = "setup_params.json" 
# cfg = SetupConfig.load_from_file(hpfile)
# # cfg.TRAINING_EPOCHS = 15
# cfg.dump_to_file(hpfile)
# train(cfg)
