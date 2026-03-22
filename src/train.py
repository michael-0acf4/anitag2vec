import torch
from at2v.dloader import TagDataset
from at2v.anitag2vec import SetupConfig, get_setup
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import asciichartpy as acp

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

def train(
    cfg: SetupConfig
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, tagtok, anitag2vec = get_setup(
        cfg,
        device,
        prefix_path="."
    )

    tags = data.tags[:cfg.TRAINING_TAKE_EXAMPLES]
    dataset = TagDataset(
        tags,
        tokenizer=tagtok,
        max_len_cut=cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP
    )
    print(f"Loaded {len(dataset)} training examples")
    dataloader = DataLoader(dataset, batch_size=cfg.TRAINING_BATCH_SIZE, shuffle=True)

    total_params = sum(p.numel() for p in anitag2vec.parameters())
    model_output = f"checkpoints/anitag2vec_i{cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP}_e{cfg.HYPERP_EPOCHS}_s{len(tags)}_b{cfg.TRAINING_BATCH_SIZE}_p{total_params}.pth"
    print(f"Cooking model with {total_params:,} parameters")

    optimizer = torch.optim.Adam(anitag2vec.parameters(), lr=1e-4)
    anitag2vec.train()
    hide_id = tagtok.pad_token_id()

    errors = []
    p_epochs = tqdm(range(cfg.HYPERP_EPOCHS), desc="Epochs")
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

    torch.save(anitag2vec.state_dict(), model_output)
    print(acp.plot(errors))
    print("Done!")


hpfile = "setup_params.json" 
cfg = SetupConfig.load_from_file(hpfile)
# cfg.HYPERP_EPOCHS = 15
cfg.dump_to_file(hpfile)
train(cfg)
