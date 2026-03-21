import torch
from at2v.dloader import TagDataset
from at2v.anitag2vec import SetupConfig, get_setup
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm


def augment_tags(x, drop_prob=0.15, shuffle=True):
    # random 0 and 1s
    mask = torch.bernoulli(torch.full(x.shape, 1 - drop_prob)).to(x.device)
    # replace dropped tags with [PAD] token id
    x_augmented = x * mask.long()
    if shuffle:
        for i in range(x_augmented.size(0)):
            perm = torch.randperm(x_augmented.size(1))
            x_augmented[i] = x_augmented[i][perm]
    return x_augmented


def compute_loss(model: torch.nn.Module, batch_data: torch.Tensor, temperature=0.07):
    view1 = augment_tags(batch_data)
    view2 = augment_tags(batch_data)
    emb1 = model(view1)                      # (B, O)
    emb2 = model(view2)                      # (B, O)
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    logits = (emb1 @ emb2.T) / temperature   # (B, B) where diagonal is self-similarity
    loss = F.cross_entropy(
        logits,
        torch.arange(emb1.size(0)).to(
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
    model_output = f"checkpoints/anitag2vec_e{cfg.HYPERP_EPOCHS}_s{len(tags)}_p{total_params}.pth"
    print(f"Cooking model with {total_params:,} parameters")

    optimizer = torch.optim.Adam(anitag2vec.parameters(), lr=1e-4)
    anitag2vec.train()

    p_epochs = tqdm(range(cfg.HYPERP_EPOCHS), desc="Epochs")
    for epoch in p_epochs:
        total_loss = 0
        p_batches = tqdm(dataloader, desc="Batches", leave=False)
        for batch in p_batches:
            batch = batch.to(device)
            optimizer.zero_grad()

            loss = compute_loss(anitag2vec, batch_data=batch, temperature=0.07)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            p_batches.set_description(f"Epoch {epoch} | Loss: {loss}")

        p_epochs.set_description(f"Mean total Loss: {total_loss / len(dataloader):.4f}")

    torch.save(anitag2vec.state_dict(), model_output)

    print("Done!")


hpfile = "setup_params.json" 
cfg = SetupConfig.load_from_file(hpfile)
# cfg.HYPERP_EPOCHS = 15
cfg.dump_to_file(hpfile)
train(cfg)
