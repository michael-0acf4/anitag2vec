import torch
from dloader import MergeSet, TagDataset
from anitag2vec import AniTag2Vec
from tokenizer import TagBPETokenizer
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

data = MergeSet.from_file("data/output/merged_tags.json")
train_data = data.extend_with_synthetic(perm_limit=5, sub_array_count=5)

tagtok = TagBPETokenizer(vocab_size=5000, min_frequency=2)
tagtok_file = f"token_vocab_size_{tagtok.vocab_size}_freq_{tagtok.min_frequency}.json"
try:
    print(f"Loading tokenizer from '{tagtok_file}'..")
    tagtok.load(tagtok_file)
except:
    print("Training new tokenizer..")
    tagtok.train(train_data, tagtok_file)
print("Done!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Training AniTag2Vec...")
anitag2vec = AniTag2Vec(
    vocab_size=tagtok.vocab_size,
    d_model=128,
    n_heads=8,
    n_layers=6,
    output_emb=128,
)
anitag2vec.to(device)

dataset = TagDataset(data.tags, tokenizer=tagtok, max_len_cut=64)
print(len(dataset.list_of_tags), "examples")
dataloader = DataLoader(dataset, batch_size=500, shuffle=True)

optimizer = torch.optim.Adam(anitag2vec.parameters(), lr=1e-4)
anitag2vec.train()

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
    emb1 = model(view1)  # (B, O)
    emb2 = model(view2)  # (B, O)
    emb1 = F.normalize(emb1, p=2, dim=1)
    emb2 = F.normalize(emb2, p=2, dim=1)
    logits = (emb1 @ emb2.T) / temperature  # (B, B) where diagonal is self-similarity
    loss = F.cross_entropy(
        logits,
        torch.arange(emb1.size(0)).to(
            batch_data.device
        ),  # target is the diagonal (item 0 in view1 == item 0 in view2)
    )
    return loss


total_params = sum(p.numel() for p in anitag2vec.parameters())
print(f"Cooking model with {total_params:,} parameters")

p_epochs = tqdm(range(10), desc="Epochs")
for epoch in p_epochs:
    total_loss = 0
    p_batches = tqdm(dataloader, desc="Batches", leave=False)
    for batch in p_batches:
        batch = batch.to(device)
        optimizer.zero_grad()

        logits = anitag2vec(batch)
        loss = compute_loss(anitag2vec, batch_data=batch, temperature=0.07)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        p_batches.set_description(f"Epoch {epoch} | Loss: {loss}")

    p_batches.set_description(f"Mean total Loss: {total_loss / len(dataloader):.4f}")

torch.save(anitag2vec.state_dict(), f"anitag2vec_{total_params}.pth")

print("Done!")
