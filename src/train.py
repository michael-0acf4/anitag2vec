from typing import Callable

import torch
from at2v.dloader import MergeSet, TagDataset
from at2v.anitag2vec import AniTag2Vec, LossLogger, ModelConfig, TrainingCfg
from torch.utils.data import DataLoader
import torch.nn.functional as F

from tqdm import tqdm
import asciichartpy as acp

from at2v.tokenizer import TagBPETokenizer

def augment_tags(
    x: torch.Tensor,
    hide_id: int,
    drop_prob: float,
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
    temperature: float,
    drop_prob: float
):
    aug1 = augment_tags(batch_data, pad_id, drop_prob)
    aug2 = augment_tags(batch_data, pad_id, drop_prob)
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
    model_config: ModelConfig,
    anitag2vec: AniTag2Vec,
    losses: LossLogger,
    hashsum: str,
    epoch: int,
    model_path: Callable
):
    torch.save(anitag2vec.state_dict(), model_path(epoch))
    model_config.dump_to_file(f"checkpoints/config_{hashsum}.json")
    losses.dump_to_file(f"checkpoints/errors_{hashsum}_{losses.training_config.build_hash()}.json")

def ascii_plot(
    losses: LossLogger
):
    config = {"height": 5}
    print("Training losses:")
    print(acp.plot(losses.training_epoch_losses, config))
    print("Eval losses:")
    print(acp.plot(losses.eval_epoch_losses, config))
    print("Test losses:")
    print(acp.plot(losses.test_losses, config))


def train(
    save_every_epoch: int,
    model_config: ModelConfig,
    training_config: TrainingCfg
):
    model_config_hash = model_config.build_hash()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # data = MergeSet.from_file(f"{prefix_path}/data/output/merged_tags.json")
    raw_data = MergeSet.from_file(f"./data/output/merged_tags_v2.json")
    data_hash = raw_data.build_hash()
    ext_train_data = raw_data.get_extend_with_synthetic_then_shuffle(
        perm_limit=training_config.TRAINING_PERM_LIMIT,
        sub_array_count=training_config.TRAINING_SUBARRAY_COUNT,
        seed=training_config.TRAINING_SHUFFLE_SEED #
    )
    hashsum = f"{model_config_hash}_{data_hash}"
    print(f"Training hash {training_config.build_hash()}, hyper parameter hash {hashsum}")

    tagtok = TagBPETokenizer(
        vocab_size=model_config.HYPERP_TAGTOK_VOCAB_SIZE,
        min_frequency=model_config.HYPERP_TAGTOK_MIN_FREQ
    )
    tagtok_file = f"./checkpoints/token_dataset_{data_hash}_vocab_size_{tagtok.vocab_size}_freq_{tagtok.min_frequency}.json"
    try:
        print(f"Loading tokenizer from '{tagtok_file}'..")
        tagtok.load(tagtok_file)
    except:
        print("Training new tokenizer..")
        tagtok.train(ext_train_data, tagtok_file)

    max_len_cut = model_config.HYPERP_TAGTOK_MAX_TOKEN_CLAMP
    anitag2vec = AniTag2Vec(
        vocab_size=tagtok.vocab_size,
        max_len_cut=max_len_cut,
        d_model=model_config.HYPERP_TRANSFORMER_D_MODEL,
        n_heads=model_config.HYPERP_TRANSFORMER_N_HEADS,
        n_layers=model_config.HYPERP_TRANSFORMER_N_LAYERS,
        output_emb=model_config.HYPERP_OUTPUT_EMB,
    )
    anitag2vec.to(device)
    total_params = sum(p.numel() for p in anitag2vec.parameters())
    print(f"Cooking model with {total_params:,} parameters")

    ext_train_data = ext_train_data
    take_eval = training_config.TRAINING_EVAL_SPLIT
    take_test = training_config.TRAINING_TEST_SPLIT
    take_samples = max(0, len(ext_train_data) - (take_eval + take_test))
    assert take_samples > 0 and take_test > 0 and take_eval > 0

    train_end = take_samples
    eval_end = train_end + take_eval
    train_dataset = TagDataset(
        ext_train_data[:train_end],
        tokenizer=tagtok,
        max_len_cut=max_len_cut
    )
    eval_dataset = TagDataset(
        ext_train_data[train_end:eval_end],
        tokenizer=tagtok,
        max_len_cut=max_len_cut
    )
    test_dataset = TagDataset(
        ext_train_data[eval_end:],
        tokenizer=tagtok,
        max_len_cut=max_len_cut
    )
    print(f"Loaded {len(ext_train_data)} total examples | Hash {data_hash}")
    print(f"Splits: training {len(train_dataset)}, eval {len(eval_dataset)}, test {len(test_dataset)}")

    batch_size = training_config.TRAINING_BATCH_SIZE
    epochs_count = training_config.TRAINING_EPOCHS
    temperature = training_config.TRAINING_LOGITS_TEMPERATURE
    drop_prob = training_config.TRAINING_AUG_DROP_PROB
    hide_id = tagtok.pad_token_id()
    print(f"Batch size {batch_size}")

    g = torch.Generator()
    g.manual_seed(training_config.TRAINING_SHUFFLE_SEED)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=g)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, generator=g)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, generator=g)

    optimizer = torch.optim.Adam(anitag2vec.parameters(), lr=training_config.TRAINING_LEARNING_RATE)
    anitag2vec.train()
    model_path = lambda epochs: f"checkpoints/anitag2vec_{hashsum}_i{max_len_cut}_e{epochs}_s{len(train_dataset)}_b{batch_size}_p{total_params}.pth"
    losses = LossLogger(
        training_config=training_config
    )
    p_epochs = tqdm(range(1, epochs_count + 1), desc="Epochs")
    for epoch in p_epochs:
        # training
        training_loss = 0
        p_train = tqdm(train_dataloader, desc="Batches", leave=False)
        for batch in p_train:
            batch = batch.to(device)
            optimizer.zero_grad()
            loss = compute_loss(anitag2vec, hide_id, batch, temperature, drop_prob)
            loss.backward()
            optimizer.step()
            training_loss += loss.item()
            p_train.set_description(f"Epoch {epoch} | Loss: {loss}")
        avg_batch_loss = training_loss / len(train_dataloader)
        losses.add_avg_training_loss(avg_batch_loss)
        p_epochs.set_description(f"Mean total Loss: {avg_batch_loss:.4f}")
        if epoch % save_every_epoch == 0 and epoch != epochs_count:
            save_checkpoint(model_config, anitag2vec, losses, hashsum, epoch, model_path)

        # eval
        p_eval = tqdm(eval_dataloader, desc="Eval", leave=False)
        eval_loss = 0
        for batch in p_eval:
            batch = batch.to(device)
            with torch.no_grad():
                loss = compute_loss(anitag2vec, hide_id, batch, temperature, drop_prob)
            eval_loss += loss.item()
            p_eval.set_description(f"Eval | Loss: {loss}")
        avg_batch_loss = eval_loss / len(eval_dataloader)
        losses.add_avg_eval_loss(avg_batch_loss)
        p_epochs.set_description(f"Mean total Loss: {avg_batch_loss:.4f}")

    # test
    print("Running tests")
    p_test = tqdm(test_dataloader, desc="Test", leave=False)
    for batch in p_test:
        batch = batch.to(device)
        with torch.no_grad():
            loss = compute_loss(anitag2vec, hide_id, batch, temperature, drop_prob)
        loss = loss.item()
        losses.add_test_loss(loss)
        p_test.set_description(f"Test | Loss: {loss}")

    ascii_plot(losses)
    save_checkpoint(model_config, anitag2vec, losses, hashsum, epoch, model_path)

    print("Done!")

# Total is around 196k so 10% ~ 19k
training_configs = [
    TrainingCfg(
        TRAINING_EPOCHS=15,
        TRAINING_EVAL_SPLIT=20_000,
        TRAINING_TEST_SPLIT=19_000,
        TRAINING_BATCH_SIZE=256,
        TRAINING_PERM_LIMIT=8,
        TRAINING_SUBARRAY_COUNT=7,
        TRAINING_LOGITS_TEMPERATURE=0.07,
        TRAINING_AUG_DROP_PROB=0.3,
        TRAINING_SHUFFLE_SEED=0x0acf4,
        TRAINING_LEARNING_RATE=1e-4
    )
]

model_configs = [
    ModelConfig(
        HYPERP_TAGTOK_MAX_TOKEN_CLAMP=128,
        HYPERP_TAGTOK_VOCAB_SIZE=5000,
        HYPERP_TAGTOK_MIN_FREQ=3,
        HYPERP_TRANSFORMER_D_MODEL=128,
        HYPERP_TRANSFORMER_N_HEADS=8,
        HYPERP_TRANSFORMER_N_LAYERS=2,
        HYPERP_OUTPUT_EMB=128,
    ),
]

setups = [(m, t) for t in training_configs for m in model_configs]
save_every_epoch = 3
for m, t in setups:
    train(save_every_epoch, m, t)
