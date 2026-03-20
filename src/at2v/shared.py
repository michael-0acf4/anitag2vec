from at2v.anitag2vec import AniTag2Vec
from at2v.dloader import MergeSet
from at2v.tokenizer import TagBPETokenizer

TRAINING_TAKE_EXAMPLES = 25000
TRAINING_BATCH_SIZE = 256

HYPERP_TAGTOK_MAX_TOKEN_CLAMP = 64
HYPERP_TAGTOK_VOCAB_SIZE = 5000
HYPERP_TAGTOK_MIN_FREQ = 3 # occurence at which we merge

HYPERP_TRANSFORMER_D_MODEL = 64
HYPERP_TRANSFORMER_N_HEADS = 32
HYPERP_TRANSFORMER_N_LAYERS = 2
HYPERP_OUTPUT_EMB = 128
HYPERP_EPOCHS = 15

def get_setup(
    device,
    prefix_path="."
):
    data = MergeSet.from_file(f"{prefix_path}/data/output/merged_tags.json")
    train_data = data.extend_with_synthetic(perm_limit=5, sub_array_count=5)

    tagtok = TagBPETokenizer(vocab_size=HYPERP_TAGTOK_VOCAB_SIZE, min_frequency=HYPERP_TAGTOK_MIN_FREQ)
    tagtok_file = f"{prefix_path}/checkpoints/token_vocab_size_{tagtok.vocab_size}_freq_{tagtok.min_frequency}.json"
    try:
        print(f"Loading tokenizer from '{tagtok_file}'..")
        tagtok.load(tagtok_file)
    except:
        print("Training new tokenizer..")
        tagtok.train(train_data, tagtok_file)
    print("Done!")

    anitag2vec = AniTag2Vec(
        vocab_size=tagtok.vocab_size,
        d_model=HYPERP_TRANSFORMER_D_MODEL,
        n_heads=HYPERP_TRANSFORMER_N_HEADS,
        n_layers=HYPERP_TRANSFORMER_N_LAYERS,
        output_emb=HYPERP_OUTPUT_EMB,
    )
    anitag2vec.to(device)

    return data, tagtok, anitag2vec
