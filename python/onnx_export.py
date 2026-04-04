import argparse
import os

def onnx_export(model_path: str, config_path: str):
    import torch
    from at2v.anitag2vec import AniTag2Vec, ModelConfig
    print("Loading PyTorch model..")
    cfg = ModelConfig.load_from_file(config_path)    
    anitag2vec = AniTag2Vec(
        vocab_size=cfg.HYPERP_TAGTOK_VOCAB_SIZE,
        max_len_cut=cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,
        d_model=cfg.HYPERP_TRANSFORMER_D_MODEL,
        n_heads=cfg.HYPERP_TRANSFORMER_N_HEADS,
        n_layers=cfg.HYPERP_TRANSFORMER_N_LAYERS,
        output_emb=cfg.HYPERP_OUTPUT_EMB,
    )
    anitag2vec.load_state_dict(torch.load(model_path))
    anitag2vec.eval()

    print("Exporting to ONNX...")
    example_input = torch.randint(0, 1, (1, cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,))
    # example_input = torch.randn((1, cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,), dtype=torch.int64)
    anitag2vec(example_input)
    onnx_program = torch.onnx.export(anitag2vec, example_input, dynamo=True)
    base, _ = os.path.splitext(model_path)
    output = f"{base}.onnx"
    onnx_program.save(f"{base}.onnx")
    print(f"ONNX model created at {output}")

def main():
    parser = argparse.ArgumentParser(description="Anitag2Vec ONNX exporter")
    print("Note: Please install onnx first if not done yet")
    print(" $ pip install --upgrade onnx onnxscript")
    print("")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model file .pt, .pth path"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model json config path"
    )
    args = parser.parse_args()
    onnx_export(args.model, args.config)

if __name__ == "__main__":
    main()
