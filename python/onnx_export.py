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
        encode_split_token_id=cfg.HYPERP_INPUT_ALLOW_POS_ENCODING_TOKEN_ID
    )
    anitag2vec.load_state_dict(torch.load(model_path))
    anitag2vec.eval()

    print("Exporting to ONNX...")
    base, _ = os.path.splitext(model_path)
    output_file = f"{base}.onnx"
    example_input = torch.randint(0, 1000, (2, cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,), dtype=torch.int64)
    # example_input = torch.randn((1, cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,), dtype=torch.int64)
    # anitag2vec(example_input)
    onnx_program = torch.onnx.export(
        anitag2vec,
        example_input,
        output_file,
        # dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
        # opset_version=17,
        # do_constant_folding=False,
        input_names=["x"],
        output_names=["y"],
        # dynamic_shapes={
        dynamic_axes={
            # "input": { 0: torch.export.Dim("batch") }
            "x": { 0: "batch" },
            "y": { 0: "batch" }
        },
        # verbose=True
    )

    print(f"ONNX model created at {output_file}")

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
