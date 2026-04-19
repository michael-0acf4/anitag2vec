import argparse
import os


def onnx_export(model_path: str, tokenizer_path: str, config_path: str):
    import torch
    from at2v.anitag2vec import AniTag2Vec, ModelConfig, TagBPETokenizer
    print("Loading PyTorch model..")
    cfg = ModelConfig.load_from_file(config_path)
    tagtok = TagBPETokenizer.load_from_file(tokenizer_path)

    anitag2vec = AniTag2Vec(
        vocab_size=cfg.HYPERP_TAGTOK_VOCAB_SIZE,
        max_len_cut=cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,
        d_model=cfg.HYPERP_TRANSFORMER_D_MODEL,
        n_heads=cfg.HYPERP_TRANSFORMER_N_HEADS,
        n_layers=cfg.HYPERP_TRANSFORMER_N_LAYERS,
        output_emb=cfg.HYPERP_OUTPUT_EMB,
        segmented_rope=cfg.HYPERP_ENABLE_SEGMENTED_ROPE
    )
    anitag2vec.load_state_dict(torch.load(model_path))
    anitag2vec.eval()

    print("Exporting to ONNX...")
    base, _ = os.path.splitext(model_path)
    output_file = f"{base}.onnx"

    input_names = ["x"]
    output_names = ["y"]
    dynamic_axes = {
        # "input": { 0: torch.export.Dim("batch") }
        "x": { 0: "batch" },
        "y": { 0: "batch" }
    }
    x = torch.randint(0, 1000, (2, cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,), dtype=torch.int64)
    if anitag2vec.segmented_rope:
        input_names.append("x_chunked_pos")
        dynamic_axes["x_chunked_pos"] = { 0: "batch" }
        example_input = (x, tagtok.get_chunked_positions_torch(x))
    else:
        example_input = (x,)
    # example_input = torch.randn((1, cfg.HYPERP_TAGTOK_MAX_TOKEN_CLAMP,), dtype=torch.int64)
    # anitag2vec(example_input)
    torch.onnx.export(
        anitag2vec,
        example_input,
        output_file,
        # dynamo=False,
        training=torch.onnx.TrainingMode.EVAL,
        opset_version=14,
        # do_constant_folding=False,
        input_names=input_names,
        output_names=output_names,
        # dynamic_shapes={
        dynamic_axes=dynamic_axes,
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
        "--tokenizer",
        type=str,
        required=True,
        help="Tokenizer model json config path"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Model json config path"
    )
    args = parser.parse_args()
    onnx_export(args.model, args.tokenizer, args.config)

if __name__ == "__main__":
    main()
