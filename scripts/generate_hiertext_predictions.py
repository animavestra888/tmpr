from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from polygon_qwen import (
    HierTextParagraphClusteringDataset,
    Qwen3VLPolygonModel,
    evaluate_pointer_outputs,
)
from polygon_qwen.device import SUPPORTED_DEVICE_NAMES, resolve_device

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on transformers version
    Qwen3VLForConditionalGeneration = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HierText paragraph pointer predictions.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--checkpoint-dir", type=Path, default=None)
    parser.add_argument("--jsonl-path", type=Path, default=Path("data/hiertext/jsonl/validation.jsonl"))
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--polygon-mode", choices=["text", "embedding"], default="text")
    parser.add_argument("--polygon-encoder", choices=["auto", "mlp", "transformer"], default="auto")
    parser.add_argument("--poly-token", type=str, default="<poly>")
    parser.add_argument("--polygon-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=1024)
    parser.add_argument("--transformer-max-positions", type=int, default=2048)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--coord-precision", type=int, default=4)
    parser.add_argument("--bbox-scale", type=int, default=1000)
    parser.add_argument("--max-pixels", type=int, default=50176)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--device", choices=SUPPORTED_DEVICE_NAMES, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def configure_processor(processor: Any, *, max_pixels: int | None) -> None:
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and max_pixels is not None:
        if hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = max_pixels
        if hasattr(image_processor, "size") and isinstance(image_processor.size, dict):
            image_processor.size["max_pixels"] = max_pixels


def load_qwen_model(model_dir: Path, *, dtype: torch.dtype) -> torch.nn.Module:
    model_cls: Any = Qwen3VLForConditionalGeneration or AutoModelForImageTextToText
    return model_cls.from_pretrained(str(model_dir), torch_dtype=dtype)


def maybe_load_lora(model: torch.nn.Module, checkpoint_dir: Path | None) -> torch.nn.Module:
    if checkpoint_dir is None:
        return model
    lora_dir = checkpoint_dir / "lora"
    if not lora_dir.exists():
        lora_dir = checkpoint_dir
    if not (lora_dir / "adapter_config.json").exists():
        return model

    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(lora_dir))


def maybe_load_polygon_adapter(model: Qwen3VLPolygonModel, checkpoint_dir: Path | None) -> None:
    if checkpoint_dir is None:
        return
    adapter_path = checkpoint_dir / "polygon_adapter.pt"
    if not adapter_path.exists():
        return
    adapter = torch.load(adapter_path, map_location="cpu")
    model.polygon_encoder.load_state_dict(adapter["polygon_encoder"])


def load_polygon_adapter_payload(checkpoint_dir: Path | None) -> dict[str, Any] | None:
    if checkpoint_dir is None:
        return None
    adapter_path = checkpoint_dir / "polygon_adapter.pt"
    if not adapter_path.exists():
        return None
    return torch.load(adapter_path, map_location="cpu")


def resolve_polygon_encoder_args(
    args: argparse.Namespace,
    adapter_payload: dict[str, Any] | None,
) -> dict[str, Any]:
    adapter_config = adapter_payload.get("polygon_encoder_config", {}) if adapter_payload else {}
    encoder_type = args.polygon_encoder
    if encoder_type == "auto":
        encoder_type = adapter_payload.get("polygon_encoder_type", "mlp") if adapter_payload else "mlp"

    return {
        "polygon_encoder_type": encoder_type,
        "dropout": adapter_config.get("dropout", args.polygon_dropout),
        "transformer_d_model": adapter_config.get("d_model", args.transformer_d_model),
        "transformer_layers": adapter_config.get("num_layers", args.transformer_layers),
        "transformer_heads": adapter_config.get("num_heads", args.transformer_heads),
        "transformer_ffn_dim": adapter_config.get("dim_feedforward", args.transformer_ffn_dim),
        "transformer_max_positions": adapter_config.get(
            "max_position_embeddings",
            args.transformer_max_positions,
        ),
    }


def build_prompt(processor: Any, prompt: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for key, value in batch.items():
        moved[key] = value.to(device) if torch.is_tensor(value) else value
    return moved


def normalize_multimodal_keys(batch: dict[str, Any]) -> dict[str, Any]:
    token_type_ids = batch.pop("token_type_ids", None)
    if token_type_ids is not None and "mm_token_type_ids" not in batch:
        batch["mm_token_type_ids"] = token_type_ids
    return batch


def generate_text_prediction(
    *,
    model: torch.nn.Module,
    processor: Any,
    example: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
) -> str:
    prompt = build_prompt(processor, example["prompt"])
    batch = processor(
        text=[prompt],
        images=[example["image"]],
        padding=True,
        return_tensors="pt",
    )
    batch = normalize_multimodal_keys(batch)
    batch = move_batch(batch, device)
    input_length = int(batch["input_ids"].shape[1])

    with torch.inference_mode():
        output_ids = model.generate(
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    return processor.tokenizer.decode(
        output_ids[0, input_length:],
        skip_special_tokens=True,
    ).strip()


def generate_embedding_prediction(
    *,
    model: Qwen3VLPolygonModel,
    processor: Any,
    example: dict[str, Any],
    device: torch.device,
    max_new_tokens: int,
) -> str:
    prompt = build_prompt(processor, example["prompt"])
    batch = processor(
        text=[prompt],
        images=[example["image"]],
        padding=True,
        return_tensors="pt",
    )
    batch = normalize_multimodal_keys(batch)
    batch["polygon_coords"] = torch.as_tensor(example["polygon_coords"], dtype=torch.float32).unsqueeze(0)
    batch["polygon_counts"] = torch.tensor([len(example["polygon_coords"])], dtype=torch.long)
    batch = move_batch(batch, device)

    with torch.inference_mode():
        input_ids = batch.pop("input_ids")
        polygon_coords = batch.pop("polygon_coords")
        polygon_counts = batch.pop("polygon_counts")
        inputs_embeds = model.base_model.get_input_embeddings()(input_ids)
        inputs_embeds = model._replace_polygon_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            polygon_coords=polygon_coords,
            polygon_counts=polygon_counts,
        )
        output_ids = model.base_model.generate(
            inputs_embeds=inputs_embeds,
            **batch,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    return processor.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)

    dataset = HierTextParagraphClusteringDataset(
        jsonl_path=args.jsonl_path,
        path_root=args.path_root,
        limit=args.max_samples,
        polygon_mode=args.polygon_mode,
        poly_token=args.poly_token,
        coord_precision=args.coord_precision,
        bbox_scale=args.bbox_scale,
    )

    if args.polygon_mode == "embedding":
        adapter_payload = load_polygon_adapter_payload(args.checkpoint_dir)
        encoder_args = resolve_polygon_encoder_args(args, adapter_payload)
        model = Qwen3VLPolygonModel.from_pretrained(
            str(args.model_dir),
            poly_token=args.poly_token,
            torch_dtype=dtype,
            freeze_base_model=True,
            **encoder_args,
        )
        configure_processor(model.processor, max_pixels=args.max_pixels)
        if adapter_payload is not None:
            model.polygon_encoder.load_state_dict(adapter_payload["polygon_encoder"])
        model.base_model = maybe_load_lora(model.base_model, args.checkpoint_dir)
        model.to(device)
        model.eval()
        processor = model.processor
    else:
        processor = AutoProcessor.from_pretrained(str(args.model_dir))
        configure_processor(processor, max_pixels=args.max_pixels)
        model = load_qwen_model(args.model_dir, dtype=dtype)
        model = maybe_load_lora(model, args.checkpoint_dir)
        model.to(device)
        model.eval()

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    metric_records: list[tuple[str, str]] = []
    with args.output_jsonl.open("w", encoding="utf-8") as output_handle:
        for index in range(len(dataset)):
            example = dataset[index]
            if args.polygon_mode == "embedding":
                prediction = generate_embedding_prediction(
                    model=model,
                    processor=processor,
                    example=example,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )
            else:
                prediction = generate_text_prediction(
                    model=model,
                    processor=processor,
                    example=example,
                    device=device,
                    max_new_tokens=args.max_new_tokens,
                )

            metric_records.append((example["answer"], prediction))
            record = {
                "image_path": example["img_path"],
                "num_lines": example["num_lines"],
                "answer": prediction,
            }
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("\n" + "=" * 88)
            print(f"index: {index} | image_id: {example['image_id']} | lines: {example['num_lines']}")
            print("GOLD:")
            print(example["answer"])
            print("PREDICTION:")
            print(prediction or "<empty>")

    metrics = evaluate_pointer_outputs(metric_records)
    print("\nMETRICS")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nwrote: {args.output_jsonl}")


if __name__ == "__main__":
    main()
