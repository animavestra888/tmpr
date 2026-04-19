from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polygon_qwen import (
    EMBEDDING_GEOMETRIES,
    HierTextParagraphClusteringDataset,
    Qwen3VLPolygonModel,
    coord_format_to_embedding_geometry,
    embedding_geometry_to_coord_format,
    evaluate_pointer_outputs,
    ocr_lines_to_pointer_text,
    pointers_to_clusters,
    sanitize_pointer_output,
)
from polygon_qwen.device import SUPPORTED_DEVICE_NAMES, resolve_device
from polygon_qwen.processor import configure_processor

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
    parser.add_argument(
        "--embedding-geometry",
        choices=["auto", *EMBEDDING_GEOMETRIES],
        default="auto",
        help="Geometry representation used for learned <poly> embeddings. auto uses the adapter config when available.",
    )
    parser.add_argument("--polygon-encoder", choices=["auto", "mlp", "transformer"], default="auto")
    parser.add_argument("--poly-token", type=str, default="<poly>")
    parser.add_argument("--polygon-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=1024)
    parser.add_argument("--transformer-max-positions", type=int, default=2048)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Limit the number of examples. Use 0 or a negative value to run the full split.",
    )
    parser.add_argument("--coord-precision", type=int, default=4)
    parser.add_argument("--bbox-scale", type=int, default=1000)
    parser.add_argument("--max-pixels", type=int, default=327680)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
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


def resolve_sample_limit(max_samples: int) -> int | None:
    return None if max_samples <= 0 else max_samples


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


def load_polygon_adapter_payload(checkpoint_dir: Path | None) -> dict[str, Any] | None:
    if checkpoint_dir is None:
        return None
    adapter_path = checkpoint_dir / "polygon_adapter.pt"
    if not adapter_path.exists():
        return None
    return torch.load(adapter_path, map_location="cpu")


def load_polygon_encoder_state(model: Qwen3VLPolygonModel, adapter_payload: dict[str, Any]) -> None:
    model.polygon_encoder.load_state_dict(adapter_payload["polygon_encoder"])


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


def resolve_embedding_geometry(
    args: argparse.Namespace,
    adapter_payload: dict[str, Any] | None,
) -> str:
    adapter_config = adapter_payload.get("polygon_encoder_config", {}) if adapter_payload else {}
    adapter_coord_format = adapter_config.get("coord_format")
    adapter_geometry = (
        coord_format_to_embedding_geometry(str(adapter_coord_format))
        if adapter_coord_format is not None
        else None
    )
    if args.embedding_geometry == "auto":
        return adapter_geometry or "bbox_corners"
    if adapter_geometry is not None and args.embedding_geometry != adapter_geometry:
        raise ValueError(
            "Embedding geometry mismatch: "
            f"checkpoint uses {adapter_geometry!r}, but --embedding-geometry={args.embedding_geometry!r}."
        )
    return args.embedding_geometry


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


def _line_id(line: dict[str, Any], fallback: int) -> str:
    return str(int(line.get("id", fallback)))


def prediction_to_ocr_lines(
    *,
    gt_ocr_lines: list[dict[str, Any]],
    prediction_text: str,
) -> tuple[list[dict[str, Any]], bool, dict[str, int]]:
    expected_line_ids = [_line_id(line, index) for index, line in enumerate(gt_ocr_lines)]
    sanitized = sanitize_pointer_output(
        prediction_text,
        expected_line_ids=expected_line_ids,
    )
    pred_clusters = pointers_to_clusters(sanitized.pointers, expected_line_ids=expected_line_ids)
    if pred_clusters is None:
        raise ValueError("Sanitized pointer output is still invalid.")
    id_order = {line_id: index for index, line_id in enumerate(expected_line_ids)}
    predicted_paragraphs: dict[str, int] = {}
    for paragraph_id, cluster in enumerate(pred_clusters):
        for line_id in sorted(cluster, key=lambda item: id_order[item]):
            predicted_paragraphs[line_id] = paragraph_id

    repair_counts: dict[str, int] = {}
    for reason in sanitized.repairs:
        repair_counts[reason] = repair_counts.get(reason, 0) + 1

    predicted_lines: list[dict[str, Any]] = []
    for index, line in enumerate(gt_ocr_lines):
        line_id = _line_id(line, index)
        predicted_line = dict(line)
        predicted_line["id"] = int(line_id)
        predicted_line["paragraph_id"] = int(predicted_paragraphs.get(line_id, -1))
        predicted_lines.append(predicted_line)
    return predicted_lines, sanitized.is_valid, repair_counts


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
        position_ids = model._compute_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=batch.get("attention_mask"),
            image_grid_thw=batch.get("image_grid_thw"),
            video_grid_thw=batch.get("video_grid_thw"),
            second_per_grid_ts=batch.get("second_per_grid_ts"),
            mm_token_type_ids=batch.get("mm_token_type_ids"),
        )
        if position_ids is not None:
            batch["position_ids"] = position_ids
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

    adapter_payload = load_polygon_adapter_payload(args.checkpoint_dir) if args.polygon_mode == "embedding" else None
    embedding_geometry = resolve_embedding_geometry(args, adapter_payload)
    coord_format = embedding_geometry_to_coord_format(embedding_geometry)

    dataset = HierTextParagraphClusteringDataset(
        jsonl_path=args.jsonl_path,
        path_root=args.path_root,
        limit=resolve_sample_limit(args.max_samples),
        polygon_mode=args.polygon_mode,
        embedding_geometry=embedding_geometry,
        poly_token=args.poly_token,
        coord_precision=args.coord_precision,
        bbox_scale=args.bbox_scale,
    )

    if args.polygon_mode == "embedding":
        encoder_args = resolve_polygon_encoder_args(args, adapter_payload)
        model = Qwen3VLPolygonModel.from_pretrained(
            str(args.model_dir),
            poly_token=args.poly_token,
            torch_dtype=dtype,
            freeze_base_model=True,
            coord_format=coord_format,
            **encoder_args,
        )
        configure_processor(model.processor, max_pixels=args.max_pixels)
        if adapter_payload is not None:
            load_polygon_encoder_state(model, adapter_payload)
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
    start_time = time.perf_counter()
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

            predicted_ocr_lines, prediction_valid, prediction_repair_counts = prediction_to_ocr_lines(
                gt_ocr_lines=example["ocr_lines"],
                prediction_text=prediction,
            )
            record = {
                "img_path": example["img_path"],
                "num_lines": example["num_lines"],
                "ocr_lines": predicted_ocr_lines,
                "gt_ocr_lines": example["ocr_lines"],
                "gt_answer": example["answer"],
                "model_answer": prediction,
                "prediction_valid": prediction_valid,
                "prediction_repair_counts": prediction_repair_counts,
            }
            metric_records.append(
                (
                    ocr_lines_to_pointer_text(record["gt_ocr_lines"]),
                    ocr_lines_to_pointer_text(record["ocr_lines"]),
                )
            )
            output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            print("\n" + "=" * 88)
            print(f"index: {index} | image_id: {example['image_id']} | lines: {example['num_lines']}")
            print("GT:")
            print(example["answer"])
            print("MODEL ANSWER:")
            print(prediction or "<empty>")
            print(f"prediction_valid: {prediction_valid}")
            if prediction_repair_counts:
                print(f"prediction_repair_counts: {json.dumps(prediction_repair_counts, ensure_ascii=False)}")

    metrics = evaluate_pointer_outputs(metric_records)
    inference_seconds = time.perf_counter() - start_time
    metrics["inference_time"] = inference_seconds
    metrics["samples_per_second"] = len(metric_records) / inference_seconds if inference_seconds > 0 else 0.0
    print("\nMETRICS")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print(f"\nwrote: {args.output_jsonl}")


if __name__ == "__main__":
    main()
