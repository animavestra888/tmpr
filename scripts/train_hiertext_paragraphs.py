from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, Trainer, TrainingArguments

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from polygon_qwen.device import SUPPORTED_DEVICE_NAMES, resolve_device
from polygon_qwen import (
    HierTextParagraphClusteringDataset,
    HierTextParagraphCollator,
    Qwen3VLPolygonModel,
)

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on transformers version
    Qwen3VLForConditionalGeneration = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-VL on HierText paragraph clustering.")
    parser.add_argument("--model-dir", type=Path, default=Path("models/Qwen3-VL-2B-Instruct"))
    parser.add_argument("--train-gt-json", type=Path, default=Path("data/hiertext/repo/gt/train.jsonl.gz"))
    parser.add_argument("--train-image-dir", type=Path, default=Path("data/hiertext/train"))
    parser.add_argument("--eval-gt-json", type=Path, default=Path("data/hiertext/repo/gt/validation.jsonl.gz"))
    parser.add_argument("--eval-image-dir", type=Path, default=Path("data/hiertext/validation"))
    parser.add_argument("--train-jsonl", type=Path, default=Path("data/hiertext/jsonl/train.jsonl"))
    parser.add_argument("--eval-jsonl", type=Path, default=Path("data/hiertext/jsonl/validation.jsonl"))
    parser.add_argument("--path-root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/hiertext_paragraphs_qwen3vl"))
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=128)
    parser.add_argument("--coord-precision", type=int, default=4)
    parser.add_argument("--bbox-scale", type=int, default=1000)
    parser.add_argument("--exclude-illegible", action="store_true")
    parser.add_argument("--polygon-mode", choices=["text", "embedding"], default="text")
    parser.add_argument("--polygon-encoder", choices=["mlp", "transformer"], default="mlp")
    parser.add_argument(
        "--polygon-adapter",
        type=Path,
        default=None,
        help="Path to polygon_adapter.pt, or to a checkpoint/final directory that contains it.",
    )
    parser.add_argument(
        "--freeze-polygon-encoder",
        action="store_true",
        help="Keep the polygon encoder fixed after loading/building it. Useful for stage-2 LoRA training.",
    )
    parser.add_argument("--poly-token", type=str, default="<poly>")
    parser.add_argument("--polygon-dropout", type=float, default=0.1)
    parser.add_argument(
        "--poly-det-loss-weight",
        type=float,
        default=0.0,
        help="Auxiliary coordinate-reconstruction loss weight for <poly> token states. 0 disables it.",
    )
    parser.add_argument(
        "--poly-det-loss-type",
        choices=["l1", "l2", "smooth_l1"],
        default="l1",
        help="Loss type for the auxiliary <poly> coordinate head.",
    )
    parser.add_argument(
        "--poly-det-source",
        choices=["embedding", "hidden"],
        default="embedding",
        help="'embedding' is memory-safe; 'hidden' uses final LM hidden states and costs more memory.",
    )
    parser.add_argument("--poly-det-dropout", type=float, default=0.0)
    parser.add_argument("--transformer-d-model", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=2)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-ffn-dim", type=int, default=1024)
    parser.add_argument("--transformer-max-positions", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=262144)
    parser.add_argument("--device", choices=SUPPORTED_DEVICE_NAMES, default="auto")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--deepspeed", type=Path, default=None)
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--disable-lora", action="store_true")
    parser.add_argument("--train-base-model", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module-name suffixes for PEFT LoRA.",
    )
    return parser.parse_args()


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def load_qwen_model(model_dir: Path, *, dtype: torch.dtype) -> torch.nn.Module:
    model_cls: Any = Qwen3VLForConditionalGeneration or AutoModelForImageTextToText
    return model_cls.from_pretrained(str(model_dir), torch_dtype=dtype)


def configure_processor(processor: Any, *, max_pixels: int | None) -> None:
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token

    image_processor = getattr(processor, "image_processor", None)
    if image_processor is not None and max_pixels is not None:
        size = getattr(image_processor, "size", None)
        if size is not None and getattr(size, "longest_edge", None) is not None:
            size["longest_edge"] = max_pixels
            if getattr(size, "shortest_edge", None) is not None and size["shortest_edge"] > max_pixels:
                size["shortest_edge"] = max_pixels
        elif hasattr(image_processor, "max_pixels"):
            image_processor.max_pixels = max_pixels


def maybe_apply_lora(model: torch.nn.Module, args: argparse.Namespace) -> torch.nn.Module:
    if args.disable_lora:
        return model

    try:
        from peft import LoraConfig, TaskType, get_peft_model
    except ImportError as exc:
        raise ImportError(
            "PEFT is required for the default LoRA training path. "
            "Install it or pass --disable-lora to full fine-tune."
        ) from exc

    config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=[item.strip() for item in args.lora_target_modules.split(",") if item.strip()],
    )
    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def resolve_polygon_adapter_path(adapter_path: Path | None) -> Path | None:
    if adapter_path is None:
        return None
    if adapter_path.is_dir():
        adapter_path = adapter_path / "polygon_adapter.pt"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Polygon adapter not found: {adapter_path}")
    return adapter_path


def infer_adapter_coord_dim(payload: dict[str, Any]) -> int | None:
    config = payload.get("polygon_encoder_config", {})
    coord_dim = config.get("coord_dim")
    if coord_dim is not None:
        return int(coord_dim)

    state_dict = payload.get("polygon_encoder", {})
    for key in ("net.0.weight", "coord_projection.weight"):
        weight = state_dict.get(key)
        if torch.is_tensor(weight) and weight.ndim == 2:
            return int(weight.shape[1])
    return None


def infer_adapter_coord_format(payload: dict[str, Any]) -> str | None:
    config = payload.get("polygon_encoder_config", {})
    coord_format = config.get("coord_format")
    return str(coord_format) if coord_format is not None else None


def load_polygon_adapter(model: Qwen3VLPolygonModel, adapter_path: Path) -> dict[str, Any]:
    payload = torch.load(adapter_path, map_location="cpu")
    saved_encoder_type = payload.get("polygon_encoder_type")
    current_encoder_type = getattr(model.polygon_encoder, "encoder_type", None)
    if saved_encoder_type is not None and current_encoder_type != saved_encoder_type:
        raise ValueError(
            "Polygon adapter encoder type mismatch: "
            f"checkpoint has '{saved_encoder_type}', current model uses '{current_encoder_type}'."
        )

    saved_coord_dim = infer_adapter_coord_dim(payload)
    saved_coord_format = infer_adapter_coord_format(payload)
    saved_config = payload.get("polygon_encoder_config", {})
    current_config = (
        model.polygon_encoder.config_dict()
        if hasattr(model.polygon_encoder, "config_dict")
        else {}
    )
    current_coord_dim = current_config.get("coord_dim")
    current_coord_format = current_config.get("coord_format")
    if saved_coord_dim is not None and current_coord_dim is not None and saved_coord_dim != current_coord_dim:
        raise ValueError(
            "Polygon adapter coordinate dimension mismatch: "
            f"checkpoint has {saved_coord_dim}, current model uses {current_coord_dim}. "
            "Current bbox-corner adapters use 8 coordinates."
        )
    if saved_coord_dim == current_coord_dim and saved_coord_format != current_coord_format:
        raise ValueError(
            "Polygon adapter coordinate format mismatch: "
            f"checkpoint has {saved_coord_format!r}, current model uses {current_coord_format!r}. "
            "Retrain the polygon adapter for axis-aligned bbox corners."
        )
    ignored_config_keys = {"dropout"}
    mismatches = [
        f"{key}: checkpoint={saved_config[key]!r}, current={current_config[key]!r}"
        for key in saved_config.keys() & current_config.keys()
        if key not in ignored_config_keys and saved_config[key] != current_config[key]
    ]
    if mismatches:
        mismatch_text = "; ".join(mismatches)
        raise ValueError(f"Polygon adapter config mismatch: {mismatch_text}")

    model.polygon_encoder.load_state_dict(payload["polygon_encoder"])
    if "poly_detection_head" in payload:
        model.load_poly_detection_head(payload["poly_detection_head"])
        print("loaded polygon detection head")
    elif model.poly_detection_loss_weight > 0.0:
        print("polygon adapter has no detection head; training a new one")
    print(f"loaded polygon adapter: {adapter_path}")
    return payload


def set_module_trainable(module: torch.nn.Module, *, trainable: bool) -> None:
    for parameter in module.parameters():
        parameter.requires_grad = trainable


def print_trainable_summary(model: torch.nn.Module) -> None:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    percent = 100.0 * trainable / total if total else 0.0
    print(f"overall trainable params: {trainable:,} || all params: {total:,} || trainable%: {percent:.4f}")


def build_training_args(args: argparse.Namespace, *, dtype: torch.dtype, device: torch.device, has_eval: bool) -> TrainingArguments:
    kwargs: dict[str, Any] = {
        "output_dir": str(args.output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "num_train_epochs": args.num_train_epochs,
        "max_steps": args.max_steps,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_total_limit": args.save_total_limit,
        "eval_strategy": "steps" if has_eval else "no",
        "eval_steps": args.eval_steps if has_eval else None,
        "fp16": dtype == torch.float16,
        "bf16": dtype == torch.bfloat16,
        "gradient_checkpointing": args.gradient_checkpointing,
        "remove_unused_columns": False,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "optim": "adamw_torch",
    }
    if args.deepspeed is not None:
        kwargs["deepspeed"] = str(args.deepspeed)
    if "use_cpu" in inspect.signature(TrainingArguments.__init__).parameters:
        kwargs["use_cpu"] = device.type == "cpu"
    return TrainingArguments(**kwargs)


class PolygonEmbeddingTrainer(Trainer):
    def _save(self, output_dir: str | None = None, state_dict: dict[str, Any] | None = None) -> None:
        if not isinstance(self.model, Qwen3VLPolygonModel):
            return super()._save(output_dir=output_dir, state_dict=state_dict)

        checkpoint_dir = Path(output_dir or self.args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model.save_polygon_adapter(checkpoint_dir)
        if hasattr(self.model.base_model, "peft_config"):
            self.model.base_model.save_pretrained(checkpoint_dir / "lora")
        torch.save(self.args, checkpoint_dir / "training_args.bin")


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    print(f"device: {device}")

    use_polygon_embeddings = args.polygon_mode == "embedding"
    if args.polygon_adapter is not None and not use_polygon_embeddings:
        raise ValueError("--polygon-adapter can only be used with --polygon-mode embedding.")
    if args.freeze_polygon_encoder and not use_polygon_embeddings:
        raise ValueError("--freeze-polygon-encoder can only be used with --polygon-mode embedding.")
    if args.poly_det_loss_weight > 0.0 and not use_polygon_embeddings:
        raise ValueError("--poly-det-loss-weight can only be used with --polygon-mode embedding.")

    if use_polygon_embeddings:
        model = Qwen3VLPolygonModel.from_pretrained(
            str(args.model_dir),
            poly_token=args.poly_token,
            torch_dtype=dtype,
            freeze_base_model=not args.train_base_model,
            dropout=args.polygon_dropout,
            polygon_encoder_type=args.polygon_encoder,
            transformer_d_model=args.transformer_d_model,
            transformer_layers=args.transformer_layers,
            transformer_heads=args.transformer_heads,
            transformer_ffn_dim=args.transformer_ffn_dim,
            transformer_max_positions=args.transformer_max_positions,
            poly_detection_loss_weight=args.poly_det_loss_weight,
            poly_detection_loss_type=args.poly_det_loss_type,
            poly_detection_source=args.poly_det_source,
            poly_detection_dropout=args.poly_det_dropout,
        )
        adapter_path = resolve_polygon_adapter_path(args.polygon_adapter)
        if adapter_path is not None:
            load_polygon_adapter(model, adapter_path)
        processor = model.processor
        configure_processor(processor, max_pixels=args.max_pixels)
        model.base_model.config.use_cache = False
        if args.gradient_checkpointing:
            model.base_model.gradient_checkpointing_enable()
            if hasattr(model.base_model, "enable_input_require_grads"):
                model.base_model.enable_input_require_grads()
            if hasattr(model.base_model.config, "use_cache"):
                model.base_model.config.use_cache = False
        if not args.disable_lora:
            model.base_model = maybe_apply_lora(model.base_model, args)
        if args.freeze_polygon_encoder:
            set_module_trainable(model.polygon_encoder, trainable=False)
            print("polygon encoder frozen")
        if args.device != "auto":
            model.to(device)
    else:
        processor = AutoProcessor.from_pretrained(str(args.model_dir))
        configure_processor(processor, max_pixels=args.max_pixels)

        model = load_qwen_model(args.model_dir, dtype=dtype)
        model.config.use_cache = False
        if args.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            if hasattr(model.config, "use_cache"):
                model.config.use_cache = False
        if not args.disable_lora:
            model = maybe_apply_lora(model, args)
        if args.device != "auto":
            model.to(device)
    print_trainable_summary(model)

    train_dataset = HierTextParagraphClusteringDataset(
        gt_json=None if args.train_jsonl else args.train_gt_json,
        image_dir=None if args.train_jsonl else args.train_image_dir,
        jsonl_path=args.train_jsonl,
        path_root=args.path_root,
        limit=args.max_train_samples,
        include_illegible=not args.exclude_illegible,
        polygon_mode=args.polygon_mode,
        poly_token=args.poly_token,
        coord_precision=args.coord_precision,
        bbox_scale=args.bbox_scale,
    )
    eval_dataset = None
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        eval_dataset = HierTextParagraphClusteringDataset(
            gt_json=None if args.eval_jsonl else args.eval_gt_json,
            image_dir=None if args.eval_jsonl else args.eval_image_dir,
            jsonl_path=args.eval_jsonl,
            path_root=args.path_root,
            limit=args.max_eval_samples,
            include_illegible=not args.exclude_illegible,
            polygon_mode=args.polygon_mode,
            poly_token=args.poly_token,
            coord_precision=args.coord_precision,
            bbox_scale=args.bbox_scale,
        )

    print(f"train_examples: {len(train_dataset)}")
    if eval_dataset is not None:
        print(f"eval_examples: {len(eval_dataset)}")

    training_args = build_training_args(
        args,
        dtype=dtype,
        device=device,
        has_eval=eval_dataset is not None,
    )

    trainer_cls = PolygonEmbeddingTrainer if use_polygon_embeddings else Trainer
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=HierTextParagraphCollator(
            processor=processor,
            use_polygon_embeddings=use_polygon_embeddings,
        ),
    )
    trainer.train()
    trainer.save_model(str(args.output_dir / "final"))
    processor.save_pretrained(args.output_dir / "final")


if __name__ == "__main__":
    main()
