from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor

try:
    from transformers import Qwen3VLForConditionalGeneration
except ImportError:  # pragma: no cover - depends on transformers version
    Qwen3VLForConditionalGeneration = None


def _call_with_supported_kwargs(fn: Any, **kwargs: Any) -> Any:
    signature = inspect.signature(fn)
    supported = {}
    for name, parameter in signature.parameters.items():
        if name in kwargs:
            supported[name] = kwargs[name]
        elif parameter.default is inspect._empty:
            raise TypeError(f"Missing required argument '{name}' for {fn}.")
    return fn(**supported)


def _find_nested_attr(root: Any, attr_name: str, *, max_depth: int = 4) -> Any | None:
    queue: list[tuple[Any, int]] = [(root, 0)]
    visited: set[int] = set()
    while queue:
        item, depth = queue.pop(0)
        if item is None or id(item) in visited:
            continue
        visited.add(id(item))
        value = getattr(item, attr_name, None)
        if value is not None:
            return value
        if depth >= max_depth:
            continue
        for child_name in ("model", "base_model", "module"):
            child = getattr(item, child_name, None)
            if child is not None:
                queue.append((child, depth + 1))
    return None


def _resolve_hidden_size(config: Any) -> int:
    for attr in ("hidden_size",):
        value = getattr(config, attr, None)
        if value is not None:
            return int(value)

    text_config = getattr(config, "text_config", None)
    if text_config is not None:
        value = getattr(text_config, "hidden_size", None)
        if value is not None:
            return int(value)

    raise AttributeError("Could not resolve hidden_size from the loaded model config.")


def _load_base_model(
    model_name_or_path: str,
    *,
    torch_dtype: torch.dtype | str | None = None,
    device_map: str | dict[str, Any] | None = None,
    trust_remote_code: bool = False,
) -> nn.Module:
    common_kwargs = {
        "torch_dtype": torch_dtype,
        "device_map": device_map,
        "trust_remote_code": trust_remote_code,
    }
    if Qwen3VLForConditionalGeneration is not None:
        return Qwen3VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            **common_kwargs,
        )
    return AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        **common_kwargs,
    )


def _ensure_poly_token(processor: Any, model: nn.Module, poly_token: str) -> int:
    tokenizer = processor.tokenizer
    poly_token_id = tokenizer.convert_tokens_to_ids(poly_token)
    if poly_token_id != tokenizer.unk_token_id:
        return int(poly_token_id)

    tokenizer.add_special_tokens({"additional_special_tokens": [poly_token]})
    model.resize_token_embeddings(len(tokenizer))
    return int(tokenizer.convert_tokens_to_ids(poly_token))


class PolygonMLPEncoder(nn.Module):
    encoder_type = "mlp"

    def __init__(self, out_dim: int = 2048, dropout: float = 0.1) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.dropout = dropout
        self.net = nn.Sequential(
            nn.Linear(8, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
            nn.Dropout(dropout),
            nn.Linear(1024, out_dim),
        )

    def forward(
        self,
        polygon_coords: torch.Tensor,
        polygon_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if polygon_coords.ndim < 2 or polygon_coords.shape[-1] != 8:
            raise ValueError(
                f"Expected polygon_coords with shape (..., 8), received {tuple(polygon_coords.shape)}."
            )
        leading_shape = polygon_coords.shape[:-1]
        flat_coords = polygon_coords.reshape(-1, 8)
        flat_embeds = self.net(flat_coords)
        return flat_embeds.reshape(*leading_shape, flat_embeds.shape[-1])

    def config_dict(self) -> dict[str, Any]:
        return {
            "out_dim": self.out_dim,
            "dropout": self.dropout,
        }


class PolygonLayoutTransformerEncoder(nn.Module):
    encoder_type = "transformer"

    def __init__(
        self,
        out_dim: int = 2048,
        *,
        d_model: int = 256,
        num_layers: int = 2,
        num_heads: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_position_embeddings: int = 2048,
    ) -> None:
        super().__init__()
        self.out_dim = out_dim
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout
        self.max_position_embeddings = max_position_embeddings

        self.coord_projection = nn.Linear(8, d_model)
        self.coord_norm = nn.LayerNorm(d_model)
        self.position_embedding = nn.Embedding(max_position_embeddings, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, out_dim)

    def _padding_mask(
        self,
        *,
        batch_size: int,
        num_polygons: int,
        polygon_counts: torch.Tensor | None,
        device: torch.device,
    ) -> torch.Tensor | None:
        if polygon_counts is None:
            return None
        polygon_counts = polygon_counts.to(device=device)
        return torch.arange(num_polygons, device=device).unsqueeze(0).expand(batch_size, -1) >= polygon_counts.unsqueeze(1)

    def forward(
        self,
        polygon_coords: torch.Tensor,
        polygon_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        squeeze_polygon_dim = False
        if polygon_coords.ndim == 2 and polygon_coords.shape[-1] == 8:
            polygon_coords = polygon_coords.unsqueeze(1)
            squeeze_polygon_dim = True
            polygon_counts = torch.ones(
                polygon_coords.shape[0],
                dtype=torch.long,
                device=polygon_coords.device,
            )
        elif polygon_coords.ndim != 3 or polygon_coords.shape[-1] != 8:
            raise ValueError(
                f"Expected polygon_coords with shape (batch, 8) or (batch, num_polygons, 8), "
                f"received {tuple(polygon_coords.shape)}."
            )

        batch_size, num_polygons, _ = polygon_coords.shape
        if num_polygons > self.max_position_embeddings:
            raise ValueError(
                f"Received {num_polygons} polygons, but max_position_embeddings="
                f"{self.max_position_embeddings}."
            )

        position_ids = torch.arange(num_polygons, device=polygon_coords.device)
        hidden_states = self.coord_norm(self.coord_projection(polygon_coords))
        hidden_states = hidden_states + self.position_embedding(position_ids).unsqueeze(0)
        padding_mask = self._padding_mask(
            batch_size=batch_size,
            num_polygons=num_polygons,
            polygon_counts=polygon_counts,
            device=polygon_coords.device,
        )
        hidden_states = self.encoder(hidden_states, src_key_padding_mask=padding_mask)
        hidden_states = self.output_norm(hidden_states)
        output = self.output_projection(hidden_states)
        if squeeze_polygon_dim:
            return output.squeeze(1)
        return output

    def config_dict(self) -> dict[str, Any]:
        return {
            "out_dim": self.out_dim,
            "d_model": self.d_model,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "dim_feedforward": self.dim_feedforward,
            "dropout": self.dropout,
            "max_position_embeddings": self.max_position_embeddings,
        }


def build_polygon_encoder(
    *,
    encoder_type: str,
    hidden_size: int,
    dropout: float,
    transformer_d_model: int = 256,
    transformer_layers: int = 2,
    transformer_heads: int = 4,
    transformer_ffn_dim: int = 1024,
    transformer_max_positions: int = 2048,
) -> nn.Module:
    if encoder_type == "mlp":
        return PolygonMLPEncoder(out_dim=hidden_size, dropout=dropout)
    if encoder_type == "transformer":
        return PolygonLayoutTransformerEncoder(
            out_dim=hidden_size,
            d_model=transformer_d_model,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            max_position_embeddings=transformer_max_positions,
        )
    raise ValueError("polygon_encoder_type must be either 'mlp' or 'transformer'.")


class Qwen3VLPolygonModel(nn.Module):
    """Qwen3-VL wrapper that replaces the `<poly>` token embedding per sample."""
    supports_gradient_checkpointing = True

    def __init__(
        self,
        *,
        base_model: nn.Module,
        processor: Any,
        polygon_encoder: nn.Module,
        poly_token_id: int,
        poly_token: str = "<poly>",
        freeze_base_model: bool = True,
    ) -> None:
        super().__init__()
        self.base_model = base_model
        self.processor = processor
        self.polygon_encoder = polygon_encoder
        self.poly_token_id = int(poly_token_id)
        self.poly_token = poly_token
        self._config = base_model.config
        self.generation_config = getattr(base_model, "generation_config", None)
        self.hidden_size = _resolve_hidden_size(base_model.config)

        if freeze_base_model:
            for parameter in self.base_model.parameters():
                parameter.requires_grad = False

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        *,
        poly_token: str = "<poly>",
        torch_dtype: torch.dtype | str | None = None,
        device_map: str | dict[str, Any] | None = None,
        freeze_base_model: bool = True,
        dropout: float = 0.1,
        polygon_encoder_type: str = "mlp",
        transformer_d_model: int = 256,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_ffn_dim: int = 1024,
        transformer_max_positions: int = 2048,
        trust_remote_code: bool = False,
    ) -> "Qwen3VLPolygonModel":
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        base_model = _load_base_model(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        poly_token_id = _ensure_poly_token(processor, base_model, poly_token)
        hidden_size = _resolve_hidden_size(base_model.config)
        polygon_encoder = build_polygon_encoder(
            encoder_type=polygon_encoder_type,
            hidden_size=hidden_size,
            dropout=dropout,
            transformer_d_model=transformer_d_model,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_ffn_dim=transformer_ffn_dim,
            transformer_max_positions=transformer_max_positions,
        )
        return cls(
            base_model=base_model,
            processor=processor,
            polygon_encoder=polygon_encoder,
            poly_token_id=poly_token_id,
            poly_token=poly_token,
            freeze_base_model=freeze_base_model,
        )

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [parameter for parameter in self.parameters() if parameter.requires_grad]

    @property
    def config(self) -> Any:
        return self.base_model.config

    @config.setter
    def config(self, value: Any) -> None:
        self._config = value

    @property
    def is_gradient_checkpointing(self) -> bool:
        return bool(getattr(self.base_model, "is_gradient_checkpointing", False))

    def gradient_checkpointing_enable(
        self,
        gradient_checkpointing_kwargs: dict[str, Any] | None = None,
    ) -> None:
        if not hasattr(self.base_model, "gradient_checkpointing_enable"):
            return
        if gradient_checkpointing_kwargs is None:
            self.base_model.gradient_checkpointing_enable()
            return
        try:
            self.base_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs
            )
        except TypeError:
            self.base_model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.base_model, "gradient_checkpointing_disable"):
            self.base_model.gradient_checkpointing_disable()

    def enable_input_require_grads(self) -> None:
        if hasattr(self.base_model, "enable_input_require_grads"):
            self.base_model.enable_input_require_grads()

    def disable_input_require_grads(self) -> None:
        if hasattr(self.base_model, "disable_input_require_grads"):
            self.base_model.disable_input_require_grads()

    def save_polygon_adapter(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.processor.save_pretrained(output_dir / "processor")
        adapter_path = output_dir / "polygon_adapter.pt"
        torch.save(
            {
                "polygon_encoder": self.polygon_encoder.state_dict(),
                "polygon_encoder_type": getattr(self.polygon_encoder, "encoder_type", "unknown"),
                "polygon_encoder_config": (
                    self.polygon_encoder.config_dict()
                    if hasattr(self.polygon_encoder, "config_dict")
                    else {}
                ),
                "poly_token": self.poly_token,
                "poly_token_id": self.poly_token_id,
                "hidden_size": self.hidden_size,
                "base_model_name_or_path": getattr(self.base_model.config, "_name_or_path", None),
            },
            adapter_path,
        )
        return adapter_path

    def _replace_polygon_embeddings(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        polygon_coords: torch.Tensor,
        polygon_counts: torch.Tensor | None = None,
    ) -> torch.Tensor:
        poly_mask = input_ids.eq(self.poly_token_id)
        occurrences = poly_mask.sum(dim=1)

        if polygon_coords.ndim == 2:
            expected_counts = torch.ones_like(occurrences)
        elif polygon_coords.ndim == 3:
            if polygon_counts is None:
                expected_counts = torch.full_like(occurrences, polygon_coords.shape[1])
            else:
                expected_counts = polygon_counts.to(device=occurrences.device, dtype=occurrences.dtype)
        else:
            raise ValueError(
                f"Expected polygon_coords with shape (batch, 8) or (batch, num_polygons, 8), "
                f"received {tuple(polygon_coords.shape)}."
            )

        if not torch.equal(occurrences, expected_counts):
            raise ValueError(
                f"Each sample must contain one {self.poly_token} token per polygon. "
                f"Observed token counts: {occurrences.tolist()}; "
                f"expected polygon counts: {expected_counts.tolist()}."
            )

        encoder_parameter = next(self.polygon_encoder.parameters())
        polygon_coords = polygon_coords.to(
            device=encoder_parameter.device,
            dtype=encoder_parameter.dtype,
        )

        encoder_is_frozen = not any(parameter.requires_grad for parameter in self.polygon_encoder.parameters())
        if encoder_is_frozen:
            was_training = self.polygon_encoder.training
            self.polygon_encoder.eval()
            with torch.no_grad():
                polygon_embeds = self.polygon_encoder(polygon_coords, polygon_counts=expected_counts)
            if was_training:
                self.polygon_encoder.train()
        else:
            polygon_embeds = self.polygon_encoder(polygon_coords, polygon_counts=expected_counts)

        updated = inputs_embeds.clone()
        if polygon_embeds.ndim == 2:
            updated[poly_mask] = polygon_embeds.to(
                device=updated.device,
                dtype=updated.dtype,
            )
            return updated

        polygon_embeds = polygon_embeds.to(device=updated.device, dtype=updated.dtype)
        for batch_index in range(input_ids.shape[0]):
            positions = poly_mask[batch_index].nonzero(as_tuple=False).squeeze(-1)
            count = int(expected_counts[batch_index].item())
            if count == 0:
                continue
            updated[batch_index, positions[:count]] = polygon_embeds[batch_index, :count]
        return updated

    def _compute_position_ids(
        self,
        *,
        input_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        video_grid_thw: torch.Tensor | None,
        second_per_grid_ts: torch.Tensor | None,
        mm_token_type_ids: torch.Tensor | None,
    ) -> torch.Tensor | None:
        compute_fn = _find_nested_attr(self.base_model, "compute_3d_position_ids")
        if compute_fn is None:
            return None

        return _call_with_supported_kwargs(
            compute_fn,
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            mm_token_type_ids=mm_token_type_ids,
            past_key_values=None,
        )

    def forward(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        polygon_coords: torch.Tensor,
        polygon_counts: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        mm_token_type_ids: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        second_per_grid_ts: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> Any:
        if token_type_ids is not None and mm_token_type_ids is None:
            mm_token_type_ids = token_type_ids

        target_device = next(self.base_model.parameters()).device
        input_ids = input_ids.to(target_device)
        attention_mask = attention_mask.to(target_device) if attention_mask is not None else None
        pixel_values = pixel_values.to(target_device) if pixel_values is not None else None
        image_grid_thw = image_grid_thw.to(target_device) if image_grid_thw is not None else None
        polygon_coords = polygon_coords.to(target_device)
        polygon_counts = polygon_counts.to(target_device) if polygon_counts is not None else None
        labels = labels.to(target_device) if labels is not None else None
        mm_token_type_ids = mm_token_type_ids.to(target_device) if mm_token_type_ids is not None else None
        video_grid_thw = video_grid_thw.to(target_device) if video_grid_thw is not None else None
        second_per_grid_ts = (
            second_per_grid_ts.to(target_device) if second_per_grid_ts is not None else None
        )

        inputs_embeds = self.base_model.get_input_embeddings()(input_ids)
        inputs_embeds = self._replace_polygon_embeddings(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            polygon_coords=polygon_coords,
            polygon_counts=polygon_counts,
        )

        position_ids = self._compute_position_ids(
            input_ids=input_ids,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            mm_token_type_ids=mm_token_type_ids,
        )

        return self.base_model(
            input_ids=None,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            labels=labels,
            position_ids=position_ids,
            mm_token_type_ids=mm_token_type_ids,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            **kwargs,
        )
