#!/usr/bin/env python3
"""Generate real-checkpoint SigLIP2 numerical reference artifacts.

The generated JSON manifest and safetensors payload are consumed by
``tests/real_numerical_parity.rs``.  Model loading is deliberately local-only:
the caller must select an explicit Hugging Face checkpoint directory, so CI
never downloads multi-gigabyte weights by accident.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
DEFAULT_TEXTS = (
    "this is a photo of a cat.",
    "this is a photo of a dog.",
    "this is a photo of an airplane.",
)

# These are the three smallest architecture scales from the official SigLIP2
# release, each represented by its lowest-resolution fixed-resolution model.
VARIANTS: dict[str, dict[str, int]] = {
    "base-patch16-224": {
        "image_size": 224,
        "patch_size": 16,
        "hidden_size": 768,
        "intermediate_size": 3072,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "projection_size": 768,
    },
    "large-patch16-256": {
        "image_size": 256,
        "patch_size": 16,
        "hidden_size": 1024,
        "intermediate_size": 4096,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "projection_size": 1024,
    },
    "so400m-patch14-224": {
        "image_size": 224,
        "patch_size": 14,
        "hidden_size": 1152,
        "intermediate_size": 4304,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "projection_size": 1152,
    },
}

TOLERANCES: dict[str, dict[str, float]] = {
    # Same-storage-dtype parity against the Burn NdArray backend.  These limits
    # allow normal reduction-order differences without hiding broken masking,
    # pooling, normalization, or weight mapping.
    "f32": {
        "embedding_max_abs": 2.0e-4,
        "embedding_rmse": 5.0e-5,
        "embedding_min_cosine": 0.999995,
        "logit_max_abs": 2.0e-2,
        "logit_rmse": 1.0e-2,
        "probability_max_abs": 2.0e-3,
        "probability_rmse": 1.0e-3,
    },
    "f16": {
        "embedding_max_abs": 5.0e-4,
        "embedding_rmse": 1.0e-4,
        "embedding_min_cosine": 0.99999,
        "logit_max_abs": 5.0e-2,
        "logit_rmse": 3.0e-2,
        "probability_max_abs": 5.0e-3,
        "probability_rmse": 3.0e-3,
    },
}

# Acceptance for the intended CDN F16 artifact relative to upstream F32.  A
# base-patch16-224 measurement is documented beside the harness README.
F16_FIDELITY_LIMITS = {
    "embedding_max_abs": 5.0e-4,
    "embedding_min_cosine": 0.9999,
    "logit_max_abs": 5.0e-2,
    "probability_max_abs": 5.0e-3,
}


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Emit Hugging Face SigLIP2 image/text/logit reference tensors"
    )
    parser.add_argument(
        "--hf-dir",
        type=Path,
        help="local Hugging Face checkpoint directory (never downloaded)",
    )
    parser.add_argument("--variant", choices=tuple(VARIANTS))
    parser.add_argument(
        "--output-base",
        type=Path,
        help="output stem; writes <stem>.json and <stem>.safetensors",
    )
    parser.add_argument(
        "--image",
        type=Path,
        help="optional image; otherwise uses the deterministic 83x61 RGB pattern",
    )
    parser.add_argument(
        "--text",
        action="append",
        dest="texts",
        help="complete text prompt; repeat for a batch (defaults to three prompts)",
    )
    parser.add_argument(
        "--weight-precision",
        choices=("f32", "f16"),
        default="f16",
        help="quantize weights through F16 before F32 compute to mirror CDN storage",
    )
    parser.add_argument(
        "--upstream-revision",
        help="optional immutable Hugging Face git revision recorded in metadata",
    )
    parser.add_argument(
        "--hash-model",
        action="store_true",
        help="SHA-256 the multi-gigabyte model.safetensors file",
    )
    parser.add_argument("--threads", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run a dependency-free deterministic smoke test and exit",
    )
    args = parser.parse_args(argv)
    if args.self_test:
        return args
    missing = [
        flag
        for flag, value in (
            ("--hf-dir", args.hf_dir),
            ("--variant", args.variant),
            ("--output-base", args.output_base),
        )
        if value is None
    ]
    if missing:
        parser.error(f"required unless --self-test: {', '.join(missing)}")
    if args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_output_base(path: Path) -> Path:
    value = str(path)
    for suffix in (".safetensors", ".json"):
        if value.endswith(suffix):
            value = value[: -len(suffix)]
            break
    if not value:
        raise ValueError("output base cannot be empty")
    return Path(value)


def deterministic_rgb() -> tuple[bytes, dict[str, Any]]:
    width, height = 83, 61
    rgb = bytearray(width * height * 3)
    cursor = 0
    for y in range(height):
        for x in range(width):
            rgb[cursor] = (x * 3 + y * 5 + 17) % 256
            rgb[cursor + 1] = (x * 7 + y * 11 + 29) % 256
            rgb[cursor + 2] = (x * 13 + y * 17 + 43) % 256
            cursor += 3
    data = bytes(rgb)
    return data, {
        "kind": "deterministic_rgb_v1",
        "width": width,
        "height": height,
        "rgb_sha256": sha256_bytes(data),
    }


def deterministic_image() -> tuple[Any, dict[str, Any]]:
    from PIL import Image

    data, metadata = deterministic_rgb()
    return Image.frombytes(
        "RGB", (metadata["width"], metadata["height"]), data
    ), metadata


def load_image(path: Path | None) -> tuple[Any, dict[str, Any]]:
    if path is None:
        return deterministic_image()

    from PIL import Image, ImageOps

    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"image does not exist: {path}")
    image = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    return image, {
        "kind": "file",
        "path": str(path),
        "file_sha256": sha256_file(path),
        "width": image.width,
        "height": image.height,
    }


def encode_reference_png(image: Any) -> bytes:
    """Encode the exact RGB input used by HF for Rust decoder parity."""
    output = io.BytesIO()
    image.save(output, format="PNG", optimize=False, compress_level=9)
    return output.getvalue()


def require_checkpoint_files(hf_dir: Path) -> dict[str, Path]:
    hf_dir = hf_dir.expanduser().resolve()
    if not hf_dir.is_dir():
        raise FileNotFoundError(f"checkpoint directory does not exist: {hf_dir}")
    files = {
        name: hf_dir / name
        for name in (
            "config.json",
            "model.safetensors",
            "preprocessor_config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        )
    }
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("checkpoint is incomplete: " + ", ".join(missing))
    return files


def config_value(config: Any, name: str) -> int:
    value = getattr(config, name)
    if isinstance(value, (tuple, list)):
        if len(value) != 2 or value[0] != value[1]:
            raise ValueError(f"expected square {name}, got {value!r}")
        value = value[0]
    return int(value)


def validate_model_config(model: Any, variant: str) -> dict[str, int]:
    expected = VARIANTS[variant]
    text = model.config.text_config
    vision = model.config.vision_config
    actual = {
        "image_size": config_value(vision, "image_size"),
        "patch_size": config_value(vision, "patch_size"),
        "hidden_size": int(vision.hidden_size),
        "intermediate_size": int(vision.intermediate_size),
        "num_hidden_layers": int(vision.num_hidden_layers),
        "num_attention_heads": int(vision.num_attention_heads),
        "projection_size": int(text.projection_size),
        "text_hidden_size": int(text.hidden_size),
        "text_intermediate_size": int(text.intermediate_size),
        "text_num_hidden_layers": int(text.num_hidden_layers),
        "text_num_attention_heads": int(text.num_attention_heads),
        "text_max_positions": int(text.max_position_embeddings),
        "text_vocab_size": int(text.vocab_size),
    }
    for key, value in expected.items():
        if actual[key] != value:
            raise ValueError(
                f"checkpoint does not match {variant}: {key}={actual[key]}, expected {value}"
            )
    paired = {
        "text_hidden_size": "hidden_size",
        "text_intermediate_size": "intermediate_size",
        "text_num_hidden_layers": "num_hidden_layers",
        "text_num_attention_heads": "num_attention_heads",
    }
    for text_key, vision_key in paired.items():
        if actual[text_key] != actual[vision_key]:
            raise ValueError(
                f"unsupported asymmetric towers: {text_key}={actual[text_key]} "
                f"but {vision_key}={actual[vision_key]}"
            )
    if actual["text_max_positions"] != 64 or actual["text_vocab_size"] != 256_000:
        raise ValueError(
            "SigLIP2 reference requires text length 64 and Gemma vocabulary 256000"
        )
    return actual


def quantize_weights_for_storage(model: Any, torch: Any, precision: str) -> None:
    if precision == "f32":
        return
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.is_floating_point():
                parameter.copy_(parameter.to(torch.float16).to(parameter.dtype))


def tensor_bytes(tensor: Any) -> bytes:
    return tensor.detach().cpu().contiguous().numpy().tobytes(order="C")


def tensor_entry(tensor: Any) -> dict[str, Any]:
    data = tensor_bytes(tensor)
    return {
        "dtype": str(tensor.dtype).removeprefix("torch."),
        "shape": list(tensor.shape),
        "sha256": sha256_bytes(data),
    }


def owned_cpu_tensor(tensor: Any) -> Any:
    """Return an independent contiguous allocation accepted by safetensors."""
    return tensor.detach().cpu().contiguous().clone()


def row_norms(tensor: Any, torch: Any) -> list[float]:
    if tensor.ndim < 2:
        return []
    return [float(value) for value in torch.linalg.vector_norm(tensor.float(), dim=-1).cpu()]


def assert_close(name: str, actual: Any, expected: Any, torch: Any) -> None:
    if actual.shape != expected.shape:
        raise AssertionError(f"{name} shape mismatch: {actual.shape} != {expected.shape}")
    if not torch.allclose(actual.float(), expected.float(), atol=2.0e-5, rtol=2.0e-5):
        delta = (actual.float() - expected.float()).abs().max().item()
        raise AssertionError(f"{name} disagrees with full model forward: max_abs={delta:.6e}")


def generate(args: argparse.Namespace) -> tuple[Path, Path]:
    try:
        import torch
        import transformers
        from safetensors.torch import save_file
        from transformers import AutoModel, AutoProcessor
        from transformers.utils import logging as transformers_logging
    except ImportError as error:
        raise RuntimeError(
            "reference generation requires torch, transformers, Pillow, and safetensors"
        ) from error

    transformers_logging.disable_progress_bar()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)

    checkpoint_files = require_checkpoint_files(args.hf_dir)
    hf_dir = checkpoint_files["config.json"].parent
    device = torch.device("cpu")
    model = AutoModel.from_pretrained(hf_dir, local_files_only=True).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        hf_dir, local_files_only=True, use_fast=False
    )
    resolved_config = validate_model_config(model, args.variant)

    tokenizer = processor.tokenizer
    tokenizer_facts = {
        "class": type(tokenizer).__name__,
        "pad_token": str(tokenizer.pad_token),
        "pad_token_id": int(tokenizer.pad_token_id),
        "eos_token": str(tokenizer.eos_token),
        "eos_token_id": int(tokenizer.eos_token_id),
        "padding_side": str(tokenizer.padding_side),
    }
    if tokenizer_facts["pad_token_id"] != 0:
        raise ValueError(f"expected SigLIP2 pad id 0, got {tokenizer_facts['pad_token_id']}")
    if tokenizer_facts["eos_token_id"] != 1:
        raise ValueError(f"expected SigLIP2 EOS id 1, got {tokenizer_facts['eos_token_id']}")
    if tokenizer_facts["padding_side"] != "right":
        raise ValueError("fixed-resolution SigLIP2 requires right padding")

    image, image_metadata = load_image(args.image)
    encoded_image = encode_reference_png(image)
    original_texts = tuple(args.texts or DEFAULT_TEXTS)
    if not original_texts or any(not text.strip() for text in original_texts):
        raise ValueError("at least one non-empty --text is required")
    # The fixed-resolution repositories still advertise GemmaTokenizer, which
    # does not lowercase by itself.  SigLIP2 training and the newer explicit
    # Siglip2Tokenizer do lowercase, so make the normalization unambiguous.
    normalized_texts = tuple(text.lower() for text in original_texts)
    processed = processor(
        text=list(normalized_texts),
        images=[image],
        padding="max_length",
        truncation=True,
        max_length=64,
        return_tensors="pt",
    )
    pixel_values = processed["pixel_values"].to(device=device, dtype=torch.float32)
    input_ids = processed["input_ids"].to(device=device, dtype=torch.int64)
    if tuple(input_ids.shape) != (len(normalized_texts), 64):
        raise ValueError(f"tokenizer returned unexpected shape {tuple(input_ids.shape)}")
    if torch.any(input_ids[:, -1] != tokenizer.pad_token_id):
        raise ValueError("short smoke prompts should end in right-padding id 0")

    processor_returned_attention_mask = "attention_mask" in processed
    quantize_weights_for_storage(model, torch, args.weight_precision)
    with torch.inference_mode():
        image_raw = model.vision_model(pixel_values=pixel_values).pooler_output.float()
        # HF fixed-resolution checkpoint parity intentionally omits the mask.
        text_raw = model.text_model(input_ids=input_ids, attention_mask=None).pooler_output.float()
        image_normalized = torch.nn.functional.normalize(image_raw, p=2, dim=-1)
        text_normalized = torch.nn.functional.normalize(text_raw, p=2, dim=-1)
        logits_per_text = (
            text_normalized @ image_normalized.transpose(0, 1)
        ) * model.logit_scale.float().exp() + model.logit_bias.float()
        logits_per_image = logits_per_text.transpose(0, 1).contiguous()
        probabilities_per_image = torch.sigmoid(logits_per_image)
        probabilities_per_text = probabilities_per_image.transpose(0, 1).contiguous()

        joined = model(input_ids=input_ids, pixel_values=pixel_values)
        assert_close("image embedding", image_normalized, joined.image_embeds, torch)
        assert_close("text embedding", text_normalized, joined.text_embeds, torch)
        assert_close("logits", logits_per_image, joined.logits_per_image, torch)

    tensors = {
        "input.pixel_values": owned_cpu_tensor(pixel_values),
        "input.input_ids": owned_cpu_tensor(input_ids),
        "output.image_embedding.raw": owned_cpu_tensor(image_raw),
        "output.image_embedding.normalized": owned_cpu_tensor(image_normalized),
        "output.text_embedding.raw": owned_cpu_tensor(text_raw),
        "output.text_embedding.normalized": owned_cpu_tensor(text_normalized),
        "output.logits_per_image": owned_cpu_tensor(logits_per_image),
        "output.logits_per_text": owned_cpu_tensor(logits_per_text),
        "output.probabilities_per_image": owned_cpu_tensor(probabilities_per_image),
        "output.probabilities_per_text": owned_cpu_tensor(probabilities_per_text),
        "model.logit_scale": owned_cpu_tensor(model.logit_scale.float()),
        "model.logit_bias": owned_cpu_tensor(model.logit_bias.float()),
    }

    output_base = normalize_output_base(args.output_base).expanduser().resolve()
    output_base.parent.mkdir(parents=True, exist_ok=True)
    input_image_path = Path(str(output_base) + ".input.png")
    tensor_path = Path(str(output_base) + ".safetensors")
    manifest_path = Path(str(output_base) + ".json")
    input_image_path.write_bytes(encoded_image)
    save_file(
        tensors,
        str(tensor_path),
        metadata={
            "schema": "burn_siglip2.hf_reference.v1",
            "variant": args.variant,
            "weight_precision": args.weight_precision,
        },
    )

    checkpoint_metadata: dict[str, Any] = {
        "hf_dir": str(hf_dir),
        "upstream_revision": args.upstream_revision,
        "model_file": checkpoint_files["model.safetensors"].name,
        "model_bytes": checkpoint_files["model.safetensors"].stat().st_size,
        "model_sha256": None,
        "config_sha256": sha256_file(checkpoint_files["config.json"]),
        "preprocessor_config_sha256": sha256_file(
            checkpoint_files["preprocessor_config.json"]
        ),
        "tokenizer_sha256": sha256_file(checkpoint_files["tokenizer.json"]),
        "tokenizer_config_sha256": sha256_file(
            checkpoint_files["tokenizer_config.json"]
        ),
    }
    if args.hash_model:
        checkpoint_metadata["model_sha256"] = sha256_file(
            checkpoint_files["model.safetensors"]
        )

    manifest = {
        "schema": "burn_siglip2.hf_reference",
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generator": {
            "script": Path(__file__).name,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "device": str(device),
            "threads": args.threads,
        },
        "model": {
            "variant": args.variant,
            "weight_precision": args.weight_precision,
            "resolved_config": resolved_config,
            "checkpoint": checkpoint_metadata,
        },
        "inputs": {
            "encoded_image": {
                "file": input_image_path.name,
                "file_sha256": sha256_bytes(encoded_image),
                "format": "png",
                "width": int(image.width),
                "height": int(image.height),
            },
            "texts": list(original_texts),
        },
        "preprocessing": {
            "image_processor_class": type(processor.image_processor).__name__,
            "use_fast": False,
            "resize_mode": "direct_square",
            "resample": int(processor.image_processor.resample),
            "rescale_factor": float(processor.image_processor.rescale_factor),
            "image_mean": list(processor.image_processor.image_mean),
            "image_std": list(processor.image_processor.image_std),
            "image": image_metadata,
            "text_lowercase": True,
            "padding": "max_length",
            "truncation": True,
            "max_length": 64,
            "attention_mask_forwarded": False,
            "processor_returned_attention_mask": processor_returned_attention_mask,
            "tokenizer": tokenizer_facts,
            "texts_original": list(original_texts),
            "texts_normalized": list(normalized_texts),
        },
        "tensors": {
            "file": tensor_path.name,
            "file_sha256": sha256_file(tensor_path),
            "entries": {name: tensor_entry(tensor) for name, tensor in tensors.items()},
            "row_norms": {
                "output.image_embedding.raw": row_norms(image_raw, torch),
                "output.image_embedding.normalized": row_norms(image_normalized, torch),
                "output.text_embedding.raw": row_norms(text_raw, torch),
                "output.text_embedding.normalized": row_norms(text_normalized, torch),
            },
        },
        "tolerances": {
            "parity": TOLERANCES[args.weight_precision],
            "f16_fidelity_against_upstream_f32": F16_FIDELITY_LIMITS,
        },
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path, tensor_path


def diff_metrics(actual: Iterable[float], reference: Iterable[float]) -> dict[str, float]:
    deltas = [float(a) - float(b) for a, b in zip(actual, reference, strict=True)]
    if not deltas:
        return {"max_abs": 0.0, "rmse": 0.0}
    return {
        "max_abs": max(abs(value) for value in deltas),
        "rmse": math.sqrt(sum(value * value for value in deltas) / len(deltas)),
    }


def self_test() -> None:
    first, first_meta = deterministic_rgb()
    second, second_meta = deterministic_rgb()
    assert first == second
    assert first_meta == second_meta
    assert first_meta["rgb_sha256"] == "37da0b4fb714077c620a406690437940d9196009eb6fd870a1be45a7ce809da3"
    metrics = diff_metrics([1.0, 2.0, 3.0], [1.5, 1.0, 3.0])
    assert abs(metrics["max_abs"] - 1.0) < 1.0e-12
    assert abs(metrics["rmse"] - math.sqrt(1.25 / 3.0)) < 1.0e-12
    assert normalize_output_base(Path("reference.json")) == Path("reference")
    assert normalize_output_base(Path("reference.safetensors")) == Path("reference")
    print("siglip2_reference self-test: ok")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        if args.self_test:
            self_test()
            return 0
        manifest_path, tensor_path = generate(args)
    except (AssertionError, FileNotFoundError, RuntimeError, ValueError) as error:
        print(f"siglip2_reference: error: {error}", file=sys.stderr)
        return 2
    print(f"reference manifest: {manifest_path}")
    print(f"reference tensors:  {tensor_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
