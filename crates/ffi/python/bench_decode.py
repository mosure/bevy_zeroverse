#!/usr/bin/env python
"""Benchmark chunked JPEG decode throughput for CPU vs GPU collate."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from bevy_zeroverse_dataloader import (
    BevyZeroverseDataset,
    ChunkedIteratorDataset,
    chunk_and_save,
    chunk_collate,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark chunked JPEG decode throughput (CPU vs GPU collate).",
    )
    parser.add_argument(
        "--chunk-dir",
        type=Path,
        default=Path("data/zeroverse/bench_decode/chunk"),
        help="Path to a chunked dataset directory.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=32,
        help="Number of samples to generate if chunk dir is missing.",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=4,
        help="Number of cameras to render when generating.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Render width when generating.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Render height when generating.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=2,
        help="Batch size for the DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader workers (use 0 for GPU tensors).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of timed batches to run per mode.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
        help="Number of warmup batches per mode.",
    )
    parser.add_argument(
        "--pin-memory",
        action="store_true",
        help="Enable DataLoader pinned memory.",
    )
    return parser.parse_args()


def ensure_chunks(args: argparse.Namespace) -> None:
    if args.chunk_dir.exists() and list(args.chunk_dir.glob("*.safetensors*")):
        return

    args.chunk_dir.mkdir(parents=True, exist_ok=True)
    dataset = BevyZeroverseDataset(
        editor=False,
        headless=True,
        num_cameras=args.num_cameras,
        width=args.width,
        height=args.height,
        num_samples=args.num_samples,
        scene_type="room",
        ovoxel_mode="disabled",
    )
    chunk_and_save(
        dataset,
        args.chunk_dir,
        bytes_per_chunk=int(256 * 1024 * 1024),
        samples_per_chunk=min(args.num_samples, 8),
        n_workers=0,
        full_size_only=False,
    )


def benchmark_loader(name: str, loader: DataLoader, steps: int, warmup: int) -> Dict[str, float]:
    timings: List[float] = []
    iterator = iter(loader)

    for _ in range(warmup):
        try:
            next(iterator)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except StopIteration:
            break

    for _ in range(steps):
        start = time.perf_counter()
        try:
            next(iterator)
        except StopIteration:
            break
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        timings.append(time.perf_counter() - start)

    if not timings:
        return {"name": name, "batches": 0, "sec_per_batch": 0.0, "batches_per_sec": 0.0}

    sec_per_batch = statistics.fmean(timings)
    return {
        "name": name,
        "batches": len(timings),
        "sec_per_batch": sec_per_batch,
        "batches_per_sec": (1.0 / sec_per_batch) if sec_per_batch > 0 else 0.0,
    }


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        print("[bench_decode] CUDA not available; benchmark requires a CUDA device.")
        return 1

    ensure_chunks(args)

    default_dataset = ChunkedIteratorDataset(args.chunk_dir)
    default_loader = DataLoader(
        default_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    gpu_dataset = ChunkedIteratorDataset(
        args.chunk_dir,
        jpeg_device="cuda",
        keep_jpeg_on_device=True,
    )
    gpu_loader = DataLoader(
        gpu_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=lambda samples: chunk_collate(samples, device="cuda", non_blocking=True),
    )

    print("[bench_decode] running with CUDA device")
    print(f"[bench_decode] chunk dir: {args.chunk_dir}")
    print(f"[bench_decode] batch size: {args.batch_size}, workers: {args.num_workers}")

    default_stats = benchmark_loader("default_cpu_collate", default_loader, args.steps, args.warmup)
    gpu_stats = benchmark_loader("gpu_collate", gpu_loader, args.steps, args.warmup)

    print()
    print(f"{'mode':>20}  {'batches':>7}  {'sec/batch':>10}  {'batches/s':>10}")
    print("-" * 56)
    for stats in (default_stats, gpu_stats):
        print(
            f"{stats['name']:>20}  "
            f"{stats['batches']:7d}  "
            f"{stats['sec_per_batch']:10.4f}  "
            f"{stats['batches_per_sec']:10.2f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
