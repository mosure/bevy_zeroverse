#!/usr/bin/env python
"""Benchmark Zeroverse sample generation throughput across worker counts."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import shutil
import statistics
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    import psutil  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    psutil = None  # type: ignore

try:
    import pynvml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    pynvml = None  # type: ignore

import torch

from bevy_zeroverse_dataloader import (
    BevyZeroverseDataset,
    chunk_and_save,
    get_chunk_sample_count,
)


def human_bytes(value: float) -> str:
    if value <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    idx = min(len(units) - 1, int(math.log(value, 1024)))
    scaled = value / (1024 ** idx)
    return f"{scaled:.2f} {units[idx]}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark dataset generation throughput for different worker counts.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/zeroverse/bench"),
        help="Directory root where per-worker results are stored.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=64,
        help="Number of samples to generate for each benchmark run.",
    )
    parser.add_argument(
        "--workers",
        type=str,
        default=None,
        help="Comma separated list of worker counts to benchmark (overrides max-workers).",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=16,
        help="Upper bound when generating a default worker sweep (0,1,2,4,8,16,…).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=2,
        help="Number of repetitions per worker configuration.",
    )
    parser.add_argument(
        "--bytes-per-chunk",
        type=int,
        default=256 * 1024 * 1024,
        help="Target byte size per chunk when saving samples.",
    )
    parser.add_argument(
        "--samples-per-chunk",
        type=int,
        default=8,
        help="Maximum samples per chunk (0 disables the cap).",
    )
    parser.add_argument(
        "--scene-type",
        type=str,
        default="room",
        help="Scene type for the generated samples.",
    )
    parser.add_argument(
        "--num-cameras",
        type=int,
        default=4,
        help="Number of cameras to render.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=640,
        help="Render width for generated samples.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=480,
        help="Render height for generated samples.",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU index to monitor for VRAM usage (requires pynvml).",
    )
    parser.add_argument(
        "--gpu-interval",
        type=float,
        default=0.5,
        help="GPU polling interval in seconds when monitoring VRAM/utilisation.",
    )
    parser.add_argument(
        "--json",
        type=Path,
        default=None,
        help="Optional path to write detailed results as JSON.",
    )
    parser.add_argument(
        "--cleanup-output",
        action="store_true",
        help="Remove per-run output directories after benchmarking.",
    )
    parser.add_argument(
        "--keep-existing",
        action="store_true",
        help="Skip regeneration if a per-worker result directory already exists.",
    )
    return parser.parse_args()


def resolve_worker_counts(args: argparse.Namespace) -> List[int]:
    if args.workers:
        counts = sorted({int(item.strip()) for item in args.workers.split(",") if item.strip()})
    else:
        template = [0, 1, 2, 4, 8, 16]
        cpu_cap = os.cpu_count() or args.max_workers
        limit = min(args.max_workers, cpu_cap)
        counts = [value for value in template if value <= limit]
        if limit not in counts:
            counts.append(limit)
    if not counts:
        raise ValueError("No worker counts specified.")
    return counts


@dataclass
class GpuStats:
    available: bool
    baseline_mem: Optional[int] = None
    peak_mem: Optional[int] = None
    final_mem: Optional[int] = None
    peak_util: Optional[int] = None


class GpuMonitor:
    def __init__(self, index: int, interval: float) -> None:
        self.index = index
        self.interval = interval
        self._records: List[tuple[int, int]] = []  # (mem_used, util)
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._handle = None
        self._available = False

        if pynvml is None:
            return

        try:
            pynvml.nvmlInit()
            device_count = pynvml.nvmlDeviceGetCount()
            if not (0 <= index < device_count):
                print(f"[bench] GPU index {index} out of range (available {device_count})", file=sys.stderr)
                return
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(index)
            self._available = True
        except Exception as exc:  # pragma: no cover - depends on runtime
            print(f"[bench] NVML unavailable: {exc}", file=sys.stderr)
            self._handle = None
            self._available = False

    def is_available(self) -> bool:
        return self._available and self._handle is not None

    def _poll(self) -> None:
        assert self._handle is not None
        while not self._stop.is_set():
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                self._records.append((mem.used, util.gpu))
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self) -> Optional[int]:
        if not self.is_available():
            return None
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self._records.clear()
        self._records.append((mem.used, 0))
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return mem.used

    def stop(self) -> GpuStats:
        if not self.is_available():
            return GpuStats(False)
        self._stop.set()
        if self._thread:
            self._thread.join()
        mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self._records.append((mem.used, 0))
        mem_values = [entry[0] for entry in self._records]
        util_values = [entry[1] for entry in self._records]
        return GpuStats(
            available=True,
            baseline_mem=self._records[0][0] if self._records else mem.used,
            peak_mem=max(mem_values) if mem_values else mem.used,
            final_mem=mem.used,
            peak_util=max(util_values) if util_values else 0,
        )


def make_dataset(args: argparse.Namespace) -> BevyZeroverseDataset:
    return BevyZeroverseDataset(
        editor=False,
        headless=True,
        num_cameras=args.num_cameras,
        width=args.width,
        height=args.height,
        num_samples=args.num_samples,
        scene_type=args.scene_type,
    )


def remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def run_generation(
    target_dir: Path,
    workers: int,
    args: argparse.Namespace,
    process: "psutil.Process | None",
    gpu_monitor: GpuMonitor,
) -> Dict[str, Any]:
    if target_dir.exists() and not args.keep_existing:
        remove_tree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    gc.collect()
    if torch.cuda.is_available():  # pragma: no cover - runtime dependent
        torch.cuda.empty_cache()

    dataset = make_dataset(args)

    base_rss = process.memory_info().rss if process else None
    gpu_monitor.start()

    start = time.perf_counter()
    chunk_paths = chunk_and_save(
        dataset,
        target_dir,
        bytes_per_chunk=args.bytes_per_chunk,
        samples_per_chunk=(args.samples_per_chunk or None),
        n_workers=workers,
        full_size_only=False,
    )
    duration = max(time.perf_counter() - start, sys.float_info.epsilon)

    gpu_stats = gpu_monitor.stop()

    if process:
        final_rss = process.memory_info().rss
        peak_rss = max(base_rss or final_rss, final_rss)
    else:
        final_rss = None
        peak_rss = None

    bytes_written = 0
    samples_written = 0
    chunk_dir = Path(target_dir)
    chunk_files = list(chunk_dir.glob('*.safetensors*'))
    if not chunk_files:
        chunk_files = [Path(p) for p in chunk_paths]
    for chunk_path in chunk_files:
        if chunk_path.exists():
            bytes_written += chunk_path.stat().st_size
            try:
                samples_written += get_chunk_sample_count(chunk_path)
            except Exception:
                pass
    if samples_written == 0:
        samples_written = args.num_samples

    throughput = samples_written / duration if samples_written else 0.0
    bandwidth = bytes_written / duration if duration > 0 else 0.0

    return {
        "workers": workers,
        "duration": duration,
        "samples": samples_written,
        "bytes": bytes_written,
        "throughput": throughput,
        "bandwidth": bandwidth,
        "rss_base": base_rss,
        "rss_peak": peak_rss,
        "rss_final": final_rss,
        "gpu_available": gpu_stats.available,
        "gpu_mem_base": gpu_stats.baseline_mem if gpu_stats.available else None,
        "gpu_mem_peak": gpu_stats.peak_mem if gpu_stats.available else None,
        "gpu_mem_final": gpu_stats.final_mem if gpu_stats.available else None,
        "gpu_util_peak": gpu_stats.peak_util if gpu_stats.available else None,
        "output_dir": str(target_dir),
    }


def summarise(results: List[Dict[str, Any]]) -> Dict[int, Dict[str, float]]:
    summary: Dict[int, Dict[str, float]] = {}
    grouped: Dict[int, List[Dict[str, Any]]] = {}
    for record in results:
        grouped.setdefault(record["workers"], []).append(record)

    for workers, records in grouped.items():
        thr = [entry["throughput"] for entry in records]
        bw = [entry["bandwidth"] for entry in records]
        peak_rss = [
            (entry["rss_peak"] - entry["rss_base"])
            for entry in records
            if entry["rss_peak"] is not None and entry["rss_base"] is not None
        ]
        gpu_delta = [
            (entry["gpu_mem_peak"] - entry["gpu_mem_base"])
            for entry in records
            if entry.get("gpu_available") and entry.get("gpu_mem_peak") is not None and entry.get("gpu_mem_base") is not None
        ]
        gpu_util = [entry["gpu_util_peak"] for entry in records if entry.get("gpu_util_peak") is not None]
        summary[workers] = {
            "throughput_mean": statistics.fmean(thr) if thr else 0.0,
            "throughput_std": statistics.pstdev(thr) if len(thr) > 1 else 0.0,
            "bandwidth_mean": statistics.fmean(bw) if bw else 0.0,
            "bandwidth_std": statistics.pstdev(bw) if len(bw) > 1 else 0.0,
            "rss_peak_delta": max(peak_rss) if peak_rss else 0.0,
            "gpu_mem_delta": max(gpu_delta) if gpu_delta else 0.0,
            "gpu_util_peak": max(gpu_util) if gpu_util else 0.0,
            "samples": int(records[0]["samples"]),
        }
    return summary


def cleanup(results: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if not args.cleanup_output:
        return
    for record in results:
        remove_tree(Path(record["output_dir"]))


def main() -> int:
    args = parse_args()
    workers = resolve_worker_counts(args)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    process = psutil.Process(os.getpid()) if psutil else None
    gpu_monitor = GpuMonitor(args.gpu_index, args.gpu_interval)

    results: List[Dict[str, Any]] = []

    print("[bench] benchmarking generation for worker counts:", workers)

    for worker_count in workers:
        for rep in range(args.repeats):
            run_dir = args.output_dir / f"workers_{worker_count}" / f"repeat_{rep+1}"
            if args.keep_existing and run_dir.exists():
                print(f"[bench] workers={worker_count} repeat={rep+1}: skipping (exists)")
                continue
            print(f"[bench] workers={worker_count} repeat={rep+1}: generating {args.num_samples} samples…", flush=True)
            record = run_generation(run_dir, worker_count, args, process, gpu_monitor)
            results.append(record)

    if not results:
        print("[bench] no runs executed")
        return 0

    summary = summarise(results)

    print()
    header = (
        f"{'workers':>7}  {'samples':>8}  {'thr (samples/s)':>18}  "
        f"{'thr std':>10}  {'bandwidth':>14}  {'rss peak':>12}  {'vram peak':>10}  {'GPU util%':>10}"
    )
    print(header)
    print("-" * len(header))
    for workers_count in sorted(summary):
        row = summary[workers_count]
        print(
            f"{workers_count:7d}  "
            f"{row['samples']:8d}  "
            f"{row['throughput_mean']:18.2f}  "
            f"{row['throughput_std']:10.2f}  "
            f"{human_bytes(row['bandwidth_mean']):>14}  "
            f"{human_bytes(row['rss_peak_delta']):>12}  "
            f"{human_bytes(row['gpu_mem_delta']):>10}  "
            f"{row['gpu_util_peak']:10.1f}"
        )

    if args.json:
        payload = {
            "args": vars(args),
            "results": results,
            "summary": summary,
        }
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2))
        print(f"[bench] wrote JSON results to {args.json}")

    cleanup(results, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

