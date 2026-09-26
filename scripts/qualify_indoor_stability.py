#!/usr/bin/env python3
"""Finite, reproducible qualification of a continuous capture process.

Uses complete captures, actual live glibc allocations, RSS and retained renderer
resources. Block means reduce scene-to-scene noise; confidence intervals describe
the measured window, not an assertion about infinite time or other devices.
"""
import argparse
import json
import math
from pathlib import Path
import statistics

from indoor_bench_report import validate_benchmark, summarize_telemetry


PROTOCOL = {
    "version": 1,
    "minimum_scenes": 4096,
    "minimum_warmup": 512,
    "tail_scenes": 2048,
    "block_scenes": 64,
    "heap_slope_upper_bytes_per_scene": 8192,
    "rss_slope_upper_bytes_per_scene": 32768,
    "heap_late_to_early_ratio": 1.08,
    "rss_late_to_early_ratio": 1.15,
    "maximum_live_heap_bytes": 2 * 1024**3,
    "maximum_rss_bytes": 4608 * 1024**2,
    "maximum_retained_bind_groups": 512,
    "maximum_retained_textures": 256,
    "maximum_retained_buffers": 1024,
}


def blocked_trend(values, block=64):
    """OLS on nonoverlapping block means, with a conservative t~2.1 upper bound."""
    means = [statistics.mean(values[i:i + block]) for i in range(0, len(values) - block + 1, block)]
    if len(means) < 16:
        raise ValueError("At least 16 full blocks are required for a stability estimate")
    x = [(i + .5) * block for i in range(len(means))]
    mx, my = statistics.mean(x), statistics.mean(means)
    sxx = sum((v - mx)**2 for v in x)
    slope = sum((a - mx) * (b - my) for a, b in zip(x, means)) / sxx
    residual = sum((b - my - slope * (a - mx))**2 for a, b in zip(x, means))
    se = math.sqrt(residual / (len(means) - 2) / sxx)
    return {"slope_bytes_per_scene": slope, "standard_error": se,
            "approximate_95pct_upper_slope": slope + 2.1 * se,
            "block_means_bytes": means, "block_scenes": block}


def align_gpu_memory(scenes, polls, maximum_age):
    """Use the last observed PID memory before each completion, with bounded age.

    This aligns memory polls only. Utilization is never interpolated or filled.
    """
    index, latest, values = 0, None, []
    for scene in scenes:
        stamp = scene["completed_unix_seconds"]
        while index < len(polls) and polls[index]["wall_unix_seconds"] <= stamp:
            latest = polls[index]
            index += 1
        if latest is None or stamp - latest["wall_unix_seconds"] > maximum_age:
            raise ValueError("GPU memory observation missing or stale at scene completion")
        value = latest["process_gpu_memory_bytes"]
        if value is None or not math.isfinite(value) or value <= 0:
            raise ValueError("GPU memory unavailable; do not substitute zero")
        values.append(value)
    return values


def add_gpu_qualification(report, directory, summary, rows):
    telemetry = summarize_telemetry(directory, summary, rows)
    if not telemetry["available"]:
        raise ValueError("This protocol requires completed PID-specific GPU memory telemetry")
    meta = json.loads((directory / "telemetry_summary.json").read_text())
    polls = [json.loads(line) for line in (directory / "telemetry.jsonl").read_text().splitlines()]
    protocol = report["protocol"]
    measured = rows[summary["warmup_scenes"]:]
    values = align_gpu_memory(measured, polls, max(1.0, 3 * meta["interval_seconds"]))
    trend = blocked_trend(values[-protocol["tail_scenes"]:], protocol["block_scenes"])
    early, late = statistics.mean(values[:512]), statistics.mean(values[-512:])
    maximum = max(p["process_gpu_memory_bytes"] for p in polls
                  if p["process_gpu_memory_bytes"] is not None)
    report["metrics"]["gpu_memory"] = {
        **trend, "early_512_mean_bytes": early, "late_512_mean_bytes": late,
        "late_to_early_ratio": late / early, "maximum_observed_bytes": maximum,
        "alignment": "Latest preceding PID-specific NVML memory poll, at most three polling intervals old; no utilization interpolation",
    }
    report["gates"].update(
        gpu_memory_evidence_present=True,
        gpu_memory_trend=trend["approximate_95pct_upper_slope"] <= protocol["gpu_memory_slope_upper_bytes_per_scene"],
        gpu_memory_window_growth=late / early <= protocol["gpu_memory_late_to_early_ratio"],
        gpu_memory_budget=maximum <= protocol["maximum_gpu_memory_bytes"],
    )
    report["accepted"] = all(report["gates"].values())


def qualify(summary, rows, protocol=None):
    protocol = dict(PROTOCOL if protocol is None else protocol)
    validate_benchmark(summary, rows)
    if summary["scenes"] < protocol["minimum_scenes"] or summary["warmup_scenes"] < protocol["minimum_warmup"]:
        raise ValueError("Run does not meet the preregistered population/warmup requirements")
    measured = rows[summary["warmup_scenes"]:]
    tail = protocol["tail_scenes"]
    if len(measured) < tail:
        raise ValueError("Insufficient measured scenes for the tail window")
    quantities = {
        "heap": [r["heap"]["live_heap_bytes"] + r["heap"]["mapped_bytes"] for r in measured],
        "rss": [r["rss_bytes"] for r in measured],
    }
    if any(not math.isfinite(v) or v <= 0 for values in quantities.values() for v in values):
        raise ValueError("Missing or invalid measured memory")
    metrics, gates = {}, {}
    for name, values in quantities.items():
        trend = blocked_trend(values[-tail:], protocol["block_scenes"])
        early, late = statistics.mean(values[:512]), statistics.mean(values[-512:])
        metrics[name] = {**trend, "early_512_mean_bytes": early, "late_512_mean_bytes": late,
                         "late_to_early_ratio": late / early, "maximum_bytes": max(values)}
        gates[name + "_trend"] = trend["approximate_95pct_upper_slope"] <= protocol[name + "_slope_upper_bytes_per_scene"]
        gates[name + "_window_growth"] = late / early <= protocol[name + "_late_to_early_ratio"]
    gates["heap_budget"] = metrics["heap"]["maximum_bytes"] <= protocol["maximum_live_heap_bytes"]
    gates["rss_budget"] = metrics["rss"]["maximum_bytes"] <= protocol["maximum_rss_bytes"]
    resource_max = {}
    for resource in ("bind_groups", "textures", "buffers"):
        resource_max[resource] = max(r["gpu_registry"][resource]["kept_from_user"] for r in measured)
        gates[resource + "_budget"] = resource_max[resource] <= protocol["maximum_retained_" + resource]
    gates["capture_staging_constant"] = len({r["staging_bytes"] for r in measured}) == 1
    gates["one_process"] = len({r["pid"] for r in rows}) == 1
    gates["all_requested_captures"] = rows[-1]["completed_capture_requests"] == len(rows)
    if "maximum_command_encoders" in protocol:
        gates["command_encoder_budget"] = max(r["hal_memory"]["command_encoders"] for r in measured) <= protocol["maximum_command_encoders"]
    if protocol.get("require_gpu_memory", False):
        gates["gpu_memory_evidence_present"] = False
    return {"protocol": protocol, "accepted": all(gates.values()), "gates": gates,
            "scope": "Finite continuous-process qualification on this adapter/configuration; no extrapolation to infinite time or untested configurations",
            "pid": summary["pid"], "run_id": summary["run_id"], "capture_engine": summary["capture_engine"],
            "adapter": summary["adapter"], "scenes": summary["scenes"],
            "views": sum(r["views"] for r in rows), "warmup_scenes": summary["warmup_scenes"],
            "metrics": metrics, "maximum_retained_resources": resource_max,
            "command_encoder_range": [min(r["hal_memory"]["command_encoders"] for r in measured),
                                       max(r["hal_memory"]["command_encoders"] for r in measured)]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    summary = json.loads((args.run / "summary.json").read_text())
    rows = [json.loads(line) for line in (args.run / "scenes.jsonl").read_text().splitlines()]
    protocol = json.loads(args.protocol.read_text())["protocol"] if args.protocol else PROTOCOL
    report = qualify(summary, rows, protocol)
    if report["protocol"].get("require_gpu_memory", False):
        add_gpu_qualification(report, args.run, summary, rows)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"accepted": report["accepted"], "gates": report["gates"]}))
    raise SystemExit(0 if report["accepted"] else 1)


if __name__ == "__main__":
    main()
