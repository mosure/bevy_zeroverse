#!/usr/bin/env python3
"""Summarize indoor_validate captures; optional figures need Pillow and matplotlib.

Raw scene/camera/object CSV and metric histograms are exported by indoor_validate.
This report adds measured RGB/annotation/pose/semantic visibility coverage. It never
filters failed captures or infers photographic realism from signal thresholds.
"""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics


def distribution(values):
    values = sorted(values)
    if not values:
        return None
    return {
        "count": len(values), "min": values[0], "max": values[-1],
        "mean": statistics.fmean(values),
        "p05_p50_p95": [values[round((len(values) - 1) * p)] for p in (0.05, 0.5, 0.95)],
    }


def summarize(root):
    metrics = json.loads((root / "metrics.json").read_text())
    selection = json.loads((root / "render_selection.json").read_text())
    completion = json.loads((root / "run_complete.json").read_text())
    if not selection.get("run_id") or completion.get("run_id") != selection["run_id"]:
        raise ValueError("capture run incomplete or stale completion marker")
    if not selection["selected_seeds"] or completion["selected_seeds"] != selection["selected_seeds"]:
        raise ValueError("capture selection missing or completion mismatch")
    reports = []
    for seed in selection["selected_seeds"]:
        directory = root / f"seed_{seed:06}"
        report = json.loads((directory / "capture.json").read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        if report.get("run_id") != selection["run_id"]:
            raise ValueError(f"stale capture from a different run: {seed}")
        if report["image_size"] != metrics["image_size"] or manifest["density"] != metrics["density"]:
            raise ValueError(f"capture and audit configuration mismatch: {seed}")
        if report["capabilities"]["quality"] != selection["quality"]:
            raise ValueError(f"capture quality mismatch: {seed}")
        if report["seed"] != seed or manifest["seed"] != seed:
            raise ValueError(f"manifest/capture seed mismatch: {seed}")
        if manifest["generator_version"] != metrics["generator_version"]:
            raise ValueError(f"mixed generator versions: {seed}")
        reports.append((directory, report, manifest))
    views = [(directory, i, view, manifest) for directory, report, manifest in reports
             for i, view in enumerate(report["views"])]
    n = len(views)
    pixel_counts = Counter()
    visibility = Counter()
    for _, _, view, _ in views:
        pixel_counts.update(view["semantic_pixel_counts"])
        visibility.update(name for name, count in view["semantic_pixel_counts"].items() if count > 0)
    expected = sum(len(m["cameras"]) * selection["playback_steps"] for _, _, m in reports)
    if n != expected:
        raise ValueError(f"view count mismatch: {n} != {expected}")
    if any(not math.isfinite(v[k]) for _, _, v, _ in views for k in ("mean_luminance", "pose_max_absolute_error")):
        raise ValueError("nonfinite rendered metrics")
    view_metrics = {key: distribution([v[key] for _, _, v, _ in views]) for key in (
        "mean_luminance", "luminance_std", "dark_fraction", "clipped_fraction", "pose_max_absolute_error")}
    annotation_metrics = {key: distribution([v["annotation_alignment"][key] for _, _, v, _ in views
                                            if v["annotation_alignment"] is not None]) for key in (
        "depth_position_p99_metres", "reprojection_p99_pixels", "normal_length_max_error",
        "position_quantization_budget_p99_ratio")}
    per_layout = defaultdict(list)
    for _, _, view, manifest in views:
        per_layout[manifest["layout"]].append(view["mean_luminance"])
    result = {
        "schema_version": 1, "run_id": selection["run_id"], "generator_version": metrics["generator_version"],
        "capture_engine": selection.get("capture_engine"),
        "selection_policy": selection["policy"], "audit_scenes": metrics["scenes"],
        "rendered_scenes": len(reports), "rendered_views": n,
        "playback_steps": selection["playback_steps"], "image_size": metrics["image_size"],
        "observed_strata": len(selection["observed_strata"]),
        "captured_layouts": dict(Counter(m["layout"] for _, _, m in reports)),
        "captured_lighting": dict(Counter(m["lighting"] for _, _, m in reports)),
        "captured_floors": dict(Counter(m["floor_style"] for _, _, m in reports)),
        "captured_furniture": dict(Counter(m["furniture_style"] for _, _, m in reports)),
        "captured_architecture": dict(Counter(m.get("architecture_style", "Contemporary") for _, _, m in reports)),
        "captured_plant_species": dict(Counter(str(o["variant"] % 6) for _, _, m in reports
                                              for o in m.get("objects", []) if o["kind"] == "Plant")),
        "rgb_and_pose": view_metrics, "annotation": annotation_metrics,
        "mean_linear_luminance_by_layout": {k: distribution(v) for k, v in per_layout.items()},
        "semantic_pixel_counts": dict(pixel_counts),
        "semantic_view_presence_fraction": {k: v / n for k, v in visibility.items()},
        "asset_counts": {k: distribution([r[k] for _, r, _ in reports])
                         for k in ("mesh_assets", "material_assets", "image_assets")},
        "capture_wall_seconds": distribution([r["elapsed_seconds"] for _, r, _ in reports]),
        "renderer": sorted({r["renderer"] for _, r, _ in reports}),
        "annotation_precision": sorted({r.get("annotation_precision", "float16_hdr") for _, r, _ in reports}),
        "capabilities": reports[0][1]["capabilities"] if reports else None,
        "limits": ["Signal and constraint gates do not measure photographic realism or real-office distribution match.",
                   "Camera paths checked at every captured timestep; geometry sweep is also audited separately.",
                   "Semantic pixel presence differs from scene instance counts; glazing is opaque in annotation passes.",
                   "Annotation precision is recorded explicitly; legacy HDR and native geometry captures have different error budgets.",
                   "Capture wall time includes generation, warmup, requested render work and disk IO; not FPS or isolated throughput."],
    }
    for name, order in [("darkest_views", lambda v: v[2]["mean_luminance"]),
                        ("brightest_views", lambda v: -v[2]["mean_luminance"]),
                        ("largest_position_error_views", lambda v: -(v[2]["annotation_alignment"] or {}).get("depth_position_p99_metres", 0))]:
        result[name] = [{"path": str(p / f"view_{i:02}_color.png"), "seed": m["seed"], "view": i,
                         "mean_luminance": v["mean_luminance"]} for p, i, v, m in sorted(views, key=order)[:8]]
    (root / "render_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return metrics, reports, result


def figures(root, metrics, reports, summary):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont

    plt.rcParams.update({"svg.fonttype": "none", "font.size": 10})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    counts = metrics["object_counts_per_scene"]["main/Chair"]
    xs = sorted(map(int, counts))
    axes[0, 0].bar(xs, [counts[str(x)] / metrics["scenes"] for x in xs], color="#356aa0")
    axes[0, 0].set(xlabel="Main-room chair instances (including zero)", ylabel="Fraction of scenes")
    for ax, key, label in zip(list(axes.flat)[1:],
                             ["vertical_fov_degrees", "fx_pixels", "camera_height_m", "trajectory_length_m", "camera_pitch_degrees"],
                             ["Vertical field of view (degrees)", "Focal length fx=fy (pixels)", "Camera start height (m)", "Translation length (m)", "Camera pitch (degrees)"]):
        d = metrics["numeric"][key]
        edges = d["bin_edges"]
        ax.bar(edges[:-1], [v / d["count"] for v in d["bin_counts"]],
               width=[b - a for a, b in zip(edges, edges[1:])], align="edge", color="#357c75")
        ax.set(xlabel=label, ylabel="Fraction of cameras")
    fig.suptitle(f"Generator v{metrics['generator_version']}: {metrics['scenes']:,} scenes at density {metrics['density']}")
    fig.savefig(root / "distributions.svg")
    fig.savefig(root / "distributions.png", dpi=140)
    plt.close(fig)
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    try:
        font = ImageFont.truetype(font_path, 13)
    except OSError:
        font = ImageFont.load_default()
    # Include every captured view, including awkward camera angles and occlusion.
    views = [(directory, i, manifest) for directory, report, manifest in reports
             for i in range(len(report["views"]))]
    columns, tile_w, tile_h = 6, 320, 278
    sheet = Image.new("RGB", (columns * tile_w, math.ceil(len(views) / columns) * tile_h), "#f4f6f8")
    draw = ImageDraw.Draw(sheet)
    for i, (directory, view, manifest) in enumerate(views):
        img = Image.open(directory / f"view_{view:02}_color.png").convert("RGB")
        img.thumbnail((tile_w, 240))
        x, y = (i % columns) * tile_w, (i // columns) * tile_h
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + 243), f"seed {manifest['seed']} view {view} | {manifest['layout']}", fill="#182433", font=font)
        draw.text((x + 4, y + 258), f"{manifest.get('architecture_style', 'Contemporary')} | {manifest['lighting']}", fill="#182433", font=font)
    sheet.save(root / "contact_all_views.jpg", quality=92)
    by_layout = defaultdict(list)
    for row in reports:
        by_layout[row[2]["layout"]].append(row)
    for layout, rows in by_layout.items():
        columns, tile_w, tile_h = 3, 320, 270
        sheet = Image.new("RGB", (columns * tile_w, math.ceil(len(rows) / columns) * tile_h), "#f4f6f8")
        draw = ImageDraw.Draw(sheet)
        for i, (directory, _, manifest) in enumerate(rows):
            image = Image.open(directory / "view_00_color.png").convert("RGB")
            image.thumbnail((tile_w, 240))
            x, y = (i % columns) * tile_w, (i // columns) * tile_h
            sheet.paste(image, (x, y))
            draw.text((x + 4, y + 243), f"seed {manifest['seed']} | {manifest['lighting']} | floor {manifest['floor_style']} / furniture {manifest['furniture_style']}", fill="#182433", font=font)
        sheet.save(root / f"contact_{layout.lower()}.png")
    worst = summary["darkest_views"][:4] + summary["brightest_views"][:4]
    sheet = Image.new("RGB", (4 * 320, 2 * 270), "#f4f6f8")
    draw = ImageDraw.Draw(sheet)
    for i, item in enumerate(worst):
        img = Image.open(item["path"]).convert("RGB")
        img.thumbnail((320, 240))
        x, y = (i % 4) * 320, (i // 4) * 270
        sheet.paste(img, (x, y))
        draw.text((x + 4, y + 243), f"seed {item['seed']} view {item['view']} | linear Y={item['mean_luminance']:.3f}", fill="#182433", font=font)
    sheet.save(root / "contact_exposure_extremes.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--figures", action="store_true", help="create charts and complete first-view contact sheets")
    args = parser.parse_args()
    metrics, reports, summary = summarize(args.directory)
    if args.figures:
        figures(args.directory, metrics, reports, summary)
    print(json.dumps({k: summary[k] for k in ("audit_scenes", "rendered_scenes", "rendered_views", "observed_strata", "rgb_and_pose", "annotation")}, indent=2))


if __name__ == "__main__":
    main()
