#!/usr/bin/env python3
"""Audit built exterior apertures and show every captured room view.

Opening areas are rough wall apertures (including their frames); they exclude
internal partitions and neighboring rooms. The CSV has one row per opening,
while exposure/full-height prevalence is counted once per scene.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

from indoor_report import summarize
from report_indoor_tabletop import contact_sheet


def report(captures, output):
    metrics, rooms, render = summarize(captures)
    output.mkdir(parents=True, exist_ok=True)
    with (captures / "windows.csv").open() as stream:
        windows = list(csv.DictReader(stream))
    assert windows, "missing exterior apertures"
    assert len(windows) == metrics["numeric"]["exterior_opening_width_m"]["count"]
    counts = metrics["categories"]["exterior_wall_count"]
    assert sum(counts.values()) == metrics["scenes"]
    seeds = {int(w["seed"]) for w in windows}
    assert len(seeds) == metrics["scenes"]
    full = {int(w["seed"]) for w in windows if w["full_height"] == "true"}
    assert len(full) == metrics["categories"]["exterior_full_height_room"].get("true", 0)
    for _, _, manifest in rooms:
        rows = [w for w in windows if int(w["seed"]) == manifest["seed"]]
        expected = sum(len(f["openings"]) for f in manifest["exterior"]["facades"])
        assert len(rows) == expected, "render/audit aperture count mismatch"
    views = [(d / f"view_{i:02}_color.png", f"seed{m['seed']} view{i} | " + "+".join(f["side"] for f in m["exterior"]["facades"]))
             for d, c, m in rooms for i in range(len(c["views"]))]
    contact_sheet(views, output / "rooms.png", 400, 300)
    plots(metrics, windows, output)
    elevations(rooms, output)
    geometry = {p.name: json.loads(p.read_text()) for p in sorted(captures.glob("geometry_*.json"))}
    artifacts = [captures / name for name in ("windows.csv", "metrics.json", "distribution.json", "render_selection.json", "run_complete.json")]
    artifacts += [d / name for d, _, _ in rooms for name in ("manifest.json", "capture.json")]
    artifacts += [p for p, _ in views]
    result = {
        "generator_version": metrics["generator_version"], "capture_engine": render["capture_engine"],
        "audit_scenes": metrics["scenes"], "openings": len(windows),
        "exterior_wall_counts": counts, "rooms_with_full_height_aperture": len(full),
        "full_height_definition": "rough sill <= 0.16m and head clearance <= 0.18m; frames reduce clear glass dimensions",
        "categories": {k: v for k, v in metrics["categories"].items() if k.startswith("exterior_")},
        "parameters": {k: v for k, v in metrics["numeric"].items() if k.startswith("exterior_")},
        "render_check": render, "mesh_checks": geometry,
        "artifact_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in artifacts},
        "limits": ["Room renders are consecutive unfiltered seeds, not a photographic-realism benchmark.",
                   "All samples retain a glazed internal wall to the neighboring room; the other three walls are sampled for exterior exposure.",
                   "Finite audit and structural tests do not establish 10M-sample training utility."],
    }
    (output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    (output / "layout_audit.json").write_bytes((captures / "distribution.json").read_bytes())
    (output / "windows.csv").write_bytes((captures / "windows.csv").read_bytes())
    return result


def plots(metrics, windows, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for ax, key, label in [(axes[0, 0], "exterior_wall_count", "Exterior window walls per room"),
                           (axes[0, 1], "exterior_sides", "Wall combinations (room count)")]:
        data = metrics["categories"][key]
        ax.bar(list(data), list(data.values()), color="#326c80")
        ax.set(xlabel=label, ylabel="Rooms")
        ax.tick_params(axis="x", rotation=30)
    for ax, key, label in [(axes[0, 2], "exterior_opening_area_fraction", "Rough opening area / facade area"),
                           (axes[1, 0], "exterior_sill_height_m", "Sill height (m)"),
                           (axes[1, 1], "exterior_opening_width_m", "Opening width (m)")]:
        data = metrics["numeric"][key]
        edges = data["bin_edges"]
        ax.bar(edges[:-1], data["bin_counts"], width=[b-a for a,b in zip(edges, edges[1:])], align="edge", color="#3b806b")
        ax.set(xlabel=label, ylabel="Facades" if "fraction" in key else "Openings")
    axes[1, 2].scatter([float(w["width_m"]) for w in windows], [float(w["height_m"]) for w in windows],
                       c=["#bf7539" if w["full_height"] == "true" else "#326c80" for w in windows], s=8, alpha=.45)
    axes[1, 2].set(xlabel="Width (m)", ylabel="Height (m)", title="Orange: near full-height openings")
    fig.suptitle(f"Exterior facade v{metrics['generator_version']} | {metrics['scenes']} consecutive scenes")
    fig.savefig(output / "distributions.svg")
    fig.savefig(output / "distributions.png", dpi=140)
    plt.close(fig)


def elevations(rooms, output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    count = len(rooms)
    fig, axes = plt.subplots(count, 3, figsize=(12, max(3, count * 1.5)), constrained_layout=True, squeeze=False)
    for row, (_, _, manifest) in enumerate(rooms):
        size = manifest["room_size"]
        for col, side in enumerate(("Left", "Rear", "Right")):
            ax = axes[row, col]
            width = size[0] if side == "Rear" else size[2]
            height = size[1]
            ax.add_patch(Rectangle((-width/2, 0), width, height, facecolor="#d5d0c7", edgecolor="#54545b"))
            facade = next((f for f in manifest["exterior"]["facades"] if f["side"] == side), None)
            if facade:
                for o in facade["openings"]:
                    x, y = o["min"]
                    w, h = [b-a for a,b in zip(o["min"], o["max"])]
                    ax.add_patch(Rectangle((x, y), w, h, facecolor="#86bac7", edgecolor="#344651", linewidth=1.5))
                    for c in range(1, o["columns"]):
                        ax.plot([x+w*c/o["columns"]]*2, [y, y+h], color="#344651", linewidth=.7)
                    if o["transom"]:
                        ax.plot([x, x+w], [y+h*o["transom"]]*2, color="#344651", linewidth=.7)
            ax.set(xlim=(-width/2-.2, width/2+.2), ylim=(-.1, height+.3), title=f"seed {manifest['seed']} | {side}")
            ax.set_aspect("equal")
            ax.axis("off")
    fig.suptitle("Stored exterior programs for every captured room (schematic elevations)")
    fig.savefig(output / "elevations.svg")
    fig.savefig(output / "elevations.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.captures, args.output)
    print(json.dumps({k: result[k] for k in ("audit_scenes", "openings", "exterior_wall_counts", "rooms_with_full_height_aperture")}, indent=2))
