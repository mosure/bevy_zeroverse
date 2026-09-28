#!/usr/bin/env python3
"""Summarize a bounded tabletop review and its unfiltered indoor captures.

Uses indoor_report's run/manifest checks. Requires Pillow and matplotlib for
contact sheets and distribution plots; no models or external image assets.
"""
import argparse
import hashlib
import json
from pathlib import Path

from indoor_report import summarize


KINDS = ("Mug", "CoffeeCup", "WaterBottle", "SodaCan", "Notepad", "Pencil",
         "Microphone", "Phone", "Laptop", "Monitor", "Clock")


def contact_sheet(rows, output, width=300, height=300):
    from PIL import Image, ImageDraw
    sheet = Image.new("RGB", (4 * width, ((len(rows) + 3) // 4) * (height + 24)), "#f4f4f4")
    draw = ImageDraw.Draw(sheet)
    for index, (path, label) in enumerate(rows):
        x, y = index % 4 * width, index // 4 * (height + 24)
        with Image.open(path) as image:
            image = image.convert("RGB")
            image.thumbnail((width, height), Image.Resampling.LANCZOS)
            sheet.paste(image, (x, y))
        draw.text((x + 6, y + height + 5), label, fill="black")
    sheet.save(output)


def report(captures, gallery, output):
    metrics, rooms, render = summarize(captures)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for kind in KINDS:
        hist = metrics["object_counts_per_scene"][f"main/{kind}"]
        assert sum(hist.values()) == metrics["scenes"], "missing zero-count rooms"
        counts[kind] = {
            "instances": sum(int(n) * frequency for n, frequency in hist.items()),
            "rooms_with_object": sum(frequency for n, frequency in hist.items() if int(n) > 0),
            "count_histogram": hist,
        }
    parameters = {key: value for key, value in metrics["numeric"].items()
                  if key.startswith(("laptop_", "display_", "clock_", "screen_",
                                     "vessel_", "drink_", "water_"))}
    categories = {key: value for key, value in metrics["categories"].items()
                  if key.endswith("_construction") or key in ("clock_display", "screen_content")}
    gallery_objects = json.loads((gallery / "objects.json").read_text())
    gallery_rows = [(gallery / f"furniture_{i}.png", f"{obj['kind']} seed {obj['seed']}")
                    for i, obj in enumerate(gallery_objects)]
    for name, start, end in [("beverages", 0, 11), ("tabletop", 11, 18), ("devices", 18, 28)]:
        contact_sheet(gallery_rows[start:end], output / f"{name}.png")
    room_rows = [(directory / f"view_{i:02}_color.png", f"seed {manifest['seed']} view {i} | {manifest['layout']}")
                 for directory, capture, manifest in rooms for i in range(len(capture["views"]))]
    contact_sheet(room_rows, output / "rooms.png", 320, 240)
    paths = [captures / name for name in ("metrics.json", "distribution.json", "render_selection.json", "run_complete.json")]
    paths += [gallery / "objects.json"] + [path for path, _ in gallery_rows + room_rows]
    paths += [directory / name for directory, _, _ in rooms for name in ("capture.json", "manifest.json")]
    summary = {
        "generator_version": metrics["generator_version"],
        "capture_engine": render["capture_engine"],
        "audit_scenes": metrics["scenes"],
        "primary_room_objects": counts,
        "parameters_all_rooms": parameters,
        "categories_all_rooms": categories,
        "render_check": render,
        "gallery_objects": gallery_objects,
        "artifact_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "limits": [
            "Count histograms include zeros and count primary-room instances only.",
            "Device and beverage parameter histograms include neighboring-room objects.",
            "Studio gallery illustrates particular programs; rooms are consecutive seeds without filtering.",
            "These checks establish bounded geometry and capture coverage, not photographic realism or 10M-sample utility.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "layout_audit.json").write_bytes((captures / "distribution.json").read_bytes())
    plot_distributions(metrics, counts, output)
    return summary


def plot_distributions(metrics, counts, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].barh(KINDS, [counts[k]["rooms_with_object"] / metrics["scenes"] for k in KINDS], color="#386b8e")
    axes[0, 0].set(xlabel="Fraction of primary rooms with object", xlim=(0, 1))
    for ax, key, label in zip(list(axes.flat)[1:],
                             ("clock_time_seconds", "screen_time_minutes", "laptop_lid_angle_radians", "display_aspect", "vessel_taper"),
                             ("Clock time (seconds after midnight)", "Screen time (minutes after midnight)", "Laptop hinge (radians)", "Display aspect ratio", "Cup/mug mouth-to-base radius")):
        d = metrics["numeric"][key]
        edges = d["bin_edges"]
        ax.bar(edges[:-1], d["bin_counts"], width=[b-a for a, b in zip(edges, edges[1:])], align="edge", color="#387c72")
        ax.set(xlabel=label, ylabel="Objects (all rooms)")
    fig.suptitle(f"Tabletop v{metrics['generator_version']} | {metrics['scenes']} consecutive scenes")
    fig.savefig(output / "distributions.svg")
    fig.savefig(output / "distributions.png", dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--gallery", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = report(args.captures, args.gallery, args.output)
    print(json.dumps({"audit_scenes": result["audit_scenes"],
                      "primary_room_objects": result["primary_room_objects"],
                      "annotations": result["render_check"]["annotation"]}, indent=2))
