#!/usr/bin/env python3
"""Report the bounded seating/activity review with unfiltered room captures.

Uses indoor_report's capture identity checks. Requires Pillow and matplotlib.
The studio gallery illustrates programs; only consecutive room seeds estimate
their sampled prevalence. No perceptual quality or learning-utility score is implied.
"""
import argparse
import hashlib
import json
from pathlib import Path

from indoor_report import summarize
from report_indoor_tabletop import contact_sheet


KINDS = ("Sofa", "Chair", "Bookcase", "Whiteboard", "Display", "WallOutlet", "LightSwitch")


def report(captures, gallery, output):
    metrics, rooms, render = summarize(captures)
    output.mkdir(parents=True, exist_ok=True)
    counts = {}
    for kind in KINDS:
        hist = metrics["object_counts_per_scene"][f"main/{kind}"]
        assert sum(hist.values()) == metrics["scenes"], "missing zero-count rooms"
        counts[kind] = {
            "instances": sum(int(n) * f for n, f in hist.items()),
            "rooms_with_object": sum(f for n, f in hist.items() if int(n) > 0),
            "count_histogram": hist,
        }
    gallery_objects = json.loads((gallery / "objects.json").read_text())
    assert len(gallery_objects) == 31, "expected the seating review gallery"
    gallery_rows = [(gallery / f"furniture_{i}.png", f"{o['kind']} v{o['variant']} seed{o['seed']}")
                    for i, o in enumerate(gallery_objects)]
    for group, start, end in [("sofas", 0, 4), ("chairs", 4, 14), ("bookshelves", 14, 18),
                              ("boards", 18, 25), ("hardware", 25, 31)]:
        contact_sheet(gallery_rows[start:end], output / f"{group}.png")
    room_rows = [(d / f"view_{i:02}_color.png", f"seed{m['seed']} view{i} | {m['layout']}")
                 for d, c, m in rooms for i in range(len(c["views"]))]
    contact_sheet(room_rows, output / "rooms.png", 320, 240)
    geometry = {p.name: json.loads(p.read_text()) for p in sorted(captures.glob("geometry_*.json"))}
    paths = [captures / name for name in ("metrics.json", "distribution.json", "render_selection.json", "run_complete.json")]
    paths += [gallery / "objects.json"] + [p for p, _ in gallery_rows + room_rows]
    paths += list(captures.glob("geometry_*.json"))
    paths += [d / name for d, _, _ in rooms for name in ("capture.json", "manifest.json")]
    summary = {
        "generator_version": metrics["generator_version"],
        "capture_engine": render["capture_engine"],
        "audit_scenes": metrics["scenes"],
        "primary_room_objects": counts,
        "parameters_all_rooms": {k: v for k, v in metrics["numeric"].items()
                                 if k.startswith(("activity_", "sofa_", "chair_", "bookshelf_"))},
        "categories_all_rooms": {k: v for k, v in metrics["categories"].items()
                                 if k.startswith(("chair_", "sofa_")) or k in ("tv_content", "whiteboard_content", "layout")},
        "render_check": render,
        "mesh_checks": geometry,
        "gallery_objects": gallery_objects,
        "artifact_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "limits": [
            "Primary-room instance counts include zero-count rooms; parameter distributions include neighboring rooms.",
            "Sofa-program parameters include lounge armchairs. Layout names are biases, not fixed floor plans.",
            "The studio gallery deliberately exercises distinct programs; room captures are consecutive unfiltered seeds.",
            "Bounds, signal and coverage checks do not establish photographic realism or training utility.",
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "layout_audit.json").write_bytes((captures / "distribution.json").read_bytes())
    plots(metrics, counts, output)
    return summary


def plots(metrics, counts, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "font.size": 9})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    axes[0, 0].barh(KINDS, [counts[k]["rooms_with_object"] / metrics["scenes"] for k in KINDS], color="#386b8e")
    axes[0, 0].set(xlabel="Fraction of primary rooms with object", xlim=(0, 1))
    for ax, key, label in zip(list(axes.flat)[1:4],
                             ("activity_weight_social", "sofa_cushion_m", "bookshelf_fill"),
                             ("Social group probability (per zone)", "Upholstery cushion thickness (m)", "Shelf filling prior")):
        d = metrics["numeric"][key]
        edges = d["bin_edges"]
        ax.bar(edges[:-1], d["bin_counts"], width=[b-a for a, b in zip(edges, edges[1:])], align="edge", color="#387c72")
        ax.set(xlabel=label, ylabel="Zones / objects (all rooms)")
    for ax, key, label in [(axes[1, 1], "chair_family", "Chair construction"),
                            (axes[1, 2], "sofa_configuration", "Upholstery configuration")]:
        data = metrics["categories"][key]
        ax.bar(list(data), list(data.values()), color="#976340")
        ax.set(xlabel=label, ylabel="Objects (all rooms)")
        ax.tick_params(axis="x", rotation=35)
    fig.suptitle(f"Seating / activity v{metrics['generator_version']} | {metrics['scenes']} consecutive scenes")
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
    print(json.dumps({"audit_scenes": result["audit_scenes"], "primary_room_objects": result["primary_room_objects"],
                      "rendered_views": result["render_check"]["rendered_views"]}, indent=2))
