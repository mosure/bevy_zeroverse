#!/usr/bin/env python3
"""Build the static project gallery from real CLI captures and recorded audits.

Requires numpy, Pillow, matplotlib, safetensors, and ffmpeg. See www/README.md
for capture commands. No model inference or image synthesis happens here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb, LinearSegmentedColormap
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps
from safetensors.numpy import load_file

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "www/project/static/media"
EVIDENCE = ROOT / "docs/evidence/generation_v18"
MODES = ["color", "depth", "normal", "position", "semantic", "optical_flow", "motion_vectors", "co_visibility"]


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def plane(folder, mode, t, c):
    with np.load(folder / f"{mode}_{t:03}_{c:02}.npz") as values:
        result = values[mode]
        assert np.isfinite(result).all(), (mode, t, c)
        return result


def save_image(path, array):
    im = Image.fromarray(np.rint(np.clip(array, 0, 1) * 255).astype(np.uint8))
    if path.suffix == ".webp":
        im.save(path, quality=92, method=6)
    else:
        im.save(path, optimize=True)


def flow_rgb(values, scale):
    dx, dy = values[..., 0], values[..., 1]
    hue = (np.arctan2(dy, dx) / (2 * np.pi)) % 1
    saturation = np.minimum(np.hypot(dx, dy) / scale, 1)
    result = hsv_to_rgb(np.stack([hue, saturation, np.ones_like(hue)], -1))
    result[values[..., 2] < 0.5] = 0
    return result


def camera_record(meta, t, c):
    return dict(index=c, time=float(meta["time"][t, c, 0]),
                fovy_degrees=float(np.degrees(meta["fovy"][t, c, 0])),
                near=float(meta["near"][t, c, 0]), far=float(meta["far"][t, c, 0]),
                world_from_view_columns=meta["world_from_view"][t, c].tolist())


def scene_assets(source, seed, title):
    folder = source / f"scene_{seed}" / "000000"
    config = read(folder.parent / "generation_config.json")
    assert config["generator_version"] == 20
    meta = load_file(folder / "meta.safetensors")
    h, w = config["height"], config["width"]
    steps, cameras = config["playback_steps"], config["cameras"]
    legend = read(folder / "co_visibility_metadata.json")
    depths = [plane(folder, "depth", t, c) for t in range(steps) for c in range(cameras)]
    positive = np.concatenate([d[d > 0] for d in depths])
    depth_max = float(np.ceil(np.percentile(positive, 99.5)))
    errors, source_hashes, frames = [], {}, []
    for t in range(steps):
        views = []
        for c in range(cameras):
            values = {m: plane(folder, m, t, c) for m in MODES}
            assert all(a.shape[:2] == (h, w) for a in values.values())
            with np.load(folder / f"co_visibility_{t:03}_{c:02}.npz") as cov:
                valid = cov["co_visibility_valid"][..., 0].astype(bool)
            mask = values["co_visibility"][..., 0]
            assert mask.dtype == np.uint16 and (mask < 2**cameras).all()
            assert not np.any(mask & (1 << c)), "source camera must not share with itself"
            assert np.allclose(values["motion_vectors"][..., :2] * [w, h],
                               values["optical_flow"][..., :2], atol=2e-5)
            assert np.array_equal(values["motion_vectors"][..., 2:], values["optical_flow"][..., 2:])
            # Check the actual captured calibration, not just matching filenames.
            world = values["position"] * (meta["aabb"][1] - meta["aabb"][0]) + meta["aabb"][0]
            view = np.linalg.inv(meta["world_from_view"][t, c].T)
            xyz = world @ view[:3, :3].T + view[:3, 3]
            depth_error = float(np.max(np.abs(-xyz[..., 2][valid] - values["depth"][valid])))
            assert depth_error < 0.0002, depth_error
            f = 1 / np.tan(float(meta["fovy"][t, c, 0]) / 2)
            y, x = np.indices((h, w))
            safe_z = np.where(valid, -xyz[..., 2], 1)
            px = (xyz[..., 0] / safe_z * f / (w / h) + 1) * w / 2
            py = (1 - xyz[..., 1] / safe_z * f) * h / 2
            reprojection = float(np.max(np.hypot(px - x - 0.5, py - y - 0.5)[valid]))
            assert reprojection < 0.02, reprojection
            errors.append(dict(t=t, camera=c, depth_error_m=depth_error, reprojection_pixels=reprojection))
            output = {}
            for mode, array in values.items():
                suffix = "webp" if mode == "color" else "png"
                filename = f"s{seed}-t{t}-c{c}-{mode}.{suffix}"
                if mode == "depth":
                    rgb = matplotlib.colormaps["viridis"](np.clip(array / depth_max, 0, 1))[..., :3]
                    rgb[~valid] = 0
                elif mode in ("normal", "position"):
                    rgb = array.copy()
                    rgb[~valid] = 0
                elif mode == "optical_flow":
                    rgb = flow_rgb(array, 32)
                elif mode == "motion_vectors":
                    rgb = flow_rgb(array, 0.05)
                elif mode in ("semantic", "co_visibility"):
                    shutil.copyfile(folder / f"{mode}_{t:03}_{c:02}.png", MEDIA / filename)
                    rgb = None
                else:
                    rgb = array
                if rgb is not None:
                    save_image(MEDIA / filename, rgb)
                output[mode] = filename
                original = folder / f"{mode}_{t:03}_{c:02}.npz"
                source_hashes[str(original.relative_to(ROOT))] = digest(original)
            # One peer's visibility over RGB: exact membership, display-only colors.
            peers = []
            for peer in range(cameras):
                shared = (mask & (1 << peer)) != 0
                overlay = values["color"] * 0.20
                overlay[shared] = values["color"][shared] * 0.65 + np.array([0.12, 0.92, 0.72]) * 0.35
                name = f"s{seed}-t{t}-c{c}-peer{peer}.webp"
                save_image(MEDIA / name, overlay)
                peers.append(name)
            output["peers"] = peers
            views.append(dict(camera=camera_record(meta, t, c), images=output,
                              shared_fraction=[float(np.mean((mask & (1 << i)) != 0)) for i in range(cameras)]))
        frames.append(dict(time=float(meta["time"][t, 0, 0]), views=views))
    calibration = dict(config=config, aabb=meta["aabb"].tolist(),
                       cameras=[[camera_record(meta, t, c) for c in range(cameras)] for t in range(steps)],
                       co_visibility=legend, source_sha256=source_hashes, alignment_checks=errors)
    write(MEDIA / f"s{seed}-calibration.json", calibration)
    # A compact, exact four-camera visibility example; PNG previews are not used as data.
    with zipfile.ZipFile(MEDIA / f"s{seed}-visibility.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        archive.write(MEDIA / f"s{seed}-calibration.json", "calibration.json")
        for c in range(cameras):
            name = f"co_visibility_000_{c:02}.npz"
            archive.write(folder / name, name)
    return dict(seed=seed, title=title, generator_version=config["generator_version"], width=w, height=h,
                depth_max=depth_max, frames=frames, legend=legend["legend"],
                calibration=f"s{seed}-calibration.json", masks=f"s{seed}-visibility.zip")


def save_svg(figure, path):
    figure.savefig(path)
    # Matplotlib path continuations include trailing spaces; keep generated
    # text assets clean in diffs without changing their SVG geometry.
    path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")


def figures():
    metrics, report = read(EVIDENCE / "metrics.json"), read(EVIDENCE / "report.json")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.edgecolor": "#c4cbc6", "axes.labelcolor": "#58655f",
                         "text.color": "#253b32", "xtick.color": "#58655f", "ytick.color": "#58655f",
                         "svg.fonttype": "none", "savefig.facecolor": "#ffffff"})
    fig, axes = plt.subplots(2, 3, figsize=(13.0, 6.8), constrained_layout=True)
    charts = [("room_area_m2", "Room footprint", "m²", None),
              (None, "Chairs per interior", "chairs · zeros included", "main/Chair"),
              (None, "People per interior", "people · zeros included", "main/Person"),
              ("vertical_fov_degrees", "Vertical field of view", "degrees", None),
              ("sun_illuminance_lux", "Solar illumination", "lux", None),
              ("camera_reference_baseline_m", "Reference-camera baseline", "metres", None)]
    for ax, (key, title, unit, category) in zip(axes.flat, charts):
        if category:
            counts = metrics["object_counts_per_scene"][category]
            centers = np.array(sorted(map(int, counts)))
            heights = np.array([counts[str(i)] for i in centers]); widths = np.full(len(centers), 0.8)
        else:
            distribution = metrics["numeric"][key]
            edges = np.array(distribution["bin_edges"])
            centers = (edges[:-1] + edges[1:]) / 2
            widths = np.diff(edges) * 0.88
            heights = np.array(distribution["bin_counts"])
        panel, single = plt.subplots(figsize=(4.3, 3.2), constrained_layout=True)
        for target in [ax, single]:
            target.bar(centers, heights, width=widths, color="#247c68", linewidth=0)
            target.set_title(f"{title}  ·  n={sum(heights):,}", loc="left", fontsize=11, pad=14)
            target.set_xlabel(unit); target.set_ylabel("count"); target.grid(axis="y", alpha=0.16); target.set_axisbelow(True)
            target.tick_params(length=0)
        name = key or category.split("/")[-1].lower()
        save_svg(panel, MEDIA / f"distribution-{name}.svg"); plt.close(panel)
    save_svg(fig, MEDIA / "distributions.svg"); plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), constrained_layout=True)
    colors = LinearSegmentedColormap.from_list("room", ["#f5f6ef", "#90c5a5", "#247c68", "#173c33"])
    for ax, key, title in zip(axes, ["main/Chair", "main/Person", "camera_path"], ["Chair centers", "Person centers", "Camera path samples"]):
        raw = np.array(metrics["placement_heatmaps"][key]).reshape(24, 24)
        values = raw / raw.sum() * 100
        panel, single = plt.subplots(figsize=(4.3, 3.8), constrained_layout=True)
        for parent, target in [(fig, ax), (panel, single)]:
            image = target.imshow(values, extent=(0, 1, 1, 0), cmap=colors, interpolation="nearest")
            target.set_title(f"{title}  ·  n={raw.sum():,}", loc="left", fontsize=11, pad=13)
            target.set_xlabel("normalized room X"); target.set_ylabel("normalized room Z")
            parent.colorbar(image, ax=target, shrink=0.8, label="% of samples in cell")
        save_svg(panel, MEDIA / f"placement-{key.split('/')[-1].lower()}.svg"); plt.close(panel)
    save_svg(fig, MEDIA / "placement.svg"); plt.close(fig)
    shutil.copyfile(EVIDENCE / "report.json", MEDIA / "audit-v18.json")
    shutil.copyfile(EVIDENCE / "metrics.json", MEDIA / "population-v18.json")
    # Consecutive, unfiltered examples from the exact population in the whitepaper.
    identities = read(EVIDENCE / "capture_sha256.json")
    for seed in range(24000, 24008):
        path = ROOT / f"out/paper_v18/cohort/seed_{seed:06}/view_00_color.png"
        assert digest(path) == identities[f"seed_{seed:06}/view_00_color.png"]
        Image.open(path).save(MEDIA / f"cohort-{seed}.webp", quality=92, method=6)
    return dict(generator_version=18, rooms=metrics["scenes"],
                seeds=[24000, 25023], rendered_rooms=report["embedding"]["scenes"],
                rendered_images=report["embedding"]["images"],
                metrics_sha256=digest(EVIDENCE / "metrics.json"), report_sha256=digest(EVIDENCE / "report.json"))


def video(source, name, views, fps=20):
    folder = source / name / "000000"
    config = read(folder.parent / "generation_config.json")
    assert config["generator_version"] == 20 and config["playback_steps"] == 120
    frames = source / f"{name}_frames"
    frames.mkdir(exist_ok=True)
    width, height = config["width"], config["height"]
    cols, rows = (2, 2) if len(views) == 4 else (2, 1)
    font = ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans"), 16)
    hashes = {}
    for t in range(config["playback_steps"]):
        canvas = Image.new("RGB", (width * cols, (height + 32) * rows), "#182c24")
        draw = ImageDraw.Draw(canvas)
        for i, c in enumerate(views):
            path = folder / f"color_{t:03}_{c:02}.npz"
            rgb = plane(folder, "color", t, c)
            im = Image.fromarray(np.rint(rgb.clip(0, 1) * 255).astype(np.uint8))
            x, y = (i % cols) * width, (i // cols) * (height + 32)
            canvas.paste(im, (x, y))
            draw.text((x + 12, y + height + 6), f"CAMERA {c}   ·   t = {t / 119:.3f}", fill="#e8eee8", font=font)
            hashes[str(path.relative_to(ROOT))] = digest(path)
        canvas.save(frames / f"{t:03}.png")
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", str(frames / "%03d.png"),
                    "-c:v", "libx264", "-preset", "slow", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    str(MEDIA / f"{name}.mp4")], check=True)
    Image.open(frames / "000.png").save(MEDIA / f"{name}.webp", quality=92, method=6)
    if len(views) == 4:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(MEDIA / f"{name}.mp4"),
                        "-filter_complex", "fps=8,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse",
                        "-loop", "0", str(MEDIA / "multiview.gif")], check=True)
    metadata = read(folder / "indoor_render_metadata.json")
    return dict(config=config, source_sha256=hashes, human_motion=metadata.get("human_motion"), output=f"{name}.mp4", frames=120, fps=fps,
                duration_seconds=6, note="Synchronized source frames, no synthetic interpolation; normalized capture t=0..1.")


def paper_figure(scene):
    views = scene["frames"][0]["views"]
    modes = ["color", "depth", "normal", "semantic"]
    canvas = Image.new("RGB", (1536, 520), "white")
    draw = ImageDraw.Draw(canvas)
    for row, c in enumerate([0, 2]):
        for col, mode in enumerate(modes):
            im = Image.open(MEDIA / views[c]["images"][mode]).resize((384, 240), Image.Resampling.LANCZOS)
            canvas.paste(im, (col * 384, row * 260))
            draw.text((col * 384 + 8, row * 260 + 243), f"Camera {c} / {mode}", fill="black")
    canvas.save(ROOT / "tex/generated/gallery_v20.jpg", quality=93)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captures", type=Path, default=ROOT / "out/project_page_v20")
    parser.add_argument("--skip-video", action="store_true")
    parser.add_argument("--keep-motion-video", action="store_true",
                        help="Retain the separately labeled v19 motion illustration and its provenance.")
    args = parser.parse_args()
    args.captures = args.captures.resolve()
    MEDIA.mkdir(parents=True, exist_ok=True)
    scenes = [scene_assets(args.captures, 24005, "Sunlit seating"), scene_assets(args.captures, 24000, "Workspace & bookshelves")]
    palette_source = (ROOT / "src/render/semantic.rs").read_text()
    palette = [{"label": m[0], "rgb8": list(map(int, m[1:]))}
               for m in re.findall(r"SemanticLabel::(\w+) => Color::srgb_u8\((\d+), (\d+), (\d+)\)", palette_source)]
    assert len(palette) == 40
    data = dict(schema_version=1, scenes=scenes, semantic_palette=palette, audit=figures())
    if not args.skip_video:
        data["videos"] = [video(args.captures, "traversal_24005", [0, 1, 2, 3])]
        if args.keep_motion_video:
            previous = read(MEDIA / "gallery.json")
            motion = next(v for v in previous["videos"] if v["output"] == "motion_13.mp4")
            assert (MEDIA / motion["output"]).is_file()
            data["videos"].append(motion)
        else:
            data["videos"].append(video(args.captures, "motion_13", [0, 1]))
    elif (MEDIA / "gallery.json").exists():
        data["videos"] = read(MEDIA / "gallery.json").get("videos", [])
    for entry in data.get("videos", []):
        metadata = args.captures / Path(entry["output"]).stem / "000000/indoor_render_metadata.json"
        if metadata.exists():
            entry["human_motion"] = read(metadata).get("human_motion")
    # Social preview is a labeled crop of the real room capture, never a mockup.
    social = Image.new("RGB", (1200, 630), "#20382e")
    source = Image.open(MEDIA / scenes[0]["frames"][0]["views"][0]["images"]["color"])
    social.paste(ImageOps.fit(source, (1200, 500)), (0, 130))
    font = ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans"), 38)
    draw = ImageDraw.Draw(social)
    draw.text((35, 18), "bevy_zeroverse", fill="white", font=font)
    draw.text((35, 76), "Procedural rooms. Shared geometry.", fill="#cce0ce",
              font=ImageFont.truetype(matplotlib.font_manager.findfont("DejaVu Sans"), 23))
    social.save(MEDIA / "social.jpg", quality=90)
    write(MEDIA / "gallery.json", data)
    paper_figure(scenes[0])
    print(f"Wrote {len(scenes)} scenes, matched annotations, calibration, charts and media to {MEDIA}")


if __name__ == "__main__":
    main()
