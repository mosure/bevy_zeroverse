#!/usr/bin/env python3
"""Paper figures from the gallery's exact exported masks, without new rendering.

The tracked calibration/NPZ archives and RGB/annotation images are sufficient.
Optionally verify the original capture hashes and depth/position reprojection.
Requires NumPy and Pillow. Numeric colors are never resized or tone mapped here.
"""
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
MEDIA = ROOT / "www/project/static/media"
OUTPUT = ROOT / "tex/generated"
EVIDENCE = ROOT / "docs/evidence/paper_co_visibility_v21"
spec = importlib.util.spec_from_file_location(
    "co_visibility", ROOT / "crates/ffi/python/bevy_zeroverse_dataloader/co_visibility.py"
)
cv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def font(size, bold=False):
    return ImageFont.truetype("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size)


def load_scene(seed, inputs):
    archive_path = MEDIA / f"s{seed}-visibility.zip"
    calibration_path = MEDIA / f"s{seed}-calibration.json"
    calibration = json.loads(calibration_path.read_text())
    for p in [archive_path, calibration_path]:
        inputs[str(p.relative_to(ROOT))] = digest(p)
    config = calibration["config"]
    assert config["generator_version"] == 21 and config["base_seed"] == seed
    assert config["cameras"] == 4
    metadata = cv.validate_metadata(calibration["co_visibility"], 4)
    views = []
    with zipfile.ZipFile(archive_path) as archive:
        assert json.loads(archive.read("calibration.json")) == calibration
        for camera in range(4):
            name = f"co_visibility_000_{camera:02}.npz"
            raw = archive.read(name)
            recorded = f"out/project_page_v21/scene_{seed}/000000/{name}"
            assert hashlib.sha256(raw).hexdigest() == calibration["source_sha256"][recorded]
            with np.load(io.BytesIO(raw), allow_pickle=False) as data:
                mask = data["co_visibility"].copy()
                valid = data["co_visibility_valid"].copy()
            assert mask.shape == valid.shape == (config["height"], config["width"], 1)
            rgb_path = MEDIA / f"s{seed}-t0-c{camera}-color.webp"
            code_path = MEDIA / f"s{seed}-t0-c{camera}-co_visibility.png"
            for p in [rgb_path, code_path]:
                inputs[str(p.relative_to(ROOT))] = digest(p)
            with Image.open(rgb_path) as image:
                rgb = image.convert("RGB")
            with Image.open(code_path) as image:
                code = np.asarray(image.convert("RGB"))
            assert rgb.size == (config["width"], config["height"])
            assert np.array_equal(code, cv.mask_to_rgb(mask[..., 0], 4))
            assert np.array_equal(cv.rgb_to_mask(code, 4), mask[..., 0])
            pose = calibration["cameras"][0][camera]
            assert pose["index"] == camera and pose["time"] == 0.0
            views.append(dict(camera=camera, mask=mask[..., 0], valid=valid[..., 0],
                              rgb=rgb, code=Image.fromarray(code), pose=pose))
    cv.validate({
        "co_visibility": np.stack([v["mask"] for v in views])[None, ..., None],
        "co_visibility_valid": np.stack([v["valid"] for v in views])[None, ..., None],
        "co_visibility_metadata": metadata,
    })
    return calibration, views


def verify_capture(folder, calibration, views):
    """Check the raster's calibration without interpreting the RGB preview."""
    bounds = np.asarray(calibration["aabb"], dtype=np.float64)
    results = []
    for view in views:
        camera = view["camera"]
        planes = {}
        for mode in ["color", "co_visibility", "depth", "position"]:
            path = folder / f"{mode}_000_{camera:02}.npz"
            key = f"out/project_page_v21/{folder.parent.name}/000000/{path.name}"
            assert digest(path) == calibration["source_sha256"][key], path
            with np.load(path, allow_pickle=False) as data:
                planes[mode] = data[mode].copy()
        assert np.array_equal(planes["co_visibility"][..., 0], view["mask"])
        # Reproduce the gallery's fixed RGB conversion and WebP settings. This
        # verifies the RGB panel's scene/camera/time, not just its dimensions.
        encoded = io.BytesIO()
        rgb = np.rint(np.clip(planes["color"], 0, 1) * 255).astype(np.uint8)
        Image.fromarray(rgb).save(encoded, format="WEBP", quality=92, method=6)
        rgb_path = MEDIA / f"s{calibration['config']['base_seed']}-t0-c{camera}-color.webp"
        assert encoded.getvalue() == rgb_path.read_bytes(), "RGB preview differs from its recorded capture"
        valid = view["valid"].astype(bool)
        assert valid.any()
        world = planes["position"] * (bounds[1] - bounds[0]) + bounds[0]
        matrix = np.linalg.inv(np.asarray(view["pose"]["world_from_view_columns"]).T)
        xyz = world @ matrix[:3, :3].T + matrix[:3, 3]
        z = -xyz[..., 2]
        depth_error = float(np.max(np.abs(z[valid] - planes["depth"][valid])))
        h, w = valid.shape
        y, x = np.indices((h, w))
        f = 1 / np.tan(np.deg2rad(view["pose"]["fovy_degrees"]) / 2)
        safe_z = np.where(valid, z, 1)
        px = (xyz[..., 0] / safe_z * f / (w / h) + 1) * w / 2
        py = (1 - xyz[..., 1] / safe_z * f) * h / 2
        error = float(np.max(np.hypot(px - x - .5, py - y - .5)[valid]))
        assert depth_error < 0.0002 and error < 0.02, (depth_error, error)
        results.append(dict(camera=camera, rgb_preview_reproduced_exactly=True, depth_position_max_m=depth_error,
                            reprojection_max_pixels=error))
    return results


def statistics(view):
    mask, valid = view["mask"], view["valid"].astype(bool)
    count = sum(((mask >> i) & 1).astype(np.uint8) for i in range(4))
    valid_pixels = int(valid.sum())
    bins = np.bincount(count[valid], minlength=4).tolist()
    assert sum(bins) == valid_pixels and len(bins) == 4
    return dict(camera=view["camera"], valid_pixels=valid_pixels,
                invalid_pixels=int((~valid).sum()), peer_count_pixels=bins,
                peer_pixels=[int(np.count_nonzero(valid & cv.camera_visible(mask, i))) for i in range(4)],
                shared_any_fraction=float(np.count_nonzero(valid & (mask != 0)) / valid_pixels))


def multi_view_figure(views, path):
    w, h = views[0]["rgb"].size
    pad, gap, header, label = 30, 30, 208, 94
    image = Image.new("RGB", (w * 4 + pad * 2 + gap * 3, header + 2 * (h + label + gap) + 154), "white")
    draw = ImageDraw.Draw(image)
    draw.text((pad, 4), "Same room · same timestep · four capture cameras", font=font(60, True), fill="#1f3440")
    draw.text((pad, 98), "Camera contribution (RGB8):", font=font(43), fill="#1f3440")
    for i, color in enumerate(cv.palette(4)):
        x = 780 + i * 610
        draw.rectangle((x, 105, x + 60, 155), fill=tuple(int(c) for c in color), outline="#444444")
        draw.text((x + 80, 98), f"C{i}  {tuple(color.tolist())}", font=font(40), fill="#1f3440")
    for column, view in enumerate(views):
        x = pad + column * (w + gap)
        c = view["camera"]
        stats = statistics(view)
        draw.text((x, header), f"Camera {c} · RGB", font=font(56, True), fill="#1f3440")
        image.paste(view["rgb"], (x, header + label))
        y = header + h + label + gap
        draw.text((x, y), f"Mask · {100*stats['shared_any_fraction']:.1f}% shared", font=font(51, True), fill="#1f3440")
        image.paste(view["code"], (x, y + label))
    y = image.height - 148
    draw.text((pad, y), "Colors add camera membership; they are not semantic classes or a heatmap.", font=font(47), fill="#1f3440")
    draw.text((pad, y + 72), "Black = no peer bits. Validity is stored separately. Each source camera excludes its own bit.", font=font(44), fill="#1f3440")
    image.save(path, optimize=True)


def decomposition_figure(view, path):
    w, h = view["rgb"].size
    pad, gap, label, header = 28, 28, 94, 110
    image = Image.new("RGB", (3 * w + 2 * gap + 2 * pad, header + 2 * (h + label + gap) + 140), "white")
    draw = ImageDraw.Draw(image)
    draw.text((pad, 8), "One source pixel, three independent peer-membership bits", font=font(43, True), fill="#1f3440")
    draw.text((pad, 66), "Seed 24000 · camera 0 · t = 0 · every panel uses the same source-pixel coordinates", font=font(29), fill="#1f3440")
    mask, valid = view["mask"], view["valid"].astype(bool)
    count = sum(((mask >> i) & 1).astype(np.uint8) for i in range(4))
    # Cardinality is a derived categorical preview, separate from the exact bit-color code.
    colors = np.array([[0, 0, 0], [90, 123, 153], [38, 163, 134], [238, 184, 60]], dtype=np.uint8)
    cardinality = colors[count]
    cardinality[~valid] = 160
    tiles = [(view["rgb"], "Source camera 0 · RGB", "Rendered color"),
             (view["code"], "Additive membership color", "M = 2·V1 + 4·V2 + 8·V3"),
             (Image.fromarray(cardinality), "Number of visible peers", "popcount(M), from 0 to 3")]
    stats = statistics(view)
    for peer in [1, 2, 3]:
        binary = np.repeat(np.where(cv.camera_visible(mask, peer), 255, 0)[..., None], 3, axis=2).astype(np.uint8)
        binary[~valid] = 160
        tiles.append((Image.fromarray(binary), f"Visible in camera {peer}",
                      f"V{peer} = (M & {1 << peer}) != 0  ·  {100*stats['peer_pixels'][peer]/stats['valid_pixels']:.1f}% of valid pixels"))
    for i, (tile, title, subtitle) in enumerate(tiles):
        x = pad + (i % 3) * (w + gap)
        y = header + (i // 3) * (h + label + gap)
        draw.text((x, y), title, font=font(37, True), fill="#1f3440")
        draw.text((x, y + 51), subtitle, font=font(26), fill="#1f3440")
        image.paste(tile, (x, y + label))
    y = image.height - 134
    draw.text((pad, y), "Peer masks: white = visible in that camera; black = not visible; gray = invalid source.", font=font(29), fill="#1f3440")
    draw.text((pad, y + 49), "Peer count:", font=font(29, True), fill="#1f3440")
    for i, color in enumerate(colors):
        x = 260 + i * 220
        draw.rectangle((x, y + 52, x + 40, y + 82), fill=tuple(color.tolist()), outline="#444444")
        draw.text((x + 55, y + 49), str(i), font=font(29), fill="#1f3440")
    draw.text((1160, y + 49), "Example: M = 10 = 0b1010 → peers C1 and C3", font=font(29), fill="#1f3440")
    image.save(path, optimize=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-captures", type=Path, help="Optional original scene_SEED/000000 capture directories")
    args = parser.parse_args()
    OUTPUT.mkdir(exist_ok=True, parents=True)
    EVIDENCE.mkdir(exist_ok=True, parents=True)
    inputs, records = {}, []
    for seed in [24005, 24000]:
        calibration, views = load_scene(seed, inputs)
        record = dict(seed=seed, generator_version=21, capture_engine=calibration["config"]["capture_engine"],
                      time=0, image_size=list(views[0]["rgb"].size), views=[statistics(v) for v in views])
        if args.verify_captures:
            record["original_capture_alignment"] = verify_capture(args.verify_captures / f"scene_{seed}" / "000000", calibration, views)
        records.append(record)
        if seed == 24005:
            multi_view_figure(views, OUTPUT / "co_visibility_multiview_v21.png")
        else:
            decomposition_figure(views[0], OUTPUT / "co_visibility_bits_v21.png")
    figures = ["co_visibility_multiview_v21.png", "co_visibility_bits_v21.png"]
    report = dict(generator_version=21, scenes=records, inputs_sha256=inputs,
                  figures_sha256={p: digest(OUTPUT / p) for p in figures},
                  checks=["Archived mask hashes match captured calibration records.",
                          "All eight t=0 PNG masks encode and decode exactly to the uint16 masks.",
                          "Shared camera legend, synchronized identities, source-bit exclusion and separate validity checked.",
                          "Cardinality and peer fractions count valid source pixels, including unshared surfaces."],
                  limits=["Selected gallery examples, not additional population measurements.",
                          "Figure images are publication previews; use NPZ plus validity for supervision.",
                          "First geometric surface including annotation-opaque glass; no reflected/refracted visibility."])
    (EVIDENCE / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(dict(checked_views=8, figures=figures, original_capture_alignment=bool(args.verify_captures)), indent=2))


if __name__ == "__main__":
    main()
