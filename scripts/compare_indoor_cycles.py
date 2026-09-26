#!/usr/bin/env python3
"""Compare calibrated native linear captures with actual-scene Cycles references.

Requires NumPy and Pillow. No fitted exposure, registration, masked-out difficult
scenes, or display-space comparison is used for the quantitative measurements.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from indoor_reference_contract import sky_radiance, validate_snapshot


LUMA = np.array([.2126, .7152, .0722])
SEMANTICS = {
    "wall": (174, 199, 232), "floor": (152, 223, 138),
    "ceiling": (78, 71, 183), "chair": (188, 189, 34),
    "window": (197, 176, 213), "lamp": (96, 207, 209),
    "person": (120, 185, 128), "desk": (247, 182, 210),
}


def measurements(native, reference, mask):
    n, r = native[mask].astype(np.float64), reference[mask].astype(np.float64)
    if len(n) == 0:
        return None
    if not np.isfinite(n).all() or not np.isfinite(r).all():
        raise ValueError("Non-finite radiance")
    nl, rl = n @ LUMA, r @ LUMA
    floor = max(float(np.median(rl)) * .01, 1e-5)
    stops = np.log2(np.maximum(nl, floor) / np.maximum(rl, floor))
    return {
        "pixels": len(n), "native_mean_luminance": float(nl.mean()),
        "reference_mean_luminance": float(rl.mean()),
        "rgb_relative_mae": float(np.abs(n - r).sum() / max(np.abs(r).sum(), 1e-12)),
        "luminance_relative_mae": float(np.abs(nl - rl).sum() / max(np.abs(rl).sum(), 1e-12)),
        "luminance_bias_ratio": float(nl.mean() / max(rl.mean(), 1e-12)),
        "median_absolute_stops": float(np.median(np.abs(stops))),
        "p90_absolute_stops": float(np.quantile(np.abs(stops), .9)),
        "fraction_within_20_percent_luminance": float((np.abs(nl-rl) <= .2*np.maximum(rl, floor)).mean()),
        "log_floor": floor,
    }


def raw(path, size):
    a = np.fromfile(path, dtype="<f4")
    w, h = size
    if a.size != w * h * 4:
        raise ValueError(f"Wrong raw image size: {path}")
    return a.reshape(h, w, 4)[..., :3]


def preview(a):
    a = np.maximum(a, 0)
    a = a / (1 + a)  # One common, fixed display transform for both renderers.
    a = np.where(a <= .0031308, 12.92*a, 1.055*a**(1/2.4)-.055)
    return (np.clip(a, 0, 1)*255 + .5).astype(np.uint8)


def blocks(a, size=16):
    """Average complete image blocks; never smooth across an arbitrary crop."""
    h, w, channels = a.shape
    if h % size or w % size:
        raise ValueError(f"Image dimensions must be divisible by block size {size}")
    return a.reshape(h // size, size, w // size, size, channels).mean(axis=(1, 3))


def block_measurements(a, b):
    result = {}
    for size in (8, 16, 32):
        if a.shape[0] % size or a.shape[1] % size:
            continue
        aa, bb = blocks(a, size), blocks(b, size)
        result[str(size)] = measurements(aa, bb, np.ones(aa.shape[:2], dtype=bool))
    return result


def optical_identity(document, directory):
    """Hash resolved optical content, independent of ECS export table ordering."""
    encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    digest = lambda data: hashlib.sha256(data).hexdigest()
    geometry = (directory / "geometry.bin").read_bytes()
    manifest = json.loads((directory.parent / "manifest.json").read_text())
    validate_snapshot(document, manifest)
    if len(geometry) != document["geometry_bytes"]:
        raise ValueError("Invalid geometry length in optical source")
    meshes = []
    for mesh in document["meshes"]:
        offset = mesh["offset"]
        length = mesh["vertices"] * 32 + mesh["indices"] * 4
        if offset < 0 or length <= 0 or offset + length > len(geometry):
            raise ValueError("Invalid mesh range in optical source")
        meshes.append({"vertices": mesh["vertices"], "indices": mesh["indices"],
                       "sha256": digest(geometry[offset:offset + length])})
    materials = []
    for material in document["materials"]:
        resolved = dict(material)
        for name, item in material.items():
            if isinstance(item, dict) and "path" in item:
                resolved[name] = {key: value for key, value in item.items() if key != "path"}
                resolved[name]["sha256"] = digest((directory / item["path"]).read_bytes())
        materials.append(resolved)
    instances = []
    for instance in document["instances"]:
        resolved = dict(instance)
        resolved["mesh"] = meshes[instance["mesh"]]
        resolved["material"] = materials[instance["material"]]
        instances.append(resolved)
    canonical = {key: value for key, value in document.items()
                 if key not in ("capture_engine", "meshes", "materials", "instances", "lights")}
    canonical.update(meshes=sorted(meshes, key=encode), materials=sorted(materials, key=encode),
                     instances=sorted(instances, key=encode), lights=sorted(document["lights"], key=encode))
    canonical["external_sky_radiance"] = sky_radiance(manifest)
    return digest(encode(canonical).encode())


def compare(native_dir, cycles_dir, output, reference_repeat=None, reference_source=None):
    capture = json.loads((native_dir / "capture.json").read_text())
    scene = json.loads((native_dir / "reference/scene.json").read_text())
    report = json.loads((cycles_dir / "report.json").read_text())
    if "exposed scene-linear" not in capture["color_encoding"]:
        raise ValueError("Native capture must use --linear-rgb, with raw output enabled")
    if not report["geometry_alignment_passed"]:
        raise ValueError("Cycles geometry/camera alignment has not passed")
    if scene["image_size"] != report["image_size"] or scene["seed"] != report["seed"]:
        raise ValueError("Reference identity/size mismatch")
    repeat_report = None
    if reference_repeat:
        repeat_report = json.loads((reference_repeat / "report.json").read_text())
        for field in ("source_sha256", "image_size", "seed", "bounces", "blender_hash", "radiometry_sha256"):
            if repeat_report[field] != report[field]:
                raise ValueError(f"Independent reference mismatch: {field}")
        if report["denoised"] or repeat_report["denoised"]:
            raise ValueError("Reference convergence requires raw, independently sampled images")
        if not repeat_report["geometry_alignment_passed"]:
            raise ValueError("Independent reference geometry audit failed")
        if len(repeat_report["views"]) != len(report["views"]):
            raise ValueError("Independent reference view count mismatch")
    source = reference_source or native_dir / "reference"
    source_scene = json.loads((source / "scene.json").read_text())
    if "../manifest.json" not in report["source_sha256"]:
        raise ValueError("Reference report must pin the manifest that supplies sky lighting")
    for relative, expected in report["source_sha256"].items():
        path = source / relative
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Native/reference source identity mismatch: {relative}")
    optical_hash = optical_identity(source_scene, source)
    if optical_hash != optical_identity(scene, native_dir / "reference"):
        raise ValueError("Reference reuse requires identical geometry, materials, lights, exposure and cameras")
    # This exposure follows Bevy's physical Exposure::exposure(), without fitting.
    exposure = 2**(-scene["ev100"]) / 1.2
    output.mkdir(parents=True, exist_ok=False)
    views = []
    for view in report["views"]:
        i = view["index"]
        npath, rpath = native_dir / f"view_{i:02}_color.rgba32f", cycles_dir / f"view_{i:02}.rgba32f"
        n = raw(npath, scene["image_size"])
        r = raw(rpath, scene["image_size"]) * exposure
        semantic_path = native_dir / f"view_{i:02}_semantic.png"
        masks = {"all_pixels": np.ones(n.shape[:2], dtype=bool)}
        if semantic_path.exists():
            semantic = np.array(Image.open(semantic_path).convert("RGB"))
            masks.update({name: np.all(semantic == color, axis=2) for name, color in SEMANTICS.items()})
        stats = {name: measurements(n, r, mask) for name, mask in masks.items()}
        spatial = block_measurements(n, r)
        convergence = None
        if reference_repeat:
            other_view = next(v for v in repeat_report["views"] if v["index"] == i)
            if other_view["seed"] == view["seed"]:
                raise ValueError("Independent reference must use a different sampling seed")
            other_path = reference_repeat / f"view_{i:02}.rgba32f"
            other = raw(other_path, scene["image_size"]) * exposure
            repeat_blocks = block_measurements(r, other)
            all_pixels = measurements(r, other, masks["all_pixels"])
            convergence = {
                "independent_samples": repeat_report["samples"],
                "independent_image_sha256": hashlib.sha256(other_path.read_bytes()).hexdigest(),
                "all_pixels": all_pixels, "blocks": repeat_blocks,
                "coarse_convergence_passed": ("32" in repeat_blocks and
                    repeat_blocks["32"]["rgb_relative_mae"] < .05 and
                    abs(all_pixels["luminance_bias_ratio"] - 1) < .02),
                "scope": "32x32 block radiance agreement below 5% and whole-image mean luminance within 2%; does not establish per-pixel convergence",
            }
        Image.fromarray(np.concatenate([preview(n), preview(r)], axis=1)).save(output / f"view_{i:02}_native_cycles.png")
        views.append({"index": i, "metrics": stats, "blocks": spatial, "reference_convergence": convergence,
                      "native_sha256": hashlib.sha256(npath.read_bytes()).hexdigest(),
                      "cycles_sha256": hashlib.sha256(rpath.read_bytes()).hexdigest()})
    if not views:
        raise ValueError("No compared views")
    result = {"seed": scene["seed"], "capture_engine": scene["capture_engine"],
              "comparison_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "contract_script_sha256": hashlib.sha256(Path(__file__).with_name("indoor_reference_contract.py").read_bytes()).hexdigest(),
              "reference_source_capture_engine": source_scene["capture_engine"],
              "native_scene_sha256": hashlib.sha256((native_dir / "reference/scene.json").read_bytes()).hexdigest(),
              "optical_content_sha256": optical_hash,
              "reference_reuse": reference_source is not None,
              "reference_context_verification": report.get("source_context_verification", "recorded by renderer"),
              "ev100": scene["ev100"], "cycles_to_native_exposure": exposure,
              "scope": "Absolute linear radiance comparison; lighting/BSDF model differences recorded in Cycles report",
              "preview": "Native left, Cycles right; same fixed Reinhard then sRGB display transform",
              "cycles_denoised": report["denoised"], "cycles_samples": report["samples"], "views": views}
    (output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--cycles", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference-repeat", type=Path, help="Independent, raw Cycles run of the identical scene for convergence diagnostics")
    p.add_argument("--reference-source", type=Path, help="Original exported reference directory; reuse across native engine revisions requires exact equality of every optical input")
    a = p.parse_args()
    r = compare(a.native, a.cycles, a.output, a.reference_repeat, a.reference_source)
    print(json.dumps({"seed": r["seed"], "views": [v["metrics"]["all_pixels"] for v in r["views"]]}))


if __name__ == "__main__":
    main()
