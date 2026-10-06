#!/usr/bin/env python3
"""Archive local release qualification after every required check completes.

Run manually, before replacing the inspected local core package. This helper
does not build, test, publish, contact the network, or change existing evidence.
"""

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import struct
import tarfile
import tempfile
import tomllib


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "out/release_031"
DESTINATION = ROOT / "docs/evidence/release_031"
DEFAULT_ARCHIVE = Path(
    "/media/mosure/hyper2/build/bevy_zeroverse-0.28.1/package/"
    "bevy_zeroverse-0.31.0.crate"
)
SMALL_FILE_LIMIT = 2 * 1024 * 1024
FINAL_CHECKS = {
    "final-fmt", "final-workspace-clippy", "final-broad-scenes",
    "final-workspace-tests", "final-wasm-motion", "final-native-gpu",
    "final-recovered-render-881", "final-recovered-render-158", "final-recovered-render-3779", "final-recovered-render-670", "final-recovered-render-3445", "final-recovered-render-682",
    "final-core-package",
}
REGISTRY_CHECKS = {
    "registry-lock", "registry-core-tests", "registry-gpu",
    "registry-burn-temporal", "registry-clippy",
}
PACKAGE_DIRECTORIES = ("src", "tests", "benches", "examples")
TEST_SUMMARY = re.compile(
    r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"(\d+) measured; (\d+) filtered out; finished in ([\d.]+)s"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def read_json(path):
    return json.loads(path.read_text())


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def capture_source_identity():
    """The same explicit input set/CRLF-normalized digest as the frozen source helper."""
    files = {}
    for name in ["Cargo.toml", "Cargo.lock", "build.rs", "crates/capture/Cargo.toml",
                 "third_party/wgpu-core/Cargo.toml", "third_party/wgpu-hal/Cargo.toml"]:
        path = ROOT / name
        if path.is_file():
            files[name] = path
    for name in ["src", "crates/capture/src", "third_party/wgpu-core/src",
                 "third_party/wgpu-hal/src", "assets/shaders", "assets/embedded"]:
        directory = ROOT / name
        if directory.exists():
            for path in directory.rglob("*"):
                require(not path.is_symlink(), f"Symlink in capture source inputs: {path}")
                if path.is_file():
                    files[str(path.relative_to(ROOT))] = path
    text = {"rs", "wgsl", "glsl", "metal", "hlsl", "vert", "frag", "toml", "lock",
            "json", "html", "css", "tex", "sty", "bib", "md", "svg", "txt"}
    digest = hashlib.sha256()
    for name, path in sorted(files.items()):
        data = path.read_bytes()
        if path.suffix.lstrip(".") in text or path.name == "LICENSE":
            try:
                data = data.decode("utf8").replace("\r\n", "\n").encode()
            except UnicodeDecodeError:
                pass
        encoded = name.encode()
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(hashlib.sha256(data).hexdigest().encode())
    return digest.hexdigest(), len(files)


def package_source_signatures(directory):
    files = {name: directory / name for name in ("Cargo.toml", "build.rs")
             if (directory / name).is_file()}
    for folder in PACKAGE_DIRECTORIES:
        for path in (directory / folder).rglob("*"):
            require(not path.is_symlink(), f"Symlink in package qualification inputs: {path}")
            if path.is_file():
                files[str(path.relative_to(directory))] = path
    return {name: sha(path) for name, path in sorted(files.items())}


def external_model_fixture(workspace=None):
    files = {}
    for name in ["assets/burn_human/fullbody_default.safetensors",
                 "assets/burn_human/fullbody_default.meta.json"]:
        path = ROOT / name
        require(path.is_file(), f"External offline model fixture is missing: {name}")
        files[name] = {"sha256": sha(path), "bytes": path.stat().st_size}
    fixture = {
        "environment_overrides": {"BEVY_ASSET_ROOT": str(ROOT)},
        "files": files,
        "scope": "Existing checkout fullbody model fixture supplied externally for offline model/render assertions. No fixture is copied into or packaged with the normalized core archive; no dependency override is introduced.",
    }
    if workspace is not None:
        target = (ROOT / "assets").resolve(strict=True)
        aliases = {}
        for path in [workspace / "assets", workspace / "core/assets"]:
            require(path.is_symlink() and path.resolve(strict=True) == target,
                    f"External asset fixture alias is missing or retargeted: {path}")
            resolved_files = {}
            for name in files:
                relative = Path(name).relative_to("assets")
                resolved = (path / relative).resolve(strict=True)
                require(resolved == (ROOT / name).resolve(strict=True),
                        f"External model fixture alias resolves a different file: {name}")
                resolved_files[name] = str(resolved)
            aliases[str(path.relative_to(ROOT))] = {
                "symlink_target": str(path.readlink()), "canonical_asset_directory": str(target),
                "resolved_fixture_files": resolved_files,
            }
        fixture["asset_directory_aliases"] = aliases
        fixture["scope"] += " Qualification-only asset symlinks cover tests that replace the inherited asset root; both link targets and effective model paths are verified before and after every gate. This does not claim every test consumes the model fixture."
    return fixture


def qualification_identity():
    """Bind every command to the freeze and complete compilation/test inputs."""
    frozen_path = OUT / "final-source.json"
    frozen = read_json(frozen_path)
    require(capture_source_identity() == (frozen["release_source_sha256"],
                                         frozen["release_source_inputs_count"]),
            "Capture inputs differ from final-source.json")
    live_rust = {str(path.relative_to(ROOT)): sha(path)
                 for folder in ["src", "tests", "crates/burn/src"]
                 for path in (ROOT / folder).rglob("*.rs")}
    require(live_rust == frozen["rust_inputs_sha256"],
            "Rust qualification inputs differ from final-source.json")
    for name, expected in frozen["burn"]["source_inputs"].items():
        data = (ROOT / "crates/burn" / name).read_bytes()
        if Path(name).suffix in {".rs", ".toml", ".json", ".wgsl"}:
            data = data.decode("utf8").replace("\r\n", "\n").encode()
        require(hashlib.sha256(data).hexdigest() == expected,
                f"Burn producer input differs from final-source.json: {name}")
    packages = {"core": package_source_signatures(ROOT)}
    for name in ["burn", "ffi", "capture", "publication", "burn_siglip2"]:
        packages[name] = package_source_signatures(ROOT / "crates" / name)
    for name, version in [("burn", "0.14.0"), ("ffi", "0.31.0")]:
        manifest = tomllib.loads((ROOT / "crates" / name / "Cargo.toml").read_text())
        require(manifest["package"]["version"] == version
                and manifest["dependencies"]["bevy_zeroverse"]["version"] == "0.31.0",
                f"Unexpected wrapper/core version in {name}")
    configuration = [".cargo/config.toml", "rust-toolchain", "rust-toolchain.toml",
                     "out/substrate_quality/cargo.sh"]
    return {
        "final_source_json_sha256": sha(frozen_path),
        "release_source_sha256": frozen["release_source_sha256"],
        "package_inputs_sha256": packages,
        "build_configuration_sha256": {name: sha(ROOT / name) for name in configuration
                                       if (ROOT / name).is_file()},
    }


def expected_commands(required):
    if required == REGISTRY_CHECKS:
        return [
            ("registry-lock", ["metadata", "--offline", "--format-version", "1"]),
            ("registry-core-tests", ["test", "--offline", "--locked", "-p", "bevy_zeroverse", "--lib", "--features", "human_motion", "--", "--test-threads=4"]),
            ("registry-gpu", ["test", "--offline", "--locked", "-p", "bevy_zeroverse", "--test", "procedural_indoor_render", "--test", "co_visibility_render", "--test", "ground_truth_render", "--test", "optical_flow_render", "--", "--ignored", "--test-threads=1"]),
            ("registry-burn-temporal", ["test", "--offline", "--locked", "-p", "bevy_zeroverse_burn", "--test", "dataset_tests", "--features", "human_motion", "headless_fs_render_mode_resets_between_timesteps", "--", "--test-threads=1"]),
            ("registry-clippy", ["clippy", "--offline", "--locked", "--workspace", "--all-targets", "--features", "human_motion", "--", "-D", "warnings"]),
        ]
    require(required == FINAL_CHECKS, "Unknown qualification command set")
    commands = [
        ("final-fmt", ["fmt", "--all", "--check"]),
        ("final-workspace-clippy", ["clippy", "--offline", "--locked", "--workspace", "--all-targets", "--features", "human_motion", "--", "-D", "warnings"]),
        ("final-workspace-tests", ["test", "--offline", "--locked", "--workspace", "--features", "human_motion", "--", "--test-threads=4"]),
        ("final-wasm-motion", ["check", "--offline", "--locked", "--target", "wasm32-unknown-unknown", "--no-default-features", "--features", "web,human_motion", "--bin", "viewer"]),
        ("final-native-gpu", ["test", "--offline", "--locked", "-p", "bevy_zeroverse", "--test", "procedural_indoor_render", "--test", "co_visibility_render", "--test", "ground_truth_render", "--test", "optical_flow_render", "--", "--ignored", "--test-threads=1"]),
    ]
    for seed, human, density in [(881, 0.25, 0.0), (158, 1.0, 0.0), (3779, 0.25, 0.35),
                                 (670, 1.0, 0.65), (3445, 0.25, 0.65), (682, 1.0, 0.65)]:
        commands.append((f"final-recovered-render-{seed}", ["run", "--offline", "--locked", "-p", "bevy_zeroverse", "--no-default-features", "--features", "multi_threaded", "--bin", "indoor_validate", "--", "--seed", str(seed), "--audit-seeds", "1", "--audit-geometry", "--renders", "1", "--cameras", "4", "--density", str(density), "--human-density", str(human), "--width", "512", "--height", "512", "--playback-steps", "2", "--labels", "--co-visibility", "--no-raw", "--output", str(OUT / f"recovered-render-{seed}")]))
    commands.append(("final-core-package", ["package", "--offline", "--locked", "--registry", "crates-io", "-p", "bevy_zeroverse", "--allow-dirty", "--no-verify"]))
    commands.append(("final-broad-scenes", ["test", "--offline", "--locked", "-p", "bevy_zeroverse", "--lib", "--features", "human_motion", "scene::procedural_indoor::tests::broad_", "--", "--ignored", "--nocapture", "--test-threads=2"]))
    return commands


def write_json(path, record):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def load_checks(path, required, identity):
    record = read_json(path)
    require(record.get("outcome") == "complete" and record.get("source_identity") == identity,
            f"Incomplete or stale qualification identity in {path}")
    checks = record.get("checks")
    require(isinstance(checks, list), f"Missing checks: {path}")
    names = [c.get("name") for c in checks]
    commands = expected_commands(required)
    require(names == [name for name, _ in commands] and set(names) == required,
            f"Incomplete or unexpected checks in {path}: {names}")
    for check, (_, arguments) in zip(checks, commands):
        require(type(check.get("exit_code")) is int and check["exit_code"] == 0,
                f"Failed or unfinished check: {check}")
        require(check.get("args") == arguments
                and check.get("source_verified_before_after") is True,
                f"Incorrect command or missing source checks: {check['name']}")
        log = OUT / (check["name"] + ".log")
        require(log.is_file(),
                f"Missing check log: {check['name']}")
        require(valid_digest(check.get("log_sha256")) and sha(log) == check["log_sha256"],
                f"Check log changed after qualification: {check['name']}")
        if required == REGISTRY_CHECKS:
            require(check.get("external_fixture_verified_before_after") is True
                    and check.get("environment_overrides") == {"BEVY_ASSET_ROOT": str(ROOT)},
                    f"Missing external model fixture/environment binding: {check['name']}")
    return record


def summaries(text):
    """Raw harness results; subprocess/child repeats are not unique fixtures."""
    results = []
    harness = None
    for line in text.splitlines():
        if line.lstrip().startswith(("Running ", "Doc-tests ")):
            harness = line.strip()
        match = TEST_SUMMARY.search(line)
        if match:
            status, passed, failed, ignored, measured, filtered, seconds = match.groups()
            require(status == "ok" and int(failed) == 0, f"Failed test summary: {line}")
            results.append({
                "harness": harness, "passed": int(passed), "failed": int(failed),
                "ignored": int(ignored), "measured": int(measured),
                "filtered_out": int(filtered), "harness_seconds": float(seconds),
            })
    return results


def broad_summary(text):
    density_rows = re.findall(
        r"density=([\d.]+): (\d+) seeds, (\d+) cameras, zero invalid scenes; layouts=([^\n]+)",
        text,
    )
    occupancy_rows = re.findall(
        r"full human density=1 furniture_density=([\d.]+): scenes=(\d+) cameras=(\d+) "
        r"people=(\d+) neighbor_people=(\d+) per_scene_min=(\d+) per_scene_max=(\d+) "
        r"layouts=([^\n]+)", text,
    )
    densities = [0.0, 0.35, 0.65, 1.0]
    require(sorted(float(row[0]) for row in density_rows) == densities,
            "Broad density sweep is missing a complete density stratum")
    require(sorted(float(row[0]) for row in occupancy_rows) == densities,
            "Full-occupancy sweep is missing a complete density stratum")
    require(all((int(row[1]), int(row[2])) == (10000, 40000) for row in density_rows),
            "Broad density denominators changed")
    require(all((int(row[1]), int(row[2])) == (1024, 4096) for row in occupancy_rows),
            "Full-occupancy denominators changed")
    geometry = re.search(
        r"full occupancy geometry: vertices=(\d+) triangles=(\d+); "
        r"poses=(.*?), outfits=(.*?), skin_tones=(.*?), hairstyles=([^\n]+)", text,
    )
    require(geometry is not None, "Full-occupancy geometry completion row is missing")
    for name in ["broad_density_sweep_is_valid",
                 "broad_full_human_occupancy_preserves_geometry_and_layout"]:
        require(name in text, f"Missing broad test: {name}")
    test_results = summaries(text)
    require(len(test_results) == 1 and test_results[0]["passed"] == 2,
            "Expected two completed broad qualification tests")
    return {
        "density_sweep": [{"furniture_density": float(d), "seed_start": 0,
                           "seeds": int(n), "cameras": int(c), "invalid_scenes": 0,
                           "layout_counts_rust_debug": layouts}
                          for d, n, c, layouts in density_rows],
        "density_scene_configurations": 40000,
        "density_distinct_seeds": 10000,
        "density_camera_configurations": 160000,
        "full_occupancy": [{"furniture_density": float(d), "human_density": 1.0,
                            "seed_start": 0, "scenes": int(n), "cameras": int(c),
                            "people": int(p), "neighbor_people": int(neighbors),
                            "per_scene_people_min": int(lo), "per_scene_people_max": int(hi),
                            "layout_counts_rust_debug": layouts}
                           for d, n, c, p, neighbors, lo, hi, layouts in occupancy_rows],
        "occupancy_scene_configurations": 4096,
        "occupancy_distinct_seeds": 1024,
        "occupancy_camera_configurations": 16384,
        "constructed_human_vertices": int(geometry[1]),
        "constructed_human_triangles": int(geometry[2]),
        "pose_set_rust_debug": geometry[3], "outfit_set_rust_debug": geometry[4],
        "skin_tone_set_rust_debug": geometry[5], "hairstyle_set_rust_debug": geometry[6],
        "assertions": [
            "Every density stratum: zero invalid scenes and exact 10000-layout denominator.",
            "Full occupancy: generation and validate_layout succeed for every scene.",
            "Constructed humans remain inside placement bounds and contact the floor.",
            "Mesh array lengths, triangular indices, finite vertices and unit normals pass.",
            "Every occupancy stratum retains all layouts, >4096 people and >128 neighbors.",
            "Eight poses, six outfits, eight skin tones and nineteen hairstyles are retained.",
        ],
        "scope": "Scene configurations repeat seeds across densities; totals are not distinct seeds.",
    }


def archive_inspection(archive, registry, frozen):
    checksum = sha(archive)
    require(checksum == registry.get("archive_sha256"),
            "Current archive differs from the exact archive used by registry_checks.py")
    require(external_model_fixture(OUT / "registry-workspace") == registry.get("external_model_fixture"),
            "External model fixture differs from the one used for registry qualification")
    prefix = "bevy_zeroverse-0.31.0/"
    with tarfile.open(archive, "r:gz") as package:
        members = package.getmembers()
        require(all(m.name.startswith(prefix) and (m.isfile() or m.isdir()) for m in members),
                "Unexpected archive root or special entry")
        names = [m.name for m in members if m.isfile()]
        require(len(names) == len(set(names)), "Duplicate archive paths")
        require(not any(n.startswith(prefix + folder) for n in names
                        for folder in ["third_party/", "out/", ".cargo/", "docs/", "www/", "assets/"]),
                "Local-only inputs entered the normalized package")
        normalized = tomllib.loads(package.extractfile(prefix + "Cargo.toml").read().decode())
        require(normalized["package"]["version"] == "0.31.0", "Wrong normalized core version")
        require(not any(k in normalized for k in ["patch", "replace", "workspace"]),
                "Normalized manifest contains local resolution overrides")
        require(package.extractfile(prefix + "Cargo.toml.orig").read()
                == (ROOT / "Cargo.toml").read_bytes(), "Archive original manifest differs from checkout")
        require(package.extractfile(prefix + "build.rs").read()
                == (ROOT / "build.rs").read_bytes(), "Archive build.rs differs from checkout")
        tables = [normalized, *normalized.get("target", {}).values()]
        for table in tables:
            for block in ["dependencies", "dev-dependencies", "build-dependencies"]:
                require(all(not isinstance(d, dict) or "path" not in d
                            for d in table.get(block, {}).values()),
                        "Normalized manifest contains a local path dependency")
        source_count = 0
        for name, expected in frozen["rust_inputs_sha256"].items():
            if name.startswith(("src/", "tests/")):
                data = package.extractfile(prefix + name).read()
                require(hashlib.sha256(data).hexdigest() == expected,
                        f"Archive differs from frozen Rust source: {name}")
                source_count += 1
        actual_rust = {n[len(prefix):] for n in names
                       if n.endswith(".rs") and n.startswith((prefix + "src/", prefix + "tests/"))}
        frozen_rust = {n for n in frozen["rust_inputs_sha256"] if n.startswith(("src/", "tests/"))}
        require(actual_rust == frozen_rust, "Archive Rust input set differs from final-source.json")
        core_inputs = registry["source_identity"]["package_inputs_sha256"]["core"]
        for name, expected in core_inputs.items():
            archived_name = "Cargo.toml.orig" if name == "Cargo.toml" else name
            require(hashlib.sha256(package.extractfile(prefix + archived_name).read()).hexdigest()
                    == expected, f"Archive compilation/test input differs from qualification: {name}")
        source_prefixes = tuple(prefix + folder + "/" for folder in PACKAGE_DIRECTORIES)
        actual_inputs = {name[len(prefix):] for name in names if name.startswith(source_prefixes)}
        expected_inputs = {name for name in core_inputs
                           if name.startswith(tuple(folder + "/" for folder in PACKAGE_DIRECTORIES))}
        require(actual_inputs == expected_inputs, "Archive compilation/test input membership differs")
        workspace = OUT / "registry-workspace"
        require(sha(workspace / "Cargo.toml") == registry.get("workspace_manifest_sha256"),
                "Tested registry workspace manifest changed")
        workspace_inputs = {name: package_source_signatures(workspace / name)
                            for name in ["core", "burn", "ffi"]}
        require(workspace_inputs == registry.get("workspace_inputs_sha256"),
                "Tested registry workspace sources/manifests changed")
        for name, expected in core_inputs.items():
            if name != "Cargo.toml":
                require(workspace_inputs["core"].get(name) == expected,
                        f"Tested core input differs from the archive: {name}")
        require(workspace_inputs["core"]["Cargo.toml"]
                == hashlib.sha256(package.extractfile(prefix + "Cargo.toml").read()).hexdigest(),
                "Tested core normalized manifest differs from the archive")
        for wrapper in ["burn", "ffi"]:
            expected = dict(registry["source_identity"]["package_inputs_sha256"][wrapper])
            text = (ROOT / "crates" / wrapper / "Cargo.toml").read_text()
            require(text.count('path = "../.."') == 1,
                    f"Unexpected wrapper core path replacement in {wrapper}")
            expected["Cargo.toml"] = hashlib.sha256(
                text.replace('path = "../.."', 'path = "../core"').encode()).hexdigest()
            require(workspace_inputs[wrapper] == expected,
                    f"Tested wrapper inputs differ from the final source: {wrapper}")
    lock = tomllib.loads((OUT / "registry-workspace/Cargo.lock").read_text())
    require(sha(OUT / "registry-workspace/Cargo.lock") == registry.get("resolved_lock_sha256"),
            "Registry qualification lock changed after testing")
    wgpu = [p for p in lock["package"] if p["name"] in ["wgpu", "wgpu-core", "wgpu-hal"]]
    require({p["name"] for p in wgpu} == {"wgpu", "wgpu-core", "wgpu-hal"},
            "Registry qualification lock lacks the expected WGPU packages")
    require(all(p["version"] == "29.0.4" and p.get("source", "").startswith("registry+")
                for p in wgpu), "WGPU qualification did not use registry 29.0.4")
    return {
        "name": archive.name, "sha256": checksum, "bytes": archive.stat().st_size,
        "regular_files": len(names), "frozen_core_rust_files_exact": source_count,
        "normalized_manifest_no_patch_replace_workspace_or_path_dependencies": True,
        "original_manifest_and_build_script_match_checkout": True,
        "all_core_compilation_and_test_inputs_exact": True,
        "tested_wrapper_inputs_exact_except_documented_core_path_rebinding": True,
        "local_only_directories_excluded": True,
        "external_model_fixture_required": True,
        "wgpu_resolution": [{k: p[k] for k in ["name", "version", "source"]} for p in wgpu],
        "scope": "Actual locally normalized archive inspected and tested with registry WGPU and a separately bound external checkout model fixture, excluded from the archive; "
                 "this checksum is not a claim about a subsequently uploaded crates.io archive.",
    }


def recovered_render(seed, frozen):
    expected_cases = {
        881: (0.0, 0.25), 158: (0.0, 1.0), 3779: (0.35, 0.25),
        670: (0.65, 1.0), 3445: (0.65, 0.25), 682: (0.65, 1.0),
    }
    require(seed in expected_cases, f"Unexpected recovered render seed {seed}")
    density, human_density = expected_cases[seed]
    directory = OUT / f"recovered-render-{seed}"
    complete = read_json(directory / "run_complete.json")
    selection = read_json(directory / "render_selection.json")
    capture = read_json(directory / f"seed_{seed:06}/capture.json")
    require(complete["captured_scenes"] == 1 and complete["selected_seeds"] == [seed],
            f"Incomplete recovered render {seed}")
    identity = complete["identity"]
    require(isinstance(identity, dict) and identity.get("schema_version") == 1
            and identity.get("source_sha256") == frozen["release_source_sha256"]
            and identity.get("crate_version") == "0.31.0"
            and type(identity.get("generator_version")) is int,
            f"Recovered render {seed} has stale generator identity")
    require(isinstance(complete["run_id"], str) and complete["run_id"]
            and selection.get("run_id") == complete["run_id"]
            and selection.get("identity") == identity
            and selection.get("selected_seeds") == [seed]
            and selection.get("playback_steps") == 2
            and selection.get("co_visibility") is True,
            f"Recovered render {seed} selection does not match its completed protocol")

    def exact_density(value, expected):
        # The validator arguments are f32; JSON may expose their widened f64 values.
        return (type(value) in (int, float) and math.isfinite(value)
                and 0.0 <= value <= 1.0
                and struct.pack("<f", value) == struct.pack("<f", expected))

    require(exact_density(selection.get("density"), density)
            and exact_density(selection.get("human_density"), human_density),
            f"Recovered render {seed} selection has incorrect furniture or human density")
    provenance = capture["build_provenance"]
    require(capture["seed"] == seed and capture["run_id"] == complete["run_id"]
            and capture["image_size"] == [512, 512]
            and capture["annotation_precision"] == "float32_geometry"
            and isinstance(provenance, dict)
            and all(provenance.get(field) == identity[field] for field in
                    ["schema_version", "crate_version", "generator_version", "source_sha256"])
            and isinstance(selection.get("capture_engine"), str)
            and selection["capture_engine"]
            and selection["capture_engine"] == provenance.get("capture_engine"),
            f"Recovered render {seed} metadata does not match the release protocol")
    views = capture["views"]
    require(len(views) == 8 and {(v["camera_index"], v["step_index"]) for v in views}
            == {(c, t) for c in range(4) for t in range(2)}, f"Incomplete view set {seed}")
    require(capture.get("co_visibility_metadata") is not None, f"Missing co-visibility {seed}")
    result = []
    for view in views:
        alignment = view.get("annotation_alignment")
        require(isinstance(alignment, dict) and alignment.get("checked_pixels", 0) > 0,
                f"Missing annotation validation {seed}")
        require(all(not isinstance(v, float) or math.isfinite(v) for v in alignment.values()),
                f"Nonfinite annotation report {seed}")
        require(view.get("co_visibility") is not None, f"Missing per-view co-visibility {seed}")
        result.append({"camera": view["camera_index"], "step": view["step_index"],
                       "time": view["time"], "pose_max_absolute_error": view["pose_max_absolute_error"],
                       "semantic_colors": view["semantic_colors"], "annotation_alignment": alignment})
    return {
        "seed": seed, "image_size": [512, 512], "cameras": 4, "timesteps": 2,
        "furniture_density": density, "human_density": human_density,
        "run_id": complete["run_id"], "generator_identity": complete["identity"],
        "capabilities": capture["capabilities"], "annotation_precision": capture["annotation_precision"],
        "views": result,
        "capture_metadata_sha256": sha(directory / f"seed_{seed:06}/capture.json"),
        "completion_metadata_sha256": sha(directory / "run_complete.json"),
        "selection_metadata_sha256": sha(directory / "render_selection.json"),
        "scope": "Completed actual recovery renders and validator alignment assertions; "
                 "raw images are retained only under out/. No throughput claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    args = parser.parse_args()
    require(not DESTINATION.exists(), f"Refusing to replace existing evidence: {DESTINATION}")
    identity = qualification_identity()
    final = load_checks(OUT / "final-checks.json", FINAL_CHECKS, identity)
    registry = load_checks(OUT / "registry-checks.json", REGISTRY_CHECKS, identity)
    frozen = read_json(OUT / "final-source.json")
    validated_inputs = {
        OUT / "final-checks.json": sha(OUT / "final-checks.json"),
        OUT / "registry-checks.json": sha(OUT / "registry-checks.json"),
        OUT / "final-source.json": identity["final_source_json_sha256"],
        Path(__file__).resolve(): sha(Path(__file__).resolve()),
    }
    require(valid_digest(frozen.get("release_source_sha256"))
            and valid_digest(frozen.get("measured_source_sha256"))
            and frozen.get("reconstructed_measured_inputs_exact") is True
            and frozen.get("transitive_resolution_unchanged") is True,
            "Missing final source identity or exact measured-source bridge")
    require(frozen["release_versions"] == {"bevy_zeroverse": "0.31.0",
            "bevy_zeroverse_burn": "0.14.0", "bevy_zeroverse_ffi": "0.31.0"},
            "Unexpected release versions")
    require(capture_source_identity() == (frozen["release_source_sha256"],
                                         frozen["release_source_inputs_count"]),
            "Live capture source/build/shader inputs drifted after freeze")
    live_rust = {str(path.relative_to(ROOT)) for folder in ["src", "tests", "crates/burn/src"]
                 for path in (ROOT / folder).rglob("*.rs")}
    require(live_rust == set(frozen["rust_inputs_sha256"]), "Live Rust source input set drifted")
    for name, expected in frozen["rust_inputs_sha256"].items():
        require(sha(ROOT / name) == expected, f"Live Rust source drifted after freeze: {name}")
    for name, expected in frozen["burn"]["source_inputs"].items():
        data = (ROOT / "crates/burn" / name).read_bytes()
        if Path(name).suffix in {".rs", ".toml", ".json", ".wgsl"}:
            data = data.decode("utf8").replace("\r\n", "\n").encode()
        require(hashlib.sha256(data).hexdigest() == expected, f"Burn producer input drifted: {name}")
    cpu_path = ROOT / "docs/evidence/cpu_efficiency/receipt.json"
    cpu = read_json(cpu_path)
    validated_inputs[cpu_path] = sha(cpu_path)
    require(cpu["current_provenance"]["source_sha256"] == frozen["measured_source_sha256"],
            "CPU measurement receipt does not match the measured source bridge")
    broad_path = OUT / "final-broad-scenes.log"
    broad = broad_summary(broad_path.read_text())
    archive = archive_inspection(args.archive, registry, frozen)
    renders = [recovered_render(seed, frozen) for seed in [881, 158, 3779, 670, 3445, 682]]
    test_results = {}
    logs = {}
    for check in final["checks"] + registry["checks"]:
        path = OUT / (check["name"] + ".log")
        text = path.read_text(errors="replace")
        parsed = summaries(text)
        if "tests" in check["name"] or check["name"] in {
                "final-broad-scenes", "final-native-gpu", "registry-gpu", "registry-burn-temporal"}:
            require(parsed and any(r["passed"] for r in parsed),
                    f"No completed passing tests in {path}")
        if parsed:
            test_results[check["name"]] = parsed
        logs[check["name"]] = {"path": path, "sha256": sha(path), "bytes": path.stat().st_size}
        validated_inputs[path] = check["log_sha256"]
    require("Packaged " in (OUT / "final-core-package.log").read_text(),
            "Missing actual package completion row")
    receipt = {
        "schema_version": 1, "assembled_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "scope": "Permanent pre-publication local qualification for release 0.31.0; "
                 "registry-WGPU archive tests are local tests, not registry upload verification.",
        "final_source": {k: v for k, v in frozen.items() if k not in ["rust_inputs_sha256", "burn"]},
        "qualification_input_identity": identity,
        "normalized_archive": archive,
        "external_model_fixture": registry["external_model_fixture"],
        "measured_cpu_snapshot": {
            "receipt": "../cpu_efficiency/receipt.json", "receipt_sha256": sha(cpu_path),
            "crate_version": cpu["current_provenance"]["crate_version"],
            "source_sha256": frozen["measured_source_sha256"],
            "binary_sha256": cpu["binary_sha256"]["final"],
            "scope": "Prior local optimized-dev/WGPU-patched measurement. Rare camera recovery "
                     "and release metadata differ. No final-source or registry performance was measured.",
        },
        "checks": {"final": final["checks"], "registry_archive": registry["checks"]},
        "raw_test_harness_results": test_results,
        "test_result_scope": "Raw Rust harness summaries, including possible child-process repeats; "
                             "these are not summed unique-fixture or sustained-throughput counts.",
        "broad_scene_qualification": broad,
        "recovered_render_probes": renders,
        "excluded_claims": ["2x throughput", "final-source/registry performance",
                            "photographic realism", "downstream training benefit",
                            "unlimited-process stability", "browser runtime qualification",
                            "post-commit CI success", "crates.io publication/checksum verification",
                            "live deployment verification"],
        "excluded_artifacts": ["Capture images/raw planes", "binaries", "crate archives",
                               "large capture manifests", "post-commit CI/registry/live-state records"],
        "artifacts": {},
    }
    # Do not write tracked evidence until every read/validation above succeeds.
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".release_031-", dir=DESTINATION.parent))
    try:
        paths = [OUT / "final-checks.json", OUT / "registry-checks.json", OUT / "final-source.json",
                 Path(__file__).resolve()]
        paths += [record["path"] for record in logs.values()
                  if record["path"] == broad_path or record["bytes"] <= SMALL_FILE_LIMIT]
        for path in paths:
            require(path == broad_path or path.stat().st_size <= SMALL_FILE_LIMIT,
                    f"Unexpectedly large permanent evidence input: {path}")
            name = path.name
            shutil.copyfile(path, stage / name)
            copied_sha256 = sha(stage / name)
            require(copied_sha256 == validated_inputs[path],
                    f"Evidence changed while being copied: {path}")
            receipt["artifacts"][name] = {"sha256": copied_sha256, "bytes": path.stat().st_size}
        receipt["large_logs_retained_only_in_out"] = [
            {"path": str(record["path"].relative_to(ROOT)), "sha256": record["sha256"],
             "bytes": record["bytes"]} for record in logs.values()
            if record["path"] != broad_path and record["bytes"] > SMALL_FILE_LIMIT
        ]
        (stage / "receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (stage / "README.md").write_text(
            "# Release 0.31 local qualification\n\n"
            "The receipt binds final frozen source, complete successful check records, the full "
            "40,000-configuration density and 4,096-configuration occupancy logs, actual recovered "
            "render assertions, and the exact locally inspected normalized core archive checksum.\n\n"
            "Registry WGPU tests exercised that local archive with a separately hashed external "
            "fullbody model fixture and verified qualification-only asset symlinks. The assets are "
            "not copied into or packaged with the archive. These tests do not prove upload, exact-commit "
            "CI, deployed Pages, or a subsequently published registry archive. Those state-dependent "
            "records remain under `out/release_031`. Raw capture files, binaries and package archives "
            "are omitted. Test harness counts can include child-process repeats.\n\n"
            "[The CPU throughput receipt](../cpu_efficiency/receipt.json) remains a separately "
            "identified pre-release 0.30.0 measurement. Its timings are not measurements of the "
            "final camera-fix source or registry packages.\n"
        )
        # A check rerun or source/package mutation must not turn a previously
        # validated snapshot into mixed evidence during the staging window.
        require(qualification_identity() == identity, "Qualification inputs changed during assembly")
        require(load_checks(OUT / "final-checks.json", FINAL_CHECKS, identity) == final
                and load_checks(OUT / "registry-checks.json", REGISTRY_CHECKS, identity) == registry,
                "Qualification reports changed during assembly")
        require(archive_inspection(args.archive, registry, frozen) == archive,
                "Archive/workspace qualification changed during assembly")
        for path, expected in validated_inputs.items():
            require(sha(path) == expected, f"Validated evidence changed during assembly: {path}")
        for render in renders:
            directory = OUT / f"recovered-render-{render['seed']}"
            for path, field in [
                (directory / f"seed_{render['seed']:06}/capture.json", "capture_metadata_sha256"),
                (directory / "run_complete.json", "completion_metadata_sha256"),
                (directory / "render_selection.json", "selection_metadata_sha256"),
            ]:
                require(sha(path) == render[field], "Recovered-render metadata changed during assembly")
        require(not DESTINATION.exists(), "Destination appeared during assembly")
        os.rename(stage, DESTINATION)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise
    print(json.dumps({"receipt": str(DESTINATION / "receipt.json"),
                      "release_source_sha256": frozen["release_source_sha256"],
                      "normalized_archive_sha256": archive["sha256"]}, indent=2))


if __name__ == "__main__":
    main()
