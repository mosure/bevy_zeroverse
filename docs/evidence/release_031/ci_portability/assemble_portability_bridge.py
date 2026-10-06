#!/usr/bin/env python3
"""Assemble the test-only NaN portability bridge after all three gates pass.

This helper does not run Cargo, modify the old release packet/freeze, publish,
or claim that inherited runtime checks were rerun. Run from this checkout after
out/release_031/portability-checks.json is complete. A new evidence directory is
installed atomically; an existing directory is never overwritten.
"""

import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import shutil
import struct
import subprocess
import tempfile


ROOT = next(path for path in Path(__file__).resolve().parents
            if (path / "Cargo.toml").is_file()
            and (path / "crates/capture/src/lib.rs").is_file())
OUT = ROOT / "out/release_031"
PREDECESSOR = ROOT / "docs/evidence/release_031"
DESTINATION = PREDECESSOR / "ci_portability"
OLD_COMMIT = "ee5084c5e407ad4f631ebd07e2e9630d90ae6174"
OLD_CAPTURE = "ab2e322b1b6121ec0fc8bfa6dfe77e0783625cb164dec80227b0a65eba83bdeb"
CHANGED = "src/scene/procedural_indoor/materials/mineral/carry/replay_tests.rs"
PARENT_MODULE = "src/scene/procedural_indoor/materials/mineral/carry.rs"
PACKAGE_DIRECTORIES = ("src", "tests", "benches", "examples")
PACKAGES = {
    "core": ROOT,
    **{name: ROOT / "crates" / name
       for name in ("burn", "ffi", "capture", "publication", "burn_siglip2")},
}
SMALL_LIMIT = 2 * 1024 * 1024
TEST_SUMMARY = re.compile(
    r"test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; "
    r"(\d+) measured; (\d+) filtered out; finished in ([\d.]+)s"
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    require(path.is_file() and not path.is_symlink(), f"Missing/linked input: {path}")
    require(path.stat().st_size <= SMALL_LIMIT, f"Oversized JSON input: {path}")
    return json.loads(path.read_text())


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def valid_digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def package_source_signatures(directory):
    """Complete compilation/test input set used by the preceding qualification."""
    files = {name: directory / name for name in ("Cargo.toml", "build.rs")
             if (directory / name).is_file()}
    for folder in PACKAGE_DIRECTORIES:
        for path in (directory / folder).rglob("*"):
            require(not path.is_symlink(), f"Linked package input: {path}")
            if path.is_file():
                files[path.relative_to(directory).as_posix()] = path
    return {name: sha(path) for name, path in sorted(files.items())}


def capture_source_inputs(overrides=None):
    """Mirror bevy_zeroverse_capture::provenance::source_inputs exactly."""
    overrides = overrides or {}
    files = {}
    for name in ("Cargo.toml", "Cargo.lock", "build.rs", "crates/capture/Cargo.toml",
                 "third_party/wgpu-core/Cargo.toml", "third_party/wgpu-hal/Cargo.toml"):
        path = ROOT / name
        if path.is_file():
            require(not path.is_symlink(), f"Linked capture input: {path}")
            files[name] = path
    for name in ("src", "crates/capture/src", "third_party/wgpu-core/src",
                 "third_party/wgpu-hal/src", "assets/shaders", "assets/embedded"):
        directory = ROOT / name
        require(not directory.is_symlink(), f"Linked capture directory: {directory}")
        for path in directory.rglob("*"):
            require(not path.is_symlink(), f"Linked capture input: {path}")
            if path.is_file():
                files[path.relative_to(ROOT).as_posix()] = path
    require(set(overrides).issubset(files), "Override names are absent from capture inputs")
    text = {"rs", "wgsl", "glsl", "metal", "hlsl", "vert", "frag", "toml", "lock",
            "json", "html", "css", "tex", "sty", "bib", "md", "svg", "txt"}
    inputs = {}
    for name, path in sorted(files.items()):
        data = overrides[name] if name in overrides else path.read_bytes()
        if path.suffix.lstrip(".") in text or path.name == "LICENSE":
            try:
                data = data.decode("utf8").replace("\r\n", "\n").encode()
            except UnicodeDecodeError:
                pass
        inputs[name] = sha_bytes(data)
    return inputs


def source_digest(inputs):
    digest = hashlib.sha256()
    for name, value in sorted(inputs.items()):
        encoded = name.encode()
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(value.encode())
    return digest.hexdigest()


def capture_source_identity():
    inputs = capture_source_inputs()
    return source_digest(inputs), len(inputs)


def qualification_identity():
    digest, count = capture_source_identity()
    return {
        "release_source_sha256": digest,
        "release_source_inputs_count": count,
        "package_inputs_sha256": {
            name: package_source_signatures(path) for name, path in PACKAGES.items()
        },
    }


def expected_commands():
    return [
        ("portability-fmt", ["fmt", "--all", "--", "--check"]),
        ("portability-core-tests", ["test", "--offline", "--locked", "-p",
                                    "bevy_zeroverse", "--lib", "--features", "human_motion",
                                    "--", "--test-threads=4"]),
        ("portability-clippy", ["clippy", "--offline", "--locked", "--workspace",
                                "--all-targets", "--features", "human_motion", "--",
                                "-D", "warnings"]),
    ]


def utc(value):
    require(isinstance(value, str), "Missing timestamp")
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(parsed.tzinfo is not None and parsed.utcoffset() == dt.timedelta(0),
            "Check timestamp must be UTC")
    return parsed


def verify_predecessor():
    receipt = read_json(PREDECESSOR / "receipt.json")
    committed_receipt = subprocess.run(
        ["git", "show", f"{OLD_COMMIT}:docs/evidence/release_031/receipt.json"], cwd=ROOT,
        check=True, stdout=subprocess.PIPE,
    ).stdout
    require(sha_bytes(committed_receipt) == sha(PREDECESSOR / "receipt.json"),
            "Predecessor receipt differs from its committed immutable packet")
    for name in ("final-source.json", "final-checks.json", "registry-checks.json"):
        expected = receipt["artifacts"][name]
        path = PREDECESSOR / name
        require(path.stat().st_size == expected["bytes"] and sha(path) == expected["sha256"],
                f"Predecessor artifact changed: {name}")
    frozen = read_json(PREDECESSOR / "final-source.json")
    checks = read_json(PREDECESSOR / "final-checks.json")
    registry = read_json(PREDECESSOR / "registry-checks.json")
    require(checks.get("outcome") == registry.get("outcome") == "complete",
            "Predecessor qualification is incomplete")
    require(len(checks["checks"]) == 13 and len(registry["checks"]) == 5
            and all(check["exit_code"] == 0 and check["source_verified_before_after"] is True
                    for record in (checks, registry) for check in record["checks"]),
            "Predecessor does not have eighteen successful frozen-source gates")
    old_identity = checks["source_identity"]
    require(old_identity["final_source_json_sha256"] == sha(PREDECESSOR / "final-source.json")
            and old_identity["release_source_sha256"] == OLD_CAPTURE
            and frozen["release_source_sha256"] == OLD_CAPTURE
            and frozen["release_source_inputs_count"] == 416,
            "Unexpected predecessor source identity")
    return receipt, frozen, old_identity


def verify_bridge(identity, frozen, old_identity):
    packages = identity["package_inputs_sha256"]
    old_packages = old_identity["package_inputs_sha256"]
    require(set(packages) == set(old_packages) == set(PACKAGES), "Package set changed")
    changes = []
    for package, current in packages.items():
        original = old_packages[package]
        require(set(current) == set(original), f"Package input membership changed: {package}")
        changes.extend({"package": package, "path": path,
                        "before_sha256": original[path], "after_sha256": current[path]}
                       for path in current if current[path] != original[path])
    require([(change["package"], change["path"]) for change in changes] == [("core", CHANGED)],
            f"Expected exactly one test-only package change, found: {changes}")
    require(identity["release_source_inputs_count"] == 416, "Capture input count changed")
    original = subprocess.run(["git", "show", f"{OLD_COMMIT}:{CHANGED}"], cwd=ROOT,
                              check=True, stdout=subprocess.PIPE).stdout
    require(sha_bytes(original) == old_packages["core"][CHANGED]
            and frozen["rust_inputs_sha256"][CHANGED] == sha_bytes(original),
            "Committed original test bytes differ from predecessor qualification")
    source = capture_source_inputs()
    rebuilt = capture_source_inputs({CHANGED: original})
    require(len(source) == len(rebuilt) == 416 and source_digest(rebuilt) == OLD_CAPTURE,
            "Restoring only old test bytes does not reconstruct the old capture digest")
    require(source_digest(source) == identity["release_source_sha256"]
            and source_digest(source) != OLD_CAPTURE,
            "Current capture identity does not match completed check identity")
    require([name for name in source if source[name] != rebuilt[name]] == [CHANGED],
            "Reconstruction changed another capture input")
    parent = ROOT / PARENT_MODULE
    require(sha(parent) == old_packages["core"][PARENT_MODULE]
            and re.search(r"#\[cfg\(test\)\]\s*mod replay_tests;", parent.read_text()) is not None,
            "Changed file is not gated by the unchanged cfg(test) parent module")
    for path, digest in old_identity["build_configuration_sha256"].items():
        require(sha(ROOT / path) == digest, f"Build configuration changed: {path}")
    return changes[0], original, source, rebuilt


def verify_checks(identity):
    record = read_json(OUT / "portability-checks.json")
    require(record.get("schema_version") == 2 and record.get("outcome") == "complete"
            and not record.get("current_check") and not record.get("error"),
            "Portability checks are incomplete or unsuccessful")
    require(record.get("source_identity") == identity,
            "Completed checks do not bind the complete current package/capture identity")
    commands = expected_commands()
    require([check.get("name") for check in record.get("checks", [])]
            == [name for name, _ in commands], "Incorrect ordered portability check inventory")
    last_finish = None
    summaries = []
    for check, (name, arguments) in zip(record["checks"], commands):
        require(check.get("args") == arguments and check.get("exit_code") == 0
                and check.get("source_verified_before_after") is True,
                f"Wrong command, failure, or absent source guards: {name}")
        start, finish = utc(check.get("started_utc")), utc(check.get("finished_utc"))
        require(finish >= start and (last_finish is None or start >= last_finish),
                f"Invalid/overlapping check chronology: {name}")
        last_finish = finish
        log = OUT / (name + ".log")
        require(log.is_file() and not log.is_symlink() and log.stat().st_size <= SMALL_LIMIT
                and valid_digest(check.get("log_sha256"))
                and sha(log) == check["log_sha256"], f"Missing/mismatched log: {name}")
        text = log.read_text()
        require(not any(result[0] == "FAILED" or result[2] != "0"
                        for result in TEST_SUMMARY.findall(text)), f"Failed harness: {name}")
        if name == "portability-core-tests":
            found = TEST_SUMMARY.findall(text)
            require(len(found) == 1 and int(found[0][1]) > 0 and found[0][5] == "0",
                    "Core gate must run the unfiltered library harness")
            required_tests = [
                "carried_decode_replays_original_transfers_without_assuming_identity",
                "carried_decode_preserves_signed_zero_nan_payloads_and_transfer_boundaries",
            ]
            for test in required_tests:
                require(re.search(r"test [^\n]*::" + re.escape(test) + r" \.\.\. ok\b", text),
                        f"Required replay test did not run successfully: {test}")
            summaries.append({"name": name, "passed": int(found[0][1]),
                              "failed": int(found[0][2]), "ignored": int(found[0][3]),
                              "filtered_out": int(found[0][5]), "seconds": float(found[0][6])})
    return record, summaries


def main():
    require(not DESTINATION.exists(), f"Refusing to replace existing evidence: {DESTINATION}")
    receipt, frozen, old_identity = verify_predecessor()
    identity = qualification_identity()
    change, original, current_inputs, reconstructed_inputs = verify_bridge(identity, frozen, old_identity)
    record, summaries = verify_checks(identity)
    old_packet_hashes = {name: sha(PREDECESSOR / name)
                         for name in ("receipt.json", "final-source.json", "final-checks.json",
                                      "registry-checks.json", "README.md")}
    with tempfile.TemporaryDirectory(prefix=".ci-portability-", dir=PREDECESSOR) as temporary:
        staged = Path(temporary) / "ci_portability"
        staged.mkdir()
        (staged / "logs").mkdir()
        shutil.copy2(OUT / "portability-checks.json", staged / "checks.json")
        for name, _ in expected_commands():
            shutil.copy2(OUT / (name + ".log"), staged / "logs" / (name + ".log"))
        shutil.copy2(Path(__file__), staged / "assemble_portability_bridge.py")
        (staged / "original-replay_tests.rs").write_bytes(original)
        shutil.copy2(ROOT / CHANGED, staged / "qualified-replay_tests.rs")
        bridge = {
            "schema_version": 1,
            "outcome": "complete",
            "assembled_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "scope": "Test-only arithmetic NaN portability follow-up to immutable release0.31 runtime qualification; no production code, wrapper, lockfile, or build configuration change.",
            "original_commit": OLD_COMMIT,
            "source_identity": identity,
            "single_changed_input": change,
            "capture_bridge": {
                "before_sha256": OLD_CAPTURE,
                "after_sha256": identity["release_source_sha256"],
                "before_inputs_count": len(reconstructed_inputs),
                "after_inputs_count": len(current_inputs),
                "current_inputs_sha256": current_inputs,
                "restoring_only_committed_original_test_bytes_reconstructs_before_exactly": True,
                "reconstruction_writes_to_checkout": False,
                "production_inputs_exact": True,
                "complete_package_input_memberships_exact": True,
                "all_other_package_inputs_exact": True,
                "all_wrapper_and_wire_crate_inputs_exact": True,
                "lockfile_manifests_and_build_configuration_exact": True,
                "changed_module_guard": "#[cfg(test)] mod replay_tests;",
                "unchanged_parent_module": {
                    "path": PARENT_MODULE,
                    "sha256": old_identity["package_inputs_sha256"]["core"][PARENT_MODULE],
                },
                "package_signature_scope": "Cargo.toml, build.rs when present, and all files under src/tests/benches/examples for each of the six packages; this is the predecessor's complete recorded input inventory, not a new archive checksum.",
            },
            "test_policy": {
                "arithmetic_nan": "Compare each channel's NaN classification; Rust does not guarantee arithmetic NaN payload/sign.",
                "non_nan_arithmetic": "Compare exact bits, including finite values, signed zero and infinity.",
                "non_arithmetic_construction": "Compare exact copied bits, including NaN payload/sign.",
                "finite_replay_corpus": "Existing exact-bit oracle and exceptional decode-reuse fallback assertions remain.",
            },
            "new_local_checks": record["checks"],
            "new_core_harness_summaries": summaries,
            "inherited_runtime_qualification": {
                "reference": "../receipt.json",
                "source_sha256": OLD_CAPTURE,
                "artifacts": old_packet_hashes,
                "completed_gates": 18,
                "density_scene_configurations": receipt["broad_scene_qualification"]["density_scene_configurations"],
                "occupancy_scene_configurations": receipt["broad_scene_qualification"]["occupancy_scene_configurations"],
                "scope": "Broad scene, full occupancy, native GPU/recovered renders and registry-WGPU checks were completed on the predecessor source. They were not rerun by this bridge. Production-input equality supports inheritance; it does not relabel old logs, binaries, archive checksums or throughput as new-source measurements.",
            },
            "new_ci_and_publication_proof": {
                "status": "separate_state_dependent_proof_required",
                "scope": "These three local reruns do not prove remote macOS/Windows CI, an uploaded archive, current canonical page/paper captures, or deployed Pages. The release closeout must separately verify those actual commit/source identities. The test-only source change still changes the capture digest and requires canonical publication refresh.",
            },
        }
        (staged / "README.md").write_text(
            "# Release 0.31 arithmetic NaN portability bridge\n\n"
            "This packet proves that the only changed compilation/test input after the "
            "[immutable runtime qualification](../receipt.json) is " + CHANGED + ". "
            "Its unchanged parent includes the module only under `cfg(test)`. All six package "
            "input memberships, every other compilation/test input, manifests, lockfile, local "
            "WGPU inputs and build configuration remain exact. Restoring the recorded "
            "`ee5084c` test bytes in hashing only reconstructs the old `ab2e322b` capture digest; "
            "both capture input counts are 416. No checkout source or old evidence is changed.\n\n"
            "The new completed local gates are formatting, the full core library harness with "
            "`human_motion`, and strict all-target workspace Clippy. The exceptional arithmetic "
            "oracle compares NaN classification per channel and exact bits for every non-NaN "
            "result, including signed zero and infinity. The finite oracle remains exact; "
            "non-arithmetic construction additionally checks exact NaN payload copying.\n\n"
            "The predecessor's eighteen runtime/package gates, 40,000 density scene "
            "configurations and 4,096 occupancy configurations are inherited, not newly run. "
            "The historical CPU timing packet remains separately bound to its measured source. "
            "This packet does not establish new-source render bytes, performance, remote CI, "
            "registry upload, canonical publication captures or deployed Pages. Actual release "
            "closeout records must verify those independently; the changed capture identity "
            "requires the canonical Rust publication refresh.\n"
        )
        bridge["artifacts"] = {
            path.relative_to(staged).as_posix(): {"sha256": sha(path), "bytes": path.stat().st_size}
            for path in sorted(staged.rglob("*")) if path.is_file()
        }
        write_json(staged / "receipt.json", bridge)
        require(qualification_identity() == identity, "Current source changed during assembly")
        require(all(sha(PREDECESSOR / name) == value for name, value in old_packet_hashes.items()),
                "Predecessor packet changed during bridge assembly")
        require(not DESTINATION.exists(), "Evidence destination appeared during assembly")
        staged.rename(DESTINATION)
    print(json.dumps({"outcome": "complete", "destination": str(DESTINATION),
                      "release_source_sha256": identity["release_source_sha256"],
                      "release_source_inputs_count": identity["release_source_inputs_count"]}))


if __name__ == "__main__":
    main()
