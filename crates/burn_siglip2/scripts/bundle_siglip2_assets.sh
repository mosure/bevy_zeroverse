#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${1:-${ROOT_DIR}/assets/models/siglip2}"
DST_ROOT="${2:-${ROOT_DIR}/dist/cdn/siglip2}"
STRICT="${BURN_SIGLIP2_CDN_BUNDLE_STRICT:-${BURN_SIGLIP2_WEB_BUNDLE_STRICT:-0}}"

if [[ -z "${DST_ROOT}" || "${DST_ROOT}" == "/" ]]; then
  echo "[siglip2-cdn] refusing unsafe destination '${DST_ROOT}'" >&2
  exit 1
fi
SRC_CANON="$(python3 - "${SRC_ROOT}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).resolve(strict=False))
PY
)"
DST_CANON="$(python3 - "${DST_ROOT}" <<'PY'
import pathlib
import sys
print(pathlib.Path(sys.argv[1]).resolve(strict=False))
PY
)"
if [[ "${DST_CANON}" == "/" \
  || "${DST_CANON}" == "${SRC_CANON}" \
  || "${DST_CANON}" == "${SRC_CANON}/"* \
  || "${SRC_CANON}" == "${DST_CANON}/"* ]]; then
  echo "[siglip2-cdn] refusing overlapping source/destination: source='${SRC_CANON}' destination='${DST_CANON}'" >&2
  exit 1
fi
if [[ "${STRICT}" != "0" && "${STRICT}" != "1" ]]; then
  echo "[siglip2-cdn] strict mode must be 0 or 1, got '${STRICT}'" >&2
  exit 1
fi

variants=(
  "base-patch16-224"
  "large-patch16-256"
  "so400m-patch14-224"
)

declare -A upstream_ids=(
  ["base-patch16-224"]="google/siglip2-base-patch16-224"
  ["large-patch16-256"]="google/siglip2-large-patch16-256"
  ["so400m-patch14-224"]="google/siglip2-so400m-patch14-224"
)

mkdir -p "${DST_ROOT}"
find "${DST_ROOT}" -maxdepth 1 -type f \( -name "index.json" -o -name "SHA256SUMS" \) -delete

processed=()

for variant in "${variants[@]}"; do
  stem="siglip2-${variant}"
  variant_src="${SRC_ROOT}/${variant}"
  if [[ ! -d "${variant_src}" ]]; then
    variant_src="${SRC_ROOT}"
  fi
  manifest_src="${variant_src}/${stem}.bpk.parts.json"
  variant_dst="${DST_ROOT}/${variant}"
  mkdir -p "${variant_dst}"
  find "${variant_dst}" -mindepth 1 -depth -delete

  if [[ ! -f "${manifest_src}" ]]; then
    if [[ "${STRICT}" == "1" ]]; then
      echo "[siglip2-cdn] missing ${variant} manifest: ${manifest_src}" >&2
      exit 1
    fi
    echo "[siglip2-cdn] ${variant}: no manifest, skipped" >&2
    continue
  fi

  part_list="$(mktemp)"
  if ! python3 - "${manifest_src}" "${variant_src}" "${variant}" "${upstream_ids[${variant}]}" >"${part_list}" <<'PY'
import hashlib
import json
import pathlib
import re
import sys

manifest_path = pathlib.Path(sys.argv[1])
source_dir = pathlib.Path(sys.argv[2])
expected_variant = sys.argv[3]
expected_model_id = sys.argv[4]

with manifest_path.open("r", encoding="utf-8") as handle:
    manifest = json.load(handle)

def fail(message: str) -> None:
    raise SystemExit(f"{manifest_path}: {message}")

if manifest.get("manifest_kind") != "siglip2_bpk_parts":
    fail("missing production manifest_kind=siglip2_bpk_parts")
if manifest.get("model_family") != "siglip2":
    fail("model_family must be siglip2")

artifact = manifest.get("artifact")
if not isinstance(artifact, dict):
    fail("missing immutable artifact metadata")
if artifact.get("model_variant") != expected_variant:
    fail(f"model_variant must be {expected_variant!r}")
if artifact.get("upstream_model_id") != expected_model_id:
    fail(f"upstream_model_id must be {expected_model_id!r}")
revision = artifact.get("upstream_revision")
if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
    fail("upstream_revision must be a full 40-character git commit")
if artifact.get("storage_dtype") != "f16":
    fail("CDN bundles require storage_dtype=f16")
if manifest.get("storage_dtypes") != ["f16"]:
    fail(f"all stored tensors must be f16, got {manifest.get('storage_dtypes')!r}")

expected_profiles = {
    "base-patch16-224": {
        "channels": 3, "hidden_dim": 768, "image_size": 224,
        "intermediate_dim": 3072, "num_heads": 12, "num_layers": 12,
        "patch_size": 16, "projection_dim": 768, "text_max_positions": 64,
        "text_pad_token_id": 0, "text_vocab_size": 256000,
    },
    "large-patch16-256": {
        "channels": 3, "hidden_dim": 1024, "image_size": 256,
        "intermediate_dim": 4096, "num_heads": 16, "num_layers": 24,
        "patch_size": 16, "projection_dim": 1024, "text_max_positions": 64,
        "text_pad_token_id": 0, "text_vocab_size": 256000,
    },
    "so400m-patch14-224": {
        "channels": 3, "hidden_dim": 1152, "image_size": 224,
        "intermediate_dim": 4304, "num_heads": 16, "num_layers": 27,
        "patch_size": 14, "projection_dim": 1152, "text_max_positions": 64,
        "text_pad_token_id": 0, "text_vocab_size": 256000,
    },
}
expected_profile = expected_profiles[expected_variant]
config = manifest.get("config")
expected_config_keys = set(expected_profile)
if not isinstance(config, dict) or set(config) != expected_config_keys | {"layer_norm_eps"}:
    fail("config must contain exactly the supported production profile fields")
for field in expected_config_keys:
    if config.get(field) != expected_profile[field]:
        fail(
            f"config field {field!r} must be {expected_profile[field]!r}, "
            f"got {config.get(field)!r}"
        )
layer_norm_eps = config.get("layer_norm_eps")
if not isinstance(layer_norm_eps, (int, float)) or abs(layer_norm_eps - 1.0e-6) > 1.0e-12:
    fail(f"config layer_norm_eps must be 1e-6, got {layer_norm_eps!r}")
expected_tensor_count = 32 * expected_profile["num_layers"] + 43
if manifest.get("tensor_count") != expected_tensor_count:
    fail(
        f"portable production tensor_count must be {expected_tensor_count}, "
        f"got {manifest.get('tensor_count')!r}"
    )

safe_name = re.compile(r"[A-Za-z0-9._-]+")
source_file = manifest.get("source_file")
if not isinstance(source_file, str) or safe_name.fullmatch(source_file) is None:
    fail("source_file is not a safe file name")

parts = manifest.get("parts")
if not isinstance(parts, list) or not parts:
    fail("parts must be a non-empty list")
seen = set()
actual_max = 0
for index, part in enumerate(parts):
    if not isinstance(part, dict):
        fail(f"part {index} is not an object")
    name = part.get("path")
    if not isinstance(name, str) or safe_name.fullmatch(name) is None or not name.endswith(".bpk"):
        fail(f"part {index} has an unsafe/non-BPK path")
    if name in seen:
        fail(f"duplicate part path {name!r}")
    seen.add(name)
    expected_bytes = part.get("bytes")
    expected_sha = part.get("sha256")
    if not isinstance(expected_bytes, int) or expected_bytes <= 0:
        fail(f"part {name!r} has invalid bytes")
    if not isinstance(expected_sha, str) or re.fullmatch(r"[0-9a-fA-F]{64}", expected_sha) is None:
        fail(f"part {name!r} has invalid sha256")
    part_path = source_dir / name
    if not part_path.is_file():
        fail(f"listed part is missing: {part_path}")
    actual_bytes = part_path.stat().st_size
    if actual_bytes != expected_bytes:
        fail(f"part {name!r} bytes mismatch: manifest={expected_bytes}, actual={actual_bytes}")
    digest = hashlib.sha256()
    with part_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    actual_sha = digest.hexdigest()
    if actual_sha.lower() != expected_sha.lower():
        fail(f"part {name!r} sha256 mismatch: manifest={expected_sha}, actual={actual_sha}")
    actual_max = max(actual_max, actual_bytes)

sharding = manifest.get("sharding")
if not isinstance(sharding, dict) or sharding.get("strategy") != "portable_text_embedding_chunks_v1":
    fail("CDN bundles require portable_text_embedding_chunks_v1 sharding metadata")
requested = sharding.get("requested_max_part_bytes")
declared_actual = sharding.get("actual_max_part_bytes")
if not isinstance(requested, int) or requested <= 0:
    fail("requested_max_part_bytes must be positive")
if declared_actual != actual_max:
    fail(f"actual_max_part_bytes mismatch: manifest={declared_actual}, actual={actual_max}")
if sharding.get("limit_enforced") != (actual_max <= requested):
    fail("limit_enforced does not match actual shard sizes")
web_max_part_bytes = 64 * 1024 * 1024
if actual_max > web_max_part_bytes:
    fail(
        f"largest shard {actual_max} exceeds the browser/CDN maximum "
        f"{web_max_part_bytes}; regenerate with portable token-embedding chunks"
    )
if sharding.get("limit_enforced") is not True:
    fail("CDN bundles require every serialized shard to honor the requested maximum")

for name in sorted(seen):
    print(name)
PY
  then
    rm -f "${part_list}"
    exit 1
  fi

  mapfile -t part_names <"${part_list}"
  rm -f "${part_list}"

  cp -f "${manifest_src}" "${variant_dst}/$(basename "${manifest_src}")"
  for part_name in "${part_names[@]}"; do
    cp -f "${variant_src}/${part_name}" "${variant_dst}/${part_name}"
  done

  required_sidecars=("tokenizer.json" "tokenizer_config.json" "preprocessor_config.json")
  optional_sidecars=("tokenizer.model")
  for sidecar in "${required_sidecars[@]}"; do
    sidecar_src="${variant_src}/${stem}.${sidecar}"
    if [[ ! -f "${sidecar_src}" ]]; then
      echo "[siglip2-cdn] ${variant}: missing required image/text sidecar ${sidecar_src}" >&2
      exit 1
    fi
    cp -f "${sidecar_src}" "${variant_dst}/$(basename "${sidecar_src}")"
  done
  for sidecar in "${optional_sidecars[@]}"; do
    sidecar_src="${variant_src}/${stem}.${sidecar}"
    if [[ -f "${sidecar_src}" ]]; then
      cp -f "${sidecar_src}" "${variant_dst}/$(basename "${sidecar_src}")"
    fi
  done

  python3 - "${variant_dst}" "$(basename "${manifest_src}")" >"${variant_dst}/bundle.manifest.json" <<'PY'
import hashlib
import json
import pathlib
import sys

bundle_dir = pathlib.Path(sys.argv[1])
parts_manifest_name = sys.argv[2]
with (bundle_dir / parts_manifest_name).open("r", encoding="utf-8") as handle:
    parts_manifest = json.load(handle)

part_names = {entry["path"] for entry in parts_manifest["parts"]}
expected_parts = {entry["path"]: entry for entry in parts_manifest["parts"]}
files = []
total_bytes = 0
for path in sorted(bundle_dir.iterdir(), key=lambda item: item.name):
    if not path.is_file() or path.name in {"bundle.manifest.json", "SHA256SUMS"}:
        continue
    if path.name == parts_manifest_name:
        role = "parts_manifest"
    elif path.name in part_names:
        role = "model_part"
    elif path.name.endswith(".tokenizer.json"):
        role = "tokenizer"
    elif path.name.endswith(".preprocessor_config.json"):
        role = "image_preprocessor"
    else:
        role = "tokenizer_sidecar"
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    size = path.stat().st_size
    if role == "model_part":
        expected = expected_parts[path.name]
        if size != expected["bytes"] or digest.hexdigest().lower() != expected["sha256"].lower():
            raise SystemExit(f"copied shard verification failed for {path}")
    if role in {"tokenizer", "image_preprocessor"} or path.name.endswith(".tokenizer_config.json"):
        try:
            json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise SystemExit(f"invalid required JSON sidecar {path}: {error}")
    total_bytes += size
    files.append({"path": path.name, "role": role, "bytes": size, "sha256": digest.hexdigest()})

document = {
    "schema_version": 1,
    "manifest_kind": "siglip2_cdn_bundle",
    "model_family": "siglip2",
    "model_variant": parts_manifest["artifact"]["model_variant"],
    "upstream_model_id": parts_manifest["artifact"]["upstream_model_id"],
    "upstream_revision": parts_manifest["artifact"]["upstream_revision"],
    "storage_dtype": parts_manifest["artifact"]["storage_dtype"],
    "parts_manifest": parts_manifest_name,
    "sharding": parts_manifest["sharding"],
    "payload_bytes": total_bytes,
    "files": files,
}
json.dump(document, sys.stdout, indent=2, ensure_ascii=True)
sys.stdout.write("\n")
PY

  (
    cd "${variant_dst}"
    find . -maxdepth 1 -type f ! -name "SHA256SUMS" -printf '%f\n' \
      | LC_ALL=C sort \
      | xargs sha256sum >SHA256SUMS
  )

  if [[ -e "${variant_dst}/${stem}.bpk" ]]; then
    echo "[siglip2-cdn] internal error: source monolith leaked into ${variant_dst}" >&2
    exit 1
  fi
  processed+=("${variant}")
  echo "[siglip2-cdn] ${variant}: ${#part_names[@]} verified f16 shards -> ${variant_dst}"
done

if [[ "${#processed[@]}" -eq 0 ]]; then
  echo "[siglip2-cdn] no complete model bundles found under ${SRC_ROOT}" >&2
  exit 0
fi

python3 - "${DST_ROOT}" "${processed[@]}" >"${DST_ROOT}/index.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
variants = sys.argv[2:]
bundles = []
for variant in variants:
    path = root / variant / "bundle.manifest.json"
    raw = path.read_bytes()
    document = json.loads(raw)
    bundles.append({
        "model_variant": variant,
        "manifest": f"{variant}/bundle.manifest.json",
        "bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "upstream_model_id": document["upstream_model_id"],
        "upstream_revision": document["upstream_revision"],
        "storage_dtype": document["storage_dtype"],
    })
json.dump({
    "schema_version": 1,
    "manifest_kind": "siglip2_cdn_index",
    "model_family": "siglip2",
    "bundles": bundles,
}, sys.stdout, indent=2, ensure_ascii=True)
sys.stdout.write("\n")
PY

(
  cd "${DST_ROOT}"
  sha256sum index.json >SHA256SUMS
)

echo "[siglip2-cdn] upload the contents of ${DST_ROOT}; source monoliths were excluded"
