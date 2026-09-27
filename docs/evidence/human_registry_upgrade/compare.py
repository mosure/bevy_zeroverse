import json
import math
import sys
from pathlib import Path

root = Path(sys.argv[1])
prefix = sys.argv[2]
summary = {"seeds": [], "limits": {"frame_component_max_absolute_error": 1e-6, "contact_disagreements": 0}}

def component_error(a, b):
    if isinstance(a, list):
        assert len(a) == len(b)
        return max((component_error(x, y) for x, y in zip(a, b)), default=0.0)
    assert math.isfinite(a) and math.isfinite(b)
    return abs(a - b)

for seed in (1, 13):
    serial = root / f"{prefix}_serial" / str(seed)
    batched = root / f"{prefix}_batch2" / str(seed)
    reports = [json.loads((path / "motion.json").read_text()) for path in (serial, batched)]
    clips = [dict(json.loads((path / "clips.json").read_text())) for path in (serial, batched)]
    assert clips[0].keys() == clips[1].keys(), "admitted actor identities differ"
    assert reports[0]["accepted"] == reports[1]["accepted"], "admitted plans differ"
    assert reports[0]["rejected"] == reports[1]["rejected"], "motion rejections differ"
    maximum, contacts, frames = 0.0, 0, 0
    for actor in clips[0]:
        a, b = clips[0][actor], clips[1][actor]
        assert a["rig"] == b["rig"] and a["fps"] == b["fps"]
        assert len(a["frames"]) == len(b["frames"]) == 120
        for f, g in zip(a["frames"], b["frames"]):
            maximum = max(maximum, component_error(f["root_translation"], g["root_translation"]),
                          component_error(f["local_rotations"], g["local_rotations"]))
            contacts += sum(x != y for x, y in zip(f["foot_contacts"], g["foot_contacts"]))
            frames += 1
    assert maximum <= 1e-6 and contacts == 0, (maximum, contacts)
    assert reports[0]["model_loads"] == reports[1]["model_loads"] <= 1, "model residency differs or reloaded"
    captures = [json.loads((path / "capture.json").read_text()) for path in (serial, batched)]
    assert all(len(capture["annotation_checks"]) == 10 for capture in captures)
    summary["seeds"].append({"seed": seed, "admitted_actors": len(clips[0]), "frames_compared": frames,
        "frame_component_max_absolute_error": maximum, "contact_disagreements": contacts,
        "model_loads": [report["model_loads"] for report in reports],
        "cumulative_batches": [report["generated_batches"] for report in reports],
        "annotation_views_checked": sum(len(capture["annotation_checks"]) for capture in captures)})
assert summary["seeds"][-1]["cumulative_batches"][0] > summary["seeds"][-1]["cumulative_batches"][1], "no batch aggregation was exercised"
assert sum(row["admitted_actors"] for row in summary["seeds"]) >= 2, "insufficient admitted actors"
assert summary["seeds"][-1]["model_loads"] == [1, 1], "real models were not exercised"
static = json.loads((root / f"{prefix}_static" / "13" / "motion.json").read_text())
assert static["model_loads"] == 0 and static["generated_batches"] == 0 and not static["accepted"]
static_capture = json.loads((root / f"{prefix}_static" / "13" / "capture.json").read_text())
assert len(static_capture["annotation_checks"]) == 10
summary["static"] = {"model_loads": 0, "generated_batches": 0, "annotation_views_checked": 10}
summary["passed"] = True
output = root / f"{prefix}_comparison.json"
output.write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
