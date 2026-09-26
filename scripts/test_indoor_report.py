"""CPU-only regression checks for the render report's evidence integrity.

Run with: python3 -m unittest discover -s scripts -p test_indoor_report.py
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import indoor_report


class ReportIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="indoor-report-test-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.run_id = "current-test-run"
        self.seeds = [3, 7]
        self.write("metrics.json", {
            "generator_version": 2, "image_size": [4, 3], "density": 0.65,
            "scenes": 20,
        })
        self.write("render_selection.json", {
            "run_id": self.run_id, "selected_seeds": self.seeds,
            "playback_steps": 2, "quality": "Auto", "policy": "test selection",
            "observed_strata": {"first": 3, "second": 7},
        })
        self.write("run_complete.json", {
            "run_id": self.run_id, "selected_seeds": self.seeds,
        })
        view = {
            "semantic_pixel_counts": {"chair": 2, "floor": 10},
            "mean_luminance": 0.2, "luminance_std": 0.1,
            "dark_fraction": 0.0, "clipped_fraction": 0.0,
            "pose_max_absolute_error": 0.0, "annotation_alignment": None,
        }
        for seed in self.seeds:
            self.write(f"seed_{seed:06}/capture.json", {
                "run_id": self.run_id, "seed": seed, "image_size": [4, 3],
                "capabilities": {"quality": "Auto"},
                "renderer": "test-adapter", "views": [copy.deepcopy(view), copy.deepcopy(view)],
                "mesh_assets": 2, "material_assets": 3, "image_assets": 4,
                "elapsed_seconds": 0.1,
            })
            self.write(f"seed_{seed:06}/manifest.json", {
                "seed": seed, "generator_version": 2, "density": 0.65,
                "cameras": [{}], "layout": "Conference", "lighting": "Daylight",
                "floor_style": 0, "furniture_style": 0,
            })

    def write(self, name, value):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def change(self, name, key, value):
        data = json.loads((self.root / name).read_text())
        data[key] = value
        self.write(name, data)

    def assert_rejected(self, expected_error=ValueError):
        with self.assertRaises(expected_error):
            indoor_report.summarize(self.root)
        self.assertFalse((self.root / "render_summary.json").exists())

    def test_matching_complete_run_preserves_view_and_pixel_counts(self):
        _, reports, summary = indoor_report.summarize(self.root)
        self.assertEqual(len(reports), 2)
        self.assertEqual(summary["run_id"], self.run_id)
        self.assertEqual(summary["rendered_scenes"], 2)
        self.assertEqual(summary["rendered_views"], 4)
        self.assertEqual(summary["semantic_pixel_counts"], {"chair": 8, "floor": 40})
        self.assertEqual(summary["semantic_view_presence_fraction"]["chair"], 1.0)
        self.assertEqual(
            json.loads((self.root / "render_summary.json").read_text()),
            json.loads(json.dumps(summary)),
        )

    def test_stale_capture_cannot_be_mixed_with_current_run(self):
        self.change("seed_000007/capture.json", "run_id", "previous-run")
        self.assert_rejected()

    def test_missing_capture_does_not_silently_reduce_population(self):
        (self.root / "seed_000007/capture.json").unlink()
        self.assert_rejected(FileNotFoundError)

    def test_incomplete_run_with_all_capture_files_is_rejected(self):
        (self.root / "run_complete.json").unlink()
        self.assert_rejected(FileNotFoundError)

    def test_stale_completion_marker_is_rejected(self):
        self.change("run_complete.json", "run_id", "previous-run")
        self.assert_rejected()

    def test_completion_must_cover_the_exact_selected_population(self):
        self.change("run_complete.json", "selected_seeds", [3])
        self.assert_rejected()

    def test_empty_selection_cannot_be_reported_as_render_qualification(self):
        self.change("render_selection.json", "selected_seeds", [])
        self.change("run_complete.json", "selected_seeds", [])
        self.assert_rejected()

    def test_same_run_still_requires_matching_dimensions(self):
        self.change("seed_000007/capture.json", "image_size", [8, 6])
        self.assert_rejected()

    def test_same_run_still_requires_matching_density(self):
        self.change("seed_000007/manifest.json", "density", 0.35)
        self.assert_rejected()

    def test_same_run_still_requires_matching_quality(self):
        self.change("seed_000007/capture.json", "capabilities", {"quality": "Portable"})
        self.assert_rejected()

    def test_same_run_still_requires_matching_seed(self):
        self.change("seed_000007/manifest.json", "seed", 8)
        self.assert_rejected()

    def test_same_run_still_requires_matching_generator(self):
        self.change("seed_000007/manifest.json", "generator_version", 1)
        self.assert_rejected()

    def test_missing_timestep_is_rejected(self):
        self.change("seed_000007/capture.json", "views", [])
        self.assert_rejected()


if __name__ == "__main__":
    unittest.main()
