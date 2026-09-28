import hashlib
import json
import shutil
from pathlib import Path
import tempfile
import unittest

import numpy as np

from compare_indoor_cycles import block_areas, block_measurements, blocks, compare, measurements, optical_identity
from indoor_reference_contract import absorption_coefficients, sky_radiance, validate_snapshot


class ReferenceComparisonTests(unittest.TestCase):
    def test_beer_lambert_distance_and_thickness_contract(self):
        color = np.array([.4,.7,.9])
        sigma = np.array(absorption_coefficients(color,.016))
        np.testing.assert_allclose(np.exp(-sigma*.016),color)
        np.testing.assert_allclose(np.exp(-sigma*.008),np.sqrt(color))
        self.assertEqual(absorption_coefficients([1,1,1],None),(0,0,0))
        for distance in (0,-1,float("nan")):
            with self.assertRaises(ValueError): absorption_coefficients(color,distance)

    def test_continuous_sky_uses_manifest_radiance_with_legacy_fallback(self):
        self.assertEqual(sky_radiance({"lighting": "Evening"}), (16, 21, 32))
        manifest = {"lighting": "Evening", "program": {"domain": {
            "photometry": {"sky_radiance": [0.03, 0.04, 0.07]}}}}
        self.assertEqual(sky_radiance(manifest), (0.03, 0.04, 0.07))

    def test_spatial_metrics_preserve_radiant_energy(self):
        rng = np.random.default_rng(39)
        image = rng.exponential(size=(64, 96, 3))
        for size in (8, 16, 32):
            np.testing.assert_allclose(blocks(image, size).mean((0, 1)), image.mean((0, 1)))
        cropped = image[:63, :89]
        for size in (8, 16, 32):
            areas = block_areas(cropped.shape, size)
            np.testing.assert_allclose((blocks(cropped, size) * areas[..., None]).sum((0, 1)), cropped.sum((0, 1)))

    def test_partial_edge_blocks_are_included_with_their_pixel_area(self):
        reference = np.ones((35, 47, 3))
        native = reference.copy()
        native[32:] *= 3
        native[:, 32:] *= 2
        raw = measurements(native, reference, np.ones(reference.shape[:2], dtype=bool))
        for metric in block_measurements(native, reference).values():
            self.assertEqual(metric["covered_pixels"], 35 * 47)
            self.assertAlmostEqual(metric["rgb_relative_mae"], raw["rgb_relative_mae"])
            self.assertAlmostEqual(metric["luminance_bias_ratio"], raw["luminance_bias_ratio"])

    def test_fixed_exposure_error_is_not_fitted_away(self):
        image = np.ones((32, 32, 3))
        metrics = measurements(image * 2, image, np.ones((32, 32), dtype=bool))
        self.assertAlmostEqual(metrics["luminance_bias_ratio"], 2)
        self.assertAlmostEqual(metrics["median_absolute_stops"], 1)
        self.assertAlmostEqual(metrics["rgb_relative_mae"], 1)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.native, self.cycles, self.repeat = (self.root / name for name in ("native", "cycles", "repeat"))
        reference = self.native / "reference"
        reference.mkdir(parents=True)
        geometry = np.array([0., 0., 0., 1., 0., 0., 0., 1., 0.,
                             0., 0., 1., 0., 0., 1., 0., 0., 1.,
                             0., 0., 1., 0., 0., 1.], dtype="<f4").tobytes()
        geometry += np.array([0, 1, 2], dtype="<u4").tobytes()
        scene = {"image_size": [32, 32], "seed": 1, "ev100": 3, "capture_engine": "test",
                 "geometry_bytes": len(geometry), "meshes": [{"offset": 0, "vertices": 3, "indices": 3}],
                 "materials": [{"roughness": .5}], "instances": [{"mesh": 0, "material": 0}], "lights": [], "cameras": [{"time": 0}]}
        (reference / "scene.json").write_text(json.dumps(scene))
        (reference / "geometry.bin").write_bytes(geometry)
        (self.native / "manifest.json").write_text(json.dumps({"lighting": "Evening", "humans": []}))
        (self.native / "capture.json").write_text(json.dumps({"color_encoding": "exposed scene-linear"}))
        radiance = np.full((32, 32, 4), 4, dtype="<f4")
        (radiance * (2**-3 / 1.2)).tofile(self.native / "view_00_color.rgba32f")
        self.report = {"source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in reference.iterdir()},
                       "seed": 1, "image_size": [32, 32], "bounces": 12,
                       "blender_hash": "test", "radiometry_sha256": "test", "denoised": False,
                       "samples": 16, "geometry_alignment_passed": True, "views": [{"index": 0, "seed": 7}]}
        self.report["source_sha256"]["../manifest.json"] = hashlib.sha256((self.native / "manifest.json").read_bytes()).hexdigest()
        for directory, offset in ((self.cycles, 0), (self.repeat, 1)):
            directory.mkdir()
            report = dict(self.report, views=[{"index": 0, "seed": 7 + offset}])
            (directory / "report.json").write_text(json.dumps(report))
            radiance.tofile(directory / "view_00.rgba32f")

    def tearDown(self):
        self.temporary.cleanup()

    def test_calibrated_exposure_and_independent_convergence(self):
        result = compare(self.native, self.cycles, self.root / "comparison", self.repeat)
        view = result["views"][0]
        self.assertLess(view["metrics"]["all_pixels"]["rgb_relative_mae"], 1e-6)
        self.assertTrue(view["reference_convergence"]["coarse_convergence_passed"])

    def test_modified_geometry_is_rejected(self):
        (self.native / "reference/geometry.bin").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "source identity"):
            compare(self.native, self.cycles, self.root / "comparison")

    def test_engine_revision_reuse_requires_identical_optical_inputs(self):
        original = self.root / "original_source/reference"
        shutil.copytree(self.native / "reference", original)
        shutil.copyfile(self.native / "manifest.json", original.parent / "manifest.json")
        scene_path = self.native / "reference/scene.json"
        scene = json.loads(scene_path.read_text())
        scene["capture_engine"] = "new_engine"
        scene_path.write_text(json.dumps(scene))
        with self.assertRaisesRegex(ValueError, "source identity"):
            compare(self.native, self.cycles, self.root / "mismatched")
        result = compare(self.native, self.cycles, self.root / "reused", reference_source=original)
        self.assertTrue(result["reference_reuse"])
        scene["ev100"] += 1
        scene_path.write_text(json.dumps(scene))
        with self.assertRaisesRegex(ValueError, "identical geometry"):
            compare(self.native, self.cycles, self.root / "changed_exposure", reference_source=original)

    def test_optical_identity_resolves_reordered_mesh_and_material_tables(self):
        directory = self.native / "reference"
        scene = json.loads((directory / "scene.json").read_text())
        geometry = (directory / "geometry.bin").read_bytes()
        changed = bytearray(geometry)
        changed[:4] = np.float32(.25).tobytes()
        (directory / "geometry.bin").write_bytes(geometry + changed)
        scene["geometry_bytes"] *= 2
        scene["meshes"].append(dict(scene["meshes"][0], offset=len(geometry)))
        scene["materials"].append({"roughness": .8})
        scene["instances"].append({"mesh": 1, "material": 1})
        expected = optical_identity(scene, directory)
        scene["meshes"].reverse()
        scene["materials"].reverse()
        scene["instances"] = [{"mesh": 1, "material": 1}, {"mesh": 0, "material": 0}]
        self.assertEqual(expected, optical_identity(scene, directory))
        scene["instances"][0]["material"] = 0
        self.assertNotEqual(expected, optical_identity(scene, directory))

    def test_animated_geometry_cannot_be_reused_across_times(self):
        document = {"cameras": [{"time": 0}, {"time": 1}]}
        validate_snapshot(document, {"humans": []})
        with self.assertRaisesRegex(ValueError, "separate geometry snapshots"):
            validate_snapshot(document, {"humans": [{"id": 1}]})

    def test_unpinned_sky_context_is_rejected(self):
        del self.report["source_sha256"]["../manifest.json"]
        (self.cycles / "report.json").write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "pin the manifest"):
            compare(self.native, self.cycles, self.root / "comparison")

    def test_correlated_reference_is_rejected(self):
        (self.repeat / "report.json").write_text(json.dumps(self.report))
        with self.assertRaisesRegex(ValueError, "different sampling seed"):
            compare(self.native, self.cycles, self.root / "comparison", self.repeat)

    def test_denoised_reference_is_not_convergence_evidence(self):
        (self.repeat / "report.json").write_text(json.dumps(dict(self.report, denoised=True)))
        with self.assertRaisesRegex(ValueError, "raw, independently"):
            compare(self.native, self.cycles, self.root / "comparison", self.repeat)


if __name__ == "__main__":
    unittest.main()
