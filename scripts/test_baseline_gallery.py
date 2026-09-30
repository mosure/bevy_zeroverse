"""Protect the denominator and matched-scene identity of the visual comparison."""
import copy
import unittest

import numpy as np

from build_baseline_gallery import scene_digest, visibility_stats


class BaselineGallery(unittest.TestCase):
    def test_visibility_counts_valid_unshared_pixels_and_excludes_background(self):
        masks = np.array([[2,14,0], [1,0,0], [0,0,0], [0,0,0]], dtype=np.uint16)
        valid = np.array([[1,1,1], [1,1,0], [0,0,0], [0,0,0]], dtype=bool)
        stats = visibility_stats(masks, valid)
        self.assertEqual(stats["valid_pixels"], 5)
        self.assertEqual(stats["peer_count_pixels"], [2,2,0,1])
        self.assertAlmostEqual(stats["shared_any"], .6)
        self.assertAlmostEqual(stats["shared_all"], .2)

    def test_source_bit_and_nonempty_background_are_rejected(self):
        for mask, valid in ((1,True), (2,False)):
            masks = np.zeros((4,2),dtype=np.uint16)
            validity = np.ones((4,2),dtype=bool)
            masks[0,0], validity[0,0] = mask, valid
            with self.assertRaises(AssertionError):
                visibility_stats(masks,validity)

    def test_camera_changes_do_not_hide_scene_changes(self):
        source = dict(seed=10,program={"materials":{"glass":.2}},humans=[{"pose":1}],
                      lighting={"sun_lux":100},objects=[{"yaw":.3}],
                      cameras=[0],camera_settings={"baseline":.5},camera_aspect_ratio=1.6)
        changed = copy.deepcopy(source)
        changed.update(cameras=[1],camera_settings={"baseline":1})
        self.assertEqual(scene_digest(source),scene_digest(changed))
        for key in ("program","humans","lighting","objects"):
            altered = copy.deepcopy(changed)
            altered[key] = None
            self.assertNotEqual(scene_digest(source),scene_digest(altered),key)


if __name__ == "__main__":
    unittest.main()
