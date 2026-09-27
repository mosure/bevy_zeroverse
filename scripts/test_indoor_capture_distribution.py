import math
from pathlib import Path
import tempfile
import unittest

from indoor_capture_distribution import object_rotations, repetition, view_flags, wilson


class CaptureDistributionTests(unittest.TestCase):
    def test_rotations_wrap_and_exclude_neighbor_objects(self):
        objects = [dict(kind='Chair', yaw=math.radians(a), neighbor=False) for a in [-15, 375, 180, 225]]
        objects.append(dict(kind='Chair', yaw=0, neighbor=True))
        rows, stats = object_rotations([(None, None, dict(seed=1, objects=objects))])
        self.assertEqual([r['off_cardinal_degrees'] for r in rows], [15, 15, 0, 45])
        self.assertEqual(stats['Chair']['instances'], 4)
        self.assertEqual(sum(stats['Chair']['yaw_bins_15_degrees']), 4)

    def test_scene_confidence_includes_zero_failure_uncertainty(self):
        low, high = wilson(0, 128)
        self.assertAlmostEqual(low, 0)
        self.assertGreater(high, .02)
        self.assertLess(high, .04)
        self.assertAlmostEqual(wilson(64, 128)[0], 1-wilson(64, 128)[1])
        self.assertIsNone(wilson(0, 0))
        self.assertGreater(1-math.pow(.05, 1/128), .02)

    def test_review_flags_preserve_intentional_dark_and_semantic_tails(self):
        flags = view_flags({'semantic_pixel_counts': {'wall': 96, 'floor': 4},
                            'dark_fraction': .96, 'clipped_fraction': .02}, 100)
        self.assertEqual(flags, {'mostly_dark': True, 'substantial_clipping': False,
                                 'single_class_dominance': True,
                                 'few_semantic_classes': True})

    def test_repetition_excludes_temporal_pairs_and_same_scene(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as tmp:
            paths = [Path(tmp)/f'{i}.png' for i in range(4)]
            for p in paths:
                Image.new('RGB', (20,20), '#445566').save(p)
            images = [(paths[0], {'seed':1, 'time':0}), (paths[1], {'seed':1, 'time':1}),
                      (paths[2], {'seed':2, 'time':0}), (paths[3], {'seed':2, 'time':0})]
            result = repetition(images)
            self.assertEqual(result['compared_views'], 3)
            self.assertEqual(result['exact_duplicate_image_groups_across_scenes'], 1)
            self.assertTrue(all(p['left'] == str(paths[0]) for p in result['closest_pairs']))
            self.assertNotIn(str(paths[1]), str(result))


if __name__ == '__main__':
    unittest.main()
