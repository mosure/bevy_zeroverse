import unittest
import numpy as np
from indoor_covisibility_report import mask_metrics, connected
from indoor_multiview_report import shared_pixels
from test_indoor_multiview_report import camera


class CoVisibilityReport(unittest.TestCase):
    def test_membership_counts_exclude_background_and_self(self):
        masks = np.array([[0, 2, 6, 14, 0]], dtype=np.uint16)
        valid = np.array([[True, True, True, True, False]])
        result = mask_metrics(masks, valid, 0, 4)
        self.assertEqual(result['cardinality_counts'], [1, 1, 1, 1])
        self.assertEqual(result['any_other_fraction_all_pixels'], 3/5)
        self.assertEqual(result['any_other_fraction_valid_pixels'], 3/4)
        self.assertEqual(result['all_others_fraction_valid_pixels'], 1/4)
        self.assertEqual(result['mean_other_cameras_valid_pixels'], 1.5)
        for value in (1, 16):
            bad = masks.copy(); bad[0, 0] = value
            with self.assertRaises(ValueError):
                mask_metrics(bad, valid, 0, 4)
        bad = masks.copy(); bad[0, -1] = 2
        with self.assertRaises(ValueError):
            mask_metrics(bad, valid, 0, 4)
        empty = mask_metrics(np.zeros((2, 2), dtype=np.uint16), np.zeros((2, 2), dtype=bool), 0, 4)
        self.assertIsNone(empty['any_other_fraction_valid_pixels'])

    def test_analytic_three_camera_plane_and_graph(self):
        depth = np.full((20, 40), 4.)
        masks = np.zeros(depth.shape, dtype=np.uint16)
        for i, x in enumerate((-2, 2), 1):
            shared, _, valid = shared_pixels(depth, camera(), depth, camera(x=x))
            masks[shared] |= 1 << i
        stats = mask_metrics(masks, valid, 0, 3)
        self.assertEqual(stats['cardinality_counts'], [0, 400, 400])
        self.assertEqual(stats['any_other_fraction_valid_pixels'], 1)
        self.assertEqual(stats['mean_other_cameras_valid_pixels'], 1.5)
        pairs = {(0,1):.5,(1,0):.5,(1,2):.5,(2,1):.5,(0,2):0,(2,0):0}
        self.assertTrue(connected(pairs, 3, .35))
        self.assertFalse(connected(pairs, 3, .6))


if __name__ == '__main__':
    unittest.main()
