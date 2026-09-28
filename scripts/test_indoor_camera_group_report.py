import unittest
import numpy as np
from indoor_camera_group_report import group_geometry


class CameraGroups(unittest.TestCase):
    def test_lines_and_square_and_crossing(self):
        line = np.array([[[0, 0, 0], [1, 0, 0], [2, 0, 0]],
                         [[0, 0, 1], [1, 0, 1], [2, 0, 1]]])
        result = group_geometry(line)
        self.assertEqual(result['min_horizontal_spread'], 0)
        self.assertEqual(result['min_relative_motion'], 0)
        square = [[[0, 0, 0], [1, 0, 0], [0, 0, 1], [1, 0, 1]]]
        result = group_geometry(square)
        self.assertEqual(result['min_horizontal_spread'], 1)
        self.assertEqual(result['min_pairwise_baseline_m'], 1)
        self.assertIsNone(result['min_relative_motion'])
        self.assertEqual(group_geometry([[[0, 0, 0], [0, 0, 0]]])['min_pairwise_baseline_m'], 0)

    def test_independent_motion_and_invalid_data(self):
        result = group_geometry([[[0, 0, 0], [1, 0, 0]], [[0, 0, 1], [2, 0, 0]]])
        self.assertAlmostEqual(result['min_relative_motion'], 2**.5)
        self.assertIsNone(result['min_horizontal_spread'])
        for data in ([], [[[0, 0, 0]]], [[[0, 0, float('nan')], [0, 0, 0]]]):
            with self.assertRaises(ValueError):
                group_geometry(data)


if __name__ == '__main__':
    unittest.main()
