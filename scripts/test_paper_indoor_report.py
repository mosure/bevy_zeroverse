import unittest
from paper_indoor_report import entropy_count, discrete_distribution, matched_speedup, primary_occupied


class PaperMetricsTests(unittest.TestCase):
    def test_entropy_and_zero_preserving_counts(self):
        self.assertEqual(entropy_count([]), 0)
        self.assertEqual(entropy_count([0, 5]), 1)
        self.assertAlmostEqual(entropy_count([4, 4, 4]), 3)
        with self.assertRaises(ValueError):
            entropy_count([1, -1])
        result = discrete_distribution({'0': 3, '4': 1})
        self.assertEqual(result['mean'], 1)
        self.assertEqual(sum(result['bin_counts']), 4)

    def test_speedup_requires_matched_completed_work(self):
        before = [dict(seed=i, views=4, elapsed_seconds=2) for i in range(3)]
        after = [dict(seed=i, views=4, elapsed_seconds=1) for i in range(3)]
        self.assertEqual(matched_speedup(before, after, 1), 50)
        with self.assertRaises(ValueError):
            matched_speedup(before, list(reversed(after)), 1)
        after[2]['views'] = 2
        with self.assertRaises(ValueError):
            matched_speedup(before, after, 1)

    def test_people_in_secondary_zones_are_not_primary_occupancy(self):
        manifest = dict(program={'zones': [dict(min=[-2, -2], max=[0, 2]),
                                          dict(min=[0, -2], max=[2, 2])]},
                        humans=[dict(position=[-1, 0, 0], neighbor=False)])
        self.assertFalse(primary_occupied(manifest))
        manifest['humans'][0]['position'] = [1, 0, 0]
        self.assertTrue(primary_occupied(manifest))
        manifest['humans'][0]['neighbor'] = True
        self.assertFalse(primary_occupied(manifest))


if __name__ == '__main__':
    unittest.main()
