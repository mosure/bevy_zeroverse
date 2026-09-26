import math
import unittest
from qualify_indoor_stability import blocked_trend, align_gpu_memory


class StabilityTests(unittest.TestCase):
    def test_noise_is_not_mistaken_for_growth(self):
        values = [1_500_000_000 + 20_000_000 * math.sin(i * .37) for i in range(2048)]
        self.assertLess(blocked_trend(values)["approximate_95pct_upper_slope"], 8192)

    def test_linear_leak_and_late_step_are_detected(self):
        for values in ([1_000_000_000 + i * 100_000 for i in range(2048)],
                       [1_000_000_000 + (200_000_000 if i >= 1500 else 0) for i in range(2048)]):
            self.assertGreater(blocked_trend(values)["approximate_95pct_upper_slope"], 8192)

    def test_short_windows_are_rejected(self):
        with self.assertRaises(ValueError):
            blocked_trend([1_000_000_000] * 128)

    def test_gpu_memory_uses_only_recent_preceding_observations(self):
        scenes = [{"completed_unix_seconds": t} for t in (1.1, 1.3, 1.6)]
        polls = [{"wall_unix_seconds": t, "process_gpu_memory_bytes": n}
                 for t, n in ((1, 100), (1.5, 200), (2, 300))]
        self.assertEqual(align_gpu_memory(scenes, polls, .6), [100, 100, 200])
        with self.assertRaisesRegex(ValueError, "missing or stale"):
            align_gpu_memory(scenes, polls, .2)
        polls[0]["process_gpu_memory_bytes"] = None
        with self.assertRaisesRegex(ValueError, "unavailable"):
            align_gpu_memory(scenes, polls, .6)


if __name__ == "__main__":
    unittest.main()
