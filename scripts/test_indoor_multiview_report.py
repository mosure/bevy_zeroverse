import unittest
import numpy as np
from indoor_multiview_report import shared_pixels


def camera(x=0, fx=20):
    pose = np.eye(4)
    pose[0, 3] = x
    return dict(world_from_view=pose.T.tolist(), fx_pixels=fx, fy_pixels=fx, near=.1, far=50)


class MultiViewTests(unittest.TestCase):
    def test_identity_and_parallel_stereo_plane(self):
        depth = np.full((20, 40), 4.)
        same, angles, valid = shared_pixels(depth, camera(), depth, camera())
        self.assertTrue(same.all())
        self.assertTrue(valid.all())
        self.assertLess(angles.max(), .000002)
        # At z=4, a 2m baseline and focal=20px shifts exactly 10 pixels.
        visible, angles, _ = shared_pixels(depth, camera(), depth, camera(x=2))
        self.assertAlmostEqual(visible.mean(), .75)
        self.assertTrue(np.all(angles[visible] > 10))

    def test_occlusion_background_and_disjoint_views(self):
        depth = np.full((20, 40), 4.)
        foreground = depth.copy()
        foreground[:, :20] = 2.
        visible, _, _ = shared_pixels(depth, camera(), foreground, camera())
        self.assertAlmostEqual(visible.mean(), .5)
        foreground[:, :20] = 0.
        visible, _, _ = shared_pixels(depth, camera(), foreground, camera())
        self.assertAlmostEqual(visible.mean(), .5)
        opposite = camera()
        opposite['world_from_view'][0][0] = -1.
        opposite['world_from_view'][2][2] = -1.
        self.assertFalse(shared_pixels(depth, camera(), depth, opposite)[0].any())

    def test_fov_asymmetry_and_invalid_calibration(self):
        depth = np.full((20, 40), 4.)
        wide_to_narrow = shared_pixels(depth, camera(), depth, camera(fx=40))[0]
        narrow_to_wide = shared_pixels(depth, camera(fx=40), depth, camera())[0]
        self.assertAlmostEqual(wide_to_narrow.mean(), .25)
        self.assertAlmostEqual(narrow_to_wide.mean(), 1.)
        with self.assertRaises(ValueError):
            shared_pixels(depth, camera(fx=0), depth, camera())
        with self.assertRaises(ValueError):
            shared_pixels(depth*np.nan, camera(), depth, camera())


if __name__ == '__main__':
    unittest.main()
