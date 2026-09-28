"""Analytic safeguards for sampling-error measurements, independent of screenshots."""
import unittest
import numpy as np
from compare_indoor_glass import errors, erode


class GlassErrorTests(unittest.TestCase):
    def test_constant_exposure_bias_is_not_called_noise_free(self):
        reference = np.ones((9, 11, 3))
        stats, _ = errors(reference*.8, reference, np.ones((9,11),dtype=bool))
        self.assertAlmostEqual(stats['relative_rmse'],.2)
        self.assertAlmostEqual(stats['relative_bias'],-.2)

    def test_denoising_to_a_constant_does_not_erase_geometric_error(self):
        reference = np.zeros((8,8,3)); reference[:,4:] = 1
        stats, _ = errors(np.full_like(reference,.5),reference,np.ones((8,8),dtype=bool))
        self.assertAlmostEqual(stats['relative_rmse'],1)
        self.assertAlmostEqual(stats['relative_bias'],0)

    def test_mask_excludes_silhouette_and_image_boundaries(self):
        mask=np.ones((7,9),dtype=bool)
        self.assertEqual(erode(mask).sum(),35)
        mask[3,4]=False
        self.assertEqual(erode(mask).sum(),26)


if __name__=='__main__': unittest.main()
