"""Camera contract parity for live, chunk and folder paths (no GPU needed)."""
import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from bevy_zeroverse_dataloader import View, Sample, chunk_and_save, load_single_sample, save_to_folders, FolderDataset
from bevy_zeroverse_dataloader import calibration


def sample():
    c = dict(schema_version=1, lens_model='pinhole', lens_model_version=1,
             pixel_convention='top_left_corner_half_pixel_centers', image_size=[7, 5],
             k=[[6., 0., 3.5], [0., 6., 2.5], [0., 0., 1.]])
    views = [View(color=np.full((5, 7, 4), 0.3, np.float32), depth=None, normal=None,
                  optical_flow=None, position=None, world_from_view=np.eye(4), fovy=1., near=.1,
                  far=50., time=i//2, width=7, height=5, calibration=copy.deepcopy(c),
                  trajectory_progress=i//2, time_seconds=0. if i < 2 else None) for i in range(4)]
    return Sample(views, 2, [[-1]*3, [1]*3], [], [], [], [], [], color_encoding='srgb').to_tensors()


class CalibrationTests(unittest.TestCase):
    def test_full_k_and_unspecified_time_survive_all_export_paths(self):
        item = sample()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for codec in ('raw', 'jpeg'):
                paths = chunk_and_save([item, sample()], root/codec, samples_per_chunk=2,
                                       color_codec=codec, compression=None, persistent_workers=False, n_workers=0)
                restored = load_single_sample(paths[0], 1)
                calibration.validate(restored)
                for name in calibration.NAMES:
                    self.assertTrue(torch.equal(item[name], restored[name]), name)
            save_to_folders([item], root/'folders', n_workers=0)
            restored = FolderDataset(root/'folders')[0]
            for name in calibration.NAMES:
                self.assertTrue(torch.equal(item[name], restored[name]), name)

    def test_invalid_or_partial_labels_are_rejected(self):
        for name in calibration.NAMES:
            item = sample()
            del item[name]
            with self.assertRaises(ValueError):
                calibration.validate(item)
        for name in ('intrinsics', 'trajectory_progress', 'time_seconds'):
            item = sample()
            item[name].flatten()[0] = float('nan')
            with self.assertRaises(ValueError):
                calibration.validate(item)
        item = sample()
        item['image_size'][0, 0, 0] = 8
        with self.assertRaises(ValueError):
            calibration.validate(item)
        calibration.validate({'fovy': torch.zeros((1, 1, 1))})


if __name__ == '__main__':
    unittest.main()
