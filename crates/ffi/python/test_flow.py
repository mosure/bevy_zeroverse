"""Numeric flow must survive the live dataloader and both dataset codecs."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from bevy_zeroverse_dataloader import Sample, View, chunk_and_save, load_chunk, load_single_sample, save_to_folders, FolderDataset, BevyZeroverseDataset
from bevy_zeroverse_dataloader import flow


def sample():
    views = []
    for t in range(2):
        for camera in range(2):
            pixels = np.zeros((5, 7, 4), dtype=np.float32)
            if t == 0:
                pixels[:] = [-17.125 - camera, 0.000123, 1, 1]
                pixels[0, 0] = 0
                pixels[1, 1, 3] = 0
            normalized = pixels.copy()
            normalized[..., 0] /= 7
            normalized[..., 1] /= 5
            views.append(View(color=None, depth=None, normal=None, position=None,
                              optical_flow=pixels, motion_vectors=normalized,
                              world_from_view=np.eye(4), fovy=1., near=.1, far=50.,
                              time=t*.25, width=7, height=5))
    return Sample(views, 2, [[-1]*3, [1]*3], [], [], [], [], []).to_tensors()


class FlowTests(unittest.TestCase):
    def test_live_conversion_keeps_sign_precision_and_masks(self):
        item = sample()
        self.assertEqual(item['optical_flow'].shape, (2, 2, 5, 7, 2))
        self.assertEqual(item['optical_flow'][0, 0, 2, 3, 0].item(), -17.125)
        self.assertAlmostEqual(item['optical_flow'][0, 0, 2, 3, 1].item(), .000123)
        self.assertTrue(torch.equal(item['motion_vectors'], item['optical_flow'] / torch.tensor([7., 5.])))
        self.assertEqual(item['optical_flow_valid'].dtype, torch.uint8)
        self.assertEqual(item['optical_flow_valid'][0, 0, 1, 1].item(), 1)
        self.assertEqual(item['optical_flow_visible'][0, 0, 1, 1].item(), 0)
        self.assertTrue(torch.all(item['optical_flow_valid'][1] == 0))

    def test_chunk_and_folder_preserve_numeric_flow_without_rgb(self):
        original = sample()
        with tempfile.TemporaryDirectory() as root:
            paths = chunk_and_save([original, sample()], Path(root)/'chunks', samples_per_chunk=2,
                                   color_codec='raw', compression=None, persistent_workers=False, n_workers=0)
            chunk = load_chunk(paths[0])
            decoded = load_single_sample(paths[0], 1)
            self.assertEqual(json.loads(bytes(chunk['flow_metadata'].tolist()))['schema_version'], 1)
            save_to_folders([original], Path(root)/'folders', n_workers=0)
            folder = FolderDataset(Path(root)/'folders')[0]
            for name in flow.NAMES:
                for suffix in ('', '_valid', '_visible'):
                    key = name + suffix
                    self.assertTrue(torch.equal(original[key], decoded[key]), key)
                    self.assertTrue(torch.equal(original[key], folder[key]), key)
            with np.load(Path(root)/'folders/000000/optical_flow_000_00.npz') as archive:
                self.assertEqual(archive['optical_flow'].shape, (5, 7, 4))

    def test_reject_colored_flow_and_invalid_masks(self):
        with self.assertRaises(ValueError):
            flow.from_rgba('optical_flow', np.full((2, 2, 4), .5, np.float32))
        item = sample()
        item['optical_flow_visible'][0, 0, 0, 0] = 1
        with self.assertRaises(ValueError):
            flow.validate(item)

    def test_indoor_accepts_both_temporal_annotations(self):
        data = BevyZeroverseDataset(False, True, 1, 7, 5, 1, scene_type='procedural_indoor',
                                   render_modes=['optical_flow', 'motion_vectors'])
        self.assertEqual(data.render_modes, ['optical_flow', 'motion_vectors'])


if __name__ == '__main__':
    unittest.main()
