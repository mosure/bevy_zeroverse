"""Camera membership is integer data, including when RGB uses JPEG."""
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from bevy_zeroverse_dataloader import Sample, View, chunk_and_save, load_single_sample, save_to_folders, FolderDataset
from bevy_zeroverse_dataloader import co_visibility as cv


def metadata(count):
    return {'schema_version': 1, 'camera_count': count,
            'legend': [{'bit': i, 'camera_index': i * 3 + 2, 'mask': 1 << i, 'rgb8': c.tolist()}
                       for i, c in enumerate(cv.palette(count))]}


def sample(count=16):
    views = []
    for t in range(2):
        for camera in range(count):
            mask = ((1 << count) - 1) ^ (1 << camera)
            rgba = np.full((5, 7, 4), [mask, mask.bit_count(), 1, 0], dtype=np.float32)
            rgba[0, 0] = 0
            rgba[0, 1] = [0, 0, 1, 0]
            views.append(View(color=None, depth=None, normal=None, optical_flow=None, position=None,
                              world_from_view=np.eye(4), fovy=1., near=.1, far=50., time=t*.3, width=7, height=5,
                              co_visibility=rgba))
    return Sample(views, count, [[-1]*3, [1]*3], [], [], [], [], [],
                  co_visibility_metadata=json.dumps(metadata(count))).to_tensors()


class CoVisibilityTests(unittest.TestCase):
    def test_every_color_sum_decodes(self):
        for count in range(1, 17):
            masks = np.arange(1 << count, dtype=np.uint16)
            np.testing.assert_array_equal(cv.rgb_to_mask(cv.mask_to_rgb(masks, count), count), masks)

    def test_live_and_both_codecs_keep_all_bits_and_validity(self):
        for count in [1, 3, 16]:
            original = sample(count)
            self.assertEqual(original['co_visibility'].dtype, torch.uint16)
            with tempfile.TemporaryDirectory() as root:
                for codec in ('raw', 'jpeg'):
                    paths = chunk_and_save([original, sample(count)], Path(root)/codec, samples_per_chunk=2,
                                           color_codec=codec, compression=None, persistent_workers=False, n_workers=0)
                    decoded = load_single_sample(paths[0], 1)
                    for name in ('co_visibility', 'co_visibility_valid'):
                        self.assertTrue(torch.equal(original[name], decoded[name]), name)
                    cv.validate(decoded)
                save_to_folders([original], Path(root)/'folders', n_workers=0)
                folder = FolderDataset(Path(root)/'folders')[0]
                for name in ('co_visibility', 'co_visibility_valid'):
                    self.assertTrue(torch.equal(original[name], folder[name]), name)
                path = Path(root)/'folders/000000/co_visibility_000_00.npz'
                path.unlink()
                with self.assertRaises(ValueError):
                    FolderDataset(Path(root)/'folders')[0]

    def test_reject_wrong_source_bits_invalid_codes_and_missing_metadata(self):
        with self.assertRaises(ValueError):
            cv.rgb_to_mask(np.array([[1, 0, 0]], np.uint8), 3)
        for bad in [np.nan, -1., 65536., .5]:
            with self.assertRaises(ValueError):
                cv.from_rgba(np.array([[bad, 0, 1, 0]], np.float32))
        item = sample(3)
        item['co_visibility'][0, 0, 0, 0, 0] = 1
        with self.assertRaises(ValueError):
            cv.validate(item)
        item = sample(3)
        del item['co_visibility_metadata']
        with self.assertRaises(ValueError):
            cv.validate(item)

    def test_incomplete_chunks_are_rejected_before_writing(self):
        original = sample(3)
        incomplete = {k: v for k, v in original.items() if not k.startswith('co_visibility')}
        with tempfile.TemporaryDirectory() as root, self.assertRaises(ValueError):
            chunk_and_save([original, incomplete], Path(root), samples_per_chunk=2,
                           color_codec='raw', compression=None, persistent_workers=False, n_workers=0)


if __name__ == '__main__':
    unittest.main()
