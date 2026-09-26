"""CPU export regressions; requires the built extension and dataloader dependencies."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from bevy_zeroverse_dataloader import Sample, View, chunk_and_save, load_chunk, ChunkedIteratorDataset, BevyZeroverseDataset, save_to_folders, FolderDataset


def sample(seed):
    view = View(
        color=np.full((4, 4, 4), 0.18, dtype=np.float32),
        depth=None, normal=None, optical_flow=None, position=None,
        world_from_view=np.eye(4), fovy=1.0, near=0.1, far=50.0, time=0.0,
        width=4, height=4, semantic=np.full((4, 4, 4), 0.25, dtype=np.float32),
    )
    return Sample(
        views=[view], view_dim=1, aabb=[[-1] * 3, [1] * 3], object_obbs=[],
        human_poses=[], human_pose_steps=[], human_bone_names=[], human_bone_parents=[],
        indoor_manifest=json.dumps({'seed': seed}), color_encoding='tonemapped_linear',
        annotation_precision='float32_geometry',
        indoor_render_metadata=json.dumps({'quality': 'auto', 'gi_rays': 32}),
    ).to_tensors()


class IndoorExportTests(unittest.TestCase):
    def test_fixed_transfer_keeps_constant_gray(self):
        tensors = sample(3)
        self.assertTrue(torch.allclose(tensors['color'], torch.full_like(tensors['color'], 0.4613561)))
        self.assertTrue(torch.all(tensors['semantic'] == 0.25))
        self.assertEqual(json.loads(bytes(tensors['indoor_manifest'].tolist())), {'seed': 3})
        self.assertEqual(tensors['color_encoding'].item(), 2)
        self.assertEqual(tensors['annotation_precision'].item(), 1)

    def test_variable_length_manifests_roundtrip_in_chunk(self):
        samples = [sample(3), sample(123456789)]
        with tempfile.TemporaryDirectory() as output:
            paths = chunk_and_save(samples, Path(output), samples_per_chunk=2,
                                   color_codec='raw', compression=None,
                                   persistent_workers=False, n_workers=0)
            loaded = load_chunk(paths[0])
        self.assertEqual([json.loads(bytes(t.tolist()))['seed'] for t in loaded['indoor_manifest']], [3, 123456789])
        self.assertTrue(torch.allclose(loaded['color'][0], samples[0]['color']))
        self.assertTrue(torch.equal(loaded['semantic'][1], samples[1]['semantic']))
        self.assertEqual(loaded['color_encoding'].tolist(), [2, 2])
        self.assertEqual(loaded['annotation_precision'].tolist(), [1, 1])
        self.assertEqual(json.loads(bytes(loaded['indoor_render_metadata'][0].tolist())), {'quality': 'auto', 'gi_rays': 32})

    def test_chunk_iteration_keeps_all_samples_with_unequal_chunks(self):
        with tempfile.TemporaryDirectory() as output:
            chunk_and_save([sample(i) for i in range(5)], Path(output), samples_per_chunk=3,
                           color_codec='raw', compression=None, persistent_workers=False, n_workers=0)
            dataset = ChunkedIteratorDataset(Path(output))
            seeds = []
            for worker_id in range(3):
                worker = SimpleNamespace(id=worker_id, num_workers=3)
                with patch('bevy_zeroverse_dataloader.get_worker_info', return_value=worker):
                    seeds.extend(json.loads(bytes(item['indoor_manifest'].tolist()))['seed'] for item in dataset)
            self.assertEqual(sorted(seeds), list(range(5)))

    def test_mixed_empty_and_populated_human_chunks_keep_rows_and_ids(self):
        samples = [sample(0), sample(1), sample(2)]
        for item, count in zip(samples, [0, 2, 1]):
            item['human_count'] = torch.tensor(count, dtype=torch.int64)
            item['human_instance_ids'] = torch.arange(10, 10 + count)
            if count:
                item['human_pose_position'] = torch.ones((1, count, 21, 3)) * count
                item['human_pose_rotation'] = torch.zeros((1, count, 21, 4))
        with tempfile.TemporaryDirectory() as output:
            paths = chunk_and_save(samples, Path(output), samples_per_chunk=3, color_codec='raw',
                                   compression=None, persistent_workers=False, n_workers=0)
            decoded = load_chunk(paths[0])
        self.assertEqual(decoded['human_count'].tolist(), [0, 2, 1])
        self.assertEqual(decoded['human_instance_ids'].tolist(), [[-1, -1], [10, 11], [10, -1]])
        self.assertEqual(tuple(decoded['human_pose_position'].shape), (3, 1, 2, 21, 3))
        self.assertTrue(torch.all(decoded['human_pose_position'][0] == 0))
        self.assertTrue(torch.all(decoded['human_pose_position'][1] == 2))

    def test_obb_classes_remap_across_ragged_sample_dictionaries(self):
        samples = [sample(0), sample(1), sample(2)]
        for item, names in zip(samples, [[], ['chair', 'person'], ['person', 'desk']]):
            if names:
                item['object_obb_center'] = torch.zeros((len(names), 3))
                item['object_obb_scale'] = torch.ones((len(names), 3))
                item['object_obb_rotation'] = torch.zeros((len(names), 4))
                item['object_obb_class_idx'] = torch.arange(len(names))
                item['object_obb_class_names'] = torch.tensor(list(json.dumps(names).encode()), dtype=torch.uint8)
        with tempfile.TemporaryDirectory() as output:
            paths = chunk_and_save(samples, Path(output), samples_per_chunk=3, color_codec='raw',
                                   compression=None, persistent_workers=False, n_workers=0)
            decoded = load_chunk(paths[0])
        names = json.loads(bytes(decoded['object_obb_class_names'].tolist()))
        actual = [[names[index] for index in row.tolist() if index >= 0] for row in decoded['object_obb_class_idx']]
        self.assertEqual(actual, [[], ['chair', 'person'], ['person', 'desk']])

    def test_corrupt_chunks_fail_closed(self):
        with tempfile.TemporaryDirectory() as output:
            Path(output, '000000.safetensors').write_bytes(b'incomplete')
            with self.assertRaises(Exception):
                ChunkedIteratorDataset(Path(output))

    def test_indoor_gi_budget_configuration_rejects_invalid_values(self):
        for rays in [0, 63, 16385, 128.5]:
            with self.assertRaises(ValueError):
                BevyZeroverseDataset(False, True, 1, 4, 4, 1, scene_type='procedural_indoor', indoor_gi_rays=rays)
        dataset = BevyZeroverseDataset(False, True, 1, 4, 4, 1, scene_type='procedural_indoor', indoor_gi_rays=1024)
        self.assertEqual(dataset.indoor_gi_rays, 1024)

    def test_indexed_capture_uses_index_not_worker_stream(self):
        dataset = BevyZeroverseDataset(False, True, 1, 4, 4, 10, scene_type='procedural_indoor',
                                      indoor_seed=100, playback_steps=1, ovoxel_mode='disabled')
        dataset.initialized = True
        def fake_next(indoor_seed):
            return SimpleNamespace(seed=indoor_seed)
        def fake_sample(value, width, height):
            return SimpleNamespace(indoor_manifest=json.dumps({'seed': value.seed}), to_tensors=lambda: value.seed)
        with patch('bevy_zeroverse_dataloader.bevy_zeroverse_ffi.next', side_effect=fake_next), \
             patch('bevy_zeroverse_dataloader.Sample.from_rust', side_effect=fake_sample):
            self.assertEqual([dataset[index] for index in [4, 0, 4]], [104, 100, 104])
        self.assertEqual(dataset.depth_format, 'linear')

    def test_folder_labels_preserve_all_rows_channels_and_metadata(self):
        item = sample(17)
        item['depth'] = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4, 1)
        item['normal'] = torch.arange(48, dtype=torch.float32).reshape(1, 1, 4, 4, 3) / 48
        item['position'] = item['normal'].clone()
        with tempfile.TemporaryDirectory() as output:
            save_to_folders([item], Path(output), n_workers=0)
            decoded = FolderDataset(Path(output))[0]
            for name in ['depth', 'normal', 'position', 'semantic', 'indoor_manifest', 'color_encoding', 'annotation_precision', 'indoor_render_metadata']:
                self.assertTrue(torch.equal(item[name], decoded[name]), name)
            with self.assertRaises(FileExistsError):
                save_to_folders([item], Path(output), n_workers=0)


if __name__ == '__main__':
    unittest.main()
