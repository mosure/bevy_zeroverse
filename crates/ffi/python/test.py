from pathlib import Path
import shutil
import time

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import unittest
import numpy as np

try:
    import psutil
except ImportError:  # pragma: no cover - optional dependency
    psutil = None

from bevy_zeroverse_dataloader import BevyZeroverseDataset, \
    ChunkedIteratorDataset, FolderDataset, MP4Dataset, \
    chunk_and_save, load_chunk, save_to_folders, save_to_mp4, write_sample, Sample, View

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import unittest
import numpy as np

from bevy_zeroverse_dataloader import BevyZeroverseDataset, \
    ChunkedIteratorDataset, FolderDataset, MP4Dataset, \
    chunk_and_save, load_chunk, save_to_folders, save_to_mp4, write_sample, Sample, View



def visualize(batch):
    print(batch['color'].shape)

    is_chunked = len(batch['color'].shape) == 6

    color_images = batch['color'].numpy()
    # depth_images = batch['depth'].numpy()
    # normal_images = batch['normal'].numpy()

    if is_chunked:
        color_images = color_images.squeeze(0)
        # depth_images = depth_images.squeeze(0)
        # normal_images = normal_images.squeeze(0)

    batch_size = color_images.shape[0]
    num_cameras = color_images.shape[1]
    num_image_types = 2  # color, depth, normal
    total_images = batch_size * num_cameras * num_image_types

    cols = 9
    rows = (total_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for batch_idx in range(batch_size):
        for cam_idx in range(num_cameras):
            index = (batch_idx * num_cameras + cam_idx) * num_image_types

            color_image = color_images[batch_idx, cam_idx]
            axes[index].imshow(color_image)
            axes[index].axis('off')
            if batch_size <= 2:
                axes[index].set_title(f'({batch_idx + 1}, {cam_idx + 1})_color')

            # depth_image = depth_images[batch_idx, cam_idx]
            # axes[index + 1].imshow(depth_image, cmap='gray')
            # axes[index + 1].axis('off')
            # if batch_size <= 2:
            #     axes[index + 1].set_title(f'({batch_idx + 1}, {cam_idx + 1})_depth')

            # normal_image = normal_images[batch_idx, cam_idx]
            # axes[index + 2].imshow(normal_image)
            # axes[index + 2].axis('off')
            # if batch_size <= 2:
            #     axes[index + 2].set_title(f'({batch_idx + 1}, {cam_idx + 1})_normal')

    for ax in axes[total_images:]:
        ax.axis('off')

    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)

    def on_key(event):
        plt.close(fig)

        if event.key == 'escape':
            exit(0)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()


def benchmark(dataloader):
    import time
    start = time.time()
    count = 0
    for batch in dataloader:
        print('batch shape:', batch['color'].shape)
        count += 1
        if count == 100:
            break
    end = time.time()
    print('seconds per batch:', (end - start) / count)


def test():
    dataset = BevyZeroverseDataset(
        editor=False, headless=True, num_cameras=6,
        width=640, height=480, num_samples=1e6,
        scene_type='room',
    )
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True, num_workers=2)

    # benchmark(dataloader)

    for batch in dataloader:
        visualize(batch)


generated = False
class TestChunkedDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.editor = True
        cls.headless = True
        cls.num_cameras = 4
        cls.width = 640
        cls.height = 480
        cls.num_samples = 10
        cls.bytes_per_chunk = int(256 * 1024 * 1024)
        cls.samples_per_chunk = 3
        cls.stage = "test"
        cls.output_dir = Path("./data/zeroverse") / cls.stage

        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)

        cls.dataset = BevyZeroverseDataset(
            cls.editor,
            cls.headless,
            cls.num_cameras,
            cls.width,
            cls.height,
            cls.num_samples,
            scene_type='room',
        )

        cls.chunk_paths = chunk_and_save(
            cls.dataset,
            cls.output_dir / 'chunk',
            bytes_per_chunk=cls.bytes_per_chunk,
            samples_per_chunk=cls.samples_per_chunk,
            n_workers=0,
        )

        save_to_folders(
            cls.dataset,
            cls.output_dir / 'fs',
            n_workers=0,
        )

        save_to_mp4(
            cls.dataset,
            cls.output_dir / 'mp4',
            n_workers=0,
        )

    @classmethod
    def tearDownClass(cls):
        if cls.output_dir.exists():
            shutil.rmtree(cls.output_dir)

    def setUp(self):
        self.output_dir = type(self).output_dir
        self.chunk_paths = type(self).chunk_paths

    def test_benchmark_chunked_dataloader(self):
        chunked_dataset = ChunkedIteratorDataset(self.output_dir / 'chunk')
        dataloader = DataLoader(chunked_dataset, batch_size=1, shuffle=False)

        print("\nBenchmarking chunks:")
        benchmark(dataloader)

    def test_benchmark_folder_dataloader(self):
        chunked_dataset = FolderDataset(self.output_dir / 'fs')
        dataloader = DataLoader(chunked_dataset, batch_size=1, shuffle=False)

        print("\nBenchmarking folder:")
        benchmark(dataloader)

    def test_benchmark_mp4_dataloader(self):
        chunked_dataset = MP4Dataset(self.output_dir / 'mp4')
        dataloader = DataLoader(chunked_dataset, batch_size=1, shuffle=False)

        print("\nBenchmarking mp4:")
        benchmark(dataloader)

    def test_update_sample_in_chunk(self):
        """
        ensure `write_sample` edits exactly one sample in-place, including jpeg
        payloads, without disturbing its neighbours.
        """
        chunked_ds = ChunkedIteratorDataset(self.output_dir / 'chunk', shuffle=False)
        sample = next(iter(chunked_ds))

        chunk_path = Path(sample["_chunk_path"])
        sample_idx = sample["_sample_idx"]

        chunk_before = load_chunk(chunk_path)
        B = chunk_before["color"].shape[0]

        self.assertGreaterEqual(B, 1, "chunk unexpectedly empty")

        other_idx = (sample_idx + 1) % B if B > 1 else None
        if other_idx is not None:
            other_color_before = chunk_before["color"][other_idx].clone()

        new_color = torch.zeros_like(sample["color"])
        mod_sample = {
            **{k: v for k, v in sample.items() if k[0] != "_"},
            "color": new_color,
            "_chunk_path": str(chunk_path),
            "_sample_idx": sample_idx,
        }

        write_sample(mod_sample)

        chunk_after = load_chunk(chunk_path)
        updated_color = chunk_after["color"][sample_idx]

        self.assertLess(
            updated_color.mean().item(),
            0.05,
            "updated sample did not change as expected",
        )

        if other_idx is not None:
            other_color_after = chunk_after["color"][other_idx]
            self.assertTrue(
                torch.equal(other_color_before, other_color_after),
                "neighbouring sample was unintentionally modified",
            )

    def test_chunk_prefetch_overlap(self):
        if len(self.chunk_paths) < 2:
            self.skipTest("need at least two chunks to test prefetch")

        load_starts = {}

        def delayed_load(path: Path):
            load_starts[str(path)] = time.time()
            time.sleep(0.05)
            return load_chunk(path)

        chunked_ds = ChunkedIteratorDataset(
            self.output_dir / 'chunk',
            shuffle=False,
            prefetch_chunks=1,
            load_chunk_fn=delayed_load,
        )

        first_chunk = None
        first_chunk_end_time = None
        second_chunk = None

        for sample in chunked_ds:
            if first_chunk is None:
                first_chunk = sample["_chunk_path"]
            elif sample["_chunk_path"] != first_chunk:
                second_chunk = sample["_chunk_path"]
                first_chunk_end_time = time.time()
                break

        self.assertIsNotNone(second_chunk, "did not advance to a second chunk")
        self.assertIn(second_chunk, load_starts, "prefetch did not start next chunk load")
        self.assertLess(
            load_starts[second_chunk],
            first_chunk_end_time,
            "next chunk load did not start before finishing current chunk iteration",
        )

    def test_chunk_cache_local(self):
        cache_dir = self.output_dir / "chunk_cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)

        def load_from_cache(path: Path):
            self.assertEqual(path.parent.resolve(), cache_dir.resolve())
            return load_chunk(path)

        chunked_ds = ChunkedIteratorDataset(
            self.output_dir / 'chunk',
            shuffle=False,
            prefetch_chunks=0,
            load_chunk_fn=load_from_cache,
            cache_dir=cache_dir,
        )

        sample = next(iter(chunked_ds))
        cached_path = cache_dir / Path(sample["_chunk_path"]).name
        self.assertTrue(cached_path.exists(), "chunk was not cached locally")



class TestPoseTensorization(unittest.TestCase):
    def test_multistep_pose_shapes(self):
        world = np.eye(4, dtype=np.float32)
        views = [
            View(
                None,
                None,
                None,
                None,
                None,
                world,
                fovy=1.0,
                near=0.1,
                far=10.0,
                time=0.0,
                width=1,
                height=1,
            ),
            View(
                None,
                None,
                None,
                None,
                None,
                world,
                fovy=1.0,
                near=0.1,
                far=10.0,
                time=0.5,
                width=1,
                height=1,
            ),
        ]
        human_pose_steps = [
            [
                {
                    "bone_positions": np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32),
                    "bone_rotations": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                },
                {
                    "bone_positions": np.array([[0.5, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=np.float32),
                    "bone_rotations": np.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                },
            ],
            [
                {
                    "bone_positions": np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
                    "bone_rotations": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                },
                {
                    "bone_positions": np.array([[3.0, 0.0, 0.0]], dtype=np.float32),
                    "bone_rotations": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float32),
                },
            ],
        ]
        sample = Sample(
            views,
            view_dim=1,
            aabb=np.zeros((2, 3), dtype=np.float32),
            object_obbs=[],
            human_poses=[],
            human_pose_steps=human_pose_steps,
            human_bone_names=["root", "spine"],
            human_bone_parents=[-1, 0],
            ovoxel=None,
        )
        tensors = sample.to_tensors()
        self.assertEqual(tuple(tensors["human_pose_position"].shape), (2, 2, 2, 3))
        self.assertEqual(tuple(tensors["human_pose_rotation"].shape), (2, 2, 2, 4))


class TestMemoryUsage(unittest.TestCase):
    def test_chunk_generation_memory_stability(self):
        if psutil is None:
            self.skipTest("psutil is required for memory checks")

        proc = psutil.Process()
        baseline = proc.memory_info().rss

        root = Path("./data/zeroverse/memory_test")
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

        runs = 3
        for idx in range(runs):
            run_dir = root / f"run_{idx}"
            dataset = BevyZeroverseDataset(
                editor=False,
                headless=True,
                num_cameras=2,
                width=640,
                height=480,
                num_samples=6,
                scene_type='room',
            )
            chunk_and_save(
                dataset,
                run_dir,
                bytes_per_chunk=64 * 1024 * 1024,
                samples_per_chunk=3,
                n_workers=0,
                full_size_only=False,
                memory_cleanup=True,
            )
            time.sleep(0.1)

        final = proc.memory_info().rss
        max_allowed = 512 * 1024 * 1024
        self.assertLess(
            final - baseline,
            max_allowed,
            f"RSS increased by {(final - baseline) / 1e6:.2f} MB which exceeds the allowed threshold",
        )

def main():
    unittest.main()
    # test()

if __name__ == "__main__":
    main()
