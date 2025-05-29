from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
import unittest

from bevy_zeroverse_dataloader import BevyZeroverseDataset, \
    ChunkedIteratorDataset, FolderDataset, MP4Dataset, \
    chunk_and_save, load_chunk, save_to_folders, save_to_mp4, write_sample



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
    def setUp(self):
        self.editor = True
        self.headless = True
        self.num_cameras = 4
        self.width = 640
        self.height = 480
        self.num_samples = 10
        self.bytes_per_chunk = int(256 * 1024 * 1024)
        self.samples_per_chunk = 3  #None
        self.stage = "test"
        self.output_dir = Path("./data/zeroverse") / self.stage

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.dataset = BevyZeroverseDataset(
            self.editor, self.headless, self.num_cameras,
            self.width, self.height, self.num_samples,
            scene_type='room',
        )

        # TODO: perform this only once
        self.chunk_paths = chunk_and_save(
            self.dataset,
            self.output_dir / 'chunk',
            bytes_per_chunk=self.bytes_per_chunk,
            samples_per_chunk=self.samples_per_chunk,
        )

        save_to_folders(
            self.dataset,
            self.output_dir / 'fs',
        )

        save_to_mp4(
            self.dataset,
            self.output_dir / 'mp4',
        )

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


def main():
    unittest.main()
    # test()

if __name__ == "__main__":
    main()
