import matplotlib.pyplot as plt
from pathlib import Path
import shutil
from torch.utils.data import DataLoader
import unittest

from bevy_zeroverse_dataloader.dataloader import BevyZeroverseDataset, ChunkedDataset



def visualize(batch):
    print(batch['color'].shape)

    color_images = batch['color'].numpy()
    depth_images = batch['depth'].numpy()
    normal_images = batch['normal'].numpy()

    batch_size = color_images.shape[0]
    num_cameras = color_images.shape[1]
    num_image_types = 3  # color, depth, normal
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

            depth_image = depth_images[batch_idx, cam_idx]
            axes[index + 1].imshow(depth_image, cmap='gray')
            axes[index + 1].axis('off')
            if batch_size <= 2:
                axes[index + 1].set_title(f'({batch_idx + 1}, {cam_idx + 1})_depth')

            normal_image = normal_images[batch_idx, cam_idx]
            axes[index + 2].imshow(normal_image)
            axes[index + 2].axis('off')
            if batch_size <= 2:
                axes[index + 2].set_title(f'({batch_idx + 1}, {cam_idx + 1})_normal')

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
        width=640, height=360, num_samples=1e6,
    )
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=1)

    # benchmark(dataloader)

    for batch in dataloader:
        visualize(batch)


generated = False
class TestChunkedDataset(unittest.TestCase):
    def setUp(self):
        self.editor = True
        self.headless = True
        self.num_cameras = 9
        self.width = 640
        self.height = 360
        self.num_samples = 100
        self.bytes_per_chunk = int(1e8)
        self.stage = "test"
        self.output_dir = Path("./data/zeroverse") / self.stage

        if self.output_dir.exists():
            shutil.rmtree(self.output_dir)

        self.dataset = BevyZeroverseDataset(self.editor, self.headless, self.num_cameras, self.width, self.height, self.num_samples)
        self.original_samples = self.dataset.chunk_and_save(self.output_dir, self.bytes_per_chunk)

    def test_chunked_dataset_loading(self):
        chunked_dataset = ChunkedDataset(self.output_dir)
        dataloader = DataLoader(chunked_dataset, batch_size=1, shuffle=False)

        num_chunks = 0
        total_loaded_samples = 0

        expected_shapes = {key: tensor.shape for key, tensor in self.original_samples[0].items()}

        for batch in dataloader:
            num_chunks += 1

            for key, tensor in batch.items():
                tensor = tensor.squeeze(0)
                expected_shape = (tensor.shape[0],) + expected_shapes[key]
                self.assertEqual(tensor.shape, expected_shape, f"Mismatch in tensor shape for key {key}")

            total_loaded_samples += batch['color'].squeeze(0).shape[0]

        expected_num_chunks = len(chunked_dataset)
        self.assertEqual(num_chunks, expected_num_chunks, "Mismatch in number of chunks")

        self.assertEqual(total_loaded_samples, len(self.original_samples))

    def test_benchmark_chunked_dataloader(self):
        chunked_dataset = ChunkedDataset(self.output_dir)
        dataloader = DataLoader(chunked_dataset, batch_size=1, shuffle=False)

        print("\nBenchmarking chunks:")
        benchmark(dataloader)


def main():
    unittest.main()
    # test()

if __name__ == "__main__":
    main()
