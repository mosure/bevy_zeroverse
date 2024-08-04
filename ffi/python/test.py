import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from bevy_zeroverse_dataloader.dataloader import BevyZeroverseDataset



def visualize(batch):
    print(batch['color'].shape)

    color_images = batch['color'].numpy()
    depth_images = batch['depth'].numpy()
    normal_images = batch['normal'].numpy()

    batch_size = color_images.shape[0]
    num_cameras = color_images.shape[1]
    num_image_types = 3  # color, depth, normal
    total_images = batch_size * num_cameras * num_image_types

    # Calculate number of columns as a multiple of 3 and close to the square root of total images
    cols = 9
    rows = (total_images + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 20))
    axes = axes.flatten()

    for batch_idx in range(batch_size):
        for cam_idx in range(num_cameras):
            # Calculate the index for the current camera's images in the batch
            index = (batch_idx * num_cameras + cam_idx) * num_image_types

            color_image = color_images[batch_idx, cam_idx]
            axes[index].imshow(color_image)
            axes[index].axis('off')
            if batch_size <= 4:
                axes[index].set_title(f'({batch_idx + 1}, {cam_idx + 1})_color')

            depth_image = depth_images[batch_idx, cam_idx]
            axes[index + 1].imshow(depth_image, cmap='gray')
            axes[index + 1].axis('off')
            if batch_size <= 4:
                axes[index + 1].set_title(f'({batch_idx + 1}, {cam_idx + 1})_depth')

            normal_image = normal_images[batch_idx, cam_idx]
            axes[index + 2].imshow(normal_image)
            axes[index + 2].axis('off')
            if batch_size <= 4:
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
        count += 1
        if count == 10:
            break
    end = time.time()
    print('seconds per batch:', (end - start) / count)


def test():
    dataset = BevyZeroverseDataset(
        editor=False, headless=True, num_cameras=5,
        width=640, height=360, num_samples=1e6,
    )
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    # benchmark(dataloader)

    for batch in dataloader:
        visualize(batch)


if __name__ == "__main__":
    test()
