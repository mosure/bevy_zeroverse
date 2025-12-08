from pathlib import Path
import shutil

from bevy_zeroverse_dataloader import BevyZeroverseDataset, chunk_and_save, save_to_folders, save_to_mp4


def generate_chunked_dataset(
    output_dir = Path("./data/zeroverse/cli/chunk"),
    dataset = BevyZeroverseDataset(
        editor=False,
        headless=True,
        num_cameras=4,
        width=640,
        height=480,
        num_samples=4,
        scene_type='room',
    )
) -> list:
    save_to_folders(
        dataset,
        output_dir,
        n_workers=4,
    )

    return chunk_and_save(
        dataset,
        output_dir,
        samples_per_chunk=2,
        n_workers=4,
    )


def video_dataset(
    output_dir = Path("./data/zeroverse/cli/video"),
    dataset = BevyZeroverseDataset(
        editor=False,
        headless=True,
        num_cameras=9,
        width=640,
        height=480,
        num_samples=1,
        scene_type='human',
        playback_step = 1.0 / 120.0,
        playback_steps = 120,
        render_modes = ['normal']
    )
):
    save_to_mp4(
        dataset,
        output_dir,
        n_workers=4,
    )


if __name__ == "__main__":
    # TODO: add cli arguments
    # chunk_paths = generate_chunked_dataset()

    # print(f"chunks:\n{chunk_paths}")

    video_dataset()
