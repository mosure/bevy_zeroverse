from pathlib import Path
import shutil

from bevy_zeroverse_dataloader import BevyZeroverseDataset, chunk_and_save, save_to_folders


def generate_chunked_dataset(
    output_dir = Path("./data/zeroverse/cli"),
    bytes_per_chunk = int(256 * 1024 * 1024),
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
        bytes_per_chunk,
        n_workers=4,
    )


if __name__ == "__main__":
    # TODO: add cli arguments
    chunk_paths = generate_chunked_dataset()

    print(f"chunks:\n{chunk_paths}")
