from pathlib import Path
import shutil

from bevy_zeroverse_dataloader import BevyZeroverseDataset, chunk_and_save


def generate_chunked_dataset(
    output_dir = Path("./data/zeroverse/cli"),
    bytes_per_chunk = int(256 * 1024 * 1024),
    dataset = BevyZeroverseDataset(
        editor=False,
        headless=True,
        num_cameras=4,
        width=640,
        height=480,
        num_samples=100,
    )
) -> list:
    if output_dir.exists():
        shutil.rmtree(output_dir)

    return chunk_and_save(
        dataset,
        output_dir,
        bytes_per_chunk,
        n_workers=4,
    )


if __name__ == "__main__":
    # TODO: add cli arguments
    _data, chunk_paths = generate_chunked_dataset()

    print(f"chunks:\n{chunk_paths}")
