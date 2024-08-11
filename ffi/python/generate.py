from pathlib import Path
import shutil

from bevy_zeroverse_dataloader import BevyZeroverseDataset


def generate_chunked_dataset(
        stage = "test",
        output_dir = Path("./data/zeroverse"),
        bytes_per_chunk = int(1e8),
        dataset = BevyZeroverseDataset(
            editor=False,
            headless=True,
            num_cameras=4,
            width=640,
            height=360,
            num_samples=10,
        )
    ) -> list:
    if output_dir.exists():
        shutil.rmtree(output_dir / stage)

    return dataset.chunk_and_save(
        output_dir / stage,
        bytes_per_chunk,
        n_workers=4,
    )


if __name__ == "__main__":
    # TODO: add cli arguments
    _data, chunk_paths = generate_chunked_dataset()

    print(f"chunks:\n{chunk_paths}")
