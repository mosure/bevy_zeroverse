from torch.utils.data import DataLoader

from bevy_zeroverse_dataloader import ChunkedIteratorDataset

from test import visualize


def main():
    chunked_dataset = ChunkedIteratorDataset("data/zeroverse/cli")
    dataloader = DataLoader(chunked_dataset, batch_size=1, shuffle=False)

    for batch in dataloader:
        visualize(batch)


if __name__ == "__main__":
    main()
