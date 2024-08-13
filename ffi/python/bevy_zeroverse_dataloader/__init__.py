import numpy as np
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.utils.data import DataLoader, Dataset

import bevy_zeroverse


# TODO: add sample-level world rotation augment
class View:
    def __init__(self, color, depth, normal, view_from_world, fovy, width, height):
        self.color = color
        self.depth = depth
        self.normal = normal
        self.view_from_world = view_from_world
        self.fovy = fovy
        self.width = width
        self.height = height

    @classmethod
    def from_rust(cls, rust_view, width, height):
        width = int(width)
        height = int(height)

        def reshape_data(data, dtype):
            return np.frombuffer(data, dtype=dtype).reshape(height, width, 4)

        if len(rust_view.color) == 0:
            print("empty color buffer")

        if len(rust_view.depth) == 0:
            print("empty depth buffer")

        if len(rust_view.normal) == 0:
            print("empty normal buffer")

        color = reshape_data(rust_view.color, np.float32)
        depth = reshape_data(rust_view.depth, np.float32)
        normal = reshape_data(rust_view.normal, np.float32)

        view_from_world = np.array(rust_view.view_from_world)
        fovy = rust_view.fovy
        return cls(color, depth, normal, view_from_world, fovy, width, height)

    def to_tensors(self):
        color_tensor = torch.tensor(self.color, dtype=torch.float32)
        depth_tensor = torch.tensor(self.depth, dtype=torch.float32)
        normal_tensor = torch.tensor(self.normal, dtype=torch.float32)

        color_tensor = color_tensor[..., :3]
        depth_tensor = depth_tensor[..., 0]
        normal_tensor = normal_tensor[..., :3]

        view_from_world_tensor = torch.tensor(self.view_from_world, dtype=torch.float32)
        fovy_tensor = torch.tensor(self.fovy, dtype=torch.float32)
        return {
            'color': color_tensor,
            'depth': depth_tensor,
            'normal': normal_tensor,
            'view_from_world': view_from_world_tensor,
            'fovy': fovy_tensor
        }

class Sample:
    def __init__(self, views):
        self.views = views

    @classmethod
    def from_rust(cls, rust_sample, width, height):
        views = [View.from_rust(view, width, height) for view in rust_sample.views]
        return cls(views)

    def to_tensors(self):
        tensor_dict = {
            'color': [],
            'depth': [],
            'normal': [],
            'view_from_world': [],
            'fovy': []
        }

        if len(self.views) == 0:
            print("empty views")
            return tensor_dict

        for view in self.views:
            tensors = view.to_tensors()
            for key in tensor_dict:
                tensor_dict[key].append(tensors[key])

        for key in tensor_dict:
            tensor_dict[key] = torch.stack(tensor_dict[key], dim=0)

        return tensor_dict


# TODO: add dataset seed parameter to config
class BevyZeroverseDataset(Dataset):
    def __init__(
        self,
        editor,
        headless,
        num_cameras,
        width,
        height,
        num_samples,
        root_asset_folder=None,
    ):
        self.editor = editor
        self.headless = headless
        self.num_cameras = num_cameras
        self.width = width
        self.height = height
        self.num_samples = int(num_samples)
        self.initialized = False
        self.root_asset_folder = root_asset_folder

    def initialize(self):
        config = bevy_zeroverse.BevyZeroverseConfig()
        config.editor = self.editor
        config.headless = self.headless
        config.num_cameras = self.num_cameras
        config.width = self.width
        config.height = self.height
        config.scene_type = bevy_zeroverse.ZeroverseSceneType.Room
        config.regenerate_scene_material_shuffle_period = 256
        bevy_zeroverse.initialize(
            config,
            self.root_asset_folder,
        )
        self.initialized = True

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.initialized:
            self.initialize()

        rust_sample = bevy_zeroverse.next()
        sample = Sample.from_rust(rust_sample, self.width, self.height)
        return sample.to_tensors()


def chunk_and_save(
    dataset,
    output_dir: Path,
    bytes_per_chunk: int = int(256 * 1024 * 1024),
    n_workers: int = 1,
):
    chunk_size = 0
    chunk_index = 0
    chunk = []
    original_samples = []
    chunk_file_paths = []

    def save_chunk():
        nonlocal chunk_size, chunk_index, chunk, original_samples, chunk_file_paths

        chunk_key = f"{chunk_index:0>6}"
        print(f"saving chunk {chunk_key} of {len(dataset)} ({chunk_size / 1e6:.2f} MB).")
        output_dir.mkdir(exist_ok=True, parents=True)
        file_path = output_dir / f"{chunk_key}.safetensors"

        batch = {}
        for sample in chunk:
            for key, tensor in sample.items():
                if key not in batch:
                    batch[key] = []
                batch[key].append(tensor)

        # Stack the tensors for each key
        flat_tensors = {key: torch.stack(tensors, dim=0) for key, tensors in batch.items()}
        save_file(flat_tensors, str(file_path))

        chunk_file_paths.append(file_path)
        chunk_size = 0
        chunk_index += 1
        chunk = []

    dataloader = DataLoader(dataset, batch_size=1, num_workers=n_workers, shuffle=False)

    for idx, sample in enumerate(dataloader):
        sample = {k: v.squeeze(0) for k, v in sample.items()}
        sample_size = sum(tensor.numel() * tensor.element_size() for tensor in sample.values())
        chunk.append(sample)
        original_samples.append(sample)
        chunk_size += sample_size

        print(f"    added sample {idx} to chunk ({sample_size / 1e6:.2f} MB).")
        if chunk_size >= bytes_per_chunk:
            save_chunk()

    if chunk_size > 0:
        save_chunk()

    return original_samples, chunk_file_paths


class ChunkedDataset(Dataset):
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.chunk_files = sorted(output_dir.glob("*.safetensors"))

    def load_chunk(self, file_path: Path):
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}

    def __len__(self):
        return len(self.chunk_files)

    def __getitem__(self, idx):
        chunk_file = self.chunk_files[idx]
        return self.load_chunk(chunk_file)
