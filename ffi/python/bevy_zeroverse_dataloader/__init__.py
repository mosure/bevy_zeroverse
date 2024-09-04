from pathlib import Path
from typing import Optional

import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.utils.data import DataLoader, Dataset

import bevy_zeroverse_ffi


# TODO: add sample-level world rotation augment
class View:
    def __init__(self, color, depth, normal, world_from_view, fovy, near, far, width, height):
        self.color = color
        self.depth = depth
        self.normal = normal
        self.world_from_view = world_from_view
        self.fovy = fovy
        self.near = near
        self.far = far
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

        world_from_view = np.array(rust_view.world_from_view)
        fovy = rust_view.fovy
        near = rust_view.near
        far = rust_view.far
        return cls(color, depth, normal, world_from_view, fovy, near, far, width, height)

    def to_tensors(self):
        color_tensor = torch.tensor(self.color, dtype=torch.float32)
        depth_tensor = torch.tensor(self.depth, dtype=torch.float32)
        normal_tensor = torch.tensor(self.normal, dtype=torch.float32)

        color_tensor = color_tensor[..., :3]
        depth_tensor = depth_tensor[..., 0:1]
        normal_tensor = normal_tensor[..., :3]

        world_from_view_tensor = torch.tensor(self.world_from_view, dtype=torch.float32)

        fovy_tensor = torch.tensor(self.fovy, dtype=torch.float32).unsqueeze(-1)
        near_tensor = torch.tensor(self.near, dtype=torch.float32).unsqueeze(-1)
        far_tensor = torch.tensor(self.far, dtype=torch.float32).unsqueeze(-1)

        return {
            'color': color_tensor,
            'depth': depth_tensor,
            'normal': normal_tensor,
            'world_from_view': world_from_view_tensor,
            'fovy': fovy_tensor,
            'near': near_tensor,
            'far': far_tensor,
        }

class Sample:
    def __init__(self, views, aabb):
        self.views = views
        self.aabb = aabb

    @classmethod
    def from_rust(cls, rust_sample, width, height):
        views = [View.from_rust(view, width, height) for view in rust_sample.views]
        aabb = np.array(rust_sample.aabb)
        return cls(views, aabb)

    def to_tensors(self):
        tensor_dict = {
            'color': [],
            'depth': [],
            'normal': [],
            'world_from_view': [],
            'fovy': [],
            'near': [],
            'far': [],
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

        tensor_dict['aabb'] = torch.tensor(self.aabb, dtype=torch.float32)

        return tensor_dict


# TODO: add dataset seed parameter to config
class BevyZeroverseDataset(Dataset):
    scene_map = {
        'cornell_cube': bevy_zeroverse_ffi.ZeroverseSceneType.CornellCube,
        'object': bevy_zeroverse_ffi.ZeroverseSceneType.Object,
        'room': bevy_zeroverse_ffi.ZeroverseSceneType.Room,
    }

    def __init__(
        self,
        editor,
        headless,
        num_cameras,
        width,
        height,
        num_samples,
        root_asset_folder=None,
        scene_type='object',
    ):
        self.editor = editor
        self.headless = headless
        self.num_cameras = num_cameras
        self.width = width
        self.height = height
        self.num_samples = int(num_samples)
        self.initialized = False
        self.root_asset_folder = root_asset_folder
        self.scene_type = scene_type

    def initialize(self):
        config = bevy_zeroverse_ffi.BevyZeroverseConfig()
        config.editor = self.editor
        config.headless = self.headless
        config.num_cameras = self.num_cameras
        config.width = self.width
        config.height = self.height
        config.scene_type = BevyZeroverseDataset.scene_map[self.scene_type]
        config.regenerate_scene_material_shuffle_period = 256
        bevy_zeroverse_ffi.initialize(
            config,
            self.root_asset_folder,
        )
        self.initialized = True

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if not self.initialized:
            self.initialize()

        rust_sample = bevy_zeroverse_ffi.next()
        sample = Sample.from_rust(rust_sample, self.width, self.height)
        return sample.to_tensors()


def chunk_and_save(
    dataset,
    output_dir: Path,
    bytes_per_chunk: Optional[int] = int(256 * 1024 * 1024),
    samples_per_chunk: Optional[int] = None,
    n_workers: int = 1,
):
    """
    if samples_per_chunk is not None, the dataset will be chunked into chunks of size samples_per_chunk, regardless of bytes_per_chunk.
    """

    output_dir.mkdir(exist_ok=True, parents=True)
    existing_chunks = sorted(output_dir.glob("*.safetensors"))
    if existing_chunks:
        latest_chunk = existing_chunks[-1]
        chunk_index = int(latest_chunk.stem)
        print(f"resuming from chunk {chunk_index}.")
    else:
        chunk_index = 0

    chunk_size = 0
    chunk = []
    chunk_file_paths = [output_dir / f"{int(chunk.stem):0>6}.safetensors" for chunk in existing_chunks]

    def save_chunk():
        nonlocal chunk_size, chunk_index, chunk, chunk_file_paths

        chunk_key = f"{chunk_index:0>6}"
        print(f"saving chunk {chunk_key} of {len(dataset)} ({chunk_size / 1e6:.2f} MB).")
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

        del batch
        torch.cuda.empty_cache()
        import gc
        gc.collect()

    dataloader = DataLoader(dataset, batch_size=1, num_workers=n_workers, shuffle=False)

    for idx, sample in enumerate(dataloader):
        sample = {k: v.squeeze(0) for k, v in sample.items()}
        sample_size = sum(tensor.numel() * tensor.element_size() for tensor in sample.values())
        chunk.append(sample)
        chunk_size += sample_size

        print(f"    added sample {idx} to chunk ({sample_size / 1e6:.2f} MB).")

        if samples_per_chunk is not None:
            if len(chunk) >= samples_per_chunk:
                save_chunk()
            continue

        if chunk_size >= bytes_per_chunk:
            save_chunk()

    if chunk_size > 0:
        save_chunk()

    return chunk_file_paths


class ChunkedDataset(Dataset):
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.chunk_files = sorted(self.output_dir.glob("*.safetensors"))

    def load_chunk(self, file_path: Path):
        with safe_open(str(file_path), framework="pt", device="cpu") as f:
            return {key: f.get_tensor(key) for key in f.keys()}

    def __len__(self):
        return len(self.chunk_files)

    def __getitem__(self, idx):
        chunk_file = self.chunk_files[idx]
        return self.load_chunk(chunk_file)
