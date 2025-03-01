from pathlib import Path
from PIL import Image
from typing import Optional

import numpy as np
import os
from safetensors import safe_open
from safetensors.torch import save_file
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.utils import save_image

import bevy_zeroverse_ffi


def normalize_hdr_image_tonemap(hdr_image: torch.Tensor) -> torch.Tensor:
    """
    normalizes an HDR image using reinhard tone mapping to handle high dynamic range values.

    Args:
        hdr_image (torch.Tensor): the input HDR image with shape (view, height, width, channel).

    Returns:
        torch.Tensor: normalized HDR image with values in [0, 1].
    """
    assert hdr_image.dim() == 4, "expected input shape to be (view, height, width, channel)."

    # apply reinhard tone mapping: L/(1 + L), where L is the luminance value
    tonemapped_image = hdr_image / (hdr_image + 1.0)

    min_val = torch.amin(tonemapped_image, dim=(1, 2, 3), keepdim=True)
    max_val = torch.amax(tonemapped_image, dim=(1, 2, 3), keepdim=True)

    normalized_image = (tonemapped_image - min_val) / (max_val - min_val + 1e-8)

    return normalized_image.clamp(0.0, 1.0)


# TODO: add sample-level world rotation augment
class View:
    def __init__(
        self,
        color,
        depth,
        normal,
        optical_flow,
        position,
        world_from_view,
        fovy,
        near,
        far,
        width,
        height,
    ):
        self.color = color
        self.depth = depth
        self.normal = normal
        self.optical_flow = optical_flow
        self.position = position
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

        color = reshape_data(rust_view.color, np.float32)

        if len(rust_view.depth) != 0:
            depth = reshape_data(rust_view.depth, np.float32)
        else:
            depth = np.zeros(color.shape, dtype=np.float32)

        if len(rust_view.normal) != 0:
            normal = reshape_data(rust_view.normal, np.float32)
        else:
            normal = np.zeros(color.shape, dtype=np.float32)

        if len(rust_view.optical_flow) != 0:
            optical_flow = reshape_data(rust_view.optical_flow, np.float32)
        else:
            optical_flow = np.zeros(color.shape, dtype=np.float32)

        if len(rust_view.position) != 0:
            position = reshape_data(rust_view.position, np.float32)
        else:
            position = np.zeros(color.shape, dtype=np.float32)

        world_from_view = np.array(rust_view.world_from_view)
        fovy = rust_view.fovy
        near = rust_view.near
        far = rust_view.far
        return cls(
            color,
            depth,
            normal,
            optical_flow,
            position,
            world_from_view,
            fovy,
            near,
            far,
            width,
            height,
        )

    def to_tensors(self):
        color_tensor = torch.tensor(self.color, dtype=torch.float32)
        color_tensor = color_tensor[..., :3]

        depth_tensor = torch.tensor(self.depth, dtype=torch.float32)
        depth_tensor = depth_tensor[..., 0:1]

        normal_tensor = torch.tensor(self.normal, dtype=torch.float32)
        normal_tensor = normal_tensor[..., :3]

        optical_flow_tensor = torch.tensor(self.optical_flow, dtype=torch.float32)
        optical_flow_tensor = optical_flow_tensor[..., :3]

        position_tensor = torch.tensor(self.position, dtype=torch.float32)
        position_tensor = position_tensor[..., :3]

        world_from_view_tensor = torch.tensor(self.world_from_view, dtype=torch.float32)

        fovy_tensor = torch.tensor(self.fovy, dtype=torch.float32).unsqueeze(-1)
        near_tensor = torch.tensor(self.near, dtype=torch.float32).unsqueeze(-1)
        far_tensor = torch.tensor(self.far, dtype=torch.float32).unsqueeze(-1)

        return {
            'color': color_tensor,
            'depth': depth_tensor,
            'normal': normal_tensor,
            'optical_flow': optical_flow_tensor,
            'position': position_tensor,
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
            'optical_flow': [],
            'position': [],
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

        tensor_dict['color'] = normalize_hdr_image_tonemap(tensor_dict['color'])
        tensor_dict['depth'] = tensor_dict['depth']
        tensor_dict['normal'] = tensor_dict['normal']
        tensor_dict['optical_flow'] = tensor_dict['optical_flow']
        tensor_dict['position'] = tensor_dict['position']

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
        max_camera_radius=0.0,
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
        self.max_camera_radius = max_camera_radius

    def initialize(self):
        config = bevy_zeroverse_ffi.BevyZeroverseConfig()
        config.editor = self.editor
        config.headless = self.headless
        config.image_copiers = True
        config.num_cameras = self.num_cameras
        config.width = self.width
        config.height = self.height
        config.scene_type = BevyZeroverseDataset.scene_map[self.scene_type]
        config.playback_mode = bevy_zeroverse_ffi.PlaybackMode.Still
        config.max_camera_radius = self.max_camera_radius
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



def save_to_folders(dataset, output_dir: Path, n_workers: int = 1):
    """
    Saves each sample from the dataset into a folder structure.

    Args:
        dataset (Dataset): The dataset to save.
        output_dir (Path): The directory where the dataset will be saved.
        n_workers (int): Number of worker processes for data loading.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    dataloader = DataLoader(dataset, batch_size=1, num_workers=n_workers, shuffle=False)

    for idx, sample in enumerate(dataloader):
        sample = {k: v.squeeze(0) for k, v in sample.items()}  # Remove batch dimension

        scene_dir = output_dir / f"{idx:06d}"
        scene_dir.mkdir(exist_ok=True)

        color_tensor = sample['color']
        depth_tensor = sample['depth']
        normal_tensor = sample['normal']
        optical_flow_tensor = sample['optical_flow']
        position_tensor = sample['position']

        views = color_tensor.shape[0]
        for view_idx in range(views):
            view_color = color_tensor[view_idx].permute(2, 0, 1)
            image_filename = scene_dir / f"color_{view_idx:02d}.jpg"
            save_image(view_color, str(image_filename))

            view_depth = normalize_hdr_image_tonemap(depth_tensor)[view_idx].permute(2, 0, 1)
            depth_filename = scene_dir / f"depth_{view_idx:02d}.jpg"
            save_image(view_depth, str(depth_filename))

            view_depth_flat = view_depth[0]
            depth_npz_filename = scene_dir / f"depth_{view_idx:02d}.npz"
            np.savez(depth_npz_filename, depth=view_depth_flat.cpu().numpy())

            view_normal = normalize_hdr_image_tonemap(normal_tensor)[view_idx].permute(2, 0, 1)
            normal_filename = scene_dir / f"normal_{view_idx:02d}.jpg"
            save_image(view_normal, str(normal_filename))

            view_normal_flat = view_normal[0]
            normal_npz_filename = scene_dir / f"normal_{view_idx:02d}.npz"
            np.savez(normal_npz_filename, normal=view_normal_flat.cpu().numpy())

            view_optical_flow = normalize_hdr_image_tonemap(optical_flow_tensor)[view_idx].permute(2, 0, 1)
            optical_flow_filename = scene_dir / f"optical_flow_{view_idx:02d}.jpg"
            save_image(view_optical_flow, str(optical_flow_filename))

            # TODO: save motion vectors as npz, not optical flow rgb
            # view_optical_flow_flat = view_optical_flow[0]
            # optical_flow_npz_filename = scene_dir / f"optical_flow_{view_idx:02d}.npz"
            # np.savez(optical_flow_npz_filename, optical_flow=view_optical_flow_flat.cpu().numpy())

            view_position = normalize_hdr_image_tonemap(position_tensor)[view_idx].permute(2, 0, 1)
            position_filename = scene_dir / f"position_{view_idx:02d}.jpg"
            save_image(view_position, str(position_filename))

            view_position_flat = view_position[0]
            position_npz_filename = scene_dir / f"position_{view_idx:02d}.npz"
            np.savez(position_npz_filename, position=view_position_flat.cpu().numpy())

        meta_tensors = {
            'world_from_view': sample['world_from_view'],
            'fovy': sample['fovy'],
            'near': sample['near'],
            'far': sample['far'],
            'aabb': sample['aabb'],
        }
        meta_filename = scene_dir / "meta.safetensors"
        save_file(meta_tensors, str(meta_filename))

        print(f"Saved sample {idx} to {scene_dir}")


class FolderDataset(Dataset):
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.scene_dirs = sorted([d for d in self.output_dir.iterdir() if d.is_dir()])

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_dir = self.scene_dirs[idx]
        meta_filename = scene_dir / "meta.safetensors"
        with safe_open(str(meta_filename), framework="pt", device="cpu") as f:
            meta_tensors = {key: f.get_tensor(key) for key in f.keys()}

        color_images = sorted(scene_dir.glob("color_*.jpg"))
        color_tensors = []
        depth_tensors = []
        normal_tensors = []
        optical_flow_tensors = []
        position_tensors = []
        for image_file in color_images:
            image = Image.open(image_file).convert("RGB")
            color_tensor = self.transform(image).permute(1, 2, 0)
            color_tensors.append(color_tensor)

            depth_file = image_file.with_name(image_file.name.replace("color", "depth").replace("jpg", "npz"))
            if depth_file.exists():
                # depth = Image.open(depth_file).convert("L")
                # depth_tensor = self.transform(depth).permute(1, 2, 0)

                data = np.load(depth_file)
                depth_np = data['depth']
                loaded_depth = torch.from_numpy(depth_np).to(torch.float32)
                loaded_depth = loaded_depth.unsqueeze(-1)

                depth_tensors.append(loaded_depth)

            normal_file = image_file.with_name(image_file.name.replace("color", "normal").replace("jpg", "npz"))
            if normal_file.exists():
                data = np.load(normal_file)
                normal_np = data['normal']
                loaded_normal = torch.from_numpy(normal_np).to(torch.float32)
                normal_tensors.append(loaded_normal)

            optical_flow_file = image_file.with_name(image_file.name.replace("color", "optical_flow"))
            if optical_flow_file.exists():
                image = Image.open(optical_flow_file).convert("RGB")
                optical_flow_tensor = self.transform(image).permute(1, 2, 0)
                optical_flow_tensors.append(optical_flow_tensor)

            position_file = image_file.with_name(image_file.name.replace("color", "position").replace("jpg", "npz"))
            if position_file.exists():
                data = np.load(position_file)
                position_np = data['position']
                loaded_position = torch.from_numpy(position_np).to(torch.float32)
                position_tensors.append(loaded_position)

        color_tensor = torch.stack(color_tensors, dim=0)

        if len(depth_tensors) == 0:
            depth_tensor = torch.zeros_like(color_tensor)[..., 0:1]
        else:
            depth_tensor = torch.stack(depth_tensors, dim=0)

        if len(normal_tensors) == 0:
            normal_tensor = torch.zeros_like(color_tensor)[..., 0:1]
        else:
            normal_tensor = torch.stack(normal_tensors, dim=0)

        if len(optical_flow_tensors) == 0:
            optical_flow_tensor = torch.zeros_like(color_tensor)[..., 0:1]
        else:
            optical_flow_tensor = torch.stack(optical_flow_tensors, dim=0)

        if len(position_tensors) == 0:
            position_tensor = torch.zeros_like(color_tensor)[..., 0:1]
        else:
            position_tensor = torch.stack(position_tensors, dim=0)

        meta_tensors['color'] = color_tensor
        meta_tensors['depth'] = depth_tensor
        meta_tensors['normal'] = normal_tensor
        meta_tensors['optical_flow'] = optical_flow_tensor
        meta_tensors['position'] = position_tensor

        return meta_tensors
