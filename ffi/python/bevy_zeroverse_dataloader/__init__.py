from io import BytesIO
import os
from pathlib import Path
from PIL import Image
import random
from typing import Optional

import numpy as np
from safetensors import safe_open
from safetensors.torch import load_file, save_file
import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import save_image

import bevy_zeroverse_ffi


def normalize_hdr_image_tonemap(hdr_image: torch.Tensor) -> torch.Tensor:
    """
    normalizes an HDR image using reinhard tone mapping to handle high dynamic range values.

    Args:
        hdr_image (torch.Tensor): the input HDR image with shape (time, view, height, width, channel).

    Returns:
        torch.Tensor: normalized HDR image with values in [0, 1].
    """
    assert hdr_image.dim() == 5, "expected input shape to be (time, view, height, width, channel)."

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
        time,
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
        self.time = time
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
            depth = None

        if len(rust_view.normal) != 0:
            normal = reshape_data(rust_view.normal, np.float32)
        else:
            normal = None

        if len(rust_view.optical_flow) != 0:
            optical_flow = reshape_data(rust_view.optical_flow, np.float32)
        else:
            optical_flow = None

        if len(rust_view.position) != 0:
            position = reshape_data(rust_view.position, np.float32)
        else:
            position = None

        world_from_view = np.array(rust_view.world_from_view)
        fovy = rust_view.fovy
        near = rust_view.near
        far = rust_view.far
        time = rust_view.time
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
            time,
            width,
            height,
        )

    def to_tensors(self):
        batch = {}

        color_tensor = torch.tensor(self.color, dtype=torch.float32)
        batch['color'] = color_tensor[..., :3]

        if self.depth is not None:
            depth_tensor = torch.tensor(self.depth, dtype=torch.float32)
            batch['depth'] = depth_tensor[..., 0:1]

        if self.normal is not None:
            normal_tensor = torch.tensor(self.normal, dtype=torch.float32)
            batch['normal'] = normal_tensor[..., :3]

        if self.optical_flow is not None:
            optical_flow_tensor = torch.tensor(self.optical_flow, dtype=torch.float32)
            batch['optical_flow'] = optical_flow_tensor[..., :3]

        if self.position is not None:
            position_tensor = torch.tensor(self.position, dtype=torch.float32)
            batch['position'] = position_tensor[..., :3]

        batch['far'] = torch.tensor(self.far, dtype=torch.float32).unsqueeze(-1)
        batch['fovy'] = torch.tensor(self.fovy, dtype=torch.float32).unsqueeze(-1)
        batch['near'] = torch.tensor(self.near, dtype=torch.float32).unsqueeze(-1)
        batch['time'] = torch.tensor(self.time, dtype=torch.float32).unsqueeze(-1)
        batch['world_from_view'] = torch.tensor(self.world_from_view, dtype=torch.float32)

        return batch

class Sample:
    def __init__(self, views, view_dim, aabb):
        self.views = views
        self.view_dim = view_dim
        self.aabb = aabb

    @classmethod
    def from_rust(cls, rust_sample, width, height):
        views = [View.from_rust(view, width, height) for view in rust_sample.views]
        view_dim = rust_sample.view_dim
        aabb = np.array(rust_sample.aabb)
        return cls(views, view_dim, aabb)

    def to_tensors(self):
        sample = {}

        if len(self.views) == 0:
            print("empty views")
            return sample

        for view in self.views:
            tensors = view.to_tensors()
            for key in tensors.keys():
                if key not in sample:
                    sample[key] = []
                sample[key].append(tensors[key])

        for key in sample:
            sample[key] = torch.stack(sample[key], dim=0)

            new_shape = (-1, self.view_dim, *sample[key].shape[1:])
            sample[key] = sample[key].reshape(new_shape)

        sample['aabb'] = torch.tensor(self.aabb, dtype=torch.float32)

        normalize_keys = ['color', 'optical_flow']
        for key in normalize_keys:
            if key in sample:
                sample[key] = normalize_hdr_image_tonemap(sample[key])

        return sample


# TODO: add dataset seed parameter to config
class BevyZeroverseDataset(Dataset):
    render_mode_map = {
        'color': bevy_zeroverse_ffi.RenderMode.Color,
        'depth': bevy_zeroverse_ffi.RenderMode.Depth,
        'normal': bevy_zeroverse_ffi.RenderMode.Normal,
        'optical_flow': bevy_zeroverse_ffi.RenderMode.OpticalFlow,
        'position': bevy_zeroverse_ffi.RenderMode.Position,
        'semantic': bevy_zeroverse_ffi.RenderMode.Semantic,
    }

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
        playback_step=0.1,
        playback_steps=5,
        render_modes=['color'],
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
        self.playback_step = playback_step
        self.playback_steps = playback_steps
        self.render_modes = render_modes

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
        config.playback_step = self.playback_step
        config.playback_steps = self.playback_steps
        config.render_modes = [BevyZeroverseDataset.render_mode_map[mode] for mode in self.render_modes]
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


def decode_jpg_tensor(jpg_tensor: torch.Tensor) -> torch.Tensor:
    buffer = BytesIO(jpg_tensor.numpy().tobytes())
    img = Image.open(buffer).convert("RGB")
    return to_tensor(img).permute(1, 2, 0)

def encode_tensor_to_jpg(tensor: torch.Tensor, quality: int = 75) -> torch.Tensor:
    buffer = BytesIO()
    img = to_pil_image(tensor.permute(2, 0, 1))
    img.save(buffer, format="JPEG", quality=quality)
    return torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)


def chunk_and_save(
    dataset,
    output_dir: Path,
    bytes_per_chunk: Optional[int] = int(256 * 1024 * 1024),
    samples_per_chunk: Optional[int] = None,
    n_workers: int = 1,
    jpg_quality: int = 75,
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

    total_chunks = len(dataset)
    if samples_per_chunk is not None:
        total_chunks = len(dataset) // samples_per_chunk

    def save_chunk():
        nonlocal chunk_size, chunk_index, chunk, chunk_file_paths

        chunk_key = f"{chunk_index:0>6}"
        print(f"saving chunk {chunk_key} of {total_chunks} ({chunk_size / 1e6:.2f} MB).")
        file_path = output_dir / f"{chunk_key}.safetensors"

        B = len(chunk)
        batch = {}
        offset = 0
        for sample in chunk:
            for key, tensor in sample.items():
                if key in ['color']:
                    T, V, H, W, C = tensor.shape
                    num_views = T * V
                    for i, view in enumerate(tensor.view(-1, H, W, C)):
                        view = encode_tensor_to_jpg(view, quality=jpg_quality)
                        batch[f'{key}_jpg_{offset + i}'] = view

                    if f'{key}_shape' not in batch:
                        batch[f'{key}_shape'] = torch.tensor([B, T, V, H, W, C])
                    offset += num_views
                else:
                    if key not in batch:
                        batch[key] = []
                    batch[key].append(tensor)

        flat_tensors = {
            key: torch.stack(tensors, dim=0) if '_jpg' not in key and '_shape' not in key else tensors for key, tensors in batch.items()
        }
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
        # save_chunk()
        pass  # ignore chunks that do not have full size

    return chunk_file_paths


def crop(tensor, shape, channels_last=True):
    if channels_last:
        if tensor.ndim == 3:
            tensor = tensor.permute(2, 0, 1)
        elif tensor.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2)
        elif tensor.ndim == 5:
            tensor = tensor.permute(0, 1, 4, 2, 3)  # (B, V, C, H, W)
        elif tensor.ndim == 6:
            tensor = tensor.permute(0, 1, 2, 5, 3, 4)  # (B, T, V, C, H, W)
        else:
            raise ValueError(f"invalid number of dimensions: {tensor.ndim}")

    *_, H, W = tensor.shape
    h_out, w_out = shape

    row = (H - h_out) // 2
    col = (W - w_out) // 2

    tensor_cropped = tensor[..., row:row + h_out, col:col + w_out]

    if channels_last:
        if tensor.ndim == 3:
            tensor_cropped = tensor_cropped.permute(1, 2, 0)
        elif tensor.ndim == 4:
            tensor_cropped = tensor_cropped.permute(0, 2, 3, 1)
        elif tensor.ndim == 5:
            tensor_cropped = tensor_cropped.permute(0, 1, 3, 4, 2)
        elif tensor.ndim == 6:
            tensor_cropped = tensor_cropped.permute(0, 1, 2, 4, 5, 3)

    return tensor_cropped


def load_chunk(file_path: Path):
    tensors = load_file(str(file_path))
    batch = {}

    # TODO: aggregate list of jpg data and call torchvision.io.decode_jpeg /w 'cuda' device
    #       note, the resulting tensor needs to be on CPU for multiprocessed dataloaders to function. it would be most efficient to decode jpeg at the batch level
    for key, tensor in tensors.items():
        if '_jpg_' in key:
            parent = key.rsplit('_jpg_', 1)[0]
            if parent in batch:
                continue
            shape = tensors[f'{parent}_shape'].tolist()
            images = shape[0] * shape[1] * shape[2]
            decoded_images = [decode_jpg_tensor(
                    tensors[f'{parent}_jpg_{i}']
                ) for i in range(images)]

            height = min(image.shape[0] for image in decoded_images)
            width = min(image.shape[1] for image in decoded_images)
            perform_crop = any(image.shape[0] != height or image.shape[1] != width for image in decoded_images)
            decoded_images = [crop(image, (height, width)) for image in decoded_images] if perform_crop else decoded_images

            batch[parent] = torch.stack(decoded_images).reshape(shape)
        elif '_shape' in key:
            continue
        else:
            batch[key] = tensor

    return batch

class ChunkedIteratorDataset(IterableDataset):
    def __init__(self, output_dir: Path, shuffle: bool = False):
        self.output_dir = Path(output_dir)
        self.chunk_files = sorted(self.output_dir.glob("*.safetensors"))
        self.shuffle = shuffle

        if self.chunk_files:
            last_chunk = self.chunk_files[0]
            tensors = load_chunk(last_chunk)
            self.samples_per_chunk_estimate = next(iter(tensors.values())).shape[0]

    # def __len__(self):
    #     return len(self.chunk_files) * self.samples_per_chunk_estimate

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            chunk_files = self.chunk_files
        else:
            chunk_files = self.chunk_files[worker_info.id::worker_info.num_workers]

        if self.shuffle:
            random.shuffle(chunk_files)

        for chunk_file in chunk_files:
            chunk_data = load_chunk(chunk_file)
            num_samples = next(iter(chunk_data.values())).shape[0]
            sample_indices = list(range(num_samples))

            if self.shuffle:
                random.shuffle(sample_indices)

            for sample_idx in sample_indices:
                yield {key: tensor[sample_idx] for key, tensor in chunk_data.items()}



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

        for timestep in range(color_tensor.shape[0]):
            for view_idx in range(color_tensor.shape[1]):
                if 'color' in sample:
                    view_color = sample['color'][timestep, view_idx].permute(2, 0, 1)
                    image_filename = scene_dir / f"color_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_color, str(image_filename))

                if 'depth' in sample:
                    view_depth = normalize_hdr_image_tonemap(sample['depth'])[timestep, view_idx].permute(2, 0, 1)
                    depth_filename = scene_dir / f"depth_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_depth, str(depth_filename))

                    view_depth_flat = sample['depth'][timestep, view_idx][0]
                    depth_npz_filename = scene_dir / f"depth_{timestep:03d}_{view_idx:02d}.npz"
                    np.savez(depth_npz_filename, depth=view_depth_flat.cpu().numpy())

                if 'normal' in sample:
                    view_normal = normalize_hdr_image_tonemap(sample['normal'])[timestep, view_idx].permute(2, 0, 1)
                    normal_filename = scene_dir / f"normal_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_normal, str(normal_filename))

                    view_normal_flat = sample['normal'][timestep, view_idx][0]
                    normal_npz_filename = scene_dir / f"normal_{timestep:03d}_{view_idx:02d}.npz"
                    np.savez(normal_npz_filename, normal=view_normal_flat.cpu().numpy())

                if 'optical_flow' in sample:
                    view_optical_flow = normalize_hdr_image_tonemap(sample['optical_flow'])[timestep, view_idx].permute(2, 0, 1)
                    optical_flow_filename = scene_dir / f"optical_flow_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_optical_flow, str(optical_flow_filename))

                # TODO: save motion vectors as npz, not optical flow rgb
                # view_optical_flow_flat = view_optical_flow[0]
                # optical_flow_npz_filename = scene_dir / f"optical_flow_{timestep:03d}_{view_idx:02d}.npz"
                # np.savez(optical_flow_npz_filename, optical_flow=view_optical_flow_flat.cpu().numpy())

                if 'position' in sample:
                    print('position shape', sample['position'].shape)

                    view_position = normalize_hdr_image_tonemap(sample['position'])[timestep, view_idx].permute(2, 0, 1)
                    position_filename = scene_dir / f"position_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_position, str(position_filename))

                    view_position_flat = sample['position'][timestep, view_idx][0]
                    position_npz_filename = scene_dir / f"position_{timestep:03d}_{view_idx:02d}.npz"
                    np.savez(position_npz_filename, position=view_position_flat.cpu().numpy())

        meta_tensors = {
            'world_from_view': sample['world_from_view'],
            'fovy': sample['fovy'],
            'near': sample['near'],
            'far': sample['far'],
            'time': sample['time'],
            'aabb': sample['aabb'],
        }
        meta_filename = scene_dir / "meta.safetensors"
        save_file(meta_tensors, str(meta_filename))

        print(f"saved sample {idx} to {scene_dir}")


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
        timesteps = sorted({int(p.stem.split('_')[1]) for p in color_images})
        views = sorted({int(p.stem.split('_')[2]) for p in color_images})

        color_tensors = []
        depth_tensors = []
        normal_tensors = []
        optical_flow_tensors = []
        position_tensors = []

        for timestep in timesteps:
            timestep_colors, timestep_depths, timestep_normals, timestep_flows, timestep_positions = [], [], [], [], []

            for view in views:
                base_name = f"{timestep:03d}_{view:02d}"

                color_file = scene_dir / f"color_{base_name}.jpg"
                color_tensor = self.transform(Image.open(color_file).convert("RGB")).permute(1, 2, 0)
                timestep_colors.append(color_tensor)

                depth_file = scene_dir / f"depth_{base_name}.npz"
                if depth_file.exists():
                    depth_np = np.load(depth_file)['depth']
                    depth_tensor = torch.from_numpy(depth_np).unsqueeze(-1).float()
                    timestep_depths.append(depth_tensor)

                normal_file = scene_dir / f"normal_{base_name}.npz"
                if normal_file.exists():
                    normal_np = np.load(normal_file)['normal']
                    normal_tensor = torch.from_numpy(normal_np).unsqueeze(-1).float()
                    timestep_normals.append(normal_tensor)

                optical_flow_file = scene_dir / f"optical_flow_{base_name}.jpg"
                if optical_flow_file.exists():
                    flow_tensor = self.transform(Image.open(optical_flow_file).convert("RGB")).permute(1, 2, 0)
                    timestep_flows.append(flow_tensor)

                position_file = scene_dir / f"position_{base_name}.npz"
                if position_file.exists():
                    position_np = np.load(position_file)['position']
                    position_tensor = torch.from_numpy(position_np).unsqueeze(-1).float()
                    timestep_positions.append(position_tensor)

            color_tensors.append(torch.stack(timestep_colors, dim=0))
            depth_tensors.append(torch.stack(timestep_depths, dim=0) if timestep_depths else torch.zeros_like(timestep_colors[0])[..., :1].unsqueeze(0))
            normal_tensors.append(torch.stack(timestep_normals, dim=0) if timestep_normals else torch.zeros_like(timestep_colors[0])[..., :1].unsqueeze(0))
            optical_flow_tensors.append(torch.stack(timestep_flows, dim=0) if timestep_flows else torch.zeros_like(timestep_colors[0])[..., :1].unsqueeze(0))
            position_tensors.append(torch.stack(timestep_positions, dim=0) if timestep_positions else torch.zeros_like(timestep_colors[0])[..., :1].unsqueeze(0))

        meta_tensors['color'] = torch.stack(color_tensors, dim=0)
        meta_tensors['depth'] = torch.stack(depth_tensors, dim=0)
        meta_tensors['normal'] = torch.stack(normal_tensors, dim=0)
        meta_tensors['optical_flow'] = torch.stack(optical_flow_tensors, dim=0)
        meta_tensors['position'] = torch.stack(position_tensors, dim=0)

        return meta_tensors
