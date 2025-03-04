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
        # depth,
        # normal,
        # optical_flow,
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
        # self.depth = depth
        # self.normal = normal
        # self.optical_flow = optical_flow
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

        # if len(rust_view.depth) != 0:
        #     depth = reshape_data(rust_view.depth, np.float32)
        # else:
        #     depth = np.zeros(color.shape, dtype=np.float32)

        # if len(rust_view.normal) != 0:
        #     normal = reshape_data(rust_view.normal, np.float32)
        # else:
        #     normal = np.zeros(color.shape, dtype=np.float32)

        # if len(rust_view.optical_flow) != 0:
        #     optical_flow = reshape_data(rust_view.optical_flow, np.float32)
        # else:
        #     optical_flow = np.zeros(color.shape, dtype=np.float32)

        if len(rust_view.position) != 0:
            position = reshape_data(rust_view.position, np.float32)
        else:
            position = np.zeros(color.shape, dtype=np.float32)

        world_from_view = np.array(rust_view.world_from_view)
        fovy = rust_view.fovy
        near = rust_view.near
        far = rust_view.far
        time = rust_view.time
        return cls(
            color,
            # depth,
            # normal,
            # optical_flow,
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
        color_tensor = torch.tensor(self.color, dtype=torch.float32)
        color_tensor = color_tensor[..., :3]

        # depth_tensor = torch.tensor(self.depth, dtype=torch.float32)
        # depth_tensor = depth_tensor[..., 0:1]

        # normal_tensor = torch.tensor(self.normal, dtype=torch.float32)
        # normal_tensor = normal_tensor[..., :3]

        # optical_flow_tensor = torch.tensor(self.optical_flow, dtype=torch.float32)
        # optical_flow_tensor = optical_flow_tensor[..., :3]

        position_tensor = torch.tensor(self.position, dtype=torch.float32)
        position_tensor = position_tensor[..., :3]

        world_from_view_tensor = torch.tensor(self.world_from_view, dtype=torch.float32)

        fovy_tensor = torch.tensor(self.fovy, dtype=torch.float32).unsqueeze(-1)
        near_tensor = torch.tensor(self.near, dtype=torch.float32).unsqueeze(-1)
        far_tensor = torch.tensor(self.far, dtype=torch.float32).unsqueeze(-1)
        time_tensor = torch.tensor(self.time, dtype=torch.float32).unsqueeze(-1)

        return {
            'color': color_tensor,
            # 'depth': depth_tensor,
            # 'normal': normal_tensor,
            # 'optical_flow': optical_flow_tensor,
            'position': position_tensor,
            'world_from_view': world_from_view_tensor,
            'fovy': fovy_tensor,
            'near': near_tensor,
            'far': far_tensor,
            'time': time_tensor,
        }

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
        tensor_dict = {
            'color': [],
            # 'depth': [],
            # 'normal': [],
            # 'optical_flow': [],
            'position': [],
            'world_from_view': [],
            'fovy': [],
            'near': [],
            'far': [],
            'time': [],
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

            new_shape = (-1, self.view_dim, *tensor_dict[key].shape[1:])
            tensor_dict[key] = tensor_dict[key].reshape(new_shape)

        tensor_dict['aabb'] = torch.tensor(self.aabb, dtype=torch.float32)

        tensor_dict['color'] = normalize_hdr_image_tonemap(tensor_dict['color'])
        # tensor_dict['depth'] = tensor_dict['depth']
        # tensor_dict['normal'] = tensor_dict['normal']
        # tensor_dict['optical_flow'] = tensor_dict['optical_flow']
        tensor_dict['position'] = tensor_dict['position']

        return tensor_dict


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

        if 'color' in dataset.render_modes:
            color_tensor = sample['color']

        if 'depth' in dataset.render_modes:
            depth_tensor = sample['depth']

        if 'normal' in dataset.render_modes:
            normal_tensor = sample['normal']

        if 'optical_flow' in dataset.render_modes:
            optical_flow_tensor = sample['optical_flow']

        if 'position' in dataset.render_modes:
            position_tensor = sample['position']

        for timestep in range(color_tensor.shape[0]):
            for view_idx in range(color_tensor.shape[1]):
                if 'color' in dataset.render_modes:
                    view_color = color_tensor[timestep, view_idx].permute(2, 0, 1)
                    image_filename = scene_dir / f"color_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_color, str(image_filename))

                if 'depth' in dataset.render_modes:
                    view_depth = normalize_hdr_image_tonemap(depth_tensor)[timestep, view_idx].permute(2, 0, 1)
                    depth_filename = scene_dir / f"depth_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_depth, str(depth_filename))

                    view_depth_flat = depth_tensor[timestep, view_idx][0]
                    depth_npz_filename = scene_dir / f"depth_{timestep:03d}_{view_idx:02d}.npz"
                    np.savez(depth_npz_filename, depth=view_depth_flat.cpu().numpy())

                if 'normal' in dataset.render_modes:
                    view_normal = normalize_hdr_image_tonemap(normal_tensor)[timestep, view_idx].permute(2, 0, 1)
                    normal_filename = scene_dir / f"normal_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_normal, str(normal_filename))

                    view_normal_flat = normal_tensor[timestep, view_idx][0]
                    normal_npz_filename = scene_dir / f"normal_{timestep:03d}_{view_idx:02d}.npz"
                    np.savez(normal_npz_filename, normal=view_normal_flat.cpu().numpy())

                if 'optical_flow' in dataset.render_modes:
                    view_optical_flow = normalize_hdr_image_tonemap(optical_flow_tensor)[timestep, view_idx].permute(2, 0, 1)
                    optical_flow_filename = scene_dir / f"optical_flow_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_optical_flow, str(optical_flow_filename))

                # TODO: save motion vectors as npz, not optical flow rgb
                # view_optical_flow_flat = view_optical_flow[0]
                # optical_flow_npz_filename = scene_dir / f"optical_flow_{timestep:03d}_{view_idx:02d}.npz"
                # np.savez(optical_flow_npz_filename, optical_flow=view_optical_flow_flat.cpu().numpy())

                if 'position' in dataset.render_modes:
                    view_position = normalize_hdr_image_tonemap(position_tensor)[timestep, view_idx].permute(2, 0, 1)
                    position_filename = scene_dir / f"position_{timestep:03d}_{view_idx:02d}.jpg"
                    save_image(view_position, str(position_filename))

                    view_position_flat = position_tensor[timestep, view_idx][0]
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
