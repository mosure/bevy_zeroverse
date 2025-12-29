from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from itertools import takewhile
import gc
import json
import math
import os
from pathlib import Path
from PIL import Image
import random
from typing import Any, Literal, Optional

import imageio
import numpy as np
from safetensors import safe_open
from safetensors.torch import load as load_bytes, load_file, save as serialize, save_file
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, IterableDataset, get_worker_info, default_collate
from torchvision.io import decode_jpeg
import torchvision.transforms as transforms
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.utils import save_image
from tqdm.auto import tqdm

import lz4.frame as lz4
import zstandard as zstd

import bevy_zeroverse_ffi


def _as_numpy_array(data, dtype, width: Optional[int] = None) -> np.ndarray:
    """
    Coerce python/ffi payloads (lists, bytes, tensors) into a predictable numpy array.
    Handles ragged ovoxel fields that may arrive as bytes objects.
    """
    if data is None:
        if width is None:
            return np.zeros((0,), dtype=dtype)
        return np.zeros((0, width), dtype=dtype)

    if isinstance(data, (bytes, bytearray, memoryview)):
        arr = np.frombuffer(data, dtype=dtype).copy()
        if width:
            arr = arr.reshape(-1, width)
        return arr

    if isinstance(data, (list, tuple)) and len(data) > 0 and isinstance(data[0], (bytes, bytearray, memoryview)):
        inferred = width or len(data[0])
        blob = b"".join(data)
        arr = np.frombuffer(blob, dtype=dtype).copy()
        if inferred:
            arr = arr.reshape(len(data), inferred)
        return arr

    arr = np.asarray(data, dtype=dtype)
    if width:
        arr = arr.reshape(-1, width)
    return arr


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

        if len(rust_view.color) != 0:
            color = reshape_data(rust_view.color, np.float32)
        else:
            color = None

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

        if self.color is not None:
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
    def __init__(
        self,
        views,
        view_dim,
        aabb,
        object_obbs,
        human_poses,
        human_pose_steps,
        human_bone_names,
        human_bone_parents,
        ovoxel=None,
    ):
        self.views = views
        self.view_dim = view_dim
        self.aabb = aabb
        self.object_obbs = object_obbs
        self.human_poses = human_poses
        self.human_pose_steps = human_pose_steps or []
        self.human_bone_names = human_bone_names
        self.human_bone_parents = human_bone_parents
        self.ovoxel = ovoxel

    @classmethod
    def from_rust(cls, rust_sample, width, height):
        if hasattr(rust_sample, "take_views"):
            rust_views = rust_sample.take_views()
        else:
            rust_views = rust_sample.views

        views = [View.from_rust(view, width, height) for view in rust_views]
        view_dim = rust_sample.view_dim
        aabb = np.array(rust_sample.aabb)
        ovoxel = None
        if hasattr(rust_sample, "ovoxel") and rust_sample.ovoxel is not None:
            ov = rust_sample.ovoxel
            coords = _as_numpy_array(getattr(ov, "coords", None), np.uint32, width=3)
            dual_vertices = _as_numpy_array(getattr(ov, "dual_vertices", None), np.uint8, width=3)
            intersected = _as_numpy_array(getattr(ov, "intersected", None), np.uint8).reshape(-1)
            base_color = _as_numpy_array(getattr(ov, "base_color", None), np.uint8, width=4)
            semantic = _as_numpy_array(getattr(ov, "semantics", None), np.uint16).reshape(-1)

            semantic_labels = list(getattr(ov, "semantic_labels", []))
            semantic_label_bytes = json.dumps(semantic_labels).encode("utf-8") if semantic_labels else b""
            semantic_label_tensor = (
                np.frombuffer(semantic_label_bytes, dtype=np.uint8)
                if len(semantic_label_bytes) > 0
                else np.zeros((0,), dtype=np.uint8)
            )

            ovoxel = {
                "coords": coords,
                "dual_vertices": dual_vertices,
                "intersected": intersected,
                "base_color": base_color,
                "semantic": semantic,
                "semantic_labels": semantic_label_tensor,
                "semantic_label_offsets": np.array([[0, semantic_label_tensor.shape[0]]], dtype=np.int64),
                "offsets": np.array([[0, coords.shape[0]]], dtype=np.int64),
                "resolution": np.array([getattr(ov, "resolution", 0)], dtype=np.uint32),
                "aabb": np.array([getattr(ov, "aabb", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])], dtype=np.float32),
            }
        object_obbs = []
        if hasattr(rust_sample, "object_obbs"):
            for obb in rust_sample.object_obbs:
                object_obbs.append(
                    {
                        'center': np.array(obb.center, dtype=np.float32),
                        'scale': np.array(obb.scale, dtype=np.float32),
                        'rotation': np.array(obb.rotation, dtype=np.float32),
                        'class_name': obb.class_name,
                    }
                )
        human_poses = []
        if hasattr(rust_sample, "human_poses"):
            for pose in rust_sample.human_poses:
                human_poses.append(
                    {
                        "bone_positions": np.array(pose.bone_positions, dtype=np.float32),
                        "bone_rotations": np.array(pose.bone_rotations, dtype=np.float32),
                    }
                )
        human_pose_steps = []
        if hasattr(rust_sample, "human_pose_steps"):
            for step in rust_sample.human_pose_steps:
                step_poses = []
                for pose in step:
                    step_poses.append(
                        {
                            "bone_positions": np.array(pose.bone_positions, dtype=np.float32),
                            "bone_rotations": np.array(pose.bone_rotations, dtype=np.float32),
                        }
                    )
                human_pose_steps.append(step_poses)
        human_bone_names = (
            list(getattr(rust_sample, "human_bone_names", []))
            if hasattr(rust_sample, "human_bone_names")
            else []
        )
        human_bone_parents = (
            list(getattr(rust_sample, "human_bone_parents", []))
            if hasattr(rust_sample, "human_bone_parents")
            else []
        )
        # drop rust_views to release lock earlier
        del rust_views
        return cls(
            views,
            view_dim,
            aabb,
            object_obbs,
            human_poses,
            human_pose_steps,
            human_bone_names,
            human_bone_parents,
            ovoxel,
        )

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
        if self.ovoxel is not None:
            sample['ovoxel_coords'] = torch.as_tensor(self.ovoxel["coords"], dtype=torch.int32)
            sample['ovoxel_dual_vertices'] = torch.as_tensor(self.ovoxel["dual_vertices"], dtype=torch.uint8)
            sample['ovoxel_intersected'] = torch.as_tensor(self.ovoxel["intersected"], dtype=torch.uint8)
            sample['ovoxel_base_color'] = torch.as_tensor(self.ovoxel["base_color"], dtype=torch.uint8)
            sample['ovoxel_semantic'] = torch.as_tensor(self.ovoxel["semantic"], dtype=torch.int32)
            labels_np = np.array(self.ovoxel["semantic_labels"], copy=True)
            sample['ovoxel_semantic_labels'] = torch.tensor(labels_np, dtype=torch.uint8)
            sample['ovoxel_semantic_label_offsets'] = torch.as_tensor(self.ovoxel["semantic_label_offsets"], dtype=torch.int64)
            sample['ovoxel_offsets'] = torch.as_tensor(self.ovoxel["offsets"], dtype=torch.int64)
            sample['ovoxel_resolution'] = torch.as_tensor(self.ovoxel["resolution"], dtype=torch.int32)
            sample['ovoxel_aabb'] = torch.as_tensor(self.ovoxel["aabb"], dtype=torch.float32)
            self.ovoxel = None

        if len(self.object_obbs) > 0:
            class_to_idx = {}
            centers = []
            scales = []
            rotations = []
            class_idxs = []
            for obb in self.object_obbs:
                cls_idx = class_to_idx.setdefault(obb['class_name'], len(class_to_idx))
                centers.append(obb['center'])
                scales.append(obb['scale'])
                rotations.append(obb['rotation'])
                class_idxs.append(cls_idx)

            sample['object_obb_center'] = torch.tensor(np.stack(centers, axis=0), dtype=torch.float32)
            sample['object_obb_scale'] = torch.tensor(np.stack(scales, axis=0), dtype=torch.float32)
            sample['object_obb_rotation'] = torch.tensor(np.stack(rotations, axis=0), dtype=torch.float32)
            sample['object_obb_class_idx'] = torch.tensor(class_idxs, dtype=torch.int64)
            names_bytes = json.dumps(list(class_to_idx.keys())).encode("utf-8")
            sample['object_obb_class_names'] = torch.tensor(list(names_bytes), dtype=torch.uint8)

        if self.human_pose_steps:
            steps = 1
            if 'time' in sample:
                steps = int(sample['time'].shape[0])
            elif self.view_dim:
                steps = max(1, len(self.views) // int(self.view_dim))

            pose_steps = list(self.human_pose_steps)

            if len(pose_steps) < steps:
                pad_step = pose_steps[-1] if pose_steps else []
                pose_steps.extend([pad_step for _ in range(steps - len(pose_steps))])
            elif len(pose_steps) > steps:
                pose_steps = pose_steps[:steps]

            bone_count = len(self.human_bone_names)
            max_humans = 0
            for step in pose_steps:
                max_humans = max(max_humans, len(step))
                for pose in step:
                    pos = _as_numpy_array(pose["bone_positions"], np.float32, width=3)
                    rot = _as_numpy_array(pose["bone_rotations"], np.float32, width=4)
                    bone_count = max(bone_count, pos.shape[0], rot.shape[0])

            if max_humans > 0 and bone_count > 0:
                pose_positions = np.zeros((steps, max_humans, bone_count, 3), dtype=np.float32)
                pose_rotations = np.zeros((steps, max_humans, bone_count, 4), dtype=np.float32)

                for step_idx, step in enumerate(pose_steps):
                    for human_idx, pose in enumerate(step[:max_humans]):
                        pos = _as_numpy_array(pose["bone_positions"], np.float32, width=3)
                        rot = _as_numpy_array(pose["bone_rotations"], np.float32, width=4)
                        if pos.shape[0] < bone_count:
                            pad = np.zeros((bone_count - pos.shape[0], 3), dtype=np.float32)
                            pos = np.concatenate([pos, pad], axis=0)
                        if rot.shape[0] < bone_count:
                            pad = np.zeros((bone_count - rot.shape[0], 4), dtype=np.float32)
                            rot = np.concatenate([rot, pad], axis=0)
                        pose_positions[step_idx, human_idx, :, :] = pos
                        pose_rotations[step_idx, human_idx, :, :] = rot

                sample['human_pose_position'] = torch.tensor(pose_positions, dtype=torch.float32)
                sample['human_pose_rotation'] = torch.tensor(pose_rotations, dtype=torch.float32)

                if len(self.human_bone_names) > 0:
                    names_bytes = json.dumps(self.human_bone_names).encode("utf-8")
                    sample['human_pose_bone_names'] = torch.tensor(list(names_bytes), dtype=torch.uint8)
                if len(self.human_bone_parents) > 0:
                    sample['human_pose_bone_parents'] = torch.tensor(self.human_bone_parents, dtype=torch.int64)

        normalize_keys = ['color', 'optical_flow']
        for key in normalize_keys:
            if key in sample:
                sample[key] = normalize_hdr_image_tonemap(sample[key])

        self.views.clear()
        self.object_obbs.clear()
        self.human_poses.clear()
        self.human_pose_steps.clear()

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
        'human': bevy_zeroverse_ffi.ZeroverseSceneType.Human,
        'object': bevy_zeroverse_ffi.ZeroverseSceneType.Object,
        'room': bevy_zeroverse_ffi.ZeroverseSceneType.Room,
        'semantic_room': bevy_zeroverse_ffi.ZeroverseSceneType.SemanticRoom,
    }

    ov_mode_map = {
        "disabled": bevy_zeroverse_ffi.OvoxelMode.Disabled,
        "cpu_async": bevy_zeroverse_ffi.OvoxelMode.CpuAsync,
        "gpu_compute": bevy_zeroverse_ffi.OvoxelMode.GpuCompute,
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
        rotation_augmentation=True,
        regenerate_scene_material_shuffle_period=256,
        cuboid_only=False,
        ovoxel_mode: str = "cpu_async",
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
        self.rotation_augmentation = rotation_augmentation
        self.regenerate_scene_material_shuffle_period = regenerate_scene_material_shuffle_period
        self.cuboid_only = cuboid_only
        self.ovoxel_mode = ovoxel_mode

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
        config.regenerate_scene_material_shuffle_period = self.regenerate_scene_material_shuffle_period
        config.playback_step = self.playback_step
        config.playback_steps = self.playback_steps
        config.render_modes = [BevyZeroverseDataset.render_mode_map[mode] for mode in self.render_modes]
        config.render_mode = config.render_modes[0]
        config.rotation_augmentation = self.rotation_augmentation
        config.cuboid_only = self.cuboid_only
        config.ovoxel_mode = BevyZeroverseDataset.ov_mode_map[self.ovoxel_mode]
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
        del rust_sample
        return sample.to_tensors()


def _compress(data: bytes, codec: str, level: int = 0) -> bytes:
    if codec == "lz4":
        return lz4.compress(data, compression_level=level)
    if codec == "zstd":
        c = zstd.ZstdCompressor(level=level)
        return c.compress(data)
    return data

def _decompress_bytes(path: Path) -> bytes:
    raw = path.read_bytes()
    if path.suffix == ".lz4":
        return lz4.decompress(raw)
    if path.suffix == ".zst":
        d = zstd.ZstdDecompressor()
        return d.decompress(raw)
    return raw


def decode_jpg_tensor(jpg_tensor: torch.Tensor) -> torch.Tensor:
    buffer = BytesIO(jpg_tensor.numpy().tobytes())
    img = Image.open(buffer).convert("RGB")
    return to_tensor(img).permute(1, 2, 0)

def encode_tensor_to_jpg(tensor: torch.Tensor, quality: int = 75) -> torch.Tensor:
    buffer = BytesIO()
    img = to_pil_image(tensor.permute(2, 0, 1))
    img.save(buffer, format="JPEG", quality=quality)
    return torch.frombuffer(bytearray(buffer.getvalue()), dtype=torch.uint8)


def numeric_prefix(s: str) -> int | None:
    digits = ''.join(takewhile(str.isdigit, s))
    return int(digits) if digits else None

def chunk_and_save(
    dataset,
    output_dir: Path,
    bytes_per_chunk: Optional[int] = int(256 * 1024 * 1024),
    samples_per_chunk: Optional[int] = None,
    n_workers: int = 0,
    jpg_quality: int = 75,
    compression: Optional[Literal["lz4", "zstd"]] = "lz4",
    full_size_only: bool = True,
    prefetch_factor: int = 2,
    pin_memory: bool = False,
    persistent_workers: bool = True,
    memory_cleanup: bool = False,
):
    """Save samples to chunk files, optionally batching by size or sample count."""

    output_dir.mkdir(exist_ok=True, parents=True)

    cache_file = output_dir / "chunk_sizes_cache.json"
    if cache_file.exists():
        os.remove(cache_file)

    existing_chunks = sorted(output_dir.glob("*.safetensors*"))
    if existing_chunks:
        latest_chunk = existing_chunks[-1]
        chunk_index = numeric_prefix(latest_chunk.stem)
        print(f"resuming from chunk {chunk_index}.")
    else:
        chunk_index = 0

    chunk_size = 0
    chunk: list[dict[str, torch.Tensor]] = []
    chunk_file_paths = list(existing_chunks)

    total_chunks = len(dataset)
    if samples_per_chunk is not None and samples_per_chunk > 0:
        total_chunks = max(1, len(dataset) // samples_per_chunk)

    def estimate_chunk_bytes(chunk_samples: list[dict[str, torch.Tensor]]) -> int:
        return sum(t.numel() * t.element_size() for sample in chunk_samples for t in sample.values())

    def save_chunk_sync(
        chunk_samples: list[dict[str, torch.Tensor]],
        chunk_idx: int,
    ) -> Path:
        key = f"{chunk_idx:0>6}"
        estimate_mb = estimate_chunk_bytes(chunk_samples) / 1e6
        print(f"saving chunk {key} of {total_chunks} ({estimate_mb:.2f} MB).")
        base_name = f"{key}.safetensors"

        batch: dict[str, Any] = {}
        offset = 0
        has_ovoxel = any(any(key.startswith("ovoxel_") for key in sample.keys()) for sample in chunk_samples)
        max_obbs = 0
        for sample in chunk_samples:
            center = sample.get("object_obb_center")
            if center is not None:
                max_obbs = max(max_obbs, int(center.shape[0]))
        max_humans = 0
        for sample in chunk_samples:
            pose = sample.get("human_pose_position")
            if pose is not None:
                if pose.dim() >= 4:
                    max_humans = max(max_humans, int(pose.shape[1]))
                else:
                    max_humans = max(max_humans, int(pose.shape[0]))

        ov_coords: list[torch.Tensor] = []
        ov_dual: list[torch.Tensor] = []
        ov_intersected: list[torch.Tensor] = []
        ov_base: list[torch.Tensor] = []
        ov_semantic: list[torch.Tensor] = []
        ov_offsets: list[torch.Tensor] = []
        ov_resolutions: list[torch.Tensor] = []
        ov_aabb: list[torch.Tensor] = []
        ov_label_offsets: list[torch.Tensor] = []
        ov_label_blob: bytearray = bytearray()
        ov_cursor = 0
        obb_class_names_tensor: torch.Tensor | None = None
        human_bone_names_tensor: torch.Tensor | None = None
        human_bone_parents_tensor: torch.Tensor | None = None

        def pad_obb_tensor(tensor: torch.Tensor, target: int, fill: float | int = 0) -> torch.Tensor:
            if target <= 0:
                return tensor.new_zeros((0, *tensor.shape[1:]), dtype=tensor.dtype)
            if tensor.shape[0] == target:
                return tensor
            pad_shape = (target - tensor.shape[0], *tensor.shape[1:])
            pad = torch.full(pad_shape, fill, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad], dim=0)

        def pad_pose_tensor(tensor: torch.Tensor, target: int) -> torch.Tensor:
            if target <= 0:
                if tensor.dim() >= 4:
                    return tensor.new_zeros((tensor.shape[0], 0, *tensor.shape[2:]), dtype=tensor.dtype)
                return tensor.new_zeros((0, *tensor.shape[1:]), dtype=tensor.dtype)
            if tensor.dim() >= 4:
                if tensor.shape[1] == target:
                    return tensor
                pad_shape = (tensor.shape[0], target - tensor.shape[1], *tensor.shape[2:])
                pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
                return torch.cat([tensor, pad], dim=1)
            if tensor.shape[0] == target:
                return tensor
            pad_shape = (target - tensor.shape[0], *tensor.shape[1:])
            pad = torch.zeros(pad_shape, dtype=tensor.dtype, device=tensor.device)
            return torch.cat([tensor, pad], dim=0)

        for sample in chunk_samples:
            for name, tensor in sample.items():
                if name in ["color"]:
                    T, V, H, W, C = tensor.shape
                    flat_views = tensor.view(-1, H, W, C)
                    for idx, view in enumerate(flat_views):
                        view_jpg = encode_tensor_to_jpg(view, quality=jpg_quality)
                        batch[f"{name}_jpg_{offset + idx}"] = view_jpg

                    if f"{name}_shape" not in batch:
                        batch[f"{name}_shape"] = torch.tensor([len(chunk_samples), T, V, H, W, C])
                    offset += flat_views.shape[0]
                elif name.startswith("object_obb_"):
                    if max_obbs == 0:
                        continue
                    if name == "object_obb_class_names":
                        if obb_class_names_tensor is None:
                            obb_class_names_tensor = tensor.cpu()
                        continue
                    tensor = tensor.cpu()
                    fill = -1 if name == "object_obb_class_idx" else 0
                    padded = pad_obb_tensor(tensor, max_obbs, fill=fill)
                    batch.setdefault(name, []).append(padded)
                    continue
                elif name.startswith("human_pose_"):
                    if max_humans == 0:
                        continue
                    if name == "human_pose_bone_names":
                        if human_bone_names_tensor is None:
                            human_bone_names_tensor = tensor.cpu()
                        continue
                    if name == "human_pose_bone_parents":
                        if human_bone_parents_tensor is None:
                            human_bone_parents_tensor = tensor.cpu()
                        continue
                    tensor = tensor.cpu()
                    padded = pad_pose_tensor(tensor, max_humans)
                    batch.setdefault(name, []).append(padded)
                    continue
                elif name.startswith("ovoxel_"):
                    continue
                else:
                    batch.setdefault(name, []).append(tensor)

            if has_ovoxel:
                coords = sample.get("ovoxel_coords")
                coords = coords.reshape(-1, 3).cpu() if coords is not None else None
                num_vox = int(coords.shape[0]) if coords is not None else 0
                ov_offsets.append(torch.tensor([ov_cursor, num_vox], dtype=torch.int64))
                ov_cursor += num_vox

                if coords is not None and num_vox > 0:
                    ov_coords.append(coords)
                    dual = sample.get("ovoxel_dual_vertices")
                    if dual is not None:
                        ov_dual.append(dual.reshape(num_vox, 3).cpu())
                    intersect = sample.get("ovoxel_intersected")
                    if intersect is not None:
                        ov_intersected.append(intersect.reshape(-1).cpu())
                    base = sample.get("ovoxel_base_color")
                    if base is not None:
                        ov_base.append(base.reshape(num_vox, 4).cpu())
                    semantic = sample.get("ovoxel_semantic")
                    if semantic is not None:
                        ov_semantic.append(semantic.reshape(-1).cpu())

                res = sample.get("ovoxel_resolution")
                res_tensor = res.reshape(-1)[0].cpu() if res is not None else torch.tensor(0, dtype=torch.int32)
                ov_resolutions.append(res_tensor)

                aabb = sample.get("ovoxel_aabb")
                aabb_tensor = aabb.reshape(1, 2, 3).cpu() if aabb is not None else torch.zeros((1, 2, 3), dtype=torch.float32)
                ov_aabb.append(aabb_tensor)

                labels = sample.get("ovoxel_semantic_labels")
                labels_tensor = labels.reshape(-1).to(torch.uint8).cpu() if labels is not None else None
                label_len = int(labels_tensor.numel()) if labels_tensor is not None else 0
                ov_label_offsets.append(torch.tensor([len(ov_label_blob), label_len], dtype=torch.int64))
                if labels_tensor is not None and label_len > 0:
                    ov_label_blob.extend(labels_tensor.numpy().tobytes())

        if has_ovoxel:
            batch["ovoxel_offsets"] = torch.stack(ov_offsets, dim=0)
            batch["ovoxel_resolution"] = torch.stack(ov_resolutions, dim=0)
            batch["ovoxel_aabb"] = torch.cat(ov_aabb, dim=0)
            batch["ovoxel_semantic_label_offsets"] = torch.stack(ov_label_offsets, dim=0)
            batch["ovoxel_semantic_labels"] = (
                torch.tensor(list(ov_label_blob), dtype=torch.uint8)
                if len(ov_label_blob) > 0
                else torch.zeros((0,), dtype=torch.uint8)
            )
            if len(ov_coords) > 0:
                batch["ovoxel_coords"] = torch.cat(ov_coords, dim=0)
            if len(ov_dual) > 0:
                batch["ovoxel_dual_vertices"] = torch.cat(ov_dual, dim=0)
            if len(ov_intersected) > 0:
                batch["ovoxel_intersected"] = torch.cat(ov_intersected, dim=0)
            if len(ov_base) > 0:
                batch["ovoxel_base_color"] = torch.cat(ov_base, dim=0)
            if len(ov_semantic) > 0:
                batch["ovoxel_semantic"] = torch.cat(ov_semantic, dim=0)
        if max_obbs > 0 and obb_class_names_tensor is not None:
            batch["object_obb_class_names"] = obb_class_names_tensor
        if max_humans > 0 and human_bone_names_tensor is not None:
            batch["human_pose_bone_names"] = human_bone_names_tensor
        if max_humans > 0 and human_bone_parents_tensor is not None:
            batch["human_pose_bone_parents"] = human_bone_parents_tensor

        flat_tensors: dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, list):
                flat_tensors[key] = torch.stack(value, dim=0)
            else:
                flat_tensors[key] = value

        if compression is None:
            final_path = output_dir / base_name
            save_file(flat_tensors, str(final_path))
        else:
            blob = serialize(flat_tensors)
            encoded = _compress(blob, compression)
            ext = ".lz4" if compression == "lz4" else ".zst"
            final_path = output_dir / (base_name + ext)
            final_path.write_bytes(encoded)

        if memory_cleanup:
            torch.cuda.empty_cache()
            gc.collect()
        return final_path

    max_save_workers = min(max(1, (os.cpu_count() or 4) // 2), 8)
    pending: list[Any] = []

    def collate_fn(samples: list[dict[str, Any]]) -> dict[str, Any]:
        if not samples:
            return {}
        collated: dict[str, Any] = {}
        ragged_prefixes = ("ovoxel_", "object_obb_", "human_pose_")
        for key in samples[0].keys():
            values = [s[key] for s in samples]
            if key.startswith(ragged_prefixes):
                collated[key] = values
            else:
                collated[key] = default_collate(values)
        return collated

    dl_bs = samples_per_chunk if samples_per_chunk else 1
    dataloader_kwargs = dict(
        batch_size=dl_bs,
        num_workers=n_workers,
        shuffle=False,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
    if n_workers > 0:
        dataloader_kwargs["prefetch_factor"] = max(1, prefetch_factor)
        dataloader_kwargs["persistent_workers"] = persistent_workers
    dataloader = DataLoader(dataset, **dataloader_kwargs)
    pbar = tqdm(total=len(dataset), unit="sample", desc="processing", smoothing=0.99)

    with ThreadPoolExecutor(max_workers=max_save_workers) as executor:
        for idx, batch in enumerate(dataloader):
            first_val = next(iter(batch.values()))
            bsz = len(first_val) if isinstance(first_val, list) else first_val.shape[0]
            pbar.update(bsz)
            for i in range(bsz):
                sample = {
                    name: (tensor[i] if not isinstance(tensor, list) else tensor[i])
                    for name, tensor in batch.items()
                }
                sample_size = sum(t.numel() * t.element_size() for t in sample.values())
                chunk.append(sample)
                chunk_size += sample_size

                flush_by_samples = samples_per_chunk and len(chunk) >= samples_per_chunk
                flush_by_bytes = not samples_per_chunk and chunk_size >= bytes_per_chunk

                if flush_by_samples or flush_by_bytes:
                    chunk_copy = chunk
                    current_index = chunk_index
                    future = executor.submit(save_chunk_sync, chunk_copy, current_index)
                    pending.append(future)
                    chunk = []
                    chunk_size = 0
                    chunk_index += 1

            pbar.set_postfix(chunk_mb=f"{chunk_size/1e6:,.1f}", idx=idx)

        if chunk_size > 0 and not full_size_only:
            chunk_copy = chunk
            current_index = chunk_index
            future = executor.submit(save_chunk_sync, chunk_copy, current_index)
            pending.append(future)
            chunk = []
            chunk_index += 1

        for future in pending:
            final_path = future.result()
            chunk_file_paths.append(final_path)

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


# TODO: support optional compression of chunks
def load_chunk(path: Path):
    raw = _decompress_bytes(path)
    meta = {}
    try:
        with safe_open(BytesIO(raw), framework="pt", device="cpu") as f:
            meta = f.metadata() or {}
    except Exception:
        meta = {}
    tensors = load_bytes(raw)
    batch = {}

    jpeg_groups = {}
    for key in tensors:
        if '_jpg_' in key:
            parent, idx = key.rsplit('_jpg_', 1)
            jpeg_groups.setdefault(parent, []).append((int(idx), tensors[key]))

    for parent, indexed_jpegs in jpeg_groups.items():
        indexed_jpegs.sort()
        shape = tensors[f'{parent}_shape'].tolist()
        images_count = shape[0] * shape[1] * shape[2]

        jpeg_data = [data for _, data in indexed_jpegs[:images_count]]
        decoded_images = decode_jpeg(
            jpeg_data,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            mode='RGB',
        )
        decoded_images = [
            img.to('cpu').float().div(255.0).permute(1, 2, 0) for img in decoded_images
        ]

        height = min(img.shape[0] for img in decoded_images)
        width = min(img.shape[1] for img in decoded_images)
        perform_crop = any(img.shape[0] != height or img.shape[1] != width for img in decoded_images)

        if perform_crop:
            decoded_images = [
                crop(img, (height, width)) for img in decoded_images
            ]

        shape[3] = height
        shape[4] = width

        batch[parent] = torch.stack(decoded_images).reshape(shape)

    for key, tensor in tensors.items():
        if '_jpg_' not in key and '_shape' not in key:
            batch[key] = tensor

    if "object_obb_class_names" in meta:
        try:
            names_bytes = meta["object_obb_class_names"].encode("utf-8")
            batch["object_obb_class_names"] = torch.tensor(list(names_bytes), dtype=torch.uint8)
        except Exception:
            pass
    elif "object_obb_class_names" in tensors:
        batch["object_obb_class_names"] = tensors["object_obb_class_names"]

    # Optional batched O-Voxel tensors (parity with Rust generator).
    if "ovoxel_coords" in tensors:
        batch["ovoxel_coords"] = tensors["ovoxel_coords"]
        if "ovoxel_dual_vertices" in tensors:
            batch["ovoxel_dual_vertices"] = tensors["ovoxel_dual_vertices"]
        if "ovoxel_intersected" in tensors:
            batch["ovoxel_intersected"] = tensors["ovoxel_intersected"]
        if "ovoxel_base_color" in tensors:
            batch["ovoxel_base_color"] = tensors["ovoxel_base_color"]
        if "ovoxel_semantic" in tensors:
            batch["ovoxel_semantic"] = tensors["ovoxel_semantic"]
        if "ovoxel_offsets" in tensors:
            batch["ovoxel_offsets"] = tensors["ovoxel_offsets"]
        if "ovoxel_resolution" in tensors:
            batch["ovoxel_resolution"] = tensors["ovoxel_resolution"]
        if "ovoxel_aabb" in tensors:
            batch["ovoxel_aabb"] = tensors["ovoxel_aabb"]
        if "ovoxel_semantic_labels" in tensors:
            batch["ovoxel_semantic_labels"] = tensors["ovoxel_semantic_labels"]
        if "ovoxel_semantic_label_offsets" in tensors:
            batch["ovoxel_semantic_label_offsets"] = tensors["ovoxel_semantic_label_offsets"]

    return batch


def load_single_sample(
    chunk_path: Path,
    sample_idx: int,
):
    chunk = load_chunk(chunk_path)
    meta_keys = {
        "object_obb_class_names",
        "human_pose_bone_names",
        "human_pose_bone_parents",
    }
    sample = {
        k: v[sample_idx]
        for k, v in chunk.items()
        if k not in meta_keys
    }
    for key in meta_keys:
        if key in chunk:
            sample[key] = chunk[key]
    sample['_chunk_path'] = str(chunk_path)
    sample['_sample_idx'] = sample_idx
    return sample


def get_chunk_sample_count(path: Path):
    if path.suffix in {".lz4", ".zst"}:
        tensors = load_bytes(_decompress_bytes(path))
    else:
        tensors = load_file(str(path))

    if "color_shape" in tensors:
        return tensors["color_shape"][0].item()
    if "near" in tensors:
        return tensors["near"].shape[0]

    raise ValueError("no shape key found in chunk file")



class ChunkedIteratorDataset(IterableDataset):
    def __init__(self, output_dir: Path, shuffle: bool = False):
        self.output_dir = Path(output_dir)
        self.shuffle = shuffle
        self.cache_file = self.output_dir / "chunk_sizes_cache.json"

        self.chunk_files: list[Path] = []
        self.chunk_sizes: list[int] = []
        self.total_samples: int = 0

        if self.cache_file.exists():
            try:
                with open(self.cache_file, "r") as f:
                    cache = json.load(f)

                self.chunk_files = [Path(p) for p in cache.get("files", [])]
                self.chunk_sizes = cache.get("chunk_sizes", [])

                if (
                    not self.chunk_files
                    or not self.chunk_sizes
                    or len(self.chunk_files) != len(self.chunk_sizes)
                    or any(not p.exists() for p in self.chunk_files)
                ):
                    self._refresh_cache()
                else:
                    self.total_samples = sum(self.chunk_sizes)
            except Exception:
                self._refresh_cache()
        else:
            self._refresh_cache()


    def _refresh_cache(self):
        self.chunk_files = sorted(self.output_dir.glob("*.safetensors*"))
        self.chunk_sizes = []
        self.total_samples = 0
        for chunk_file in self.chunk_files:
            samples = get_chunk_sample_count(chunk_file)
            self.chunk_sizes.append(samples)
            self.total_samples += samples

        cache_data = {
            "files": [str(f.resolve()) for f in self.chunk_files],
            "chunk_sizes": self.chunk_sizes,
        }
        with open(self.cache_file, "w") as f:
            json.dump(cache_data, f)


    def __iter__(self):
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info else 1
        worker_id = worker_info.id if worker_info else 0

        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0

        total_workers = world_size * num_workers
        worker_global_id = rank * num_workers + worker_id

        ideal_samples_per_worker = self.total_samples // total_workers

        worker_chunks = [[] for _ in range(total_workers)]
        worker_samples = [0] * total_workers

        current_worker = 0
        for idx, chunk_size in enumerate(self.chunk_sizes):
            if worker_samples[current_worker] + chunk_size > ideal_samples_per_worker:
                current_worker += 1
                if current_worker >= total_workers:
                    break

            worker_chunks[current_worker].append(idx)
            worker_samples[current_worker] += chunk_size

        min_assigned_samples = min(worker_samples)
        chunk_files = worker_chunks[worker_global_id]

        if self.shuffle:
            random.shuffle(chunk_files)

        emitted_samples = 0

        for chunk_idx in chunk_files:
            chunk_path = self.chunk_files[chunk_idx]
            chunk_data = load_chunk(chunk_path)
            ov_offsets = chunk_data.get("ovoxel_offsets")
            ov_label_offsets = chunk_data.get("ovoxel_semantic_label_offsets")
            ov_labels = chunk_data.get("ovoxel_semantic_labels")
            ov_coords = chunk_data.get("ovoxel_coords")
            ov_dual = chunk_data.get("ovoxel_dual_vertices")
            ov_intersected = chunk_data.get("ovoxel_intersected")
            ov_base = chunk_data.get("ovoxel_base_color")
            ov_semantic = chunk_data.get("ovoxel_semantic")
            ov_resolution = chunk_data.get("ovoxel_resolution")
            ov_aabb = chunk_data.get("ovoxel_aabb")
            local_indices = list(range(self.chunk_sizes[chunk_idx]))

            if self.shuffle:
                random.shuffle(local_indices)

            for sample_idx in local_indices:
                if emitted_samples >= min_assigned_samples:
                    return
                meta_keys = {
                    "object_obb_class_names",
                    "human_pose_bone_names",
                    "human_pose_bone_parents",
                }
                sample = {
                    key: tensor[sample_idx]
                    for key, tensor in chunk_data.items()
                    if not key.startswith("ovoxel_") and key not in meta_keys
                }
                for key in meta_keys:
                    if key in chunk_data:
                        sample[key] = chunk_data[key]

                if ov_offsets is not None and sample_idx < len(ov_offsets):
                    start, length = ov_offsets[sample_idx].tolist()
                    start = int(start)
                    length = int(length)

                    sample["ovoxel_offsets"] = ov_offsets[sample_idx]
                    if ov_resolution is not None:
                        sample["ovoxel_resolution"] = ov_resolution[sample_idx]
                    if ov_aabb is not None:
                        sample["ovoxel_aabb"] = ov_aabb[sample_idx]
                    if ov_coords is not None and length > 0:
                        sample["ovoxel_coords"] = ov_coords[start:start + length]
                    if ov_dual is not None and length > 0:
                        sample["ovoxel_dual_vertices"] = ov_dual[start:start + length]
                    if ov_intersected is not None and length > 0:
                        sample["ovoxel_intersected"] = ov_intersected[start:start + length]
                    if ov_base is not None and length > 0:
                        sample["ovoxel_base_color"] = ov_base[start:start + length]
                    if ov_semantic is not None and length > 0:
                        sample["ovoxel_semantic"] = ov_semantic[start:start + length]
                    if ov_label_offsets is not None and ov_labels is not None:
                        l_start, l_len = ov_label_offsets[sample_idx].tolist()
                        l_start = int(l_start)
                        l_len = int(l_len)
                        sample["ovoxel_semantic_label_offsets"] = ov_label_offsets[sample_idx]
                        sample["ovoxel_semantic_labels"] = ov_labels[l_start:l_start + l_len]

                sample["_chunk_path"] = str(chunk_path)
                sample["_sample_idx"] = sample_idx
                yield sample
                emitted_samples += 1



def write_sample(sample: dict, *, jpg_quality: int = 75) -> None:
    """
    overwrite *one* sample inside its original chunk.

    required keys (they are removed before saving):
        sample["_chunk_path"] : Path to the chunk file
        sample["_sample_idx"]  : integer position of the sample in that chunk
    any remaining (key, tensor) pairs are written back.

    jpeg-backed groups (e.g. `color`) touch ONLY the views that belong to
    `_sample_idx`; the rest of the chunk stays byte-for-byte identical.
    """

    assert {"_chunk_path", "_sample_idx"} <= sample.keys(), \
        "`_chunk_path` and `_sample_idx` must be present"
    meta_keys = {
        "object_obb_class_names",
        "human_pose_bone_names",
        "human_pose_bone_parents",
    }
    for key in list(sample.keys()):
        if key.startswith("ovoxel_") or key in meta_keys:
            sample.pop(key)

    chunk_path = Path(sample.pop("_chunk_path"))
    idx = int(sample.pop("_sample_idx"))

    codec = (
        "lz4" if chunk_path.suffix == ".lz4" else
        "zstd" if chunk_path.suffix == ".zst" else
        None
    )

    tensors = load_bytes(_decompress_bytes(chunk_path))

    def _update_jpeg_group(parent: str, tensor: torch.Tensor) -> None:
        shape = tensors[f"{parent}_shape"].tolist()  # [B,T,V,H,W,C]
        B, T, V, H, W, C = shape
        assert idx < B, f"idx {idx} >= batch {B}"

        n_views = T * V
        offset = idx * n_views
        views = tensor.view(n_views, H, W, C)

        for v_i, view in enumerate(views):
            key = f"{parent}_jpg_{offset + v_i}"
            tensors[key] = encode_tensor_to_jpg(view, quality=jpg_quality)

    for key, val in sample.items():
        if key in tensors:
            tensors[key][idx] = val.to(tensors[key].dtype)
            continue

        shape_key = f"{key}_shape"
        if shape_key in tensors:
            _update_jpeg_group(key, val)
            continue

        raise KeyError(f"unrecognised tensor key: {key}")

    blob = serialize(tensors)
    data_out = _compress(blob, codec) if codec else blob
    chunk_path.write_bytes(data_out)



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
        for key in ['object_obb_center', 'object_obb_scale', 'object_obb_rotation', 'object_obb_class_idx']:
            if key in sample:
                meta_tensors[key] = sample[key]
        if 'object_obb_class_names' in sample:
            names_val = sample['object_obb_class_names']
            if isinstance(names_val, torch.Tensor):
                decoded = []
                if names_val.numel() > 0:
                    decoded = json.loads(bytes(names_val.cpu().tolist()).decode("utf-8"))
            else:
                decoded = names_val
            names_bytes = json.dumps(decoded).encode('utf-8')
            meta_tensors['object_obb_class_names'] = torch.tensor(list(names_bytes), dtype=torch.uint8)
        for key in ['human_pose_position', 'human_pose_rotation', 'human_pose_bone_parents']:
            if key in sample:
                meta_tensors[key] = sample[key]
        if 'human_pose_bone_names' in sample:
            names_val = sample['human_pose_bone_names']
            if isinstance(names_val, torch.Tensor):
                decoded = []
                if names_val.numel() > 0:
                    decoded = json.loads(bytes(names_val.cpu().tolist()).decode("utf-8"))
            else:
                decoded = names_val
            names_bytes = json.dumps(decoded).encode('utf-8')
            meta_tensors['human_pose_bone_names'] = torch.tensor(list(names_bytes), dtype=torch.uint8)
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


def prepare_video_frames(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dim() == 3:
        if tensor.shape[0] in [1, 3, 4]:
            tensor = tensor.permute(1, 2, 0)
        elif tensor.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Unexpected channel size: {tensor.shape[2]}")
    else:
        raise ValueError(f"Unexpected tensor dimensions: {tensor.dim()}")

    tensor = tensor.numpy()

    if tensor.shape[2] == 1:
        tensor = np.repeat(tensor, 3, axis=2)
    elif tensor.shape[2] > 3:
        tensor = tensor[:, :, :3]

    if tensor.dtype != np.uint8:
        tensor = (tensor * 255).astype(np.uint8)

    return tensor

def save_to_mp4(dataset, output_dir: Path, fps: int = 24, n_workers: int = 1):
    output_dir.mkdir(exist_ok=True, parents=True)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=n_workers, shuffle=False)

    for idx, sample in enumerate(dataloader):
        sample = {k: v.squeeze(0) for k, v in sample.items()}

        scene_dir = output_dir / f"{idx:06d}"
        scene_dir.mkdir(exist_ok=True)

        planes = ['color', 'depth', 'normal', 'optical_flow', 'position']

        for plane in planes:
            if plane in sample:
                plane_tensor = sample[plane]  # (T, V, H, W, C)
                plane_tensor_normalized = normalize_hdr_image_tonemap(plane_tensor)
                num_views = plane_tensor.shape[1]

                for view_idx in range(num_views):
                    plane_video = [
                        prepare_video_frames(plane_tensor_normalized[t, view_idx])
                        for t in range(plane_tensor.shape[0])
                    ]

                    video_path = scene_dir / f"{plane}_view_{view_idx:02d}.mp4"
                    imageio.mimwrite(video_path, plane_video, fps=fps, codec='libx264')

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

        print(f"Saved sample {idx} to {scene_dir}")


class MP4Dataset(Dataset):
    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.scene_dirs = sorted([d for d in self.output_dir.iterdir() if d.is_dir()])

    def __len__(self):
        return len(self.scene_dirs)

    def __getitem__(self, idx):
        scene_dir = self.scene_dirs[idx]
        meta_filename = scene_dir / "meta.safetensors"
        with safe_open(str(meta_filename), framework="pt", device="cpu") as f:
            meta_tensors = {key: f.get_tensor(key) for key in f.keys()}

        planes = ['color', 'depth', 'normal', 'optical_flow', 'position']

        for plane in planes:
            video_files = sorted(scene_dir.glob(f"{plane}_view_*.mp4"))

            if not video_files:
                continue

            plane_tensors = []
            for video_file in video_files:
                reader = imageio.get_reader(video_file, 'ffmpeg')
                frames = [to_tensor(frame) for frame in reader]
                plane_tensors.append(torch.stack(frames, dim=0))
                reader.close()

            meta_tensors[plane] = torch.stack(plane_tensors, dim=1)  # [T, V, C, H, W]

        return meta_tensors
