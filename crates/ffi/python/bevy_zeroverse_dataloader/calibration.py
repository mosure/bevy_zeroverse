"""Versioned camera tensors. Never infer calibration for legacy archives."""
import json

import torch
import bevy_zeroverse_ffi

NAMES = ('camera_calibration', 'intrinsics', 'image_size', 'trajectory_progress',
         'time_seconds', 'time_seconds_valid')


def metadata_tensor():
    return torch.tensor(list(bevy_zeroverse_ffi.CAMERA_CALIBRATION_METADATA.encode()), dtype=torch.uint8)


def validate(sample):
    if not any(name in sample for name in NAMES):
        return
    if not all(name in sample for name in NAMES):
        raise ValueError('incomplete calibrated capture')
    raw = sample['camera_calibration']
    if raw.dtype != torch.uint8 or raw.ndim != 1:
        raise ValueError('invalid calibration metadata tensor')
    if json.loads(bytes(raw.tolist())) != json.loads(bevy_zeroverse_ffi.CAMERA_CALIBRATION_METADATA):
        raise ValueError('unsupported camera calibration contract')
    prefix = sample['fovy'].shape[:-1]
    for name, dtype, tail in [('intrinsics', torch.float32, (3, 3)), ('image_size', torch.int64, (2,)),
                              ('trajectory_progress', torch.float32, (1,)), ('time_seconds', torch.float32, (1,)),
                              ('time_seconds_valid', torch.uint8, (1,))]:
        value = sample[name]
        if value.shape != (*prefix, *tail) or value.dtype != dtype or not torch.isfinite(value).all():
            raise ValueError(f'invalid {name} shape, dtype or values')
    k = sample['intrinsics']
    if (k[..., 0, 0] <= 0).any() or (k[..., 1, 1] <= 0).any() or (k[..., 1, 0] != 0).any() or not torch.all(k[..., 2, :] == k.new_tensor([0., 0., 1.])):
        raise ValueError('invalid pinhole K')
    dimensions = sample['image_size']
    if (dimensions <= 0).any():
        raise ValueError('image dimensions must be positive')
    for name in ('color', 'depth', 'normal', 'position', 'semantic', 'co_visibility', 'optical_flow', 'motion_vectors'):
        if name in sample:
            plane = sample[name]
            if tuple(plane.shape[:-3]) != tuple(prefix) or not torch.all(dimensions == dimensions.new_tensor([plane.shape[-2], plane.shape[-3]])):
                raise ValueError(f'{name} dimensions differ from calibration')
    progress, seconds, valid = (sample[name] for name in NAMES[-3:])
    if ((progress < 0) | (progress > 1)).any() or (seconds < 0).any() or (valid > 1).any() or ((valid == 0) & (seconds != 0)).any():
        raise ValueError('invalid capture time or validity')
