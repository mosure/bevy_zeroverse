"""Versioned camera tensors. Never infer calibration for legacy archives."""
import json

import torch
import bevy_zeroverse_ffi

NAMES = ('camera_calibration', 'intrinsics', 'image_size', 'trajectory_progress',
         'time_seconds', 'time_seconds_valid')


def metadata_tensor(mixed=False):
    text = bevy_zeroverse_ffi.CAMERA_CALIBRATION_METADATA_V2 if mixed else bevy_zeroverse_ffi.CAMERA_CALIBRATION_METADATA
    return torch.tensor(list(text.encode()), dtype=torch.uint8)


def validate(sample):
    if not any(name in sample for name in NAMES):
        return
    if not all(name in sample for name in NAMES):
        raise ValueError('incomplete calibrated capture')
    raw = sample['camera_calibration']
    if raw.dtype != torch.uint8 or raw.ndim != 1:
        raise ValueError('invalid calibration metadata tensor')
    contract = json.loads(bytes(raw.tolist()))
    mixed = contract.get('schema_version') == 2
    expected = bevy_zeroverse_ffi.CAMERA_CALIBRATION_METADATA_V2 if mixed else bevy_zeroverse_ffi.CAMERA_CALIBRATION_METADATA
    if contract != json.loads(expected):
        raise ValueError('unsupported camera calibration contract')
    prefix = sample['fovy'].shape[:-1]
    for name, dtype, tail in [('intrinsics', torch.float32, (3, 3)), ('image_size', torch.int64, (2,)),
                              ('trajectory_progress', torch.float32, (1,)), ('time_seconds', torch.float32, (1,)),
                              ('time_seconds_valid', torch.uint8, (1,))]:
        value = sample[name]
        if value.shape != (*prefix, *tail) or value.dtype != dtype or not torch.isfinite(value).all():
            raise ValueError(f'invalid {name} shape, dtype or values')
    if not mixed and ('lens_model' in sample or 'lens_parameters' in sample):
        raise ValueError('angular lens tensors require version 2 metadata')
    if mixed and not all(name in sample for name in ('lens_model', 'lens_parameters')):
        raise ValueError('incomplete angular lens tensors')
    if mixed:
        ids, params = sample['lens_model'], sample['lens_parameters']
        if ids.dtype != torch.uint8 or ids.shape != (*prefix, 1) or params.dtype != torch.float32 or params.shape != (*prefix, 5) or not torch.isfinite(params).all() or (ids > 1).any():
            raise ValueError('invalid per-view lens tensors')
        if (params[ids[..., 0] == 0] != 0).any():
            raise ValueError('pinhole angular parameters must be zero')
        angular = params[ids[..., 0] == 1]
        if ((angular[..., 4] < .15) | (angular[..., 4] > 2.)).any():
            raise ValueError('unsupported angular domain')
        # Certify the entire derivative interval, matching Rust's conservative
        # contract. Sampling a few radii would miss a narrow polynomial fold.
        def certify(k, lo, hi, remaining):
            lower = 1. + sum(v * (lo if v >= 0 else hi)**(i+1) for i, v in enumerate(k))
            if lower > .02:
                return True
            if remaining == 0:
                return False
            mid = (lo+hi)*.5
            return certify(k,lo,mid,remaining-1) and certify(k,mid,hi,remaining-1)
        for p in angular.detach().cpu().reshape(-1,5).tolist():
            magnitude = 1. + sum(abs(k) * p[4]**(2*i+2) for i,k in enumerate(p[:4]))
            if magnitude * max(p[4],9.) > torch.finfo(torch.float32).max * .25:
                raise ValueError('angular coefficients exceed portable f32 arithmetic')
            if not certify([v*(2*i+3) for i,v in enumerate(p[:4])],0.,p[4]**2,12):
                raise ValueError('folded or uncertified angular lens')
    k = sample['intrinsics']
    if (k[..., 0, 0] <= 0).any() or (k[..., 1, 1] <= 0).any() or (k[..., 1, 0] != 0).any() or not torch.all(k[..., 2, :] == k.new_tensor([0., 0., 1.])):
        raise ValueError('invalid calibration K')
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
