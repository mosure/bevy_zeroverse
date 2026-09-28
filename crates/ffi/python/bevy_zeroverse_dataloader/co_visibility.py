"""Exact per-camera membership; no image tone mapping or lossy color codec."""
import json
import numpy as np

MAX_CAMERAS = 16


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def palette(count):
    if not 1 <= count <= MAX_CAMERAS:
        raise ValueError('co-visibility supports 1..16 capture cameras')
    colors = np.zeros((count, 3), dtype=np.uint8)
    for i in range(count):
        channel, rank = i % 3, i // 3
        bits = (count - channel + 2) // 3
        colors[i, channel] = (255 // ((1 << bits) - 1)) * (1 << (bits - 1 - rank))
    return colors


def camera_visible(mask, camera):
    if not 0 <= camera < MAX_CAMERAS:
        raise ValueError('camera bit must be in 0..15')
    return (_array(mask).astype(np.uint16) & (1 << camera)) != 0


def mask_to_rgb(mask, count):
    palette(count)
    masks = _array(mask)
    if masks.dtype != np.uint16 or (masks.astype(np.uint32) >= (1 << count)).any():
        raise ValueError('membership must be uint16 with no out-of-range bits')
    colors = palette(count)
    rgb = np.zeros((*masks.shape, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        rgb += camera_visible(masks, i)[..., None] * color
    return rgb


def rgb_to_mask(rgb, count):
    rgb = _array(rgb)
    palette(count)
    if rgb.dtype != np.uint8 or rgb.shape[-1] != 3:
        raise ValueError('expected unmodified lossless RGB8 codes')
    masks = np.zeros(rgb.shape[:-1], dtype=np.uint16)
    for channel in range(3):
        if channel >= count:
            if rgb[..., channel].any():
                raise ValueError('unused color channel is nonzero')
            continue
        bits = (count - channel + 2) // 3
        scale = 255 // ((1 << bits) - 1)
        value = rgb[..., channel].astype(np.uint16)
        if (value % scale).any() or (value // scale >= (1 << bits)).any():
            raise ValueError('RGB values do not match the camera legend')
        for rank in range(bits):
            masks |= ((value // scale >> (bits - 1 - rank)) & 1) << (channel + 3 * rank)
    return masks


def from_rgba(values):
    values = _array(values)
    if values.shape[-1] != 4 or not np.isfinite(values).all():
        raise ValueError('expected finite co-visibility RGBA data')
    mask, count, valid, reserved = np.moveaxis(values, -1, 0)
    if (mask < 0).any() or (mask > 65535).any() or (mask != np.floor(mask)).any():
        raise ValueError('invalid camera membership')
    masks = mask.astype(np.uint16)
    popcount = sum(((masks >> i) & 1).astype(np.uint8) for i in range(16))
    if (count != popcount).any() or not np.isin(valid, [0, 1]).all() or reserved.any() or ((valid == 0) & (masks != 0)).any():
        raise ValueError('invalid co-visibility count or validity')
    return {'co_visibility': masks[..., None], 'co_visibility_valid': valid[..., None].astype(np.uint8)}


def validate_metadata(metadata, count):
    if not isinstance(metadata, dict):
        metadata = json.loads(bytes(_array(metadata).tolist()))
    if metadata.get('schema_version') != 1 or metadata.get('camera_count') != count:
        raise ValueError('missing or unknown co-visibility convention')
    colors = palette(count)
    legend = metadata.get('legend', [])
    if len(legend) != count:
        raise ValueError('invalid co-visibility legend length')
    indices = set()
    for bit, item in enumerate(legend):
        index = item.get('camera_index')
        if (item.get('bit') != bit or item.get('mask') != 1 << bit or item.get('rgb8') != colors[bit].tolist()
                or not isinstance(index, int) or index < 0 or index in indices):
            raise ValueError('invalid camera ordering or RGB legend')
        indices.add(index)
    return metadata


def validate(sample):
    if 'co_visibility' not in sample:
        if 'co_visibility_valid' in sample:
            raise ValueError('validity without membership')
        return
    mask = _array(sample['co_visibility'])
    valid = _array(sample.get('co_visibility_valid'))
    if mask.ndim not in (5, 6) or mask.shape[-1] != 1 or mask.dtype != np.uint16:
        raise ValueError('co_visibility must be [batch?,time,camera,height,width,1] uint16')
    count = mask.shape[-4]
    if not 1 <= count <= MAX_CAMERAS or (mask.astype(np.uint32) >= (1 << count)).any():
        raise ValueError('out-of-range camera membership')
    if valid.dtype != np.uint8 or valid.shape != mask.shape or (valid > 1).any() or ((valid == 0) & (mask != 0)).any():
        raise ValueError('invalid co-visibility validity')
    for i in range(count):
        if (mask[..., i, :, :, :] & (1 << i)).any():
            raise ValueError('source camera must not appear in its own membership')
    metadata = sample.get('co_visibility_metadata')
    if metadata is None:
        raise ValueError('missing co-visibility metadata')
    rows = metadata if mask.ndim == 6 else [metadata]
    if len(rows) != (mask.shape[0] if mask.ndim == 6 else 1):
        raise ValueError('co-visibility metadata batch size mismatch')
    for row in rows:
        validate_metadata(row, count)
    for name in ('color', 'depth', 'normal', 'semantic', 'position', 'optical_flow'):
        if name in sample and sample[name].shape[:-1] != mask.shape[:-1]:
            raise ValueError('co-visibility image dimensions differ')


def save_folder(sample, directory):
    """Write exact masks plus a lossless additive preview for every view."""
    from PIL import Image
    validate(sample)
    if 'co_visibility' not in sample:
        return
    masks, valid = (_array(sample[name]) for name in ('co_visibility', 'co_visibility_valid'))
    if masks.ndim != 5:
        raise ValueError('folder export expects one sample')
    count = masks.shape[1]
    metadata = validate_metadata(sample['co_visibility_metadata'], count)
    (directory / 'co_visibility_metadata.json').write_text(json.dumps(metadata))
    for t in range(masks.shape[0]):
        for camera in range(count):
            stem = directory / f'co_visibility_{t:03d}_{camera:02d}'
            np.savez_compressed(stem.with_suffix('.npz'), co_visibility=masks[t, camera], co_visibility_valid=valid[t, camera])
            Image.fromarray(mask_to_rgb(masks[t, camera, ..., 0], count)).save(stem.with_suffix('.png'))


def load_folder(directory, steps, cameras):
    path = directory / 'co_visibility_metadata.json'
    planes = list(directory.glob('co_visibility_*.npz'))
    if not path.exists() and not planes:
        return {}
    if not path.exists() or len(planes) != steps * cameras:
        raise ValueError('incomplete co-visibility planes or metadata')
    metadata = validate_metadata(json.loads(path.read_text()), cameras)
    result = {name: [] for name in ('co_visibility', 'co_visibility_valid')}
    for step in range(steps):
        frame = {name: [] for name in result}
        for camera in range(cameras):
            with np.load(directory / f'co_visibility_{step:03d}_{camera:02d}.npz') as archive:
                for name in result:
                    frame[name].append(np.array(archive[name], copy=True))
        for name in result:
            result[name].append(np.stack(frame[name]))
    result = {name: np.stack(value) for name, value in result.items()}
    result['co_visibility_metadata'] = np.frombuffer(json.dumps(metadata).encode(), dtype=np.uint8).copy()
    validate(result)
    return result
