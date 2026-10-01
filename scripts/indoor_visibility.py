"""Validate exact production co-visibility exports and their display encoding."""
import math
from io import BytesIO

import numpy as np
from PIL import Image


def palette(metadata, count):
    if not isinstance(metadata,dict) or metadata.get('schema_version') != 1 or metadata.get('camera_count') != count:
        raise ValueError('missing or incompatible co-visibility metadata')
    legend = metadata.get('legend', [])
    if len(legend) != count or not 1 <= count <= 16:
        raise ValueError('invalid co-visibility camera count')
    codes = np.zeros((1 << count, 3), dtype=np.uint16)
    membership = np.arange(1 << count, dtype=np.uint32)
    for bit, entry in enumerate(legend):
        channel = bit % 3
        bits = (count - channel + 2) // 3
        expected = [0, 0, 0]
        expected[channel] = (255 // ((1 << bits) - 1)) * (1 << (bits - 1 - bit // 3))
        if entry != {'bit': bit, 'camera_index': bit, 'mask': 1 << bit, 'rgb8': expected}:
            raise ValueError('inconsistent camera ordering or additive legend')
        codes += ((membership >> bit) & 1)[:, None].astype(np.uint16) * np.asarray(expected, dtype=np.uint16)
    if codes.max() > 255:
        raise ValueError('camera colors overflow RGB')
    return codes.astype(np.uint8)


def load(folder, capture, index, count):
    """Return validated uint16 masks, validity and counts, never decoded RGB bits."""
    stem = f'view_{index:02}_co_visibility'
    return decode((folder/f'{stem}_mask.png').read_bytes(),
                  (folder/f'{stem}_valid.png').read_bytes(),capture,index,count,
                  (folder/f'{stem}.png').read_bytes())


def decode(mask_png, valid_png, capture, index, count, preview_png=None):
    """The same numeric checks apply to raw captures and downloadable archives."""
    view = capture['views'][index]
    source = view['camera_index']
    if not 0 <= source < count:
        raise ValueError('invalid source camera')
    codes = palette(capture.get('co_visibility_metadata', {}), count)
    header = mask_png[:26]
    if header[:8] != b'\x89PNG\r\n\x1a\n' or header[24:26] != bytes([16, 0]):
        raise ValueError('membership must be an exact 16-bit grayscale PNG')
    with Image.open(BytesIO(mask_png)) as im:
        masks = np.asarray(im).astype(np.uint16)
    with Image.open(BytesIO(valid_png)) as im:
        valid = np.asarray(im)
    shape = tuple(reversed(capture['image_size']))
    if masks.shape != shape or valid.shape != shape or not np.isin(valid, [0, 1]).all():
        raise ValueError('invalid co-visibility dimensions or validity encoding')
    valid = valid.astype(bool)
    if (masks.astype(np.uint32) >= (1 << count)).any() or (masks & (1 << source)).any() or masks[~valid].any():
        raise ValueError('invalid membership bits, source exclusion or background')
    peer_pixels = [int(np.count_nonzero(masks & (1 << i))) for i in range(count)]
    cardinality = np.zeros(shape, dtype=np.uint8)
    for i in range(count):
        cardinality += ((masks >> i) & 1).astype(np.uint8)
    counts = np.bincount(cardinality[valid], minlength=count).tolist()
    n = int(valid.sum())
    shared = int(np.count_nonzero(masks))
    stats = dict(valid_pixels=n, shared_pixels=shared, peer_pixels=peer_pixels,
                 cardinality_pixels=counts, shared_fraction_valid=shared / max(n, 1),
                 peer_fraction_valid=[v / max(n, 1) for v in peer_pixels])
    recorded = view.get('co_visibility', {})
    for key, value in stats.items():
        actual = recorded.get(key)
        if isinstance(value, float):
            matches = actual is not None and math.isclose(value, actual, abs_tol=1e-12)
        elif key == 'peer_fraction_valid':
            matches = actual is not None and np.allclose(value, actual, atol=1e-12, rtol=0)
        else:
            matches = actual == value
        if not matches:
            raise ValueError(f'co-visibility report does not match masks: {key}')
    if preview_png is not None:
        with Image.open(BytesIO(preview_png)) as im:
            if not np.array_equal(codes[masks], np.asarray(im.convert('RGB'))):
                raise ValueError('co-visibility preview does not match exact masks and legend')
    return masks, valid, stats
