#!/usr/bin/env python3
"""Measure same-time shared visible surfaces using actual calibrated depth captures.

Run indoor_validate with --labels (retain raw buffers), then pass its output dir.
No inference, learned features or RGB similarity are used. Cameras are paired to
camera zero at every captured time, exactly like the placement policy.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from indoor_report import distribution, summarize


def shared_pixels(source_depth, source, target_depth, target):
    """Return per-source-pixel visibility and parallax, using camera-space Z depth.

    Pixel centres are half-integers, extrinsics are column-major world_from_view,
    Bevy cameras face -Z, and source/target intrinsics may differ.
    """
    if source_depth.shape != target_depth.shape or source_depth.ndim != 2:
        raise ValueError('paired captures must have the same 2D image shape')
    h, w = source_depth.shape
    if not np.isfinite(source_depth).all() or not np.isfinite(target_depth).all():
        raise ValueError('nonfinite depth')
    for view in (source, target):
        if not all(np.isfinite(view[k]) and view[k] > 0 for k in ('fx_pixels', 'fy_pixels')):
            raise ValueError('invalid focal length')
    sw = np.asarray(source['world_from_view'], dtype=np.float64).T
    tw = np.asarray(target['world_from_view'], dtype=np.float64).T
    for pose in (sw, tw):
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5)
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)):
            raise ValueError('invalid rigid world_from_view matrix')
    y, x = np.indices((h, w), dtype=np.float64)
    valid = (source_depth >= source.get('near', .1)) & (source_depth <= source.get('far', 50))
    local = np.stack(((x+.5-w/2)/source['fx_pixels']*source_depth,
                      -(y+.5-h/2)/source['fy_pixels']*source_depth, -source_depth), axis=-1)
    world = local @ sw[:3, :3].T + sw[:3, 3]
    other = (world - tw[:3, 3]) @ tw[:3, :3]
    depth = -other[..., 2]
    safe = np.where(depth > 0, depth, 1)
    # Coordinates in pixel-edge convention; floor selects nearest pixel centre.
    u = target['fx_pixels'] * other[..., 0]/safe + w/2
    v = -target['fy_pixels'] * other[..., 1]/safe + h/2
    inside = (valid & (depth >= target.get('near', .1)) & (depth <= target.get('far', 50))
              & (u >= 0) & (u < w) & (v >= 0) & (v < h))
    ix = np.clip(np.floor(u), 0, w-1).astype(np.int64)
    iy = np.clip(np.floor(v), 0, h-1).astype(np.int64)
    observed = target_depth[iy, ix]
    # Allows finite pixel footprint / slope error; never counts occluded points
    # solely because they project inside the target frustum.
    visible = inside & (observed > 0) & (np.abs(observed-depth) <= .01 + .002*depth)
    ray_a = world - sw[:3, 3]
    ray_b = world - tw[:3, 3]
    denom = np.linalg.norm(ray_a, axis=-1) * np.linalg.norm(ray_b, axis=-1)
    cosine = np.sum(ray_a*ray_b, axis=-1) / np.maximum(denom, 1e-12)
    angles = np.degrees(np.arccos(np.clip(cosine, -1, 1)))
    return visible, angles, valid


def report(root, previews=False):
    metrics, captures, base = summarize(root)  # checks run IDs, completion, seed selection
    rows = []
    hashes = {name: hashlib.sha256((root/name).read_bytes()).hexdigest()
              for name in ('metrics.json', 'render_selection.json', 'run_complete.json')}
    for directory, capture, manifest in captures:
        w, h = capture['image_size']
        if metrics.get('camera_settings') != manifest.get('camera_settings'):
            raise ValueError('capture and audit camera policies differ')
        if abs(manifest.get('camera_aspect_ratio', w/h) - w/h) > 1e-5:
            raise ValueError('capture and sampler aspect ratios differ')
        groups = {}
        for index, view in enumerate(capture['views']):
            group = groups.setdefault(view['step_index'], {})
            if view['camera_index'] in group:
                raise ValueError('duplicate camera at timestep')
            group[view['camera_index']] = (index, view)
        for step, group in sorted(groups.items()):
            if set(group) != set(range(len(manifest['cameras']))):
                raise ValueError('incomplete synchronized camera set')
            depths = {}
            for camera, (index, view) in group.items():
                if abs(view['time'] - group[0][1]['time']) > 1e-6:
                    raise ValueError('different times in same multi-view group')
                path = directory / f'view_{index:02}_depth.rgba32f'
                raw = path.read_bytes()
                hashes[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
                data = np.frombuffer(raw, dtype='<f4')
                if data.size != w*h*4:
                    raise ValueError(f'incorrect depth buffer size: {path}')
                depths[camera] = data.reshape(h, w, 4)[..., 0]
            reference = group[0][1]
            for camera, (index, view) in sorted(group.items()):
                if camera == 0:
                    continue
                ab, aa, av = shared_pixels(depths[0], reference, depths[camera], view)
                ba, bb, bv = shared_pixels(depths[camera], view, depths[0], reference)
                parallax = np.concatenate((aa[ab], bb[ba]))
                pose_a = np.asarray(reference['world_from_view']).T
                pose_b = np.asarray(view['world_from_view']).T
                policy = manifest.get('camera_settings', {}).get('multiview')
                shared_min = float(min(ab.mean(), ba.mean()))
                row = dict(seed=manifest['seed'], step=step, time=view['time'], reference=0, camera=camera,
                           reference_to_view=float(ab.mean()), view_to_reference=float(ba.mean()),
                           bidirectional_min=shared_min, reference_geometry_fraction=float(av.mean()),
                           view_geometry_fraction=float(bv.mean()),
                           baseline_m=float(np.linalg.norm(pose_a[:3, 3]-pose_b[:3, 3])),
                           triangulation_mean_degrees=float(parallax.mean()) if len(parallax) else None,
                           triangulation_p10_degrees=float(np.percentile(parallax, 10)) if len(parallax) else None,
                           requested_overlap=policy['min_overlap'] if policy else None,
                           below_requested_overlap=shared_min < policy['min_overlap'] if policy else None)
                rows.append(row)
                if previews:
                    from PIL import Image
                    for source_index, mask in [(group[0][0], ab), (index, ba)]:
                        rgb = np.asarray(Image.open(directory / f'view_{source_index:02}_color.png').convert('RGB')).copy()
                        rgb[~mask] = (rgb[~mask].astype(float)*.2).astype(np.uint8)
                        Image.fromarray(rgb).save(directory / f'overlap_{step:02}_0_{camera}_{source_index}.png')
        for name in ['capture.json', 'manifest.json']:
            path = directory/name
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if not rows:
        raise ValueError('at least two cameras are required')
    keys = ['bidirectional_min', 'reference_to_view', 'view_to_reference', 'baseline_m',
            'triangulation_mean_degrees', 'triangulation_p10_degrees']
    scene_min = [min(r['bidirectional_min'] for r in rows if r['seed'] == seed)
                 for seed in sorted({r['seed'] for r in rows})]
    result = dict(schema_version=1, capture_engine=base['capture_engine'], run_id=base['run_id'],
                  selection_policy=base['selection_policy'], scenes=len(captures), pairs=len(rows),
                  image_size=metrics['image_size'], camera_settings=metrics.get('camera_settings'),
                  distributions={k: distribution([r[k] for r in rows if r[k] is not None]) for k in keys},
                  scene_worst_overlap=distribution(scene_min),
                  pairs_below_requested_overlap=sum(r['below_requested_overlap'] is True for r in rows),
                  scenes_with_under_10_percent_overlap=sum(v < .1 for v in scene_min),
                  input_sha256=hashes,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  definition='Minimum of A-to-B and B-to-A visible surface pixels / all image pixels, paired with camera zero at equal time. Depth match uses nearest pixel and 0.01m + 0.002*z tolerance.',
                  limitations=['Placement uses approximate boxes, this audit uses rendered first-surface depth; neither measures image matchability or downstream training utility.',
                               'Glass is opaque for annotations; transmitted RGB content is not measured.',
                               'Only captured timesteps are measured; moving people can change overlap.',
                               'Pairs/times within a room are correlated; scene minima are also reported.'])
    with (root/'rendered_camera_overlap.csv').open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root/'multiview_report.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--previews', action='store_true')
    parser.add_argument('--require-min-overlap', type=float, help='Fail after writing reports if any rendered pair is below this fraction')
    args = parser.parse_args()
    if args.require_min_overlap is not None and not 0 <= args.require_min_overlap <= 1:
        parser.error('--require-min-overlap must be in [0,1]')
    result = report(args.root, args.previews)
    print(json.dumps({k: result[k] for k in ['scenes', 'pairs', 'scene_worst_overlap', 'pairs_below_requested_overlap']}, indent=2))

    if (args.require_min_overlap is not None
            and result['scene_worst_overlap']['min'] < args.require_min_overlap):
        raise SystemExit('Rendered overlap requirement failed; see rendered_camera_overlap.csv')
