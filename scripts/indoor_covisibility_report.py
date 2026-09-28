#!/usr/bin/env python3
"""All-camera co-visibility audit from completed rendered depth/calibration runs.

An independent depth-reprojection diagnostic, not a count of the production GPU
annotation masks (which use a different tangent-plane test). Keeps run identities
when aggregating bounded process chunks; never rewrites capture provenance.
"""
import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from indoor_multiview_report import shared_pixels
from indoor_report import distribution, summarize
from paper_indoor_report import primary_occupied


def mask_metrics(masks, valid, source, cameras):
    if masks.shape != valid.shape or masks.dtype != np.uint16 or valid.dtype != np.bool_ or not valid.size or not 1 <= cameras <= 16:
        raise ValueError('expected same-shaped uint16 membership and validity for 1..16 cameras')
    if not 0 <= source < cameras or np.any(masks & (1 << source)):
        raise ValueError('source camera must be excluded')
    if np.any(masks.astype(np.uint32) >> cameras) or np.any(masks[~valid]):
        raise ValueError('unknown camera bit or membership on invalid source')
    membership = np.bincount(masks[valid], minlength=1 << cameras)
    popcounts = np.array([i.bit_count() for i in range(1 << cameras)])
    cardinality = np.bincount(popcounts, weights=membership, minlength=cameras)[:cameras].astype(np.int64)
    pixels = int(valid.size)
    hits = int(valid.sum())
    return dict(pixels=pixels, valid_pixels=hits,
        valid_fraction=hits / pixels,
        any_other_fraction_all_pixels=int(cardinality[1:].sum()) / pixels,
        any_other_fraction_valid_pixels=int(cardinality[1:].sum()) / hits if hits else None,
        all_others_fraction_valid_pixels=int(cardinality[-1]) / hits if hits else None,
        mean_other_cameras_valid_pixels=float(np.dot(cardinality, np.arange(cameras)) / hits) if hits else None,
        cardinality_counts=cardinality.tolist(), membership_counts=membership.tolist())


def connected(pairs, cameras, threshold):
    neighbors = {i: set() for i in range(cameras)}
    for a in range(cameras):
        for b in range(a + 1, cameras):
            if min(pairs[a, b], pairs[b, a]) >= threshold:
                neighbors[a].add(b)
                neighbors[b].add(a)
    visited, pending = set(), [0]
    while pending:
        camera = pending.pop()
        if camera not in visited:
            visited.add(camera)
            pending.extend(neighbors[camera] - visited)
    return len(visited) == cameras


def scene_report(task):
    root, directory, capture, manifest = task
    count = len(manifest['cameras'])
    if not 2 <= count <= 16:
        raise ValueError('audit requires 2..16 capture cameras')
    width, height = capture['image_size']
    groups, hashes = {}, {}
    for index, view in enumerate(capture['views']):
        group = groups.setdefault(view['step_index'], {})
        if view['camera_index'] in group:
            raise ValueError('duplicate camera at a timestep')
        group[view['camera_index']] = (index, view)
    rows, pair_rows, time_rows = [], [], []
    total_cardinality = np.zeros(count, dtype=np.int64)
    total_membership = np.zeros(1 << count, dtype=np.int64)
    for step, group in sorted(groups.items()):
        if set(group) != set(range(count)):
            raise ValueError('incomplete camera set')
        depths = {}
        for camera, (index, view) in group.items():
            if abs(view['time'] - group[0][1]['time']) > 1e-6:
                raise ValueError('unsynchronized camera times')
            name = directory / f'view_{index:02}_depth.rgba32f'
            raw = name.read_bytes()
            hashes[str(name.relative_to(root))] = hashlib.sha256(raw).hexdigest()
            data = np.frombuffer(raw, dtype='<f4')
            if data.size != width * height * 4:
                raise ValueError('incorrect depth buffer size')
            depths[camera] = data.reshape(height, width, 4)[..., 0]
        pair_fractions = {}
        for source, (_, view) in sorted(group.items()):
            masks = np.zeros((height, width), dtype=np.uint16)
            valid = (depths[source] >= view['near']) & (depths[source] <= view['far'])
            for other, (_, target) in sorted(group.items()):
                if other == source:
                    continue
                shared, _, _ = shared_pixels(depths[source], view, depths[other], target)
                masks[shared] |= 1 << other
                fraction = float(shared.mean())
                pair_fractions[source, other] = fraction
                pair_rows.append(dict(seed=manifest['seed'], step=step, time=view['time'], source=source,
                    other=other, fraction_all_pixels=fraction,
                    fraction_valid_pixels=int(shared.sum()) / int(valid.sum()) if valid.any() else None))
            stats = mask_metrics(masks, valid, source, count)
            total_cardinality += stats.pop('cardinality_counts')
            total_membership += stats.pop('membership_counts')
            rows.append(dict(seed=manifest['seed'], step=step, time=view['time'], camera=source, **stats))
        reference = [min(pair_fractions[0, c], pair_fractions[c, 0]) for c in range(1, count)]
        all_pairs = [min(pair_fractions[a, b], pair_fractions[b, a])
                     for a in range(count) for b in range(a + 1, count)]
        policy = manifest['camera_settings'].get('multiview')
        time_rows.append(dict(seed=manifest['seed'], step=step, time=group[0][1]['time'],
            reference_min_overlap=min(reference), reference_mean_overlap=float(np.mean(reference)),
            all_pairs_min_overlap=min(all_pairs), all_pairs_mean_overlap=float(np.mean(all_pairs)),
            reference_pairs_below_requested=sum(v < policy['min_overlap'] for v in reference) if policy else 0,
            connected_at_10_percent=connected(pair_fractions, count, .1),
            connected_at_35_percent=connected(pair_fractions, count, .35)))
    for name in ('manifest.json', 'capture.json'):
        hashes[str((directory / name).relative_to(root))] = hashlib.sha256((directory / name).read_bytes()).hexdigest()
    return rows, pair_rows, time_rows, total_cardinality, total_membership, hashes


def write_csv(path, rows):
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def report(roots, output, expected_scenes=None, workers=4):
    tasks, runs, seeds, contract = [], [], set(), None
    for root in roots:
        metrics, captures, base = summarize(root)
        selection = json.loads((root / 'render_selection.json').read_text())
        current = dict(generator_version=metrics['generator_version'], capture_engine=base['capture_engine'],
            image_size=metrics['image_size'], density=metrics['density'], human_density=metrics['human_density'],
            camera_settings=metrics['camera_settings'], playback_steps=base['playback_steps'],
            settings={k: selection.get(k) for k in ('quality', 'diffuse_gi_enabled', 'gi_rays', 'gi_bounces')})
        if contract is not None and contract != current:
            raise ValueError('capture chunks use different generation/render configurations')
        contract = current
        for directory, capture, manifest in captures:
            if manifest['seed'] in seeds:
                raise ValueError('duplicate scene across runs')
            seeds.add(manifest['seed'])
            if metrics['camera_settings'] != manifest['camera_settings']:
                raise ValueError('capture camera policy mismatch')
            if not np.isclose(manifest['camera_aspect_ratio'], metrics['image_size'][0] / metrics['image_size'][1]):
                raise ValueError('capture camera aspect mismatch')
            tasks.append((root, directory, capture, manifest))
        runs.append(dict(directory=str(root), run_id=base['run_id'], seeds=selection['selected_seeds'],
            sha256={f: hashlib.sha256((root / f).read_bytes()).hexdigest()
                    for f in ('metrics.json', 'render_selection.json', 'run_complete.json')}))
    if not tasks or expected_scenes is not None and len(tasks) != expected_scenes:
        raise ValueError('missing requested rendered scenes')
    if sorted(seeds) != list(range(min(seeds), max(seeds) + 1)):
        raise ValueError('expected consecutive seeds without omissions')
    counts = {len(t[3]['cameras']) for t in tasks}
    if len(counts) != 1:
        raise ValueError('inconsistent camera count')
    cameras = counts.pop()
    output.mkdir(parents=True, exist_ok=True)
    views, pairs, times, hashes = [], [], [], {}
    cardinality = np.zeros(cameras, dtype=np.int64)
    membership = np.zeros(1 << cameras, dtype=np.int64)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for task, (v, p, t, c, m, h) in zip(tasks, pool.map(scene_report, tasks)):
            views.extend(v); pairs.extend(p); times.extend(t)
            cardinality += c; membership += m
            hashes[task[2]['run_id'] + '/' + str(task[3]['seed'])] = h
    scene_worst = [min(t['reference_min_overlap'] for t in times if t['seed'] == seed) for seed in sorted(seeds)]
    pair_lookup = {(p['seed'],p['step'],p['source'],p['other']):p['fraction_all_pixels'] for p in pairs}
    reference = [min(p['fraction_all_pixels'], pair_lookup[p['seed'],p['step'],p['other'],0])
        for p in pairs if p['source'] == 0]
    hit_total = int(cardinality.sum())
    capture_views = [view for _, _, capture, _ in tasks for view in capture['views']]
    occupied = [capture for _, _, capture, manifest in tasks if primary_occupied(manifest)]
    quality = dict(
        layouts=dict(Counter(manifest['layout'] for _, _, _, manifest in tasks)),
        rgb_and_pose={k:distribution([v[k] for v in capture_views]) for k in
                     ('mean_luminance','dark_fraction','clipped_fraction','pose_max_absolute_error')},
        annotations={k:distribution([v['annotation_alignment'][k] for v in capture_views if v['annotation_alignment']])
                     for k in ('depth_position_p99_metres','reprojection_p99_pixels','normal_length_max_error')},
        rooms_with_a_view_having_at_most_two_semantic_classes=sum(any(len(v['semantic_pixel_counts']) <= 2 for v in capture['views']) for _, _, capture, _ in tasks),
        rooms_with_primary_people=len(occupied),
        occupied_rooms_without_person_pixels=sum(not any(v['semantic_pixel_counts'].get('person',0) for v in capture['views']) for capture in occupied),
    )
    result = dict(schema_version=1, method='independent rendered-depth reprojection', **contract,
        scenes=len(seeds), views=len(views), cameras=cameras, synchronized_sets=len(times),
        directed_pairs=len(pairs), reference_pairs=len(reference), seed_range=[min(seeds), max(seeds)],
        runs=runs, rendered_quality=quality, source_pixels=sum(v['pixels'] for v in views), source_valid_pixels=hit_total,
        other_camera_count_histogram_valid=cardinality.tolist(),
        other_camera_count_fractions_valid=(cardinality / hit_total).tolist() if hit_total else None,
        membership_mask_counts_valid=membership.tolist(),
        view_distributions={k: distribution([v[k] for v in views if v[k] is not None]) for k in
            ('valid_fraction','any_other_fraction_all_pixels','any_other_fraction_valid_pixels',
             'all_others_fraction_valid_pixels','mean_other_cameras_valid_pixels')},
        reference_overlap=distribution(reference), scene_worst_reference_overlap=distribution(scene_worst),
        reference_pairs_below_requested=sum(t['reference_pairs_below_requested'] for t in times),
        scenes_with_reference_pair_below_10_percent=sum(v < .1 for v in scene_worst),
        disconnected_sets_at_10_percent=sum(not t['connected_at_10_percent'] for t in times),
        disconnected_sets_at_35_percent=sum(not t['connected_at_35_percent'] for t in times),
        all_pair_min_overlap=distribution([t['all_pairs_min_overlap'] for t in times]),
        worst_sets=sorted(times, key=lambda t:t['reference_min_overlap'])[:16],
        input_sha256=hashes, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                           for name in ('indoor_multiview_report.py','indoor_report.py')},
        definition='All ordered source/target camera pairs at equal time. Visible iff projected depth agrees with the nearest target pixel within 0.01 m + 0.002*z. Bits name target cameras; self excluded. Histograms count valid source pixels (observations, not unique 3D points); pair overlap uses all source image pixels.',
        limitations=['These reconstructed masks are an independent depth diagnostic; production GPU co-visibility uses a different symmetric tangent-plane test. Do not equate their values.',
            'Glass is annotation-opaque; refracted/reflected RGB content is not included.',
            'Samples and cameras within a scene are correlated. Only captured times are measured.',
            'No rendered room is replaced, filtered by appearance or dropped for low overlap.'])
    for name, rows in [('views',views),('pairs',pairs),('times',times)]:
        write_csv(output / f'covisibility_{name}.csv', rows)
    (output / 'covisibility_report.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-scenes', type=int)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    result = report(args.roots, args.output, args.expected_scenes, args.workers)
    print(json.dumps({k:v for k,v in result.items() if k not in ('input_sha256','runs','membership_mask_counts_valid')}, indent=2))
