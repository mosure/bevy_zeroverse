#!/usr/bin/env python3
"""Capture-derived coverage, scene-level uncertainty, and visual repetition audit.

Uses indoor_report's complete-run checks. Consecutive and stress cohorts stay
separate; all selected scenes and all captured views contribute, including tails.
"""
import argparse
from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path

from indoor_report import distribution, summarize


def wilson(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    p = successes / total
    center = (p + z*z/(2*total)) / (1 + z*z/total)
    half = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / (1 + z*z/total)
    return [max(0.0, center-half), min(1.0, center+half)]


def view_flags(view, pixels):
    counts = view['semantic_pixel_counts']
    return {
        'mostly_dark': view['dark_fraction'] > 0.95,
        'substantial_clipping': view['clipped_fraction'] > 0.10,
        'single_class_dominance': max(counts.values(), default=0) / pixels > 0.90,
        'few_semantic_classes': len(counts) <= 2,
    }


def write_csv(path, rows):
    keys = sorted(set().union(*(row.keys() for row in rows)))
    with path.open('w') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def capture_rows(metrics, reports):
    scenes, views, images = [], [], []
    kinds = sorted(key.removeprefix('main/') for key in metrics['object_counts_per_scene'] if key.startswith('main/'))
    for directory, report, manifest in reports:
        size = manifest['room_size']
        program = manifest['program']
        domain = program['domain']
        counts = Counter(o['kind'] for o in manifest['objects'] if not o['neighbor'])
        counts['Person'] = sum(not h['neighbor'] for h in manifest['humans'])
        scene = {
            'seed': manifest['seed'], 'layout': manifest['layout'],
            'architecture': manifest['architecture_style'],
            'room_area_m2': size[0]*size[2], 'room_height_m': size[1],
            'zones': len(program['zones']), 'clutter_prior': domain['clutter'],
            'sun_lux': manifest['daylight_lux'], 'electric_target_lux': manifest['target_lux'],
            'ev100': domain['photometry']['ev100'],
            'wood_roughness': next(m['roughness'] for m in program['materials'] if m['surface'] == 'Wood'),
            'wood_repeat_m': next(m['period_m'] for m in program['materials'] if m['surface'] == 'Wood'),
            'elapsed_seconds': report['elapsed_seconds'],
            **{f'count_{kind}': counts[kind] for kind in kinds},
        }
        flags = []
        people_visible = False
        for i, v in enumerate(report['views']):
            pixels = math.prod(report['image_size'])
            labels = v['semantic_pixel_counts']
            if sum(labels.values()) != pixels:
                raise ValueError('capture distributions require complete semantic pixel counts; capture with --labels')
            f = view_flags(v, pixels)
            flags.append(f)
            people_visible |= labels.get('person', 0) > 0
            camera = manifest['cameras'][v['camera_index']]
            distance = math.dist(camera['start'], camera['end'])
            row = {
                'seed': manifest['seed'], 'view': i, 'camera': v['camera_index'], 'time': v['time'],
                'vertical_fov_degrees': math.degrees(v['fov_y']), 'fx_pixels': v['fx_pixels'],
                'fy_pixels': v['fy_pixels'], 'cx_pixels': report['image_size'][0]/2,
                'cy_pixels': report['image_size'][1]/2,
                **{f'position_{axis}_m': v['world_from_view'][3][j] for j,axis in enumerate('xyz')},
                **{f'forward_{axis}': -v['world_from_view'][2][j] for j,axis in enumerate('xyz')},
                'camera_height_m': v['world_from_view'][3][1], 'endpoint_baseline_m': distance,
                'camera_start_boundary_clearance_m': min(size[0]/2-abs(camera['start'][0]), size[2]/2-abs(camera['start'][2])),
                **{k: v[k] for k in ['mean_luminance', 'luminance_std', 'dark_fraction', 'clipped_fraction']},
                **{f'visible_{k}_fraction': labels.get(k, 0)/pixels for k in ['person', 'chair', 'table', 'desk', 'window', 'floor', 'wall']},
                'semantic_classes': len(labels), **f,
            }
            views.append(row)
            images.append((directory/f'view_{i:02}_color.png', row))
        for key in flags[0]:
            scene[f'any_view_{key}'] = any(f[key] for f in flags)
        scene['all_views_flagged'] = all(any(f.values()) for f in flags)
        scene['main_people_present'] = counts['Person'] > 0
        scene['person_visible'] = people_visible
        scenes.append(scene)
    return scenes, views, images


def placement(reports, grid=20):
    maps = {key: [0]*(grid*grid) for key in ['furniture', 'chairs', 'plants', 'camera_start', 'camera_path']}
    def add(key, p, size):
        x = min(grid-1, max(0, int((p[0]/size[0]+.5)*grid)))
        z = min(grid-1, max(0, int((p[2]/size[2]+.5)*grid)))
        maps[key][z*grid+x] += 1
    for _, _, manifest in reports:
        size = manifest['room_size']
        for obj in manifest['objects']:
            if obj['neighbor']:
                continue
            if obj['solid']:
                add('furniture', obj['position'], size)
            for kind, key in [('Chair', 'chairs'), ('Plant', 'plants')]:
                if obj['kind'] == kind:
                    add(key, obj['position'], size)
        for camera in manifest['cameras']:
            start, end = camera['start'], camera['end']
            add('camera_start', start, size)
            for i in range(33):
                t = i/32
                if camera.get('motion'):
                    controls = [start, *camera['motion']['control'], end]
                    weights = [(1-t)**3, 3*(1-t)**2*t, 3*(1-t)*t*t, t**3]
                    point = [sum(w*p[a] for w,p in zip(weights, controls)) for a in range(3)]
                else:
                    point = [(1-t)*a+t*b for a,b in zip(start, end)]
                add('camera_path', point, size)
    return {'grid': grid, 'policy': 'Captured manifests only; normalized main-room x/z. Object centers, not visible pixels. Camera paths use 33 cubic-Bezier samples per camera; row index increases with +Z.', 'counts': maps}


def object_rotations(reports):
    """Object-weighted circular coverage; distance from the nearest cardinal axis."""
    rows, result = [], {}
    for _, _, manifest in reports:
        for obj in manifest['objects']:
            if obj['neighbor'] or obj['kind'] not in ['Chair', 'Table', 'Desk', 'CoffeeTable']:
                continue
            yaw = math.degrees(obj['yaw']) % 360
            rows.append(dict(seed=manifest['seed'], kind=obj['kind'], yaw_degrees=yaw,
                             off_cardinal_degrees=abs((yaw+45) % 90-45)))
    for kind in sorted({r['kind'] for r in rows}):
        subset = [r for r in rows if r['kind'] == kind]
        angles = [r['off_cardinal_degrees'] for r in subset]
        histogram = [0]*24
        for row in subset:
            histogram[min(23, int(row['yaw_degrees']/15))] += 1
        result[kind] = dict(instances=len(subset), yaw_bins_15_degrees=histogram,
                            off_cardinal_degrees=distribution(angles),
                            fraction_more_than_15_degrees_off_cardinal=sum(a > 15 for a in angles)/len(angles))
    return rows, result


def repetition(images):
    """One start view per camera; comparisons exclude the same scene."""
    import numpy as np
    from PIL import Image
    starts = [(p, row) for p, row in images if row['time'] == 0]
    hashes, exact = [], {}
    for path, row in starts:
        with Image.open(path) as img:
            exact.setdefault(hashlib.sha256(img.convert('RGB').tobytes()).hexdigest(), set()).add(row['seed'])
            pixels = np.asarray(img.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
            bits = (pixels[:, 1:] > pixels[:, :-1]).ravel()
            hashes.append(sum(int(bit) << i for i, bit in enumerate(bits)))
    nearest, pairs = [], []
    for i, (_, a) in enumerate(starts):
        candidates = [(int(hashes[i] ^ hashes[j]).bit_count(), j) for j, (_, b) in enumerate(starts) if a['seed'] != b['seed']]
        if not candidates:
            continue
        distance, j = min(candidates)
        nearest.append(distance)
        pairs.append((distance, min(i, j), max(i, j)))
    closest = sorted(set(pairs))[:12]
    return {
        'policy': '64-bit luminance difference hash; start frames only; nearest view from another scene. Diagnostic, not semantic uniqueness.',
        'compared_views': len(starts), 'nearest_hamming_bits': distribution(nearest),
        'exact_duplicate_image_groups_across_scenes': sum(len(seeds) > 1 for seeds in exact.values()),
        'nearest_distance_at_most_4': sum(d <= 4 for d in nearest),
        'closest_pairs': [{'distance_bits': d, 'left': str(starts[i][0]), 'right': str(starts[j][0])} for d, i, j in closest],
    }


def audit(root, make_figures=True):
    metrics, reports, base = summarize(root)
    scenes, views, images = capture_rows(metrics, reports)
    rotation_rows, rotations = object_rotations(reports)
    inputs = [root/'metrics.json', root/'render_selection.json', root/'run_complete.json']
    for directory, _, _ in reports:
        inputs.extend([directory/'capture.json', directory/'manifest.json'])
    # Bind derived reports to the exact measurements used, not just a seed range.
    digest = hashlib.sha256()
    for path in inputs:
        digest.update(str(path.relative_to(root)).encode() + b'\0')
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    n = len(scenes)
    representative = base['selection_policy'] == 'consecutive seeds'
    outcomes = {}
    for key in ['any_view_mostly_dark', 'any_view_substantial_clipping', 'any_view_single_class_dominance', 'any_view_few_semantic_classes', 'all_views_flagged']:
        count = sum(s[key] for s in scenes)
        outcomes[key] = {'scenes': count, 'fraction': count/n,
                         'wilson_95_interval': wilson(count, n) if representative else None}
    report = {
        'schema_version': 1, 'run_id': base['run_id'], 'generator_version': metrics['generator_version'],
        'capture_engine': base['capture_engine'], 'selection_policy': base['selection_policy'],
        'input_metrics_sha256': digest.hexdigest(),
        'report_script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'population_estimate': representative, 'scenes': n, 'views': len(views),
        'selected_scenes_completed': n, 'capture_failures': 0,
        'zero_failure_one_sided_95_upper': 1-math.pow(0.05, 1/n) if representative else None,
        'uncertainty_policy': 'Scene is the statistical unit; views/time samples are correlated. Intervals assume consecutive seeds act as independent generator draws. No population intervals for stratified cohorts.',
        'capture_scene_distributions': {k: distribution([s[k] for s in scenes]) for k in scenes[0] if k not in ['seed', 'layout', 'architecture'] and not isinstance(scenes[0][k], bool)},
        'capture_view_distributions': {k: distribution([v[k] for v in views]) for k in views[0] if k not in ['seed', 'view', 'camera', 'time'] and not isinstance(views[0][k], bool)},
        'scene_layout_counts': dict(Counter(s['layout'] for s in scenes)),
        'scene_quality_flags': outcomes,
        'main_people_scenes': sum(s['main_people_present'] for s in scenes),
        'main_people_scenes_with_no_person_pixels': sum(s['main_people_present'] and not s['person_visible'] for s in scenes),
        'repetition': repetition(images), 'annotation': base['annotation'],
        'placement_heatmaps': placement(reports),
        'object_rotations': rotations,
        'threshold_policy': 'Review flags, not rejection: >95% pixels below linear luminance .002; >10% above .99; >90% one semantic class; at most two semantic classes. Intentional low light or close views can be flagged.',
        'limits': ['Small-sample evidence cannot certify 10M samples or photographic realism.',
                   'Instance counts describe captured manifests; semantic fractions measure visibility. Glass is opaque for annotations.',
                   'Plant parameters/placement and finish maps remain procedural approximations.'],
    }
    write_csv(root/'captured_scenes.csv', scenes)
    write_csv(root/'captured_views.csv', views)
    if rotation_rows:
        write_csv(root/'captured_object_rotations.csv', rotation_rows)
    (root/'capture_distribution.json').write_text(json.dumps(report, indent=2)+'\n')
    if make_figures:
        figures(root, scenes, views, images, report)
    return report


def figures(root, scenes, views, images, report):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw
    plt.rcParams.update({'svg.fonttype': 'none', 'font.size': 9})
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    for ax, data, key, label in [
        (axes[0,0], scenes, 'count_Chair', 'Chairs per captured scene (including zero)'),
        (axes[0,1], scenes, 'count_Plant', 'Plants per captured scene'),
        (axes[0,2], scenes, 'count_Person', 'People per captured main room'),
        (axes[1,0], views, 'vertical_fov_degrees', 'Captured vertical FOV (degrees)'),
        (axes[1,1], views, 'camera_height_m', 'Captured camera height (m)'),
        (axes[1,2], views, 'mean_luminance', 'Mean rendered linear luminance'),
    ]:
        ax.hist([r[key] for r in data], bins=16, color='#387c85')
        ax.set(xlabel=label, ylabel='Count')
    points = axes[2,0].scatter([s['room_area_m2'] for s in scenes], [s['count_Chair'] for s in scenes], c=[s['zones'] for s in scenes], cmap='viridis', s=22)
    axes[2,0].set(xlabel='Room area (m²)', ylabel='Chairs')
    fig.colorbar(points, ax=axes[2,0], label='Zones')
    axes[2,1].scatter([s['sun_lux'] for s in scenes], [s['electric_target_lux'] for s in scenes], c=['#b54837' if s['any_view_mostly_dark'] else '#387c85' for s in scenes], s=22)
    axes[2,1].set(xscale='log', yscale='log', xlabel='Solar illuminance (lux)', ylabel='Electric design target (lux)', title='Red: scene includes a mostly-dark view')
    axes[2,2].hist([p['distance_bits'] for p in report['repetition']['closest_pairs']], bins=12, color='#89654e')
    axes[2,2].set(xlabel='Difference-hash distance (bits)', ylabel='Pairs', title='12 closest cross-scene view pairs')
    fig.suptitle(f"Generator {report['generator_version']} • {len(scenes)} captured scenes / {len(views)} views\n{report['selection_policy']}")
    fig.savefig(root/'capture_dashboard.svg')
    fig.savefig(root/'capture_dashboard.png', dpi=140)
    plt.close(fig)
    import numpy as np
    maps = report['placement_heatmaps']
    fig, axes = plt.subplots(1, 5, figsize=(16, 3.6), constrained_layout=True)
    for ax, (key, counts) in zip(axes, maps['counts'].items()):
        values = np.array(counts).reshape(maps['grid'], maps['grid'])
        values = values / max(1, values.sum()) * 100
        plot = ax.imshow(values, origin='lower', extent=[0,1,0,1], cmap='magma')
        ax.set(title=key.replace('_', ' '), xlabel='x / width + 0.5', ylabel='z / depth + 0.5')
        fig.colorbar(plot, ax=ax, label='% of centers / path samples', shrink=.75)
    fig.suptitle(f'{len(scenes)} captured rooms • placement, not visibility')
    fig.savefig(root/'capture_placement.svg')
    fig.savefig(root/'capture_placement.png', dpi=140)
    plt.close(fig)
    # Bounded pages preserve every view without an unreadably tall contact sheet.
    def sheet(rows, path, columns=6):
        tw, th = 240, 212
        output = Image.new('RGB', (columns*tw, math.ceil(len(rows)/columns)*th), '#f4f6f8')
        draw = ImageDraw.Draw(output)
        for i, (path_in, label) in enumerate(rows):
            with Image.open(path_in) as source:
                source = source.convert('RGB'); source.thumbnail((tw, 180))
                x, y = i%columns*tw, i//columns*th
                output.paste(source, (x, y)); draw.text((x+4, y+182), label, fill='#182433')
        output.save(path, quality=90)
    for page in range(0, len(images), 48):
        rows = [(p, f"seed {v['seed']} camera {v['camera']} t={v['time']:.2f}\nL={v['mean_luminance']:.4f} dark={v['dark_fraction']:.1%}") for p,v in images[page:page+48]]
        sheet(rows, root/f'contact_{page//48:02}.jpg')
    worst = sorted(images, key=lambda row: row[1]['mean_luminance'])[:24]
    sheet([(p, f"seed {v['seed']} view {v['view']} L={v['mean_luminance']:.5f}") for p,v in worst], root/'darkest_views.jpg')
    pairs = []
    for p in report['repetition']['closest_pairs']:
        pairs.extend((Path(p[k]), f"{Path(p[k]).parent.name} • {p['distance_bits']} bits") for k in ['left', 'right'])
    if pairs:
        sheet(pairs, root/'closest_pairs.jpg', columns=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--no-figures', action='store_true')
    args = parser.parse_args()
    print(json.dumps(audit(args.root, not args.no_figures), indent=2))
