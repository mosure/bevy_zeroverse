#!/usr/bin/env python3
"""Generate paper tables from completed capture/embedding/benchmark evidence.

No seeds or views are filtered for appearance. Effective counts/ranks are
sample-dependent diagnostics, not a finite procedural-space cardinality.
"""
import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from indoor_bench_report import summarize as benchmark_summary
from indoor_embedding_report import load_embeddings, cohort_metrics
from indoor_report import summarize as capture_summary, distribution


def entropy_count(counts):
    a = np.asarray(list(counts), dtype=float)
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError('invalid counts')
    a = a[a > 0]
    if not len(a):
        return 0.0
    p = a / a.sum()
    return float(np.exp(-(p * np.log(p)).sum()))


def matched_speedup(before, after, warmup):
    a, b = before[warmup:], after[warmup:]
    if len(a) != len(b) or [r['seed'] for r in a] != [r['seed'] for r in b]:
        raise ValueError('speed comparison must use identical ordered seeds')
    if any(x['views'] != y['views'] for x, y in zip(a, b)):
        raise ValueError('speed comparison changed completed view count')
    return 100 * (1 - sum(x['elapsed_seconds'] for x in b) / sum(x['elapsed_seconds'] for x in a))


def primary_occupied(manifest):
    zones = (manifest.get('program') or {}).get('zones', [])
    if zones:
        # Rust max_by selects the last equal-area zone; calculations are f32.
        _, zone = max(enumerate(zones), key=lambda iz: (np.prod(np.asarray(iz[1]['max'], dtype=np.float32)
                                      - np.asarray(iz[1]['min'], dtype=np.float32)), iz[0]))
        lo, hi = np.array(zone['min']), np.array(zone['max'])
    else:
        hi = np.array(manifest['room_size'])[[0, 2]] * .5
        lo = -hi
    return any(not h.get('neighbor', False) and np.all(np.array(h['position'])[[0, 2]] >= lo)
               and np.all(np.array(h['position'])[[0, 2]] <= hi) for h in manifest['humans'])


def discrete_distribution(counts):
    values = np.repeat([int(k) for k in counts], list(counts.values()))
    if not len(values) or (values < 0).any():
        raise ValueError('invalid scene counts')
    counts, edges = np.histogram(values, bins=np.arange(values.max()+2)-.5)
    return dict(min=float(values.min()), max=float(values.max()), mean=float(values.mean()),
                bin_counts=counts.tolist(), bin_edges=edges.tolist())


def table(path, rows, caption, label):
    lines = [r'\begin{table}[ht]', r'\centering\small',
             r'\begin{tabular}{p{.53\linewidth}p{.36\linewidth}}', r'\toprule']
    lines.extend(f'{key} & {value} \\\\' for key, value in rows)
    lines += [r'\bottomrule', r'\end{tabular}', f'\\caption{{{caption}}}\\label{{{label}}}', r'\end{table}']
    path.write_text('\n'.join(lines) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--captures', type=Path, required=True)
    parser.add_argument('--embeddings', type=Path, required=True)
    parser.add_argument('--benchmark', action='append', default=[], metavar='LABEL=DIR')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--tex', type=Path, required=True)
    args = parser.parse_args()
    out, tex = args.output, args.tex
    out.mkdir(parents=True, exist_ok=True)
    tex.mkdir(parents=True, exist_ok=True)
    metrics, captures, summary = capture_summary(args.captures)
    audit = json.loads((args.captures / 'distribution.json').read_text())
    overlap = json.loads((args.captures / 'multiview_report.json').read_text())
    metadata, embeddings = load_embeddings(args.embeddings)
    actual_images = {(m['seed'], v['camera_index'], v['time']): (p / f'view_{i:02}_color.png')
                     for p, c, m in captures for i, v in enumerate(c['views'])}
    if len(metadata['samples']) != len(actual_images):
        raise ValueError('embedding cohort does not contain every rendered view')
    for row in metadata['samples']:
        path = actual_images[(row['seed'], row['camera'], row['time'])]
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('embedding image identity mismatch')
    embedding = cohort_metrics(np.asarray(embeddings), metadata['samples'])
    occupied = [(c, m) for _, c, m in captures if any(not h.get('neighbor', False) for h in m['humans'])]
    occupied_primary = [(c, m) for _, c, m in captures if primary_occupied(m)]
    semantic = dict(
        occupied_primary_rooms=len(occupied_primary),
        primary_rooms_without_person_label=sum(not any(v['semantic_pixel_counts'].get('person', 0) >= 32 for v in c['views']) for c, _ in occupied_primary),
        class_pixels_threshold=32,
        classes_per_view=distribution([sum(v >= 32 for v in view['semantic_pixel_counts'].values())
                                      for _, c, _ in captures for view in c['views']]),
        occupied_non_neighbor_rooms=len(occupied),
        occupied_rooms_without_person_label=sum(not any(v['semantic_pixel_counts'].get('person', 0) >= 32
                                                        for v in c['views']) for c, _ in occupied))
    entropy = {key: dict(total=sum(counts.values()), observed=len(counts),
                         effective_count=entropy_count(counts.values()))
               for key, counts in metrics['categories'].items()}
    varying = sum(d['standard_deviation'] > 1e-10 for d in metrics['numeric'].values())
    benches, rows = {}, {}
    for spec in args.benchmark:
        label, directory = spec.split('=', 1)
        if label in benches:
            raise ValueError('duplicate benchmark label')
        benches[label], _ = benchmark_summary(directory)
        rows[label] = [json.loads(x) for x in (Path(directory) / 'scenes.jsonl').read_text().splitlines()]
        target = out / 'benchmarks' / label
        target.mkdir(parents=True, exist_ok=True)
        for name in ['summary.json', 'telemetry_summary.json', 'baseline.jsonl']:
            shutil.copyfile(Path(directory) / name, target / name)
        for name in ['scenes.jsonl', 'telemetry.jsonl', 'process.log']:
            with (Path(directory) / name).open('rb') as src, gzip.open(target / (name + '.gz'), 'wb') as dst:
                shutil.copyfileobj(src, dst)
    speedup = matched_speedup(rows['baseline'], rows['parallel'], benches['baseline']['summary']['warmup_scenes'])
    if benches['baseline']['summary']['config'] != benches['parallel']['summary']['config']:
        raise ValueError('performance comparison changed configuration')
    data = dict(schema_version=1, generator_version=metrics['generator_version'],
                capture_engine=summary['capture_engine'], audit=audit, rendered=summary,
                categorical_entropy=entropy, numeric_series=len(metrics['numeric']), varying_numeric_series=varying,
                embedding=embedding, overlap=overlap, semantics=semantic, benchmarks=benches,
                capture_time_reduction_percent=speedup,
                interpretation='Bounded diagnostics, no downstream training or ten-million-scene qualification.')
    (out / 'report.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    for name in ['metrics.json', 'distribution.json', 'render_summary.json', 'multiview_report.json',
                 'cameras.csv', 'camera_overlap.csv', 'rendered_camera_overlap.csv', 'scenes.csv', 'windows.csv',
                 'placement_heatmaps.svg']:
        shutil.copyfile(args.captures / name, out / name)
    for name in ['embeddings.json', 'embeddings.f32']:
        shutil.copyfile(args.embeddings.parent / name, out / name)
    # Preserve audit identity without committing multi-gigabyte numeric captures.
    hashes = {}
    for path, _, _ in captures:
        for file in sorted(path.iterdir()):
            if file.suffix in ('.json', '.png', '.rgba32f'):
                with file.open('rb') as stream:
                    hashes[str(file.relative_to(args.captures))] = hashlib.file_digest(stream, 'sha256').hexdigest()
    (out / 'capture_sha256.json').write_text(json.dumps(hashes, indent=2) + '\n')
    macros = {'AuditRooms': metrics['scenes'], 'RenderRooms': summary['rendered_scenes'],
              'NumericSeries': len(metrics['numeric']), 'VaryingSeries': varying,
              'CategorySeries': len(entropy), 'InvalidRooms': len(audit['invalid_seeds']),
              'SpeedupPercent': f'{speedup:.1f}'}
    (tex / 'results.tex').write_text('% Generated by scripts/paper_indoor_report.py\n' +
        ''.join(f'\\newcommand{{\\{key}}}{{{value}}}\n' for key, value in macros.items()))
    with (args.captures / 'cameras.csv').open() as stream:
        camera_count = len(list(csv.DictReader(stream)))
    n = metrics['numeric']
    chairs = discrete_distribution(metrics['object_counts_per_scene']['main/Chair'])
    fmt = lambda d: f"{d['min']:.2f}--{d['max']:.2f} (mean {d['mean']:.2f})"
    table(tex / 'diversity_table.tex', [
        ('Audited rooms / cameras', f"{metrics['scenes']} / {camera_count}"),
        ('Room area (m$^2$)', fmt(n['room_area_m2'])),
        ('Main-interior chairs per scene (zeros included)', fmt(chairs)),
        ('People per scene', fmt(n['people_per_scene'])),
        ('Exterior window walls / near full-height rooms', f"1--3 / {metrics['categories']['exterior_full_height_room']['true']}"),
        ('Solar illuminance (lux)', fmt(n['sun_illuminance_lux'])),
        ('Vertical field of view (degrees)', fmt(n['vertical_fov_degrees'])),
        ('Observed / entropy-effective activity categories', f"{entropy['layout']['observed']} / {entropy['layout']['effective_count']:.2f}"),
        ('Quantized occupancy-signature estimate', f"{metrics['occupancy_signature_estimate']:.1f}"),
    ], 'Consecutive layout population. Occupancy uses HyperLogLog with 4096 registers (approximately 1.63\\% relative standard error), 12$\\times$12 class occupancy and 10 cm partition quantization; random seeds and appearance are excluded.', 'tab:diversity')
    nn = embedding['cross_scene_nearest_cosine_distance']
    table(tex / 'embedding_table.tex', [
        ('Rooms / images / embedding dimensions', f"{embedding['scenes']} / {embedding['images']} / {metadata['shape'][1]}"),
        ('Cross-room nearest cosine distance: min / median / max', f"{nn['min']:.4f} / {nn['p05_p50_p95'][1]:.4f} / {nn['max']:.4f}"),
        ('View fraction with cross-room nearest distance below 0.02', f"{100*embedding['nearest_distance_threshold_fractions']['0.02']:.2f}\\%"),
        ('Scene-centroid entropy-effective rank', f"{embedding['scene_centroid_spectrum']['effective_rank']:.2f}"),
        ('Dimensions explaining 90\\% of centroid variance', str(embedding['scene_centroid_spectrum']['dimensions_for_90_percent_variance'])),
    ], 'Frozen SigLIP2 base-patch16-224 audit. Direct bilinear RGB resize, mean/std 0.5, L2-normalized embeddings. Thresholds are exploratory; no real-data baseline or training-utility claim.', 'tab:embedding')
    table(tex / 'overlap_table.tex', [
        ('Semantic classes/view: minimum / mean (at least 32 pixels)', f"{semantic['classes_per_view']['min']:.0f} / {semantic['classes_per_view']['mean']:.2f}"),
        ('Occupied primary rooms with no person label', f"{semantic['primary_rooms_without_person_label']} / {semantic['occupied_primary_rooms']}"),
        ('Occupied non-neighbor rooms with no person label', f"{semantic['occupied_rooms_without_person_label']} / {semantic['occupied_non_neighbor_rooms']}"),
        ('Rendered reference edges / rooms', f"{overlap['pairs']} / {overlap['scenes']}"),
        ('Worst-pair overlap per room: min / mean', f"{overlap['scene_worst_overlap']['min']:.3f} / {overlap['scene_worst_overlap']['mean']:.3f}"),
        ('Rendered pairs below the 0.35 proxy requirement', str(overlap['pairs_below_requested_overlap'])),
        ('Maximum per-view depth/position p99 error (m)', f"{summary['annotation']['depth_position_p99_metres']['max']:.2e}"),
        ('Maximum per-view reprojection p99 error (pixels)', f"{summary['annotation']['reprojection_p99_pixels']['max']:.5f}"),
    ], 'Actual geometry audit at the captured timestep. Visibility requires depth agreement within 0.01 m plus 0.002 times depth; nearest-pixel raster boundaries and thin geometry differ from proxy rays.', 'tab:overlap')
    perf = []
    for label, report in benches.items():
        s = report['summary']
        perf.append((label.replace('_', ' '), f"{s['views_per_second']:.2f} views/s; {s['scene_seconds_p50_p95'][0]:.3f}/{s['scene_seconds_p50_p95'][1]:.3f} s"))
    table(tex / 'performance_table.tex', perf,
          'Completed views per second; room wall-time median/p95. Baseline/shared geometry/optimized/parallel are local-patch runs with identical independent-camera settings. Registry rows use the normalized published dependency graph and the new grouped-camera default. See the evidence for modes, startup, memory and stage timings.', 'tab:performance')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 5.5), constrained_layout=True)
    plots = [(n['room_area_m2'], 'Room area (m²)'),
             (chairs, 'Main-interior chairs'),
             (n['vertical_fov_degrees'], 'Vertical field of view (°)'),
             (n['exterior_opening_area_fraction'], 'Rough opening / facade area'),
             (n['camera_reference_baseline_m'], 'Reference baseline (m)')]
    for ax, (d, name) in zip(axes.flat, plots):
        edges = np.array(d['bin_edges'])
        ax.bar(edges[:-1], d['bin_counts'], width=np.diff(edges), align='edge', color='#376c82')
        ax.set(xlabel=name, ylabel='Count')
    bins = metrics['joint_histograms']['sun_log10_vs_electric_log10']
    # The export stores a named histogram object; reject schema drift.
    if isinstance(bins, dict):
        bins = bins['counts']
    axes[1, 2].imshow(np.asarray(bins).reshape(16, 16), origin='lower', extent=(-1, 5, 0, 3), aspect='auto')
    axes[1, 2].set(xlabel='Solar illuminance (log10 lux)', ylabel='Electric target (log10 lux)')
    fig.savefig(tex / 'distributions.png', dpi=170)
    plt.close(fig)
    from PIL import Image, ImageDraw
    sheet = Image.new('RGB', (384 * 4, 310 * 2), 'white')
    draw = ImageDraw.Draw(sheet)
    for i, (path, _, manifest) in enumerate(captures[:8]):
        tile = Image.open(path / 'view_00_color.png').convert('RGB').resize((384, 288))
        x, y = (i % 4)*384, (i // 4)*310
        sheet.paste(tile, (x, y))
        draw.text((x + 5, y + 291), f"seed {manifest['seed']}", fill='black')
    sheet.save(tex / 'scenes.jpg', quality=94)
    print(json.dumps(dict(audit=metrics['scenes'], rendered=summary['rendered_scenes'],
                         time_reduction_percent=speedup, effective_rank=embedding['scene_centroid_spectrum']['effective_rank'])))


if __name__ == '__main__':
    main()
