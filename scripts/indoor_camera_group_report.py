#!/usr/bin/env python3
"""Audit camera-group spread and motion; optionally compare the same room seeds.

Requires NumPy and matplotlib. Uses camera_paths.csv when available; legacy
cohorts expose endpoint checks only, explicitly labeled in the output.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def group_geometry(positions):
    """positions[time, camera, xyz], in metres at synchronized times."""
    p = np.asarray(positions, dtype=np.float64)
    if p.ndim != 3 or p.shape[2] != 3 or not np.isfinite(p).all():
        raise ValueError('expected finite time x camera x xyz positions')
    if p.shape[0] < 1 or p.shape[1] < 2:
        raise ValueError('need at least one time and two cameras')
    a, b = np.triu_indices(p.shape[1], 1)
    planar = p[..., [0, 2]]
    singular = np.linalg.svd(planar - planar.mean(axis=1, keepdims=True), compute_uv=False)
    spread = np.divide(singular[:, 1], singular[:, 0], out=np.zeros(len(p)), where=singular[:, 0] > 1e-12)
    displacement = p - p[0:1]
    travel = np.mean(np.sum(displacement**2, axis=2), axis=0)
    denominator = np.maximum(travel[a], travel[b])
    residual = np.mean(np.sum((displacement[:, a] - displacement[:, b])**2, axis=2), axis=0)
    moving = denominator > 1e-8 / 33
    return dict(
        min_pairwise_baseline_m=float(np.linalg.norm(p[:, a] - p[:, b], axis=2).min()),
        min_reference_baseline_m=float(np.linalg.norm(p[:, 1:] - p[:, :1], axis=2).min()),
        max_reference_baseline_m=float(np.linalg.norm(p[:, 1:] - p[:, :1], axis=2).max()),
        min_horizontal_spread=float(spread.min()) if p.shape[1] >= 3 else None,
        min_relative_motion=float(np.sqrt(residual[moving] / denominator[moving]).min()) if moving.any() else None,
    )


def load(root):
    rows = list(csv.DictReader((root / 'cameras.csv').open()))
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row['seed']), []).append(row)
    paths = {}
    path_file = root / 'camera_paths.csv'
    if path_file.exists():
        for row in csv.DictReader(path_file.open()):
            seed_paths = paths.setdefault(int(row['seed']), {})
            key = (int(row['camera']), float(row['time']))
            if key in seed_paths:
                raise ValueError('duplicate camera/time in paths')
            seed_paths[key] = [float(row[f'{axis}_m']) for axis in 'xyz']
        if set(paths) != set(grouped):
            raise ValueError('camera paths and metadata have different seeds')
    samples, starts = {}, {}
    for seed, cameras in grouped.items():
        cameras.sort(key=lambda r: int(r['camera']))
        if [int(r['camera']) for r in cameras] != list(range(len(cameras))):
            raise ValueError('non-contiguous camera indices')
        endpoints = np.array([[[float(r[f'{end}_{axis}_m']) for axis in 'xyz']
                               for r in cameras] for end in ('start', 'end')])
        starts[seed] = endpoints[:1]
        if paths:
            seed_paths = paths[seed]
            times = sorted({t for _, t in seed_paths})
            if len(seed_paths) != len(cameras) * len(times):
                raise ValueError('camera paths and metadata have different camera counts')
            samples[seed] = np.array([[seed_paths[(i, t)] for i in range(len(cameras))] for t in times])
            if len(times) != 33 or not np.allclose(times, np.linspace(0, 1, 33)):
                raise ValueError('expected 33 uniformly sampled times')
            if not np.allclose(samples[seed][[0, -1]], endpoints, atol=1e-5):
                raise ValueError('path endpoints differ from camera metadata')
        else:
            samples[seed] = endpoints
    return samples, starts, '33 synchronized times' if paths else 'endpoints only; intermediate motion unmeasured'


def distribution(values):
    values = np.array([v for v in values if v is not None])
    return None if not len(values) else dict(count=len(values), min=float(values.min()),
        median=float(np.median(values)), mean=float(values.mean()), max=float(values.max()))


def report(roots, output, seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output.mkdir(parents=True, exist_ok=True)
    report = dict(schema_version=1, cohorts={}, limitations='Finite time samples. Spatial and path diversity do not establish rendered overlap or learning utility.')
    fig, axes = plt.subplots(1, len(roots), figsize=(6 * len(roots), 6), squeeze=False)
    previous = None
    for root, ax in zip(roots, axes[0]):
        samples, starts, convention = load(root)
        hashes = {f: hashlib.sha256((root / f).read_bytes()).hexdigest()
                  for f in ('cameras.csv', 'objects.csv', 'humans.csv', 'scenes.csv', 'metrics.json')}
        contract = (sorted(samples), hashes['objects.csv'], hashes['humans.csv'], hashes['scenes.csv'])
        if previous is not None and previous != contract:
            raise ValueError('comparison requires the same seeds, architecture, objects and people')
        previous = contract
        groups = {s: group_geometry(p) for s, p in samples.items()}
        start_metrics = {s: group_geometry(p) for s, p in starts.items()}
        if (root / 'camera_groups.csv').exists():
            exported = list(csv.DictReader((root / 'camera_groups.csv').open()))
            if sorted(int(r['seed']) for r in exported) != sorted(groups):
                raise ValueError('incomplete camera-group export')
            for row in exported:
                for key, value in groups[int(row['seed'])].items():
                    if key == 'min_reference_baseline_m' and key not in row:
                        continue  # Older exports did not record this bound.
                    actual = float(row[key]) if row[key] else None
                    if (actual is None) != (value is None) or (value is not None and not np.isclose(actual, value, atol=2e-5)):
                        raise ValueError(f'exported geometry disagrees with independent NumPy calculation: {row["seed"]} {key}')
            hashes['camera_paths.csv'] = hashlib.sha256((root / 'camera_paths.csv').read_bytes()).hexdigest()
        report['cohorts'][root.name] = dict(scenes=len(groups), convention=convention, hashes=hashes,
            generator_version=json.loads((root / 'metrics.json').read_text())['generator_version'],
            distributions={key: distribution(g[key] for g in groups.values()) for key in next(iter(groups.values()))},
            start_spread=distribution(g['min_horizontal_spread'] for g in start_metrics.values()),
            starts_below_quarter_spread=sum(g['min_horizontal_spread'] is not None and g['min_horizontal_spread'] < .25 - 1e-5 for g in start_metrics.values()),
            sample_seed=seed, sample_geometry=groups[seed])
        p = samples[seed]
        for i in range(p.shape[1]):
            ax.plot(p[:, i, 0], p[:, i, 2], label=f'Camera {i}', color=f'C{i}')
            ax.scatter(*p[0, i, [0, 2]], color=f'C{i}', marker='o')
            ax.scatter(*p[-1, i, [0, 2]], color=f'C{i}', marker='x')
        measurement = '33 samples' if p.shape[0] == 33 else 'endpoint chords'
        ax.set(title=f'{root.name}: seed {seed} ({measurement})', xlabel='x (m)', ylabel='z (m)', aspect='equal')
        ax.grid(alpha=.2)
        ax.legend(fontsize=9)
    # Identical axes make the before/after footprint directly comparable.
    xlims, ylims = [ax.get_xlim() for ax in axes[0]], [ax.get_ylim() for ax in axes[0]]
    for ax in axes[0]:
        ax.set_xlim(min(x[0] for x in xlims), max(x[1] for x in xlims))
        ax.set_ylim(min(y[0] for y in ylims), max(y[1] for y in ylims))
    fig.suptitle('Camera paths in the primary room (circle=start, cross=end)')
    fig.tight_layout()
    fig.savefig(output / 'camera_paths.png', dpi=160)
    plt.close(fig)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=24005)
    args = parser.parse_args()
    if len({p.name for p in args.roots}) != len(args.roots):
        parser.error('cohort directories must have unique names')
    print(json.dumps(report(args.roots, args.output, args.seed), indent=2))
