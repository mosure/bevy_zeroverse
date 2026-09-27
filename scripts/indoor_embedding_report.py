#!/usr/bin/env python3
"""Index completed captures for indoor_embed; audit spacing in SigLIP2 image space.

Nearest neighbors exclude the same scene. Trajectory distances are positive
controls, reported separately. Thresholds are diagnostics, not utility gates.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from indoor_report import distribution, summarize


def make_index(cohorts, output):
    samples, records = [], {}
    for spec in cohorts:
        name, directory = spec.split('=', 1)
        if not name or name in records:
            raise ValueError('cohort names must be nonempty and unique')
        root = Path(directory).resolve()
        _, captures, summary = summarize(root)
        records[name] = {k: summary[k] for k in [
            'run_id', 'generator_version', 'capture_engine', 'selection_policy',
            'rendered_scenes', 'rendered_views', 'image_size']}
        records[name]['root'] = str(root)
        for path, capture, manifest in captures:
            for i, view in enumerate(capture['views']):
                image = path/f'view_{i:02}_color.png'
                samples.append(dict(path=str(image), sha256=hashlib.sha256(image.read_bytes()).hexdigest(),
                                    cohort=name, seed=manifest['seed'], camera=view['camera_index'], time=view['time']))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(dict(schema_version=1, samples=samples, cohorts=records), indent=2)+'\n')
    return dict(images=len(samples), cohorts=records)


def load_embeddings(path):
    metadata = json.loads(path.read_text())
    tensor_path = path.parent/'embeddings.f32'
    with tensor_path.open('rb') as file:
        digest = hashlib.file_digest(file, 'sha256').hexdigest()
    if metadata['schema_version'] != 1 or digest != metadata['tensor_sha256']:
        raise ValueError('embedding identity/checksum mismatch')
    rows, dimensions = metadata['shape']
    if rows != len(metadata['samples']) or tensor_path.stat().st_size != rows*dimensions*4:
        raise ValueError('embedding shape mismatch')
    vectors = np.memmap(tensor_path, dtype='<f4', mode='r', shape=(rows, dimensions))
    if not np.isfinite(vectors).all() or not np.allclose(np.linalg.norm(vectors, axis=1), 1, atol=2e-5):
        raise ValueError('nonfinite or unnormalized embeddings')
    return metadata, vectors


def cross_scene_nearest(vectors, scene_ids):
    """Exact, blockwise nearest neighbors for bounded audit cohorts."""
    ids = np.asarray(scene_ids)
    if len(set(scene_ids)) < 2:
        raise ValueError('at least two scenes required')
    distances, neighbors = [], []
    for start in range(0, len(vectors), 64):
        similarity = np.clip(vectors[start:start+64] @ vectors.T, -1, 1)
        similarity[ids[start:start+64, None] == ids[None, :]] = -np.inf
        nearest = np.argmax(similarity, axis=1)
        distances.extend((1-similarity[np.arange(len(nearest)), nearest]).tolist())
        neighbors.extend(nearest.tolist())
    return np.asarray(distances), np.asarray(neighbors)


def spectrum(vectors):
    centered = vectors - vectors.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    power = singular.astype(np.float64)**2
    if power.sum() <= 1e-12:
        return dict(effective_rank=0, dimensions_for_90_percent_variance=0)
    p = power[power > 1e-12]/power.sum()
    return dict(effective_rank=float(np.exp(-(p*np.log(p)).sum())),
                dimensions_for_90_percent_variance=int(np.searchsorted(np.cumsum(p), .90)+1))


def cohort_metrics(vectors, rows):
    # At 10M scale, use a declared sample, not a quadratic full-dataset search.
    if len(rows) > 4096:
        raise ValueError('exact report is bounded to 4096 views per cohort; sample scenes before capture')
    ids = [r['seed'] for r in rows]
    distances, neighbors = cross_scene_nearest(vectors, ids)
    temporal, other_view, duplicates = [], [], []
    for i, row in enumerate(rows):
        for j in range(i+1, len(rows)):
            other = rows[j]
            if row['sha256'] == other['sha256']:
                duplicates.append(float(max(0, 1-np.dot(vectors[i], vectors[j]))))
            if row['seed'] != other['seed']:
                continue
            distance = float(np.clip(1-np.dot(vectors[i], vectors[j]), 0, 2))
            if row['camera'] == other['camera'] and row['time'] != other['time']:
                temporal.append(distance)
            elif row['camera'] != other['camera'] and row['time'] == other['time']:
                other_view.append(distance)
    # Every scene contributes once to the centroid/spectrum, regardless of view count.
    centroids = np.array([vectors[np.array(ids) == seed].mean(axis=0) for seed in sorted(set(ids))])
    centroids /= np.maximum(np.linalg.norm(centroids, axis=1, keepdims=True), 1e-12)
    scene_distances, _ = cross_scene_nearest(centroids, sorted(set(ids)))
    # Show each room pair once, keeping its closest view pair. Trajectory frames
    # must not crowd the review sheet with repetitions of the same two rooms.
    unique_pairs = {}
    for i, j in enumerate(neighbors):
        key = tuple(sorted((ids[i], ids[j])))
        pair = (float(distances[i]), min(i, int(j)), max(i, int(j)))
        if key not in unique_pairs or pair < unique_pairs[key]:
            unique_pairs[key] = pair
    pairs = sorted(unique_pairs.values())[:16]
    return {
        'images': len(rows), 'scenes': len(set(ids)),
        'cross_scene_nearest_cosine_distance': distribution(distances.tolist()),
        'scene_centroid_nearest_cosine_distance': distribution(scene_distances.tolist()),
        'same_camera_trajectory_cosine_distance': distribution(temporal),
        'same_scene_other_camera_cosine_distance': distribution(other_view),
        'exact_file_duplicate_cosine_distance': distribution(duplicates),
        'scene_centroid_spectrum': spectrum(centroids),
        'nearest_distance_threshold_fractions': {str(t): float(np.mean(distances < t)) for t in [.005,.01,.02,.05,.10]},
        'closest_cross_scene_pairs': [dict(distance=d, left=rows[i], right=rows[j]) for d,i,j in pairs],
    }


def report(path, figures=True):
    metadata, vectors = load_embeddings(path)
    names = sorted({r['cohort'] for r in metadata['samples']})
    cohorts = {}
    for name in names:
        indices = [i for i,r in enumerate(metadata['samples']) if r['cohort'] == name]
        cohorts[name] = cohort_metrics(np.asarray(vectors[indices]), [metadata['samples'][i] for i in indices])
    result = {
        'schema_version': 1, 'encoder': metadata['encoder'], 'variant': metadata['variant'],
        'model_sha256': metadata['loaded_weights']['loaded_weight_sha256'],
        'input_tensor_sha256': metadata['tensor_sha256'], 'cohorts': cohorts,
        'protocol': 'Unit-normalized image embeddings; cosine distance = 1 - dot product. Cross-scene comparisons exclude every camera and timestep of the same scene. Each cohort is evaluated separately with exact neighbors.',
        'interpretation': 'Smaller distance means greater encoder similarity. Inspect nearest pairs. Thresholds are exploratory; no universal acceptable spacing or photographic-realism threshold is assumed. Scene centroids have equal scene weight. Trajectory frames are correlated positive controls, not new independent scenes.',
        'limits': ['This is a sampled perceptual redundancy audit, not evidence of downstream pretraining utility.',
                   'Results depend on model size, preprocessing, cohort size and camera policy. Compare matched cohort sizes.',
                   'A global semantic embedding may ignore geometry, material defects or annotation errors.'],
    }
    (path.parent/'spacing.json').write_text(json.dumps(result, indent=2)+'\n')
    if figures:
        make_figures(path.parent, result)
    return result


def make_figures(output, result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw, ImageFont
    fig, axes = plt.subplots(1, 2, figsize=(11,4), constrained_layout=True)
    cohorts = result['cohorts']
    labels = list(cohorts)
    for axis, field, title in zip(axes, [
        'cross_scene_nearest_cosine_distance','same_camera_trajectory_cosine_distance'],
        ['Nearest image from another room', 'Same-camera trajectory endpoints']):
        for i,name in enumerate(labels):
            d = cohorts[name][field]
            if d is None:
                continue
            lo, median, hi = d['p05_p50_p95']
            axis.errorbar(i, median, yerr=[[median-lo],[hi-median]], fmt='o', capsize=6)
        axis.set(xticks=range(len(labels)), xticklabels=labels, ylabel='Cosine distance (5th/median/95th)', title=title)
        axis.set_ylim(bottom=0)
    fig.suptitle(f"SigLIP2 {result['variant']} • matched-cohort spacing diagnostics")
    fig.savefig(output/'spacing.svg'); fig.savefig(output/'spacing.png', dpi=140); plt.close(fig)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 12)
    except OSError:
        font = ImageFont.load_default()
    for name, cohort in cohorts.items():
        pairs = cohort['closest_cross_scene_pairs']
        sheet = Image.new('RGB',(960,math.ceil(len(pairs)/2)*212),'#eef1f4')
        draw = ImageDraw.Draw(sheet)
        for i,pair in enumerate(pairs):
            for j,key in enumerate(['left','right']):
                row = pair[key]; x,y=(i%2*2+j)*240,i//2*212
                with Image.open(row['path']) as image:
                    image=image.convert('RGB'); image.thumbnail((240,180)); sheet.paste(image,(x,y))
                draw.text((x+3,y+181),f"seed {row['seed']} cam {row['camera']} t={row['time']}\nd={pair['distance']:.5f}",font=font,fill='#162134')
        safe = ''.join(c if c.isalnum() or c in '_-' else '_' for c in name)
        sheet.save(output/f'nearest_{safe}.jpg',quality=90)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    index=sub.add_parser('index'); index.add_argument('--cohort',action='append',required=True); index.add_argument('--output',type=Path,required=True)
    audit=sub.add_parser('report'); audit.add_argument('embeddings',type=Path); audit.add_argument('--no-figures',action='store_true')
    args=parser.parse_args()
    result=make_index(args.cohort,args.output) if args.command=='index' else report(args.embeddings,not args.no_figures)
    print(json.dumps(result,indent=2))
