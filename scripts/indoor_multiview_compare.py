#!/usr/bin/env python3
"""Compare paired camera policies after indoor_multiview_report.py has run."""
import argparse
import csv
import hashlib
import json
from pathlib import Path


def compare(roots, output):
    from PIL import Image, ImageDraw, ImageFont

    if len({root.name for root in roots}) != len(roots):
        raise ValueError('cohort directory names must be unique')
    cohorts = []
    canonical = None
    for root in roots:
        report = json.loads((root/'multiview_report.json').read_text())
        metrics = json.loads((root/'metrics.json').read_text())
        selection = json.loads((root/'render_selection.json').read_text())
        rows = list(csv.DictReader((root/'rendered_camera_overlap.csv').open()))
        snapshots = []
        for seed in selection['selected_seeds']:
            manifest = json.loads((root/f'seed_{seed:06}'/'manifest.json').read_text())
            for key in ['cameras', 'camera_settings', 'camera_aspect_ratio']:
                manifest.pop(key, None)
            snapshots.append(manifest)
        settings = {k: selection[k] for k in ('quality', 'playback_steps', 'density', 'human_density', 'diffuse_gi_enabled', 'gi_rays', 'gi_bounces')}
        contract = (snapshots, report['image_size'], report['capture_engine'], settings)
        if canonical is not None and canonical != contract:
            raise ValueError('comparison requires identical scenes, render engine, size and time schedule')
        canonical = contract
        if report['pairs'] != len(rows):
            raise ValueError('stale overlap CSV')
        cohorts.append((root, report, metrics, selection, rows))
    if not cohorts:
        raise ValueError('no cohorts')
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(schema_version=1, seeds=cohorts[0][3]['selected_seeds'],
                   capture_engine=cohorts[0][1]['capture_engine'],
                   comparison_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   cohorts={})
    w, h = 320, 240
    seeds = summary['seeds']
    sheet = Image.new('RGB', (len(roots)*2*w, len(seeds)*(h+42)+50), '#f3f5f8')
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 17)
        small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    except OSError:
        font = small = ImageFont.load_default()
    for column, (root, report, metrics, selection, rows) in enumerate(cohorts):
        policy = (report['camera_settings'] or {}).get('multiview')
        title = f"Minimum estimated overlap: {policy['min_overlap']:.0%}" if policy else 'Independent cameras'
        draw.text((column*2*w+10, 14), title, font=font, fill='#142d40')
        for row_index, seed in enumerate(seeds):
            row = next(r for r in rows if int(r['seed']) == seed and int(r['step']) == 0 and int(r['camera']) == 1)
            top = 50+row_index*(h+42)
            for camera in range(2):
                image = Image.open(root/f'seed_{seed:06}'/f'view_{camera:02}_color.png').convert('RGB')
                image.thumbnail((w, h))
                sheet.paste(image, (column*2*w+camera*w, top))
            draw.text((column*2*w+8, top+h+5), f"seed {seed} | shared {float(row['bidirectional_min']):.1%} | baseline {float(row['baseline_m']):.2f} m", font=small, fill='#142d40')
        summary['cohorts'][root.name] = dict(
            camera_settings=report['camera_settings'], audited_scenes=metrics['scenes'],
            rendered_scenes=report['scenes'], rendered_pairs=report['pairs'],
            rendered_distributions=report['distributions'], scene_worst_overlap=report['scene_worst_overlap'],
            pairs_below_requested_overlap=report['pairs_below_requested_overlap'],
            scenes_with_under_10_percent_overlap=report['scenes_with_under_10_percent_overlap'],
            cpu_distributions={k: metrics['numeric'][k] for k in ['camera_overlap_bidirectional_min', 'camera_reference_baseline_m', 'camera_mean_triangulation_degrees', 'vertical_fov_degrees', 'trajectory_length_m']},
            report_sha256=hashlib.sha256((root/'multiview_report.json').read_bytes()).hexdigest())
    sheet.save(output/'comparison.jpg', quality=91)
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    compare(args.roots, args.output)
