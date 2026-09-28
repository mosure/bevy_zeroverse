#!/usr/bin/env python3
"""Build v20 paper/site camera results from retained, completed audit reports."""
import argparse
import csv
import json
from pathlib import Path
import shutil

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def build(evidence, audit):
    groups = json.loads((evidence / 'summary.json').read_text())['cohorts']
    before, after = groups['before'], groups['after']
    cv = json.loads((evidence / 'covisibility_report.json').read_text())
    metrics = json.loads((audit / 'metrics.json').read_text())
    assert cv['scenes'] == 512 and cv['seed_range'] == [24000,24511] and after['scenes'] == 2048
    assert after['scenes'] == before['scenes'] == metrics['scenes']
    assert cv['generator_version'] == metrics['generator_version'] == 20
    assert after['starts_below_quarter_spread'] == 0
    media = ROOT / 'www/project/static/media'
    generated = ROOT / 'tex/generated'
    plt.rcParams.update({'svg.fonttype':'none', 'font.size':10})
    times = list(csv.DictReader((evidence / 'covisibility_times.csv').open()))
    worst = {}
    for row in times:
        seed = int(row['seed'])
        worst[seed] = min(worst.get(seed, 1), float(row['reference_min_overlap']))
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    axes[0].bar(range(cv['cameras']), np.array(cv['other_camera_count_fractions_valid']) * 100, color='#17624d')
    axes[0].set(xlabel='Other cameras seeing the source surface', ylabel='Valid source-pixel observations (%)',
                title=f"{cv['views']:,} rendered views / {cv['scenes']} rooms", xticks=range(cv['cameras']))
    axes[1].hist(list(worst.values()), bins=np.linspace(0, 1, 21), color='#356aa0')
    axes[1].axvline(.35, color='#a55437', linestyle='--', label='35% proxy target')
    axes[1].set(xlabel='Worst reference-pair overlap per room', ylabel='Rooms', title='All three captured times included')
    axes[1].legend(fontsize=8)
    for ax in axes:
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
    fig.savefig(evidence / 'covisibility.png', dpi=160)
    fig.savefig(evidence / 'covisibility.svg')
    plt.close(fig)
    for source, destinations in [
        (evidence / 'camera_paths.png', [media / 'camera-paths-v20.png', generated / 'camera_paths_v20.png']),
        (evidence / 'covisibility.png', [media / 'covisibility-v20.png', generated / 'covisibility_v20.png']),
        (evidence / 'summary.json', [media / 'camera-groups-v20.json']),
        (evidence / 'covisibility_report.json', [media / 'covisibility-v20.json']),
        (audit / 'metrics.json', [evidence / 'metrics.json', media / 'population-v20.json']),
        (audit / 'distribution.json', [evidence / 'distribution.json']),
        (audit / 'camera_groups.csv', [evidence / 'camera_groups.csv']),
    ]:
        for destination in destinations:
            shutil.copyfile(source, destination)
    mean_overlap = cv['reference_overlap']['mean'] * 100
    worst_overlap = cv['reference_overlap']['min'] * 100
    fractions = np.array(cv['other_camera_count_fractions_valid']) * 100
    before_spread, after_spread = before['start_spread']['median'], after['start_spread']['median']
    varying = sum(d['standard_deviation'] > 1e-12 for d in metrics['numeric'].values())
    macros = dict(CameraAuditRooms=f"{after['scenes']:,}", CameraRenderRooms=str(cv['scenes']),
        CameraRenderViews=f"{cv['views']:,}", CameraReferencePairs=f"{cv['reference_pairs']:,}",
        CameraMeanOverlap=f'{mean_overlap:.1f}', CameraMinimumOverlap=f'{worst_overlap:.1f}',
        CameraTargetMisses=f"{cv['reference_pairs_below_requested']:,}",
        CameraOldLines=str(before['starts_below_quarter_spread']), CameraNewLines=str(after['starts_below_quarter_spread']))
    (generated / 'camera_results.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    quality = cv['rendered_quality']
    tex = rf"""We compare generators 19 and 20 on \CameraAuditRooms{{}} consecutive layout seeds (24000--26047), with four cameras, furnishing density 0.65 and static human density 0.25. Object, people and architecture CSV hashes agree exactly. All new layouts pass geometric, swept-path and sampled group checks. The new export contains {len(metrics['numeric'])} numeric series, {varying} with observed nonzero variance; these remain correlated diagnostics, not independent degrees of freedom. The source reports and commands are retained in \texttt{{docs/camera\_evaluation\_v20.md}}.

\begin{{table}}[ht]
\centering\small
\begin{{tabular}}{{lrr}}\toprule
Camera-group statistic & Generator 19 & Generator 20\\\midrule
Near-collinear starting groups (spread $<0.25$) & \CameraOldLines{{}} & \CameraNewLines{{}}\\
Median starting spread & {before_spread:.3f} & {after_spread:.3f}\\
Median minimum pair separation (m) & {before['distributions']['min_pairwise_baseline_m']['median']:.3f} & {after['distributions']['min_pairwise_baseline_m']['median']:.3f}\\
Median minimum relative motion & approximately zero & {after['distributions']['min_relative_motion']['median']:.3f}\\\bottomrule
\end{{tabular}}
\caption{{Matched layout comparison. Legacy CSVs expose endpoint checks; new path metrics use 33 synchronized samples. Generator 19 translated the same reference path for all views.}}
\end{{table}}

\begin{{figure}}[ht]
\centering\includegraphics[width=\linewidth]{{generated/camera_paths_v20.png}}
\caption{{Seed 24005 before and after the group/path diversity constraints. Starts are circles, endpoints crosses.}}
\end{{figure}}

The rendered cohort contains \CameraRenderRooms{{}} consecutive rooms (24000--24511), four $320\times240$ cameras and times $0,0.5,1$: \CameraRenderViews{{}} views in {len(cv['runs'])} completed process chunks. Native lights and shadows remain enabled; baked GI is disabled for this geometric audit. No dark, weak or low-overlap view is discarded. Shared-depth reference overlap over \CameraReferencePairs{{}} pair/time observations averages \CameraMeanOverlap{{}}\%, with minimum \CameraMinimumOverlap{{}}\%. \CameraTargetMisses{{}} observations miss the 35\% proxy target; {cv['scenes_with_reference_pair_below_10_percent']} rooms contain a reference pair below 10\%.

We additionally evaluate all {cv['directed_pairs']:,} directed camera-pair/time observations. For each valid source pixel we form a membership mask of other cameras passing rendered-depth reprojection, excluding the source bit. Across {cv['source_valid_pixels']:,} valid source-pixel observations, the fractions seen by zero, one, two and three other cameras are {fractions[0]:.1f}\%, {fractions[1]:.1f}\%, {fractions[2]:.1f}\% and {fractions[3]:.1f}\%, respectively. Observations repeat surfaces across views and times; these are not counts of unique 3D points. {cv['disconnected_sets_at_10_percent']} of {cv['synchronized_sets']:,} synchronized sets have a disconnected graph when edges require 10\% bidirectional overlap.

This independent diagnostic uses nearest-pixel depth agreement within $0.01\,\mathrm{{m}}+0.002z$. It is deliberately separate from the production GPU co-visibility annotation's symmetric tangent-plane test described below; its masks must not be interpreted as identical annotation values. Both treat glass as an opaque geometric surface. Gallery mask exports retain and validate the production annotation contract.

\begin{{figure}}[ht]
\centering\includegraphics[width=\linewidth]{{generated/covisibility_v20.png}}
\caption{{Expanded co-visibility evaluation: source-pixel membership cardinality and per-room worst reference overlap. All captured times and failures are retained.}}
\end{{figure}}

The same captures expose quality tails: {quality['rooms_with_a_view_having_at_most_two_semantic_classes']} rooms contain a view with at most two visible semantic classes; {quality['occupied_rooms_without_person_pixels']} of {quality['rooms_with_primary_people']} rooms containing primary-room people have no person pixels across the captured views. These diagnostics neither establish photographic realism nor repeat the historical embedding and throughput experiments. The largest depth/position p99 disagreement is {quality['annotations']['depth_position_p99_metres']['max']:.6f}\,m. Four bounded processes do not establish unlimited-process memory stability.
"""
    (generated / 'camera_evaluation.tex').write_text(tex)
    html = f"""
    <div class="cohort-heading"><h3>Camera robustness and co-visibility</h3><span>generator v20 · consecutive seeds</span></div>
    <div class="metric-strip">
      <div><strong>{after['scenes']:,}</strong><span>audited layouts</span><small>24000–26047 · zero invalid layouts</small></div>
      <div><strong>{cv['scenes']:,}</strong><span>rendered rooms / {cv['views']:,} views</span><small>four cameras · three times · no GI</small></div>
      <div><strong>{before['starts_below_quarter_spread']} → 0</strong><span>near-collinear starting groups</span><small>matched layouts · spread threshold 0.25</small></div>
    </div>
    <figure class="chart"><img src="static/media/camera-paths-v20.png" loading="lazy" width="1920" height="960" alt="Four camera paths before and after the spread and independent-motion constraints"><figcaption>Median starting spread increases from {before_spread:.3f} to {after_spread:.3f}. Paths keep shared-content, primary-room and swept-clearance checks. New group metrics inspect 33 synchronized times.</figcaption></figure>
    <figure class="chart"><img src="static/media/covisibility-v20.png" loading="lazy" width="1760" height="576" alt="Histogram of other-camera visibility and the worst reference overlap per rendered room"><figcaption>All {cv['directed_pairs']:,} directed camera-pair/time observations. Membership excludes the source camera; counts describe valid source pixels, not unique 3D points.</figcaption></figure>
    <div class="evidence-grid">
      <article><div class="eyebrow">ALL-CAMERA CO-VISIBILITY</div><h3>{100-fractions[0]:.1f}% shared with another camera</h3><p>Of valid source pixels, {fractions[1]:.1f}% are visible in one other camera, {fractions[2]:.1f}% in two, and {fractions[3]:.1f}% in all three. {fractions[0]:.1f}% have no other observing camera.</p><p class="note">Independent rendered-depth reprojection with a declared tolerance. The gallery's GPU annotation uses its separate tangent-plane test; these metrics do not claim identical masks.</p></article>
      <article><div class="eyebrow">RETAINED DIFFICULT PAIRS</div><h3>{mean_overlap:.1f}% mean reference overlap</h3><p>{cv['reference_pairs']:,} pair/time observations; minimum {worst_overlap:.1f}%. {cv['reference_pairs_below_requested']:,} miss the 35% proxy target. {cv['scenes_with_reference_pair_below_10_percent']} rooms contain a reference pair below 10%.</p><p class="note">All 512 rooms and all captured times are included. {quality['rooms_with_a_view_having_at_most_two_semantic_classes']} rooms include a view with at most two semantic classes. Photographic realism and downstream training gains remain unproven.</p></article>
    </div>
    <div class="evidence-downloads"><a href="static/media/covisibility-v20.json" download>Co-visibility report (JSON) ↓</a><a href="static/media/camera-groups-v20.json" download>Camera comparison (JSON) ↓</a><a href="static/media/population-v20.json" download>2,048-room distributions (JSON) ↓</a><a href="https://github.com/mosure/bevy_zeroverse/blob/main/docs/camera_evaluation_v20.md">Protocol and limitations ↗</a></div>
"""
    page = ROOT / 'www/project/index.html'
    source = page.read_text()
    start, end = '<!-- CAMERA_EVALUATION_START -->', '<!-- CAMERA_EVALUATION_END -->'
    assert source.count(start) == source.count(end) == 1
    page.write_text(source.split(start)[0] + start + html + '    ' + end + source.split(end)[1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, default=ROOT / 'docs/evidence/camera_v20')
    parser.add_argument('--audit', type=Path, default=ROOT / 'out/camera_release/after')
    args = parser.parse_args()
    build(args.evidence, args.audit)
