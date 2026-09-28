#!/usr/bin/env python3
"""Current-generator camera-baseline sweep: matched scenes, absolute metrics only."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
from indoor_camera_group_report import group_geometry, load
from indoor_covisibility_report import report as covisibility_report
from indoor_report import distribution

ROOT = Path(__file__).resolve().parents[1]
LEVELS = [0.0, 0.25, 0.5, 0.75, 1.0]


def read(path):
    return json.loads(path.read_text())


def write(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def scene_hash(path):
    manifest = read(path)
    geometry = {k:v for k,v in manifest.items()
                if k not in ('cameras', 'camera_settings', 'camera_aspect_ratio')}
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def camera_metrics(root, seeds):
    paths, _, _ = load(root)
    exported = {int(row['seed']):row for row in csv.DictReader((root/'camera_groups.csv').open())}
    lengths = {}
    for row in csv.DictReader((root/'cameras.csv').open()):
        lengths.setdefault(int(row['seed']), []).append(float(row['path_length_m']))
    values = []
    for seed in seeds:
        p = paths[seed]
        group = group_geometry(p)
        for key, expected in group.items():
            actual = float(exported[seed][key]) if exported[seed][key] else None
            assert (expected is None) == (actual is None)
            if actual is not None:
                assert np.isclose(expected, actual, atol=2e-5), (seed, key)
        values.append(dict(seed=seed, mean_reference_baseline_m=float(np.linalg.norm(p[:,1:]-p[:,:1],axis=2).mean()),
            mean_path_length_m=float(np.mean(lengths[seed])), **group))
    return values, paths


def collect(root, output, analyze):
    default_roots = [root/f'b050_{seed}' for seed in [24000,24128,24256,24384]]
    if analyze:
        covisibility_report(default_roots, output/'default', 512, 4)
    default = read(output/'default/covisibility_report.json')
    assert default['scenes'] == 512 and default['generator_version'] == 21
    seeds = list(range(24000,24128))
    expected_geometry = {seed:scene_hash(default_roots[0]/f'seed_{seed:06}/manifest.json') for seed in seeds}
    rows, group_rows, paths = [], [], {}
    for level in LEVELS:
        name = f'b{round(level*100):03}'
        source = root/f'{name}_24000'
        if analyze:
            covisibility_report([source], output/name, 128, 4)
        cv = read(output/name/'covisibility_report.json')
        assert cv['scenes'] == 128 and cv['seed_range'] == [24000,24127] and cv['generator_version'] == 21
        assert cv['capture_engine'] == default['capture_engine'] and cv['image_size'] == [320,240]
        assert all(scene_hash(source/f'seed_{seed:06}/manifest.json') == expected_geometry[seed] for seed in seeds)
        groups, sampled = camera_metrics(source, seeds)
        paths[level] = sampled[24005]
        group_rows.extend(dict(baseline=level, **g) for g in groups)
        times = list(csv.DictReader((output/name/'covisibility_times.csv').open()))
        room_overlap = [float(np.mean([float(t['reference_mean_overlap']) for t in times if int(t['seed']) == seed])) for seed in seeds]
        rows.append(dict(baseline=level, rooms=128, views=cv['views'], policy=cv['camera_settings']['multiview'],
            reference_baseline_m=distribution([g['mean_reference_baseline_m'] for g in groups]),
            room_mean_path_length_m=distribution([g['mean_path_length_m'] for g in groups]),
            minimum_pair_separation_m=distribution([g['min_pairwise_baseline_m'] for g in groups]),
            room_mean_reference_overlap=distribution(room_overlap), reference_overlap=cv['reference_overlap'],
            other_camera_fractions=cv['other_camera_count_fractions_valid'], valid_pixels=cv['source_valid_pixels'],
            reference_pairs=cv['reference_pairs'], proxy_target_misses=cv['reference_pairs_below_requested'],
            disconnected_at_10_percent=cv['disconnected_sets_at_10_percent'],
            capture_export_seconds=distribution([read(source/f'seed_{seed:06}/capture.json')['elapsed_seconds'] for seed in seeds]),
            covisibility_report_sha256=hashlib.sha256((output/name/'covisibility_report.json').read_bytes()).hexdigest()))
    with (output/'camera_groups.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(group_rows[0])); writer.writeheader(); writer.writerows(group_rows)
    metrics = read(default_roots[0]/'metrics.json')
    assert metrics['scenes'] == 2048 and metrics['generator_version'] == 21
    audit = read(default_roots[0]/'distribution.json')
    assert not audit['invalid_seeds']
    result = dict(schema_version=1, generator_version=21, default_baseline=0.5, audit_rooms=2048,
        distinct_rendered_rooms=512, rendered_configurations=1024, rendered_views=12288,
        matched_seeds=[24000,24127], default_seeds=[24000,24511], cameras=4, times=[0,.5,1],
        image_size=[320,240], levels=rows, matched_geometry_sha256=expected_geometry,
        default_report_sha256=hashlib.sha256((output/'default/covisibility_report.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        dependency_sha256={name:hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
            for name in ('indoor_camera_group_report.py','indoor_covisibility_report.py','indoor_report.py')},
        software=dict(python=sys.version.split()[0], numpy=np.__version__),
        limitations=['Same current generator at every slider value; no historical version comparison.',
            'Co-visibility is an independent nearest-depth reprojection diagnostic, not exact production GPU mask counts.',
            'Pixel observations repeat surfaces across views and time; scenes are the independent units.',
            'Captures include all consecutive seeds and times; no filtering by overlap or appearance.',
            'Native Auto lighting/shadows, no baked GI for the sweep; full GI is used in the selected gallery.',
            'Capture/export time includes regeneration, readiness, all modes/times, validation and image/raw-file writes; excludes process startup.',
            'Captures use the checkout-local wgpu command-cache/upload optimizations; published crates resolve registry wgpu. These timings do not qualify registry performance.',
            'No photographic-realism, unlimited-process-memory, or downstream-learning qualification.'])
    write(output/'report.json',result)
    for name in ['metrics.json','distribution.json']:
        shutil.copyfile(default_roots[0]/name,output/name)
    return result, default, metrics, paths


def figures(output, result, paths):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','svg.hashsalt':'zeroverse-baseline-21'})
    rows=result['levels']; x=np.array(LEVELS)
    fig,axes=plt.subplots(1,3,figsize=(14,3.7),constrained_layout=True)
    for ax,key,title,ylabel in [(axes[0],'reference_baseline_m','Realized camera spacing','Mean reference distance per room (m)'),
        (axes[1],'room_mean_reference_overlap','Shared geometry','Mean reference overlap per room')]:
        d=[r[key] for r in rows]
        ax.plot(x,[v['p05_p50_p95'][1] for v in d],'o-',color='#17624d',label='median')
        ax.fill_between(x,[v['p05_p50_p95'][0] for v in d],[v['p05_p50_p95'][2] for v in d],alpha=.18,color='#17624d',label='5th–95th percentile')
        ax.set(title=title,xlabel='Camera baseline control',ylabel=ylabel,xticks=x);ax.legend(fontsize=8)
    bottom=np.zeros(len(rows))
    for count,color in enumerate(['#9da7a4','#adc8c2','#3f917c','#164f41']):
        heights=np.array([r['other_camera_fractions'][count] for r in rows])*100
        axes[2].bar(x,heights,bottom=bottom,width=.17,color=color,label=str(count));bottom+=heights
    axes[2].set(title='Co-visibility membership',xlabel='Camera baseline control',ylabel='Valid source-pixel observations (%)',xticks=x,ylim=(0,100))
    axes[2].legend(title='Other cameras',ncols=4,fontsize=8,loc='lower left')
    for ax in axes: ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.savefig(output/'baseline_sweep.png',dpi=160);fig.savefig(output/'baseline_sweep.svg',metadata={'Date':None});plt.close(fig)
    svg=output/'baseline_sweep.svg'
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    fig,axes=plt.subplots(1,3,figsize=(12,4),sharex=True,sharey=True,constrained_layout=True)
    for ax,b in zip(axes,[0,.5,1]):
        p=paths[b]
        for camera in range(4):
            ax.plot(p[:,camera,0],p[:,camera,2],color=f'C{camera}',label=f'Camera {camera}')
            ax.scatter(*p[0,camera,[0,2]],color=f'C{camera}',marker='o')
            ax.scatter(*p[-1,camera,[0,2]],color=f'C{camera}',marker='x')
        ax.set(title=f'Baseline {b:g}',xlabel='x (m)',ylabel='z (m)',aspect='equal');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=8);fig.suptitle('Seed 24005 · current generator · circles=start, crosses=end')
    fig.savefig(output/'baseline_paths.png',dpi=160);plt.close(fig)


def publish(output, result, default, metrics):
    media=ROOT/'www/project/static/media';generated=ROOT/'tex/generated'
    for name in ['baseline_sweep.png','baseline_paths.png']:
        shutil.copyfile(output/name,media/name);shutil.copyfile(output/name,generated/name)
    shutil.copyfile(output/'report.json',media/'baseline-v21.json')
    shutil.copyfile(output/'metrics.json',media/'population-v21.json')
    shutil.copyfile(output/'default/covisibility_report.json',media/'covisibility-v21.json')
    macros=dict(CameraAuditRooms='2,048',CameraRenderRooms='512',CameraRenderViews='6,144')
    (generated/'camera_results.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    table=[]
    for row in result['levels']:
        f=row['other_camera_fractions']
        table.append(f"{row['baseline']:.2f} & {row['reference_baseline_m']['p05_p50_p95'][1]:.2f} & {100*row['reference_overlap']['mean']:.1f} & {100*(1-f[0]):.1f} & {100*f[3]:.1f} & {row['proxy_target_misses']} / {row['reference_pairs']} \\\\")
    quality=default['rendered_quality'];overlap=default['reference_overlap'];fractions=default['other_camera_count_fractions_valid']
    tex=rf'''The current generator is audited on 2,048 consecutive layout seeds (24000--26047) at furnishing density 0.65, static human density 0.25, four cameras and default baseline $b=0.5$. All layouts pass geometry, swept collision and sampled camera constraints. The population contains {len(metrics['numeric'])} numeric series, {sum(v['standard_deviation']>1e-12 for v in metrics['numeric'].values())} with nonzero observed variance. These are correlated measurements.

The rendered population contains 512 distinct consecutive rooms (24000--24511), four $320\times240$ cameras and times $0,0.5,1$. The first 128 rooms are additionally captured at $b=0,0.25,0.75,1$, yielding 1,024 room/configuration captures and 12,288 views. Every row in the parameter sweep uses the same 128 rooms, with identical geometry, people, materials and lighting, verified by manifest hashes. No failed, dark, or low-overlap sample is replaced. Native Auto lighting and shadows are enabled; baked diffuse GI is disabled for the sweep. The selected gallery uses a separate full-lighting recipe. Captures use the checkout-local wgpu command-cache/upload optimizations; published crates resolve registry wgpu. Timing observations do not qualify unpatched package performance.

\begin{{table}}[ht]\centering\small
\begin{{tabular}}{{rrrrrr}}\toprule
Baseline & Distance (m) & Overlap (\%) & Shared (\%) & All three (\%) & Proxy misses\\\midrule
'''+ '\n'.join(table)+rf'''
\bottomrule\end{{tabular}}
\caption{{Absolute measurements at five settings of the current camera program. Distance is the median room-mean reference distance, measured at 33 times. Overlap averages bidirectional reference pairs at three rendered times. Shared and all-three fractions use valid source-pixel observations. Each row contains 128 rooms, 1,536 views and 1,152 reference pairs; its proxy threshold depends on baseline.}}
\end{{table}}
\begin{{figure}}[ht]\centering\includegraphics[width=\linewidth]{{generated/baseline_sweep.png}}
\caption{{Realized spacing, room-level overlap quantiles and co-visibility cardinality versus the camera-baseline parameter. Pixel observations and views from the same room are correlated.}}\end{{figure}}
\begin{{figure}}[ht]\centering\includegraphics[width=\linewidth]{{generated/baseline_paths.png}}
\caption{{Current camera programs for one fixed room at three baseline settings, using common metric axes.}}\end{{figure}}

At default baseline, all 512 rooms contribute 6,144 views and 4,608 reference-pair/time observations. Mean reference overlap is {100*overlap['mean']:.1f}\%, with minimum {100*overlap['min']:.1f}\%; {default['reference_pairs_below_requested']} pairs miss the proxy target. {100*(1-fractions[0]):.1f}\% of valid source-pixel observations are visible to another camera, and {100*fractions[3]:.1f}\% to all three others. The denominator is {default['source_valid_pixels']:,} valid pixel observations, not unique 3D points. Membership excludes the source camera.

The independent diagnostic requires nearest-target-pixel depth agreement within $0.01\,\mathrm{{m}}+0.002z$. The production GPU annotation uses a separate symmetric tangent-plane predicate; its exact masks are exported with the gallery. Both use the first geometric surface, treating glass as annotation-opaque. Reflections and refractions are not counted as geometric correspondence.

There are {quality['rooms_with_a_view_having_at_most_two_semantic_classes']} rooms with a view containing at most two semantic classes, and {quality['occupied_rooms_without_person_pixels']} of {quality['rooms_with_primary_people']} occupied primary rooms have no person pixels in any captured view. Maximum per-view p99 depth/position disagreement is {quality['annotations']['depth_position_p99_metres']['max']:.7f}\,m. These are annotation and coverage checks, not photographic-realism metrics. Complete distributions, source identities, configurations and commands are in \texttt{{docs/camera\_baseline\_v21.md}}. Bounded capture processes do not establish unlimited-process memory stability.
'''
    (generated/'camera_evaluation.tex').write_text(tex)
    tr=''.join(f"<tr><td>{r['baseline']:.2f}</td><td>{r['reference_baseline_m']['p05_p50_p95'][1]:.2f} m</td><td>{100*r['reference_overlap']['mean']:.1f}%</td><td>{100*(1-r['other_camera_fractions'][0]):.1f}%</td><td>{100*r['other_camera_fractions'][3]:.1f}%</td></tr>" for r in result['levels'])
    html=f'''
    <div class="cohort-heading"><h3>Camera baseline: close rigs to wide views</h3><span>generator v21 · current measurements</span></div>
    <div class="metric-strip"><div><strong>512</strong><span>distinct rendered rooms</span><small>6,144 views at default baseline</small></div><div><strong>5 settings</strong><span>128 identical rooms at each</span><small>baseline 0, 0.25, 0.5, 0.75, 1</small></div><div><strong>12,288</strong><span>views across the full experiment</span><small>1,024 room/configuration captures</small></div></div>
    <p>The [0,1] Camera baseline slider coordinates reference distance, minimum pair separation, overlap, spread and path variation. Default: 0.5. Advanced edits switch to custom settings; Regenerate applies them. Wider spacing trades shared pixels for stronger viewpoint variation.</p>
    <figure class="chart"><img src="static/media/baseline_sweep.png" loading="lazy" width="2240" height="592" alt="Current-generation spacing, overlap quantiles and co-visibility cardinality across five camera baseline settings"><figcaption>Same 128 room seeds, geometry and lighting at each value. Bands show room-level 5th–95th percentiles. Co-visibility fractions count valid source-pixel observations, not unique 3D points.</figcaption></figure>
    <div class="table-scroll"><table><caption>Matched 128-room camera-baseline sweep · 1,536 views per setting</caption><thead><tr><th>Baseline</th><th>Median room-mean distance</th><th>Mean reference overlap</th><th>Shared with any peer</th><th>Shared with all three</th></tr></thead><tbody>{tr}</tbody></table></div>
    <figure class="chart"><img src="static/media/baseline_paths.png" loading="lazy" width="1920" height="640" alt="Paths in the same current-generation room at baseline 0, 0.5 and 1"><figcaption>Seed 24005 · common metric axes · circles mark starts and crosses mark ends. Camera travel is separate from spacing between views.</figcaption></figure>
    <p>Across all 512 rooms at default baseline, mean reference overlap is {100*overlap['mean']:.1f}% (minimum {100*overlap['min']:.1f}%). {default['reference_pairs_below_requested']:,}/{default['reference_pairs']:,} observations miss the proxy target and remain reported. {100*(1-fractions[0]):.1f}% of valid source pixels are shared with another camera.</p>
    <p class="note">These are independent rendered-depth diagnostics; the gallery exports exact production GPU masks using a different surface test. Sweep: four 320 × 240 cameras, three times, native shadows, baked GI disabled. The gallery uses full lighting. No appearance filtering, photographic-realism claim or downstream training qualification.</p>
    <div class="evidence-downloads"><a href="static/media/baseline-v21.json" download>Baseline sweep (JSON) ↓</a><a href="static/media/covisibility-v21.json" download>512-room co-visibility (JSON) ↓</a><a href="static/media/population-v21.json" download>2,048-room distributions (JSON) ↓</a><a href="https://github.com/mosure/bevy_zeroverse/blob/main/docs/camera_baseline_v21.md">Protocol and limitations ↗</a></div>
'''
    page=ROOT/'www/project/index.html';text=page.read_text();a=text.index('<!-- CAMERA_EVALUATION_START -->');b=text.index('<!-- CAMERA_EVALUATION_END -->',a)
    page.write_text(text[:a]+'<!-- CAMERA_EVALUATION_START -->'+html+'    '+text[b:])
    rows = []
    for row in result['levels']:
        q = row['reference_baseline_m']['p05_p50_p95']
        f = row['other_camera_fractions']
        rows.append(f"| {row['baseline']:.2f} | {q[1]:.2f} ({q[0]:.2f}–{q[2]:.2f}) | "
                    f"{100*row['reference_overlap']['mean']:.2f}% | {100*(1-f[0]):.2f}% | "
                    f"{100*f[3]:.2f}% | {row['proxy_target_misses']}/{row['reference_pairs']} "
                    f"at {100*row['policy']['min_overlap']:.1f}% |")
    md = '\n\n| Baseline | Median room-mean reference distance, m (5th–95th percentile) | Mean reference overlap | Shared with any peer | Shared with all three | Proxy target misses |\n'
    md += '| --- | ---: | ---: | ---: | ---: | ---: |\n'+'\n'.join(rows)+'\n\n'
    md += (f"At the default baseline across **all 512 rooms / 6,144 views**, mean reference overlap is "
           f"**{100*overlap['mean']:.2f}%** (minimum **{100*overlap['min']:.2f}%**). "
           f"**{default['reference_pairs_below_requested']}/{default['reference_pairs']}** pair/time "
           f"observations miss the proxy target. **{100*(1-fractions[0]):.2f}%** of valid source pixels "
           f"are shared with another camera and **{100*fractions[3]:.2f}%** with all three. "
           f"The denominator is **{default['source_valid_pixels']:,}** valid pixel observations.\n\n"
           f"There are **{default['disconnected_sets_at_10_percent']}** disconnected camera sets "
           f"at a 10% edge-overlap threshold, **{quality['rooms_with_a_view_having_at_most_two_semantic_classes']}** "
           f"rooms with a view containing at most two semantic classes, and "
           f"**{quality['occupied_rooms_without_person_pixels']}/{quality['rooms_with_primary_people']}** "
           f"occupied rooms without person pixels across their views. Maximum per-view p99 "
           f"depth/position error is **{quality['annotations']['depth_position_p99_metres']['max']:.8f} m**.\n\n"
           f"The 2,048-room audit has **0 invalid seeds**, **{len(metrics['numeric'])} numeric series**, "
           f"and **{sum(v['standard_deviation']>1e-12 for v in metrics['numeric'].values())}** with "
           f"nonzero observed variance. The occupancy-signature estimate is "
           f"**{metrics['occupancy_signature_estimate']:.1f}** for 2,048 rooms; it is an approximate "
           "HyperLogLog statistic, not an exact unique count. These measurements are correlated.\n\n")
    md += '| Baseline | Median room-mean camera travel (m) | Capture/export seconds per room: median / 95th percentile |\n| --- | ---: | ---: |\n'
    for row in result['levels']:
        q = row['capture_export_seconds']['p05_p50_p95']
        md += f"| {row['baseline']:.2f} | {row['room_mean_path_length_m']['p05_p50_p95'][1]:.2f} | {q[1]:.2f} / {q[2]:.2f} |\n"
    md += ('\nTiming uses the matched 128-room subset at each setting and the declared capture protocol. '
           'Travel bounds are separate controls, but joint collision/overlap rejection correlates the realized distributions. '
           'Wide spacing can favor shorter accepted routes; request a larger minimum travel when needed.\n\n')
    doc = ROOT/'docs/camera_baseline_v21.md'
    content = doc.read_text(); a = content.index('<!-- RESULTS_START -->'); b = content.index('<!-- RESULTS_END -->',a)
    doc.write_text(content[:a]+'<!-- RESULTS_START -->'+md+content[b:])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'out/baseline_v21/captures')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/evidence/baseline_v21')
    parser.add_argument('--analyze',action='store_true',help='Recompute depth reports from raw captures')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    result,default,metrics,paths=collect(args.root,args.output,args.analyze)
    figures(args.output,result,paths);publish(args.output,result,default,metrics)
    print(json.dumps({k:result[k] for k in ['generator_version','distinct_rendered_rooms','rendered_configurations','rendered_views']}))


if __name__=='__main__':
    main()
