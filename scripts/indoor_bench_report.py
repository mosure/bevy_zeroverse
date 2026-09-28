#!/usr/bin/env python3
"""Validate and summarize complete capture runs, with explicitly scoped GPU evidence."""
import argparse
import json
import math
import pathlib
import statistics

NVML_SOURCE = 'https://docs.nvidia.com/deploy/nvml-api/api/group__nvmlDeviceQueries.html'


def number(value, name, minimum=0, positive=False, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if integer and not isinstance(value, int):
        raise ValueError(f'{name} must be an integer')
    if value < minimum or (positive and value == minimum):
        raise ValueError(f'{name} is outside its valid bounds')
    return value


def close(actual, expected, name):
    number(actual, name)
    if not math.isclose(actual, expected, rel_tol=1e-7, abs_tol=1e-8):
        raise ValueError(f'{name} does not match completed scene records: {actual} != {expected}')


def read_json(text):
    def invalid(value):
        raise ValueError(f'nonfinite JSON value: {value}')
    return json.loads(text, parse_constant=invalid)


def read_rows(path):
    return [read_json(line) for line in path.read_text().splitlines()]


def slope(values):
    if len(values) < 2:
        return None  # One point cannot establish a memory plateau.
    mx = (len(values) - 1) / 2
    my = statistics.mean(values)
    return sum((i-mx)*(v-my) for i,v in enumerate(values))/sum((i-mx)**2 for i in range(len(values)))


def validate_benchmark(summary, rows):
    version = number(summary['schema_version'], 'schema_version', integer=True)
    if version not in (1, 2, 3):
        raise ValueError(f'unsupported benchmark schema: {version}')
    count = number(summary['scenes'], 'scenes', positive=True, integer=True)
    warmup = number(summary['warmup_scenes'], 'warmup_scenes', integer=True)
    if warmup >= count or len(rows) != count:
        raise ValueError('incomplete benchmark or no measured scenes after warmup')
    pid = number(summary['pid'], 'pid', positive=True, integer=True)
    config = summary['config']
    views = number(config['num_cameras'], 'num_cameras', positive=True, integer=True) * number(config['playback_steps'], 'playback_steps', positive=True, integer=True)
    if version >= 2 and (not isinstance(summary.get('run_id'), str) or not summary['run_id']):
        raise ValueError('v2 benchmark requires a nonempty run_id')
    absolute_time = 'loop_started_unix_seconds' in summary
    previous_wall = 0
    previous_unix = number(summary['loop_started_unix_seconds'], 'loop_started_unix_seconds', positive=True) if absolute_time else None
    for index, row in enumerate(rows):
        if number(row['index'], 'index', integer=True) != index:
            raise ValueError('incomplete or unordered benchmark')
        elapsed = number(row['elapsed_seconds'], 'elapsed_seconds', positive=True)
        preparation = number(row['request_seconds'] if version >= 3 else row['preparation_seconds'], 'request_seconds')
        capture = number(row['capture_seconds'], 'capture_seconds')
        close(preparation + capture, elapsed, 'preparation plus capture seconds')
        if version >= 3 and row.get('preparation_stages') is not None:
            for field, value in row['preparation_stages'].items():
                number(value, field)
        for field in ('rss_bytes', 'staging_bytes', 'views'):
            number(row[field], field, integer=True)
        if row['views'] != views:
            raise ValueError('completed view count differs from requested cameras and steps')
        if row['annotation_precision'] not in ('float16_hdr', 'float32_geometry'):
            raise ValueError('unknown annotation precision')
        if row.get('heap') is not None:
            for field in ('live_heap_bytes', 'mapped_bytes'):
                number(row['heap'][field], field, integer=True)
        if version >= 2:
            if row.get('run_id') != summary['run_id'] or row.get('pid') != pid:
                raise ValueError('scene run_id/PID differs from completed benchmark')
            wall = number(row['wall_elapsed_seconds'], 'wall_elapsed_seconds', positive=True)
            if wall <= previous_wall or wall - previous_wall + 1e-6 < elapsed:
                raise ValueError('invalid or nonmonotonic scene wall elapsed time')
            previous_wall = wall
        if absolute_time:
            completed = number(row['completed_unix_seconds'], 'completed_unix_seconds', positive=True)
            if completed <= previous_unix:
                raise ValueError('nonmonotonic scene completion timestamp')
            previous_unix = completed
        elif 'completed_unix_seconds' in row:
            raise ValueError('scene absolute timestamps require loop_started_unix_seconds')
    measured = rows[warmup:]
    seconds = sum(row['elapsed_seconds'] for row in measured)
    close(summary['measured_capture_seconds'], seconds, 'measured_capture_seconds')
    number(summary['measured_completed_views'], 'measured_completed_views', integer=True)
    close(summary['measured_completed_views'], len(measured) * views, 'measured_completed_views')
    close(summary['scenes_per_second'], len(measured) / seconds, 'scenes_per_second')
    close(summary['views_per_second'], len(measured) * views / seconds, 'views_per_second')
    wall = number(summary['total_wall_seconds'], 'total_wall_seconds', positive=True)
    if wall + 1e-6 < max(sum(row['elapsed_seconds'] for row in rows), previous_wall):
        raise ValueError('total wall time is shorter than completed capture work')
    return measured


def summarize_spans(rows):
    gpu = {}
    for row in rows:
        for name, stat in row.get('render_diagnostics', {}).items():
            if any(s in name for s in ('indoor_diffuse_bake', 'indoor_ground_truth', 'dataset_readback_copy')) and name.endswith('/elapsed_gpu'):
                count = number(stat['count'], 'diagnostic count', integer=True)
                total = number(stat['sum'], 'diagnostic sum')
                if stat['unit'] != 'ms':
                    raise ValueError('GPU elapsed diagnostic must use milliseconds')
                if count:
                    close(stat['mean'], total / count, 'diagnostic mean')
                elif total != 0 or stat['mean'] is not None:
                    raise ValueError('empty GPU diagnostic has nonempty aggregate')
                values = gpu.setdefault(name, {'samples': 0, 'sum_ms': 0.0, 'record_batches': 0, 'max_reported_batch_mean_ms': None})
                values['samples'] += count
                values['sum_ms'] += total
                if count:
                    values['record_batches'] += 1
                    values['max_reported_batch_mean_ms'] = max(values['max_reported_batch_mean_ms'] or 0, stat['mean'])
    for value in gpu.values():
        value['mean_ms'] = value['sum_ms'] / value['samples'] if value['samples'] else None
    return gpu


def telemetry_statistics(polls, samples):
    memory = [r['process_gpu_memory_bytes'] for r in polls if r['process_gpu_memory_bytes'] is not None]
    tail = polls[-50:]
    tail_memory = [r['process_gpu_memory_bytes'] for r in tail if r['process_gpu_memory_bytes'] is not None]
    times = sorted(s['timestamp_us'] for s in samples)
    gaps = [(b - a) / 1e6 for a, b in zip(times, times[1:])]
    device_activity = [r['device_gpu_utilization_percent'] for r in polls if 'device_gpu_utilization_percent' in r]
    return {
        'polls': len(polls), 'known_process_memory_polls': len(memory),
        'device_gpu_utilization_poll_mean_percent': statistics.mean(device_activity) if device_activity else None,
        'other_observed_gpu_pids': sorted({int(pid) for r in polls for pid in r.get('other_processes', {})}),
        'process_gpu_memory_observed_peak_bytes': max(memory) if memory else None,
        'tail_polls': len(tail), 'tail_known_process_memory_polls': len(tail_memory),
        'process_gpu_memory_tail_min_max_bytes': [min(tail_memory), max(tail_memory)] if tail_memory else None,
        'process_utilization_reported_samples': len(samples),
        'process_sm_utilization_reported_sample_mean_percent': statistics.mean(s['sm'] for s in samples) if samples else None,
        'process_memory_utilization_reported_sample_mean_percent': statistics.mean(s['memory'] for s in samples) if samples else None,
        'process_utilization_reported_timestamp_span_seconds': (times[-1] - times[0]) / 1e6 if len(times) > 1 else None,
        'process_utilization_timestamp_gap_seconds_min_median_max': [min(gaps), statistics.median(gaps), max(gaps)] if gaps else None,
    }


def child_cpu_statistics(summary):
    wall = summary.get('command_observed_wall_seconds')
    if wall is not None:
        number(wall, 'command observed wall seconds', positive=True)
    result = {'observed_wall_seconds': wall,
              'child_cpu_user_seconds': None, 'child_cpu_system_seconds': None,
              'child_cpu_total_seconds': None, 'child_cpu_core_equivalent': None,
              'child_cpu_source': None}
    cpu = summary.get('child_cpu_usage')
    if cpu is not None:
        if wall is None:
            raise ValueError('child CPU accounting requires observed monotonic wall duration')
        if cpu.get('source') != 'resource.getrusage(RUSAGE_CHILDREN)':
            raise ValueError('unknown child CPU accounting source')
        user = number(cpu['user_seconds'], 'child CPU user seconds')
        system = number(cpu['system_seconds'], 'child CPU system seconds')
        total = number(cpu['total_seconds'], 'child CPU total seconds')
        close(total, user + system, 'child CPU total seconds')
        result.update(child_cpu_user_seconds=user, child_cpu_system_seconds=system,
                      child_cpu_total_seconds=total, child_cpu_core_equivalent=total / wall,
                      child_cpu_source=cpu['source'])
    return result


def summarize_telemetry(directory, benchmark, scenes):
    telemetry_path, summary_path = directory / 'telemetry.jsonl', directory / 'telemetry_summary.json'
    if not telemetry_path.exists() and not summary_path.exists():
        return {'available': False}
    if not telemetry_path.exists() or not summary_path.exists():
        raise ValueError('incomplete telemetry: records and completion summary are both required')
    summary = read_json(summary_path.read_text())
    rows = read_rows(telemetry_path)
    version = number(summary['schema_version'], 'telemetry schema_version', integer=True)
    if version not in (1, 2, 3):
        raise ValueError('unsupported telemetry schema')
    if not isinstance(summary['exit_code'], int) or isinstance(summary['exit_code'], bool) or summary['exit_code'] != 0:
        raise ValueError('telemetry command did not complete successfully')
    if number(summary['samples'], 'telemetry samples', integer=True) != len(rows):
        raise ValueError('incomplete telemetry records')
    pid = number(summary['pid'], 'telemetry pid', positive=True, integer=True)
    if pid != benchmark['pid']:
        raise ValueError('telemetry PID differs from benchmark PID')
    number(summary['interval_seconds'], 'telemetry interval', positive=True)
    command_window = None
    if version >= 2:
        if not isinstance(summary.get('telemetry_run_id'), str) or not summary['telemetry_run_id']:
            raise ValueError('v2 telemetry requires a nonempty telemetry_run_id')
        command_window = (number(summary['command_started_unix_seconds'], 'command start', positive=True),
                          number(summary['command_exit_observed_unix_seconds'], 'command exit observation', positive=True))
        if command_window[0] >= command_window[1]:
            raise ValueError('invalid command time bounds')
    unique, duplicate_count, previous_elapsed, previous_wall = {}, 0, -1, -1
    for row in rows:
        if row['pid'] != pid:
            raise ValueError('telemetry poll PID differs from benchmark PID')
        elapsed = number(row['elapsed_seconds'], 'telemetry elapsed_seconds')
        if elapsed <= previous_elapsed:
            raise ValueError('nonmonotonic telemetry polls')
        previous_elapsed = elapsed
        if version >= 2:
            if row.get('telemetry_run_id') != summary['telemetry_run_id']:
                raise ValueError('telemetry records belong to another run')
            wall = number(row['wall_unix_seconds'], 'telemetry wall timestamp', positive=True)
            if wall <= previous_wall or not command_window[0] <= wall <= command_window[1]:
                raise ValueError('telemetry poll is outside command bounds or not monotonic')
            previous_wall = wall
            status = row['process_utilization_status']
            if status not in ('available', 'no_samples', 'error'):
                raise ValueError('unknown process utilization availability')
            if status == 'error' and (row['process_utilization'] is not None or not row['process_utilization_error']):
                raise ValueError('invalid unavailable process-utilization poll')
            if status != 'error' and (row['process_utilization'] is None or row['process_utilization_error'] is not None):
                raise ValueError('invalid available process-utilization poll')
            if status == 'no_samples' and row['process_utilization']:
                raise ValueError('no-samples poll contains process utilization samples')
        if row['process_gpu_memory_bytes'] is not None:
            memory = number(row['process_gpu_memory_bytes'], 'process GPU memory', integer=True)
            if memory == (1 << 64) - 1:
                raise ValueError('NVML unavailable-memory sentinel was not normalized')
        for field in ('device_gpu_utilization_percent', 'device_memory_utilization_percent'):
            if number(row[field], field) > 100:
                raise ValueError('device utilization is outside 0..100')
        for sample in row.get('process_utilization') or []:
            if sample.get('pid', pid if version == 1 else None) != pid:
                raise ValueError('NVML sample PID differs from benchmark PID')
            timestamp = number(sample['timestamp_us'], 'NVML timestamp', positive=True, integer=True)
            for field in ('sm', 'memory'):
                if number(sample[field], f'process utilization {field}') > 100:
                    raise ValueError('process utilization is outside 0..100')
            normalized = {'pid': pid, 'timestamp_us': timestamp, 'sm': sample['sm'], 'memory': sample['memory']}
            if timestamp in unique:
                if unique[timestamp] != normalized:
                    raise ValueError('conflicting NVML samples at the same timestamp')
                duplicate_count += 1
            unique[timestamp] = normalized
    samples = sorted(unique.values(), key=lambda s: s['timestamp_us'])
    inside = samples if command_window is None else [s for s in samples if command_window[0] <= s['timestamp_us'] / 1e6 <= command_window[1]]
    result = {
        'available': True, 'schema_version': version, 'pid': pid,
        'identity_verification': 'matching_pid_and_telemetry_run_id' if version >= 2 else 'legacy_matching_poll_pid_only',
        'adapter': summary.get('adapter'), 'adapter_uuid': summary.get('adapter_uuid'),
        'interval_seconds': summary['interval_seconds'],
        'command_timestamp_bounds_available': command_window is not None,
        'process_utilization_unavailable_polls': sum(r.get('process_utilization') is None for r in rows),
        'process_utilization_errors': sorted({r['process_utilization_error'] for r in rows if r.get('process_utilization_error')}),
        'duplicate_nvml_samples_removed': duplicate_count,
        'nvml_samples_outside_command_bounds_excluded': len(samples) - len(inside) if command_window else None,
        'whole_command': {**telemetry_statistics(rows, inside), **child_cpu_statistics(summary)},
        'cpu_accounting_policy': 'Optional Unix RUSAGE_CHILDREN delta for the reaped command and OS-accounted waited descendants, excluding collector CPU. Whole-command CPU-seconds divided by monotonic observed command wall-seconds gives CPU-core equivalents and can exceed 1. Includes startup/warmup/cleanup and exit-observation polling delay; no measured-scene-window CPU attribution. Older runs and unsupported platforms remain null.',
        'measured_scene_window': None,
        'policy': 'Process utilization is an arithmetic mean of deduplicated matching-PID NVML reported samples; NVML may report only processes with nonzero activity. Missing samples remain unknown: no zero fill, interpolation, time weighting, or occupancy claim. GPU memory peaks are observed polling peaks. Whole-command scope includes startup, warmup, and cleanup. Timestamp-window filters select reported CPU timestamps; an NVML sample can include activity preceding a boundary. Device utilization is not attributed to the process.',
        'nvml_api_source': NVML_SOURCE,
    }
    if 'loop_started_unix_seconds' in benchmark:
        warmup = benchmark['warmup_scenes']
        start = scenes[warmup - 1]['completed_unix_seconds'] if warmup else benchmark['loop_started_unix_seconds']
        end = scenes[-1]['completed_unix_seconds']
        if command_window and (start < command_window[0] or end > command_window[1]):
            raise ValueError('benchmark measured window is outside telemetry command bounds')
        window_samples = [s for s in inside if start < s['timestamp_us'] / 1e6 <= end]
        polls = [r for r in rows if start < r.get('wall_unix_seconds', -1) <= end]
        result['measured_scene_window'] = {
            'start_unix_seconds': start, 'end_unix_seconds': end,
            'poll_timestamps_available': all('wall_unix_seconds' in r for r in rows),
            **telemetry_statistics(polls, window_samples)}
    return result


def summarize(directory):
    directory = pathlib.Path(directory)
    summary = read_json((directory / 'summary.json').read_text())
    rows = read_rows(directory / 'scenes.jsonl')
    measured = validate_benchmark(summary, rows)
    tail = measured[-250:]
    report = {'source': str(directory), 'summary': summary, 'tail_scenes': len(tail),
              'run_identity_verification': 'matching_row_run_id_and_pid' if summary['schema_version'] >= 2 else 'unverified_legacy_v1_rows',
              'tail_rss_slope_bytes_per_scene': slope([r['rss_bytes'] for r in tail]),
              'tail_rss_min_max_bytes': [min(r['rss_bytes'] for r in tail), max(r['rss_bytes'] for r in tail)],
              'request_seconds_mean': statistics.mean(r.get('request_seconds', r.get('preparation_seconds')) for r in measured),
              'capture_seconds_mean': statistics.mean(r['capture_seconds'] for r in measured),
              'whole_loop_scenes_per_second': len(rows) / summary['total_wall_seconds'],
              'staging_bytes_min_max': [min(r['staging_bytes'] for r in rows), max(r['staging_bytes'] for r in rows)],
              'annotation_precision': sorted({r['annotation_precision'] for r in rows})}
    if all(r.get('heap') is not None for r in rows):
        report['tail_live_heap_slope_bytes_per_scene'] = slope([r['heap']['live_heap_bytes'] + r['heap']['mapped_bytes'] for r in tail])
        report['tail_heap_last'] = tail[-1]['heap']
    stages = [r['preparation_stages'] for r in measured if r.get('preparation_stages') is not None]
    report['preparation_stage_seconds_mean'] = {key: statistics.mean(r[key] for r in stages) for key in stages[0]} if stages else None
    report['gpu_timestamp_spans'] = summarize_spans(measured)
    report['gpu_timestamp_policy'] = 'Per-invocation elapsed GPU timestamp spans from available measured-scene diagnostic batches. Asynchronous delivery can cross scene boundaries; these are not exact per-seed attributions. Nested spans are not summed, and elapsed spans are not GPU occupancy.'
    report['telemetry'] = summarize_telemetry(directory, summary, rows)
    return report, rows

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=pathlib.Path, nargs='+')
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = [summarize(run) for run in args.runs]
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'generation_performance.json').write_text(json.dumps([r[0] for r in result],indent=2,allow_nan=False)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2,2,figsize=(12,8))
    for (report,rows), run in zip(result,args.runs):
        x=[r['index'] for r in rows]
        axes[0,0].plot(x,[r['rss_bytes']/2**30 for r in rows],label=run.name)
        if all(r.get('heap') is not None for r in rows):
            axes[0,1].plot(x,[(r['heap']['live_heap_bytes']+r['heap']['mapped_bytes'])/2**30 for r in rows],label=run.name)
        # Disjoint groups retain complete scenes; the final group may be shorter.
        groups=[rows[i:i+25] for i in range(0,len(rows),25)]
        axes[1,0].plot([g[-1]['index'] for g in groups],[statistics.median(r['elapsed_seconds'] for r in g) for g in groups],label=run.name)
        axes[1,1].plot(x,[r['staging_bytes']/2**20 for r in rows],label=run.name)
    for ax,title,y in zip(axes.flat,['Process resident memory','Live glibc heap + mmap allocations','Median completed-scene time per 25 scenes','Bounded readback staging'],['GiB','GiB','Seconds','MiB']):
        ax.set(title=title,xlabel='Scene index',ylabel=y);ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('Actual indoor captures: memory and sustained generation',fontsize=15)
    fig.tight_layout()
    fig.savefig(args.output/'generation_performance.svg')
    fig.savefig(args.output/'generation_performance.png',dpi=140)


if __name__ == '__main__':
    main()
