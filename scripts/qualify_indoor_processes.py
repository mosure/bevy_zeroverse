#!/usr/bin/env python3
"""Qualify finite indoor CLI worker recycling on Linux/NVIDIA.

Records sampled process-tree RSS and per-PID NVML residency, then validates raw
safetensor headers and small manifest tensors without loading image tensors.
All artifacts are retained, including on failure. Run --self-test without a GPU.
"""
import argparse
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import sys
import time
import uuid

PAGE_SIZE = os.sysconf('SC_PAGE_SIZE')
CLOCK_TICKS = os.sysconf('SC_CLK_TCK')
DTYPE_BYTES = {'BOOL': 1, 'U8': 1, 'I8': 1, 'U16': 2, 'I16': 2,
               'F16': 2, 'BF16': 2, 'U32': 4, 'I32': 4, 'F32': 4,
               'U64': 8, 'I64': 8, 'F64': 8}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def parse_stat(text):
    """The comm field may contain spaces and closing parentheses."""
    prefix, separator, tail = text.rpartition(') ')
    require(bool(separator), 'invalid /proc stat record')
    fields = tail.split()
    require(len(fields) >= 22, 'short /proc stat record')
    return {'pid': int(prefix.split(' ', 1)[0]), 'state': fields[0],
            'ppid': int(fields[1]), 'pgrp': int(fields[2]),
            'start_ticks': int(fields[19]),
            'rss_bytes': max(0, int(fields[21])) * PAGE_SIZE,
            'cpu_seconds': (int(fields[11]) + int(fields[12])) / CLOCK_TICKS}


def process_table():
    result = {}
    for entry in os.scandir('/proc'):
        if not entry.name.isdecimal():
            continue
        try:
            row = parse_stat(Path(entry.path, 'stat').read_text())
            result[row['pid']] = row
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return result


def identity(row):
    return f"{row['pid']}:{row['start_ticks']}"


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def gpu_processes(nvml, device):
    """An absent PID differs from a present PID whose memory is unavailable."""
    processes = {}
    for query in (nvml.nvmlDeviceGetGraphicsRunningProcesses,
                  nvml.nvmlDeviceGetComputeRunningProcesses):
        for item in query(device):
            memory = item.usedGpuMemory
            if memory is None or memory == (1 << 64) - 1:
                memory = None
            else:
                memory = int(memory)
                require(memory >= 0, 'negative NVML memory')
            old = processes.get(item.pid)
            processes[item.pid] = (max(old, memory) if old is not None and memory is not None
                                   else old if memory is None else memory)
    return processes


class ProcessTracker:
    def __init__(self, parent_pid):
        self.parent_pid = parent_pid
        self.processes = {}
        self.peak_active_children = 0
        self.tree_rss_peak = 0
        self.tree_gpu_known_peak = 0

    def observe(self, table, gpu, elapsed):
        # The new session also keeps workers discoverable if the parent exits.
        members = {pid: row for pid, row in table.items() if row['pgrp'] == self.parent_pid}
        active_children = 0
        rows = []
        for pid, current in members.items():
            key = identity(current)
            record = self.processes.setdefault(key, {
                'pid': pid, 'start_ticks': current['start_ticks'],
                'role': 'parent' if pid == self.parent_pid else 'worker',
                'first_seen_seconds': elapsed, 'last_seen_seconds': elapsed,
                'exit_observed_seconds': None, 'rss_observed_peak_bytes': 0,
                'cpu_seconds_last_observed': 0.0, 'gpu_observed_peak_bytes': None,
                'gpu_first_seen_seconds': None, 'gpu_release_observed_seconds': None})
            record['last_seen_seconds'] = elapsed
            record['rss_observed_peak_bytes'] = max(record['rss_observed_peak_bytes'], current['rss_bytes'])
            record['cpu_seconds_last_observed'] = max(record['cpu_seconds_last_observed'], current['cpu_seconds'])
            if pid in gpu:
                if record['gpu_first_seen_seconds'] is None:
                    record['gpu_first_seen_seconds'] = elapsed
                if gpu[pid] is not None:
                    record['gpu_observed_peak_bytes'] = max(record['gpu_observed_peak_bytes'] or 0, gpu[pid])
            alive = current['state'] != 'Z'
            active_children += int(alive and pid != self.parent_pid)
            rows.append({**current, 'identity': key, 'gpu_listed': pid in gpu,
                         'gpu_memory_bytes': gpu.get(pid)})
        for key, record in self.processes.items():
            current = table.get(record['pid'])
            gone = current is None or identity(current) != key or current['state'] == 'Z'
            if gone and record['exit_observed_seconds'] is None:
                record['exit_observed_seconds'] = elapsed
            if gone and record['pid'] not in gpu and record['gpu_release_observed_seconds'] is None:
                record['gpu_release_observed_seconds'] = elapsed
        self.peak_active_children = max(self.peak_active_children, active_children)
        self.tree_rss_peak = max(self.tree_rss_peak, sum(row['rss_bytes'] for row in rows))
        tracked_pids = {record['pid'] for record in self.processes.values()}
        known_gpu = sum(gpu.get(pid) or 0 for pid in tracked_pids)
        self.tree_gpu_known_peak = max(self.tree_gpu_known_peak, known_gpu)
        return {'elapsed_seconds': elapsed, 'active_children': active_children,
                'processes': rows, 'tree_known_gpu_memory_bytes': known_gpu,
                'tree_listed_gpu_memory_complete': all(gpu[pid] is not None for pid in tracked_pids if pid in gpu),
                'retired_gpu_processes': {str(pid): memory for pid, memory in gpu.items() if pid in tracked_pids and pid not in members},
                'other_gpu_processes': {str(pid): memory for pid, memory in gpu.items() if pid not in tracked_pids}}

    def released(self):
        return bool(self.processes) and all(
            record['exit_observed_seconds'] is not None
            and record['gpu_release_observed_seconds'] is not None
            for record in self.processes.values())

    def check_release_deadlines(self, elapsed, grace):
        for record in self.processes.values():
            exited = record['exit_observed_seconds']
            if exited is not None and record['gpu_release_observed_seconds'] is None:
                require(elapsed - exited <= grace,
                        f"terminated PID {record['pid']} retained GPU residency beyond {grace}s grace")


def terminate_group(process):
    """Terminate the private session; never signal an unrelated reused PID."""
    for sig, grace in [(signal.SIGTERM, 3.0), (signal.SIGKILL, 3.0)]:
        members = [row for row in process_table().values() if row['pgrp'] == process.pid and row['state'] != 'Z']
        if not members:
            break
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            process.poll()
            if not any(row['pgrp'] == process.pid and row['state'] != 'Z'
                       for row in process_table().values()):
                break
            time.sleep(.05)
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        raise RuntimeError('qualification could not terminate its process group')
    # Adopted grandchildren (Linux subreaper) may otherwise remain zombies.
    while True:
        try:
            pid, _ = os.waitpid(-process.pid, os.WNOHANG)
            if pid == 0:
                break
        except ChildProcessError:
            break
    require(not any(row['pgrp'] == process.pid and row['state'] != 'Z'
                    for row in process_table().values()), 'worker survived process teardown')


def tensor_header(path):
    size = path.stat().st_size
    with path.open('rb') as stream:
        raw = stream.read(8)
        require(len(raw) == 8, f'{path}: truncated safetensor header length')
        length = struct.unpack('<Q', raw)[0]
        require(2 <= length <= min(16 * 1024 * 1024, size - 8), f'{path}: invalid header length')
        header = json.loads(stream.read(length))
    ranges = []
    for name, tensor in header.items():
        if name == '__metadata__':
            continue
        shape, dtype, offsets = tensor['shape'], tensor['dtype'], tensor['data_offsets']
        require(dtype in DTYPE_BYTES and isinstance(shape, list)
                and all(type(n) is int and n >= 0 for n in shape), f'{path}: invalid tensor {name}')
        require(len(offsets) == 2 and all(type(n) is int and n >= 0 for n in offsets), f'{path}: invalid offsets')
        begin, end = offsets
        require(end - begin == math.prod(shape) * DTYPE_BYTES[dtype], f'{path}: tensor byte length mismatch: {name}')
        ranges.append((begin, end))
    cursor = 0
    for begin, end in sorted(ranges):
        require(begin == cursor, f'{path}: tensor payload gap/overlap')
        cursor = end
    require(8 + length + cursor == size, f'{path}: truncated or trailing tensor payload')
    return header, 8 + length


def small_tensor(path, header, offset, name):
    tensor = header[name]
    begin, end = tensor['data_offsets']
    require(tensor['dtype'] == 'U8' and len(tensor['shape']) == 1 and end - begin <= 4 * 1024 * 1024,
            f'{path}: invalid/oversized metadata tensor {name}')
    with path.open('rb') as stream:
        stream.seek(offset + begin)
        result = stream.read(end - begin)
    require(len(result) == end - begin, f'{path}: incomplete metadata tensor')
    return result


def verify_dataset(directory, args):
    config = json.loads((directory / 'generation_config.json').read_text())
    capture_engine = config.get('capture_engine')
    require(isinstance(capture_engine, str) and capture_engine, 'missing capture engine identity')
    for key, expected in [('base_seed', args.seed), ('width', args.width), ('height', args.height),
                          ('cameras', args.cameras), ('quality', args.quality), ('color_codec', 'raw'),
                          ('playback_steps', 1), ('generator_version', 4)]:
        require(config.get(key) == expected, f'generation contract mismatch: {key}')
    require(config.get('gi_effective_enabled') is (args.quality == 'auto'), 'GI enablement contract mismatch')
    require(config.get('gi_settings', {}).get('bake', {}).get('rays_per_probe') == 256,
            'GI ray budget contract mismatch')
    paths = sorted(directory.glob('*.safetensors'))
    require(bool(paths), 'no output chunks')
    require([path.name for path in paths] == [f'{i:06}.safetensors' for i in range(len(paths))],
            'missing, duplicated, or noncontiguous chunk indices')
    count, chunks = 0, []
    for path in paths:
        header, offset = tensor_header(path)
        precision = small_tensor(path, header, offset, 'annotation_precision')
        batch = len(precision)
        require(0 < batch <= args.chunk_size and precision == bytes([1]) * batch, 'invalid batch/native annotation precision')
        require(small_tensor(path, header, offset, 'color_encoding') == bytes([2]) * batch, 'wrong color encoding')
        for name, channels in [('color', 3), ('depth', 1), ('normal', 3), ('position', 3), ('semantic', 3)]:
            tensor = header.get(name, {})
            require(tensor.get('dtype') == 'F32' and tensor.get('shape') ==
                    [batch, 1, args.cameras, args.height, args.width, channels], f'{path}: missing/incorrect {name} plane')
        require({key for key in header if key.startswith('indoor_manifest_')} ==
                {f'indoor_manifest_{i}' for i in range(batch)}, f'{path}: manifest indices mismatch')
        for index in range(batch):
            manifest = json.loads(small_tensor(path, header, offset, f'indoor_manifest_{index}'))
            require(manifest['seed'] == (args.seed + count + index) % (1 << 64), f'{path}: sample seed/index mismatch')
            require(manifest['generator_version'] == 4 and len(manifest['cameras']) == args.cameras,
                    f'{path}: incorrect scene manifest')
            provenance = json.loads(small_tensor(path, header, offset, f'indoor_render_metadata_{index}'))
            require(provenance.get('capture_engine') == capture_engine, f'{path}: mixed capture engine identity')
            require(provenance['quality'].lower() == args.quality, f'{path}: quality provenance mismatch')
            require(provenance.get('diffuse_gi_supported') is (args.quality == 'auto'), f'{path}: GI support provenance mismatch')
            if args.quality == 'auto':
                settings = provenance.get('gi_settings') or {}
                stats = provenance.get('gi_statistics') or {}
                require(settings.get('enabled') is True and settings.get('gpu') is True
                        and settings.get('bake', {}).get('rays_per_probe') == 256,
                        f'{path}: production Auto256 GI settings missing')
                require(stats.get('backend') == 'gpu_bvh_compute' and stats.get('probes', 0) > 0
                        and stats.get('primary_rays') == stats['probes'] * 256,
                        f'{path}: production Auto256 GI statistics missing')
        chunks.append({'file': path.name, 'sample_offset': count, 'samples': batch,
                       'bytes': path.stat().st_size})
        count += batch
    require(count == args.samples, f'expected {args.samples} samples, found {count}')
    require(not list(directory.glob('*.tmp')), 'uncommitted temporary chunk files remain')
    return {'samples': count, 'chunks': chunks, 'capture_engine': capture_engine,
            'planes': ['color', 'depth', 'normal', 'position', 'semantic'],
            'scope': 'All tensor byte ranges/shapes and all small manifests/provenance checked; image tensor values were not loaded.'}


def verify_lifecycle(path, args, tracker, chunks):
    events = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    require(all(event['event'] in ('started', 'completed') for event in events), 'failed/cancelled worker lifecycle event')
    starts = [event for event in events if event['event'] == 'started']
    ends = [event for event in events if event['event'] == 'completed']
    require(starts and len(starts) == len(ends), 'incomplete worker lifecycle')
    require(len({event['run_id'] for event in events}) == 1, 'mixed lifecycle run IDs')
    require(isinstance(starts[0]['run_id'], str) and starts[0]['run_id'], 'empty lifecycle run ID')
    active, lifecycle_peak = {}, 0
    for event in events:
        require(event.get('schema_version') == 1, 'unsupported lifecycle schema')
        require(event['parent_pid'] == tracker.parent_pid and event['max_scenes_per_process'] == args.max_scenes_per_process,
                'lifecycle parent or budget mismatch')
        slot = event['worker_id']
        require(type(slot) is int and 0 <= slot < args.workers, 'invalid worker slot')
        if event['event'] == 'started':
            require(slot not in active, 'worker slot reused before preceding child completed')
            active[slot] = (event['job_id'], event['pid'])
        else:
            require(active.pop(slot, None) == (event['job_id'], event['pid']), 'mismatched lifecycle completion')
            require(event['success'] is True, 'unsuccessful lifecycle completion')
        lifecycle_peak = max(lifecycle_peak, len(active))
    require(not active and lifecycle_peak <= args.workers, 'incomplete lifecycle or worker cap exceeded')
    require(len({event['job_id'] for event in starts}) == len(starts), 'duplicate worker job ID')
    by_job = {event['job_id']: event for event in ends}
    worker_records = [row for row in tracker.processes.values() if row['role'] == 'worker']
    observed = {row['pid']: row for row in worker_records}
    require(len(observed) == len(worker_records), 'worker PID reuse prevents unambiguous lifecycle attribution')
    require(len(observed) == len(starts), 'some workers were missed by process sampling')
    cursor, chunk_cursor = 0, 0
    for event in sorted(starts, key=lambda row: row['sample_offset']):
        require(event['samples'] > 0 and (args.max_scenes_per_process == 0 or
                event['samples'] <= args.max_scenes_per_process), 'worker exceeded scene budget')
        require(event['sample_offset'] == cursor, 'overlapping or missing worker sample span')
        require(event['chunk_offset'] == chunk_cursor and event['chunks'] == math.ceil(event['samples'] / args.chunk_size),
                'overlapping or missing worker chunk span')
        for local, chunk in enumerate(chunks[chunk_cursor:chunk_cursor + event['chunks']]):
            require(chunk['sample_offset'] == cursor + local * args.chunk_size and
                    chunk['samples'] == min(args.chunk_size, event['samples'] - local * args.chunk_size),
                    'worker lifecycle does not match stored chunk/sample indices')
        chunk_cursor += event['chunks']
        cursor += event['samples']
        completed = by_job[event['job_id']]
        require(completed['pid'] == event['pid'] and completed['exit_code'] == 0, 'failed/mismatched worker completion')
        require(all(completed.get(key) == event.get(key) for key in
                    ('run_id', 'worker_id', 'job_id', 'sample_offset', 'samples', 'chunk_offset', 'chunks')),
                'worker completion span differs from start')
        record = observed.get(event['pid'])
        require(record is not None and record['exit_observed_seconds'] is not None, 'worker termination unobserved')
        require(record['gpu_first_seen_seconds'] is not None, 'worker GPU residency was never observed')
        require(record['gpu_observed_peak_bytes'] is not None, 'worker GPU memory was unavailable in every poll')
        require(record['gpu_release_observed_seconds'] is not None, 'terminated worker GPU residency did not disappear')
        record['lifecycle'] = {key: event[key] for key in
                               ('run_id', 'job_id', 'worker_id', 'sample_offset', 'samples', 'chunk_offset', 'chunks')}
        record['lifecycle']['started_unix_millis'] = event.get('unix_millis')
        record['lifecycle']['completed_unix_millis'] = completed.get('unix_millis')
        record['observed_lifetime_seconds'] = record['exit_observed_seconds'] - record.get('first_seen_seconds', 0)
    require(cursor == args.samples, 'worker lifecycle sample count mismatch')
    require(chunk_cursor == len(chunks), 'worker lifecycle chunk count mismatch')
    return {'run_id': starts[0]['run_id'], 'jobs': len(starts),
            'lifecycle_peak_active_workers': lifecycle_peak,
            'recycling_observed': len(starts) > args.workers,
            'all_worker_exits_and_gpu_release_observed': True}


def command_for(args, dataset):
    return [str(args.binary.resolve()), '--output', str(dataset.resolve()), '--scene-type', 'procedural-indoor',
            '--seed', str(args.seed), '--samples', str(args.samples), '--workers', str(args.workers),
            '--per-process=true', '--max-scenes-per-process', str(args.max_scenes_per_process),
            '--chunk-size', str(args.chunk_size), '--width', str(args.width), '--height', str(args.height),
            '--cameras', str(args.cameras), '--playback-steps', '1', '--render-modes',
            'color', 'depth', 'normal', 'semantic', 'position', '--indoor-human-density', '.5',
            '--indoor-quality', args.quality, '--indoor-gi-rays', '256', '--color-codec', 'raw',
            '--compression', 'none', '--ov-mode', 'disabled', '--no-ui', '--timeout-secs', '120']


def qualify(args):
    import pynvml as nvml

    require(sys.platform == 'linux', 'qualification requires Linux /proc')
    require(args.binary.is_file(), 'CLI binary is missing')
    args.output.mkdir(parents=True, exist_ok=False)
    # Reap worker grandchildren if an interrupted CLI parent cannot do so itself.
    require(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0, 'could not enable Linux child subreaper')
    command = command_for(args, args.output / 'dataset')
    run_id = str(uuid.uuid4())
    report = {'schema_version': 1, 'run_id': run_id, 'command': command, 'result': 'failed',
              'interval_seconds': args.interval, 'max_scenes_per_process': args.max_scenes_per_process,
              'gpu_release_grace_seconds': args.release_grace,
              'worker_limit': args.workers, 'quality': args.quality,
              'proc_clock_ticks_per_second': CLOCK_TICKS,
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
              'timing_policy': 'Memory and process-lifetime qualification; concurrent host activity is not controlled. Wall/CPU times are diagnostic, not an isolated throughput benchmark.',
              'policy': 'Peaks and lifetimes are observed /proc/NVML polls, not continuous maxima or exact exit times. Missing GPU memory remains unknown. Device activity from other processes is reported separately. PID identity includes /proc start_ticks.'}
    process, tracker, initialized = None, None, False
    started = time.monotonic()
    previous_handlers = {}
    try:
        nvml.nvmlInit()
        initialized = True
        device = nvml.nvmlDeviceGetHandleByIndex(args.device_index)
        report['adapter'] = nvml.nvmlDeviceGetName(device)
        report['adapter_uuid'] = nvml.nvmlDeviceGetUUID(device)
        report['binary_sha256'] = file_sha256(args.binary)
        report['script_sha256'] = file_sha256(Path(__file__))
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f'interrupted by signal {signum}')
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous_handlers[sig] = signal.signal(sig, interrupted)
        with (args.output / 'generation.log').open('x') as log, (args.output / 'telemetry.jsonl').open('x') as trace:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            report['command_started_unix_seconds'] = time.time()
            tracker = ProcessTracker(process.pid)
            report['parent_pid'] = process.pid
            exit_time = None
            while True:
                elapsed = time.monotonic() - started
                row = tracker.observe(process_table(), gpu_processes(nvml, device), elapsed)
                row.update(run_id=run_id, wall_unix_seconds=time.time())
                trace.write(json.dumps(row, allow_nan=False) + '\n')
                trace.flush()
                require(row['active_children'] <= args.workers, 'active worker cap exceeded')
                tracker.check_release_deadlines(elapsed, args.release_grace)
                require(elapsed <= args.timeout, 'qualification timeout')
                if process.poll() is not None:
                    if exit_time is None:
                        exit_time = time.monotonic()
                        report['exit_code'] = process.returncode
                        report['command_exit_observed_unix_seconds'] = time.time()
                    require(process.returncode == 0, f'CLI exited with {process.returncode}')
                    if tracker.released():
                        break
                    require(time.monotonic() - exit_time < args.release_grace, 'worker/process GPU release not observed within grace period')
                time.sleep(args.interval)
        report['dataset'] = verify_dataset(args.output / 'dataset', args)
        report['lifecycle'] = verify_lifecycle(args.output / 'dataset' / 'worker_lifecycle.jsonl', args, tracker, report['dataset']['chunks'])
        report['result'] = 'passed'
    except BaseException as error:
        report['error'] = f'{type(error).__name__}: {error}'
    finally:
        # A second interrupt must not abandon live GPU workers during teardown.
        for sig in previous_handlers:
            signal.signal(sig, signal.SIG_IGN)
        if process is not None:
            try:
                terminate_group(process)
            except Exception as error:
                report['result'] = 'failed'
                report['teardown_error'] = str(error)
        if initialized and tracker is not None and not tracker.released():
            try:
                deadline = time.monotonic() + args.release_grace
                with (args.output / 'telemetry.jsonl').open('a') as trace:
                    while True:
                        row = tracker.observe(process_table(), gpu_processes(nvml, device), time.monotonic() - started)
                        row.update(run_id=run_id, wall_unix_seconds=time.time(), phase='teardown')
                        trace.write(json.dumps(row, allow_nan=False) + '\n')
                        trace.flush()
                        if tracker.released():
                            break
                        require(time.monotonic() < deadline, 'GPU release unobserved after teardown')
                        time.sleep(args.interval)
            except Exception as error:
                report['result'] = 'failed'
                report['teardown_telemetry_error'] = str(error)
        if tracker is not None:
            report['processes'] = list(tracker.processes.values())
            report['peak_active_children'] = tracker.peak_active_children
            report['tree_rss_observed_peak_bytes'] = tracker.tree_rss_peak
            report['tree_gpu_known_memory_observed_peak_bytes'] = tracker.tree_gpu_known_peak
            parent = [row for row in tracker.processes.values() if row['role'] == 'parent']
            report['parent_rss_observed_peak_bytes'] = max((row['rss_observed_peak_bytes'] for row in parent), default=None)
        report['observed_wall_seconds'] = time.monotonic() - started
        for sig, handler in previous_handlers.items():
            signal.signal(sig, handler)
        if initialized:
            try:
                nvml.nvmlShutdown()
            except Exception as error:
                report['result'] = 'failed'
                report['nvml_shutdown_error'] = str(error)
        (args.output / 'report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('processes', 'dataset')}, indent=2))
    return 0 if report['result'] == 'passed' else 1


def self_test():
    import tempfile
    import types
    import unittest

    class Contracts(unittest.TestCase):
        def test_stat_parser_handles_parentheses_and_spaces(self):
            fields = ['S'] + ['0'] * 21
            for index, value in [(1, 11), (2, 22), (11, 30), (12, 20), (19, 12345), (21, 7)]:
                fields[index] = str(value)
            row = parse_stat('42 (worker ) name) ' + ' '.join(fields))
            self.assertEqual((row['pid'], row['ppid'], row['pgrp'], row['start_ticks']), (42, 11, 22, 12345))
            self.assertEqual(row['rss_bytes'], 7 * PAGE_SIZE)
            self.assertEqual(row['cpu_seconds'], 50 / CLOCK_TICKS)

        def test_process_identity_release_and_pid_reuse(self):
            tracker = ProcessTracker(10)
            parent = dict(pid=10, ppid=1, pgrp=10, state='S', start_ticks=11, rss_bytes=100, cpu_seconds=.1)
            child = dict(pid=20, ppid=10, pgrp=10, state='R', start_ticks=22, rss_bytes=200, cpu_seconds=.2)
            tracker.observe({10: parent, 20: child}, {20: 500}, 0)
            tracker.observe({10: parent}, {20: 500}, 1)
            record = tracker.processes['20:22']
            self.assertEqual(record['exit_observed_seconds'], 1)
            self.assertIsNone(record['gpu_release_observed_seconds'])
            tracker.check_release_deadlines(2, 1)
            with self.assertRaisesRegex(ValueError, 'retained GPU residency'):
                tracker.check_release_deadlines(3, 1)
            tracker.observe({10: parent, 20: {**child, 'start_ticks': 33}}, {}, 2)
            self.assertEqual(record['gpu_release_observed_seconds'], 2)
            self.assertIn('20:33', tracker.processes)
            self.assertEqual(tracker.peak_active_children, 1)

        def test_nvml_duplicate_pid_and_unavailable_memory(self):
            class Nvml:
                def nvmlDeviceGetGraphicsRunningProcesses(self, device):
                    return [types.SimpleNamespace(pid=1, usedGpuMemory=500), types.SimpleNamespace(pid=2, usedGpuMemory=(1 << 64)-1)]
                def nvmlDeviceGetComputeRunningProcesses(self, device):
                    return [types.SimpleNamespace(pid=1, usedGpuMemory=None)]
            self.assertEqual(gpu_processes(Nvml(), None), {1: 500, 2: None})

        def test_header_rejects_truncation_overlap_and_accepts_empty_tensors(self):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'test.safetensors'
                def write(header, data):
                    encoded = json.dumps(header).encode()
                    path.write_bytes(struct.pack('<Q', len(encoded)) + encoded + data)
                valid = {'a': {'dtype': 'U8', 'shape': [2], 'data_offsets': [0, 2]},
                         'empty': {'dtype': 'U8', 'shape': [0], 'data_offsets': [2, 2]}}
                write(valid, b'12')
                tensor_header(path)
                write(valid, b'1')
                with self.assertRaisesRegex(ValueError, 'truncated'):
                    tensor_header(path)
                write({**valid, 'b': {'dtype': 'U8', 'shape': [1], 'data_offsets': [1, 2]}}, b'12')
                with self.assertRaisesRegex(ValueError, 'overlap'):
                    tensor_header(path)

        def test_output_refuses_overwrite(self):
            with tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(FileExistsError):
                    Path(directory).mkdir(parents=True, exist_ok=False)

        def test_dataset_indices_and_manifests_are_checked_without_image_loading(self):
            with tempfile.TemporaryDirectory() as directory:
                directory = Path(directory)
                args = types.SimpleNamespace(seed=7, width=1, height=1, cameras=1, quality='portable', samples=2, chunk_size=2)
                config = dict(base_seed=7, width=1, height=1, cameras=1, quality='portable', color_codec='raw', playback_steps=1,
                              generator_version=4, capture_engine='test-engine', gi_effective_enabled=False,
                              gi_settings={'bake': {'rays_per_probe': 256}})
                (directory / 'generation_config.json').write_text(json.dumps(config))
                def write_chunk(seeds, auto_stats=None):
                    tensors = {'annotation_precision': ('U8', [2], b'\1\1'), 'color_encoding': ('U8', [2], b'\2\2')}
                    for name, channels in [('color', 3), ('depth', 1), ('normal', 3), ('position', 3), ('semantic', 3)]:
                        tensors[name] = ('F32', [2, 1, 1, 1, 1, channels], bytes(2 * channels * 4))
                    for index, seed in enumerate(seeds):
                        provenance = dict(quality=args.quality, diffuse_gi_supported=args.quality == 'auto',
                                          capture_engine='test-engine')
                        if args.quality == 'auto':
                            provenance.update(gi_settings=dict(enabled=True, gpu=True, bake={'rays_per_probe': 256}),
                                              gi_statistics=auto_stats)
                        for name, value in [('indoor_manifest', dict(seed=seed, generator_version=4, cameras=[{}])),
                                            ('indoor_render_metadata', provenance)]:
                            data = json.dumps(value).encode()
                            tensors[f'{name}_{index}'] = ('U8', [len(data)], data)
                    header, payload = {}, bytearray()
                    for name, (dtype, shape, data) in tensors.items():
                        start = len(payload)
                        payload.extend(data)
                        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[start, len(payload)])
                    encoded = json.dumps(header).encode()
                    (directory / '000000.safetensors').write_bytes(struct.pack('<Q', len(encoded)) + encoded + payload)
                write_chunk([7, 8])
                self.assertEqual(verify_dataset(directory, args)['samples'], 2)
                (directory / 'generation_config.json').write_text(json.dumps({**config, 'capture_engine': 'different-engine'}))
                with self.assertRaisesRegex(ValueError, 'mixed capture engine'):
                    verify_dataset(directory, args)
                (directory / 'generation_config.json').write_text(json.dumps(config))
                args.quality = 'auto'
                config.update(quality='auto', gi_effective_enabled=True)
                (directory / 'generation_config.json').write_text(json.dumps(config))
                write_chunk([7, 8], dict(backend='gpu_bvh_compute', probes=3, primary_rays=768))
                self.assertEqual(verify_dataset(directory, args)['samples'], 2)
                write_chunk([7, 8], dict(backend='gpu_bvh_compute', probes=3, primary_rays=0))
                with self.assertRaisesRegex(ValueError, 'GI statistics'):
                    verify_dataset(directory, args)
                args.quality = 'portable'
                config.update(quality='portable', gi_effective_enabled=False)
                (directory / 'generation_config.json').write_text(json.dumps(config))
                write_chunk([7, 7])
                with self.assertRaisesRegex(ValueError, 'sample seed/index mismatch'):
                    verify_dataset(directory, args)
                write_chunk([7, 8])
                (directory / '000000.safetensors').rename(directory / '000001.safetensors')
                with self.assertRaisesRegex(ValueError, 'chunk indices'):
                    verify_dataset(directory, args)

        def test_teardown_kills_and_reaps_worker_process_tree(self):
            self.assertEqual(ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0), 0)
            child = subprocess.Popen([sys.executable, '-c',
                'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); time.sleep(30)'],
                start_new_session=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            try:
                deadline = time.monotonic() + 3
                while not any(row['ppid'] == child.pid for row in process_table().values()):
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(.01)
                terminate_group(child)
                self.assertFalse(any(row['pgrp'] == child.pid for row in process_table().values()))
            finally:
                terminate_group(child)

        def test_lifecycle_caps_indices_and_gpu_release_are_enforced(self):
            args = types.SimpleNamespace(samples=2, workers=1, max_scenes_per_process=1, chunk_size=1)
            tracker = ProcessTracker(10)
            events, chunks = [], []
            for index in range(2):
                pid = 20 + index
                tracker.processes[f'{pid}:100'] = dict(pid=pid, role='worker', exit_observed_seconds=2,
                    gpu_first_seen_seconds=0, gpu_observed_peak_bytes=500, gpu_release_observed_seconds=3)
                shared = dict(schema_version=1, run_id='r', parent_pid=10, max_scenes_per_process=1, worker_id=0,
                    pid=pid, job_id=index, sample_offset=index, samples=1, chunk_offset=index, chunks=1)
                events.extend([{**shared, 'event':'started', 'exit_code':None, 'success':None},
                               {**shared, 'event':'completed', 'exit_code':0, 'success':True}])
                chunks.append(dict(sample_offset=index, samples=1))
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / 'worker_lifecycle.jsonl'
                def write(rows):
                    path.write_text(''.join(json.dumps(row)+'\n' for row in rows))
                write(events)
                self.assertTrue(verify_lifecycle(path, args, tracker, chunks)['recycling_observed'])
                write([events[0], {**events[1], 'samples': 2}, *events[2:]])
                with self.assertRaisesRegex(ValueError, 'completion span'):
                    verify_lifecycle(path, args, tracker, chunks)
                write([events[0], events[2], events[1], events[3]])
                with self.assertRaisesRegex(ValueError, 'slot reused'):
                    verify_lifecycle(path, args, tracker, chunks)
                write(events)
                tracker.processes['20:100']['gpu_release_observed_seconds'] = None
                with self.assertRaisesRegex(ValueError, 'did not disappear'):
                    verify_lifecycle(path, args, tracker, chunks)

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Contracts))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--binary', type=Path, default=Path('target/debug/zeroverse_gen'))
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--max-scenes-per-process', type=int, default=16)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--chunk-size', type=int, default=8)
    parser.add_argument('--width', type=int, default=161)
    parser.add_argument('--height', type=int, default=119)
    parser.add_argument('--cameras', type=int, default=3)
    parser.add_argument('--seed', type=int, default=23000)
    parser.add_argument('--quality', choices=['auto', 'portable'], default='portable')
    parser.add_argument('--interval', type=float, default=.1)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--release-grace', type=float, default=15)
    parser.add_argument('--device-index', type=int, default=0)
    args = parser.parse_args()
    if args.self_test:
        return self_test()
    if args.output is None:
        parser.error('--output is required')
    for name in ('samples', 'workers', 'chunk_size', 'width', 'height', 'cameras', 'interval', 'timeout', 'release_grace'):
        value = getattr(args, name)
        if not math.isfinite(value) or value <= 0:
            parser.error(f'{name} must be positive and finite')
    if args.max_scenes_per_process < 0 or args.device_index < 0 or not 0 <= args.seed < (1 << 64):
        parser.error('invalid scene budget, device index, or seed')
    return qualify(args)


if __name__ == '__main__':
    raise SystemExit(main())
