#!/usr/bin/env python3
"""Collect bounded, PID-specific NVML telemetry around an actual command.

Requires nvidia-ml-py. Device activity includes other processes. NVML process
utilization is sparse: its reported samples are not a wall-time occupancy trace.
"""
import argparse
from collections import Counter, deque
import json
import math
import pathlib
import subprocess
import time
import uuid

try:
    import resource  # Unix only; no estimate is substituted on other platforms.
except ImportError:
    resource = None


ARTIFACTS = ('process.log', 'telemetry.jsonl', 'telemetry_summary.json',
             'telemetry_summary.json.tmp', 'baseline.jsonl', 'scenes.jsonl', 'summary.json')


def validate_options(interval, baseline_seconds):
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError('interval must be finite and positive')
    if not math.isfinite(baseline_seconds) or baseline_seconds < 0:
        raise ValueError('baseline seconds must be finite and nonnegative')


def prepare_output(directory):
    directory.mkdir(parents=True, exist_ok=True)
    existing = [name for name in ARTIFACTS if (directory / name).exists()]
    if existing:
        raise ValueError(f'refusing to overwrite existing run artifacts: {existing}')


def child_cpu_snapshot():
    if resource is None or not hasattr(resource, 'RUSAGE_CHILDREN'):
        return None
    usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    return (usage.ru_utime, usage.ru_stime)


def child_cpu_delta(before, after):
    if before is None or after is None:
        return None
    user, system = (end - start for start, end in zip(before, after))
    if any(not math.isfinite(value) or value < 0 for value in (user, system)):
        raise ValueError('invalid child CPU accounting delta')
    return {'source': 'resource.getrusage(RUSAGE_CHILDREN)',
            'user_seconds': user, 'system_seconds': system,
            'total_seconds': user + system}


def gpu_memory(value):
    """NVML uses UINT64_MAX for unavailable memory on some platforms."""
    return None if value is None or value == (1 << 64) - 1 else int(value)


class NvmlSampler:
    def __init__(self, nvml, device, started):
        self.nvml, self.device, self.started = nvml, device, started
        self.latest_timestamp = 0

    def measure(self, pid):
        nvml, device = self.nvml, self.device
        processes = {}
        for query in (nvml.nvmlDeviceGetGraphicsRunningProcesses,
                      nvml.nvmlDeviceGetComputeRunningProcesses):
            for item in query(device):
                memory = gpu_memory(item.usedGpuMemory)
                previous = processes.get(item.pid)
                # A PID may appear in both lists; never double-count or replace
                # known memory with an unavailable value from the other API.
                processes[item.pid] = max(previous, memory) if previous is not None and memory is not None else (previous if memory is None else memory)
        utilization = nvml.nvmlDeviceGetUtilizationRates(device)
        process_usage, status, error_text = None, 'available', None
        try:
            samples = nvml.nvmlDeviceGetProcessUtilization(device, self.latest_timestamp)
            if samples:
                self.latest_timestamp = max(self.latest_timestamp, max(s.timeStamp for s in samples))
            process_usage = [{'pid': s.pid, 'sm': s.smUtil, 'memory': s.memUtil,
                              'timestamp_us': s.timeStamp} for s in samples if s.pid == pid]
        except nvml.NVMLError as error:
            if getattr(error, 'value', None) == nvml.NVML_ERROR_NOT_FOUND:
                status, process_usage = 'no_samples', []
            else:
                status, error_text = 'error', str(error)
        return {'elapsed_seconds': time.monotonic() - self.started,
                'wall_unix_seconds': time.time(), 'pid': pid,
                'process_gpu_memory_bytes': processes.get(pid),
                'device_gpu_utilization_percent': utilization.gpu,
                'device_memory_utilization_percent': utilization.memory,
                'device_memory_bytes': nvml.nvmlDeviceGetMemoryInfo(device).used,
                'process_utilization': process_usage,
                'process_utilization_status': status,
                'process_utilization_error': error_text,
                'other_processes': {str(p): m for p, m in processes.items() if p != pid}}


class Totals:
    """Online statistics: collector memory does not grow with command duration."""
    def __init__(self):
        self.samples = 0
        self.known_memory_samples = 0
        self.peak_memory = self.last_known_memory = None
        self.device_utilization_sum = 0
        self.statuses = Counter()
        self.last_error = None

    def add(self, sample):
        self.samples += 1
        self.device_utilization_sum += sample['device_gpu_utilization_percent']
        self.statuses[sample['process_utilization_status']] += 1
        if sample['process_utilization_error']:
            self.last_error = sample['process_utilization_error']
        memory = sample['process_gpu_memory_bytes']
        if memory is not None:
            self.known_memory_samples += 1
            self.peak_memory = max(self.peak_memory or 0, memory)
            self.last_known_memory = memory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    parser.add_argument('--interval', type=float, default=0.5)
    parser.add_argument('--baseline-seconds', type=float, default=3.0)
    parser.add_argument('--device-index', type=int, default=0)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    try:
        validate_options(args.interval, args.baseline_seconds)
        if not command or args.device_index < 0:
            raise ValueError('a command and nonnegative device index are required')
        prepare_output(args.output)
    except ValueError as error:
        parser.error(str(error))

    import pynvml as nvml
    nvml.nvmlInit()
    try:
        device = nvml.nvmlDeviceGetHandleByIndex(args.device_index)
        started = time.monotonic()
        sampler = NvmlSampler(nvml, device, started)
        telemetry_run_id = str(uuid.uuid4())
        baseline, baseline_samples = deque(maxlen=32), 0
        with (args.output / 'baseline.jsonl').open('x') as stream:
            while time.monotonic() - started < args.baseline_seconds:
                sample = sampler.measure(None)
                baseline.append(sample)
                baseline_samples += 1
                stream.write(json.dumps(sample, allow_nan=False) + '\n')
                stream.flush()
                time.sleep(min(args.interval, max(0, args.baseline_seconds - (time.monotonic() - started))))
        totals = Totals()
        with (args.output / 'process.log').open('x') as log, (args.output / 'telemetry.jsonl').open('x') as stream:
            cpu_before = child_cpu_snapshot()
            command_start_monotonic = time.monotonic()
            command_start = time.time()
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    sample = sampler.measure(child.pid)
                    sample['telemetry_run_id'] = telemetry_run_id
                    totals.add(sample)
                    stream.write(json.dumps(sample, allow_nan=False) + '\n')
                    stream.flush()
                    time.sleep(args.interval)
                exit_code = child.wait()
                command_end = time.time()
                command_wall_seconds = time.monotonic() - command_start_monotonic
                child_cpu = child_cpu_delta(cpu_before, child_cpu_snapshot())
            except BaseException:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                raise
        summary = {
            'schema_version': 2, 'telemetry_run_id': telemetry_run_id,
            'command': command, 'pid': child.pid, 'exit_code': exit_code,
            'command_started_unix_seconds': command_start,
            'command_exit_observed_unix_seconds': command_end,
            'command_observed_wall_seconds': command_wall_seconds,
            'child_cpu_usage': child_cpu,
            'child_cpu_policy': 'Optional Unix RUSAGE_CHILDREN delta before Popen to after wait, excluding collector CPU. Includes this child and any waited descendants accounted by the OS. Whole-command scope includes startup/warmup/cleanup; monotonic wall duration ends when exit is observed, so it includes polling delay. Unavailable accounting remains null.',
            'adapter': nvml.nvmlDeviceGetName(device),
            'adapter_uuid': nvml.nvmlDeviceGetUUID(device), 'device_index': args.device_index,
            'interval_seconds': args.interval, 'baseline_seconds': args.baseline_seconds,
            'baseline': list(baseline), 'baseline_samples': baseline_samples,
            'baseline_policy': 'Last 32 polls in summary; complete baseline in baseline.jsonl.',
            'samples': totals.samples, 'known_process_memory_samples': totals.known_memory_samples,
            'process_gpu_memory_observed_peak_bytes': totals.peak_memory,
            'process_gpu_memory_last_known_bytes': totals.last_known_memory,
            'device_gpu_utilization_poll_mean_percent': totals.device_utilization_sum / totals.samples if totals.samples else None,
            'process_utilization_poll_status_counts': dict(totals.statuses),
            'process_utilization_last_error': totals.last_error,
            'utilization_policy': 'Whole command including startup/warmup/cleanup. Device activity includes other processes. Process utilization is sparse NVML reported samples, not wall-time utilization or occupancy. Memory peaks are observed polls; missing values are unknown. Command exit timestamp is polling observation, not exact process exit.'}
        temporary = args.output / 'telemetry_summary.json.tmp'
        with temporary.open('x') as stream:
            stream.write(json.dumps(summary, indent=2, allow_nan=False) + '\n')
        temporary.replace(args.output / 'telemetry_summary.json')
    finally:
        nvml.nvmlShutdown()
    raise SystemExit(exit_code)


if __name__ == '__main__':
    main()
