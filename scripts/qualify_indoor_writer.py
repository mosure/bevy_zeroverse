#!/usr/bin/env python3
"""Bounded real-CLI writer/RSS qualification; retains trace and all generated data."""
import argparse
import json
import statistics
import struct
import subprocess
import time
from pathlib import Path


def stored_samples(directory):
    count = 0
    for path in directory.glob('*.safetensors'):
        with path.open('rb') as stream:
            header_size = struct.unpack('<Q', stream.read(8))[0]
            header = json.loads(stream.read(header_size))
            count += header['annotation_precision']['shape'][0]
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, default=128)
    parser.add_argument('--binary', type=Path, default=Path('target/debug/zeroverse_gen'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    dataset = args.output / 'dataset'
    command = [str(args.binary.resolve()), '--output', str(dataset), '--scene-type', 'procedural-indoor',
               '--seed', '11000', '--samples', str(args.samples), '--workers', '1', '--per-process=false',
               '--chunk-size', '8', '--width', '161', '--height', '119', '--cameras', '3',
               '--playback-steps', '1', '--render-modes', 'color', 'depth', 'normal', 'semantic', 'position',
               '--indoor-human-density', '.5', '--indoor-quality', 'portable', '--color-codec', 'raw',
               '--compression', 'none', '--ov-mode', 'disabled', '--no-ui']
    started = time.monotonic()
    trace = []
    with (args.output / 'generation.log').open('w') as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                status = dict(line.split(':', 1) for line in Path(f'/proc/{process.pid}/status').read_text().splitlines())
                trace.append({'seconds': time.monotonic() - started, 'committed_samples': stored_samples(dataset),
                              'rss_mib': int(status['VmRSS'].split()[0]) / 1024,
                              'hwm_mib': int(status['VmHWM'].split()[0]) / 1024})
            except (FileNotFoundError, KeyError):
                pass
            time.sleep(.5)
    elapsed = time.monotonic() - started
    (args.output / 'rss_trace.json').write_text(json.dumps(trace, indent=2))
    report = {'command': command, 'exit_code': process.returncode, 'seconds': elapsed,
              'samples': stored_samples(dataset), 'samples_per_second_including_startup': stored_samples(dataset) / elapsed,
              'peak_rss_mib': max(row['hwm_mib'] for row in trace),
              'encoding_batches_in_flight_limit': 1, 'capturing_batches_in_flight_limit': 1,
              'quality': 'portable; no GI/shadow benchmark claim'}
    for label, low, high in [('warm', 16, args.samples // 2), ('tail', args.samples * 3 // 4, args.samples)]:
        values = [row['rss_mib'] for row in trace if low <= row['committed_samples'] < high]
        report[f'{label}_rss_median_mib'] = statistics.median(values) if values else None
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    assert process.returncode == 0 and report['samples'] == args.samples, report
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
