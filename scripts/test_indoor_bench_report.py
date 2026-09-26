"""CPU-only regressions for truthful, fail-closed benchmark evidence."""
import json
import pathlib
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import indoor_bench_report as report
import indoor_telemetry as telemetry


def fixture(version=2):
    rows = [dict(index=i, run_id='42-test', pid=42,
                 elapsed_seconds=1., preparation_seconds=.2, capture_seconds=.8,
                 wall_elapsed_seconds=(i+1)*1.1, completed_unix_seconds=1000+(i+1)*1.1,
                 rss_bytes=100+i, staging_bytes=20, views=6,
                 annotation_precision='float32_geometry',
                 heap=dict(live_heap_bytes=40+i, mapped_bytes=20),
                 render_diagnostics={'render/indoor_diffuse_bake/elapsed_gpu':
                                     dict(count=2, sum=10., mean=5., unit='ms')}) for i in range(4)]
    summary = dict(schema_version=version, run_id='42-test', pid=42, scenes=4,
                   warmup_scenes=1, config=dict(num_cameras=3, playback_steps=2),
                   loop_started_unix_seconds=1000., total_wall_seconds=4.5,
                   measured_capture_seconds=3., measured_completed_views=18,
                   scenes_per_second=1., views_per_second=6.)
    if version == 1:
        summary.pop('loop_started_unix_seconds')
        summary.pop('run_id')
        for row in rows:
            for field in ('run_id', 'pid', 'completed_unix_seconds', 'wall_elapsed_seconds'):
                row.pop(field)
    return summary, rows


def nvml_sample(seconds, sm, pid=42):
    return dict(pid=pid, timestamp_us=int(seconds*1e6), sm=sm, memory=sm/2)


def telemetry_fixture():
    times = [1000.6, 1001.4, 1002.6, 1004.5]
    usage = [[nvml_sample(999., 100), nvml_sample(1000.5, 20)],
             [nvml_sample(1001.2, 40)], [nvml_sample(1001.2, 40)],
             [nvml_sample(1003.6, 80)]]
    rows = [dict(pid=42, telemetry_run_id='telemetry-test', elapsed_seconds=i+1.,
                 wall_unix_seconds=t, process_gpu_memory_bytes=50 if i < 3 else None,
                 device_gpu_utilization_percent=75, device_memory_utilization_percent=20,
                 process_utilization=u, process_utilization_status='available',
                 process_utilization_error=None) for i, (t, u) in enumerate(zip(times, usage))]
    summary = dict(schema_version=2, telemetry_run_id='telemetry-test', pid=42,
                   exit_code=0, samples=4, interval_seconds=.5,
                   command_started_unix_seconds=999.9, command_exit_observed_unix_seconds=1004.6)
    return summary, rows


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.temp.name)
        self.addCleanup(self.temp.cleanup)

    def write(self, summary=None, rows=None, gpu=False):
        if summary is None:
            summary, rows = fixture()
        (self.path/'summary.json').write_text(json.dumps(summary))
        (self.path/'scenes.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
        if gpu:
            self.write_telemetry(*telemetry_fixture())

    def write_telemetry(self, summary, rows):
        (self.path/'telemetry_summary.json').write_text(json.dumps(summary))
        (self.path/'telemetry.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))

    def test_verified_v2_and_legacy_identity(self):
        for version in (1, 2):
            self.write(*fixture(version))
            result, _ = report.summarize(self.path)
            self.assertEqual(result['run_identity_verification'], 'unverified_legacy_v1_rows' if version == 1 else 'matching_row_run_id_and_pid')
            self.assertEqual(result['tail_rss_slope_bytes_per_scene'], 1.)
            self.assertEqual(result['gpu_timestamp_spans']['render/indoor_diffuse_bake/elapsed_gpu']['samples'], 6)
            self.assertFalse(result['telemetry']['available'])

    def test_sparse_nvml_is_conditional_and_window_filtered(self):
        self.write(gpu=True)
        result, _ = report.summarize(self.path)
        gpu = result['telemetry']
        self.assertEqual(gpu['duplicate_nvml_samples_removed'], 1)
        self.assertEqual(gpu['nvml_samples_outside_command_bounds_excluded'], 1)
        self.assertAlmostEqual(gpu['whole_command']['process_sm_utilization_reported_sample_mean_percent'], 140/3)
        measured = gpu['measured_scene_window']
        self.assertEqual(measured['process_sm_utilization_reported_sample_mean_percent'], 60.)
        self.assertEqual(measured['process_utilization_reported_samples'], 2)
        self.assertEqual(measured['process_utilization_timestamp_gap_seconds_min_median_max'], [2.4]*3)
        self.assertEqual(measured['polls'], 2)
        self.assertNotIn('process_sm_utilization_mean_percent', gpu)

    def test_reject_mixed_run_identity_and_invalid_counts(self):
        modifications = [
            lambda s, r: r[2].update(run_id='other-run'),
            lambda s, r: r[2].update(pid=43),
            lambda s, r: s.pop('run_id'),
            lambda s, r: r.pop(),
            lambda s, r: r[2].update(index=1),
            lambda s, r: s.update(warmup_scenes=4),
            lambda s, r: r[2].update(views=5),
            lambda s, r: s.update(measured_completed_views=20),
            lambda s, r: s.update(scenes_per_second=2),
            lambda s, r: r[2].update(wall_elapsed_seconds=1),
            lambda s, r: r[2].update(completed_unix_seconds=999),
            lambda s, r: s.update(total_wall_seconds=3),
            lambda s, r: r[2].update(elapsed_seconds=0),
            lambda s, r: r[2].update(capture_seconds=-1),
            lambda s, r: r[2].update(rss_bytes=float('nan')),
            lambda s, r: r[2].update(staging_bytes=-1),
            lambda s, r: r[2].update(annotation_precision='unknown'),
        ]
        for mutate in modifications:
            with self.subTest(mutate=mutate):
                summary, rows = fixture()
                mutate(summary, rows)
                self.write(summary, rows)
                with self.assertRaises((ValueError, KeyError)):
                    report.summarize(self.path)

    def test_reject_incomplete_or_mismatched_telemetry(self):
        self.write(gpu=True)
        mutations = [
            lambda s, r: s.update(pid=43),
            lambda s, r: s.update(exit_code=1),
            lambda s, r: s.update(samples=5),
            lambda s, r: r[1].update(pid=43),
            lambda s, r: r[1].update(telemetry_run_id='stale'),
            lambda s, r: r[1]['process_utilization'][0].update(pid=43),
            lambda s, r: r[1]['process_utilization'][0].update(sm=101),
            lambda s, r: r[1]['process_utilization'][0].update(memory=float('inf')),
            lambda s, r: r[2]['process_utilization'][0].update(sm=50),
            lambda s, r: r[2].update(elapsed_seconds=1),
            lambda s, r: r[2].update(process_gpu_memory_bytes=(1 << 64)-1),
            lambda s, r: s.update(command_started_unix_seconds=1002),
            lambda s, r: r[2].update(process_utilization_status='error'),
        ]
        for mutate in mutations:
            with self.subTest(mutate=mutate):
                summary, rows = telemetry_fixture()
                mutate(summary, rows)
                self.write_telemetry(summary, rows)
                with self.assertRaises(ValueError):
                    report.summarize(self.path)
        (self.path/'telemetry_summary.json').unlink()
        with self.assertRaisesRegex(ValueError, 'incomplete telemetry'):
            report.summarize(self.path)

    def test_tail_does_not_reach_past_last_50_polls(self):
        polls = [dict(process_gpu_memory_bytes=100)] + [dict(process_gpu_memory_bytes=None) for _ in range(50)]
        result = report.telemetry_statistics(polls, [])
        self.assertEqual(result['process_gpu_memory_observed_peak_bytes'], 100)
        self.assertIsNone(result['process_gpu_memory_tail_min_max_bytes'])
        self.assertEqual(result['tail_known_process_memory_polls'], 0)
        self.assertIsNone(result['process_sm_utilization_reported_sample_mean_percent'])
        self.assertIsNone(report.slope([100]))

    def test_legacy_nvml_parent_pid_is_required(self):
        self.write(*fixture(1))
        summary, rows = telemetry_fixture()
        summary['schema_version'] = 1
        for row in rows:
            for sample in row['process_utilization']:
                sample.pop('pid')
        self.write_telemetry(summary, rows)
        result, _ = report.summarize(self.path)
        self.assertEqual(result['telemetry']['identity_verification'], 'legacy_matching_poll_pid_only')
        self.assertIsNone(result['telemetry']['measured_scene_window'])

    def test_cpu_core_equivalents_use_total_child_cpu_and_command_wall(self):
        self.write(gpu=True)
        summary, rows = telemetry_fixture()
        summary['command_observed_wall_seconds'] = 2.
        summary['child_cpu_usage'] = dict(source='resource.getrusage(RUSAGE_CHILDREN)',
                                          user_seconds=3., system_seconds=2., total_seconds=5.)
        self.write_telemetry(summary, rows)
        result, _ = report.summarize(self.path)
        cpu = result['telemetry']['whole_command']
        self.assertEqual(cpu['child_cpu_total_seconds'], 5.)
        self.assertEqual(cpu['child_cpu_core_equivalent'], 2.5)
        self.assertNotIn('child_cpu_total_seconds', result['telemetry']['measured_scene_window'])
        summary.pop('child_cpu_usage')
        self.assertIsNone(report.child_cpu_statistics(summary)['child_cpu_total_seconds'])
        self.assertIsNone(report.child_cpu_statistics({})['observed_wall_seconds'])

    def test_reject_invalid_cpu_accounting(self):
        cpu = dict(source='resource.getrusage(RUSAGE_CHILDREN)',
                   user_seconds=3., system_seconds=2., total_seconds=5.)
        for mutate in (lambda s: s.update(command_observed_wall_seconds=0),
                       lambda s: s.pop('command_observed_wall_seconds'),
                       lambda s: s['child_cpu_usage'].update(user_seconds=-1),
                       lambda s: s['child_cpu_usage'].update(total_seconds=6),
                       lambda s: s['child_cpu_usage'].update(total_seconds=float('nan')),
                       lambda s: s['child_cpu_usage'].update(source='unknown')):
            summary = dict(command_observed_wall_seconds=2., child_cpu_usage=dict(cpu))
            mutate(summary)
            with self.assertRaises(ValueError):
                report.child_cpu_statistics(summary)

    def test_bad_timestamp_span_rejected(self):
        for field, value in [('unit', 'seconds'), ('count', -1), ('mean', 6.)]:
            summary, rows = fixture()
            rows[1]['render_diagnostics']['render/indoor_diffuse_bake/elapsed_gpu'][field] = value
            self.write(summary, rows)
            with self.assertRaises(ValueError):
                report.summarize(self.path)


class FakeNvml:
    NVML_ERROR_NOT_FOUND = 6

    class NVMLError(Exception):
        def __init__(self, value):
            self.value = value
            super().__init__(str(value))

    def __init__(self):
        self.calls = []
        self.usage = [SimpleNamespace(pid=42, smUtil=50, memUtil=20, timeStamp=100),
                      SimpleNamespace(pid=99, smUtil=10, memUtil=2, timeStamp=110)]

    def nvmlDeviceGetGraphicsRunningProcesses(self, device):
        return [SimpleNamespace(pid=42, usedGpuMemory=100)]

    def nvmlDeviceGetComputeRunningProcesses(self, device):
        return [SimpleNamespace(pid=42, usedGpuMemory=(1 << 64)-1),
                SimpleNamespace(pid=99, usedGpuMemory=None)]

    def nvmlDeviceGetUtilizationRates(self, device):
        return SimpleNamespace(gpu=60, memory=30)

    def nvmlDeviceGetMemoryInfo(self, device):
        return SimpleNamespace(used=1000)

    def nvmlDeviceGetProcessUtilization(self, device, timestamp):
        self.calls.append(timestamp)
        if isinstance(self.usage, Exception):
            raise self.usage
        return self.usage


class CollectorTests(unittest.TestCase):
    def test_pid_cursor_unknown_memory_and_error_recovery(self):
        nvml = FakeNvml()
        sampler = telemetry.NvmlSampler(nvml, 'gpu', 0)
        first = sampler.measure(42)
        self.assertEqual(first['process_gpu_memory_bytes'], 100)
        self.assertIsNone(first['other_processes']['99'])
        self.assertEqual(first['process_utilization'][0]['pid'], 42)
        self.assertEqual(sampler.latest_timestamp, 110)
        nvml.usage = FakeNvml.NVMLError(6)
        missing = sampler.measure(42)
        self.assertEqual(missing['process_utilization_status'], 'no_samples')
        self.assertEqual(missing['process_utilization'], [])
        self.assertIsNone(missing['process_utilization_error'])
        nvml.usage = FakeNvml.NVMLError(3)
        failed = sampler.measure(42)
        self.assertEqual(failed['process_utilization_status'], 'error')
        self.assertIsNone(failed['process_utilization'])
        nvml.usage = []
        recovered = sampler.measure(42)
        self.assertIsNone(recovered['process_utilization_error'])
        self.assertEqual(recovered['process_utilization_status'], 'available')
        self.assertEqual(nvml.calls, [0, 110, 110, 110])
        totals = telemetry.Totals()
        for row in (first, missing, failed, recovered):
            totals.add(row)
        self.assertEqual(totals.samples, 4)
        self.assertEqual(totals.statuses['error'], 1)
        self.assertEqual(totals.peak_memory, 100)
        self.assertFalse(hasattr(totals, 'measurements'))

    def test_child_cpu_delta_subtracts_previous_children_and_is_optional(self):
        usage = iter([SimpleNamespace(ru_utime=101., ru_stime=202.),
                      SimpleNamespace(ru_utime=104., ru_stime=204.)])
        targets = []
        def getrusage(target):
            targets.append(target)
            return next(usage)
        fake = SimpleNamespace(RUSAGE_CHILDREN='children', getrusage=getrusage)
        with patch.object(telemetry, 'resource', fake):
            result = telemetry.child_cpu_delta(telemetry.child_cpu_snapshot(), telemetry.child_cpu_snapshot())
        self.assertEqual(targets, ['children', 'children'])
        self.assertEqual(result['user_seconds'], 3.)
        self.assertEqual(result['system_seconds'], 2.)
        self.assertEqual(result['total_seconds'], 5.)
        with patch.object(telemetry, 'resource', None):
            self.assertIsNone(telemetry.child_cpu_snapshot())
        self.assertIsNone(telemetry.child_cpu_delta(None, None))
        self.assertIsNone(telemetry.child_cpu_delta((1., 2.), None))
        with self.assertRaises(ValueError):
            telemetry.child_cpu_delta((3., 2.), (1., 4.))

    def test_real_cpu_command_with_mocked_nvml_completes_atomically(self):
        nvml = FakeNvml()
        lifecycle = []
        nvml.nvmlInit = lambda: lifecycle.append('init')
        nvml.nvmlShutdown = lambda: lifecycle.append('shutdown')
        nvml.nvmlDeviceGetHandleByIndex = lambda index: 'mock-device'
        nvml.nvmlDeviceGetName = lambda device: 'mock GPU (no hardware calls)'
        nvml.nvmlDeviceGetUUID = lambda device: 'GPU-mock'
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            argv = ['indoor_telemetry.py', '--output', directory, '--interval', '.005',
                    '--baseline-seconds', '0', '--', sys.executable, '-c',
                    'import time; time.sleep(.02)']
            with patch.dict(sys.modules, {'pynvml': nvml}), patch.object(sys, 'argv', argv):
                with self.assertRaises(SystemExit) as exit_result:
                    telemetry.main()
            self.assertEqual(exit_result.exception.code, 0)
            summary = json.loads((path/'telemetry_summary.json').read_text())
            rows = report.read_rows(path/'telemetry.jsonl')
            self.assertEqual(summary['schema_version'], 2)
            self.assertEqual(summary['samples'], len(rows))
            self.assertGreater(len(rows), 0)
            self.assertTrue(all(row['pid'] == summary['pid'] for row in rows))
            self.assertTrue(all(row['telemetry_run_id'] == summary['telemetry_run_id'] for row in rows))
            self.assertLess(summary['command_started_unix_seconds'], summary['command_exit_observed_unix_seconds'])
            self.assertGreater(summary['command_observed_wall_seconds'], 0)
            cpu = report.child_cpu_statistics(summary)
            if telemetry.resource is not None:
                self.assertGreater(cpu['child_cpu_total_seconds'], 0)
                self.assertAlmostEqual(cpu['child_cpu_core_equivalent'], cpu['child_cpu_total_seconds'] / summary['command_observed_wall_seconds'])
            self.assertFalse((path/'telemetry_summary.json.tmp').exists())
        self.assertEqual(lifecycle, ['init', 'shutdown'])

    def test_bounds_and_output_reuse(self):
        for interval, baseline in [(0, 1), (-1, 0), (float('nan'), 0), (float('inf'), 0), (.5, -1), (.5, float('nan'))]:
            with self.assertRaises(ValueError):
                telemetry.validate_options(interval, baseline)
        telemetry.validate_options(.5, 0)
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory)
            telemetry.prepare_output(path)
            for artifact in telemetry.ARTIFACTS:
                (path/artifact).touch()
                with self.assertRaises(ValueError):
                    telemetry.prepare_output(path)
                (path/artifact).unlink()


if __name__ == '__main__':
    unittest.main()
