"""Bounded GPU qualification of indexed Python capture and the canonical CLI.

Run with the built extension and package on PYTHONPATH. Dependencies match the
dataloader. Artifacts are retained at --output; the destination must be new.
"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

import torch
from bevy_zeroverse_dataloader import BevyZeroverseDataset, FolderDataset, load_chunk
from torch.utils.data import DataLoader, Subset


def run_isolated(command, log, timeout):
    """On timeout/interruption, stop the CLI's descendants as well as its parent."""
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                          start_new_session=os.name == 'posix') as process:
        try:
            return process.wait(timeout=timeout)
        except BaseException:
            if os.name == 'posix':
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                process.kill()
            process.wait()
            raise


def read_samples(directory):
    result = []
    contract = json.loads((directory / 'generation_config.json').read_text())
    for path in sorted(directory.glob('*.safetensors')):
        batch = load_chunk(path, jpeg_device='cpu')
        for index, raw in enumerate(batch['indoor_manifest']):
            manifest = json.loads(bytes(raw.tolist()))
            assert batch['color_encoding'][index].item() == 2
            assert batch['annotation_precision'][index].item() == 1
            assert manifest['generator_version'] == 4
            provenance = json.loads(bytes(batch['indoor_render_metadata'][index].tolist()))
            assert provenance['capture_engine'] == contract['capture_engine']
            assert provenance['light_clustering'] == 'cpu_deterministic'
            assert provenance['pipeline_compilation'] == 'synchronous_on_render_thread'
            assert provenance['draw_submission'] == 'direct_gpu_preprocessing'
            assert provenance['quality'].lower() == contract['quality']
            assert provenance['gi_settings'] == contract['gi_settings']
            if contract['gi_effective_enabled']:
                assert provenance['gi_statistics'] is not None
            obb_ids = batch['object_obb_instance_ids'][index].tolist()
            assert [value for value in obb_ids if value >= 0] == list(range(len(manifest['objects']) + len(manifest['humans'])))
            people = len(manifest['humans'])
            if people:
                assert batch['human_pose_position'][index].shape[0] == 3
                assert tuple(batch['human_pose_position'][index].shape[2:]) == (21, 3)
                assert batch['human_count'][index].item() == people
                assert batch['human_instance_ids'][index, :people].tolist() == [human['id'] for human in manifest['humans']]
                assert torch.isfinite(batch['human_pose_position'][index]).all()
            assert tuple(batch['color'][index].shape) == (3, 2, 119, 161, 3)
            assert torch.allclose(batch['time'][index, :, 0, 0], torch.tensor([0., .5, 1.]))
            for name in ['depth', 'normal', 'position', 'semantic']:
                assert torch.isfinite(batch[name][index]).all(), name
            result.append((manifest, batch['world_from_view'][index], batch['fovy'][index], batch['semantic'][index], batch['color'][index]))
    return result


def verify_lifecycle(directory, expected_samples, expected_caps):
    records = [json.loads(line) for line in (directory / 'worker_lifecycle.jsonl').read_text().splitlines()]
    starts = [record for record in records if record['event'] == 'started']
    complete = {(record['run_id'], record['job_id']): record for record in records if record['event'] == 'completed'}
    assert len(starts) == len(complete)
    assert all(record['event'] in {'started', 'completed'} for record in records)
    assert {record['max_scenes_per_process'] for record in starts} == set(expected_caps)
    indices = []
    for record in starts:
        cap = record['max_scenes_per_process']
        assert cap == 0 or record['samples'] <= cap
        terminal = complete[(record['run_id'], record['job_id'])]
        assert terminal['pid'] == record['pid'] and terminal['success'] and terminal['exit_code'] == 0
        indices.extend(range(record['sample_offset'], record['sample_offset'] + record['samples']))
    assert sorted(indices) == list(range(expected_samples))
    return len(starts)


def python_capture(output, quality, gi_rays):
    dataset = BevyZeroverseDataset(False, True, 2, 161, 119, 8,
        scene_type='procedural_indoor', indoor_seed=900, indoor_layout='training',
        indoor_density=.9, indoor_human_density=.75, indoor_quality=quality, indoor_gi_rays=gi_rays, rotation_augmentation=True, playback_step=.5,
        playback_steps=3, render_modes=['color', 'depth', 'normal', 'semantic', 'position'],
        ovoxel_mode='disabled')
    captured = [dataset[index] for index in [4, 0, 4]]
    manifests = [json.loads(bytes(item['indoor_manifest'].tolist())) for item in captured]
    assert [item['seed'] for item in manifests] == [904, 900, 904]
    for item, manifest in zip(captured, manifests):
        assert item['annotation_precision'].item() == 1
        assert item['human_instance_ids'].tolist() == [person['id'] for person in manifest['humans']]
        assert item['human_count'].item() == len(manifest['humans'])
        assert item['object_obb_instance_ids'].tolist() == list(range(len(manifest['objects']) + len(manifest['humans'])))
        provenance = json.loads(bytes(item['indoor_render_metadata'].tolist()))
        assert provenance['light_clustering'] == 'cpu_deterministic'
        assert provenance['pipeline_compilation'] == 'synchronous_on_render_thread'
        assert provenance['draw_submission'] == 'direct_gpu_preprocessing'
        assert provenance['quality'].lower() == quality
        assert provenance['gi_settings']['bake']['rays_per_probe'] == gi_rays
    assert manifests[0] == manifests[2]
    assert torch.equal(captured[0]['world_from_view'], captured[2]['world_from_view'])
    assert torch.equal(captured[0]['fovy'], captured[2]['fovy'])
    assert torch.equal(captured[0]['semantic'], captured[2]['semantic'])
    parallel = list(DataLoader(Subset(dataset, [4, 0, 4]), batch_size=None, num_workers=2,
                               multiprocessing_context='spawn'))
    assert len(parallel) == len(captured)
    for expected, actual in zip(captured, parallel):
        assert torch.equal(expected['indoor_manifest'], actual['indoor_manifest'])
        assert torch.equal(expected['world_from_view'], actual['world_from_view'])
        assert torch.equal(expected['semantic'], actual['semantic'])
    report = {'indices': [4, 0, 4], 'seeds': [904, 900, 904],
              'repeated_manifest_exact': True, 'repeated_camera_exact': True,
              'repeated_semantic_exact': True,
              'python_precision_provenance_person_and_obb_ids': True,
              'two_spawn_workers_exact_manifests_cameras_semantics': True,
              'repeat_rgb_max_error': (captured[0]['color'] - captured[2]['color']).abs().max().item(),
              'shape': list(captured[0]['color'].shape)}
    output.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--binary', type=Path, default=Path('target/debug/zeroverse_gen'))
    parser.add_argument('--python-only', action='store_true')
    parser.add_argument('--quality', choices=['auto', 'portable'], default='auto')
    parser.add_argument('--gi-rays', type=int, default=256)
    parser.add_argument('--max-scenes-per-process', type=int, default=1)
    args = parser.parse_args()
    if args.max_scenes_per_process < 1:
        parser.error('qualification requires a positive child limit; cap0 is tested separately')
    if args.python_only:
        python_capture(args.output, args.quality, args.gi_rays)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    common = [str(args.binary.resolve()), '--scene-type', 'procedural-indoor',
              '--indoor-layout', 'training', '--indoor-density', '.9', '--indoor-human-density', '.75', '--rotation-augmentation',
              '--indoor-quality', args.quality, '--indoor-gi-rays', str(args.gi_rays), '--chunk-size', '2', '--width', '161', '--height', '119', '--cameras', '2',
              '--playback-steps', '3', '--playback-step', '.5', '--compression', 'none',
              '--render-modes', 'color', 'depth', 'normal', 'semantic', 'position',
              '--ov-mode', 'disabled', '--no-ui', '--timeout-secs', '120']
    def run(name, directory, arguments, success=True, cap=None):
        with (args.output / f'{name}.log').open('w') as log:
            limit = args.max_scenes_per_process if cap is None else cap
            returncode = run_isolated(common + ['--output', str(directory), '--max-scenes-per-process', str(limit)] + arguments,
                                      log, timeout=300)
        assert (returncode == 0) == success, f'{name} failed; inspect its log'
    invalid_direct = args.output / 'invalid_direct'
    run('reject_direct_process_limit', invalid_direct, ['--workers', '1', '--samples', '1', '--per-process=false'], success=False)
    assert not invalid_direct.exists(), 'invalid scheduling options wrote a dataset contract'
    one, two = args.output / 'one_worker', args.output / 'two_workers'
    run('one_worker', one, ['--workers', '1', '--samples', '3', '--seed', '777'])
    run('two_workers', two, ['--workers', '2', '--samples', '3', '--seed', '777'])
    first, second = read_samples(one), read_samples(two)
    initial_jobs = verify_lifecycle(one, 3, [args.max_scenes_per_process])
    verify_lifecycle(two, 3, [args.max_scenes_per_process])
    if args.max_scenes_per_process == 1:
        assert initial_jobs == 3
    assert [item[0]['seed'] for item in first] == [777, 778, 779]
    assert len(first) == len(second) == 3
    for a, b in zip(first, second):
        assert a[0] == b[0], 'worker count changed manifest'
        assert torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]), 'worker count changed cameras'
        assert torch.equal(a[3], b[3]), 'worker count changed labels'
        assert torch.equal(a[4], b[4]), 'worker count changed raw RGB'
    persistent = args.output / 'cap_zero'
    run('cap_zero', persistent, ['--workers', '1', '--samples', '3', '--seed', '777'], cap=0)
    baseline = read_samples(persistent)
    assert len(baseline) == len(first)
    assert verify_lifecycle(persistent, 3, [0]) == 1
    for actual, expected in zip(first, baseline):
        assert actual[0] == expected[0]
        assert all(torch.equal(a, b) for a, b in zip(actual[1:], expected[1:])), 'process replacement changed captured data'
    resume_cap = args.max_scenes_per_process + 1
    run('resume', one, ['--workers', '1', '--samples', '2', '--resume'], cap=resume_cap)
    total_jobs = verify_lifecycle(one, 5, [args.max_scenes_per_process, resume_cap])
    resumed = read_samples(one)
    assert [item[0]['seed'] for item in resumed] == list(range(777, 782))
    metrics = json.loads((one / 'metrics' / 'metrics.json').read_text())
    assert metrics['scenes'] == 5
    run('reject_changed_config', one, ['--workers', '1', '--samples', '1', '--resume', '--seed', '778'], success=False)
    run('reject_overwrite', one, ['--workers', '1', '--samples', '1', '--seed', '777'], success=False)
    contract_path = one / 'generation_config.json'
    original_contract = contract_path.read_text()
    incompatible_engine = json.loads(original_contract)
    incompatible_engine['capture_engine'] = 'older-incompatible-capture-engine'
    try:
        contract_path.write_text(json.dumps(incompatible_engine))
        run('reject_old_engine', one, ['--workers', '1', '--samples', '1', '--resume'], success=False)
    finally:
        contract_path.write_text(original_contract)
    folders = args.output / 'folders'
    run('folders', folders, ['--workers', '1', '--samples', '1', '--seed', '777', '--output-mode', 'fs', '--per-process=false'], cap=0)
    folder = FolderDataset(folders)[0]
    assert json.loads(bytes(folder['indoor_manifest'].tolist())) == first[0][0]
    assert torch.equal(folder['world_from_view'], first[0][1])
    assert torch.equal(folder['semantic'], first[0][3])
    assert torch.equal(folder['color'], first[0][4]), 'raw folder RGB differs from chunk RGB'
    with (args.output / 'python_indexed.log').open('w') as log:
        returncode = run_isolated([sys.executable, __file__, '--python-only', '--quality', args.quality, '--gi-rays', str(args.gi_rays), '--output', str(args.output / 'python_indexed.json')],
                                  log, timeout=300)
        assert returncode == 0, 'Python indexed capture failed; inspect its log'
    report = {'cli_seeds': [777, 778, 779], 'workers_compared': [1, 2],
              'generator_version': 4, 'human_density': .75, 'quality': args.quality, 'gi_rays': args.gi_rays, 'annotation_precision': 'float32_geometry',
              'capture_engine': json.loads(original_contract)['capture_engine'],
              'max_scenes_per_process': args.max_scenes_per_process, 'initial_child_jobs': initial_jobs,
              'total_child_jobs_after_resume': total_jobs, 'resume_child_cap': resume_cap,
              'cap_zero_exact_rgb_manifests_cameras_semantics': True, 'old_engine_resume_rejected': True,
              'explicit_positive_direct_process_limit_rejected': True,
              'color_codec': json.loads((one / 'generation_config.json').read_text())['color_codec'],
              'raw_rgb_exact_between_workers': True, 'raw_rgb_exact_folder_chunk': True,
              'render_provenance_matches_contract': True, 'all_obb_instance_ids_match_manifest': True,
              'light_clustering': 'cpu_deterministic',
              'pipeline_compilation': 'synchronous_on_render_thread',
              'exact_manifests_cameras_semantics': True, 'resume_seeds': list(range(777, 782)),
              'metrics_scenes': 5, 'changed_resume_rejected': True, 'overwrite_rejected': True,
              'rust_folder_python_manifest_camera_semantic_exact': True,
              'cameras': 2, 'timesteps': [0, .5, 1], 'modes': ['color', 'depth', 'normal', 'semantic', 'position'],
              'resolution': [161, 119], 'python_indexed': json.loads((args.output / 'python_indexed.json').read_text())}
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
