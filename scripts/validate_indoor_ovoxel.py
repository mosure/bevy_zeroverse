#!/usr/bin/env python3
"""Validate zeroverse_gen O-voxel chunks (uncompressed) or fs exports; compare CPU/GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import numpy as np


def tensors(path):
    raw = path.read_bytes()
    length = struct.unpack('<Q', raw[:8])[0]
    if length > len(raw)-8:
        raise ValueError('use --compression none for this raw safetensors audit')
    header = json.loads(raw[8:8+length])
    payload = memoryview(raw)[8+length:]
    types = {'U8':'u1', 'U16':'<u2', 'U32':'<u4', 'I64':'<i8', 'F32':'<f4'}
    result = {}
    for name, info in header.items():
        if name == '__metadata__':
            continue
        lo, hi = info['data_offsets']
        result[name] = np.frombuffer(payload[lo:hi], dtype=types[info['dtype']]).reshape(info['shape'])
    return result


def audit(root):
    rows, samples, hashes = [], [], {}
    paths = sorted(root.glob('*.safetensors')) + sorted(root.glob('[0-9]*/meta.safetensors'))
    for path in paths:
        hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = tensors(path)
        if path.name == 'meta.safetensors':
            # Normalize a filesystem sample to the chunk's leading batch axis.
            data['ovoxel_offsets'] = np.array([[0, len(data['ovoxel_coords'])]])
            data['ovoxel_semantic_label_offsets'] = np.array([[0, len(data['ovoxel_semantic_labels'])]])
            data['time'] = data['time'][None]
            data['aabb'] = data['aabb'][None]
            data['world_from_view'] = data['world_from_view'][None]
            metadata = path.parent/'indoor_render_metadata.json'
            provenance = path.parent/'render_metadata.json'
            for p in (metadata, provenance):
                hashes[str(p.relative_to(root))] = hashlib.sha256(p.read_bytes()).hexdigest()
            manifest = json.loads(provenance.read_text())[1]
            data['indoor_manifest_0'] = np.frombuffer(json.dumps(manifest).encode(), dtype='u1')
            data['indoor_render_metadata_0'] = np.frombuffer(metadata.read_bytes(), dtype='u1')
        for key, value in data.items():
            if value.dtype.kind == 'f' and not np.isfinite(value).all():
                raise ValueError(f'nonfinite tensor: {key}')
        offsets = data['ovoxel_offsets']
        label_offsets = data['ovoxel_semantic_label_offsets']
        assert offsets[0,0] == 0 and offsets[-1].sum() == len(data['ovoxel_coords'])
        assert (offsets[:,1] > 0).all() and np.array_equal(offsets[:-1].sum(axis=1), offsets[1:,0])
        for index, (lo, count) in enumerate(offsets):
            hi = lo + count
            manifest = json.loads(data[f'indoor_manifest_{index}'].tobytes())
            metadata = json.loads(data[f'indoor_render_metadata_{index}'].tobytes())
            ovmeta = metadata['ovoxel']
            coords = data['ovoxel_coords'][lo:hi].astype(np.int64)
            dual = data['ovoxel_dual_vertices'][lo:hi]
            flags = data['ovoxel_intersected'][lo:hi].flatten()
            colors = data['ovoxel_base_color'][lo:hi]
            semantic = data['ovoxel_semantic'][lo:hi].flatten()
            labels = json.loads(data['ovoxel_semantic_labels'].flatten()[label_offsets[index,0]:label_offsets[index].sum()].tobytes())
            bounds = data['ovoxel_aabb'][index]
            assert np.array_equal(data['aabb'][index], bounds), 'annotation AABB must equal the primary-room voxel AABB'
            resolution = int(data['ovoxel_resolution'].flatten()[index])
            assert data['time'].shape[1] == 1, 'O-voxel is single-timestep'
            assert data['time'][index].min() == data['time'][index].max() == 0
            assert metadata['human_motion'] is None or not metadata['human_motion']['accepted']
            assert len(coords) == len(dual) == len(flags) == len(colors) == len(semantic)
            assert labels[0] == 'unlabeled' and semantic.max() < len(labels)
            assert coords.min() >= 0 and coords.max() < resolution and flags.max() <= 7
            assert len(np.unique(coords, axis=0)) == len(coords)
            assert np.array_equal(np.lexsort(coords[:, ::-1].T), np.arange(len(coords)))
            # Check the scope independently of metadata: same largest program zone
            # used by reconstruction cameras, with enclosing structural slabs.
            # f32 areas and last maximum on ties match the generator's contract.
            zone = max(enumerate(manifest['program']['zones']), key=lambda iz: (
                float(np.prod(np.array(iz[1]['max'], np.float32)-np.array(iz[1]['min'], np.float32))), iz[0]))[1]
            region = np.array([[zone['min'][0]-.2, -.2, zone['min'][1]-.2],
                               [zone['max'][0]+.2, manifest['room_size'][1]+.14, zone['max'][1]+.2]])
            assert np.allclose([ovmeta['local_region']['min'], ovmeta['local_region']['max']], region, atol=1e-5)
            yaw = manifest['world_yaw']; c, s = np.cos(yaw), np.sin(yaw)
            rotation = np.array([[c,0,s],[0,1,0],[-s,0,c]])
            # Captured transforms are column-major, in world coordinates. All
            # reconstruction cameras must belong to this room, not its context.
            cameras_local = data['world_from_view'][index, ..., 3, :3] @ rotation
            camera_xz = cameras_local[..., [0, 2]]
            assert (camera_xz >= np.array(zone['min'])-1e-5).all(), 'camera outside primary room'
            assert (camera_xz <= np.array(zone['max'])+1e-5).all(), 'camera outside primary room'
            assert ((cameras_local[..., 1] > 0) & (cameras_local[..., 1] < manifest['room_size'][1])).all()
            corners = np.array([[x,y,z] for x in region[:,0] for y in region[:,1] for z in region[:,2]]) @ rotation.T
            assert np.allclose(bounds, [corners.min(axis=0), corners.max(axis=0)], atol=1e-5)
            size = (bounds[1]-bounds[0])/resolution
            world = bounds[0]+(coords+dual/255.)*size
            local = world @ rotation
            error = float(np.maximum(region[0]-local, local-region[1]).max())
            # Conservative occupancy clamps dual points to each voxel; allow a cell
            # diagonal near the rotated crop, plus uint8 quantization.
            assert error <= np.linalg.norm(size)*1.01, (manifest['seed'], error)
            assert ovmeta['cache_version'] == 1, 'static capture redundantly rebaked'
            counts = {label: int((semantic == i).sum()) for i,label in enumerate(labels) if (semantic == i).any()}
            for label in ['floor','ceiling','wall']:
                assert counts.get(label, 0) > 0, f'missing architecture {label}'
            row = dict(seed=manifest['seed'], occupied_voxels=len(coords), resolution=resolution,
                       world_aabb=bounds.tolist(), annotation_aabb_matches_ovoxel=True,
                       cameras_in_primary_room=int(np.prod(cameras_local.shape[:-1])), semantic_counts=counts,
                       primary_objects=sum(not o['neighbor'] and zone['min'][0] <= o['position'][0] <= zone['max'][0] and zone['min'][1] <= o['position'][2] <= zone['max'][1] for o in manifest['objects']),
                       total_objects=len(manifest['objects']), primary_zone=zone,
                       primary_static_people=sum(not p['neighbor'] and zone['min'][0] <= p['position'][0] <= zone['max'][0] and zone['min'][1] <= p['position'][2] <= zone['max'][1] for p in manifest['humans']),
                       total_people=len(manifest['humans']),
                       max_crop_escape_m=max(0.,error), voxel_diagonal_m=float(np.linalg.norm(size)),
                       statistics=ovmeta['statistics'], cache_version=ovmeta['cache_version'],
                       completed_capture_requests=ovmeta['completed_capture_requests'], wait_updates=ovmeta['wait_updates'],
                       intersection_flag_counts=np.bincount(flags.astype(int), minlength=8).tolist())
            rows.append(row)
            samples.append((manifest, coords, dual, flags, semantic, labels, colors))
    assert rows, 'no exported chunks found'
    result = dict(schema_version=2, samples=rows, input_sha256=hashes,
                  limits='Conservative surface occupancy, not solid voxels or watertight mesh reconstruction. Semantic colors, not textured albedo. Primary-room world AABB is shared with scene annotation normalization; visible context can normalize outside [0, 1].')
    return result, samples


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--compare', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report, samples = audit(args.root)
    if args.compare:
        other, reference = audit(args.compare)
        assert len(samples) == len(reference)
        comparisons = []
        for a,b,row in zip(samples,reference,report['samples']):
            assert a[0] == b[0], 'different scene manifests'
            for i in [1,3,4]:
                assert np.array_equal(a[i],b[i]), f'CPU/GPU field mismatch: {i}'
            assert a[5] == b[5]
            error = int(np.abs(a[2].astype(int)-b[2].astype(int)).max())
            color_error = int(np.abs(a[6].astype(int)-b[6].astype(int)).max())
            # Different float32 operation/accumulation orders need not round to
            # identical uint8 attributes. Limit worst error to ~3% of one cell
            # component and require 99.9% to agree within one quantization unit.
            dual_errors = np.abs(a[2].astype(int)-b[2].astype(int))
            fraction_over_one = float((dual_errors > 1).mean())
            assert error <= 8 and color_error <= 8 and fraction_over_one <= .001, (
                a[0]['seed'], error, color_error, fraction_over_one)
            bounds = np.array(row['world_aabb'])
            cell_size = (bounds[1]-bounds[0])/row['resolution']
            max_world_error = float(np.linalg.norm(dual_errors*cell_size/255, axis=1).max())
            comparisons.append(dict(seed=a[0]['seed'], coordinates_flags_semantics_equal=True,
                                    dual_max_u8_error=error, dual_fraction_over_one_u8=fraction_over_one,
                                    dual_max_world_error_metres=max_world_error,
                                    semantic_color_max_u8_error=color_error))
        report['comparison'] = dict(other=other, samples=comparisons,
                                    tolerance={'exact':['coords','flags','semantic_ids','palette'],
                                               'max_attribute_u8_error':8, 'max_dual_fraction_over_one_u8':.001})
    report['script_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = args.output or args.root/'ovoxel_validation.json'
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['samples']}, indent=2))
