
struct Params {
    min: vec4<f32>,
    voxel: vec4<f32>,
    tile_dims: vec4<u32>,
    half_diag: f32,
    resolution: u32,
    tri_count: u32,
    max_output: u32,
    pair_cap: u32,
    _pad_params: vec3<u32>,
};

struct Triangle {
    a: vec4<f32>,
    b: vec4<f32>,
    c: vec4<f32>,
    min: vec4<f32>,
    max: vec4<f32>,
    color: vec4<f32>,
    semantic: u32,
    _pad_sem: vec3<u32>,
};

struct Voxel {
    coord: vec3<u32>,
    mask: u32,
    dual_sum: vec3<f32>,
    _pad_dual: f32,
    color_sum: vec4<f32>,
    semantic: u32,
    count: u32,
    _pad: vec2<u32>,
};

struct OutputMeta {
    count: atomic<u32>,
    overflow: atomic<u32>,
    _pad: vec2<u32>,
};

struct ActiveTile {
    tile_id: u32,
    start: u32,
    len: u32,
    _pad: u32,
};

struct TilePair {
    tile: u32,
    tri: u32,
};

struct ActiveCounter {
    pair_counter: atomic<u32>,
    pair_cursor: atomic<u32>,
    active_count: atomic<u32>,
    _pad: u32,
};

struct VoxelBuffer {
    header: OutputMeta,
    voxels: array<Voxel>,
};

struct DispatchArgs {
    x: u32,
    y: u32,
    z: u32,
};

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> tris: array<Triangle>;

// classify-only
@group(1) @binding(0) var<storage, read_write> tile_meta: array<atomic<u32>>;
@group(1) @binding(1) var<storage, read_write> tile_pairs: array<TilePair>;
@group(1) @binding(2) var<storage, read_write> active_counter: ActiveCounter;

// dispatch args
@group(3) @binding(0) var<storage, read_write> scatter_indirect: DispatchArgs;
@group(3) @binding(1) var<storage, read_write> voxel_indirect: DispatchArgs;

// voxel-only
@group(2) @binding(0) var<storage, read_write> active_tiles: array<ActiveTile>;
@group(2) @binding(1) var<storage, read_write> tile_indices: array<u32>;
@group(2) @binding(2) var<storage, read_write> voxel_buffer: VoxelBuffer;

const TILE_SIZE: u32 = 4u;
const GPU_SCATTER_WG: u32 = 128u;

fn total_tiles() -> u32 {
    return params.tile_dims.x * params.tile_dims.y * params.tile_dims.z;
}

fn tile_stride() -> u32 {
    return total_tiles();
}

fn count_index(tile_id: u32) -> u32 {
    return tile_id;
}

fn offset_index(tile_id: u32) -> u32 {
    return tile_id + tile_stride();
}

fn head_index(tile_id: u32) -> u32 {
    return tile_id + tile_stride() * 2u;
}

fn closest_point_on_triangle(p: vec3<f32>, tri: Triangle) -> vec3<f32> {
    let a = tri.a.xyz;
    let b = tri.b.xyz;
    let c = tri.c.xyz;
    let ab = b - a;
    let ac = c - a;
    let ap = p - a;
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    var result = a;

    if d1 <= 0.0 && d2 <= 0.0 {
        result = a;
    } else {
        let bp = p - b;
        let d3 = dot(ab, bp);
        let d4 = dot(ac, bp);
        if d3 >= 0.0 && d4 <= d3 {
            result = b;
        } else {
            let vc = d1 * d4 - d3 * d2;
            if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
                let v = d1 / (d1 - d3);
                result = a + v * ab;
            } else {
                let cp = p - c;
                let d5 = dot(ab, cp);
                let d6 = dot(ac, cp);
                if d6 >= 0.0 && d5 <= d6 {
                    result = c;
                } else {
                    let vb = d5 * d2 - d1 * d6;
                    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
                        let w = d2 / (d2 - d6);
                        result = a + w * ac;
                    } else {
                        let va = d3 * d6 - d5 * d4;
                        if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
                            let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
                            result = b + w * (c - b);
                        } else {
                            let denom = 1.0 / (va + vb + vc);
                            let v = vb * denom;
                            let w = vc * denom;
                            result = a + ab * v + ac * w;
                        }
                    }
                }
            }
        }
    }

    return result;
}

@compute @workgroup_size(256, 1, 1)
fn classify_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tri_idx = gid.x;
    if tri_idx >= params.tri_count {
        return;
    }
    let tri = tris[tri_idx];
    let tri_min = tri.min.xyz;
    let tri_max = tri.max.xyz;

    let min = params.min.xyz;
    let voxel = params.voxel.xyz;
    let start = clamp(
        floor((tri_min - min) / voxel),
        vec3<f32>(0.0),
        vec3<f32>(f32(params.resolution - 1u)),
    );
    let end = clamp(
        ceil((tri_max - min) / voxel),
        vec3<f32>(0.0),
        vec3<f32>(f32(params.resolution - 1u)),
    );

    let tile_dim = vec3<u32>(
        (params.resolution + TILE_SIZE - 1u) / TILE_SIZE,
        (params.resolution + TILE_SIZE - 1u) / TILE_SIZE,
        (params.resolution + TILE_SIZE - 1u) / TILE_SIZE,
    );

    let start_tile = clamp(
        floor(start / f32(TILE_SIZE)),
        vec3<f32>(0.0),
        vec3<f32>(f32(tile_dim.x - 1u)),
    );
    let end_tile = clamp(
        floor(end / f32(TILE_SIZE)),
        vec3<f32>(0.0),
        vec3<f32>(f32(tile_dim.x - 1u)),
    );

    for (var tz: u32 = u32(start_tile.z); tz <= u32(end_tile.z); tz = tz + 1u) {
        for (var ty: u32 = u32(start_tile.y); ty <= u32(end_tile.y); ty = ty + 1u) {
            for (var tx: u32 = u32(start_tile.x); tx <= u32(end_tile.x); tx = tx + 1u) {
                let tile_id = tx + tile_dim.x * (ty + tile_dim.y * tz);
                let pair_idx = atomicAdd(&active_counter.pair_counter, 1u);
                if pair_idx >= params.pair_cap {
                    return;
                }
                tile_pairs[pair_idx] = TilePair(tile_id, tri_idx);
                _ = atomicAdd(&tile_meta[count_index(tile_id)], 1u);
            }
        }
    }
}

@compute @workgroup_size(256, 1, 1)
fn prefix_tiles(@builtin(global_invocation_id) gid: vec3<u32>) {
    let tile_id = gid.x;
    let tiles_total = total_tiles();
    if tile_id >= tiles_total {
        return;
    }
    let count = atomicLoad(&tile_meta[count_index(tile_id)]);
    if count == 0u {
        atomicStore(&tile_meta[offset_index(tile_id)], 0u);
        atomicStore(&tile_meta[head_index(tile_id)], 0u);
        return;
    }
    let offset = atomicAdd(&active_counter.pair_cursor, count);
    if offset >= params.pair_cap {
        return;
    }
    let remaining = params.pair_cap - offset;
    let clamped = select(count, remaining, remaining < count);
    atomicStore(&tile_meta[offset_index(tile_id)], offset);
    atomicStore(&tile_meta[head_index(tile_id)], offset);
    atomicStore(&tile_meta[count_index(tile_id)], clamped);
    let active_idx = atomicAdd(&active_counter.active_count, 1u);
    if active_idx >= tiles_total {
        return;
    }
    active_tiles[active_idx] = ActiveTile(tile_id, offset, clamped, 0u);
}

@compute @workgroup_size(1, 1, 1)
fn prepare_dispatch(@builtin(global_invocation_id) gid: vec3<u32>) {
    if gid.x > 0u {
        return;
    }
    let pairs = atomicLoad(&active_counter.pair_cursor);
    let scatter = (pairs + GPU_SCATTER_WG - 1u) / GPU_SCATTER_WG;
    scatter_indirect.x = max(scatter, 1u);
    scatter_indirect.y = 1u;
    scatter_indirect.z = 1u;

    let tiles = atomicLoad(&active_counter.active_count);
    voxel_indirect.x = max(tiles, 1u);
    voxel_indirect.y = 1u;
    voxel_indirect.z = 1u;
}

@compute @workgroup_size(128, 1, 1)
fn scatter_pairs(@builtin(global_invocation_id) gid: vec3<u32>) {
    let pair_idx = gid.x;
    let total = min(atomicLoad(&active_counter.pair_cursor), params.pair_cap);
    if pair_idx >= total {
        return;
    }
    let pair = tile_pairs[pair_idx];
    let tile_id = pair.tile;
    let offset = atomicLoad(&tile_meta[offset_index(tile_id)]);
    let count = atomicLoad(&tile_meta[count_index(tile_id)]);
    if count == 0u {
        return;
    }
    let head = atomicAdd(&tile_meta[head_index(tile_id)], 1u);
    if head < offset + count && head < params.pair_cap {
        tile_indices[head] = pair.tri;
    }
}

fn decode_morton_local(idx: u32) -> vec3<u32> {
    let x = ((idx >> 0u) & 1u) | (((idx >> 3u) & 1u) << 1u);
    let y = ((idx >> 1u) & 1u) | (((idx >> 4u) & 1u) << 1u);
    let z = ((idx >> 2u) & 1u) | (((idx >> 5u) & 1u) << 1u);
    return vec3<u32>(x, y, z);
}

@compute @workgroup_size(64, 1, 1)
fn voxel_main(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tile_idx = wid.x;
    let active_max = atomicLoad(&active_counter.active_count);
    if tile_idx >= active_max {
        return;
    }

    let at = active_tiles[tile_idx];
    if at.len == 0u {
        return;
    }

    let tiles_xy = params.tile_dims.x * params.tile_dims.y;
    let tz = at.tile_id / tiles_xy;
    let rem = at.tile_id - tz * tiles_xy;
    let ty = rem / params.tile_dims.x;
    let tx = rem - ty * params.tile_dims.x;
    let origin = vec3<u32>(tx * TILE_SIZE, ty * TILE_SIZE, tz * TILE_SIZE);

    let local_coord = decode_morton_local(lid.x);
    let voxel_coord = origin + local_coord;
    if any(voxel_coord >= vec3<u32>(params.resolution)) {
        return;
    }

    let voxel_extent = params.voxel.xyz;
    let voxel_min = params.min.xyz + vec3<f32>(voxel_coord) * voxel_extent;
    let center = voxel_min + voxel_extent * 0.5;

    var count: u32 = 0u;
    var dual_sum: vec3<f32> = vec3<f32>(0.0);
    var color_sum: vec4<f32> = vec4<f32>(0.0);
    var mask: u32 = 0u;
    var semantic: u32 = 0u;

    let range_end = at.start + at.len;
    for (var idx: u32 = at.start; idx < range_end; idx = idx + 1u) {
        let tri_idx = tile_indices[idx];
        let tri = tris[tri_idx];
        let tri_min = tri.min.xyz;
        let tri_max = tri.max.xyz;
        if any(tri_min > voxel_min + voxel_extent) || any(tri_max < voxel_min) {
            continue;
        }
        let closest = closest_point_on_triangle(center, tri);
        let dist = distance(center, closest);
        if dist > params.half_diag {
            continue;
        }
        let offset = clamp(
            (closest - voxel_min) / voxel_extent,
            vec3<f32>(0.0),
            vec3<f32>(1.0),
        );
        if tri_min.x < voxel_min.x && tri_max.x > voxel_min.x { mask = mask | 1u; }
        if tri_min.y < voxel_min.y && tri_max.y > voxel_min.y { mask = mask | 2u; }
        if tri_min.z < voxel_min.z && tri_max.z > voxel_min.z { mask = mask | 4u; }
        dual_sum = dual_sum + offset;
        color_sum = color_sum + tri.color;
        if count == 0u {
            semantic = tri.semantic;
        }
        count = count + 1u;
    }

    if count == 0u {
        return;
    }

    let write_idx = atomicAdd(&voxel_buffer.header.count, 1u);
    if write_idx >= params.max_output {
        _ = atomicAdd(&voxel_buffer.header.overflow, 1u);
        return;
    }

    voxel_buffer.voxels[write_idx].coord = voxel_coord;
    voxel_buffer.voxels[write_idx].mask = mask;
    voxel_buffer.voxels[write_idx].dual_sum = dual_sum;
    voxel_buffer.voxels[write_idx].color_sum = color_sum;
    voxel_buffer.voxels[write_idx].semantic = semantic;
    voxel_buffer.voxels[write_idx].count = count;
}
