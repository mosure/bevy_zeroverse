//! One storage binding for full-precision intersection and shading data.
//! Shadow rays touch only the first three vec4s per triangle. Small scenes
//! directly address two shading vec4s per hit. Large uploads share material
//! and UV indices in one word and store charts only for mapped materials.
use super::{BakeScene, DenseTriangle, GpuMaterial, TransportTree, Triangle};

fn compact_from() -> usize {
    (wgpu::Limits::default().max_storage_buffer_binding_size as usize / 2)
        .div_ceil(size_of::<DenseTriangle>())
}

fn intersection(t: &Triangle) -> [[u32; 4]; 3] {
    [
        t.a.extend(t.normal.x).to_array().map(f32::to_bits),
        t.ab.extend(t.normal.y).to_array().map(f32::to_bits),
        t.ac.extend(t.normal.z).to_array().map(f32::to_bits),
    ]
}

pub(super) fn pack(
    tree: &TransportTree,
    scene: &BakeScene,
    materials: &[GpuMaterial],
) -> (Vec<[u32; 4]>, [u32; 4]) {
    let count = tree.triangles.len();
    if count < compact_from() {
        // Small upload savings did not repay indexed shading on the measured
        // three-camera streams. Keep the fixed-stride transport for them.
        let mut records = Vec::with_capacity(5 * count);
        for t in tree.surfaces(scene) {
            records.extend(intersection(t));
        }
        for t in tree.surfaces(scene) {
            records.extend([
                [t.uv[0].x, t.uv[0].y, t.uv[1].x, t.uv[1].y].map(f32::to_bits),
                [t.uv[2].x, t.uv[2].y, t.material as f32, 0.].map(f32::to_bits),
            ]);
        }
        return (records, [(3 * count) as u32, 0, 0, 0]);
    }
    // Use all available low bits for the UV index. In typical rooms the
    // shader decodes both indices from the tag without another memory load.
    let material_bits = u32::BITS - (materials.len().saturating_sub(1) as u32).leading_zeros();
    encode(tree, scene, materials, (u32::BITS - material_bits).min(31))
}

pub(super) fn encode(
    tree: &TransportTree,
    scene: &BakeScene,
    materials: &[GpuMaterial],
    rank_bits: u32,
) -> (Vec<[u32; 4]>, [u32; 4]) {
    let count = tree.triangles.len();
    let tags = 3 * count;
    let blocks = tags + count.div_ceil(4);
    let block_size = 1usize << rank_bits;
    let block_count = count.div_ceil(block_size);
    let charts = blocks
        + if block_count > 1 {
            block_count.div_ceil(4)
        } else {
            0
        };
    let textured: Vec<_> = materials
        .iter()
        .map(|m| m.base_map[1] != 0 || m.emission_map[1] != 0)
        .collect();
    let mut records = Vec::with_capacity(charts + (3 * count).div_ceil(2));
    records.resize(charts, [0; 4]);
    let mut mapped = 0u32;
    let mut block_start = 0;
    for (i, t) in tree.surfaces(scene).enumerate() {
        records[3 * i..3 * i + 3].copy_from_slice(&intersection(t));
        if block_count > 1 && i.is_multiple_of(block_size) {
            block_start = mapped;
            let block = i >> rank_bits;
            records[blocks + block / 4][block % 4] = block_start;
        }
        records[tags + i / 4][i % 4] = ((t.material as u32) << rank_bits) | (mapped - block_start);
        if textured[t.material] {
            let [a, b, c] = t.uv.map(|v| v.to_array().map(f32::to_bits));
            if mapped.is_multiple_of(2) {
                records.extend([[a[0], a[1], b[0], b[1]], [c[0], c[1], 0, 0]]);
            } else {
                records.last_mut().unwrap()[2..].copy_from_slice(&a);
                records.push([b[0], b[1], c[0], c[1]]);
            }
            mapped += 1;
        }
    }
    (
        records,
        [tags as u32, blocks as u32, charts as u32, rank_bits],
    )
}

#[cfg(test)]
pub(super) mod tests {
    use super::*;
    use crate::scene::procedural_indoor::gi::{Triangle, Vec2, Vec3};

    pub(in super::super) fn assert_surfaces(
        data: &[[u32; 4]],
        offsets: [u32; 4],
        originals: &[Triangle],
        materials: &[GpuMaterial],
    ) {
        let rank_bits = offsets[3];
        let [tags, blocks, charts] = [offsets[0], offsets[1], offsets[2]].map(|x| x as usize * 4);
        let words = data.as_flattened();
        assert_eq!(tags, originals.len() * 12);
        let mut mapped = 0;
        for (i, t) in originals.iter().enumerate() {
            for (j, v) in [t.a, t.ab, t.ac].into_iter().enumerate() {
                assert_eq!(
                    data[3 * i + j],
                    v.extend(t.normal[j]).to_array().map(f32::to_bits)
                );
            }
            if rank_bits == 0 {
                let at = tags + i * 8;
                let uv = t.uv.map(|v| v.to_array().map(f32::to_bits));
                assert_eq!(&words[at..at + 6], uv.as_flattened());
                assert_eq!(words[at + 6], (t.material as f32).to_bits());
                continue;
            }
            let tag = words[tags + i];
            assert_eq!((tag >> rank_bits) as usize, t.material);
            let mat = &materials[t.material];
            if mat.base_map[1] != 0 || mat.emission_map[1] != 0 {
                let base = if blocks == charts {
                    0
                } else {
                    words[blocks + (i >> rank_bits)]
                };
                assert_eq!(base + (tag & ((1 << rank_bits) - 1)), mapped as u32);
                let uv = t.uv.map(|v| v.to_array().map(f32::to_bits));
                let at = charts + mapped * 6;
                assert_eq!(&words[at..at + 6], uv.as_flattened());
                mapped += 1;
            }
        }
        if rank_bits == 0 {
            assert_eq!(data.len(), originals.len() * 5);
        } else {
            assert_eq!(data.len(), (charts + mapped * 6).div_ceil(4));
        }
    }

    #[test]
    fn compact_layout_starts_at_half_the_portable_binding_capacity() {
        let threshold = compact_from();
        let half = wgpu::Limits::default().max_storage_buffer_binding_size as usize / 2;
        let stride = size_of::<DenseTriangle>();
        assert!((threshold - 1) * stride < half);
        assert!(threshold * stride >= half);
    }

    #[test]
    fn mixed_materials_partial_blocks_and_six_float_charts_round_trip() {
        use bytemuck::Zeroable;
        let mut materials = [GpuMaterial::zeroed(); 3];
        materials[1].base_map = [0, 64];
        materials[2].emission_map = [0, 64];
        for count in [0, 1, 31, 32, 33, 127, 128, 129, 197] {
            for pattern in 0..4 {
                let mut scene = super::super::super::tests::empty();
                for i in 0..count {
                    let u = i as f32 * 0.137 - 2.0;
                    let material = match pattern {
                        0 => 0,
                        1 => 1,
                        2 => 2 * usize::from(i % 31 == 0),
                        _ => (i + i / 17) % 3,
                    };
                    scene.triangles.push(Triangle {
                        a: Vec3::new(u, u * u, -0.0),
                        ab: Vec3::X,
                        ac: Vec3::Y,
                        uv: [
                            Vec2::new(-0., u),
                            Vec2::new(-u, u + 1.),
                            Vec2::new(3.0 * u, -2.0),
                        ],
                        normal: Vec3::Z,
                        material,
                    });
                }
                let tree = scene.gpu_transport();
                let originals: Vec<_> = tree.surfaces(&scene).copied().collect();
                let (data, offsets) = pack(tree, &scene, &materials);
                assert_eq!(offsets[3], 0);
                assert_surfaces(&data, offsets, &originals, &materials);
                let (data, offsets) = encode(tree, &scene, &materials, 30);
                assert_eq!(offsets[1], offsets[2]);
                assert_surfaces(&data, offsets, &originals, &materials);
                // Exercise the same relative ranks used when very large material
                // and triangle populations cannot share a full pair of indices.
                let (data, offsets) = encode(tree, &scene, &materials, 5);
                assert_surfaces(&data, offsets, &originals, &materials);
            }
        }
    }
}
