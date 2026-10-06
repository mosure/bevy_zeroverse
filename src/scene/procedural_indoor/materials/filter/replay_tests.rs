//! Original allocating reducer as an independent exact mip oracle.
use super::*;

fn reference(mut normals: Vec<u8>, mut data: Vec<u8>, mut size: usize) -> (Vec<u8>, Vec<u8>) {
    let mut moments: Vec<_> = normals
        .as_chunks::<4>()
        .0
        .iter()
        .zip(data.as_chunks::<4>().0)
        .map(|(n, r)| Moment {
            normal: decode(n).normalize_or(Vec3::Z),
            // Bevy uses perceptual roughness r; GGX alpha = r^2. Mix alpha^2,
            // not r, to keep a rough patch from filtering into a mirror.
            roughness4: (r[1] as f32 / 255.).powi(4),
        })
        .collect();
    let mut offset = 0;
    while size > 1 {
        let next_size = size / 2;
        let mut next = Vec::with_capacity(next_size * next_size);
        for y in 0..next_size {
            for x in 0..next_size {
                let mut m = Moment::default();
                let mut channels = [0_u32; 2];
                for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                    let i = (y * 2 + dy) * size + x * 2 + dx;
                    m.normal += moments[i].normal * 0.25;
                    m.roughness4 += moments[i].roughness4 * 0.25;
                    channels[0] += data[offset + i * 4] as u32;
                    channels[1] += data[offset + i * 4 + 2] as u32;
                }
                // Carry the UNNORMALIZED first moment to every subsequent mip.
                // Renormalizing it in the reduction would erase fine variance.
                let length = m.normal.length().clamp(0.0001, 1.);
                let variance = (1. - length) / length;
                let roughness = (m.roughness4 + variance).min(1.).sqrt().sqrt();
                normals.extend(encode(m.normal / length));
                data.extend([
                    (channels[0] / 4) as u8,
                    (roughness * 255.).round() as u8,
                    (channels[1] / 4) as u8,
                    255,
                ]);
                next.push(m);
            }
        }
        offset += size * size * 4;
        size = next_size;
        moments = next;
    }
    (normals, data)
}

#[test]
fn reused_moment_buffer_preserves_every_pbr_mip_byte() {
    for size in [1, 2, 8, 64, 256, 512] {
        for seed in [0, 202, 43_084_584] {
            for field in 0..3 {
                let mut normals = Vec::with_capacity(size * size * 4);
                let mut data = Vec::with_capacity(size * size * 4);
                for y in 0..size {
                    for x in 0..size {
                        let value =
                            |stream: u64| super::super::hash(x as u32, y as u32, seed + stream);
                        let normal = match field {
                            0 => Vec3::Z,
                            1 => Vec3::new(value(17) - 0.5, value(37) - 0.5, value(51) + 0.01)
                                .normalize(),
                            _ => Vec3::new(if (x + y) % 2 == 0 { 0.8 } else { -0.8 }, 0., 1.)
                                .normalize(),
                        };
                        normals.extend(encode(normal));
                        data.extend([
                            (value(67) * 255.) as u8,
                            (value(83) * 255.) as u8,
                            (value(97) * 255.) as u8,
                            255,
                        ]);
                    }
                }
                let expected = reference(normals.clone(), data.clone(), size);
                assert_eq!(
                    filter(normals, data, size),
                    expected,
                    "size {size} seed {seed} field {field}"
                );
            }
        }
    }
}
