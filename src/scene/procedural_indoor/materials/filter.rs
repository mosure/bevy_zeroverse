//! Joint normal / GGX roughness filtering. Subpixel relief becomes a broader
//! specular lobe instead of vanishing when unit normals are averaged into mips.
use super::{mip_chain_image, mip_image, MapType, TextureMaps};
use bevy::prelude::*;

pub(super) fn images((color, normal, data): TextureMaps, size: u32) -> [Image; 3] {
    let color = mip_image(color, size, MapType::Color);
    let (normal, data) = filter(normal, data, size as usize);
    [
        color,
        mip_chain_image(normal, size, MapType::Normal),
        mip_chain_image(data, size, MapType::Data),
    ]
}

#[derive(Clone, Copy, Default)]
struct Moment {
    normal: Vec3,
    roughness4: f32,
}

fn filter(mut normals: Vec<u8>, mut data: Vec<u8>, mut size: usize) -> (Vec<u8>, Vec<u8>) {
    let mut mip_bytes = 0;
    let mut level = size;
    while level > 1 {
        level /= 2;
        mip_bytes += level * level * 4;
    }
    normals.reserve(mip_bytes);
    data.reserve(mip_bytes);
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
                // A row-major 2x2 reduction has consumed every source at or
                // before this destination. Reuse the prefix without changing
                // the unnormalized moment or floating-point reduction order.
                moments[y * next_size + x] = m;
            }
        }
        offset += size * size * 4;
        size = next_size;
        moments.truncate(next_size * next_size);
    }
    (normals, data)
}

fn decode(p: &[u8; 4]) -> Vec3 {
    Vec3::new(p[0] as f32, p[1] as f32, p[2] as f32) / 127.5 - Vec3::ONE
}

pub(super) fn encode(n: Vec3) -> [u8; 4] {
    let p = (n * 0.5 + Vec3::splat(0.5)) * 255.;
    [p.x.round() as u8, p.y.round() as u8, p.z.round() as u8, 255]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unresolved_relief_survives_all_mips_as_roughness_not_a_false_normal() {
        let mut normals = Vec::new();
        for y in 0..8 {
            for x in 0..8 {
                normals.extend(encode(
                    Vec3::new(if (x + y) % 2 == 0 { 0.3 } else { -0.3 }, 0., 1.).normalize(),
                ));
            }
        }
        let data = [255, 51, 255, 255].repeat(64);
        let (normals, data) = filter(normals, data, 8);
        assert_eq!(normals.len(), (64 + 16 + 4 + 1) * 4);
        for n in normals[64 * 4..].as_chunks::<4>().0 {
            let n = decode(n);
            assert!(n.x.abs() < 0.005 && n.y.abs() < 0.005 && n.z > 0.99);
        }
        for r in data[64 * 4..].as_chunks::<4>().0 {
            assert!(
                r[1] > 100 && r[1] < 150,
                "lost or accumulated variance: {r:?}"
            );
            assert_eq!([r[0], r[2], r[3]], [255; 3]);
        }
    }

    #[test]
    fn a_smooth_uniform_surface_keeps_its_roughness_and_metalness() {
        let (_, data) = filter(encode(Vec3::Z).repeat(64), [255, 90, 0, 255].repeat(64), 8);
        for r in data.as_chunks::<4>().0 {
            assert!((r[1] as i32 - 90).abs() <= 1);
            assert_eq!(r[2], 0);
        }
    }
}

#[cfg(test)]
mod replay_tests;
