//! Test-only mesh preview. Flat CPU shading, not a photometric/PBR render.
use super::*;
use crate::scene::procedural_indoor::{materials::Surface, objects::Assembly};
use image::{Rgb, RgbImage};
pub(crate) fn orthographic(
    a: &Assembly,
    center: Vec3,
    span: f32,
    n: u32,
    forward: Vec3,
) -> RgbImage {
    let mut image = RgbImage::from_pixel(n, n, Rgb([241, 243, 246]));
    let mut depth = vec![f32::NEG_INFINITY; (n * n) as usize];
    let forward = forward.normalize();
    let right = Vec3::Y.cross(forward).normalize();
    let up = forward.cross(right);
    let project = |p: Vec3| {
        let p = (p - center) / span;
        Vec3::new(
            (0.5 + right.dot(p)) * n as f32,
            (0.5 - up.dot(p)) * n as f32,
            forward.dot(p),
        )
    };
    for ((surface, _), g) in &a.parts {
        let base = match surface {
            Surface::Metal | Surface::Chrome => [140., 155., 169.],
            Surface::Wood | Surface::WoodEdge => [162., 114., 72.],
            Surface::Art => [65., 143., 160.],
            Surface::Screen | Surface::PhoneScreen | Surface::Television => [57., 100., 138.],
            Surface::Leaf | Surface::LeafLight | Surface::LeafVariegated | Surface::Succulent => {
                [85., 122., 65.]
            }
            Surface::Bark => [117., 91., 68.],
            Surface::Glass => [74., 105., 121.],
            Surface::Terracotta => [171., 109., 81.],
            Surface::Soil => [105., 91., 69.],
            Surface::Paper | Surface::Whiteboard => [239., 235., 220.],
            Surface::Ink | Surface::Rubber => [43., 49., 55.],
            Surface::Light => [251., 234., 192.],
            Surface::FabricAlt => [114., 139., 128.],
            _ => [182., 174., 164.],
        };
        for tri in g.indices.as_chunks::<3>().0 {
            let world = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
            let p = world.map(project);
            let cross = |a: Vec2, b: Vec2, c: Vec2| (b - a).perp_dot(c - a);
            let area = cross(p[0].xy(), p[1].xy(), p[2].xy());
            if area.abs() < 1e-5 {
                continue;
            }
            let lo = p[0]
                .min(p[1])
                .min(p[2])
                .xy()
                .floor()
                .max(Vec2::ZERO)
                .as_uvec2();
            let hi = p[0]
                .max(p[1])
                .max(p[2])
                .xy()
                .ceil()
                .min(Vec2::splat((n - 1) as f32))
                .as_uvec2();
            let normal = (world[1] - world[0])
                .cross(world[2] - world[0])
                .normalize_or_zero();
            let brightness = 0.40 + 0.60 * normal.dot(Vec3::new(-0.4, 0.8, 1.).normalize()).abs();
            let color = Rgb(base.map(|v| (v * brightness).round() as u8));
            for y in lo.y..=hi.y {
                for x in lo.x..=hi.x {
                    let q = Vec2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let w = [
                        cross(p[1].xy(), p[2].xy(), q) / area,
                        cross(p[2].xy(), p[0].xy(), q) / area,
                        cross(p[0].xy(), p[1].xy(), q) / area,
                    ];
                    if w.iter().any(|&v| v < -1e-5) {
                        continue;
                    }
                    let z = w[0] * p[0].z + w[1] * p[1].z + w[2] * p[2].z;
                    let i = (y * n + x) as usize;
                    if z > depth[i] {
                        depth[i] = z;
                        image.put_pixel(x, y, color);
                    }
                }
            }
        }
    }
    image
}
