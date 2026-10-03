//! Static, scene-derived HDR reflection radiance and split-sum IBL convolution.
//! One probe samples real geometry; it approximates secondary bounces and has
//! no per-object parallax or live reflections of moving people.
use super::*;
use crate::scene::procedural_indoor::{architecture, gi::BakeScene};
use std::f32::consts::{PI, TAU};

const SIZE: u32 = 64;
const SAMPLES: u32 = 64;

pub(super) fn build(scene: &IndoorManifest, transport: &BakeScene) -> [Image; 2] {
    let origin = transport.reflection_origin(scene);
    let ambient = scene
        .domain()
        .map_or(45., |d| d.photometry.environment_intensity);
    let fixture_size = architecture::fixture_size(scene).xz() - Vec2::new(0.06, 0.055);
    let emitters: Vec<_> = architecture::fixture_positions(scene)
        .into_iter()
        .enumerate()
        .map(|(i, p)| {
            let (color, lumens) = architecture::fixture_photometry(scene, i);
            let c = Color::srgb(color.x, color.y, color.z).to_linear();
            (
                p - Vec3::Y * 0.029,
                Vec3::new(c.red, c.green, c.blue)
                    * (lumens / (PI * fixture_size.element_product())),
            )
        })
        .collect();
    let trace_face = |face| {
        let mut pixels = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE {
            for x in 0..SIZE {
                // Bevy negates Z when sampling its left-handed cubemaps.
                let d = direction(face, x as f32, y as f32, SIZE);
                let ray = Vec3::new(d.x, d.y, -d.z);
                let (mut distance, mut radiance) =
                    transport.reflection_radiance(origin, ray, ambient);
                for &(p, emission) in &emitters {
                    let t = (p.y - origin.y) / ray.y;
                    let q = origin + ray * t;
                    if ray.y > 0.
                        && t > 0.
                        && t < distance
                        && (q.xz() - p.xz()).abs().cmplt(fixture_size * 0.5).all()
                    {
                        distance = t;
                        radiance = emission;
                    }
                }
                pixels.push(radiance);
            }
        }
        pixels
    };
    #[cfg(not(target_arch = "wasm32"))]
    let faces = crate::scene::procedural_indoor::preparation::workers::pool().scope(|scope| {
        for face in 0..6 {
            let trace = &trace_face;
            scope.spawn(async move { trace(face) });
        }
    });
    #[cfg(target_arch = "wasm32")]
    let faces: Vec<_> = (0..6).map(trace_face).collect();
    convolve(&faces.into_iter().flatten().collect::<Vec<_>>(), SIZE)
}

fn direction(face: u32, x: f32, y: f32, size: u32) -> Vec3 {
    let u = 2. * (x + 0.5) / size as f32 - 1.;
    let v = 2. * (y + 0.5) / size as f32 - 1.;
    match face {
        0 => Vec3::new(1., -v, -u),
        1 => Vec3::new(-1., -v, u),
        2 => Vec3::new(u, 1., v),
        3 => Vec3::new(u, -1., -v),
        4 => Vec3::new(u, -v, 1.),
        _ => Vec3::new(-u, -v, -1.),
    }
    .normalize()
}

fn coordinates(d: Vec3, size: u32) -> (u32, Vec2) {
    let a = d.abs();
    let (face, uv, major) = if a.x >= a.y && a.x >= a.z {
        if d.x > 0. {
            (0, Vec2::new(-d.z, -d.y), a.x)
        } else {
            (1, Vec2::new(d.z, -d.y), a.x)
        }
    } else if a.y >= a.z {
        if d.y > 0. {
            (2, Vec2::new(d.x, d.z), a.y)
        } else {
            (3, Vec2::new(d.x, -d.z), a.y)
        }
    } else if d.z > 0. {
        (4, Vec2::new(d.x, -d.y), a.z)
    } else {
        (5, Vec2::new(-d.x, -d.y), a.z)
    };
    (
        face,
        (uv / major + Vec2::ONE) * (size as f32 * 0.5) - Vec2::splat(0.5),
    )
}

fn sample(pixels: &[Vec3], size: u32, d: Vec3) -> Vec3 {
    let (face, p) = coordinates(d, size);
    let lo = p.floor();
    let f = p - lo;
    let tap = |x: f32, y: f32| {
        // Edge taps cross onto the adjacent face instead of clamping a seam.
        let (face, p) = coordinates(direction(face, x, y, size), size);
        let x = p.x.round().clamp(0., (size - 1) as f32) as u32;
        let y = p.y.round().clamp(0., (size - 1) as f32) as u32;
        pixels[(face * size * size + y * size + x) as usize]
    };
    tap(lo.x, lo.y).lerp(tap(lo.x + 1., lo.y), f.x).lerp(
        tap(lo.x, lo.y + 1.).lerp(tap(lo.x + 1., lo.y + 1.), f.x),
        f.y,
    )
}

fn basis(n: Vec3) -> (Vec3, Vec3) {
    let t = if n.y.abs() < 0.999 {
        Vec3::Y.cross(n)
    } else {
        Vec3::X.cross(n)
    }
    .normalize();
    (t, n.cross(t))
}
fn hammersley(i: u32) -> Vec2 {
    Vec2::new(
        (i as f32 + 0.5) / SAMPLES as f32,
        i.reverse_bits() as f32 * 2.328_306_4e-10,
    )
}
struct RadianceMips(Vec<(u32, Vec<Vec3>)>);
impl RadianceMips {
    fn new(pixels: &[Vec3], size: u32) -> Self {
        let mut levels = vec![(size, pixels.to_vec())];
        while levels.last().unwrap().0 > 1 {
            let (n, source) = levels.last().unwrap();
            let next = n / 2;
            let mut pixels = Vec::with_capacity((6 * next * next) as usize);
            for face in 0..6 {
                for y in 0..next {
                    for x in 0..next {
                        let mut sum = Vec3::ZERO;
                        let mut weights = 0.;
                        for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                            let u = 2. * (2 * x + dx) as f32 / *n as f32 + 1. / *n as f32 - 1.;
                            let v = 2. * (2 * y + dy) as f32 / *n as f32 + 1. / *n as f32 - 1.;
                            let w = (1. + u * u + v * v).powf(-1.5);
                            sum +=
                                source[(face * n * n + (2 * y + dy) * n + 2 * x + dx) as usize] * w;
                            weights += w;
                        }
                        pixels.push(sum / weights);
                    }
                }
            }
            levels.push((next, pixels));
        }
        Self(levels)
    }
    fn sample(&self, direction: Vec3, lod: f32) -> Vec3 {
        let lod = lod.clamp(0., (self.0.len() - 1) as f32);
        let i = lod.floor() as usize;
        let (size, pixels) = &self.0[i];
        let a = sample(pixels, *size, direction);
        let (size, pixels) = &self.0[(i + 1).min(self.0.len() - 1)];
        a.lerp(sample(pixels, *size, direction), lod.fract())
    }
}
fn convolve(pixels: &[Vec3], size: u32) -> [Image; 2] {
    let source = RadianceMips::new(pixels, size);
    let texel_angle = 4. * PI / (6 * size * size) as f32;
    let levels = size.ilog2() + 1;
    let mut specular = Vec::new();
    let mut diffuse = Vec::new();
    // Explicit layer-major ordering matches Bevy/wgpu Image's upload contract.
    for face in 0..6 {
        for level in 0..levels {
            let n = size >> level;
            let roughness = level as f32 / (levels - 1) as f32;
            for y in 0..n {
                for x in 0..n {
                    let normal = direction(face, x as f32, y as f32, n);
                    let (t, b) = basis(normal);
                    let mut value = Vec3::ZERO;
                    let mut weight = 0.;
                    if level == 0 {
                        value = pixels[(face * size * size + y * size + x) as usize];
                    } else {
                        let alpha2 = roughness.powi(4);
                        for i in 0..SAMPLES {
                            let xi = hammersley(i);
                            let cos = ((1. - xi.x) / (1. + (alpha2 - 1.) * xi.x)).sqrt();
                            let sin = (1. - cos * cos).sqrt();
                            let phi = TAU * xi.y;
                            let h = t * (sin * phi.cos()) + b * (sin * phi.sin()) + normal * cos;
                            let l = h * (2. * cos) - normal;
                            let w = normal.dot(l).max(0.);
                            // Sample the source footprint implied by the GGX PDF.
                            // Reading only level zero made small HDR emitters turn
                            // into isolated noisy dots in rough reflections.
                            let denominator = cos * cos * (alpha2 - 1.) + 1.;
                            let pdf = alpha2 / (4. * PI * denominator * denominator);
                            let lod = 0.5 * (1. / (SAMPLES as f32 * pdf * texel_angle)).log2();
                            value += source.sample(l, lod) * w;
                            weight += w;
                        }
                        value /= weight.max(1e-6);
                    }
                    encode(&mut specular, value);
                }
            }
        }
        for y in 0..16 {
            for x in 0..16 {
                let n = direction(face, x as f32, y as f32, 16);
                let (t, b) = basis(n);
                let mut value = Vec3::ZERO;
                for i in 0..SAMPLES {
                    let xi = hammersley(i);
                    let r = xi.x.sqrt();
                    let phi = TAU * xi.y;
                    let l = t * (r * phi.cos()) + b * (r * phi.sin()) + n * (1. - xi.x).sqrt();
                    let pdf = (1. - xi.x).sqrt() / PI;
                    let lod = 0.5 * (1. / (SAMPLES as f32 * pdf * texel_angle)).log2();
                    value += source.sample(l, lod);
                }
                encode(&mut diffuse, value / SAMPLES as f32);
            }
        }
    }
    [cube(diffuse, 16, 1), cube(specular, size, levels)]
}
fn encode(bytes: &mut Vec<u8>, value: Vec3) {
    for v in [value.x, value.y, value.z, 1.] {
        bytes.extend(half::f16::from_f32(v.clamp(0., 65000.)).to_le_bytes());
    }
}
fn cube(bytes: Vec<u8>, size: u32, levels: u32) -> Image {
    let mut image = Image::new(
        Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 6,
        },
        TextureDimension::D2,
        vec![0; (size * size * 6 * 8) as usize],
        TextureFormat::Rgba16Float,
        RenderAssetUsages::default(),
    );
    image.data = Some(bytes);
    image.texture_descriptor.mip_level_count = levels;
    image.texture_view_descriptor = Some(TextureViewDescriptor {
        dimension: Some(TextureViewDimension::Cube),
        ..default()
    });
    image.sampler = ImageSampler::linear();
    image
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn cubemap_faces_roundtrip_and_keep_cross_face_sampling_continuous() {
        let size = 8;
        let pixels: Vec<_> = (0..6)
            .flat_map(|f| {
                (0..size).flat_map(move |y| {
                    (0..size).map(move |x| direction(f, x as f32, y as f32, size) + Vec3::ONE)
                })
            })
            .collect();
        for face in 0..6 {
            for y in 0..size {
                for x in 0..size {
                    let d = direction(face, x as f32, y as f32, size);
                    let (f, p) = coordinates(d, size);
                    assert_eq!(face, f);
                    assert!(p.distance(Vec2::new(x as f32, y as f32)) < 1e-5);
                    assert!(sample(&pixels, size, d).distance(d + Vec3::ONE) < 1e-5);
                }
            }
        }
        let a = sample(&pixels, size, Vec3::new(1., 0.3, 1. - 1e-5));
        let b = sample(&pixels, size, Vec3::new(1. - 1e-5, 0.3, 1.));
        assert!(a.distance(b) < 1e-4);
    }
    #[test]
    fn hdr_convolution_preserves_constant_radiance_without_clipping_or_srgb() {
        let radiance = Vec3::new(0.25, 37., 2048.);
        for image in convolve(&vec![radiance; 6 * 8 * 8], 8) {
            assert_eq!(image.texture_descriptor.format, TextureFormat::Rgba16Float);
            for pixel in image.data.unwrap().as_chunks::<8>().0 {
                let values: Vec<_> = pixel
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|b| half::f16::from_le_bytes(*b).to_f32())
                    .collect();
                assert_eq!(values, [radiance.x, radiance.y, radiance.z, 1.]);
            }
        }
    }
}
