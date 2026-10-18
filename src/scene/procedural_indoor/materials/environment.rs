//! Static, scene-derived HDR reflection radiance and split-sum IBL convolution.
//! Bounded room probes sample actual static geometry. Secondary bounces remain
//! approximate; moving people are not re-traced into the static cubemaps.
use super::*;
use crate::scene::procedural_indoor::gi::{BakeScene, ReflectionEmitters};
use std::f32::consts::{PI, TAU};
#[cfg(any(test, target_arch = "wasm32"))]
pub(super) mod cooperative;
pub(crate) mod gpu;
mod kernel;
#[cfg(test)]
mod round4_replay;
mod stencil;
#[cfg(any(test, target_arch = "wasm32"))]
pub(super) use cooperative::build_async;
use kernel::ConvolutionKernel;

const SIZE: u32 = 64;
const SAMPLES: u32 = 64;

#[cfg(test)]
pub(super) fn build(scene: &IndoorManifest, transport: &BakeScene) -> [Image; 2] {
    build_at(scene, transport, transport.reflection_origin(scene))
}

pub(super) fn insert_many(
    scene: &IndoorManifest,
    transport: &BakeScene,
    origins: &[Vec3],
    images: &mut impl super::super::preparation::AssetStore<Image>,
    requests: &mut super::gpu::Requests,
    gpu: bool,
) -> Vec<[Handle<Image>; 2]> {
    let sources = trace_many(scene, transport, origins);
    if gpu {
        sources
            .into_iter()
            .map(|pixels| gpu::prepare(pixels, images, requests))
            .collect()
    } else {
        #[cfg(not(target_arch = "wasm32"))]
        let maps = crate::scene::procedural_indoor::preparation::workers::pool().scope(|scope| {
            for pixels in sources {
                scope.spawn(async move { convolve(&pixels, SIZE) });
            }
        });
        #[cfg(target_arch = "wasm32")]
        let maps: Vec<_> = sources
            .into_iter()
            .map(|pixels| convolve(&pixels, SIZE))
            .collect();
        maps.into_iter()
            .map(|maps| maps.map(|image| images.add(image)))
            .collect()
    }
}

fn trace_many(scene: &IndoorManifest, transport: &BakeScene, origins: &[Vec3]) -> Vec<Vec<Vec3>> {
    // At most four room probes. Share immutable light/fixture inputs and queue
    // all faces together so a dense face does not leave other workers idle.
    assert!(origins.len() <= 4);
    let trace = ProbeTrace::new(scene, transport, Vec3::ZERO);
    let face = |origin, face| {
        let mut pixels = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE {
            trace.row_at(origin, face, y, &mut pixels);
        }
        pixels
    };
    #[cfg(not(target_arch = "wasm32"))]
    let faces = crate::scene::procedural_indoor::preparation::workers::pool().scope(|scope| {
        for &origin in origins {
            for i in 0..6 {
                let face = &face;
                scope.spawn(async move { face(origin, i) });
            }
        }
    });
    #[cfg(target_arch = "wasm32")]
    let faces: Vec<_> = origins
        .iter()
        .flat_map(|&origin| (0..6).map(move |i| face(origin, i)))
        .collect();
    let mut faces = faces.into_iter();
    origins
        .iter()
        .map(|_| faces.by_ref().take(6).flatten().collect())
        .collect()
}

#[cfg(test)]
pub(super) fn build_at(scene: &IndoorManifest, transport: &BakeScene, origin: Vec3) -> [Image; 2] {
    convolve(&trace_at(scene, transport, origin), SIZE)
}

#[cfg(test)]
pub(super) fn trace_at(scene: &IndoorManifest, transport: &BakeScene, origin: Vec3) -> Vec<Vec3> {
    let trace = ProbeTrace::new(scene, transport, origin);
    let trace_face = |face| {
        let mut pixels = Vec::with_capacity((SIZE * SIZE) as usize);
        for y in 0..SIZE {
            trace.row(face, y, &mut pixels);
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
    faces.into_iter().flatten().collect()
}

struct ProbeTrace<'a> {
    transport: &'a BakeScene,
    #[cfg(any(test, target_arch = "wasm32"))]
    origin: Vec3,
    ambient: f32,
    emitters: ReflectionEmitters,
}

impl<'a> ProbeTrace<'a> {
    fn new(scene: &IndoorManifest, transport: &'a BakeScene, origin: Vec3) -> Self {
        #[cfg(not(any(test, target_arch = "wasm32")))]
        let _ = origin;
        let ambient = scene
            .domain()
            .map_or(45., |d| d.photometry.environment_intensity);
        let emitters = ReflectionEmitters::new(scene);
        Self {
            transport,
            #[cfg(any(test, target_arch = "wasm32"))]
            origin,
            ambient,
            emitters,
        }
    }

    #[cfg(any(test, target_arch = "wasm32"))]
    fn row(&self, face: u32, y: u32, pixels: &mut Vec<Vec3>) {
        self.row_at(self.origin, face, y, pixels);
    }

    fn row_at(&self, origin: Vec3, face: u32, y: u32, pixels: &mut Vec<Vec3>) {
        for x in 0..SIZE {
            // Bevy negates Z when sampling its left-handed cubemaps.
            let d = direction(face, x as f32, y as f32, SIZE);
            let ray = Vec3::new(d.x, d.y, -d.z);
            let (distance, radiance) =
                self.transport
                    .reflection_radiance(origin, ray, self.ambient);
            let radiance = self
                .emitters
                .radiance(origin, ray, distance)
                .unwrap_or(radiance);
            pixels.push(radiance);
        }
    }
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

#[cfg(test)]
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
    fn sample(&self, kernel: &ConvolutionKernel, direction: Vec3, lod: f32) -> Vec3 {
        let lod = lod.clamp(0., (self.0.len() - 1) as f32);
        let i = lod.floor() as usize;
        let (_, pixels) = &self.0[i];
        let a = kernel.taps[i].sample(pixels, direction);
        let j = (i + 1).min(self.0.len() - 1);
        let (_, pixels) = &self.0[j];
        a.lerp(kernel.taps[j].sample(pixels, direction), lod.fract())
    }
}
fn convolve(pixels: &[Vec3], size: u32) -> [Image; 2] {
    if size == SIZE {
        return stencil::convolve(pixels);
    }
    let source = RadianceMips::new(pixels, size);
    // Only the source radiance changes between rooms. Integer cube-edge taps and
    // the quadrature coefficients are immutable and preserve the original math.
    // The production-size kernel has fixed process residency; other sizes are
    // owned by this call instead of creating an unbounded size-keyed cache.
    let owned_kernel = (size != SIZE).then(|| ConvolutionKernel::new(size));
    let kernel = owned_kernel
        .as_ref()
        .unwrap_or_else(|| ConvolutionKernel::production());
    let levels = size.ilog2() + 1;
    let mut specular = Vec::new();
    let mut diffuse = Vec::new();
    // Explicit layer-major ordering matches Bevy/wgpu Image's upload contract.
    for face in 0..6 {
        for level in 0..levels {
            let n = size >> level;
            for y in 0..n {
                for x in 0..n {
                    let normal = direction(face, x as f32, y as f32, n);
                    let (t, b) = basis(normal);
                    let mut value = Vec3::ZERO;
                    let mut weight = 0.;
                    if level == 0 {
                        value = pixels[(face * size * size + y * size + x) as usize];
                    } else {
                        for q in &kernel.specular[level as usize] {
                            let h = t * q.x + b * q.y + normal * q.z;
                            let l = h * (2. * q.z) - normal;
                            let w = normal.dot(l).max(0.);
                            // Sample the source footprint implied by the GGX PDF.
                            // Reading only level zero made small HDR emitters turn
                            // into isolated noisy dots in rough reflections.
                            value += source.sample(kernel, l, q.lod) * w;
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
                for q in &kernel.diffuse {
                    let l = t * q.x + b * q.y + n * q.z;
                    value += source.sample(kernel, l, q.lod);
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
    fn cooperative_probe_matches_synchronous_radiance_and_mips() {
        let scene = IndoorManifest::generate_with_humans(
            3,
            super::super::super::layout::IndoorLayout::Mixed,
            0.3,
            0,
            0.,
        )
        .unwrap();
        let mut images = Assets::<Image>::default();
        let mut materials = Assets::<StandardMaterial>::default();
        let set = IndoorMaterials::build(&scene, &mut images, &mut materials);
        let transport = BakeScene::from_manifest(&scene, &set, &materials, &images);
        let origins = [
            transport.reflection_origin(&scene),
            Vec3::new(0.5, 1.3, 0.8),
            Vec3::new(-1., 1.8, -0.5),
        ];
        let batched = trace_many(&scene, &transport, &origins);
        for (origin, actual) in origins.into_iter().zip(batched) {
            assert_eq!(
                actual,
                trace_at(&scene, &transport, origin),
                "probe/face order and radiance must be exact"
            );
        }
        let expected = build(&scene, &transport);
        let actual = bevy::tasks::block_on(build_async(&scene, &transport));
        for (expected, actual) in expected.into_iter().zip(actual) {
            assert_eq!(expected.texture_descriptor, actual.texture_descriptor);
            assert_eq!(expected.data, actual.data);
        }
    }
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

#[cfg(test)]
mod replay_tests;
