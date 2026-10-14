//! Fixed-size reflection filtering: cache scene-independent addresses and weights,
//! then interpolate the room's HDR radiance with the original arithmetic/order.
//! One bounded process-wide cache is shared by native and Web preparation.
use super::*;

#[derive(Clone, Copy)]
struct Bilinear {
    addresses: [u16; 4],
    blend: Vec2,
}

impl Bilinear {
    fn new(taps: &kernel::CubeTaps, d: Vec3) -> Self {
        let (face, p) = coordinates(d, taps.size);
        let lo = p.floor();
        let stride = (taps.size + 2) as usize;
        let offset = face as usize * stride * stride
            + (lo.y as i32 + 1) as usize * stride
            + (lo.x as i32 + 1) as usize;
        Self {
            addresses: [offset, offset + 1, offset + stride, offset + stride + 1]
                .map(|i| u16::try_from(taps.addresses[i]).expect("fixed 64px cube address")),
            blend: p - lo,
        }
    }

    fn sample(&self, pixels: &[Vec3]) -> Vec3 {
        let [a, b, c, d] = self.addresses.map(|i| pixels[usize::from(i)]);
        a.lerp(b, self.blend.x)
            .lerp(c.lerp(d, self.blend.x), self.blend.y)
    }
}

struct Footprint {
    a: usize,
    b: usize,
    blend: f32,
}

impl Footprint {
    fn new(lod: f32, count: usize) -> Self {
        let lod = lod.clamp(0., (count - 1) as f32);
        let a = lod.floor() as usize;
        Self {
            a,
            b: (a + 1).min(count - 1),
            blend: lod.fract(),
        }
    }
}

struct Tap {
    a: Bilinear,
    b: Bilinear,
    weight: f32,
}

impl Tap {
    fn sample(&self, source: &RadianceMips, fp: &Footprint) -> Vec3 {
        self.a
            .sample(&source.0[fp.a].1)
            .lerp(self.b.sample(&source.0[fp.b].1), fp.blend)
    }
}

struct Level {
    footprints: Vec<Footprint>,
    taps: Vec<Tap>,
    denominators: Vec<f32>,
}

struct Face {
    specular: Vec<Level>,
    diffuse: Level,
}

struct Stencil {
    faces: Vec<Face>,
}

impl Stencil {
    fn production() -> &'static Self {
        static STENCIL: std::sync::OnceLock<Stencil> = std::sync::OnceLock::new();
        STENCIL.get_or_init(Self::new)
    }

    fn new() -> Self {
        let kernel = ConvolutionKernel::production();
        let levels = SIZE.ilog2() + 1;
        let prepare = |face, size, samples: &[kernel::Quadrature], specular| {
            let footprints: Vec<_> = samples
                .iter()
                .map(|q| Footprint::new(q.lod, levels as usize))
                .collect();
            let mut taps = Vec::with_capacity((size * size * SAMPLES) as usize);
            let mut denominators = Vec::with_capacity((size * size) as usize);
            for y in 0..size {
                for x in 0..size {
                    let n = direction(face, x as f32, y as f32, size);
                    let (t, b) = basis(n);
                    let mut weight = 0.;
                    for (q, fp) in samples.iter().zip(&footprints) {
                        let h = t * q.x + b * q.y + n * q.z;
                        let l = if specular { h * (2. * q.z) - n } else { h };
                        let w = if specular { n.dot(l).max(0.) } else { 1. };
                        taps.push(Tap {
                            a: Bilinear::new(&kernel.taps[fp.a], l),
                            b: Bilinear::new(&kernel.taps[fp.b], l),
                            weight: w,
                        });
                        weight += w;
                    }
                    denominators.push(if specular {
                        weight.max(1e-6)
                    } else {
                        SAMPLES as f32
                    });
                }
            }
            Level {
                footprints,
                taps,
                denominators,
            }
        };
        Self {
            faces: (0..6)
                .map(|face| Face {
                    specular: (1..levels)
                        .map(|level| {
                            prepare(face, SIZE >> level, &kernel.specular[level as usize], true)
                        })
                        .collect(),
                    diffuse: prepare(face, 16, &kernel.diffuse, false),
                })
                .collect(),
        }
    }
}

/// Serialize the exact fixed quadrature once. Absolute source addresses fit in
/// 16 bits across the complete 64px cubemap mip chain (32766 texels).
pub(super) fn gpu_program() -> super::gpu::Program {
    use super::gpu::{Copy, Pixel, Program, Tap as GpuTap};
    let stencil = Stencil::production();
    let mut result = Program {
        taps: Vec::new(),
        pixels: Vec::new(),
        copies: Vec::new(),
        bytes: 0,
    };
    let mut offsets = Vec::new();
    let mut offset = 0;
    for level in 0..=SIZE.ilog2() {
        offsets.push(offset);
        offset += 6 * (SIZE >> level).pow(2);
    }
    for (face, kernels) in stencil.faces.iter().enumerate() {
        for target in 0..2 {
            let levels = if target == 0 { 1 } else { SIZE.ilog2() + 1 };
            for mip in 0..levels {
                let size = if target == 0 { 16 } else { SIZE >> mip };
                let stride = (size * 8).div_ceil(256) * 256;
                let destination = result.bytes;
                result.copies.push(Copy {
                    target,
                    face: face as u32,
                    mip,
                    size,
                    offset: destination,
                    stride,
                });
                result.bytes += u64::from(stride * size);
                let level = if target == 0 {
                    Some(&kernels.diffuse)
                } else if mip > 0 {
                    Some(&kernels.specular[mip as usize - 1])
                } else {
                    None
                };
                for y in 0..size {
                    for x in 0..size {
                        let index = (y * size + x) as usize;
                        let pixel = Pixel {
                            taps: result.taps.len() as u32,
                            offset: (destination / 8) as u32 + y * (stride / 8) + x,
                            denominator: level.map_or(1., |l| l.denominators[index]),
                            kind: if target == 0 {
                                1
                            } else if mip == 0 {
                                0
                            } else {
                                2
                            },
                        };
                        let Some(level) = level else {
                            result.pixels.push(Pixel {
                                taps: face as u32 * SIZE * SIZE + y * SIZE + x,
                                ..pixel
                            });
                            continue;
                        };
                        result.pixels.push(pixel);
                        for (tap, fp) in level.taps
                            [index * SAMPLES as usize..(index + 1) * SAMPLES as usize]
                            .iter()
                            .zip(&level.footprints)
                        {
                            let packed = |a: &Bilinear, level: usize| {
                                let p = a.addresses.map(|i| u32::from(i) + offsets[level]);
                                assert!(p.iter().all(|&i| i <= u16::MAX as u32));
                                [p[0] | p[1] << 16, p[2] | p[3] << 16]
                            };
                            let a = packed(&tap.a, fp.a);
                            let b = packed(&tap.b, fp.b);
                            result.taps.push(GpuTap {
                                addresses: [a[0], a[1], b[0], b[1]],
                                blend: [tap.a.blend.x, tap.a.blend.y, tap.b.blend.x, tap.b.blend.y],
                                weight_lod: [tap.weight, fp.blend, 0., 0.],
                            });
                        }
                    }
                }
            }
        }
    }
    result
}

fn filtered_value(
    source: &RadianceMips,
    taps: &[Tap],
    footprints: &[Footprint],
    denominator: f32,
    specular: bool,
) -> Vec3 {
    let mut value = Vec3::ZERO;
    for (tap, fp) in taps.iter().zip(footprints) {
        let sample = tap.sample(source, fp);
        value += if specular {
            sample * tap.weight
        } else {
            sample
        };
    }
    value / denominator
}

fn filter(source: &RadianceMips, level: &Level, specular: bool, bytes: &mut Vec<u8>) {
    for (taps, denominator) in level
        .taps
        .as_chunks::<{ SAMPLES as usize }>()
        .0
        .iter()
        .zip(&level.denominators)
    {
        encode(
            bytes,
            filtered_value(source, taps, &level.footprints, *denominator, specular),
        );
    }
}

#[cfg(any(test, target_arch = "wasm32"))]
async fn filter_async(source: &RadianceMips, level: &Level, specular: bool, bytes: &mut Vec<u8>) {
    for (index, (taps, denominator)) in level
        .taps
        .as_chunks::<{ SAMPLES as usize }>()
        .0
        .iter()
        .zip(&level.denominators)
        .enumerate()
    {
        encode(
            bytes,
            filtered_value(source, taps, &level.footprints, *denominator, specular),
        );
        if index % 128 == 127 {
            crate::scene::procedural_indoor::preparation::cooperate().await;
        }
    }
}

#[cfg(any(test, target_arch = "wasm32"))]
pub(super) async fn convolve_async(pixels: &[Vec3]) -> [Image; 2] {
    let source = RadianceMips::new(pixels, SIZE);
    let stencil = Stencil::production();
    let mut diffuse = Vec::with_capacity(6 * 16 * 16 * 8);
    let mut specular = Vec::new();
    for (face, kernel) in stencil.faces.iter().enumerate() {
        for &value in &pixels[face * (SIZE * SIZE) as usize..(face + 1) * (SIZE * SIZE) as usize] {
            encode(&mut specular, value);
        }
        for level in &kernel.specular {
            filter_async(&source, level, true, &mut specular).await;
        }
        filter_async(&source, &kernel.diffuse, false, &mut diffuse).await;
        crate::scene::procedural_indoor::preparation::cooperate().await;
    }
    [cube(diffuse, 16, 1), cube(specular, SIZE, SIZE.ilog2() + 1)]
}

pub(super) fn convolve(pixels: &[Vec3]) -> [Image; 2] {
    let source = RadianceMips::new(pixels, SIZE);
    let stencil = Stencil::production();
    let levels = SIZE.ilog2() + 1;
    let render_face = |face: usize, kernel: &Face| {
        let mut specular = Vec::with_capacity((SIZE * SIZE * 8 * 4 / 3 + 8) as usize);
        let mut diffuse = Vec::with_capacity(16 * 16 * 8);
        for &value in &pixels[face * (SIZE * SIZE) as usize..(face + 1) * (SIZE * SIZE) as usize] {
            encode(&mut specular, value);
        }
        for level in &kernel.specular {
            filter(&source, level, true, &mut specular);
        }
        filter(&source, &kernel.diffuse, false, &mut diffuse);
        (diffuse, specular)
    };
    #[cfg(not(target_arch = "wasm32"))]
    let faces = crate::scene::procedural_indoor::preparation::workers::pool().scope(|scope| {
        for (face, kernel) in stencil.faces.iter().enumerate() {
            let render_face = &render_face;
            scope.spawn(async move { render_face(face, kernel) });
        }
    });
    #[cfg(target_arch = "wasm32")]
    let faces: Vec<_> = stencil
        .faces
        .iter()
        .enumerate()
        .map(|(face, kernel)| render_face(face, kernel))
        .collect();
    let mut diffuse = Vec::with_capacity(6 * 16 * 16 * 8);
    let mut specular = Vec::new();
    for (d, s) in faces {
        diffuse.extend(d);
        specular.extend(s);
    }
    [cube(diffuse, 16, 1), cube(specular, SIZE, levels)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cooperative_filter_preserves_all_hdr_mip_bytes() {
        let pixels: Vec<_> = (0..6 * SIZE * SIZE)
            .map(|i| {
                Vec3::new(
                    (i % 127) as f32 * 0.3,
                    (i % 39) as f32 * 12.,
                    (i % 113) as f32 * 0.001,
                )
            })
            .collect();
        let expected = convolve(&pixels);
        let actual = bevy::tasks::block_on(convolve_async(&pixels));
        for (expected, actual) in expected.into_iter().zip(actual) {
            assert_eq!(expected.texture_descriptor, actual.texture_descriptor);
            assert_eq!(expected.data, actual.data);
        }
    }

    #[test]
    fn reflection_stencil_has_fixed_bounded_residency() {
        let stencil = Stencil::production();
        let taps: usize = stencil
            .faces
            .iter()
            .flat_map(|f| f.specular.iter().chain(std::iter::once(&f.diffuse)))
            .map(|l| l.taps.len())
            .sum();
        assert_eq!(taps, 622_464);
        let storage: usize = stencil
            .faces
            .iter()
            .flat_map(|f| f.specular.iter().chain(std::iter::once(&f.diffuse)))
            .map(|l| {
                l.taps.capacity() * std::mem::size_of::<Tap>()
                    + l.footprints.capacity() * std::mem::size_of::<Footprint>()
                    + l.denominators.capacity() * std::mem::size_of::<f32>()
            })
            .sum();
        assert!(storage < 24 * 1024 * 1024);
        assert!(std::ptr::eq(stencil, Stencil::production()));
    }
}
