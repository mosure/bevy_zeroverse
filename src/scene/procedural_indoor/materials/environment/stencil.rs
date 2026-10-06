//! Fixed-size reflection filtering: cache scene-independent addresses and weights,
//! then interpolate the room's HDR radiance with the original arithmetic/order.
//! One process-wide cache has bounded residency; Web keeps the smaller kernel.
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

fn filter(source: &RadianceMips, level: &Level, specular: bool, bytes: &mut Vec<u8>) {
    for (taps, denominator) in level
        .taps
        .as_chunks::<{ SAMPLES as usize }>()
        .0
        .iter()
        .zip(&level.denominators)
    {
        let mut value = Vec3::ZERO;
        for (tap, fp) in taps.iter().zip(&level.footprints) {
            let sample = tap.sample(source, fp);
            value += if specular {
                sample * tap.weight
            } else {
                sample
            };
        }
        encode(bytes, value / *denominator);
    }
}

pub(super) fn convolve(pixels: &[Vec3]) -> [Image; 2] {
    let source = RadianceMips::new(pixels, SIZE);
    let stencil = Stencil::production();
    let levels = SIZE.ilog2() + 1;
    let faces = crate::scene::procedural_indoor::preparation::workers::pool().scope(|scope| {
        for (face, kernel) in stencil.faces.iter().enumerate() {
            let source = &source;
            scope.spawn(async move {
                let mut specular = Vec::with_capacity((SIZE * SIZE * 8 * 4 / 3 + 8) as usize);
                let mut diffuse = Vec::with_capacity(16 * 16 * 8);
                for &value in
                    &pixels[face * (SIZE * SIZE) as usize..(face + 1) * (SIZE * SIZE) as usize]
                {
                    encode(&mut specular, value);
                }
                for level in &kernel.specular {
                    filter(source, level, true, &mut specular);
                }
                filter(source, &kernel.diffuse, false, &mut diffuse);
                (diffuse, specular)
            });
        }
    });
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
