//! Scene-independent IBL quadrature and cube-edge addresses. The stored scalar
//! values and addresses execute the original expressions once, without changing
//! radiance samples, interpolation, world-space weights or accumulation order.
use super::{coordinates, direction, hammersley, Vec3, PI, SAMPLES, SIZE, TAU};

pub(super) struct ConvolutionKernel {
    pub specular: Vec<Vec<Quadrature>>,
    pub diffuse: Vec<Quadrature>,
    pub taps: Vec<CubeTaps>,
}

pub(super) struct Quadrature {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub lod: f32,
}

impl ConvolutionKernel {
    pub fn production() -> &'static Self {
        static KERNEL: std::sync::OnceLock<ConvolutionKernel> = std::sync::OnceLock::new();
        KERNEL.get_or_init(|| Self::new(SIZE))
    }

    pub fn new(size: u32) -> Self {
        let levels = size.ilog2() + 1;
        let texel_angle = 4. * PI / (6 * size * size) as f32;
        let specular = (0..levels)
            .map(|level| {
                // Level zero copies its source directly and has no quadrature.
                if level == 0 {
                    return Vec::new();
                }
                let roughness = level as f32 / (levels - 1) as f32;
                let alpha2 = roughness.powi(4);
                (0..SAMPLES)
                    .map(|i| {
                        let xi = hammersley(i);
                        let cos = ((1. - xi.x) / (1. + (alpha2 - 1.) * xi.x)).sqrt();
                        let sin = (1. - cos * cos).sqrt();
                        let phi = TAU * xi.y;
                        let denominator = cos * cos * (alpha2 - 1.) + 1.;
                        let pdf = alpha2 / (4. * PI * denominator * denominator);
                        Quadrature {
                            x: sin * phi.cos(),
                            y: sin * phi.sin(),
                            z: cos,
                            lod: 0.5 * (1. / (SAMPLES as f32 * pdf * texel_angle)).log2(),
                        }
                    })
                    .collect()
            })
            .collect();
        let diffuse = (0..SAMPLES)
            .map(|i| {
                let xi = hammersley(i);
                let r = xi.x.sqrt();
                let phi = TAU * xi.y;
                let pdf = (1. - xi.x).sqrt() / PI;
                Quadrature {
                    x: r * phi.cos(),
                    y: r * phi.sin(),
                    z: (1. - xi.x).sqrt(),
                    lod: 0.5 * (1. / (SAMPLES as f32 * pdf * texel_angle)).log2(),
                }
            })
            .collect();
        Self {
            specular,
            diffuse,
            taps: (0..levels)
                .map(|level| CubeTaps::new(size >> level))
                .collect(),
        }
    }
}

/// Bilinear taps are integer coordinates from -1 through size. Resolve their
/// exact original cross-face pixel address once, including edge/corner rounding.
pub(super) struct CubeTaps {
    pub(super) size: u32,
    pub(super) addresses: Vec<usize>,
}

impl CubeTaps {
    fn new(size: u32) -> Self {
        let mut addresses = Vec::with_capacity((6 * (size + 2) * (size + 2)) as usize);
        for face in 0..6 {
            for y in -1..=size as i32 {
                for x in -1..=size as i32 {
                    let (face, p) = coordinates(direction(face, x as f32, y as f32, size), size);
                    let x = p.x.round().clamp(0., (size - 1) as f32) as u32;
                    let y = p.y.round().clamp(0., (size - 1) as f32) as u32;
                    addresses.push((face * size * size + y * size + x) as usize);
                }
            }
        }
        Self { size, addresses }
    }

    pub fn sample(&self, pixels: &[Vec3], d: Vec3) -> Vec3 {
        let (face, p) = coordinates(d, self.size);
        let lo = p.floor();
        let f = p - lo;
        let stride = (self.size + 2) as usize;
        let offset = face as usize * stride * stride
            + (lo.y as i32 + 1) as usize * stride
            + (lo.x as i32 + 1) as usize;
        let tap = |offset| pixels[self.addresses[offset]];
        tap(offset).lerp(tap(offset + 1), f.x).lerp(
            tap(offset + stride).lerp(tap(offset + stride + 1), f.x),
            f.y,
        )
    }
}
