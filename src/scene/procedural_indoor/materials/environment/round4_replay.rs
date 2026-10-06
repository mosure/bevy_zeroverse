//! Frozen accepted round4 convolution oracle. Do not share optimized stencil code.
use super::*;

pub(super) fn round4_convolve(pixels: &[Vec3], size: u32) -> [Image; 2] {
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
