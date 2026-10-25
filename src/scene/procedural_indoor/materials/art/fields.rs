//! Coherent pigment and contour fields. Random controls are sampled on a small
//! lattice once; smooth interpolation drives large-scale composition, never
//! independent per-pixel noise. Bounded work is shared by prints and screens.
use super::*;
const SIDE: usize = 17;

#[derive(Clone, Debug, serde::Serialize)]
pub struct FieldProgram {
    pub frequency: f32,
    pub warp: f32,
    pub contrast: f32,
    pub bands: u32,
    pub stroke_density: u32,
    pub palette_offset: f32,
}
impl FieldProgram {
    pub(super) fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x5049474d454e54);
        Self {
            frequency: rng.random_range(1.3..5.5),
            warp: rng.random_range(0.0..0.8),
            contrast: rng.random_range(0.65..2.0),
            bands: rng.random_range(3..=14),
            stroke_density: rng.random_range(28..=100),
            palette_offset: rng.random_range(0.0..1.0),
        }
    }
}
struct Field {
    values: [[f32; SIDE]; SIDE],
}
impl Field {
    fn new(seed: u64, p: &FieldProgram) -> Self {
        let mut rng = stream(seed, 0x4649454c44);
        let controls: [[f32; 9]; 9] =
            std::array::from_fn(|_| std::array::from_fn(|_| rng.random_range(0.0..1.0)));
        let interpolate = |u: f32, v: f32| {
            let x = u.clamp(0., 7.999);
            let y = v.clamp(0., 7.999);
            let i = x as usize;
            let j = y as usize;
            let f = |a: f32| a * a * (3. - 2. * a);
            let a = f(x.fract());
            let b = f(y.fract());
            (controls[j][i] * (1. - a) + controls[j][i + 1] * a) * (1. - b)
                + (controls[j + 1][i] * (1. - a) + controls[j + 1][i + 1] * a) * b
        };
        let values = std::array::from_fn(|y| {
            std::array::from_fn(|x| {
                let u = x as f32 / (SIDE - 1) as f32;
                let v = y as f32 / (SIDE - 1) as f32;
                let dx = (interpolate(1. + u * 2., 1. + v * 2.) - 0.5) * p.warp;
                let dy = (interpolate(4. + u * 2., 4. + v * 2.) - 0.5) * p.warp;
                let n = interpolate(1. + u * p.frequency + dx, 1. + v * p.frequency + dy);
                0.5 + 0.5 * ((n - 0.5) * 2. * p.contrast).tanh()
            })
        });
        Self { values }
    }
    fn at(&self, u: f32, v: f32) -> f32 {
        let x = u.clamp(0., 0.99999) * (SIDE - 1) as f32;
        let y = v.clamp(0., 0.99999) * (SIDE - 1) as f32;
        let i = x as usize;
        let j = y as usize;
        // Cubic interpolation keeps contour tangents continuous across the
        // lattice. Bilinear derivatives left a visible rectangular patchwork.
        let cubic = |v: [f32; 4], t: f32| {
            v[1] + 0.5
                * t
                * (v[2] - v[0]
                    + t * (2. * v[0] - 5. * v[1] + 4. * v[2] - v[3]
                        + t * (3. * (v[1] - v[2]) + v[3] - v[0])))
        };
        let rows = std::array::from_fn(|row| {
            let yy = (j as isize + row as isize - 1).clamp(0, (SIDE - 1) as isize) as usize;
            let values = std::array::from_fn(|col| {
                let xx = (i as isize + col as isize - 1).clamp(0, (SIDE - 1) as isize) as usize;
                self.values[yy][xx]
            });
            cubic(values, x.fract())
        });
        cubic(rows, y.fract())
    }
}

pub(super) fn paint(c: &mut Canvas, bounds: [i32; 4], seed: u64, p: &ArtComposition, style: u32) {
    let [x, y, w, h] = bounds;
    let field = Field::new(seed, &p.field);
    let color = |t: f32| {
        let q = (t.clamp(0., 0.99999) * 3. + p.field.palette_offset).min(3.9999);
        let i = q as usize;
        let a = q.fract();
        std::array::from_fn(|k| {
            (p.palette[i][k] as f32 * (1. - a) + p.palette[i + 1][k] as f32 * a) as u8
        })
    };
    if style != 3 {
        for yy in 0..h {
            for xx in 0..w {
                let u = xx as f32 / w as f32;
                let v = yy as f32 / h as f32;
                let value = field.at(u, v);
                let q = value * p.field.bands as f32;
                let rgb = if style == 1 {
                    color(q.floor() / p.field.bands as f32)
                } else {
                    color(value)
                };
                if style == 2 {
                    let edge = q.fract().min(1. - q.fract());
                    c.blend(
                        x + xx,
                        y + yy,
                        p.palette[1],
                        ((0.09 - edge) * 12.).clamp(0., 1.),
                    );
                } else {
                    c.rect(x + xx, y + yy, 1, 1, rgb);
                }
            }
        }
    } else {
        // Streamlines follow the field gradient. They share the underlying
        // composition but have unequal starts, lengths and pen pressures.
        let mut rng = stream(seed, 0x464c4f57);
        for _ in 0..p.field.stroke_density {
            let mut u = rng.random_range(0.02..0.98);
            let mut v = rng.random_range(0.02..0.98);
            let length = rng.random_range(12..=64);
            let ink = color(rng.random_range(0.2..1.0));
            let dir = if rng.random_bool(0.5) { 1. } else { -1. };
            for _ in 0..length {
                let dx = field.at(u + 0.01, v) - field.at(u - 0.01, v);
                let dy = field.at(u, v + 0.01) - field.at(u, v - 0.01);
                let norm = (dx * dx + dy * dy).sqrt().max(0.0001);
                let a = [x as f32 + u * w as f32, y as f32 + v * h as f32];
                u += dir * (-dy / norm) * 0.008;
                v += dir * (dx / norm) * 0.008;
                if !(0.01..0.99).contains(&u) || !(0.01..0.99).contains(&v) {
                    break;
                }
                c.line(
                    a,
                    [x as f32 + u * w as f32, y as f32 + v * h as f32],
                    1,
                    ink,
                );
            }
        }
    }
}
