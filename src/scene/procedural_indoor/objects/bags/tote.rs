//! An open sewn carrier. The outside, lining, folded binding and handle roots
//! use the same panel stations, so an oblique view can see the actual bag floor.
use super::*;
use crate::scene::procedural_indoor::materials::variants::object_slot;
use serde::Serialize;

const STATIONS: usize = 24;
const LEVELS: [f32; 5] = [0., 0.14, 0.60, 0.90, 1.];
type Ring = [Vec3; STATIONS];

mod pocket;
#[cfg(test)]
mod tests;

#[derive(Debug, Serialize)]
pub(crate) struct ToteCut {
    pub mouth_depth_m: f32,
    pub gusset_m: f32,
    pub lip_sag_m: [f32; 2],
    lip_cant_m: f32,
    hem_m: f32,
    base_m: f32,
    pub draped_handles: [bool; 2],
    handle_offset: [f32; 2],
    pub lining: Surface,
    pub slip_pocket: bool,
    contrast_binding: bool,
}

impl ToteCut {
    pub fn sample(o: &IndoorObject, p: &BagProgram) -> Option<Self> {
        if p.style != BagStyle::Tote {
            return None;
        }
        let mut r = stream(o.seed, 0x544f_5445_4355_5453);
        if !r.random_bool(0.82) {
            return None;
        }
        let s = o.size;
        let h = s.y * p.body_height;
        let mut dye = stream(o.finish_seed.unwrap_or(o.seed), 0x544f_5445_4c49_4e45);
        Some(Self {
            mouth_depth_m: s.z * r.random_range(0.28..0.83),
            gusset_m: s.x * r.random_range(0.025..0.136),
            lip_sag_m: [0; 2].map(|_| h * r.random_range(0.030..0.17)),
            lip_cant_m: h * r.random_range(-0.060..0.060),
            hem_m: r.random_range(0.0016_f32..0.0035).min(s.z * 0.016),
            base_m: r.random_range(0.003..0.009),
            draped_handles: [0; 2].map(|_| r.random_bool(0.51)),
            handle_offset: [0; 2].map(|_| r.random_range(-0.12..0.12)),
            lining: if dye.random_bool(0.27) {
                p.finish
            } else if p.finish == Surface::FabricAlt {
                Surface::Fabric
            } else {
                Surface::FabricAlt
            },
            slip_pocket: p.flap,
            contrast_binding: dye.random_bool(0.44),
        })
    }

    pub fn clear_mouth_depth_m(&self) -> f32 {
        self.mouth_depth_m - 2. * self.hem_m
    }

    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject, p: &BagProgram) {
        let shell = Shell::new(self, o.size, p);
        let l = o.kind.class_name();
        let lining_slot = object_slot(
            self.lining,
            o.finish_seed.unwrap_or(o.seed) ^ 0x00b0_770a,
            None,
        );
        let lining_label = format!("{l}#finish{lining_slot}");
        let trim = if self.contrast_binding {
            Surface::Leather
        } else {
            p.finish
        };
        shell.skin(a.part(p.finish, l), false);
        shell.skin(a.part(self.lining, &lining_label), true);
        shell.binding(a.part(trim, l));
        if self.slip_pocket {
            pocket::build(a.part(self.lining, &lining_label), &shell, o.size, p);
        }
        for face in 0..2 {
            let grip = shell.handle(self, o.size, p, face);
            details::strap(
                a.part(trim, l),
                &grip.points,
                grip.width,
                grip.thickness,
                Vec3::Z,
                true,
            );
        }
    }
}

/// Each broad panel has eight intervals. The four intervals of each narrow end
/// turn into the inward gusset. Vertices on the six sewn creases have separate normals.
struct Shell {
    outside: [Ring; LEVELS.len()],
    inside: [Ring; LEVELS.len()],
    along: [f32; STATIONS + 1],
}

impl Shell {
    fn new(c: &ToteCut, s: Vec3, p: &BagProgram) -> Self {
        let height = s.y * p.body_height;
        let outside = LEVELS.map(|t| {
            let half_x = s.x * 0.46 * p.taper.lerp(1., t);
            let half_z = (s.z * 0.34 * p.fullness).lerp(c.mouth_depth_m * 0.5, t)
                + s.z * 0.055 * 4. * t * (1. - t);
            let broad = |u: f32, face: usize| {
                let sign = if face == 0 { 1. } else { -1. };
                let corner = ((u.abs() - 0.25) / 0.75).max(0.);
                Vec3::new(
                    u * half_x,
                    t * height + t * t * (c.lip_cant_m * u - c.lip_sag_m[face] * (1. - u * u)),
                    sign * half_z * (1. - 0.19 * corner.powi(3)),
                )
            };
            std::array::from_fn(|i| match i {
                0..=8 => broad(1. - i as f32 / 4., 0),
                12..=20 => broad((i - 12) as f32 / 4. - 1., 1),
                _ => {
                    let (sign, cross) = if i < 12 {
                        (-1., (10. - i as f32) * 0.5)
                    } else {
                        (1., (i as f32 - 22.) * 0.5)
                    };
                    let mut point = broad(sign, 1).lerp(broad(sign, 0), (1. + cross) * 0.5);
                    let crease = 1. - cross.abs();
                    point.x += sign * crease * (s.x * 0.023 * (1. - t) - c.gusset_m * t * t);
                    point.y -= crease * height * 0.026 * t * t;
                    point
                }
            })
        });
        let inside = std::array::from_fn(|i| {
            let ring = &outside[i];
            // Offset each sewn edge in its own normal direction. Scaling the
            // two axes separately can flip the binding over an inward gusset.
            let inward: [Vec2; STATIONS] = std::array::from_fn(|j| {
                (ring[(j + 1) % STATIONS] - ring[j]).xz().normalize().perp()
            });
            std::array::from_fn(|j| {
                let a = inward[(j + STATIONS - 1) % STATIONS];
                let b = inward[j];
                let offset = (a + b) * (c.hem_m / (1. + a.dot(b)));
                ring[j] + Vec3::new(offset.x, c.base_m * (1. - LEVELS[i]), offset.y)
            })
        });
        let mut along = [0.; STATIONS + 1];
        for i in 0..STATIONS {
            along[i + 1] = along[i] + outside[2][i].distance(outside[2][(i + 1) % STATIONS]);
        }
        Self {
            outside,
            inside,
            along,
        }
    }

    fn skin(&self, g: &mut Geometry, inner: bool) {
        let rings = if inner { &self.inside } else { &self.outside };
        let sign = if inner { -1. } else { 1. };
        for (first, last) in [(0, 8), (8, 10), (10, 12), (12, 20), (20, 22), (22, 24)] {
            let start = g.positions.len() as u32;
            let stride = (last - first + 1) as u32;
            let mut length = [0.; STATIONS];
            for (row, ring) in rings.iter().enumerate() {
                for j in first..=last {
                    let i = j % STATIONS;
                    let v = ring[i];
                    let up =
                        rings[(row + 1).min(LEVELS.len() - 1)][i] - rings[row.saturating_sub(1)][i];
                    let across = ring[(j + 1).min(last) % STATIONS]
                        - ring[j.saturating_sub(1).max(first) % STATIONS];
                    if row > 0 {
                        length[i] += v.distance(rings[row - 1][i]);
                    }
                    g.positions.push(v.to_array());
                    g.normals
                        .push((up.cross(across).normalize() * sign).to_array());
                    g.uvs.push([self.along[j], length[i]]);
                    if row + 1 < LEVELS.len() && j < last {
                        let a = start + row as u32 * stride + (j - first) as u32;
                        let b = a + stride;
                        g.indices.extend(if inner {
                            [a, b + 1, b, a, a + 1, b + 1]
                        } else {
                            [a, b, b + 1, a, b + 1, a + 1]
                        });
                    }
                }
            }
        }
        // Cut a separate flat sole/liner floor, with a planar texture chart.
        let bottom = &rings[0];
        let start = g.positions.len() as u32;
        for v in std::iter::once(Vec3::Y * bottom[0].y).chain(bottom.iter().copied()) {
            g.positions.push(v.to_array());
            g.normals.push((Vec3::NEG_Y * sign).to_array());
            g.uvs.push(v.xz().to_array());
        }
        for i in 0..STATIONS {
            let a = start + 1 + i as u32;
            let b = start + 1 + ((i + 1) % STATIONS) as u32;
            g.indices
                .extend(if inner { [start, b, a] } else { [start, a, b] });
        }
    }

    fn binding(&self, g: &mut Geometry) {
        let top = LEVELS.len() - 1;
        for i in 0..STATIONS {
            let j = (i + 1) % STATIONS;
            let p = [
                self.outside[top][i],
                self.inside[top][i],
                self.inside[top][j],
                self.outside[top][j],
            ];
            let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize();
            let w0 = p[0].distance(p[1]);
            let w1 = p[2].distance(p[3]);
            g.vertex_quad(
                p,
                [n; 4],
                [
                    [self.along[i], 0.],
                    [self.along[i], w0],
                    [self.along[i + 1], w1],
                    [self.along[i + 1], 0.],
                ],
            );
        }
    }

    /// Same diagonal as the indexed outer cut. Sewn roots and resting handles
    /// can land between stations without floating above a curved panel.
    fn panel(rings: &[Ring; LEVELS.len()], face: usize, u: f32, t: f32) -> Vec3 {
        let col = if face == 0 {
            (1. - u) * 4.
        } else {
            (u + 1.) * 4.
        };
        let j = (col as usize).min(7);
        let w = col - j as f32;
        let j = j + if face == 0 { 0 } else { 12 };
        let row = LEVELS[1..]
            .partition_point(|&l| l < t)
            .min(LEVELS.len() - 2);
        let rise = (t - LEVELS[row]) / (LEVELS[row + 1] - LEVELS[row]);
        let [lo, hi] = [&rings[row], &rings[row + 1]];
        if rise >= w {
            lo[j] + (hi[j] - lo[j]) * rise + (hi[j + 1] - hi[j]) * w
        } else {
            lo[j] + (lo[j + 1] - lo[j]) * w + (hi[j + 1] - lo[j + 1]) * rise
        }
    }

    fn handle(&self, c: &ToteCut, size: Vec3, p: &BagProgram, face: usize) -> Grip {
        let height = p.body_height * size.y;
        let drop = p.handle_drop * size.y;
        let span = size.x * p.handle_span;
        let rise = drop.min(height * 0.62);
        let half = span * 0.5;
        let width = p
            .strap_width
            .min(size.x * 0.075)
            .min((rise * rise / half).min(half * half / rise) * 0.75);
        let thickness = 0.0025;
        let sign = if face == 0 { 1. } else { -1. };
        let center = c.handle_offset[face];
        let points = std::array::from_fn(|i| {
            let phase = PI * i as f32 / 16.;
            let (sn, cs) = phase.sin_cos();
            let u = center + cs * half / self.outside[3][0].x;
            let anchor = Self::panel(&self.outside, face, u, 0.9);
            let dy = rise * sn;
            let t = if c.draped_handles[face] {
                0.9 - dy / height
            } else {
                (0.9 + dy / height).min(1.)
            };
            let v = Self::panel(&self.outside, face, u, t);
            let y = if c.draped_handles[face] {
                v.y
            } else {
                v.y.max(anchor.y + dy)
            };
            // At both roots the webbing straddles the outside panel. Lift only
            // the free span, leaving the bag cavity and its mouth unobstructed.
            Vec3::new(
                v.x,
                y,
                v.z + sign * (0.0003 + sn * (thickness + size.z * 0.018)),
            )
        });
        Grip {
            points,
            width,
            thickness,
        }
    }
}

struct Grip {
    points: [Vec3; 17],
    width: f32,
    thickness: f32,
}
