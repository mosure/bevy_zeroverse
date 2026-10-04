//! Dimensioned shoe lasts and separate soles. Built once in Anny's rest frame,
//! then attached rigidly to each foot; toe bones cannot stretch the shoe upper.
use super::{HumanAssembly, HumanSurface};
use crate::scene::procedural_indoor::{geometry::Geometry, layout::stream};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts::{FRAC_PI_2, TAU};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct FootwearProgram {
    pub sole_height: f32,
    pub toe_room: f32,
    pub toe_roundness: f32,
    pub collar_raise: f32,
    pub heel_width: f32,
    pub vamp: f32,
    pub laces: u8,
    pub leather: f32,
    pub upper_color: [f32; 3],
    pub sole_color: [f32; 3],
}
impl Default for FootwearProgram {
    fn default() -> Self {
        Self {
            sole_height: 0.015,
            toe_room: 0.008,
            toe_roundness: 3.1,
            collar_raise: 0.006,
            heel_width: 0.82,
            vamp: 0.45,
            laces: 5,
            leather: 0.7,
            upper_color: [0.08, 0.07, 0.06],
            sole_color: [0.10, 0.09, 0.08],
        }
    }
}
impl FootwearProgram {
    pub fn validate(&self) -> Result<(), String> {
        for (value, lo, hi) in [
            (self.sole_height, 0.008, 0.04),
            (self.toe_room, 0.001, 0.025),
            (self.toe_roundness, 2., 5.),
            (self.collar_raise, 0., 0.08),
            (self.heel_width, 0.6, 1.),
            (self.vamp, 0., 1.),
            (self.leather, 0., 1.),
        ] {
            if !value.is_finite() || !(lo..=hi).contains(&value) {
                return Err("invalid shoe last or finish parameter".into());
            }
        }
        if self.laces > 8
            || self
                .upper_color
                .iter()
                .chain(&self.sole_color)
                .any(|v| !v.is_finite() || !(0.0..=1.).contains(v))
        {
            return Err("invalid shoe detail count or color".into());
        }
        Ok(())
    }
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x53484f454c415354);
        let leather: f32 = rng.random_range(0.0..1.0);
        let value: f32 = if rng.random_bool(0.78) {
            rng.random_range(0.025..0.32)
        } else {
            rng.random_range(0.52..0.85)
        };
        let warm = rng.random_range(0.0..0.30);
        let rubber: f32 = if leather < 0.45 && rng.random_bool(0.75) {
            rng.random_range(0.60..0.88)
        } else {
            rng.random_range(0.025..0.15)
        };
        Self {
            sole_height: rng.random_range(0.010..0.026),
            toe_room: rng.random_range(0.005..0.016),
            toe_roundness: rng.random_range(2.3..4.2),
            collar_raise: rng.random_range(0.0..0.050),
            heel_width: rng.random_range(0.72..0.95),
            vamp: rng.random_range(0.25..0.75),
            laces: if rng.random_bool(0.72) {
                rng.random_range(3..8)
            } else {
                0
            },
            leather,
            upper_color: [value, value * (1. - warm), value * (1. - warm * 1.5)],
            sole_color: [rubber, rubber * 0.98, rubber * 0.95],
        }
    }
}

#[derive(Clone, Copy)]
struct Section {
    y: f32,
    centre: f32,
    radii: Vec2,
    heel: f32,
}
impl Section {
    fn point(self, angle: f32, roundness: f32) -> Vec3 {
        let power = 2. / roundness;
        let x = angle.cos().signum() * angle.cos().abs().powf(power);
        let z = angle.sin().signum() * angle.sin().abs().powf(power);
        let heel = 1. - (1. - self.heel) * angle.sin().max(0.);
        Vec3::new(
            x * self.radii.x * heel,
            self.y,
            self.centre + z * self.radii.y,
        )
    }
    fn lerp(self, next: Self, t: f32) -> Self {
        Self {
            y: self.y + (next.y - self.y) * t,
            centre: self.centre + (next.centre - self.centre) * t,
            radii: self.radii.lerp(next.radii, t),
            heel: self.heel + (next.heel - self.heel) * t,
        }
    }
}

/// The source points are foot/ankle vertices only, in the unscaled Anny frame.
pub(super) struct FootFrame {
    pub ankle: Vec3,
    pub toe: Vec3,
    pub posed_from_rest: Mat4,
    pub floor: f32,
    pub shoe_top: f32,
}

pub(super) fn append(
    program: &FootwearProgram,
    points: &[Vec3],
    frame: FootFrame,
    mesh: &mut HumanAssembly,
) {
    let FootFrame {
        ankle,
        toe,
        posed_from_rest,
        floor,
        shoe_top,
    } = frame;
    let back = (ankle - toe).with_y(0.).normalize_or(Vec3::Z);
    let right = Vec3::Y.cross(back);
    let basis = Mat4::from_cols(
        right.extend(0.),
        Vec3::Y.extend(0.),
        back.extend(0.),
        ankle.extend(1.),
    );
    let inverse = basis.inverse();
    let local: Vec<_> = points
        .iter()
        .map(|p| inverse.transform_point3(*p))
        .collect();
    let (lo, hi) = local
        .iter()
        .copied()
        .filter(|p| p.y <= shoe_top - ankle.y)
        .fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), p| (lo.min(p), hi.max(p)),
        );
    if !lo.is_finite() || !hi.is_finite() {
        return;
    }
    let width = lo.x.abs().max(hi.x.abs()) + 0.005;
    let front = lo.z - program.toe_room;
    let back = hi.z + 0.006;
    let centre = (front + back) * 0.5;
    let radii = Vec2::new(width, (back - front) * 0.5);
    let sole_top = lo.y + program.sole_height;
    let collar_top = (shoe_top - ankle.y + program.collar_raise).max(sole_top + 0.075);
    let height = collar_top - sole_top;
    let collar = local
        .iter()
        .filter(|p| p.y > collar_top - 0.025)
        .fold(Vec2::new(width * 0.50, width * 0.55), |radius, p| {
            radius.max(Vec2::new(p.x.abs() + 0.004, p.z.abs() + 0.004))
        });
    let profiles = [
        Section {
            y: sole_top,
            centre,
            radii,
            heel: program.heel_width,
        },
        Section {
            y: sole_top + height * 0.24,
            centre,
            radii: radii * Vec2::new(0.99, 0.99),
            heel: program.heel_width,
        },
        Section {
            y: sole_top + height * (0.43 + program.vamp * 0.14),
            centre: centre * 0.75,
            radii: radii * Vec2::new(0.97, 0.77),
            heel: program.heel_width,
        },
        Section {
            y: sole_top + height * 0.80,
            centre: centre * 0.20,
            radii: radii.lerp(collar, 0.70),
            heel: 0.70 + program.heel_width * 0.30,
        },
        Section {
            y: collar_top,
            centre: 0.,
            radii: collar,
            heel: 1.,
        },
        Section {
            y: collar_top,
            centre: 0.,
            radii: collar - Vec2::splat(0.003),
            heel: 1.,
        },
        Section {
            y: collar_top - 0.018,
            centre: 0.,
            radii: collar - Vec2::splat(0.003),
            heel: 1.,
        },
    ];
    let sole_profile = [
        Section {
            y: lo.y,
            centre,
            radii: radii * 0.94,
            heel: program.heel_width,
        },
        Section {
            y: lo.y + 0.0025,
            centre,
            radii: radii * 1.014,
            heel: program.heel_width,
        },
        Section {
            y: sole_top - 0.002,
            centre,
            radii: radii * 1.014,
            heel: program.heel_width,
        },
        profiles[0],
    ];
    let mut sole = loft(&sole_profile, program.toe_roundness, 1);
    cap(&mut sole, sole_profile[0], program.toe_roundness, false);
    cap(&mut sole, sole_profile[3], program.toe_roundness, true);
    let mut upper = loft(&profiles, program.toe_roundness, 3);
    let mut laces = Geometry::default();
    for row in 0..program.laces {
        let t = (row as f32 + 1.) / (program.laces as f32 + 1.);
        let at = t * 1.7;
        let section = if at < 1. {
            profiles[2].lerp(profiles[3], at)
        } else {
            profiles[3].lerp(profiles[4], at - 1.)
        };
        let points: Vec<_> = (0..=8)
            .map(|step| {
                let angle = -FRAC_PI_2 + (step as f32 / 8. - 0.5) * 0.85;
                section.point(angle, program.toe_roundness)
                    + Vec3::new(angle.cos(), 0.2, angle.sin()) * 0.0018
            })
            .collect();
        laces.tube(&points, 0.0012, 6);
    }
    let transform = posed_from_rest * basis;
    let normal = Mat3::from_mat4(transform).inverse().transpose();
    for g in [&mut upper, &mut sole, &mut laces] {
        for p in &mut g.positions {
            let q = transform.transform_point3(Vec3::from_array(*p)) - Vec3::Y * floor;
            *p = q.with_y(q.y.max(0.)).to_array();
        }
        for n in &mut g.normals {
            *n = (normal * Vec3::from_array(*n))
                .normalize_or(Vec3::Y)
                .to_array();
        }
    }
    mesh.part(HumanSurface::Shoes).append(upper);
    mesh.part(HumanSurface::Sole).append(sole);
    mesh.part(HumanSurface::ShoeDetail).append(laces);
}

const SIDES: usize = 40;
fn loft(profiles: &[Section], roundness: f32, subdivisions: usize) -> Geometry {
    let mut g = Geometry::default();
    for pair in profiles.windows(2) {
        let base = g.positions.len() as u32;
        for row in 0..=subdivisions {
            let section = pair[0].lerp(pair[1], row as f32 / subdivisions as f32);
            for side in 0..=SIDES {
                // Exactly welded wrap coordinates, including noninteger powers.
                let a = (side % SIDES) as f32 * TAU / SIDES as f32;
                let p = section.point(a, roundness);
                g.positions.push(p.to_array());
                g.normals.push([0.; 3]);
                g.uvs.push([a * section.radii.x, section.y]);
            }
        }
        for row in 0..subdivisions {
            for side in 0..SIDES {
                let a = base + (row * (SIDES + 1) + side) as u32;
                let b = a + (SIDES + 1) as u32;
                g.indices.extend([a, b, a + 1, a + 1, b, b + 1]);
            }
        }
        // Each construction crease has its own averaged shading field.
        for ids in g
            .indices
            .iter()
            .skip(g.indices.len() - subdivisions * SIDES * 6)
            .copied()
            .collect::<Vec<_>>()
            .as_chunks::<3>()
            .0
        {
            let [a, b, c] = ids.map(|i| Vec3::from_array(g.positions[i as usize]));
            let n = (b - a).cross(c - a);
            for &i in ids {
                g.normals[i as usize] = (Vec3::from_array(g.normals[i as usize]) + n).to_array();
            }
        }
        for row in 0..=subdivisions {
            let first = base as usize + row * (SIDES + 1);
            let n = Vec3::from_array(g.normals[first]) + Vec3::from_array(g.normals[first + SIDES]);
            g.normals[first] = n.to_array();
            g.normals[first + SIDES] = n.to_array();
        }
    }
    for n in &mut g.normals {
        *n = Vec3::from_array(*n).normalize_or(Vec3::Y).to_array();
    }
    g
}
fn cap(g: &mut Geometry, section: Section, roundness: f32, top: bool) {
    let base = g.positions.len() as u32;
    g.positions.push([0., section.y, section.centre]);
    g.normals
        .push((if top { Vec3::Y } else { Vec3::NEG_Y }).to_array());
    g.uvs.push([0., 0.]);
    for i in 0..SIDES {
        let p = section.point(i as f32 * TAU / SIDES as f32, roundness);
        g.positions.push(p.to_array());
        g.normals.push(g.normals[base as usize]);
        g.uvs.push([p.x, p.z]);
    }
    for i in 0..SIDES as u32 {
        let a = base + 1 + i;
        let b = base + 1 + (i + 1) % SIDES as u32;
        g.indices
            .extend(if top { [base, b, a] } else { [base, a, b] });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn shoe_programs_replay_and_solids_have_outward_finite_triangles() {
        let points = [Vec3::new(-0.045, 0., -0.18), Vec3::new(0.045, 0.09, 0.04)];
        for seed in 0..64 {
            let p = FootwearProgram::sample(seed);
            assert_eq!(p, FootwearProgram::sample(seed));
            assert!(p.validate().is_ok());
            let mut mesh = HumanAssembly::default();
            append(
                &p,
                &points,
                FootFrame {
                    ankle: Vec3::new(0., 0.09, 0.),
                    toe: Vec3::new(0., 0.015, -0.14),
                    posed_from_rest: Mat4::IDENTITY,
                    floor: 0.,
                    shoe_top: 0.105,
                },
                &mut mesh,
            );
            for (surface, g) in &mesh.parts {
                for ids in g.indices.as_chunks::<3>().0 {
                    let [a, b, c] = ids.map(|i| Vec3::from_array(g.positions[i as usize]));
                    let n = (b - a).cross(c - a);
                    assert!(
                        n.is_finite() && n.length_squared() > 1e-16,
                        "collapsed {surface:?}"
                    );
                    for &i in ids {
                        assert!(
                            n.dot(Vec3::from_array(g.normals[i as usize])) >= -1e-10,
                            "inverted {surface:?}"
                        );
                    }
                }
                assert!(g.indices.len() / 3 < 4000, "unbounded shoe complexity");
            }
        }
    }
    #[test]
    fn malformed_shoe_programs_are_rejected_before_geometry() {
        let mut p = FootwearProgram {
            toe_roundness: f32::NAN,
            ..Default::default()
        };
        assert!(p.validate().is_err());
        p = FootwearProgram {
            laces: 200,
            ..Default::default()
        };
        assert!(p.validate().is_err());
    }
}
