use super::{Section, SIDES};
use crate::scene::procedural_indoor::{geometry::Geometry, layout::stream};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::f32::consts::{FRAC_PI_2, TAU};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ShoeConstruction {
    /// Medial shift of the toe box and scoop of the waist, in half-widths.
    pub toe_bias: f32,
    pub instep_scoop: f32,
    /// Rise of the front tip above the ball of the foot, in rest-frame metres.
    pub toe_spring: f32,
    pub crossed: bool,
    pub apron: bool,
    pub fastening_color: [f32; 3],
}

impl ShoeConstruction {
    pub(super) fn sample(seed: u64, upper: [f32; 3], sole: [f32; 3]) -> Self {
        let mut rng = stream(seed, 0x53484f4557454c54);
        let tint = if rng.random_bool(0.63) { upper } else { sole };
        let factor = rng.random_range(0.80..1.12);
        Self {
            toe_bias: rng.random_range(0.055..0.24),
            instep_scoop: rng.random_range(0.035..0.19),
            toe_spring: rng.random_range(0.002..0.014),
            crossed: rng.random_bool(0.70),
            apron: rng.random_bool(0.67),
            fastening_color: tint.map(|v| (v * factor).clamp(0.015, 0.96)),
        }
    }
    pub(super) fn valid(&self) -> bool {
        (0.0..=0.32).contains(&self.toe_bias)
            && (0.0..=0.24).contains(&self.instep_scoop)
            && (0.0..=0.020).contains(&self.toe_spring)
            && self.fastening_color.iter().all(|v| (0.0..=1.).contains(v))
    }
}

pub(super) struct Last<'a> {
    pub construction: Option<&'a ShoeConstruction>,
    pub medial: f32,
    pub width: f32,
    pub back: f32,
    pub length: f32,
}
impl Last<'_> {
    pub fn shape(&self, mut p: Vec3) -> Vec3 {
        let Some(c) = self.construction else { return p };
        let t = ((self.back - p.z) / self.length).clamp(0., 1.);
        let toe = ((t - 0.61) / 0.39).clamp(0., 1.);
        let waist = (1. - ((t - 0.38) / 0.32).powi(2)).max(0.).powi(2);
        let inner = (p.x * self.medial / self.width).clamp(0., 1.);
        p.x += self.medial * self.width * (c.toe_bias * toe * toe - c.instep_scoop * waist * inner);
        p.y += c.toe_spring * toe * toe;
        p
    }
    /// Sole caps follow each transverse section. A fan to a single centre would
    /// lift the outsole across the ball as soon as its front tip was raised.
    pub fn cap(&self, g: &mut Geometry, section: Section, roundness: f32, top: bool) {
        let normal = if top { Vec3::Y } else { Vec3::NEG_Y };
        let point = |i: usize| {
            self.shape(section.point((i % SIDES) as f32 * TAU / SIDES as f32, roundness))
        };
        let toe = SIDES * 3 / 4;
        for row in 0..SIDES / 2 {
            let p = [
                point(toe + row),
                point(toe + row + 1),
                point(toe + SIDES - row - 1),
                point(toe + SIDES - row),
            ];
            for idx in [[0, 1, 2], [0, 2, 3]] {
                let [a, mut b, mut c] = idx.map(|i| p[i]);
                let mut n = (b - a).cross(c - a);
                if n.length_squared() <= 1e-18 {
                    continue;
                }
                if n.dot(normal) < 0. {
                    std::mem::swap(&mut b, &mut c);
                    n = -n;
                }
                let base = g.positions.len() as u32;
                g.positions.extend([a, b, c].map(|v| v.to_array()));
                g.normals.extend([n.normalize().to_array(); 3]);
                g.uvs.extend([a, b, c].map(|v| [v.x, v.z]));
                g.indices.extend([base, base + 1, base + 2]);
            }
        }
    }
    pub fn fastenings(&self, profile: &[Section], roundness: f32, rows: u8, g: &mut Geometry) {
        let section = |t: f32| {
            let at = t * 1.7;
            if at < 1. {
                profile[2].lerp(profile[3], at)
            } else {
                profile[3].lerp(profile[4], at - 1.)
            }
        };
        let point = |t: f32, side: f32, clearance: f32| {
            let a = -FRAC_PI_2 + side * 0.425;
            self.shape(section(t).point(a, roundness))
                + Vec3::new(a.cos(), 0.65, a.sin()).normalize() * clearance
        };
        if rows == 0 {
            if self.construction.is_some_and(|c| c.apron) {
                // A stitched apron follows the vamp around the toe, returning
                // towards the quarters on each side of a slip-on.
                let path: [_; 19] = std::array::from_fn(|i| {
                    let u = i as f32 / 18.;
                    let a = -FRAC_PI_2 + (u - 0.5) * 2.68;
                    let section = profile[1].lerp(profile[2], 0.67 + 0.25 * (2. * u - 1.).powi(2));
                    self.shape(section.point(a, roundness))
                        + Vec3::new(a.cos(), 0.55, a.sin()).normalize() * 0.0014
                });
                g.tube(&path, 0.0015, 5);
            }
            return;
        }
        let at = |row: u8| (row as f32 + 1.) / (rows as f32 + 1.);
        let crossed = self.construction.is_some_and(|c| c.crossed) && rows > 1;
        let mut lace = |lo: f32, hi: f32, side: f32, over: bool| {
            let path: [_; 9] = std::array::from_fn(|i| {
                let t = i as f32 / 8.;
                // Separate the two cords at their crossing; both ends meet the eyestays.
                let lift = if over {
                    0.0032 * (4. * t * (1. - t))
                } else {
                    0.
                };
                point(lo + (hi - lo) * t, side * (1. - 2. * t), 0.0017 + lift)
            });
            g.tube(&path, 0.0012, 6);
        };
        lace(at(0), at(0), 1., false);
        for row in 1..rows {
            if crossed {
                lace(at(row - 1), at(row), -1., false);
                lace(at(row - 1), at(row), 1., true);
            } else {
                lace(at(row), at(row), 1., false);
            }
        }
    }
}
