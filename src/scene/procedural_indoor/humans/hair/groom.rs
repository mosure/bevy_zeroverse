//! Continuous groom coordinates, relief and closed, curved tied-hair bundles.
use super::{Geometry, IndoorHuman};
use bevy::prelude::*;
use std::f32::consts::{PI, TAU};

pub(super) struct Groom {
    pub centre: Vec3,
    pub size: Vec3,
    length: f32,
    part: f32,
    curl: f32,
    style: u8,
}
impl Groom {
    pub fn new(h: &IndoorHuman, lo: Vec3, hi: Vec3) -> Self {
        Self {
            centre: (lo + hi) * 0.5,
            size: hi - lo,
            length: h.appearance.as_ref().map_or(0.015, |a| a.hair_length),
            part: h.appearance.as_ref().map_or(0.2, |a| a.hair_part),
            curl: h.appearance.as_ref().map_or(0.0, |a| a.hair_curl / 0.06),
            style: h.hairstyle,
        }
    }
    pub fn uv_period(&self) -> f32 {
        TAU * self.size.x * 0.5
    }
    /// Offset in metres, signed hairline distance, and fibre-aligned metric UVs.
    pub fn sample(&self, p: Vec3) -> (f32, f32, Vec2) {
        let local = (p - self.centre) / (self.size * 0.5);
        let front = (-local.z).clamp(0.0, 1.0);
        let side = local.x.abs().clamp(0.0, 1.0);
        let phi = local.x.atan2(local.z);
        let irregular =
            (phi * 31.0 + 0.6 * (phi * 7.0).sin()).sin() * 0.0011 + (phi * 67.0).sin() * 0.00035;
        // Lower nape, recessed temples and a shallow widow's peak instead of a
        // horizontal boundary across the occiput.
        let line = self.centre.y
            + self.size.y
                * (-0.20 + front * front * 0.55 + side * side * 0.24 + 0.075 * side * front
                    - 0.025 * front * (1.0 - side).powi(4))
            + irregular
            - if self.style == 2 {
                self.length * 0.40 * (1.0 - front)
            } else {
                0.0
            };
        let field = p.y - line;
        let taper = (field / 0.014).clamp(0.0, 1.0);
        let taper = taper * taper * (3.0 - 2.0 * taper);
        // Comb from a displaced crown/part. Metric meridians supply fibre UVs
        // rather than stretching a planar stripe across the entire skull.
        let direction = (local - Vec3::new(self.part * 0.65, 0.0, 0.22)).normalize_or(Vec3::Y);
        let theta = direction.y.clamp(-1.0, 1.0).acos();
        let longitude = direction.x.atan2(direction.z);
        let flow = longitude
            + self.part * theta * 0.65
            + self.curl * 0.08 * (theta * 8.0 + (longitude * 3.0).sin()).sin();
        let crown = local.y.clamp(0.0, 1.0);
        let clipped = if self.style == 0 { 0.16 } else { 1.0 };
        let swept = 0.65 + 0.30 * (local.x * self.part).clamp(-1.0, 1.0);
        let bulk = 0.002 + self.length * clipped * (0.12 + 0.28 * crown) * swept;
        let clumps = (flow * 54.0 + theta.sin() * self.curl * 5.0).cos();
        let wave = (flow * 17.0 + theta * 16.0).sin() * (theta * 21.0).sin();
        let relief =
            (0.00010 + self.curl * 0.00035) * clipped * clumps * (theta / 0.24).clamp(0.0, 1.0)
                + if self.style == 4 {
                    self.curl * 0.00065 * wave
                } else {
                    0.0
                };
        let part_distance = (local.x - self.part * 0.65).abs();
        let part = if self.style == 1 || self.style == 3 {
            (-part_distance * part_distance / 0.0018).exp() * crown * bulk * 0.55
        } else {
            0.0
        };
        let height = 0.0008 + taper * (bulk + relief - part).max(0.0005);
        (
            height,
            field,
            Vec2::new(flow * self.size.x * 0.5, theta * self.size.y * 0.5),
        )
    }
    pub fn bundle(&self, bun: bool, origin: Vec3, rotation: Quat, g: &mut Geometry) {
        let length = 0.10 + self.length * 1.45;
        let width = (self.size.x * 0.11 + self.length * 0.055).clamp(0.013, 0.026);
        let centre = |t: f32| {
            if bun {
                let angle = TAU * t * 2.1;
                let r = (PI * t).sin().max(0.0) * width * 1.15;
                Vec3::new(r * angle.sin(), r * angle.cos() + 0.008, 0.006 + t * 0.026)
            } else {
                Vec3::new(
                    self.part * 0.014 * t * t,
                    -length * t,
                    0.008 + 0.045 * (PI * t * 0.72).sin() + self.curl * 0.012 * (TAU * t).sin(),
                )
            }
        };
        const RINGS: u32 = 32;
        const SIDES: u32 = 48;
        let base = g.positions.len() as u32;
        let mut distance = 0.0;
        let mut previous = centre(0.0);
        for row in 0..=RINGS {
            let t = row as f32 / RINGS as f32;
            let p = centre(t);
            distance += p.distance(previous);
            previous = p;
            let tangent = (centre((t + 0.001).min(1.0)) - centre((t - 0.001).max(0.0))).normalize();
            let x = (Vec3::X - tangent * tangent.x).normalize_or(Vec3::Z);
            let y = tangent.cross(x);
            let radius = if bun {
                width * 0.64 * (1.0 - 0.7 * t.powi(8))
            } else {
                width * (0.72 + 0.35 * (PI * t).sin()) * (1.0 - 0.96 * t.powf(1.8))
            };
            for side in 0..=SIDES {
                let a = TAU * side as f32 / SIDES as f32;
                let radial = x * a.cos() + y * a.sin();
                let relief =
                    1.0 + 0.045 * (a * 16.0 + t * 4.0).cos() + 0.018 * (a * 23.0 - t * 7.0).sin();
                g.positions
                    .push((origin + rotation * (p + radial * radius * relief)).to_array());
                g.normals.push((rotation * radial).to_array());
                g.uvs.push([a * width, distance]);
            }
        }
        for row in 0..RINGS {
            for side in 0..SIDES {
                let a = base + row * (SIDES + 1) + side;
                let b = a + SIDES + 1;
                g.indices.extend([a, a + 1, b, b, a + 1, b + 1]);
            }
        }
        // Opaque closures at the scalp-embedded root and tapered tip.
        for row in [0, RINGS] {
            let ring = base + row * (SIDES + 1);
            let cap = g.positions.len() as u32;
            let t = row as f32 / RINGS as f32;
            g.positions.push((origin + rotation * centre(t)).to_array());
            let n = (centre((t + 0.001).min(1.0)) - centre((t - 0.001).max(0.0))).normalize();
            g.normals
                .push((rotation * n * if row == 0 { -1.0 } else { 1.0 }).to_array());
            g.uvs.push([0.0, t * distance]);
            for side in 0..SIDES {
                if row == 0 {
                    g.indices.extend([cap, ring + side + 1, ring + side]);
                } else {
                    g.indices.extend([cap, ring + side, ring + side + 1]);
                }
            }
        }
    }
}
