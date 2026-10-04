//! Continuous groom coordinates, relief and closed, curved tied-hair bundles.
use super::{HairProgram, HairStyle, IndoorHuman};
use bevy::prelude::*;
use noise::NoiseFn;
use std::f32::consts::TAU;

pub(super) struct Groom {
    pub centre: Vec3,
    pub size: Vec3,
    pub(super) length: f32,
    pub(super) part: f32,
    pub(super) curl: f32,
    style: u8,
    pub(super) volume: f32,
    pub(super) program: HairProgram,
    hairline: f32,
    clusters: noise::OpenSimplex,
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
            volume: h.appearance.as_ref().map_or(1.0, |a| a.hair_volume),
            program: h
                .appearance
                .as_ref()
                .map_or_else(HairProgram::default, |a| a.hair_program.clone()),
            hairline: h.appearance.as_ref().map_or(0.0, |a| a.hairline_raise),
            clusters: noise::OpenSimplex::new((h.seed ^ (h.seed >> 32)) as u32),
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
        let irregular = self.noise(Vec3::new(phi.sin(), 0.2, phi.cos()), 7.) * 0.003;
        // Lower nape, recessed temples and a shallow widow's peak instead of a
        // horizontal boundary across the occiput.
        let line = self.centre.y
            + self.size.y
                * (-0.20 + front * front * 0.55 + side * side * 0.24 + 0.075 * side * front
                    - 0.05 * front * (1.0 - side).powi(4))
            + irregular
            + self.hairline * (0.35 + front * 0.65)
            + if self.style == HairStyle::Pixie as u8 {
                front * 0.013 - side * 0.014
            } else {
                0.
            };
        let field = p.y - line;
        let edge_width = if self.style == HairStyle::Afro as u8 {
            0.04
        } else {
            0.014
        };
        let taper = (field / edge_width).clamp(0.0, 1.0);
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
        let style = HairStyle::from_id(self.style).expect("validated groom");
        let clipped = match style {
            HairStyle::Buzz => 0.09,
            HairStyle::Pixie => 0.40,
            s if s.falls() => 0.30,
            _ => 1.,
        };
        let swept = 0.65 + 0.30 * (local.x * self.part).clamp(-1.0, 1.0);
        let side_taper = if matches!(
            style,
            HairStyle::SidePart | HairStyle::Swept | HairStyle::Pixie
        ) {
            0.25 + 0.75 * crown.powf(0.7)
        } else {
            1.
        };
        let lift = if style == HairStyle::Swept {
            front * crown * 0.018 * self.volume
        } else {
            0.
        };
        let bulk = 0.0015
            + self.length * self.volume * clipped * (0.10 + 0.23 * crown) * swept * side_taper
            + lift;
        // Smooth aggregate locks alter the silhouette without alpha noise,
        // repeating bands or seams at the poles of the cranial surface.
        let cluster = self.clusters.get([
            local.x as f64 * 5.2,
            local.y as f64 * 5.2,
            local.z as f64 * 5.2,
        ]) as f32
            * 0.5
            + 0.5;
        let curl_mass = self.length
            * self.volume
            * self.curl
            * clipped
            * if self.style == 4 { 0.45 } else { 0.08 };
        let clumps = (flow * 19.0 + theta.sin() * self.curl * 5.0).cos();
        let wave = (flow * 11.0 + theta * 7.0).sin() * (theta * 9.0).sin();
        let relief =
            (0.00010 + self.curl * 0.00045) * clipped * clumps * (theta / 0.24).clamp(0.0, 1.0)
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
        let afro = if style == HairStyle::Afro {
            // Dense, rounded volume with multiscale curl relief. Sparse ring
            // attachments looked disconnected from the mass at the silhouette.
            (0.05 + 0.04 * self.volume) * (0.45 + crown * 0.55)
                + (cluster - 0.5) * 0.004
                + self.noise(local, 21.) * 0.0015
                + self.noise(local, 57.) * 0.0006
        } else {
            0.
        };
        let height =
            0.0012 + taper * (bulk + afro + curl_mass * cluster + relief - part).max(0.0005);
        (
            height,
            field,
            // Combing from a part runs across the cranial arc. This chart has
            // a regular tangent frame at the crown; longitude UVs have a pole
            // there and caused a bright anisotropic zigzag through the hair.
            Vec2::new(
                (local.z + local.x * self.program.sweep * 0.08) * self.size.z * 0.5,
                (local.x - self.part * 0.4).atan2(local.y + 0.05) * self.size.x * 0.5,
            ),
        )
    }
    pub(super) fn noise(&self, p: Vec3, frequency: f64) -> f32 {
        self.clusters.get([
            p.x as f64 * frequency,
            p.y as f64 * frequency,
            p.z as f64 * frequency,
        ]) as f32
    }
}
