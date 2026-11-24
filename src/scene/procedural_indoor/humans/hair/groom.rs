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
    style: HairStyle,
    pub(super) volume: f32,
    pub(super) program: HairProgram,
    pub(super) parting: Option<super::cut::Part>,
    hairline: f32,
    clusters: noise::OpenSimplex,
}
impl Groom {
    pub fn new(h: &IndoorHuman, lo: Vec3, hi: Vec3) -> Self {
        let style = HairStyle::from_id(h.hairstyle).expect("validated groom");
        let program = h
            .appearance
            .as_ref()
            .map_or_else(HairProgram::default, |a| a.hair_program.clone());
        let part = h.appearance.as_ref().map_or(0.2, |a| a.hair_part);
        let parting = program.scalp_cut.and_then(|cut| {
            let width = cut.part_width(style);
            (width > 0.).then_some(super::cut::Part {
                width,
                crown: cut.part_end,
                start_x: part * 0.76,
                curve: program.part_curve,
            })
        });
        Self {
            centre: (lo + hi) * 0.5,
            size: hi - lo,
            length: h.appearance.as_ref().map_or(0.015, |a| a.hair_length),
            part,
            curl: h.appearance.as_ref().map_or(0.0, |a| a.hair_curl / 0.06),
            style,
            volume: h.appearance.as_ref().map_or(1.0, |a| a.hair_volume),
            program,
            parting,
            hairline: h.appearance.as_ref().map_or(0.0, |a| a.hairline_raise),
            clusters: noise::OpenSimplex::new((h.seed ^ (h.seed >> 32)) as u32),
        }
    }
    pub fn uv_period(&self) -> f32 {
        TAU * self.size.x * 0.5
    }
    pub(super) fn part_coordinates(&self, p: Vec3) -> Vec2 {
        self.parting.map_or(Vec2::ZERO, |cut| {
            cut.coordinates((p - self.centre) / (self.size * 0.5), self.size.x * 0.5)
        })
    }
    pub fn hairline_field(&self, p: Vec3) -> f32 {
        let local = (p - self.centre) / (self.size * 0.5);
        p.y - self.hairline_height(local, (-local.z).clamp(0., 1.), local.x.abs().min(1.))
    }

    fn hairline_height(&self, local: Vec3, front: f32, side: f32) -> f32 {
        let phi = local.x.atan2(local.z);
        let irregular = self.noise(Vec3::new(phi.sin(), 0.2, phi.cos()), 7.) * 0.003;
        // Lower nape, recessed temples and a shallow widow's peak instead of a
        // horizontal boundary across the occiput.
        self.centre.y
            + self.size.y
                * (-0.20 + front * front * 0.55 + side * side * 0.24 + 0.075 * side * front
                    - 0.05 * front * (1.0 - side).powi(4))
            + irregular
            + self.hairline * (0.35 + front * 0.65)
            + if self.style == HairStyle::Pixie {
                front * 0.013 - side * 0.014
            } else {
                0.
            }
            + self.cut_contour(local)
            + self
                .program
                .scalp_cut
                .map_or(0., |cut| cut.hairline_offset(local, front, side))
    }

    /// Offset in metres, signed hairline distance, and fibre-aligned metric UVs.
    pub fn sample(&self, p: Vec3) -> (f32, f32, Vec2) {
        let local = (p - self.centre) / (self.size * 0.5);
        let front = (-local.z).clamp(0.0, 1.0);
        let side = local.x.abs().clamp(0.0, 1.0);
        let field = p.y - self.hairline_height(local, front, side);
        // This chart is regular at the crown and periodic across the nape.
        let mut uv = Vec2::new(
            (local.z + local.x * self.program.sweep * 0.08) * self.size.z * 0.5,
            (local.x - self.part * 0.4).atan2(local.y + 0.05) * self.size.x * 0.5,
        );
        if self.style.coiled() {
            return super::coils::sample(self, self.style, p, local, field, uv);
        }
        let taper = (field / 0.014).clamp(0.0, 1.0);
        let taper = taper * taper * (3.0 - 2.0 * taper);
        if self.style == HairStyle::Buzz {
            if let Some(cut) = self.program.scalp_cut {
                let crown = local.y.clamp(0., 1.);
                let grain = self.noise(local, 5.2);
                let nap = (0.36 + 0.53 * crown + 0.08 * grain) * cut.clip_length_m;
                // A clipper leaves short nap at the temples and nape. Preserve
                // the scalp silhouette and comb around the crown.
                uv.x += self.size.x * 0.12 * local.x * (0.40 + crown * self.program.crown_whorl);
                return (0.0008 + taper * nap, field, uv);
            }
        }
        // Comb from a displaced crown/part. Metric meridians supply fibre UVs
        // rather than stretching a planar stripe across the entire skull.
        let direction = (local - Vec3::new(self.part * 0.65, 0.0, 0.22)).normalize_or(Vec3::Y);
        let theta = direction.y.clamp(-1.0, 1.0).acos();
        let longitude = direction.x.atan2(direction.z);
        let flow = longitude
            + self.part * theta * 0.65
            + self.program.crown_whorl * (-theta * theta / 0.35).exp()
            + self.curl * 0.08 * (theta * 8.0 + (longitude * 3.0).sin()).sin();
        let crown = local.y.clamp(0.0, 1.0);
        let style = self.style;
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
        let curl_mass = self.length * self.volume * self.curl * clipped * 0.08;
        let clumps =
            (flow * 19.0 + theta.sin() * self.curl * 5.0 + self.noise(local, 3.7) * 1.3).cos();
        let relief =
            (0.00010 + self.curl * 0.00045) * clipped * clumps * (theta / 0.24).clamp(0.0, 1.0);
        let combed = if matches!(
            style,
            HairStyle::SidePart | HairStyle::Swept | HairStyle::Pixie
        ) {
            // Scissor-cut locks rise from the part in broad sections. Keep the
            // sides close to the ear and let the longer sections carry the crest.
            let breadth = (0.5 + 0.5 * clumps).powi(2);
            self.length.min(0.085)
                * (0.035 + 0.038 * self.program.layers)
                * breadth
                * (0.24 + 0.76 * crown)
                * (0.5 + 0.5 * cluster)
                * (theta / 0.34).clamp(0., 1.)
        } else {
            0.
        };
        let part_distance =
            (local.x - self.part * 0.65 - self.program.part_curve * local.z * 0.16).abs();
        let part = if matches!(style, HairStyle::SidePart | HairStyle::Swept) {
            (-part_distance * part_distance / 0.0018).exp() * crown * bulk * 0.55
        } else {
            0.0
        };
        let mut height =
            0.0012 + taper * (bulk + curl_mass * cluster + relief + combed - part).max(0.0005);
        if let Some(cut) = self.parting {
            let bank = cut.coordinates(local, self.size.x * 0.5);
            let active = 2. * bank.y / cut.width;
            let distance = (bank.x.abs() - bank.y).max(0.);
            let root = super::cut::smooth(distance / (0.007 + cut.width));
            height = 0.0012 + (height - 0.0012) * (1. - active * (1. - root));
            // The combed fibres turn away from the part on each side. Smooth
            // the bend across its centre so finite mip footprints stay bounded.
            let span = (bank.x * bank.x + 0.000025).sqrt() - 0.005;
            uv.x += span * (0.40 + self.program.sweep * 0.21) * active;
        }
        (height, field, uv)
    }

    fn cut_contour(&self, local: Vec3) -> f32 {
        if !matches!(
            self.style,
            HairStyle::TightCurls | HairStyle::SidePart | HairStyle::Swept | HairStyle::Pixie
        ) {
            return 0.;
        }
        let lobe = |x: f32| (1. - x * x).max(0.).powi(2);
        let side = local.x.abs().min(1.).powi(2);
        // Cut back over the ear, retain a tapered sideburn just in front and a
        // longer centre at the nape. These are scalp coordinates on the fitted head.
        let ear = lobe((local.z - 0.08) / 0.40) * side;
        let sideburn = lobe((local.z + 0.34) / 0.28) * side;
        let nape = local.z.clamp(0., 1.).powi(2) * lobe(local.x / 0.92);
        (0.009 + 0.010 * self.program.layers) * ear
            - (0.005 + 0.012 * self.program.spread / 1.4) * sideburn
            - (0.011 + 0.020 * self.program.layers) * nape
    }
    pub(super) fn noise(&self, p: Vec3, frequency: f64) -> f32 {
        self.clusters.get([
            p.x as f64 * frequency,
            p.y as f64 * frequency,
            p.z as f64 * frequency,
        ]) as f32
    }
}
