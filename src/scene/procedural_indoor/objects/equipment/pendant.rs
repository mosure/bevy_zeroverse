//! Hanging luminaires. The suspension supports a shade and exactly one optical
//! assembly. Geometry, photometry and mounting checks use the installed dimensions.
use super::super::luminaires::{self, ShadeProgram};
use super::*;
use crate::scene::procedural_indoor::photometry::FixtureBeam;
mod optics;
pub use optics::{BulbEnvelope, Optic, PendantLens};
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Suspension {
    Flex,
    Downrod,
    ThreeWire,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Canopy {
    pub radius_m: f32,
    pub height_m: f32,
    /// The raised fixing boss bears on the ceiling. The skirt clears a slight slope.
    pub pad_radius_m: f32,
    pub pad_height_m: f32,
}

impl Canopy {
    pub(crate) fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"RoseSize"));
        let radius_m = r
            .random_range(0.031_f32..0.082)
            .min(o.size.x.min(o.size.z) * 0.27);
        let height_m = (radius_m * r.random_range(0.28..0.66)).min(o.size.y * 0.14);
        Self {
            radius_m,
            height_m,
            pad_radius_m: (radius_m * r.random_range(0.25..0.41)).min(0.020),
            pad_height_m: height_m * r.random_range(0.21..0.38),
        }
    }
}

#[derive(Debug, serde::Serialize)]
pub struct PendantProgram {
    pub shade: ShadeProgram,
    pub shade_radius_m: f32,
    pub shade_height_m: f32,
    pub lip_y_m: f32,
    pub taper: f32,
    pub lens: PendantLens,
    pub suspension: Suspension,
    pub canopy: Canopy,
    pub lead_radius_m: f32,
    pub spider_phase_rad: f32,
    pub beam_inner_rad: f32,
    pub beam_outer_rad: f32,
}

pub fn parameters(o: &IndoorObject) -> PendantProgram {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"PendForm"));
    let mut stock = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"PendCoat"),
    );
    let mut shade = luminaires::parameters(o.seed);
    shade.trim = [Surface::Chrome, Surface::Metal, Surface::Metal][stock.random_range(0..3)];
    shade.finish = match shade.shape {
        3 => Surface::FabricAlt,
        4 => shade.trim,
        _ => [
            Surface::Metal,
            Surface::Metal,
            Surface::Ceramic,
            Surface::FabricAlt,
            Surface::WoodEdge,
        ][stock.random_range(0..5)],
    };
    let radius = o.size.x.min(o.size.z) * r.random_range(0.38..0.48);
    let height = (o.size.y * r.random_range(0.20..0.52)).min(radius * r.random_range(0.72..1.95));
    let taper = r.random_range(0.42..0.98);
    let opening = luminaires::shade_opening(&shade, radius, radius);
    let (lip, lens) = PendantLens::sample(o, &shade, radius, height, taper);
    let suspension = match r.random_range(0..10) {
        0..=4 => Suspension::Flex,
        5..=7 => Suspension::Downrod,
        _ => Suspension::ThreeWire,
    };
    let lead_radius_m = if suspension == Suspension::Downrod {
        r.random_range(0.0045_f32..0.009)
    } else {
        r.random_range(0.0020_f32..0.0043)
    }
    .min(radius * 0.09);
    let center_recess = (lens.bottom_y_m + lens.rise_m * 0.5 - lip).max(0.);
    let cutoff = if shade.shape == 4 {
        FRAC_PI_2
    } else {
        opening.atan2(center_recess)
    };
    let spread = match lens.optic {
        Optic::Recessed => r.random_range(0.78_f32..1.30),
        Optic::OpalBowl => r.random_range(1.18..1.49),
        Optic::OpenBulb(_) => r.random_range(1.00..1.48),
    };
    let outer = spread.min(cutoff);
    PendantProgram {
        shade,
        shade_radius_m: radius,
        shade_height_m: height,
        taper,
        lip_y_m: lip,
        lens,
        canopy: Canopy::sample(o),
        suspension,
        lead_radius_m,
        spider_phase_rad: r.random_range(0.0..TAU),
        beam_inner_rad: outer * r.random_range(0.43..0.72),
        beam_outer_rad: outer,
    }
}

impl PendantProgram {
    pub(crate) fn diffuser_fraction(&self) -> Option<f32> {
        match self.lens.optic {
            Optic::OpenBulb(_) => None,
            Optic::Recessed | Optic::OpalBowl => Some(self.lens.radius_m / self.shade_radius_m),
        }
    }

    pub(crate) fn beam(&self, o: &IndoorObject) -> FixtureBeam {
        let tf = o.transform();
        FixtureBeam {
            position: tf
                .transform_point(Vec3::Y * (self.lip_y_m.min(self.lens.bottom_y_m) - 0.003)),
            direction: tf.rotation * Vec3::NEG_Y,
            radius: self.lens.radius_m * 0.5,
            inner: self.beam_inner_rad,
            outer: self.beam_outer_rad,
        }
    }
}

// The ends of the eight-sided stock terminate inside a socket, hub or ring.
// Smooth vertex normals give the wires a round highlight.
fn wire(g: &mut Geometry, start: Vec3, end: Vec3, radius: f32) {
    let delta = end - start;
    g.lathe(
        &[(radius, 0.), (radius, delta.length())],
        8,
        Transform::from_translation(start)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, delta.normalize())),
    );
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let label = o.kind.class_name();
    let top = p.lip_y_m + p.shade_height_m;
    let r = p.shade_radius_m;
    luminaires::shade(
        a,
        label,
        &p.shade,
        Vec3::Y * p.lip_y_m,
        r,
        p.shade_height_m,
        p.taper,
    );
    let opening = luminaires::shade_opening(&p.shade, r, r);
    let socket = p
        .lens
        .build(a, label, p.shade.trim, opening, p.lip_y_m, top);
    let upper = luminaires::upper_radius(&p.shade, r, p.taper);
    let wall = luminaires::wall_thickness(r);
    let seat = (upper + luminaires::shade_opening(&p.shade, r, upper)) * 0.5;
    let c = p.canopy;
    let bottom = o.size.y - c.height_m;
    let shoulder = o.size.y - c.pad_height_m;
    let cap = a.part(p.shade.trim, label);
    // Open at the fixing plane; the mounting surface closes the shoe in a room.
    cap.lathe(
        &[
            (0., bottom),
            (c.radius_m * 0.82, bottom),
            (c.radius_m, bottom + c.height_m * 0.16),
            (c.radius_m, shoulder),
            (c.pad_radius_m, shoulder),
            (c.pad_radius_m, o.size.y),
        ],
        24,
        Transform::IDENTITY,
    );
    let joint_y = top - wall * 0.5;
    let hub = (p.lead_radius_m * 2.1).min(upper * 0.31);
    cap.lathe(
        &[
            (0., joint_y - hub * 0.62),
            (hub, joint_y - hub * 0.62),
            (hub, top + hub * 0.50),
            (p.lead_radius_m, top + hub * 0.66),
        ],
        16,
        Transform::IDENTITY,
    );
    wire(
        cap,
        Vec3::Y * socket,
        Vec3::Y * joint_y,
        p.lead_radius_m.min(hub * 0.75),
    );
    let spoke_radius = (wall * 0.42).min(0.0028);
    for i in 0..3 {
        let (s, c0) = (p.spider_phase_rad + TAU * i as f32 / 3.).sin_cos();
        let radial = Vec3::new(c0, 0., s);
        // Meet the rolled rim on a chord of its actual 32-sided stock.
        let end = Vec3::Y * joint_y
            + radial * seat * optics::rim_scale(p.spider_phase_rad + TAU * i as f32 / 3.);
        wire(cap, Vec3::Y * joint_y, end, spoke_radius);
        if p.suspension == Suspension::ThreeWire {
            wire(
                cap,
                end,
                Vec3::Y * (bottom + c.height_m * 0.06) + radial * c.radius_m * 0.62,
                spoke_radius,
            );
        }
    }
    let lead = if p.suspension == Suspension::Downrod {
        p.shade.trim
    } else {
        Surface::Rubber
    };
    wire(
        a.part(lead, label),
        Vec3::Y * (top + hub * 0.5),
        Vec3::Y * (bottom + c.height_m * 0.06),
        p.lead_radius_m,
    );
}
