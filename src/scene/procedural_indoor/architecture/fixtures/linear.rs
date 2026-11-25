//! Linear luminaires fitted to the existing ceiling reservations. Tube counts,
//! lampholders and reflectors share a section; a wraparound has a continuous lens.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;
mod mesh;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum LinearForm {
    BareStrip,
    WingReflector,
    OpalWrap,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct LinearFixture {
    pub form: LinearForm,
    /// Number of independently visible emitting strips. An opal lens is one strip.
    pub strips: usize,
    pub strip_width_m: f32,
    pub pitch_m: f32,
    pub lit_length_m: f32,
    pub body_length_m: f32,
    pub pan_width_m: f32,
    pub housing: Surface,
    pub trim: Surface,
    end_m: f32,
    top_y: f32,
    pan_bottom_y: f32,
    light_bottom_y: f32,
    reflector_half_m: f32,
    reflector_edge_y: f32,
    flat_fraction: f32,
    hanger_fraction: f32,
    along_z: bool,
}

impl LinearFixture {
    pub(crate) fn fit(
        scene: &IndoorManifest,
        shape: FixtureShape,
        at: Option<Vec3>,
        s: Vec3,
    ) -> Option<Self> {
        if shape != FixtureShape::Tube || scene.generator_version < 247 {
            return None;
        }
        let program = scene.program.as_ref()?;
        // A shared purchase commonly covers the whole building. Other fit-outs have
        // a different product in a separate zone; fittings within a zone stay a set.
        let mut r = stream(scene.seed, u64::from_le_bytes(*b"StripSku"));
        if r.random_bool(0.48) {
            if let Some(zone) = at.and_then(|p| {
                program
                    .zones
                    .iter()
                    .position(|z| p.xz().cmpge(z.min).all() && p.xz().cmple(z.max).all())
            }) {
                r.set_stream(u64::from_le_bytes(*b"ZoneTube") + zone as u64);
            }
        }
        let form = match r.random_range(0..10) {
            0..4 => LinearForm::BareStrip,
            4..7 => LinearForm::WingReflector,
            _ => LinearForm::OpalWrap,
        };
        let width = s.x.min(s.z);
        let body_length_m = s.x.max(s.z) * r.random_range(0.87..0.98);
        let usable = width * r.random_range(0.76..0.96);
        let end_m = r.random_range(0.014_f32..0.032).min(body_length_m * 0.10);
        let lit_length_m = body_length_m - 2. * end_m;
        let top_y = s.y * 0.47;
        let pan_bottom_y = s.y * r.random_range(0.13..0.30);
        let light_bottom_y = -s.y * r.random_range(0.35..0.45);
        let (strips, strip_width_m, pitch_m, pan_width_m) = if form == LinearForm::OpalWrap {
            let lens = usable.min(r.random_range(0.078..0.31)) * r.random_range(0.75..0.92);
            (1, lens, 0., (lens + 0.014).min(width * 0.98))
        } else {
            // Narrow, conventional and large-diameter stock; socket cheeks and
            // electrical separation must fit even in a very narrow reservation.
            let nominal: f32 = [0.016, 0.026, 0.038][r.random_range(0..3)];
            let diameter = (nominal * r.random_range(0.94_f32..1.06))
                .min((pan_bottom_y - light_bottom_y) * 0.68)
                .min(usable * 0.39);
            let socket_width = diameter * 1.34;
            let min_pitch = socket_width + 0.010;
            let pan_limit = if form == LinearForm::WingReflector {
                usable * 0.76
            } else {
                usable
            };
            let capacity = (1. + ((pan_limit - socket_width) / min_pitch).floor()).max(1.) as usize;
            let requested = [1, 1, 2, 2, 2, 3, 4][r.random_range(0..7)];
            let count = requested.min(capacity);
            let stock_pitch = r.random_range(min_pitch..min_pitch + 0.114);
            let pitch = if count > 1 {
                stock_pitch.min((pan_limit - socket_width) / (count - 1) as f32)
            } else {
                0.
            };
            let bank = (count - 1) as f32 * pitch + socket_width;
            (
                count,
                diameter,
                pitch,
                (bank + r.random_range(0.008..0.031)).min(pan_limit),
            )
        };
        Some(Self {
            form,
            strips,
            strip_width_m,
            pitch_m,
            lit_length_m,
            body_length_m,
            pan_width_m,
            housing: [Surface::Paint, Surface::Metal, Surface::Plastic][r.random_range(0..3)],
            trim: match form {
                LinearForm::WingReflector => {
                    [Surface::Paint, Surface::Chrome, Surface::Metal][r.random_range(0..3)]
                }
                _ => [Surface::Plastic, Surface::Ceramic, Surface::Metal][r.random_range(0..3)],
            },
            end_m,
            top_y,
            pan_bottom_y,
            light_bottom_y,
            reflector_half_m: (usable * 0.5).min(pan_width_m * 0.5 + r.random_range(0.022..0.10)),
            reflector_edge_y: light_bottom_y + 0.002_f32.min(s.y * 0.04),
            flat_fraction: r.random_range(0.15..0.73),
            hanger_fraction: r.random_range(0.24..0.42),
            along_z: s.z > s.x,
        })
    }

    pub fn projected_area_m2(&self) -> f32 {
        self.strips as f32 * self.strip_width_m * self.lit_length_m
    }

    fn along(&self) -> Vec3 {
        if self.along_z {
            Vec3::Z
        } else {
            Vec3::X
        }
    }

    pub(super) fn anchors(&self) -> [Vec3; 2] {
        [-1., 1.].map(|side| self.along() * (side * self.body_length_m * self.hanger_fraction))
    }

    /// Approximate the reflector's transverse cutoff with one axisymmetric Bevy spot.
    /// The longitudinal ends are open. The emitted lumens are normalized separately.
    pub(super) fn beam(&self, broad: f32) -> f32 {
        if self.form != LinearForm::WingReflector {
            return broad;
        }
        let axis_y = self.light_bottom_y + self.strip_width_m * 0.5;
        let across = self.reflector_half_m + (self.strips - 1) as f32 * self.pitch_m * 0.5;
        let cutoff = across.atan2((axis_y - self.reflector_edge_y).max(0.));
        0.5 * (broad + broad.min(cutoff))
    }
}
