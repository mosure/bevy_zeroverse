//! Installed geometry for human-powered and motorized cardio stations.
//! Local +Z is the user's forward axis for both products and their floor clearances.
use super::*;
use crate::scene::procedural_indoor::materials::appliance_panel::{Film, Glyph};
mod cycle;
mod treadmill;

#[derive(Serialize)]
pub(crate) enum Equipment {
    Cycle(CycleFit),
    Treadmill(BeltFit),
}

#[derive(Serialize)]
pub(crate) struct CycleFit {
    rear_drive: bool,
    radius: f32,
    saddle_height: f32,
    saddle_setback: f32,
    bar_span: f32,
    crank_angle: f32,
    readout: Option<Readout>,
    frame: Surface,
}

#[derive(Serialize)]
pub(crate) struct BeltFit {
    rise: f32,
    belt_width: f32,
    console_width: f32,
    screen_tilt: f32,
    readout: Readout,
    grip_reach: f32,
    frame: Surface,
}

pub(crate) fn parameters(o: &IndoorObject) -> Equipment {
    let mut shape = stream(o.seed, u64::from_le_bytes(*b"CardioFt"));
    let mut finish = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"FrameLot"),
    );
    let frame = [Surface::Metal, Surface::Accent, Surface::Paint][finish.random_range(0..3)];
    match o.kind {
        ObjectKind::StationaryBike => Equipment::Cycle(CycleFit {
            rear_drive: finish.random_bool(0.42),
            radius: finish.random_range(0.175..0.238),
            saddle_height: o.size.y * shape.random_range(0.68..0.78),
            saddle_setback: o.size.z * shape.random_range(0.13..0.20),
            bar_span: o.size.x * finish.random_range(0.63..0.89),
            crank_angle: shape.random_range(0.0..std::f32::consts::TAU),
            readout: finish
                .random_bool(0.78)
                .then(|| Readout::sample(&mut shape, false)),
            frame,
        }),
        ObjectKind::Treadmill => Equipment::Treadmill(BeltFit {
            rise: shape.random_range(0.009..0.072),
            belt_width: o.size.x * shape.random_range(0.60..0.72),
            console_width: o.size.x * shape.random_range(0.68..0.93),
            screen_tilt: shape.random_range(0.35..0.77),
            readout: Readout::sample(&mut shape, true),
            grip_reach: o.size.z * shape.random_range(0.16..0.28),
            frame,
        }),
        _ => unreachable!("expected cardio equipment"),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match parameters(o) {
        Equipment::Cycle(fit) => cycle::build(a, o, fit),
        Equipment::Treadmill(fit) => treadmill::build(a, o, fit),
    }
}

#[derive(Clone, Copy, Serialize)]
struct Readout {
    film: Film,
    seconds: u16,
}

impl Readout {
    fn sample(r: &mut impl Rng, powered: bool) -> Self {
        let film = if r.random_bool(0.32) {
            if powered {
                Film::DarkWindow
            } else {
                Film::BlankLcd
            }
        } else if powered {
            Film::Led
        } else {
            Film::ReflectiveLcd
        };
        Self {
            film,
            seconds: if r.random_bool(0.60) {
                0
            } else {
                r.random_range(1..3600)
            },
        }
    }

    /// Opaque tiled readout facing the operator. The timer shares the existing
    /// filtered instrument atlas across every machine and digit in a room.
    fn build(self, a: &mut Assembly, label: &str, screen: Transform, size: Vec2) {
        let face = a.part(Surface::Ink, &self.film.label(label));
        let mins = (self.seconds / 60) as u8;
        let secs = (self.seconds % 60) as u8;
        let timer = [
            Glyph::Number(mins / 10),
            Glyph::Number(mins % 10),
            Glyph::Colon,
            Glyph::Number(secs / 10),
            Glyph::Number(secs % 10),
        ];
        let edges = [-0.5, -0.28, -0.06, 0.06, 0.28, 0.5];
        for (glyph, e) in timer.into_iter().zip(edges.windows(2)) {
            let corners = [(e[0], -0.5), (e[1], -0.5), (e[1], 0.5), (e[0], 0.5)];
            // Looking along +Z, the operator's right lies on local -X.
            face.vertex_quad(
                corners
                    .map(|(x, z)| screen.transform_point(Vec3::new(-x * size.x, 0., z * size.y))),
                [screen.rotation * Vec3::Y; 4],
                glyph.uvs(),
            );
        }
    }
}

pub(crate) fn display_film(o: &IndoorObject) -> Option<Film> {
    match parameters(o) {
        Equipment::Cycle(fit) => fit.readout.map(|r| r.film),
        Equipment::Treadmill(fit) => Some(fit.readout.film),
    }
}
