//! Tap envelopes are sampled in the vertical plane from deck mount to the selected bowl.
use super::*;
use crate::scene::procedural_indoor::materials::worktop_finish::TAP_LABEL;

#[derive(Clone, Copy, Debug, Serialize, PartialEq, Eq)]
pub enum Spout {
    Gooseneck,
    BentSquare,
    LowMixer,
    PullDown,
}

#[derive(Clone, Debug, Serialize)]
pub struct TapFitting {
    pub spout: Spout,
    pub bend_fraction: f32,
    pub head_length_m: f32,
    pub head_radius_ratio: f32,
    pub control_angle: f32,
    pub outlet_height_fraction: f32,
}

impl TapFitting {
    pub(super) fn sample(seed: u64, family: u8) -> Self {
        let mut r = stream(seed, 0x5441_5046_4954);
        let spout = match family {
            0 => Spout::Gooseneck,
            1 => Spout::BentSquare,
            2 => Spout::LowMixer,
            _ => Spout::PullDown,
        };
        let pull_down = spout == Spout::PullDown;
        Self {
            spout,
            bend_fraction: r.random_range(0.20..0.39),
            head_length_m: if pull_down {
                r.random_range(0.043..0.072)
            } else {
                r.random_range(0.011..0.024)
            },
            head_radius_ratio: if pull_down {
                r.random_range(1.40..1.70)
            } else {
                r.random_range(1.10..1.36)
            },
            control_angle: r.random_range(-0.85..0.85),
            outlet_height_fraction: r.random_range(match spout {
                Spout::LowMixer => 0.22..0.42,
                Spout::BentSquare => 0.33..0.65,
                Spout::Gooseneck | Spout::PullDown => 0.40..0.68,
            }),
        }
    }

    pub(super) fn centerline(
        &self,
        mount: Vec3,
        travel: Vec3,
        rise: f32,
        available: f32,
    ) -> Vec<Vec3> {
        let reach = travel.length();
        let forward = travel / reach;
        let at = |x, y| mount + forward * x + Vec3::Y * y;
        // Start with the usable outlet, leaving room for the spray sleeve and bend.
        // Extending a pull-down head below an independently sampled arch can
        // otherwise leave a high-arc fixture with only a few centimetres of clearance.
        let ceiling = available - 0.015;
        let minimum_arch = (0.5 * reach * 0.034).sqrt();
        let minimum_drop = if matches!(self.spout, Spout::Gooseneck | Spout::PullDown) {
            minimum_arch + 0.014
        } else {
            0.054
        };
        let end = (available * self.outlet_height_fraction + self.head_length_m)
            .min(ceiling - minimum_drop);
        let mut points = Vec::with_capacity(18);
        points.push(mount);
        match self.spout {
            Spout::BentSquare => {
                let desired = reach * self.bend_fraction;
                let height = rise.max(end + desired + 0.012).min(ceiling);
                let bend = desired.min(height - end - 0.010);
                points.push(at(0., height - bend));
                for i in 1..=4 {
                    let (sin, cos) = (i as f32 * PI / 8.).sin_cos();
                    points.push(at(bend * (1. - cos), height - bend + bend * sin));
                }
                points.push(at(reach - bend, height));
                for i in 1..=4 {
                    let (sin, cos) = (i as f32 * PI / 8.).sin_cos();
                    points.push(at(reach - bend + bend * sin, height - bend * (1. - cos)));
                }
            }
            Spout::LowMixer => {
                // Two circular elbows join a rising straight arm. A fitted cubic
                // can curl more tightly than the tube itself at the high outlet.
                let radius = (reach * self.bend_fraction).min(0.044).min(reach * 0.30);
                let run = reach - 2. * radius;
                let lift = (run * (0.15 + self.bend_fraction)).min(end * 0.48);
                let angle = lift.atan2(run);
                let heel = end + 0.010 - lift;
                points.push(at(0., heel));
                for i in 1..=5 {
                    let (sin, cos) = ((PI * 0.5 - angle) * i as f32 / 5.).sin_cos();
                    points.push(at(radius * (1. - cos), heel + radius * sin));
                }
                let (sa, ca) = angle.sin_cos();
                let x = radius * (1. - sa) + run;
                let y = heel + radius * ca + lift;
                points.push(at(x, y));
                for i in 1..=6 {
                    let (sin, cos) = (angle - (PI * 0.5 + angle) * i as f32 / 6.).sin_cos();
                    points.push(at(x + radius * (sa - sin), y + radius * (cos - ca)));
                }
            }
            Spout::Gooseneck | Spout::PullDown => {
                let desired = reach
                    * (if self.spout == Spout::PullDown {
                        0.22
                    } else {
                        0.31
                    } + self.bend_fraction);
                let desired = desired.max(minimum_arch);
                let height = rise.max(end + desired + 0.014).min(ceiling);
                let arch = desired.min(height - end - 0.012);
                let heel = height - arch;
                points.push(at(0., heel));
                for i in 1..=14 {
                    let (sin, cos) = (PI * i as f32 / 14.).sin_cos();
                    points.push(at(0.5 * reach * (1. - cos), heel + arch * sin));
                }
            }
        }
        points.push(at(reach, end));
        points
    }

    pub fn outlet(&self, p: &program::SinkProgram) -> Vec3 {
        *p.faucet_points.last().unwrap() - Vec3::Y * self.head_length_m
    }

    pub(super) fn build(&self, a: &mut Assembly, p: &program::SinkProgram, handles: u8) {
        let r = p.faucet_radius_m;
        let metal = a.part(p.faucet_finish, TAP_LABEL);
        metal.tube(&p.faucet_points, r, 16);
        let at = |v| Transform::from_translation(v);
        let deck = Vec3::new(p.mount.x, p.worktop_m, p.mount.z);
        metal.lathe(
            &[
                (2.25 * r, 0.),
                (2.25 * r, 0.003),
                (1.66 * r, 0.008),
                (1.42 * r, 0.031),
                (r, 0.036),
            ],
            20,
            at(deck),
        );

        // A real aerator recess. The tube's end cap is hidden above the insert.
        let outlet = self.outlet(p);
        let wide = r * self.head_radius_ratio;
        let h = self.head_length_m;
        metal.lathe(
            &[
                (wide * 0.74, h * 0.19),
                (wide * 0.74, 0.0015),
                (wide * 0.93, 0.),
                (wide, 0.002),
                (wide * 0.97, h * 0.58),
                (r * 1.02, h + 0.002),
                (r * 0.97, h + 0.002),
            ],
            20,
            at(outlet),
        );
        let recess = (h * 0.19).min(0.006);
        a.part(Surface::Metal, "faucet").lathe(
            &[(0., recess), (wide * 0.76, recess)],
            16,
            at(outlet),
        );

        for &h in &p.handles {
            let center = Vec3::new(h.x, p.worktop_m, h.z);
            let metal = a.part(p.faucet_finish, TAP_LABEL);
            metal.lathe(
                &[
                    (0.018, 0.),
                    (0.018, 0.004),
                    (0.012, 0.009),
                    (0.012, 0.031),
                    (0.009, 0.035),
                    (0., 0.035),
                ],
                16,
                at(center),
            );
            let side = (h.x - p.mount.x).signum();
            let dir = Vec3::new(
                side * self.control_angle.cos(),
                0.,
                self.control_angle.sin(),
            );
            let root = center + Vec3::Y * 0.030;
            if handles == 1 {
                // A pair of crossheads, each on its own supported valve spindle.
                for axis in [dir, Vec3::Y.cross(dir)] {
                    metal.tube(&[root - axis * 0.022, root + axis * 0.022], 0.0035, 8);
                }
            } else {
                let reach = if handles == 2 { 0.037 } else { 0.025 };
                let lift = if handles == 2 { 0.006 } else { 0.041 };
                metal.tapered_tube(
                    &[root, root + dir * reach + Vec3::Y * lift],
                    0.005,
                    0.0035,
                    12,
                );
            }
        }
    }
}
