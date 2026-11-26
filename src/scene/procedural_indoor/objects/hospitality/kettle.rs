//! Fitted vessel, attached handles and a hollow pouring spout share one profile.
use super::*;
mod lid;
mod vessel;

pub const FORMED_KETTLE_VERSION: u32 = 268;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum Outlet {
    FormedLip,
    TaperedSpout,
    Gooseneck,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
pub enum LidGrip {
    PressButton,
    LiftKnob,
    Bridge,
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Hardware {
    pub outlet: Outlet,
    pub lid_grip: LidGrip,
    pub lid_rise_m: f32,
    pub grip_span_m: f32,
    pub grip_height_m: f32,
    pub wall_m: f32,
    pub lip_half_angle: f32,
    pub bend: f32,
}

impl Hardware {
    pub(crate) fn exit_elevation(self, chord: f32) -> f32 {
        if self.outlet == Outlet::Gooseneck {
            chord * (1. - self.bend)
        } else {
            chord * (1. - self.bend * 0.86)
        }
    }
    fn sample(o: &IndoorObject, p: &KettleProgram) -> Self {
        let mut shape = stream(o.seed, 0x4b45_5454_504f_5552);
        let outlet = if shape.random_bool(if p.overhead_handle { 0.09 } else { 0.76 }) {
            Outlet::FormedLip
        } else if shape.random_bool(if p.overhead_handle { 0.21 } else { 0.62 }) {
            Outlet::Gooseneck
        } else {
            Outlet::TaperedSpout
        };
        let mut lid = stream(o.seed, 0x4b45_5454_4c49_4654);
        let lid_grip = if !p.overhead_handle && lid.random_bool(0.49) {
            LidGrip::PressButton
        } else if lid.random_bool(0.42) {
            LidGrip::Bridge
        } else {
            LidGrip::LiftKnob
        };
        let neck = p.neck * p.radius;
        Self {
            outlet,
            lid_grip,
            lid_rise_m: p.height * lid.random_range(0.018..0.063),
            grip_span_m: neck * lid.random_range(0.68..1.13),
            grip_height_m: o.size.y * lid.random_range(0.032..0.067),
            wall_m: (p.spout_radius * shape.random_range(0.12..0.22)).min(0.0021),
            lip_half_angle: shape.random_range(0.56..0.94),
            bend: if outlet == Outlet::Gooseneck {
                shape.random_range(0.78..0.94)
            } else {
                shape.random_range(0.10..0.35)
            },
        }
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct KettleProgram {
    pub radius: f32,
    pub height: f32,
    pub base: f32,
    pub belly_height: f32,
    pub foot_fraction: f32,
    pub shoulder: f32,
    pub neck: f32,
    pub overhead_handle: bool,
    pub handle_radius: f32,
    pub handle_reach: f32,
    pub spout_root: f32,
    pub spout_tip: f32,
    pub spout_reach: f32,
    pub spout_radius: f32,
    pub finish: Surface,
    pub grip: Surface,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hardware: Option<Hardware>,
}
pub fn parameters(o: &IndoorObject) -> KettleProgram {
    parameters_at(o, bevy_zeroverse_capture::GENERATOR_VERSION)
}

pub(crate) fn parameters_at(o: &IndoorObject, version: u32) -> KettleProgram {
    let mut r = stream(o.seed, 0x4b4554544c455052);
    let s = o.size;
    let overhead_handle = r.random_bool(0.3);
    let radius = (s.x * r.random_range(0.35_f32..0.46)).min(s.z * r.random_range(0.25..0.31));
    let mut p = KettleProgram {
        radius,
        height: s.y
            * if overhead_handle {
                r.random_range(0.53..0.68)
            } else {
                r.random_range(0.75..0.87)
            },
        base: if overhead_handle {
            0.
        } else {
            s.y * r.random_range(0.018..0.06)
        },
        belly_height: r.random_range(0.25..0.56),
        foot_fraction: r.random_range(0.68..0.92),
        shoulder: r.random_range(0.62..0.96),
        neck: r.random_range(0.37..0.65),
        overhead_handle,
        handle_radius: s.x.min(s.z) * r.random_range(0.027..0.047),
        handle_reach: r.random_range(0.74..0.96),
        spout_root: r.random_range(0.43..0.61),
        spout_tip: r.random_range(0.80..0.94),
        spout_reach: s.z * r.random_range(0.36..0.45),
        spout_radius: s.x.min(s.z) * r.random_range(0.034..0.061),
        finish: [
            Surface::Metal,
            Surface::Chrome,
            Surface::Plastic,
            Surface::Ceramic,
        ][r.random_range(0..4)],
        grip: [Surface::Plastic, Surface::Rubber, Surface::Wood][r.random_range(0..3)],
        hardware: None,
    };
    if version >= FORMED_KETTLE_VERSION {
        let mut hardware = Hardware::sample(o, &p);
        if hardware.outlet == Outlet::FormedLip {
            let neck = 0.63 + 0.31 * p.neck;
            hardware.grip_span_m *= neck / p.neck;
            p.neck = neck;
            p.shoulder = p.shoulder.max(neck);
        } else {
            p.spout_root = if hardware.outlet == Outlet::TaperedSpout {
                p.belly_height + (0.78 - p.belly_height) * p.spout_root
            } else {
                p.belly_height * (p.spout_root + 0.15)
            };
            p.spout_reach = p.spout_reach.max(p.radius + o.size.z * 0.11);
        }
        hardware.grip_height_m = hardware
            .grip_height_m
            .min((o.size.y - p.base - p.height - hardware.lid_rise_m) * 0.84);
        let mut finish = stream(o.finish_seed.unwrap_or(o.seed), 0x4b45_5454_4649_4e49);
        let finishes: &[Surface] = match hardware.outlet {
            Outlet::Gooseneck => &[Surface::Metal, Surface::Chrome],
            Outlet::FormedLip => &[
                Surface::Plastic,
                Surface::Metal,
                Surface::Ceramic,
                Surface::Chrome,
            ],
            Outlet::TaperedSpout => &[Surface::Metal, Surface::Chrome, Surface::Ceramic],
        };
        p.finish = finishes[finish.random_range(0..finishes.len())];
        p.grip = [Surface::Plastic, Surface::Rubber, Surface::Wood][finish.random_range(0..3)];
        p.hardware = Some(hardware);
    }
    p
}
impl KettleProgram {
    pub(crate) fn profile(&self) -> [(f32, f32); 5] {
        [
            (self.radius * self.foot_fraction, 0.),
            (self.radius, self.height * self.belly_height),
            (self.radius * self.shoulder, self.height * 0.78),
            (self.radius * self.neck, self.height * 0.96),
            (self.radius * self.neck, self.height),
        ]
    }
    pub(crate) fn radius_at(&self, height: f32) -> f32 {
        let q = self.profile();
        let w = q.windows(2).find(|w| height <= w[1].1).unwrap();
        w[0].0 + (w[1].0 - w[0].0) * (height - w[0].1) / (w[1].1 - w[0].1)
    }
    pub(crate) fn port_levels(&self, hw: Hardware) -> [f32; 3] {
        let belly = self.height * self.belly_height;
        let center = self.height * self.spout_root;
        let half = if hw.outlet == Outlet::TaperedSpout {
            ((center - belly).min(self.height * 0.78 - center) * 0.87).min(self.radius * 0.32)
        } else {
            (belly * 0.24).min(self.radius * 0.28)
        };
        [center - half, center, center + half]
    }
    pub(crate) fn socket_z(&self, hw: Hardware) -> f32 {
        let radius = if hw.outlet == Outlet::TaperedSpout {
            let [lo, _, hi] = self.port_levels(hw);
            self.radius_at(lo).max(self.radius_at(hi))
        } else {
            self.radius
        };
        radius + hw.wall_m * 2.
    }
    pub(crate) fn handle(&self, s: Vec3) -> Vec<Vec3> {
        let h = self.height;
        if self.overhead_handle {
            let bottom = h * 0.69;
            let half = self.radius_at(bottom) * 0.97;
            let rise = s.y * 0.98 - self.handle_radius - self.base - bottom;
            (0..=20)
                .map(|i| {
                    let t = PI * i as f32 / 20.;
                    Vec3::new(half * t.cos(), self.base + bottom + rise * t.sin(), 0.)
                })
                .collect()
        } else {
            let low = h * 0.24;
            let high = h * 0.80;
            let ends = [self.radius_at(low) * 0.97, self.radius_at(high) * 0.97];
            (0..=20)
                .map(|i| {
                    let t = i as f32 / 20.;
                    let attached = ends[0] + (ends[1] - ends[0]) * t;
                    let limit = s.z * 0.5 - self.handle_radius - 0.001;
                    Vec3::new(
                        0.,
                        self.base + low + (high - low) * (0.5 - 0.5 * (PI * t).cos()),
                        -attached - (limit - attached) * self.handle_reach * (PI * t).sin(),
                    )
                })
                .collect()
        }
    }
    pub(crate) fn spout(&self) -> (Vec3, Vec3) {
        let root = self.height * self.spout_root;
        (
            Vec3::new(0., self.base + root, self.radius_at(root) * 0.84),
            Vec3::new(
                0.,
                self.base + self.height * self.spout_tip,
                self.spout_reach,
            ),
        )
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters_at(o, version);
    if let Some(hardware) = p.hardware {
        let l = o.kind.class_name();
        vessel::build(a, l, &p, &hardware);
        lid::build(a, l, &p, &hardware);
        a.part(p.grip, l)
            .tube(&p.handle(o.size), p.handle_radius, 10);
        if p.overhead_handle {
            // A bail rotates on two bosses attached to opposite vessel flanks.
            let elevation = p.height * 0.69;
            let shell = p.radius_at(elevation);
            let depth = (o.size.x * 0.49 - shell).min(p.handle_radius * 0.90);
            let r = p.handle_radius * 1.28;
            for side in [-1., 1.] {
                let mount = Transform::from_xyz(
                    (shell - p.handle_radius * 0.45) * side,
                    p.base + elevation,
                    0.,
                )
                .with_rotation(Quat::from_rotation_arc(Vec3::Y, Vec3::X * side));
                let reach = depth + p.handle_radius * 0.45;
                a.part(Surface::Metal, l).lathe(
                    &[(r, 0.), (r, reach * 0.72), (r * 0.78, reach), (0., reach)],
                    16,
                    mount,
                );
            }
        }
    } else {
        legacy(a, o, &p);
    }
}

fn legacy(a: &mut Assembly, o: &IndoorObject, p: &KettleProgram) {
    let l = o.kind.class_name();
    let mut profile = vec![(0., 0.)];
    profile.extend(p.profile());
    profile.push((0., p.height));
    a.part(p.finish, l)
        .lathe(&profile, 32, Transform::from_xyz(0., p.base, 0.));
    if p.base > 0. {
        a.part(Surface::Plastic, l).cylinder(
            p.radius * p.foot_fraction * 1.07,
            p.base,
            Transform::from_xyz(0., p.base * 0.5, 0.),
        );
    }
    let lid_h = o.size.y * 0.025;
    a.part(p.finish, l).lathe(
        &[
            (0., 0.),
            (p.radius * p.neck * 1.025, 0.),
            (p.radius * p.neck * 0.88, lid_h),
            (0., lid_h),
        ],
        24,
        Transform::from_xyz(0., p.base + p.height + 0.0004, 0.),
    );
    a.part(p.grip, l).cylinder(
        p.radius * 0.13,
        o.size.y * 0.028,
        Transform::from_xyz(
            0.,
            p.base + p.height + lid_h + o.size.y * 0.014 + 0.0004,
            0.,
        ),
    );
    a.part(p.grip, l)
        .tube(&p.handle(o.size), p.handle_radius, 10);
    let (root, tip) = p.spout();
    let length = root.distance(tip);
    let r = p.spout_radius;
    // Inward profile returns down the bore; no dark disk covers the mouth.
    a.part(p.finish, l).lathe(
        &[
            (r * 1.5, 0.),
            (r, length),
            (r * 0.72, length),
            (r * 1.15, length * 0.12),
        ],
        20,
        Transform::from_translation(root)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, (tip - root).normalize())),
    );
}
