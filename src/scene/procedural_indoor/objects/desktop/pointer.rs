//! Moulded pointing devices. The palm shell, control opening and retaining
//! bezel share one rim so their silhouettes agree in every capture attachment.
pub use super::super::super::materials::input_housing::BallFinish;
use super::super::super::materials::input_housing::Housing;
use super::*;

mod shell;
#[cfg(test)]
mod tests;

pub(crate) const INTRODUCED: u32 = 225;

// Low variant bits retain the recorded product variant. These two bits archive
// the operating side selected from an actual neighboring keyboard, if present.
const LEFT: u32 = 1 << 8;
const RIGHT: u32 = 1 << 9;

pub(crate) fn fit_operator_side(mouse: &mut IndoorObject, objects: &[IndoorObject]) {
    if mouse.support.is_none() {
        return;
    }
    let keyboard = objects
        .iter()
        .rev()
        .filter(|k| {
            k.kind == ObjectKind::Keyboard
                && k.support == mouse.support
                && k.interaction_target == mouse.interaction_target
                && (k.yaw - mouse.yaw).cos() > 0.66
        })
        .filter_map(|k| {
            let offset = Quat::from_rotation_y(-k.yaw) * (mouse.position - k.position);
            (offset.x.abs() > k.size.x * 0.32 && offset.x.abs() < 0.62 && offset.z.abs() < 0.21)
                .then_some((offset.length_squared(), offset.x))
        })
        .min_by(|a, b| a.0.total_cmp(&b.0));
    if let Some((_, x)) = keyboard {
        mouse.variant = (mouse.variant & !(LEFT | RIGHT)) | if x < 0. { LEFT } else { RIGHT };
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Grip {
    Symmetric,
    Left,
    Right,
}

impl Grip {
    fn side(self) -> f32 {
        match self {
            Self::Left => -1.,
            Self::Right => 1.,
            Self::Symmetric => 0.,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Control {
    ScrollWheel { alloy: bool, ribs: u8 },
    FingerTrackball { finish: BallFinish },
    ThumbTrackball { finish: BallFinish },
}

impl Control {
    pub(crate) fn ball(self) -> Option<BallFinish> {
        match self {
            Self::ScrollWheel { .. } => None,
            Self::FingerTrackball { finish } | Self::ThumbTrackball { finish } => Some(finish),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub struct PointerProgram {
    pub grip: Grip,
    pub keyboard_side_known: bool,
    pub control: Control,
    /// Local X/Z, relative to the centre of the desk footprint, in metres.
    pub control_position_m: Vec2,
    pub radius_m: f32,
    pub control_width_m: f32,
    pub exposure_fraction: f32,
    pub clearance_m: f32,
    pub palm_crest_fraction: f32,
    pub seam_width_m: f32,
    pub waist: f32,
    outline_power: f32,
    bank: f32,
    front_drop: f32,
    rear_drop: f32,
    side_drop: f32,
    seam_station: usize,
    base_height: f32,
}

impl PointerProgram {
    pub fn sample(o: &IndoorObject, input: &InputProgram) -> Self {
        let mut rng = stream(o.seed, u64::from_le_bytes(*b"Pointing"));
        let control = if input.trackball {
            let finish = BallFinish::sample(&mut rng);
            if rng.random_bool(0.54) {
                Control::ThumbTrackball { finish }
            } else {
                Control::FingerTrackball { finish }
            }
        } else {
            Control::ScrollWheel {
                alloy: rng.random_bool(0.24),
                ribs: rng.random_range(12..=22),
            }
        };
        let hand = o.variant & (LEFT | RIGHT);
        let grip = if !matches!(control, Control::ThumbTrackball { .. }) && rng.random_bool(0.54) {
            Grip::Symmetric
        } else if hand == LEFT || (hand != RIGHT && rng.random_bool(0.17)) {
            Grip::Left
        } else {
            Grip::Right
        };
        let s = o.size;
        let (x, z, radius) = match control {
            Control::ThumbTrackball { .. } => (
                -grip.side() * rng.random_range(0.155..0.202),
                rng.random_range(-0.11..0.015),
                (s.x * rng.random_range(0.174..0.207)).min(s.y * 0.44),
            ),
            Control::FingerTrackball { .. } => (
                0.,
                rng.random_range(-0.24..-0.135),
                (s.x * rng.random_range(0.225..0.288)).min(s.y * 0.45),
            ),
            Control::ScrollWheel { .. } => (
                grip.side() * rng.random_range(-0.036..0.018),
                rng.random_range(-0.295..-0.19),
                (s.y * rng.random_range(0.19..0.273)).min(s.z * 0.095),
            ),
        };
        // Allow stock above the desk for the cup floor even on a low shell.
        let radius = radius.min((s.y - 0.0045).max(s.y * 0.5) * 0.5);
        Self {
            grip,
            keyboard_side_known: matches!(hand, LEFT | RIGHT),
            control,
            control_position_m: Vec2::new(x * s.x, z * s.z),
            radius_m: radius,
            control_width_m: if input.trackball {
                2. * radius
            } else {
                (s.x * rng.random_range(0.072..0.125)).min(0.009)
            },
            exposure_fraction: if input.trackball {
                rng.random_range(0.48..0.69)
            } else {
                rng.random_range(0.36..0.58)
            },
            clearance_m: (s.x * rng.random_range(0.012..0.019)).clamp(0.00068, 0.0013),
            palm_crest_fraction: rng.random_range(0.59..0.76),
            seam_width_m: rng.random_range(0.00070..0.00125),
            waist: rng.random_range(0.025..0.19),
            outline_power: input.mouse_roundness.min(3.65),
            bank: grip.side() * rng.random_range(0.08..0.16),
            front_drop: rng.random_range(0.32..0.42),
            rear_drop: rng.random_range(0.17..0.26),
            side_drop: rng.random_range(0.18..0.28),
            seam_station: rng.random_range(0..=3),
            base_height: rng.random_range(0.19..0.28),
        }
    }

    /// Smooth palm hump with separate nose and heel slopes and a thumb-side bank.
    /// Returns the height and its derivatives in local metric coordinates.
    fn roof(&self, s: Vec3, p: Vec2) -> (f32, Vec2) {
        let u = 2. * p.x / s.x;
        let t = 0.5 + p.y / s.z;
        let front = t < self.palm_crest_fraction;
        let (drop, run) = if front {
            (self.front_drop, self.palm_crest_fraction)
        } else {
            (self.rear_drop, 1. - self.palm_crest_fraction)
        };
        let v = (t - self.palm_crest_fraction) / run;
        let h = 0.935 - drop * v * v - self.side_drop * u * u - self.bank * u;
        (
            s.y * h,
            Vec2::new(
                -2. * s.y / s.x * (2. * self.side_drop * u + self.bank),
                -2. * drop * v * s.y / (run * s.z),
            ),
        )
    }

    fn socket(&self, size: Vec3) -> Socket {
        let pos = self.control_position_m;
        let (height, slope) = self.roof(size, pos);
        let normal = Vec3::new(-slope.x, 1., -slope.y).normalize();
        let span = self.radius_m;
        let buried = span * (1. - self.exposure_fraction);
        let center_floor = 0.0015 + span + self.clearance_m;
        let y = height
            .max(center_floor + buried * normal.y)
            .min(size.y - (span - buried * normal.y) - 0.0012);
        let tf = Transform::from_xyz(pos.x, y, pos.y)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, normal));
        // The aperture is cut at the exact intersection of the ball/wheel
        // and the tangent plane, plus clearance for free rotation.
        let opening_radius = (span * span - buried * buried).sqrt();
        let (half, power) = if self.control.ball().is_some() {
            (Vec2::splat(opening_radius + self.clearance_m), 2.)
        } else {
            (
                Vec2::new(
                    self.control_width_m * 0.5 + self.clearance_m * 1.6,
                    opening_radius + self.clearance_m * 1.5,
                ),
                4.8,
            )
        };
        Socket {
            tf,
            half,
            power,
            buried,
            lip: self.clearance_m * 1.60,
        }
    }
}

struct Socket {
    tf: Transform,
    half: Vec2,
    power: f32,
    buried: f32,
    lip: f32,
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, input: &InputProgram) {
    let program = PointerProgram::sample(o, input);
    let socket = program.socket(o.size);
    let stock = Housing::for_object(o);
    shell::build(a, o, &program, &stock, &socket);
    let center = socket.tf.transform_point(Vec3::NEG_Y * socket.buried);
    match program.control {
        Control::FingerTrackball { finish } | Control::ThumbTrackball { finish } => {
            a.part(Surface::Plastic, finish.label()).ellipsoid(
                Vec3::splat(program.radius_m),
                Transform::from_translation(center),
            );
        }
        Control::ScrollWheel { alloy, ribs } => {
            let r = program.radius_m;
            let w = program.control_width_m * 0.5;
            let tf = Transform::from_translation(center)
                .with_rotation(socket.tf.rotation * Quat::from_rotation_z(FRAC_PI_2));
            a.part(
                if alloy {
                    Surface::Chrome
                } else {
                    Surface::Rubber
                },
                "mouse",
            )
            .fluted_lathe(
                &[
                    (0., -w),
                    (r * 0.77, -w),
                    (r, -w * 0.69),
                    (r, w * 0.69),
                    (r * 0.77, w),
                    (0., w),
                ],
                u32::from(ribs) * 3,
                if alloy { 0.020 } else { 0.045 },
                u32::from(ribs),
                tf,
            );
        }
    }
    let port = super::ports::input_local(o, input);
    // A moulded cable bushing seats in the nose and terminates on the
    // registered cable departure plane, for both connected and idle desks.
    let r = (o.size.x * 0.057).clamp(0.0029, 0.0045);
    a.part(Surface::Rubber, "mouse").lathe(
        &[
            (0., 0.),
            (r * 0.85, 0.),
            (r, 0.0026),
            (r * 0.90, 0.0055),
            (r * 1.18, 0.008),
            (r * 1.18, 0.015),
            (0., 0.015),
        ],
        16,
        Transform::from_translation(port).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
}
