//! The desk slopes toward local +Z, where the presenter stands. Dimensions are metres.
use super::*;
use crate::scene::procedural_indoor::{materials::variants, placement::rotated_half};

pub const VERSION: u32 = 326;
const INSTALLED: u32 = 1 << 8;
const NOTES: u32 = 1 << 9;

pub(crate) fn install(o: &mut IndoorObject, clutter: f32) {
    o.variant |= INSTALLED;
    o.variant &= !NOTES;
    if stream(o.seed, u64::from_le_bytes(*b"DeskUsed")).random::<f32>() < clutter {
        o.variant |= NOTES;
    }
}

pub fn installed(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::Lectern && o.variant & INSTALLED != 0
}

pub fn standing_depth(o: &IndoorObject) -> f32 {
    0.72 + 0.22 * f32::from((o.seed >> 32) as u16) / f32::from(u16::MAX)
}

pub fn presenter_position(o: &IndoorObject) -> Vec3 {
    o.transform()
        .transform_point(Vec3::Z * (o.size.z + standing_depth(o)) * 0.5)
}

/// The crossbar runs along X. Its cylindrical crown meets the mounting shoe below the desk.
pub(super) fn mount(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &TeachingProgram,
    desk: Transform,
    thickness: f32,
) -> Vec3 {
    let radius = o.size.x.min(o.size.z) * p.support.stem_fraction;
    let plate = (0.48 * radius).clamp(0.006, 0.014);
    let crossbar = p.support.foundation == 3;
    let width = if crossbar {
        o.size.x * p.support.span.x * 0.16 + 0.02
    } else {
        radius * 2.8
    };
    let underside = desk * Transform::from_xyz(0., -thickness * 0.5, 0.);
    a.part(Surface::Metal, o.kind.class_name()).cuboid(
        Vec3::new(width, plate, 2.8 * radius),
        0.001,
        underside * Transform::from_xyz(0., -plate * 0.5 + 0.0004, 0.),
    );
    let offset = if crossbar {
        plate + radius - 0.0015
    } else {
        plate * 0.5
    };
    underside.transform_point(Vec3::NEG_Y * offset)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MicStyle {
    Boundary,
    Gooseneck,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Microphone {
    pub style: MicStyle,
    pub socket_xz_m: Vec2,
    pub rise_m: f32,
    pub reach_m: f32,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Notes {
    pub sheet_m: Vec2,
    pub center_xz_m: Vec2,
    pub yaw_rad: f32,
    pub stack_m: f32,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ReadingDesk {
    pub deck_drop_m: f32,
    pub stop_z_m: f32,
    pub stop_depth_m: f32,
    pub stop_height_m: f32,
    pub microphone: Option<Microphone>,
    pub notes: Option<Notes>,
}

impl ReadingDesk {
    pub(super) fn fit(o: &IndoorObject, p: &TeachingProgram) -> Self {
        let mut kit = stream(o.seed, u64::from_le_bytes(*b"DeskMics"));
        let width = o.size.x * p.board_fraction.x;
        let depth = o.size.z * 0.88;
        let microphone = p.microphone.then(|| {
            let sign = if kit.random_bool(0.5) { 1. } else { -1. };
            let socket_xz_m = Vec2::new(
                sign * width * kit.random_range(0.32..0.42),
                -depth * kit.random_range(0.23..0.37),
            );
            // Reserve the neck inside the admitted height, without making the writing deck too low.
            let headroom =
                (o.size.y * 0.97 - depth * p.desk_angle.sin() * 0.5 - p.thickness - 0.90)
                    .min(0.215);
            let tall = headroom > 0.105 && kit.random_bool(0.67);
            Microphone {
                style: if tall {
                    MicStyle::Gooseneck
                } else {
                    MicStyle::Boundary
                },
                socket_xz_m,
                rise_m: if tall {
                    kit.random_range(0.105..headroom)
                } else {
                    kit.random_range(0.010..0.021)
                },
                reach_m: if tall {
                    depth * kit.random_range(0.19..0.35)
                } else {
                    0.
                },
            }
        });
        let deck_drop_m = microphone
            .filter(|m| m.style == MicStyle::Gooseneck)
            .map_or(0., |m| m.rise_m);
        let stop_z_m = depth * kit.random_range(0.435..0.465);
        let stop_depth_m = width.min(depth) * kit.random_range(0.031..0.044);
        let stop_height_m = kit.random_range(0.011..0.019);
        let notes = if o.variant & NOTES == 0 {
            None
        } else {
            Notes::fit(
                o.seed,
                width,
                depth,
                stop_z_m - stop_depth_m * 0.5,
                microphone,
            )
        };
        Self {
            deck_drop_m,
            stop_z_m,
            stop_depth_m,
            stop_height_m,
            microphone,
            notes,
        }
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        finish: Surface,
        size: Vec3,
        pose: Transform,
    ) {
        let mut cuts = TimberCuts::new(o.seed, u64::from_le_bytes(*b"ReadDesk"));
        let g = a.part(finish, o.kind.class_name());
        cuts.cuboid(g, finish, size, 0.005, pose, GrainAxis::X);
        let top = pose * Transform::from_xyz(0., size.y * 0.5, 0.);
        cuts.cuboid(
            g,
            finish,
            Vec3::new(size.x * 0.94, self.stop_height_m, self.stop_depth_m),
            0.002,
            top * Transform::from_xyz(0., self.stop_height_m * 0.5 - 0.0004, self.stop_z_m),
            GrainAxis::X,
        );
        if let Some(mic) = self.microphone {
            mic.build(a, top);
        }
        if let Some(notes) = self.notes {
            notes.build(a, o, top);
        }
    }
}

impl Microphone {
    fn build(self, a: &mut Assembly, top: Transform) {
        let at = top * Transform::from_xyz(self.socket_xz_m.x, -0.0004, self.socket_xz_m.y);
        let h = self.rise_m;
        let metal = a.part(Surface::Metal, "microphone");
        if self.style == MicStyle::Boundary {
            metal.lathe(
                &[
                    (0., 0.),
                    (0.024, 0.),
                    (0.026, 0.003),
                    (0.024, h * 0.60),
                    (0.019, h * 0.70),
                ],
                16,
                at,
            );
            a.part(Surface::Rubber, "microphone").lathe(
                &[(0.019, h * 0.70), (0.017, h * 0.88), (0.009, h), (0., h)],
                16,
                at,
            );
        } else {
            metal.lathe(
                &[
                    (0., 0.),
                    (0.018, 0.),
                    (0.018, 0.005),
                    (0.008, 0.011),
                    (0.005, 0.014),
                ],
                12,
                at,
            );
            let d = self.reach_m;
            let points = [
                (0.013, 0.),
                (h * 0.38, 0.),
                (h * 0.61, d * 0.06),
                (h * 0.78, d * 0.24),
                (h * 0.91, d * 0.58),
                (h, d),
            ]
            .map(|(y, z)| at.transform_point(Vec3::new(0., y, z)));
            metal.tube(&points, 0.004, 8);
            let dir = (points[5] - points[4]).normalize();
            a.part(Surface::Rubber, "microphone").small_ellipsoid(
                Vec3::new(0.009, 0.017, 0.009),
                Transform::from_translation(points[5] + dir * 0.010)
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, dir)),
            );
        }
    }
}

impl Notes {
    fn fit(seed: u64, width: f32, depth: f32, stop: f32, mic: Option<Microphone>) -> Option<Self> {
        let mut use_state = stream(seed, u64::from_le_bytes(*b"DeskNote"));
        let yaw_rad = use_state.random_range(-0.12..0.12);
        let letter = if use_state.random_bool(0.5) {
            Vec2::new(0.210, 0.297)
        } else {
            Vec2::new(0.216, 0.279)
        };
        let sizes = [letter, Vec2::new(0.148, 0.210), Vec2::new(0.152, 0.102)];
        let first = use_state.random_range(0..3);
        for &sheet_m in &sizes[first..] {
            let half = rotated_half(sheet_m, yaw_rad);
            let mut left = -width * 0.46 + half.x;
            let mut right = width * 0.46 - half.x;
            if let Some(m) = mic {
                if m.socket_xz_m.x > 0. {
                    right = right.min(m.socket_xz_m.x - 0.031 - half.x);
                } else {
                    left = left.max(m.socket_xz_m.x + 0.031 + half.x);
                }
            }
            if right <= left || stop - 2. * half.y < -depth * 0.47 {
                continue;
            }
            return Some(Self {
                sheet_m,
                yaw_rad,
                center_xz_m: Vec2::new(use_state.random_range(left..right), stop - half.y),
                stack_m: use_state.random_range(0.0015..0.006),
            });
        }
        None
    }
    fn build(self, a: &mut Assembly, o: &IndoorObject, top: Transform) {
        let at = top
            * Transform::from_xyz(self.center_xz_m.x, self.stack_m, self.center_xz_m.y)
                .with_rotation(Quat::from_rotation_y(self.yaw_rad));
        let half = self.sheet_m * 0.5;
        let corners = [(-1., -1.), (-1., 1.), (1., 1.), (1., -1.)]
            .map(|(x, z)| at.transform_point(Vec3::new(x * half.x, 0., z * half.y)));
        let normal = top.rotation * Vec3::Y;
        let slot = variants::object_slot(Surface::PrintedPaper, o.seed, o.finish_seed);
        a.part(
            Surface::PrintedPaper,
            &format!("paper#finish{slot}#uv_local"),
        )
        .vertex_quad(
            corners,
            [normal; 4],
            [[0., 0.], [0., 1.], [1., 1.], [1., 0.]],
        );
        let edge = a.part(Surface::Paper, "paper");
        for i in 0..4 {
            let (a, b) = (corners[i], corners[(i + 1) % 4]);
            edge.metric_quad(
                [b, a, a - normal * self.stack_m, b - normal * self.stack_m],
                (b - a).cross(normal).normalize(),
            );
        }
    }
}

#[cfg(test)]
mod tests;
