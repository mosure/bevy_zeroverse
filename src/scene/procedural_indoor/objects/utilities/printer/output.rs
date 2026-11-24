//! Fit finished sheets, pickup rollers and a sliding support to the actual outlet.
use super::*;
use crate::scene::procedural_indoor::{materials::variants, placement::rotated_half};

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct OutputProgram {
    pub media: Option<&'static str>,
    pub sheet_m: Vec2,
    pub count: usize,
    pub face_up: bool,
    pub caliper_m: f32,
    pub yaw: f32,
    pub center: Vec2,
    pub tray_start_z: f32,
    pub tray_tip_z: f32,
    pub bed_y: f32,
    pub roller_radius_m: f32,
    pub rollers: u8,
}

fn preferred_media(r: &mut impl Rng) -> [(&'static str, Vec2); 4] {
    use crate::scene::procedural_indoor::objects::literature::media::{A4, A5, LETTER, PHOTO};
    match r.random_range(0..20) {
        0 => [PHOTO, A5, LETTER, A4],
        1..=3 => [A5, LETTER, A4, PHOTO],
        4..=11 => [LETTER, A4, A5, PHOTO],
        _ => [A4, LETTER, A5, PHOTO],
    }
}

/// A full finished A4 or Letter page needs clearance beyond a compact
/// chassis. Size the extended tray before admission, using the same media
/// draw as the mesh. An archived or user-sized printer still fits the
/// nearest compatible stock inside its declared envelope.
pub(crate) fn reserve_media(o: &mut IndoorObject) {
    let p = super::sample(o, None);
    if p.sheet_count == 0 {
        return;
    }
    let mut r = stream(o.seed, u64::from_le_bytes(*b"PrintOut"));
    let radius = r.random_range(0.004..0.011_f32).min(p.slot_height * 0.32);
    let size = preferred_media(&mut r)[0].1;
    let extent = 2. * rotated_half(size, p.sheet_yaw);
    let opening_ratio = (p.body.x / o.size.x) * p.slot_fraction * 0.91;
    let usable_ratio = 1. - 0.17 * p.body.z / o.size.z;
    o.size.x = o.size.x.max((extent.x + 0.006) / opening_ratio);
    o.size.z = o
        .size
        .z
        .max((extent.y + 2. * radius + 0.018) / usable_ratio);
}

impl OutputProgram {
    pub fn sample(o: &IndoorObject, p: &PrinterProgram) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"PrintOut"));
        let rear = -o.size.z * 0.5 + 0.004;
        let front = rear + p.body.z;
        let start = rear + p.body.z * 0.17;
        let max = o.size.z * 0.5 - 0.002;
        let w = p.body.x * p.slot_fraction;
        let radius = r.random_range(0.004..0.011_f32).min(p.slot_height * 0.32);
        let from = start + 2. * radius + 0.003;
        let usable = Vec2::new(w * 0.91 - 0.004, max - from - 0.006);
        let choices = preferred_media(&mut r);
        let sheet = choices
            .into_iter()
            .find(|(_, size)| (2. * rotated_half(*size, p.sheet_yaw)).cmple(usable).all())
            .filter(|_| p.sheet_count > 0);
        let size = sheet.map_or(Vec2::ZERO, |(_, s)| s);
        let half = rotated_half(size, p.sheet_yaw);
        let slack = (usable - 2. * half).max(Vec2::ZERO);
        let cx = p.body.x * p.slot_center;
        let center = Vec2::new(
            cx + r.random_range(-0.5..0.5) * slack.x.min(0.010),
            from + half.y + r.random_range(0.70..1.0) * slack.y,
        );
        let tip = if sheet.is_some() {
            (center.y + half.y + 0.004).max(front)
        } else {
            front + (max - front) * r.random::<f32>().powi(2)
        };
        Self {
            media: sheet.map(|(name, _)| name),
            sheet_m: size,
            count: sheet.map_or(0, |_| p.sheet_count),
            face_up: sheet.is_some() && r.random_bool(0.57),
            caliper_m: r.random_range(0.00009..0.00013),
            yaw: p.sheet_yaw,
            center,
            tray_start_z: start,
            tray_tip_z: tip,
            bed_y: p.lower_top + 0.0035,
            roller_radius_m: radius,
            rollers: r.random_range(2..=4),
        }
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        p: &PrinterProgram,
        trim_label: &str,
    ) {
        let l = o.kind.class_name();
        let x = p.body.x * p.slot_center;
        let w = p.body.x * p.slot_fraction;
        let front = -o.size.z * 0.5 + 0.004 + p.body.z;
        // The bed and extension meet edge-to-edge. Both support the sheet;
        // there is no second hidden paper plane or stack of coincident boxes.
        for (start, end, width) in [
            (self.tray_start_z, front, 0.97 * w),
            (front, self.tray_tip_z, 0.91 * w),
        ] {
            if end - start > 0.0001 {
                a.box_part(
                    Surface::Plastic,
                    trim_label,
                    Vec3::new(x, self.bed_y - 0.00175, (start + end) * 0.5),
                    Vec3::new(width, 0.0035, end - start),
                    0.0006,
                );
            }
        }
        let radius = self.roller_radius_m;
        let axle = Vec3::new(x, self.bed_y + radius, self.tray_start_z + radius);
        let rot = Quat::from_rotation_z(FRAC_PI_2);
        a.part(Surface::Metal, l).cylinder(
            radius * 0.40,
            w,
            Transform::from_translation(axle).with_rotation(rot),
        );
        for i in 0..self.rollers {
            let dx = w * 0.79 * ((f32::from(i) + 0.5) / f32::from(self.rollers) - 0.5);
            a.part(Surface::Rubber, l).cylinder(
                radius,
                w * 0.056,
                Transform::from_translation(axle + Vec3::X * dx).with_rotation(rot),
            );
        }
        if self.count == 0 {
            return;
        }
        let h = self.caliper_m * self.count as f32;
        let half = self.sheet_m * 0.5;
        let rotation = Quat::from_rotation_y(self.yaw);
        let center = Vec3::new(self.center.x, self.bed_y + h, self.center.y);
        let top = [(-1., -1.), (-1., 1.), (1., 1.), (1., -1.)]
            .map(|(x, z)| center + rotation * Vec3::new(x * half.x, 0., z * half.y));
        let (surface, label) = if self.face_up {
            (
                Surface::PrintedPaper,
                format!(
                    "paper#finish{}#uv_local",
                    variants::object_slot(Surface::PrintedPaper, o.seed, o.finish_seed)
                ),
            )
        } else {
            (Surface::Paper, "paper".to_owned())
        };
        a.part(surface, &label).vertex_quad(
            top,
            [Vec3::Y; 4],
            [[0., 0.], [0., 1.], [1., 1.], [1., 0.]],
        );
        let edge = a.part(Surface::Paper, "paper");
        for i in 0..4 {
            let (a, b) = (top[i], top[(i + 1) % 4]);
            edge.metric_quad(
                [b, a, a - Vec3::Y * h, b - Vec3::Y * h],
                (b - a).cross(Vec3::Y).normalize(),
            );
        }
    }
}
