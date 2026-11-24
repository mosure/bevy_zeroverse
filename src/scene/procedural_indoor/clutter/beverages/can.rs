//! Drawn-metal beverage cans with coordinated heel/neck/label and lid geometry.
//! Product dimensions are sampled separately, before support/collision admission.
use super::*;
use crate::scene::procedural_indoor::materials::package_print;
mod end;
pub use end::CanEnd;
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct CanProgram {
    pub radius_fraction: f32,
    pub height_fraction: f32,
    pub heel_height: f32,
    pub heel_radius: f32,
    pub shoulder: f32,
    pub neck: f32,
    pub waist: f32,
    pub label_bottom: f32,
    pub label_top: f32,
    pub bare: bool,
    pub opened: bool,
    pub tab_width: f32,
    pub tab_length: f32,
    pub lid_yaw: f32,
    pub end: CanEnd,
}
pub fn parameters(o: &IndoorObject) -> CanProgram {
    let mut r = stream(o.seed, 0x43414e50524f4649);
    let radius_fraction = r.random_range(0.39..0.49);
    let height_fraction = r.random_range(0.92..1.);
    let heel_height = r.random_range(0.025..0.105);
    let heel_radius = r.random_range(0.76..0.95);
    let shoulder = r.random_range(0.82..0.935);
    let neck = r.random_range(0.73..0.95);
    let neck_radius = o.size.x.min(o.size.z) * radius_fraction * neck;
    let mut p = CanProgram {
        radius_fraction,
        height_fraction,
        heel_height,
        heel_radius,
        shoulder,
        neck,
        waist: r.random_range(0.93..1.),
        label_bottom: r.random_range(0.12..0.32),
        label_top: r.random_range(0.66..0.91),
        bare: r.random_bool(0.12),
        opened: r.random_bool(0.64),
        tab_width: r.random_range(0.26..0.42),
        tab_length: r.random_range(0.37..0.58),
        lid_yaw: r.random_range(0.0..TAU),
        end: CanEnd::sample(o.seed, neck_radius),
    };
    let mut print = stream(o.seed, u64::from_le_bytes(*b"CanPrint"));
    if !p.bare && print.random_bool(0.76) {
        // Direct can printing follows the tangent of the drawn heel and shoulder.
        // Other products carry a narrower band with exposed aluminium on each side.
        p.label_bottom = p.heel_height * print.random_range(0.86..1.20);
        p.label_top = p.shoulder + (1. - p.shoulder) * print.random_range(0.02..0.27);
    }
    if p.opened {
        let panel = neck_radius * p.end.panel_fraction;
        let available_rise =
            (neck_radius * p.end.recess_fraction - 1.9 * p.end.gauge_m - 0.00018).max(0.);
        let tab_length = (0.16 + 0.83 * p.tab_length) * 2. * panel;
        p.end.tab_lift_radians = p
            .end
            .tab_lift_radians
            .min((available_rise / tab_length).clamp(0., 1.).asin());
    } else {
        p.end.fold_radians = 0.;
        p.end.tab_lift_radians = 0.;
        p.end.fill_fraction = None;
    }
    p
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let h = o.size.y * p.height_fraction;
    let r = o.size.x.min(o.size.z) * p.radius_fraction;
    let neck = r * p.neck;
    let end_y = h - (0.65 * h * (1. - p.shoulder)).min(0.15 * neck);
    let foot = r * p.heel_radius;
    let heel_y = h * p.heel_height;
    let shoulder_y = h * p.shoulder;
    let shoulder_rise = end_y - shoulder_y;
    let stations = vec![
        (foot, 0.),
        (foot + (r - foot) * 0.17, heel_y * 0.43),
        (foot + (r - foot) * 0.85, heel_y * 0.79),
        (r, heel_y),
        (r * p.waist, h * 0.5),
        (r, shoulder_y),
        (r + (neck - r) * 0.16, shoulder_y + 0.27 * shoulder_rise),
        (neck + (r - neck) * 0.14, end_y - 0.24 * shoulder_rise),
        (neck, end_y),
    ];
    let label = package_print::ALUMINUM;
    let g = a.part(Surface::Metal, label);
    let round = outline(32);
    let tf = Transform::from_rotation(Quat::from_rotation_y(p.lid_yaw));
    // The push-up bottom rests on an annular foot. Only an open can exposes its interior.
    round.sweep(
        g,
        &[
            (0., 0.034 * h),
            (0.57 * r, 0.025 * h),
            (r * p.heel_radius * 0.88, 0.004 * h),
            (r * p.heel_radius, 0.),
        ],
        ProfileJoints::Smooth,
        tf,
    );
    if p.opened {
        let gauge = p.end.gauge_m;
        let floor = h * 0.036;
        // Below the liquid, opaque aluminium and drink already occlude the can lining.
        let level = p.end.fill_fraction.map_or(floor, |f| h * f);
        let i = stations.partition_point(|&(_, y)| y < level);
        let (a0, y0) = stations[i - 1];
        let (a1, y1) = stations[i];
        let wet_radius = a0 + (a1 - a0) * (level - y0) / (y1 - y0) - gauge;
        let mut lining: Vec<_> = stations[i..]
            .iter()
            .rev()
            .filter(|s| s.1 > level + 1e-5)
            .map(|&(rad, y)| (rad - gauge, y))
            .collect();
        lining.push((wet_radius, level));
        if p.end.fill_fraction.is_none() {
            lining.push((0., level));
        }
        round.sweep(g, &lining, ProfileJoints::Smooth, tf);
        if p.end.fill_fraction.is_some() {
            round.sweep(
                a.part(Surface::Drink, package_print::DRINK),
                &[(wet_radius, level), (0., level)],
                ProfileJoints::Creased,
                tf,
            );
        }
    }
    wrap::side(
        a,
        (Surface::Metal, label),
        stations,
        (!p.bare).then(|| wrap::Wrap::new(o, Vec2::new(p.label_bottom, p.label_top) * h)),
        tf,
    );
    p.end.draw(a, &p, neck, h, end_y);
}
