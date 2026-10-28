//! Drawn-metal beverage cans with coordinated heel/neck/label and lid geometry.
//! Product dimensions are sampled separately, before support/collision admission.
use super::*;
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
}
pub fn parameters(o: &IndoorObject) -> CanProgram {
    let mut r = stream(o.seed, 0x43414e50524f4649);
    CanProgram {
        radius_fraction: r.random_range(0.39..0.49),
        height_fraction: r.random_range(0.92..1.),
        heel_height: r.random_range(0.025..0.105),
        heel_radius: r.random_range(0.76..0.95),
        shoulder: r.random_range(0.82..0.935),
        neck: r.random_range(0.73..0.95),
        waist: r.random_range(0.93..1.),
        label_bottom: r.random_range(0.12..0.32),
        label_top: r.random_range(0.66..0.91),
        bare: r.random_bool(0.12),
        opened: r.random_bool(0.64),
        tab_width: r.random_range(0.16..0.29),
        tab_length: r.random_range(0.23..0.37),
        lid_yaw: r.random_range(0.0..TAU),
    }
}
impl CanProgram {
    fn radius(&self, t: f32) -> f32 {
        if t < 0.5 {
            1. + (self.waist - 1.)
                * ((t - self.heel_height) / (0.5 - self.heel_height)).clamp(0., 1.)
        } else if t < self.shoulder {
            self.waist + (1. - self.waist) * (t - 0.5) / (self.shoulder - 0.5)
        } else {
            1. + (self.neck - 1.) * ((t - self.shoulder) / (0.965 - self.shoulder)).clamp(0., 1.)
        }
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let h = o.size.y * p.height_fraction;
    let r = o.size.x.min(o.size.z) * p.radius_fraction;
    let label = o.kind.class_name();
    a.part(Surface::Metal, label).lathe(
        &[
            (0., h * 0.02),
            (r * p.heel_radius, 0.),
            (r, h * p.heel_height),
            (r * p.waist, h * 0.5),
            (r, h * p.shoulder),
            (r * p.neck, h * 0.965),
            (0., h * 0.965),
        ],
        32,
        Transform::IDENTITY,
    );
    if !p.bare {
        let mut levels = vec![p.label_bottom, p.label_top];
        levels.extend(
            [0.5, p.shoulder]
                .into_iter()
                .filter(|&t| t > p.label_bottom && t < p.label_top),
        );
        levels.sort_by(f32::total_cmp);
        let profile: Vec<_> = levels
            .into_iter()
            .map(|t| (r * p.radius(t) + 0.00025, h * t))
            .collect();
        let g = a.part(Surface::Art, label);
        let start = g.positions.len();
        g.lathe(&profile, 32, Transform::IDENTITY);
        // Print wraps exactly once, independent of can diameter and shoulder
        // height. Substrate materials retain the normal metric lathe UVs.
        for (i, (v, uv)) in g.positions[start..]
            .iter()
            .zip(&mut g.uvs[start..])
            .enumerate()
        {
            *uv = [
                (i % 33) as f32 / 32.,
                (v[1] / h - p.label_bottom) / (p.label_top - p.label_bottom),
            ];
        }
    }
    for (y, radius) in [(h * 0.022, r * p.heel_radius), (h * 0.975, r * p.neck)] {
        a.part(Surface::Chrome, label).lathe(
            &[
                (radius * 0.94, y - 0.0008),
                (radius, y),
                (radius, y + 0.001),
                (radius * 0.94, y + 0.0012),
            ],
            32,
            Transform::IDENTITY,
        );
    }
    a.part(Surface::Chrome, label).cylinder(
        r * p.neck * 0.94,
        0.0012,
        Transform::from_xyz(0., h * 0.972, 0.),
    );
    let rotation = Quat::from_rotation_y(p.lid_yaw);
    let tab: Vec<_> = (0..=20)
        .map(|i| {
            let t = TAU * i as f32 / 20.;
            rotation
                * Vec3::new(
                    t.cos() * r * p.tab_width,
                    h * 0.983,
                    r * 0.12 + t.sin() * r * p.tab_length,
                )
        })
        .collect();
    a.part(Surface::Chrome, label)
        .tube(&tab, (r * 0.025).min(h * 0.008), 6);
    a.part(Surface::Chrome, label).cylinder(
        r * 0.07,
        0.0008,
        Transform::from_translation(rotation * Vec3::new(0., h * 0.981, -r * 0.18)),
    );
    if p.opened {
        a.part(Surface::Ink, label).ellipsoid(
            Vec3::new(r * 0.27, 0.00025, r * 0.30),
            Transform::from_translation(rotation * Vec3::new(0., h * 0.981, -r * 0.36))
                .with_rotation(rotation),
        );
    }
}
