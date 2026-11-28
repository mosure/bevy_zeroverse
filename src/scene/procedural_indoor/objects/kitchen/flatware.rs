//! Stamped table cutlery. The hollow spoon and separated fork tines share a
//! tapered handle that actually rests on the supporting surface.
use super::*;
mod handle;
pub use handle::FittedGrip;
const FITTED_GRIP_VERSION: u32 = 264;

#[derive(Debug, Serialize)]
pub struct FlatwareProgram {
    pub tines: u8, // Zero denotes a spoon.
    pub head_length_fraction: f32,
    pub handle_width_fraction: f32,
    pub metal: Surface,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub fitted_grip: Option<FittedGrip>,
}

pub fn parameters(o: &IndoorObject) -> FlatwareProgram {
    parameters_at(
        o,
        crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
    )
}

pub(super) fn parameters_at(o: &IndoorObject, version: u32) -> FlatwareProgram {
    let mut r = stream(o.seed, 0x4355544c455259);
    let mut p = FlatwareProgram {
        tines: if r.random_bool(0.54) {
            r.random_range(3..=4)
        } else {
            0
        },
        head_length_fraction: r.random_range(0.25..0.38),
        handle_width_fraction: r.random_range(0.31..0.55),
        metal: if r.random_bool(0.75) {
            Surface::Chrome
        } else {
            Surface::Metal
        },
        fitted_grip: None,
    };
    if version >= FITTED_GRIP_VERSION {
        let mut stock = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"CutStock"),
        );
        p.metal = if stock.random_bool(0.73) {
            Surface::Chrome
        } else {
            Surface::Metal
        };
        p.handle_width_fraction = stock.random_range(0.31..0.55);
        p.fitted_grip = FittedGrip::sample(&mut stock);
    }
    p
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters_at(o, version);
    build_sampled(a, o, &p);
}

pub(super) fn build_sampled(a: &mut Assembly, o: &IndoorObject, p: &FlatwareProgram) {
    let s = o.size;
    let slot = super::super::super::materials::variants::object_slot(
        p.metal,
        o.finish_seed.unwrap_or(o.seed),
        None,
    );
    let head_len = s.z * p.head_length_fraction;
    let nose = 0.485 * s.z;
    let tail = -0.485 * s.z;
    let neck = nose - head_len;
    let radius = s.x * p.handle_width_fraction * 0.5;
    let half_height = s.y * 0.24;
    // A spoon shank climbs to the near rim. Extending a flat handle beneath the
    // centre would pierce the concave eating surface.
    let (run, rise, tip) = if p.tines == 0 {
        (
            neck - tail + head_len * 0.018,
            (s.y * 0.82 - half_height) / 0.92,
            0.31,
        )
    } else {
        (neck - tail + head_len * 0.16, 0., 0.54)
    };
    let metal_label = format!("kitchen_utensil#finish{slot}");
    if let Some(grip) = &p.fitted_grip {
        grip.build(
            a,
            o,
            p,
            &metal_label,
            tail..tail + run,
            half_height + rise * 0.92,
        );
    } else {
        let reach = run.hypot(rise);
        a.part(p.metal, &metal_label).lathe(
            &[
                (0., 0.),
                (radius * 0.64, 0.),
                (radius, reach * 0.08),
                (radius * 0.80, reach * 0.42),
                (radius * 0.45, reach * 0.83),
                (radius * tip, reach),
                (0., reach),
            ],
            12,
            Transform::from_xyz(0., half_height - rise * 0.08, tail)
                .with_rotation(Quat::from_rotation_x(run.atan2(rise)))
                .with_scale(Vec3::new(1., 1., half_height / radius)),
        );
    }
    let g = a.part(p.metal, &metal_label);
    let w = s.x * 0.47;
    if p.tines == 0 {
        // Exterior and interior profiles meet at a rolled rim. Normal direction
        // changes across the rim, so highlights describe the concave scoop.
        let h = s.y * 0.90;
        let t = (s.y * 0.12).min(0.0015);
        g.lathe(
            &[
                (0., 0.),
                (w * 0.17, 0.),
                (w * 0.62, h * 0.27),
                (w, h - t),
                (w, h),
                (w * 0.93, h),
                (w * 0.58, h * 0.27 + t),
                (w * 0.17, t),
                (0., t),
            ],
            24,
            Transform::from_xyz(0., 0., neck + head_len * 0.5).with_scale(Vec3::new(
                1.,
                1.,
                head_len * 0.5 / w,
            )),
        );
        return;
    }
    let heel_depth = head_len * 0.41;
    let heel = heel_depth * 0.5;
    g.profile_slab(
        &[
            Vec2::new(-w * 0.25, -heel),
            Vec2::new(w * 0.25, -heel),
            Vec2::new(w * 0.95, heel * 0.28),
            Vec2::new(w, heel),
            Vec2::new(-w, heel),
            Vec2::new(-w * 0.95, heel * 0.28),
        ],
        s.y * 0.31,
        s.y * 0.075,
        Transform::from_xyz(0., s.y * 0.29, neck + heel),
    );
    for tine in 0..p.tines {
        let u = (tine as f32 + 0.5) / p.tines as f32 * 2. - 1.;
        let start = Vec3::new(w * u, s.y * 0.30, neck + heel_depth * 0.94);
        let end = Vec3::new(w * u * 0.93, s.y * 0.83, nose - u.abs() * head_len * 0.05);
        let d = end - start;
        let radius = w * 0.67 / p.tines as f32;
        let length = d.length();
        g.lathe(
            &[
                (0., 0.),
                (radius, 0.),
                (radius * 0.75, length * 0.72),
                (radius * 0.30, length * 0.97),
                (0., length),
            ],
            8,
            Transform::from_translation(start)
                .with_rotation(
                    Quat::from_rotation_y(d.x.atan2(d.z))
                        * Quat::from_rotation_x(d.xz().length().atan2(d.y)),
                )
                .with_scale(Vec3::new(1., 1., s.y * 0.13 / radius)),
        );
    }
}
