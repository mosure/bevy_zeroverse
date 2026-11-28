//! Free-weight equipment in metres. The bench height is the installed backrest
//! height; its foot spacing and seat datum remain usable at every inclination.
use super::*;
use std::f32::consts::FRAC_PI_2;

pub(super) fn bench_dimensions(r: &mut impl Rng) -> Vec3 {
    let length = r.random_range(1.20..1.57);
    let seat = r.random_range(0.406..0.474);
    let incline = if r.random_bool(0.37) {
        0.
    } else {
        r.random_range(0.14_f32..1.12)
    };
    Vec3::new(
        r.random_range(0.46..0.70),
        seat + length * 0.59 * incline.sin(),
        length,
    )
}

#[derive(Serialize)]
#[serde(untagged)]
pub(crate) enum Fit {
    Bench(BenchFit),
    Rack(RackFit),
    Mat { radius: f32, face: Surface },
}

#[derive(Serialize)]
pub(crate) struct BenchFit {
    frame: Surface,
    seat_y: f32,
    pad: f32,
    width: f32,
    seat_width: f32,
    tilt: f32,
}

#[derive(Serialize)]
pub(crate) struct RackFit {
    frame: Surface,
    tiers: usize,
    slots: usize,
    segments: u32,
    stocking: f64,
    full_slot: usize,
    head: Surface,
}

pub(crate) fn parameters(o: &IndoorObject) -> Fit {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"GymShape"));
    let mut finish = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"GymStock"),
    );
    let frame = if finish.random_bool(0.65) {
        Surface::Metal
    } else {
        Surface::Accent
    };
    let s = o.size;
    match o.kind {
        ObjectKind::WeightBench => {
            let seat_y = if s.y < 0.50 {
                s.y
            } else {
                r.random_range(0.413..0.465)
            };
            let width = (s.x * r.random_range(0.57..0.74)).clamp(0.26, 0.37);
            Fit::Bench(BenchFit {
                frame,
                seat_y,
                width,
                pad: r.random_range(0.048..0.073),
                seat_width: width * r.random_range(0.85..1.02),
                tilt: ((s.y - seat_y) / (0.615 * s.z)).clamp(0., 0.95).asin(),
            })
        }
        ObjectKind::DumbbellRack => {
            let slots =
                (((s.x - 0.13) / r.random_range(0.195..0.254)).floor() as usize).clamp(2, 12);
            Fit::Rack(RackFit {
                frame,
                slots,
                tiers: if s.y > 0.94 && r.random_bool(0.5) {
                    3
                } else {
                    2
                },
                segments: if r.random_bool(0.62) { 6 } else { 16 },
                stocking: r.random_range(0.64..1.),
                full_slot: r.random_range(0..slots),
                head: if finish.random_bool(0.78) {
                    Surface::Rubber
                } else {
                    Surface::Metal
                },
            })
        }
        ObjectKind::ExerciseMat => Fit::Mat {
            radius: r.random_range(0.018..0.080),
            face: if finish.random_bool(0.45) {
                Surface::FabricAlt
            } else {
                Surface::Rubber
            },
        },
        _ => unreachable!("fitness equipment geometry"),
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match parameters(o) {
        Fit::Bench(fit) => bench(a, o, fit),
        Fit::Rack(fit) => rack(a, o, fit),
        Fit::Mat { radius, face } => mat(a, o, radius, face),
    }
}

fn bench(a: &mut Assembly, o: &IndoorObject, fit: BenchFit) {
    let BenchFit {
        frame,
        seat_y,
        pad,
        width,
        seat_width,
        tilt,
    } = fit;
    let s = o.size;
    let label = o.kind.class_name();
    let pivot = Vec3::new(0., seat_y, -0.14 * s.z);
    let back_length = 0.615 * s.z;
    let rotation = Quat::from_rotation_x(-tilt);
    let seat_end = pivot.z - 0.028;
    let seat_start = -s.z * 0.475;
    a.box_part(
        Surface::Leather,
        label,
        Vec3::new(0., seat_y - 0.5 * pad, (seat_end + seat_start) * 0.5),
        Vec3::new(seat_width, pad, seat_end - seat_start),
        pad * 0.28,
    );
    let back = |y, z| pivot + rotation * Vec3::new(0., y, z);
    a.part(Surface::Leather, label).cuboid(
        Vec3::new(width, pad, back_length),
        pad * 0.29,
        Transform::from_translation(back(-pad * 0.5, back_length * 0.5)).with_rotation(rotation),
    );

    let spine_y = seat_y - pad - 0.035;
    a.box_part(
        frame,
        label,
        Vec3::new(0., spine_y, -s.z * 0.015),
        Vec3::new(0.064, 0.072, s.z * 0.80),
        0.004,
    );
    for z in [-0.34 * s.z, 0.34 * s.z] {
        a.box_part(
            frame,
            label,
            Vec3::new(0., 0.055, z),
            Vec3::new(s.x - 0.03, 0.056, 0.062),
            0.005,
        );
        a.box_part(
            frame,
            label,
            Vec3::new(0., (spine_y + 0.055) * 0.5, z),
            Vec3::new(0.052, spine_y - 0.055, 0.055),
            0.004,
        );
        for side in [-1., 1.] {
            a.box_part(
                Surface::Rubber,
                label,
                Vec3::new(side * (s.x * 0.5 - 0.041), 0.041, z),
                Vec3::new(0.082, 0.082, 0.080),
                0.008,
            );
        }
    }
    // The upper rail follows its pad. A pin and diagonal prop carry its load
    // to the base even at full incline.
    a.part(frame, label).cuboid(
        Vec3::new(0.052, 0.040, back_length * 0.93),
        0.004,
        Transform::from_translation(back(-pad - 0.018, back_length * 0.48)).with_rotation(rotation),
    );
    a.part(Surface::Chrome, label).rod(
        pivot + Vec3::new(-0.061, -pad - 0.02, 0.),
        pivot + Vec3::new(0.061, -pad - 0.02, 0.),
        0.021,
    );
    a.part(frame, label).rod(
        Vec3::new(0., 0.10, s.z * 0.34),
        back(-pad - 0.02, back_length * 0.68),
        0.023,
    );
}

fn rack(a: &mut Assembly, o: &IndoorObject, fit: RackFit) {
    let RackFit {
        frame,
        tiers,
        slots,
        segments,
        stocking,
        full_slot,
        head,
    } = fit;
    let mut r = stream(o.seed, u64::from_le_bytes(*b"GymLoads"));
    let s = o.size;
    let label = o.kind.class_name();
    let pitch = (s.x - 0.13) / slots as f32;
    let max_radius = (pitch * 0.445).min(0.103);
    for side in [-1., 1.] {
        let x = side * (s.x * 0.5 - 0.035);
        a.box_part(
            Surface::Rubber,
            label,
            Vec3::new(x, 0.027, 0.),
            Vec3::new(0.068, 0.054, s.z),
            0.008,
        );
        a.part(frame, label).rod(
            Vec3::new(x, 0.036, s.z * 0.37),
            Vec3::new(x, s.y - 0.16, -s.z * 0.23),
            0.025,
        );
        a.part(frame, label).rod(
            Vec3::new(x, 0.036, -s.z * 0.37),
            Vec3::new(x, s.y - 0.16, -s.z * 0.23),
            0.022,
        );
    }
    for row in 0..tiers {
        let t = row as f32 / (tiers - 1) as f32;
        let top = 0.19 + t * (s.y - 2. * max_radius - 0.20);
        let z = (0.16 - 0.32 * t) * s.z;
        let half_grip = r.random_range(0.064..0.084);
        let head_width = r.random_range(0.057..0.086);
        let rail_z = half_grip + head_width * 0.5;
        for side in [-1., 1.] {
            a.box_part(
                frame,
                label,
                Vec3::new(0., top - 0.021, z + side * rail_z),
                Vec3::new(s.x - 0.065, 0.042, 0.041),
                0.003,
            );
        }
        // End brackets join both support rails to each upright.
        for side in [-1., 1.] {
            a.box_part(
                frame,
                label,
                Vec3::new(side * (s.x * 0.5 - 0.037), top - 0.041, z),
                Vec3::new(0.043, 0.046, 2. * rail_z + 0.05),
                0.003,
            );
        }
        for slot in 0..slots {
            if slot != full_slot && !r.random_bool(stocking) {
                continue;
            }
            let pair = (slot / 2) as f32 / ((slots - 1) / 2).max(1) as f32;
            let radius = max_radius * (0.64 + 0.30 * pair + 0.06 * (1. - t));
            let x = pitch * (slot as f32 + 0.5 - slots as f32 * 0.5);
            let y = top
                + if segments == 6 {
                    radius * 0.8660254
                } else {
                    radius
                };
            let axis = Quat::from_rotation_x(FRAC_PI_2);
            let tf = |along| Transform::from_xyz(x, y, z + along).with_rotation(axis);
            a.part(Surface::Chrome, label).cylinder(
                r.random_range(0.013..0.018),
                2. * half_grip + 0.012,
                tf(0.),
            );
            let w = head_width * 0.5;
            for side in [-1., 1.] {
                a.part(head, label).lathe(
                    &[
                        (0., -w),
                        (radius * 0.83, -w),
                        (radius, -w * 0.72),
                        (radius, w * 0.72),
                        (radius * 0.83, w),
                        (0., w),
                    ],
                    segments,
                    tf(side * rail_z),
                );
                if segments != 6 {
                    // Individually cradled round heads cannot roll down the rail.
                    for dx in [-0.56, 0.56] {
                        a.box_part(
                            Surface::Rubber,
                            label,
                            Vec3::new(x + dx * radius, top + 0.05 * radius, z + side * rail_z),
                            Vec3::new(0.022, 0.20 * radius, 0.035),
                            0.003,
                        );
                    }
                }
            }
        }
    }
}

fn mat(a: &mut Assembly, o: &IndoorObject, radius: f32, face: Surface) {
    let s = o.size;
    let half = s.xz() * 0.5;
    let corners = super::mats::rounded_outline(half, radius);
    let label = o.kind.class_name();
    let tf = Transform::from_xyz(0., s.y * 0.5, 0.);
    let profile = Geometry::slab_profile(&corners, s.y, 0.002);
    let rubber = a.part(Surface::Rubber, label);
    rubber.slab_edge(&corners, s.y, profile, &tf);
    rubber.slab_cap(&corners, profile.1, Vec2::ZERO, -s.y * 0.5, &tf, |v| v);
    a.part(face, label)
        .slab_cap(&corners, profile.1, Vec2::ZERO, s.y * 0.5, &tf, |v| v);
}
