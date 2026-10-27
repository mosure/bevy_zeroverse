//! Stapler levers and tape cradles retain their mechanical relationships while
//! varying occupied dimensions, materials and operating/resting configuration.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct StaplerProgram {
    pub base: Vec3,
    pub hinge: Vec3,
    pub arm: Vec3,
    pub opening: f32,
    pub nose_taper: f32,
    pub roundness: f32,
    pub finish: Surface,
    pub base_finish: Surface,
}
pub fn stapler_parameters(o: &IndoorObject) -> StaplerProgram {
    let mut r = stream(o.seed, 0x535441504c455250);
    let s = o.size;
    let base = Vec3::new(
        s.x * r.random_range(0.81..0.99),
        s.y * r.random_range(0.10..0.23),
        s.z * r.random_range(0.85..0.99),
    );
    let hinge = Vec3::new(
        0.,
        base.y + s.y * r.random_range(0.14..0.23),
        -base.z * r.random_range(0.30..0.39),
    );
    let length = base.z * r.random_range(0.72..0.83);
    let maximum = ((s.y - hinge.y - s.y * 0.18) / length)
        .clamp(0., 0.65)
        .asin();
    let opening = maximum * r.random_range(0.02..0.90);
    let thickness =
        ((s.y - hinge.y - length * opening.sin()) / opening.cos() * 0.75).min(s.y * 0.39);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x535441504c454649);
    StaplerProgram {
        base,
        hinge,
        arm: Vec3::new(base.x * r.random_range(0.70..0.94), thickness, length),
        opening,
        nose_taper: r.random_range(0.56..0.94),
        roundness: r.random_range(3.0..9.),
        finish: if f.random_bool(0.6) {
            Surface::Plastic
        } else {
            Surface::Metal
        },
        base_finish: if f.random_bool(0.7) {
            Surface::Rubber
        } else {
            Surface::Metal
        },
    }
}
pub(super) fn stapler(a: &mut Assembly, o: &IndoorObject) {
    let p = stapler_parameters(o);
    let l = o.kind.class_name();
    let b = p.base;
    let s = p.arm;
    a.part(p.base_finish, l).housing(
        &[
            (0., b.xz() * 0.45),
            (b.y * 0.35, b.xz() * 0.5),
            (b.y, b.xz() * 0.46),
        ],
        p.roundness,
        Transform::IDENTITY,
    );
    let rotation = Quat::from_rotation_x(-p.opening);
    let tf = Transform::from_translation(p.hinge).with_rotation(rotation);
    let case_tf = tf
        .mul_transform(Transform::from_xyz(0., s.y * 0.5, 0.))
        .with_rotation(rotation * Quat::from_rotation_x(FRAC_PI_2));
    a.part(p.finish, l).housing(
        &[
            (0., Vec2::new(s.x * 0.42, s.y * 0.40)),
            (s.z * 0.18, Vec2::new(s.x * 0.5, s.y * 0.5)),
            (s.z * 0.80, Vec2::new(s.x * 0.48, s.y * 0.47)),
            (s.z, Vec2::new(s.x * p.nose_taper * 0.5, s.y * 0.32)),
        ],
        p.roundness,
        case_tf,
    );
    a.part(Surface::Metal, l).cuboid(
        Vec3::new(s.x * 0.68, s.y * 0.14, s.z * 0.94),
        s.y * 0.025,
        tf.mul_transform(Transform::from_xyz(0., s.y * 0.015, s.z * 0.49)),
    );
    let axle = o.size.y * 0.09;
    a.part(Surface::Metal, l).cylinder(
        axle,
        b.x * 0.96,
        Transform::from_translation(p.hinge + Vec3::Y * axle * 0.25)
            .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    );
    // The rear support carries the axle; the front anvil stays on the base.
    a.box_part(
        p.finish,
        l,
        Vec3::new(0., (b.y + p.hinge.y) * 0.5, p.hinge.z),
        Vec3::new(b.x * 0.70, p.hinge.y - b.y, axle * 1.5),
        axle * 0.1,
    );
    a.box_part(
        Surface::Metal,
        l,
        Vec3::new(0., b.y + 0.0006, b.z * 0.32),
        Vec3::new(b.x * 0.42, 0.0012, b.z * 0.14),
        0.0002,
    );
}

#[derive(Clone, Debug, serde::Serialize)]
pub struct TapeProgram {
    pub base: Vec3,
    pub center: Vec3,
    pub radius: f32,
    pub core: f32,
    pub width: f32,
    pub blade: Vec3,
    pub tangent: Vec3,
    pub roundness: f32,
    pub open_frame: bool,
    pub web: bool,
    pub teeth: u32,
    pub finish: Surface,
    pub tape: Surface,
}
pub fn tape_parameters(o: &IndoorObject) -> TapeProgram {
    let mut r = stream(o.seed, 0x5441504543524144);
    let s = o.size;
    let base = Vec3::new(
        s.x * r.random_range(0.84..0.99),
        s.y * r.random_range(0.09..0.19),
        s.z * r.random_range(0.89..0.99),
    );
    let radius = ((s.y - base.y) * r.random_range(0.34..0.48)).min(base.z * 0.25);
    let center = Vec3::new(
        0.,
        base.y + radius + 0.0005,
        -base.z * r.random_range(0.11..0.21),
    );
    let blade = Vec3::new(0., s.y * r.random_range(0.26..0.43), base.z * 0.42);
    let v = (blade + Vec3::Y * 0.0006).yz() - center.yz();
    let len = v.length();
    let axis = v / len;
    let q = radius / len;
    let perp = Vec2::new(axis.y, -axis.x);
    let tangent = center
        + Vec3::new(
            0.,
            (axis * q + perp * (1. - q * q).sqrt()).x * radius,
            (axis * q + perp * (1. - q * q).sqrt()).y * radius,
        );
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x5441504546494e49);
    TapeProgram {
        base,
        center,
        radius,
        blade,
        tangent,
        core: r.random_range(0.34..0.63),
        width: base.x * r.random_range(0.40..0.60),
        roundness: r.random_range(2.5..8.),
        open_frame: r.random_bool(0.45),
        web: r.random_bool(0.73),
        teeth: r.random_range(5..=12),
        finish: [Surface::Plastic, Surface::Metal, Surface::Wood][f.random_range(0..3)],
        tape: if f.random_bool(0.55) {
            Surface::Paper
        } else {
            Surface::Plastic
        },
    }
}
pub(super) fn tape(a: &mut Assembly, o: &IndoorObject) {
    let p = tape_parameters(o);
    let l = o.kind.class_name();
    let b = p.base;
    let r = p.radius;
    a.part(p.finish, l).housing(
        &[
            (0., b.xz() * 0.47),
            (b.y * 0.3, b.xz() * 0.5),
            (b.y, b.xz() * 0.46),
        ],
        p.roundness,
        Transform::IDENTITY,
    );
    for side in [-1., 1.] {
        let x = side * b.x * 0.39;
        if p.open_frame {
            a.part(p.finish, l).rod(
                Vec3::new(x, b.y, p.center.z - b.z * 0.10),
                Vec3::new(x, p.center.y, p.center.z),
                b.x * 0.055,
            );
        } else {
            a.box_part(
                p.finish,
                l,
                Vec3::new(x, (b.y + p.center.y) * 0.5, p.center.z),
                Vec3::new(b.x * 0.10, p.center.y - b.y, r * 1.25),
                b.x * 0.035,
            );
        }
    }
    let tf = Transform::from_translation(p.center).with_rotation(Quat::from_rotation_z(FRAC_PI_2));
    let c = r * p.core;
    let w = p.width;
    a.part(Surface::Paper, l).lathe(
        &[
            (c, -w * 0.52),
            (c + 0.001, -w * 0.52),
            (c + 0.001, w * 0.52),
            (c, w * 0.52),
            (c, -w * 0.52),
        ],
        32,
        tf,
    );
    a.part(p.tape, l).lathe(
        &[
            (c + 0.0011, -w * 0.5),
            (r, -w * 0.5),
            (r, w * 0.5),
            (c + 0.0011, w * 0.5),
            (c + 0.0011, -w * 0.5),
        ],
        32,
        tf,
    );
    a.part(Surface::Plastic, l)
        .cylinder(c * 0.70, b.x * 0.87, tf);
    let foot = Vec3::new(0., (b.y + p.blade.y) * 0.5, p.blade.z);
    a.box_part(
        p.finish,
        l,
        foot,
        Vec3::new(b.x * 0.88, p.blade.y - b.y, b.z * 0.065),
        b.x * 0.04,
    );
    a.box_part(
        Surface::Metal,
        l,
        p.blade,
        Vec3::new(b.x * 0.91, 0.0009, b.z * 0.045),
        0.0001,
    );
    for i in 0..p.teeth {
        a.box_part(
            Surface::Metal,
            l,
            p.blade
                + Vec3::new(
                    (i as f32 / (p.teeth - 1) as f32 - 0.5) * b.x * 0.84,
                    0.,
                    b.z * 0.027,
                ),
            Vec3::new(b.x * 0.38 / p.teeth as f32, 0.0009, b.z * 0.009),
            0.0001,
        );
    }
    if p.web {
        let end = p.blade + Vec3::Y * 0.0006;
        let d = end - p.tangent;
        a.part(p.tape, l).cuboid(
            Vec3::new(w, 0.00018, d.length()),
            0.,
            Transform::from_translation((p.tangent + end) * 0.5)
                .with_rotation(Quat::from_rotation_arc(Vec3::Z, d.normalize())),
        );
    }
}
