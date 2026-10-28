//! Sharpened, mechanical and capped writing tools with fitted tips and clips.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct WritingProgram {
    pub mechanism: u8,
    pub length: f32,
    pub radius: f32,
    pub sides: u32,
    pub tip_fraction: f32,
    pub grip_fraction: f32,
    pub grip_radius: f32,
    pub tail_fraction: f32,
    pub eraser: bool,
    pub clip: bool,
    pub finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> WritingProgram {
    let mut r = stream(o.seed, 0x57524954494e4750);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x57524954494e4746);
    let mechanism = r.random_range(0..3);
    let clip = mechanism != 0 && r.random_bool(0.8);
    let radius = o.size.x.min(o.size.y) * r.random_range(0.30..if clip { 0.37 } else { 0.45 });
    WritingProgram {
        mechanism,
        length: o.size.z * r.random_range(0.76..0.99),
        radius,
        sides: if mechanism == 0 {
            [6, 8, 16][r.random_range(0..3)]
        } else {
            16
        },
        tip_fraction: r.random_range(0.06..0.19),
        grip_fraction: r.random_range(0.14..0.28),
        grip_radius: r.random_range(0.78..1.0),
        tail_fraction: r.random_range(0.04..0.12),
        eraser: mechanism == 0 && r.random_bool(0.55),
        clip,
        finish: if mechanism == 0 && f.random_bool(0.25) {
            Surface::Wood
        } else if mechanism != 0 && f.random_bool(0.40) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let l = o.kind.class_name();
    let h = p.length;
    let r = p.radius;
    // Put a polygon vertex on the supporting plane for every barrel section.
    let tf = Transform::from_xyz(0., r, h * 0.5)
        .with_rotation(Quat::from_rotation_x(-FRAC_PI_2) * Quat::from_rotation_y(FRAC_PI_2));
    let tail = if p.eraser || p.mechanism == 1 {
        h * p.tail_fraction
    } else {
        0.
    };
    let end = h * (1. - p.tip_fraction);
    if p.mechanism == 2 {
        // A cap sits on a narrower barrel neck, with a physical butt joint.
        let cap = h * (0.20 + p.grip_fraction * 0.40);
        a.part(p.finish, l).lathe(
            &[
                (0., 0.),
                (r * 0.75, 0.),
                (r, cap * 0.07),
                (r, cap * 0.96),
                (r * 0.84, cap * 0.96),
            ],
            p.sides,
            tf,
        );
        a.part(p.finish, l).lathe(
            &[
                (0., cap),
                (r * 0.84, cap),
                (r * 0.84, h * 0.94),
                (r * 0.50, h),
                (0., h),
            ],
            p.sides,
            tf,
        );
        a.part(Surface::Metal, l)
            .lathe(&[(r, cap * 0.96), (r, cap)], 16, tf);
    } else {
        let grip = if p.mechanism == 1 {
            end - h * p.grip_fraction
        } else {
            end
        };
        a.part(p.finish, l)
            .lathe(&[(0., tail), (r, tail), (r, grip), (0., grip)], p.sides, tf);
        if p.mechanism == 1 {
            a.part(Surface::Rubber, l).lathe(
                &[(0., grip), (r, grip), (r * p.grip_radius, end), (0., end)],
                16,
                tf,
            );
            a.part(Surface::Metal, l).lathe(
                &[(0., 0.), (r * 0.68, 0.), (r * 0.68, tail), (0., tail)],
                16,
                tf,
            );
        }
        a.part(
            if p.mechanism == 0 {
                Surface::Wood
            } else {
                Surface::Metal
            },
            l,
        )
        .lathe(
            &[
                (r * if p.mechanism == 0 { 1. } else { p.grip_radius }, end),
                (r * 0.16, h * 0.98),
            ],
            p.sides,
            tf,
        );
        a.part(Surface::Ink, l)
            .lathe(&[(r * 0.16, h * 0.98), (0., h)], 12, tf);
        if p.eraser {
            a.part(Surface::Rubber, l).lathe(
                &[
                    (0., 0.),
                    (r * 0.88, 0.),
                    (r * 0.88, tail * 0.65),
                    (0., tail * 0.65),
                ],
                16,
                tf,
            );
            a.part(Surface::Metal, l)
                .lathe(&[(r, tail * 0.65), (r, tail)], 16, tf);
        }
    }
    if p.clip {
        // Thin bent sheet-metal clip. A tube swept around near-hairpin bends
        // folds its cross-section on short pens; these fitted strips do not.
        let near = h * 0.46;
        let far = h * (0.38 - p.tail_fraction);
        let width = r * 0.45;
        a.box_part(
            Surface::Metal,
            l,
            Vec3::new(0., r * 2.13, near),
            Vec3::new(width, r * 0.66, r * 0.24),
            r * 0.05,
        );
        a.box_part(
            Surface::Metal,
            l,
            Vec3::new(0., r * 2.45, (near + far) * 0.5),
            Vec3::new(width, r * 0.22, near - far + r * 0.24),
            r * 0.05,
        );
        a.box_part(
            Surface::Metal,
            l,
            Vec3::new(0., r * 2.20, far),
            Vec3::new(width, r * 0.50, r * 0.24),
            r * 0.05,
        );
    }
}
