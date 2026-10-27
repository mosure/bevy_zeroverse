//! Boundary, gooseneck and desktop capsule microphones with fitted head axes.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct MicrophoneProgram {
    pub construction: u8,
    pub base: Vec3,
    pub roundness: f32,
    pub head: Vec3,
    pub axis: Vec3,
    pub radius: f32,
    pub length: f32,
    pub shaft: f32,
    pub button: Vec2,
    pub finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> MicrophoneProgram {
    let mut r = stream(o.seed, 0x4d4943524f50484f);
    let s = o.size;
    let construction = if s.y < s.x.min(s.z) * 0.60 {
        0
    } else if r.random_bool(0.58) {
        1
    } else {
        2
    };
    let base = Vec3::new(
        s.x * r.random_range(0.66..0.97),
        s.y * if construction == 0 {
            r.random_range(0.65..0.93)
        } else {
            r.random_range(0.055..0.13)
        },
        s.z * r.random_range(0.65..0.96),
    );
    let radius = s.x.min(s.z)
        * if construction == 1 {
            r.random_range(0.055..0.10)
        } else {
            r.random_range(0.13..0.21)
        };
    let length = (s.y * r.random_range(0.15..0.32)).min(s.x.min(s.z) * 0.86);
    let axis = Vec3::new(r.random_range(-0.35..0.35), 1., r.random_range(0.15..0.95)).normalize();
    let reserve = Vec2::splat(radius) + axis.xz().abs() * length * 0.5;
    let freedom = (s.xz() * 0.45 - reserve).max(Vec2::ZERO);
    let head = Vec3::new(
        freedom.x * r.random_range(-0.8..0.8),
        s.y - length * 0.5 - radius * 0.7,
        freedom.y * r.random_range(0.1..0.85),
    );
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4d494346494e4953);
    MicrophoneProgram {
        construction,
        base,
        head,
        axis,
        radius,
        length,
        roundness: r.random_range(2.2..8.),
        shaft: s.x.min(s.z) * r.random_range(0.018..0.030),
        button: Vec2::new(r.random_range(-0.15..0.15), r.random_range(0.29..0.33)),
        finish: if f.random_bool(0.53) {
            Surface::Metal
        } else {
            Surface::Plastic
        },
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = p.base;
    let l = o.kind.class_name();
    a.part(p.finish, l).housing(
        &[
            (0., s.xz() * 0.44),
            (s.y * 0.25, s.xz() * 0.5),
            (s.y, s.xz() * 0.42),
        ],
        p.roundness,
        Transform::IDENTITY,
    );
    if p.construction == 0 {
        a.part(Surface::Fabric, l).housing(
            &[(s.y * 1.01, s.xz() * 0.29), (s.y * 1.03, s.xz() * 0.28)],
            p.roundness,
            Transform::from_xyz(0., 0., -s.z * 0.10),
        );
    } else {
        let end = p.head - p.axis * p.length * 0.46;
        let start = Vec3::new(0., s.y, -s.z * 0.16);
        let points: Vec<_> = (0..=16)
            .map(|i| {
                let t = i as f32 / 16.;
                let mut q = start.lerp(end, t);
                q.x = start.x + (end.x - start.x) * t * t;
                q.z = start.z + (end.z - start.z) * t * t;
                q
            })
            .collect();
        a.part(
            if p.construction == 1 {
                Surface::Rubber
            } else {
                Surface::Metal
            },
            l,
        )
        .tube(&points, p.shaft, 10);
        let tf = Transform::from_translation(p.head)
            .with_rotation(Quat::from_rotation_arc(Vec3::Y, p.axis));
        let r = p.radius;
        let h = p.length;
        a.part(p.finish, l).lathe(
            &[
                (0., -h * 0.5),
                (r * 0.62, -h * 0.5),
                (r, -h * 0.30),
                (r, h * 0.08),
                (0., h * 0.08),
            ],
            24,
            tf,
        );
        a.part(Surface::FabricAlt, l).lathe(
            &[
                (0., h * 0.081),
                (r, h * 0.081),
                (r * 0.94, h * 0.32),
                (r * 0.60, h * 0.46),
                (0., h * 0.5),
            ],
            24,
            tf,
        );
    }
    // Controls stay inside the flat top, beside the boundary grille/neck.
    a.box_part(
        Surface::Plastic,
        l,
        Vec3::new(s.x * p.button.x, s.y * 1.03, s.z * p.button.y),
        Vec3::new(s.x * 0.13, s.y * 0.04, s.z * 0.065),
        s.y * 0.008,
    );
}
