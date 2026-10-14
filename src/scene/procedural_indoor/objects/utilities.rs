//! Dimensioned office equipment and belongings with independently sampled mechanisms.
use super::*;
#[derive(Debug, Clone, serde::Serialize)]
pub struct UtilityProgram {
    pub construction: u8,
    pub taper: f32,
    pub roundness: f32,
    pub split: f32,
    pub lid_fraction: f32,
    pub handle_span: f32,
    pub arms: usize,
    pub branch_height: f32,
}
pub fn parameters(o: &IndoorObject) -> UtilityProgram {
    let mut r = stream(o.seed, 7430);
    UtilityProgram {
        construction: r.random_range(0..4),
        taper: r.random_range(0.72..0.98),
        roundness: r.random_range(3.5..9.),
        split: r.random_range(0.42..0.72),
        lid_fraction: r.random_range(0.07..0.19),
        handle_span: r.random_range(0.28..0.58),
        arms: r.random_range(3..=6),
        branch_height: r.random_range(0.62..0.84),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    match o.kind {
        ObjectKind::StorageBox => {
            let lid = s.y * p.lid_fraction;
            let body = s.y - lid - 0.001;
            let finish = if p.construction < 2 {
                Surface::Plastic
            } else {
                Surface::Paper
            };
            a.part(finish, l).housing(
                &[(0., s.xz() * 0.46 * p.taper), (body, s.xz() * 0.47)],
                p.roundness,
                Transform::IDENTITY,
            );
            a.box_part(
                if p.construction == 3 {
                    Surface::FabricAlt
                } else {
                    finish
                },
                l,
                Vec3::Y * (s.y - lid * 0.5),
                Vec3::new(s.x, lid, s.z),
                s.y * 0.035,
            );
            if p.construction.is_multiple_of(2) {
                for side in [-1., 1.] {
                    a.box_part(
                        Surface::Rubber,
                        l,
                        Vec3::new(
                            side * s.x * (0.46 * p.taper + (0.47 - 0.46 * p.taper) * 0.78 + 0.006),
                            body * 0.78,
                            0.,
                        ),
                        Vec3::new(s.x * 0.025, s.y * 0.12, s.z * p.handle_span),
                        s.y * 0.025,
                    );
                }
            } else {
                a.box_part(
                    Surface::Ink,
                    l,
                    Vec3::new(
                        0.,
                        body * 0.70,
                        s.z * (0.46 * p.taper + (0.47 - 0.46 * p.taper) * 0.70 + 0.004),
                    ),
                    Vec3::new(s.x * p.handle_span, s.y * 0.08, s.z * 0.016),
                    s.y * 0.025,
                );
            }
            if p.construction == 1 {
                for side in [-1., 1.] {
                    a.box_part(
                        Surface::Metal,
                        l,
                        Vec3::new(side * s.x * 0.31, s.y - lid * 0.35, s.z * 0.47),
                        Vec3::new(s.x * 0.11, lid * 0.6, s.z * 0.035),
                        0.002,
                    );
                }
            }
        }
        ObjectKind::Printer => {
            let split = p.split;
            let lid_y = s.y * (0.93 + p.lid_fraction * 0.20);
            let body_top = lid_y / s.y - 0.016;
            let top_width = s.x * (if p.construction == 0 { 0.84 } else { 0.97 });
            a.part(Surface::Plastic, l).housing(
                &[(0., s.xz() * 0.46), (s.y * split, s.xz() * 0.49)],
                p.roundness,
                Transform::IDENTITY,
            );
            a.box_part(
                Surface::Plastic,
                l,
                Vec3::new(0., s.y * (split + (body_top - split) * 0.5), -s.z * 0.025),
                Vec3::new(top_width, s.y * (body_top - split), s.z * 0.92),
                0.008,
            );
            // A feed lid and output ledge, with paper resting on its upper face.

            a.box_part(
                Surface::Plastic,
                l,
                Vec3::new(0., lid_y, -s.z * 0.03),
                Vec3::new(top_width * 0.92, s.y * 0.032, s.z * (0.60 + p.lid_fraction)),
                0.005,
            );
            let ledge = s.y * (split * 0.58);
            a.box_part(
                Surface::Plastic,
                l,
                Vec3::new(0., ledge, s.z * 0.36),
                Vec3::new(s.x * 0.78, s.y * 0.025, s.z * 0.26),
                0.004,
            );
            a.box_part(
                Surface::Ink,
                l,
                Vec3::new(0., ledge + s.y * 0.055, s.z * 0.483),
                Vec3::new(s.x * 0.72, s.y * 0.07, 0.001),
                0.001,
            );
            a.box_part(
                Surface::Paper,
                "paper",
                Vec3::new(0., ledge + s.y * 0.015, s.z * 0.36),
                Vec3::new(s.x * 0.65, s.y * 0.005, s.z * 0.25),
                0.0001,
            );
            let x = s.x * (if p.construction < 2 { -0.29 } else { 0.23 });
            a.box_part(
                Surface::Screen,
                l,
                Vec3::new(x, s.y * 0.79, s.z * 0.438),
                Vec3::new(s.x * 0.22, s.y * (0.05 + p.lid_fraction * 0.22), 0.001),
                0.0003,
            );
        }
        ObjectKind::CoatRack => {
            let r = s.x.min(s.z) * 0.43;
            let material = if p.construction == 0 {
                Surface::Wood
            } else {
                Surface::Metal
            };
            a.part(material, l)
                .rod(Vec3::Y * 0.025, Vec3::Y * s.y * 0.96, 0.013);
            if p.construction == 1 {
                a.part(Surface::Metal, l).cylinder(
                    r * 0.83,
                    0.025,
                    Transform::from_xyz(0., 0.0125, 0.),
                );
            }
            for i in 0..p.arms {
                let t = i as f32 * TAU / p.arms as f32;
                let d = Vec3::new(t.cos(), 0., t.sin());
                if p.construction != 1 {
                    a.part(material, l)
                        .rod(Vec3::Y * s.y * 0.12, d * r + Vec3::Y * 0.012, 0.012);
                    a.part(Surface::Rubber, l).ellipsoid(
                        Vec3::new(0.015, 0.012, 0.015),
                        Transform::from_translation(d * r + Vec3::Y * 0.012),
                    );
                }
                let end = d * r * (0.68 + p.handle_span * 0.3)
                    + Vec3::Y * s.y * (0.88 + 0.05 * (i % 2) as f32);
                a.part(material, l)
                    .rod(Vec3::Y * s.y * p.branch_height, end, 0.009);
                a.part(Surface::Rubber, l)
                    .ellipsoid(Vec3::splat(0.014), Transform::from_translation(end));
            }
        }
        ObjectKind::Bag => {
            let body = s.y * (0.61 + p.lid_fraction);
            let w = s.x * 0.95;
            let d = s.z * 0.93;
            a.part(
                if p.construction < 2 {
                    Surface::Leather
                } else {
                    Surface::FabricAlt
                },
                l,
            )
            .housing(
                &[
                    (0., Vec2::new(w, d) * 0.46),
                    (body * 0.72, Vec2::new(w, d) * 0.5),
                    (body, Vec2::new(w, d) * 0.5 * p.taper),
                ],
                p.roundness,
                Transform::IDENTITY,
            );
            for side in [-1., 1.] {
                let points: Vec<_> = (0..=16)
                    .map(|i| {
                        let t = PI * i as f32 / 16.;
                        Vec3::new(
                            t.cos() * w * p.handle_span * 0.5,
                            body + (s.y - body - 0.005) * t.sin(),
                            side * d * 0.22,
                        )
                    })
                    .collect();
                a.part(Surface::Leather, l).tube(&points, 0.004, 8);
            }
            if p.construction > 1 {
                a.box_part(
                    Surface::Fabric,
                    l,
                    Vec3::new(0., body * 0.40, d * 0.48),
                    Vec3::new(w * 0.70, body * 0.45, d * 0.045),
                    d * 0.02,
                );
            }
        }
        _ => unreachable!(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn utilities_keep_all_parts_inside_placement_envelopes() {
        for kind in [
            ObjectKind::Printer,
            ObjectKind::StorageBox,
            ObjectKind::CoatRack,
            ObjectKind::Bag,
        ] {
            for seed in 0..12 {
                let size = if kind == ObjectKind::CoatRack {
                    Vec3::new(0.55, 1.5, 0.55)
                } else {
                    Vec3::new(0.4, 0.28, 0.3)
                };
                let o = IndoorObject {
                    finish_seed: None,
                    id: 0,
                    kind,
                    position: Vec3::ZERO,
                    size,
                    yaw: 0.0,
                    variant: 0,
                    seed,
                    solid: true,
                    support: None,
                    neighbor: false,
                    interaction_target: None,
                    cable_path: None,
                    cable_target: None,
                };
                let a = super::super::build_object(&o);
                let (lo, hi) = a.bounds();
                assert!(lo.is_finite() && hi.is_finite());
                assert!(
                    lo.cmpge(Vec3::new(-size.x * 0.5, 0.0, -size.z * 0.5) - Vec3::splat(0.001))
                        .all(),
                    "{kind:?} lower {lo:?}"
                );
                assert!(
                    hi.cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.001))
                        .all(),
                    "{kind:?} upper {hi:?}"
                );
            }
        }
    }
}
