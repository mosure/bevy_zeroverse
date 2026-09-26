//! Dimensioned office equipment and stored belongings; parts retain surface roles.
use super::*;
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let mut rng = stream(o.seed, 218);
    match o.kind {
        ObjectKind::Printer => {
            let split = rng.random_range(0.48..0.70);
            a.box_part(
                Surface::Plastic,
                "other_prop",
                Vec3::Y * s.y * split * 0.5,
                Vec3::new(s.x, s.y * split, s.z),
                0.012,
            );
            a.box_part(
                Surface::Ceramic,
                "other_prop",
                Vec3::Y * s.y * (split + (0.98 - split) * 0.5),
                Vec3::new(s.x * 0.97, s.y * (0.98 - split), s.z * 0.97),
                0.007,
            );
            a.box_part(
                Surface::Ink,
                "other_prop",
                Vec3::new(0.0, s.y * 0.35, s.z * 0.499),
                Vec3::new(s.x * 0.76, s.y * 0.10, 0.002),
                0.0,
            );
            a.box_part(
                Surface::Paper,
                "paper",
                Vec3::new(0.0, s.y * 0.405, s.z * 0.25),
                Vec3::new(s.x * 0.62, 0.004, s.z * 0.48),
                0.0,
            );
            a.box_part(
                Surface::Screen,
                "other_prop",
                Vec3::new(-s.x * 0.29, s.y - 0.0005, s.z * 0.27),
                Vec3::new(s.x * 0.20, 0.001, s.z * 0.21),
                0.0,
            );
        }
        ObjectKind::StorageBox => {
            a.box_part(
                Surface::Paper,
                "other_prop",
                Vec3::Y * s.y * 0.46,
                Vec3::new(s.x * 0.95, s.y * 0.92, s.z * 0.95),
                0.003,
            );
            a.box_part(
                Surface::Plastic,
                "other_prop",
                Vec3::Y * s.y * 0.96,
                Vec3::new(s.x, s.y * 0.08, s.z),
                0.004,
            );
            a.box_part(
                Surface::Ink,
                "other_prop",
                Vec3::new(0.0, s.y * 0.64, s.z * 0.476),
                Vec3::new(s.x * 0.26, s.y * 0.07, 0.001),
                0.003,
            );
        }
        ObjectKind::CoatRack => {
            let r = s.x.min(s.z) * 0.43;
            let metal = a.part(Surface::Metal, "other_prop");
            metal.rod(Vec3::Y * 0.03, Vec3::Y * s.y * 0.96, 0.018);
            let arms = rng.random_range(3..7);
            for i in 0..arms {
                let t = i as f32 * TAU / arms as f32;
                let dir = Vec3::new(t.cos(), 0.0, t.sin());
                metal.rod(Vec3::Y * 0.07, dir * r + Vec3::Y * 0.025, 0.014);
                metal.ellipsoid(
                    Vec3::new(0.025, 0.010, 0.025),
                    Transform::from_translation(dir * r + Vec3::Y * 0.010),
                );
                let end = dir * r * 0.70 + Vec3::Y * s.y * 0.94;
                metal.rod(Vec3::Y * s.y * 0.79, end, 0.012);
                metal.ellipsoid(Vec3::splat(0.019), Transform::from_translation(end));
            }
        }
        ObjectKind::Bag => {
            let body = rng.random_range(0.66..0.79);
            a.box_part(
                Surface::FabricAlt,
                "other_prop",
                Vec3::Y * s.y * body * 0.5,
                Vec3::new(s.x, s.y * body, s.z),
                s.z.min(s.x) * 0.12,
            );
            a.box_part(
                Surface::Fabric,
                "other_prop",
                Vec3::new(0.0, s.y * 0.26, s.z * 0.47),
                Vec3::new(s.x * 0.66, s.y * 0.35, s.z * 0.06),
                0.01,
            );
            for z in [-s.z * 0.27, s.z * 0.27] {
                for i in 0..12 {
                    let point = |t: f32| {
                        Vec3::new(
                            t.cos() * s.x * 0.25,
                            s.y * (body + (0.98 - body) * t.sin()),
                            z,
                        )
                    };
                    a.part(Surface::Rubber, "other_prop").rod(
                        point(i as f32 * PI / 12.0),
                        point((i + 1) as f32 * PI / 12.0),
                        0.004,
                    );
                }
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
