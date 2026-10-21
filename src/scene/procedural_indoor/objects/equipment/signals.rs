//! Wall-mounted life-safety and decorative electrical fixtures. Dimensions are
//! metric placement envelopes; all faces are separated, never coplanar decals.
use super::*;
mod lettering;

pub mod fire;
pub mod strings;
pub use strings::program as string_program;

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::ExitSign => exit(a, o),
        ObjectKind::FireAlarm | ObjectKind::FireDetector | ObjectKind::FirePullStation => {
            fire::build(a, o)
        }
        ObjectKind::StringLights => strings::build(a, o),
        _ => unreachable!(),
    }
}
fn exit(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let l = o.kind.class_name();
    let mut r = stream(o.seed, 0x455849545349474e);
    let trim = [Surface::Plastic, Surface::Metal, Surface::Chrome][r.random_range(0..3)];
    a.box_part(
        trim,
        l,
        Vec3::new(0., s.y * 0.5, -s.z * 0.04),
        s * Vec3::new(1., 1., 0.92),
        (s.y * 0.04).min(0.008),
    );
    a.box_part(
        Surface::Ink,
        l,
        Vec3::new(0., s.y * 0.5, s.z * 0.42 + 0.0005),
        Vec3::new(s.x * 0.92, s.y * 0.84, 0.0008),
        0.002,
    );
    lettering::build(
        a,
        l,
        "EXIT",
        Vec2::new(s.x * 0.78, s.y * 0.53),
        Vec3::new(0., s.y * 0.5, s.z * 0.42 + 0.0014),
        Surface::Light,
    );
    // Test button and indicator, beside the main legend.
    a.part(Surface::Rubber, l).ellipsoid(
        Vec3::splat((s.y * 0.018).min(0.002)),
        Transform::from_xyz(s.x * 0.43, s.y * 0.13, s.z * 0.42),
    );
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn strand_reservations_contain_rendered_geometry_at_dimension_extremes() {
        for seed in 0..32 {
            for size in [Vec3::new(1.8, 0.20, 0.08), Vec3::new(7.5, 0.90, 1.2)] {
                let mut o = super::super::super::audit::fixture(ObjectKind::StringLights, seed);
                o.size = size;
                let p = string_program(&o);
                for bay in 0..p.bays {
                    let left = p.point(size, bay as f32 / p.bays as f32);
                    let right = p.point(size, (bay + 1) as f32 / p.bays as f32);
                    let mid = p.point(size, (bay as f32 + 0.5) / p.bays as f32);
                    assert!((mid.xz() - (left.xz() + right.xz()) * 0.5).length() < 1e-5);
                }
                let boxes = p.local_reservations(&o);
                let a = super::super::super::build_object(&o);
                super::super::super::audit::check(&o, &a).unwrap();
                for g in a.parts.values() {
                    for v in &g.positions {
                        let v = Vec3::from_array(*v);
                        assert!(
                            boxes
                                .iter()
                                .any(|&(lo, hi)| v.cmpge(lo - Vec3::splat(0.00001)).all()
                                    && v.cmple(hi + Vec3::splat(0.00001)).all()),
                            "unreserved strand vertex {seed} {v:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn signals_have_bounded_metric_geometry_and_typed_semantics() {
        for kind in [
            ObjectKind::ExitSign,
            ObjectKind::FireAlarm,
            ObjectKind::FireDetector,
            ObjectKind::FirePullStation,
            ObjectKind::StringLights,
        ] {
            for seed in 0..64 {
                let o = super::super::super::audit::fixture(kind, seed);
                let a = super::super::super::build_object(&o);
                super::super::super::audit::check(&o, &a).unwrap();
                assert!(a
                    .parts
                    .keys()
                    .all(|(_, l)| l.split('#').next() == Some(o.kind.class_name())));
            }
        }
    }
}
