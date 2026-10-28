//! Wall-mounted life-safety and decorative electrical fixtures. Dimensions are
//! metric placement envelopes; all faces are separated, never coplanar decals.
use super::*;
pub mod exit;
mod lettering;

pub mod fire;
pub mod strings;
pub use strings::program as string_program;

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::ExitSign => exit::build(a, o),
        ObjectKind::FireAlarm | ObjectKind::FireDetector | ObjectKind::FirePullStation => {
            fire::build(a, o)
        }
        ObjectKind::StringLights => strings::build(a, o),
        _ => unreachable!(),
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn safety_housings_vary_actual_geometry_and_preserve_mounting_faces() {
        use super::super::super::audit;
        for kind in [
            ObjectKind::ExitSign,
            ObjectKind::FireAlarm,
            ObjectKind::FireDetector,
            ObjectKind::FirePullStation,
        ] {
            let mut representatives: Vec<Vec<f32>> = Vec::new();
            for seed in 0..32 {
                let o = audit::fixture(kind, seed);
                let a = super::super::super::build_object(&o);
                audit::check(&o, &a).unwrap();
                let (lo, hi) = a.bounds();
                if kind == ObjectKind::FireDetector
                    || (kind == ObjectKind::ExitSign && exit::mount(&o) != exit::ExitMount::Wall)
                {
                    assert!((hi.y - o.size.y).abs() < 1e-5, "detached ceiling detector");
                } else {
                    let mounting = super::super::super::wall::mount_offset(&o).unwrap() - 0.001;
                    assert!((lo.z + mounting).abs() < 1e-5, "detached {kind:?}/{seed}");
                }
                let descriptor = audit::descriptor(&o, &a);
                assert_eq!(
                    descriptor,
                    audit::descriptor(&o, &super::super::super::build_object(&o))
                );
                if representatives
                    .iter()
                    .all(|r| audit::distance(r, &descriptor) > 0.015)
                {
                    representatives.push(descriptor);
                }
            }
            assert!(
                representatives.len() >= 10,
                "collapsed {kind:?} geometry: {}",
                representatives.len()
            );
        }
    }
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
