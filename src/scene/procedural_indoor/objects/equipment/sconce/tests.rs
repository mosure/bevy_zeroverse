use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};
#[test]
fn sconces_share_mounting_bounds_emitter_and_finish_independent_construction() {
    let mut families = std::collections::BTreeSet::new();
    let mut heights = std::collections::BTreeSet::new();
    let mut up = 0;
    let mut canted = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::WallSconce, seed);
        o.size = dimensions(&mut stream(seed, 9892));
        let p = parameters(&o);
        let a = build_object(&o);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let (lo, hi) = a.bounds();
        assert!(
            (lo.z + o.size.z * 0.5).abs() < 1e-6,
            "{seed}: {lo:?} / {:?}",
            o.size
        );
        assert!(
            lo.cmpge(Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5) - Vec3::splat(1e-5))
                .all(),
            "{seed}: {lo:?}"
        );
        assert!(
            hi.cmple(Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5) + Vec3::splat(1e-5))
                .all(),
            "{seed}: {hi:?}"
        );
        let beam = fixture_beam(&o);
        let q = beam.position;
        up += usize::from(beam.direction.y > 0.);
        canted += usize::from(beam.direction.y.abs() < 0.997);
        assert_eq!(q, o.transform().transform_point(p.emitter));
        let front = a
            .parts
            .iter()
            .filter(|((s, _), _)| *s == Surface::Light)
            .flat_map(|(_, g)| &g.positions)
            .map(|v| {
                o.transform()
                    .transform_point(Vec3::from(*v))
                    .dot(beam.direction)
            })
            .fold(f32::NEG_INFINITY, f32::max);
        assert!((q.dot(beam.direction) - front - 0.002).abs() < 2e-5);
        let disk_area: f32 = a
            .parts
            .iter()
            .filter(|((s, _), _)| *s == Surface::Light)
            .flat_map(|(_, g)| {
                g.indices
                    .as_chunks::<3>()
                    .0
                    .iter()
                    .map(move |t| t.map(|i| Vec3::from(g.positions[i as usize])))
            })
            .filter_map(|t| {
                let n = (t[1] - t[0]).cross(t[2] - t[0]);
                let dir = p.mount_rotation * Vec3::NEG_Y;
                t.iter()
                    .all(|v| ((p.emitter - *v).dot(dir) - 0.002).abs() < 1e-5)
                    .then_some(n.dot(dir).max(0.) * 0.5)
            })
            .sum();
        let area = super::super::super::super::materials::emitters::area(&o);
        assert!(
            (disk_area / area - 1.).abs() < 0.025,
            "{seed}: {disk_area} {area}"
        );
        let replay = build_object(&o);
        for (key, g) in &a.parts {
            assert_eq!(g.positions, replay.parts[key].positions);
            assert_eq!(g.indices, replay.parts[key].indices);
        }
        families.insert((p.shade.shape, p.arm_style, p.plate_round));
        heights.insert((p.head_height * 1000.) as u32);
        o.finish_seed = Some(333);
        let alt = parameters(&o);
        assert_eq!(p.head_span, alt.head_span);
        assert_eq!(p.head_bottom, alt.head_bottom);
        assert_eq!(p.emitter, alt.emitter);
        assert_eq!(p.shade.shape, alt.shade.shape);
    }
    assert!((70..145).contains(&up));
    assert!((55..135).contains(&canted));
    assert_eq!(families.len(), 30);
    assert!(heights.len() > 100);
}
