use super::*;
use crate::scene::procedural_indoor::objects::{audit, build_object};
#[test]
fn sconces_share_mounting_bounds_emitter_and_finish_independent_construction() {
    let mut families = std::collections::BTreeSet::new();
    let mut heights = std::collections::BTreeSet::new();
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
        let q = fixture_photometry(&o).0;
        assert_eq!(q, o.transform().transform_point(p.emitter));
        let min_light = a
            .parts
            .iter()
            .filter(|((s, _), _)| *s == Surface::Light)
            .flat_map(|(_, g)| &g.positions)
            .map(|v| v[1])
            .fold(f32::INFINITY, f32::min);
        assert!(q.y < min_light);
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
    assert_eq!(families.len(), 30);
    assert!(heights.len() > 100);
}
