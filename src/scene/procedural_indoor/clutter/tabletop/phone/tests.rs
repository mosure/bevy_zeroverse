use super::*;
use crate::scene::procedural_indoor::{
    items::sizing,
    objects::{audit, build_object},
};
#[test]
fn handsets_fit_replay_and_preserve_screen_and_semantic_contracts() {
    let mut layouts = std::collections::BTreeSet::new();
    let mut dimensions = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Phone, seed);
        o.size = sizing::loose(o.kind, &mut stream(seed, 9891)).unwrap();
        let p = parameters(&o);
        let a = build_object(&o);
        audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let (lo, hi) = a.bounds();
        assert!(lo.y.abs() < 1e-6);
        assert!(lo
            .cmpge(Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5) - Vec3::splat(1e-5))
            .all());
        assert!(
            hi.cmple(Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5) + Vec3::splat(1e-5))
                .all(),
            "{seed}: {hi:?} / {:?}",
            o.size
        );
        for index in 0..32 {
            let theta = TAU * index as f32 / 32.;
            let shape = |v: f32| v.signum() * v.abs().powf(2. / p.island_roundness);
            let point = p.island_center
                + p.island_span * 0.5 * Vec2::new(shape(theta.cos()), shape(theta.sin()));
            let normalized = point.abs() / (p.span * 0.47);
            assert!(
                normalized.x.powf(p.roundness) + normalized.y.powf(p.roundness) < 1.,
                "island outside back: {seed} {point:?}"
            );
            for lens in &p.lens_centers[..p.lens_count] {
                let point = *lens + Vec2::new(theta.cos(), theta.sin()) * p.lens_radius;
                let normalized = (point - p.island_center).abs() / (p.island_span * 0.48);
                assert!(
                    normalized.x.powf(p.island_roundness) + normalized.y.powf(p.island_roundness)
                        < 1.,
                    "lens outside island: {seed} {point:?}"
                );
            }
        }
        assert!(a.parts.keys().all(|(_, label)| label.starts_with("phone")));
        let replay = build_object(&o);
        for (key, g) in &a.parts {
            assert_eq!(g.positions, replay.parts[key].positions);
            assert_eq!(g.indices, replay.parts[key].indices);
            if key.0 == Surface::PhoneScreen {
                for axis in 0..2 {
                    let min = g.uvs.iter().map(|v| v[axis]).fold(f32::INFINITY, f32::min);
                    let max = g
                        .uvs
                        .iter()
                        .map(|v| v[axis])
                        .fold(f32::NEG_INFINITY, f32::max);
                    assert!(min.abs() < 1e-6 && (max - 1.).abs() < 1e-6);
                }
            }
        }
        assert_eq!(
            a.parts.keys().any(|(s, _)| *s == Surface::PhoneScreen),
            p.face_up
        );
        layouts.insert((p.case, p.camera_layout, p.face_up));
        dimensions.insert(((p.span.x * 1000.) as u32, (p.body_height * 10000.) as u32));
        o.finish_seed = Some(1234);
        let q = parameters(&o);
        assert_eq!(p.span, q.span);
        assert_eq!(p.lens_centers, q.lens_centers);
        assert_eq!(p.face_up, q.face_up);
    }
    assert_eq!(layouts.len(), 24);
    assert!(dimensions.len() > 230);
}
