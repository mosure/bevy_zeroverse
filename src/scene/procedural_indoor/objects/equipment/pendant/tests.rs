use super::*;
use crate::scene::procedural_indoor::{
    materials::emitters,
    objects::{audit, build_object},
};
use std::collections::BTreeSet;

#[test]
fn each_fitted_optic_owns_one_supported_emitting_surface_and_the_measured_area() {
    let mut combinations = BTreeSet::new();
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::CeilingPendant, seed);
        o.size = match seed % 3 {
            0 => Vec3::new(0.13, 0.88, 0.13),
            1 => Vec3::new(0.55, 0.22, 0.55),
            _ => o.size,
        };
        let p = parameters(&o);
        combinations.insert((p.suspension as u8, p.lens.optic.name()));
        let a = build_object(&o);
        audit::check(&o, &a).unwrap_or_else(|why| panic!("{seed}: {why}"));
        let (lo, hi) = a.bounds();
        assert!(lo.y >= 0. && (hi.y - o.size.y).abs() < 1e-6);
        let mut lights = a.parts.iter().filter(|((s, _), _)| *s == Surface::Light);
        let ((_, label), g) = lights.next().unwrap();
        assert_eq!(emitters::source_key(label), Some((o.id, 0)));
        assert!(lights.next().is_none());
        let actual: f32 = g
            .indices
            .as_chunks::<3>()
            .0
            .iter()
            .map(|ix| {
                let [v, w, x] = ix.map(|i| Vec3::from(g.positions[i as usize]));
                (w - v).cross(x - v).length() * 0.5
            })
            .sum();
        assert!((actual / emitters::area(&o) - 1.).abs() < 2e-5);
        let crown = p.lens.bottom_y_m + p.lens.rise_m;
        let seat_radius = if matches!(p.lens.optic, Optic::OpenBulb(_)) {
            p.lens.radius_m * 0.32
        } else {
            p.lens.radius_m
        };
        let seam =
            |pt: Vec3| (pt.y - crown).abs() < 1e-6 && (pt.xz().length() - seat_radius).abs() < 1e-6;
        let fixed: Vec<_> = a
            .parts
            .iter()
            .filter(|((s, _), _)| *s != Surface::Light)
            .flat_map(|(_, q)| &q.positions)
            .copied()
            .map(Vec3::from)
            .filter(|v| seam(*v))
            .collect();
        let mut joined = 0;
        for pt in g
            .positions
            .iter()
            .copied()
            .map(Vec3::from)
            .filter(|v| seam(*v))
        {
            assert!(
                fixed.iter().any(|v| v.distance(pt) < 2e-6),
                "{seed}: lost optical seat at {pt}"
            );
            joined += 1;
        }
        assert!(joined >= 32);
        o.position = Vec3::new(1.4, 2.2, -2.8);
        o.yaw = 0.72;
        let beam = super::super::fixture_beam(&o);
        assert!(p.lens.bottom_y_m + o.position.y >= beam.position.y + 0.0029);
        assert!(beam.direction.abs_diff_eq(Vec3::NEG_Y, 1e-6));
        assert!(beam.inner > 0. && beam.inner < beam.outer && beam.outer < FRAC_PI_2);
        o.finish_seed = Some(983);
        let recoated = parameters(&o);
        assert_eq!(p.lens.emitting_area_m2(), recoated.lens.emitting_area_m2());
        assert_eq!(p.canopy.radius_m, recoated.canopy.radius_m);
        assert_eq!(p.suspension, recoated.suspension);
    }
    assert_eq!(combinations.len(), 9);
}
