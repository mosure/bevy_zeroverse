use super::*;
use crate::scene::procedural_indoor::{
    geometry::Geometry,
    layout::ObjectKind,
    objects::{audit, chairs::stool},
};

#[test]
fn elevated_stools_ground_both_constructed_shoes_on_the_front_footrest() {
    let mut checked = [false; 3];
    for seed in 0..64 {
        let mut chair = audit::fixture(ObjectKind::Chair, seed);
        chair.variant = 6;
        chair.size = Vec3::new(0.61, 0.76, 0.59);
        let params = stool::parameters(&chair);
        if checked[params.base as usize] {
            continue;
        }
        let support = params.support(&chair);
        let mut h = super::super::sample_person(
            seed,
            1,
            Vec3::ZERO,
            0.,
            super::super::HumanPoseKind::SeatedListening,
            Some(0),
            false,
        );
        let pose = h.pose_program.as_mut().unwrap();
        pose.stool_support = Some(support);
        h.joints = pose.solve(h.stature, h.build, h.shoulder_width, true);
        super::super::update_bounds(&mut h);
        let mesh = build(&h);
        assert!((h.bounds_min.y - mesh.bounds().0.y).abs() < 0.003);
        let sole = &mesh.parts[&HumanSurface::Sole];
        let rail_mesh = crate::scene::procedural_indoor::objects::build_object(&chair);
        for i in [15, 19] {
            let ankle = h.joints[i];
            assert!((ankle.y - 0.105 - support.rail_top_m).abs() < 0.002);
            let xz = ankle.xz() - Vec2::Y * (0.065 * h.stature / 1.75);
            let rail = rail_mesh
                .parts
                .values()
                .flat_map(|g| vertical_hits(g, xz))
                .filter(|&y| y < support.rail_top_m + 0.02)
                .max_by(f32::total_cmp)
                .expect("rail below foot");
            let sole = vertical_hits(sole, xz)
                .min_by(f32::total_cmp)
                .expect("foot above rail");
            assert!(
                (rail - support.rail_top_m).abs() < 0.003,
                "{seed} {i} {rail} {support:?}"
            );
            assert!(
                (-0.003..0.016).contains(&(sole - rail)),
                "{seed} {i}: sole {sole}, rail {rail}"
            );
        }
        checked[params.base as usize] = true;
        if checked.iter().all(|v| *v) {
            break;
        }
    }
    assert_eq!(checked, [true; 3]);
}

// Intersect the exported triangles directly; this also exercises registration
// of the skinned shoe mesh with the local joint annotation.
fn vertical_hits(g: &Geometry, xz: Vec2) -> impl Iterator<Item = f32> + '_ {
    g.indices.as_chunks::<3>().0.iter().filter_map(move |tri| {
        let [a, b, c] = tri.map(|i| Vec3::from(g.positions[i as usize]));
        let ab = b.xz() - a.xz();
        let ac = c.xz() - a.xz();
        let q = xz - a.xz();
        let det = ab.perp_dot(ac);
        let u = q.perp_dot(ac) / det;
        let v = ab.perp_dot(q) / det;
        (u >= -1e-5 && v >= -1e-5 && u + v <= 1.00001)
            .then_some(a.y + u * (b.y - a.y) + v * (c.y - a.y))
    })
}
