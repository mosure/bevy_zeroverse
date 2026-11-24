use super::*;
use crate::scene::procedural_indoor::{
    materials::{self, variants},
    objects::audit,
};

#[test]
fn facing_pages_stay_supported_outward_and_read_in_the_same_direction() {
    let mut open = 0;
    for seed in 0..192 {
        let mut o = audit::fixture(ObjectKind::Notebook, seed);
        let p = parameters(&o);
        if !p.opened {
            continue;
        }
        open += 1;
        for size in [
            Vec3::new(0.14, 0.009, 0.23),
            Vec3::new(0.28, 0.032, 0.15),
            Vec3::new(0.2, 0.018, 0.2),
        ] {
            o.size = size;
            let s = Spread::new(&o, &p);
            let assembly = build_object(&o);
            audit::check(&o, &assembly).unwrap();
            let (lo, hi) = assembly.bounds();
            assert!(lo.y.abs() < 1e-6 && hi.y <= o.size.y);
            assert!(
                lo.x >= -size.x * 0.5
                    && hi.x <= size.x * 0.5
                    && lo.z >= -size.z * 0.5
                    && hi.z <= size.z * 0.5
            );
            assert!((0.60..0.92).contains(&(0.5 * s.size.x / s.size.z)));
            assert!(assembly
                .parts
                .keys()
                .all(|(surface, label)| *surface != Surface::Ink
                    && part_label(label) == o.kind.class_name()));
            let label = format!(
                "{}#finish{}#uv_local",
                o.kind.class_name(),
                paper::spread_slot(seed)
            );
            let top = &assembly.parts[&(Surface::PrintedPaper, label)];
            assert!(top
                .uvs
                .iter()
                .all(|uv| uv.iter().all(|v| (0.0..=1.).contains(v))));
            let seam: Vec<_> = top
                .uvs
                .iter()
                .enumerate()
                .filter(|(_, uv)| uv[0] == 0.5)
                .map(|(i, uv)| (uv[1], top.positions[i]))
                .collect();
            assert_eq!(seam.len(), 4);
            assert_eq!(seam[0], seam[2]);
            assert_eq!(seam[1], seam[3]);
            for indices in top.indices.chunks_exact(3) {
                let [a, b, c] = [indices[0], indices[1], indices[2]].map(|i| i as usize);
                let [v0, v1, v2] = [a, b, c].map(|i| Vec3::from_array(top.positions[i]));
                let face = (v1 - v0).cross(v2 - v0).normalize();
                assert!(
                    face.y > 0.
                        && [a, b, c]
                            .iter()
                            .all(|&i| face.dot(Vec3::from_array(top.normals[i])) > 0.88)
                );
                let [a, b, c] = [a, b, c].map(|i| Vec2::from_array(top.uvs[i]));
                assert!(
                    (b - a).perp_dot(c - a) < 0.,
                    "reversed or collapsed page chart"
                );
            }
            for side in [false, true] {
                let (a, _) = s.leaf(&p, side, 0., 0.);
                let (b, _) = s.leaf(&p, side, 1., 0.);
                let reader_right = s.rotation * Vec3::X;
                assert_eq!((b - a).dot(reader_right) > 0., side);
            }
        }
    }
    assert!(open > 50);
}

#[test]
fn old_and_new_forms_share_placement_identity_and_differentiate_writing() {
    let mut o = audit::fixture(ObjectKind::Notebook, 2_530_243_425_158_947_287);
    o.size = Vec3::new(0.13320178, 0.031643983, 0.15321597);
    assert!(parameters(&o).opened);
    let current = build_object(&o);
    o.variant = 0;
    let archived = build_object(&o);
    for a in [&current, &archived] {
        audit::check(&o, a).unwrap();
    }
    let n = |a: &Assembly| a.parts.values().map(|g| g.indices.len() / 3).sum::<usize>();
    assert!(current
        .parts
        .keys()
        .any(|(s, _)| *s == Surface::PrintedPaper));
    assert!(archived.parts.keys().any(|(s, _)| *s == Surface::Ink));
    assert!(n(&current) < n(&archived));
    let s = paper::spread_slot(o.seed);
    assert!(s >= variants::COUNT && materials::registered_finish(Surface::PrintedPaper, s));
}
