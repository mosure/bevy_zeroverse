use super::*;
use crate::scene::procedural_indoor::{
    objects::{audit, build_object},
    photometry,
};
#[test]
fn exit_styles_mounts_and_sizes_have_valid_semantic_geometry() {
    let mut combinations = std::collections::BTreeSet::new();
    for seed in 0..64 {
        for variant in 0..4 {
            for size in [Vec3::new(0.28, 0.14, 0.025), Vec3::new(0.53, 0.90, 0.065)] {
                let mut o = audit::fixture(ObjectKind::ExitSign, seed);
                o.variant = variant;
                o.size = size;
                let p = parameters(&o);
                let a = build_object(&o);
                audit::check(&o, &a).unwrap_or_else(|e| panic!("{seed}/{variant}/{size:?}: {e}"));
                combinations.insert((p.legend, p.mount));
                let (lo, hi) = a.bounds();
                if variant == 0 {
                    assert!((lo.z + size.z * 0.5).abs() < 1e-5);
                } else {
                    assert!((hi.y - size.y).abs() < 1e-5);
                }
                for (_, label) in a.parts.keys() {
                    assert_eq!(
                        crate::scene::procedural_indoor::objects::part_label(label),
                        "exit_sign"
                    );
                }
                // Both faces use the same two/one color channels; no duplicate
                // emitter allocation per face and no hidden extra lights.
                assert_eq!(
                    photometry::emission_channels(&o),
                    if p.legend == ExitLegend::EuropeanPictogram {
                        2
                    } else {
                        1
                    }
                );
                let count = a.parts.keys().filter(|(s, _)| *s == Surface::Light).count();
                assert_eq!(count, photometry::emission_channels(&o));
            }
        }
    }
    assert_eq!(combinations.len(), 12);
}
#[test]
fn ceiling_back_legends_are_rotated_not_mirrored_or_flipped() {
    for seed in 0..32 {
        let mut o = audit::fixture(ObjectKind::ExitSign, seed);
        o.variant = 2;
        let a = build_object(&o);
        let p = parameters(&o);
        for ((s, _), g) in a.parts.iter().filter(|((s, _), _)| *s == Surface::Light) {
            assert_eq!(*s, Surface::Light);
            let front: Vec<_> = g
                .positions
                .iter()
                .zip(&g.normals)
                .filter(|(_, n)| n[2] > 0.99)
                .map(|(v, _)| Vec3::from_array(*v))
                .collect();
            let back: Vec<_> = g
                .positions
                .iter()
                .zip(&g.normals)
                .filter(|(_, n)| n[2] < -0.99)
                .map(|(v, _)| Vec3::from_array(*v))
                .collect();
            assert_eq!(front.len(), back.len());
            for v in front {
                let reflected = Vec3::new(-v.x, v.y, 2. * p.center.z - v.z);
                assert!(back.iter().any(|b| b.distance(reflected) < 1e-5));
            }
        }
    }
}
#[test]
fn legend_palette_is_independent_of_mount() {
    for seed in 0..32 {
        let mut o = audit::fixture(ObjectKind::ExitSign, seed);
        let legend = parameters(&o).legend;
        let color = appearance(&o, 0);
        for variant in 0..4 {
            o.variant = variant;
            assert_eq!(parameters(&o).legend, legend);
            assert_eq!(appearance(&o, 0), color);
        }
    }
}
