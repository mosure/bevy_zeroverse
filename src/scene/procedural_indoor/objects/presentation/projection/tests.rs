use super::super::tests::ray_hit;
use super::*;
use std::collections::BTreeSet;

#[test]
fn manufactured_stock_fits_the_reservation_and_the_first_surface_matches_the_visible_aperture() {
    let mut formats = BTreeSet::new();
    let mut states = BTreeSet::new();
    let mut coatings = BTreeSet::new();
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::ProjectionScreen, seed);
        o.finish_seed = Some(seed);
        for (width, depth) in [(1.18, 0.057), (2.7, 0.115)] {
            o.size = dimensions(seed, width, depth);
            let p = parameters(&o);
            let i = p.installation.as_ref().unwrap();
            let a = build_object(&o);
            audit::check(&o, &a).unwrap();
            let (lo, hi) = a.bounds();
            assert!((lo.z + depth * 0.5).abs() < 1e-5 && lo.y >= -1e-5 && hi.y <= o.size.y + 1e-5);
            formats.insert(i.format as usize);
            states.insert(i.deployment as usize);
            coatings.insert(i.coating as usize);
            let key = (
                Surface::Paper,
                super::super::super::super::materials::projection_cloth::label(i.coating),
            );
            if i.deployment == Deployment::Stowed {
                assert!(!a.parts.contains_key(&key));
                assert_eq!(p.surface_lo, p.surface_hi);
                assert!(p.projection_target_local().is_none());
                continue;
            }
            assert!((p.image_width_m() / i.full_image_height_m - i.format.aspect()).abs() < 1e-5);
            let visible = p.surface_hi.y - p.surface_lo.y;
            if i.deployment == Deployment::Presented {
                assert_eq!(p.projection_target_local(), Some(p.target_local()));
                assert!((visible - i.full_image_height_m).abs() < 2e-6);
            } else {
                assert!(
                    p.projection_target_local().is_none() && visible < i.full_image_height_m * 0.87
                );
            }
            let cloth = &a.parts[&key];
            assert_eq!(cloth.indices.len(), 6);
            for (pos, uv) in cloth.positions.iter().zip(&cloth.uvs) {
                assert_eq!(*uv, [pos[0], -pos[1]]);
            }
            for u in [0.01, 0.37, 0.99] {
                for v in [0.01, 0.61, 0.99] {
                    let point = Vec3::new(
                        p.surface_lo.x + p.image_width_m() * u,
                        p.surface_lo.y + visible * v,
                        p.sheet_front_m,
                    );
                    let (surface, t) = ray_hit(&a, point + Vec3::Z, Vec3::NEG_Z).unwrap();
                    assert_eq!(surface, Surface::Paper, "seed {seed}");
                    assert!((t - 1.).abs() < 1e-5);
                }
            }
            if !p.framed {
                let mouth = Vec3::new(
                    p.sheet_center_x_m,
                    p.sheet_top_m - 0.002,
                    p.sheet_front_m - 0.0018,
                );
                let (surface, t) = ray_hit(&a, mouth, Vec3::Y).unwrap();
                assert_eq!(surface, Surface::Ink);
                assert!(t > p.case_height_m * 0.32, "cassette has a solid underside");
            }
        }
    }
    assert_eq!((formats.len(), states.len(), coatings.len()), (5, 3, 3));
}
