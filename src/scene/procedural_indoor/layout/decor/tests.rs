use super::*;
use crate::scene::procedural_indoor::{objects, validation};
#[test]
fn electrical_installations_vary_heights_and_sizes_without_mixing_socket_standards() {
    use std::collections::{BTreeMap, BTreeSet};
    let mut rows = Vec::new();
    let mut standards = BTreeSet::new();
    let mut depths = BTreeSet::new();
    let mut heights = BTreeSet::new();
    let mut raised = 0;
    for seed in 0..96 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.75, 0, 0.2).unwrap();
        validation::validate_layout(&scene).unwrap_or_else(|e| panic!("{seed}: {e}"));
        let p = electrical_parameters(&scene);
        assert_eq!(p, electrical_parameters(&scene));
        let mut counts = BTreeMap::new();
        for o in scene
            .objects
            .iter()
            .filter(|o| matches!(o.kind, ObjectKind::WallOutlet | ObjectKind::LightSwitch))
        {
            let h = o.position.y - scene.floor_height(o.position.xz());
            assert!(
                h > 0.12 && h < 1.75,
                "{seed}: invalid switch/outlet height {h}"
            );
            assert!(scene.wall_attachment_clear(o));
            let a = objects::build_object(o);
            objects::audit::check(o, &a).unwrap();
            if o.kind == ObjectKind::WallOutlet {
                assert_eq!(objects::wall::parameters(o).style as u32 % 3, p.standard);
                heights.insert((h * 100.).round() as i32);
                raised += usize::from(h > 0.7);
                if h > 0.7 {
                    assert!(
                        scene.objects.iter().any(|support| {
                            if !matches!(
                                support.kind,
                                ObjectKind::Desk
                                    | ObjectKind::KitchenCounter
                                    | ObjectKind::ServiceCounter
                                    | ObjectKind::Cabinet
                            ) {
                                return false;
                            }
                            let above = o.position.y - support.position.y - support.size.y;
                            let q = support
                                .transform()
                                .compute_affine()
                                .inverse()
                                .transform_point3(o.position)
                                .xz();
                            let half = support.size.xz() * 0.5;
                            (0.09..0.29).contains(&above) && q.distance(q.clamp(-half, half)) < 0.47
                        }),
                        "{seed}: raised outlet without a nearby worktop"
                    );
                }
            }
            standards.insert(p.standard);
            depths.insert((o.size.z * 1000.).round() as u32);
            *counts.entry(format!("{:?}", o.kind)).or_insert(0usize) += 1;
        }
        rows.push(serde_json::json!({"seed":seed,"installation":p,"accepted":counts}));
    }
    assert_eq!(standards.len(), 3);
    assert!(depths.len() >= 8 && heights.len() >= 30);
    assert!(raised > 5, "raised outlets never admitted: {raised}");
    println!(
        "electrical population: {}",
        serde_json::json!({"rooms":96,"standards":standards,"plate_depth_mm_bins":depths,"outlet_height_cm_bins":heights,"raised_outlets":raised,"samples":rows})
    );
}
