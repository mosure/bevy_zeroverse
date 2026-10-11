use super::*;
use crate::scene::procedural_indoor::architecture::overlap_tests::intersection_area;
use crate::scene::procedural_indoor::layout::IndoorManifest;
use std::collections::BTreeMap;

#[test]
fn lids_panels_and_wall_hardware_have_no_competing_flat_faces() {
    let template = IndoorManifest::generate_with_humans(
        0,
        super::super::layout::IndoorLayout::Mixed,
        0.5,
        0,
        0.,
    )
    .unwrap()
    .objects[0]
        .clone();
    for seed in 0..8 {
        for (kind, size) in [
            (ObjectKind::StorageBox, Vec3::new(0.40, 0.28, 0.30)),
            (ObjectKind::Printer, Vec3::new(0.40, 0.28, 0.30)),
            (ObjectKind::WallOutlet, Vec3::new(0.225, 0.115, 0.018)),
            (ObjectKind::LightSwitch, Vec3::new(0.15, 0.115, 0.018)),
            (ObjectKind::Bookcase, Vec3::new(0.80, 1.65, 0.32)),
            (ObjectKind::Cabinet, Vec3::new(0.9, 1.1, 0.42)),
            (ObjectKind::Whiteboard, Vec3::new(1.5, 0.9, 0.04)),
            (ObjectKind::Phone, Vec3::new(0.074, 0.009, 0.16)),
        ] {
            let mut o = template.clone();
            o.kind = kind;
            o.size = size;
            o.seed = seed;
            o.variant = (seed % 4) as u32;
            let mut planes = BTreeMap::<(usize, i32, i32), Vec<([Vec2; 3], String)>>::new();
            for ((surface, label), g) in build_object(&o).parts {
                for tri in g.indices.as_chunks::<3>().0 {
                    let p: [Vec3; 3] =
                        std::array::from_fn(|i| Vec3::from_array(g.positions[tri[i] as usize]));
                    let n = (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero();
                    let axis = (0..3)
                        .max_by(|&a, &b| n[a].abs().total_cmp(&n[b].abs()))
                        .unwrap();
                    if n[axis].abs() < 0.99999 {
                        continue;
                    }
                    let key = (
                        axis,
                        n[axis].signum() as i32,
                        (p[0][axis] * 10000.).round() as i32,
                    );
                    planes.entry(key).or_default().push((
                        p.map(|p| Vec2::new(p[(axis + 1) % 3], p[(axis + 2) % 3])),
                        format!("{surface:?}/{label}"),
                    ));
                }
            }
            for (plane, triangles) in planes {
                for (i, (a, la)) in triangles.iter().enumerate() {
                    for (b, lb) in &triangles[i + 1..] {
                        let area = intersection_area(*a, *b);
                        assert!(area < 0.00001, "{kind:?} seed {seed} plane {plane:?}: {la}/{lb} compete over {area} square metres: {a:?} / {b:?}; program={:?}", storage::parameters(&o));
                    }
                }
            }
        }
    }
}
