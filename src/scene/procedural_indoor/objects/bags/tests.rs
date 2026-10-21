use super::*;
use std::collections::BTreeSet;

#[test]
fn luggage_programs_replay_stay_inside_bounds_and_preserve_semantics() {
    let mut styles = BTreeSet::new();
    let mut finishes = BTreeSet::new();
    let mut largest = 0;
    for seed in 0..256 {
        let mut o = audit::fixture(ObjectKind::Bag, seed);
        o.size = crate::scene::procedural_indoor::items::sizing::loose(
            ObjectKind::Bag,
            &mut stream(seed, 918),
        )
        .unwrap();
        let p = parameters(&o);
        styles.insert(format!("{:?}", p.style));
        finishes.insert(format!("{:?}", p.finish));
        assert_eq!(
            serde_json::to_value(&p).unwrap(),
            serde_json::to_value(parameters(&o)).unwrap()
        );
        let a = build_object(&o);
        let stats =
            audit::check(&o, &a).unwrap_or_else(|e| panic!("{e}; {p:?}; size={:?}", o.size));
        assert!(a.parts.keys().all(|(_, label)| part_label(label) == "bag"));
        assert!(stats.batches <= 4);
        assert!(stats.min_y_m.abs() < 1e-6);
        largest = largest.max(stats.triangles);
    }
    assert_eq!(styles.len(), 5);
    assert_eq!(finishes.len(), 3);
    assert!(largest < 6500, "unbounded bag detail {largest}");
    println!(
        "bag qualification: 256 dimensioned meshes, {largest} max triangles, styles={styles:?}"
    );
}

#[test]
fn luggage_silhouettes_change_at_fixed_size_without_material_shortcuts() {
    let mut descriptors = Vec::new();
    for seed in 0..20 {
        let o = audit::fixture(ObjectKind::Bag, seed);
        let a = build_object(&o);
        descriptors.push(audit::descriptor_at_resolution(&o, &a, 12));
    }
    let mut distances = Vec::new();
    for (i, a) in descriptors.iter().enumerate() {
        for b in &descriptors[i + 1..] {
            let distance =
                a.iter().zip(b).map(|(a, b)| (a - b).abs()).sum::<f32>() / a.len() as f32;
            assert!(
                distance > 0.003,
                "duplicate fixed-envelope bag silhouette: {distance}"
            );
            distances.push(distance);
        }
    }
    distances.sort_by(f32::total_cmp);
    println!(
        "bag shape L1 distances: min={} median={}",
        distances[0],
        distances[distances.len() / 2]
    );
}
