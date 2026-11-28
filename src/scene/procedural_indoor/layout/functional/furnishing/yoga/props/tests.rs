use super::*;
use crate::scene::procedural_indoor::{envelope::ray_triangle, objects::build_object, validation};

#[test]
fn practice_props_keep_mat_centres_folds_and_aisles_clear_and_replay() {
    let mut bare = 0;
    let mut paired = 0;
    let mut folded = 0;
    let mut seen = [0; 2];
    let mut ends = [0; 2];
    let mut tested_invalid = false;
    for seed in 71300..71348 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::YogaStudio, 0.65, 3, 0.35)
                .unwrap();
        validation::validate_layout(&scene).unwrap();
        let props: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| matches!(o.kind, ObjectKind::YogaBlock | ObjectKind::YogaBolster))
            .collect();
        bare += usize::from(props.is_empty());
        assert!(props.len() <= 14);
        for &obj in &props {
            let mat = &scene.objects[obj.support.unwrap()];
            assert_eq!(mat.kind, ObjectKind::ExerciseMat);
            assert!(!mat.kind.is_surface() && !mat.supports_item(ObjectKind::Laptop));
            assert!(scene.prop_clear(obj, mat, 0.01));
            let p = &scene.functional_programs()[0];
            assert!(p.members.contains(&obj.id) && !p.counts_toward_capacity(obj));
            let (lo, hi) = obj.bounds();
            assert!(p.reservations.iter().all(|r| !r.overlaps(lo, hi)));
            let tf = mat.transform().compute_affine().inverse() * obj.transform().compute_affine();
            seen[usize::from(obj.kind == ObjectKind::YogaBolster)] += 1;
            ends[usize::from(tf.translation.z > 0.)] += 1;
            paired += usize::from(
                props
                    .iter()
                    .any(|other| other.support == obj.support && other.id < obj.id),
            );
            folded += usize::from(mats::PracticeMat::for_object(mat).panels > 1);
            let mesh = build_object(mat);
            for x in [-0.5, 0.5] {
                for z in [-0.5, 0.5] {
                    let pos = tf.transform_point3(Vec3::new(x * obj.size.x, 0., z * obj.size.z));
                    let top = mesh
                        .parts
                        .values()
                        .flat_map(|g| {
                            g.indices.as_chunks::<3>().0.iter().filter_map(|tri| {
                                ray_triangle(
                                    pos + Vec3::Y * 0.05,
                                    Vec3::NEG_Y,
                                    tri.map(|i| Vec3::from(g.positions[i as usize])),
                                )
                            })
                        })
                        .min_by(f32::total_cmp)
                        .unwrap();
                    assert!((top - 0.05).abs() < 1e-5 && pos.z.abs() >= mat.size.z / 6.);
                }
            }
            if !tested_invalid {
                let replay = IndoorManifest::generate_with_humans(
                    seed,
                    IndoorLayout::YogaStudio,
                    0.65,
                    3,
                    0.35,
                )
                .unwrap();
                assert_eq!(
                    serde_json::to_value(&scene.objects).unwrap(),
                    serde_json::to_value(&replay.objects).unwrap()
                );
                let mut wrong = scene.clone();
                wrong.objects[obj.id].position = mat.position + Vec3::Y * mat.size.y;
                assert!(validation::validate_layout(&wrong).is_err());
                wrong.objects[obj.id].support = None;
                assert!(validation::validate_layout(&wrong).is_err());
                tested_invalid = true;
            }
        }
    }
    assert!(
        tested_invalid
            && (2..43).contains(&bare)
            && paired > 2
            && folded > 2
            && seen.iter().all(|&n| n > 2)
            && ends.iter().all(|&n| n > 2),
        "{bare} bare, {paired} pairs, {folded} folding, {seen:?}, {ends:?}"
    );
}
