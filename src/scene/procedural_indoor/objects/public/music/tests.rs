use super::*;
use crate::scene::procedural_indoor::{
    materials::registered_finish,
    objects::{audit, build_object},
};
use std::collections::BTreeSet;

#[test]
fn playable_keyboards_have_correct_pitch_groups_and_bounded_meshes() {
    let mut compass = BTreeSet::new();
    let mut supports = BTreeSet::new();
    for seed in 0..96 {
        let o = audit::fixture(ObjectKind::StagePiano, seed);
        let k = Keyboard::sample(&o);
        let mesh = build_object(&o);
        let check = audit::check(&o, &mesh).unwrap();
        assert!(check.batches <= 8 && check.min_y_m.abs() < 0.001);
        let count = (k.first_midi..k.first_midi + k.keys)
            .filter(|n| matches!(n % 12, 1 | 3 | 6 | 8 | 10))
            .count();
        assert_eq!(count + k.white_keys, k.keys);
        assert_eq!((k.first_midi + k.keys - 1) % 12, 0);
        let naturals = &mesh.parts[&(Surface::Plastic, "musical_instrument#finish128".to_owned())];
        assert_eq!(naturals.positions.len(), 20 * k.white_keys);
        let raised = &mesh.parts[&(Surface::Plastic, "musical_instrument#finish126".to_owned())];
        let white_top = naturals.positions[0][1];
        // Each accidental has a distinct raised center above the white-key boundary.
        let centers: BTreeSet<_> = raised
            .positions
            .iter()
            .map(|p| {
                ((p[0] + 0.5 * k.key_pitch_m * k.white_keys as f32) / k.key_pitch_m).round()
                    as usize
            })
            .collect();
        let mut boundary = 0;
        for pitch in k.first_midi..k.first_midi + k.keys {
            if matches!(pitch % 12, 1 | 3 | 6 | 8 | 10) {
                assert!(centers.contains(&boundary));
            } else {
                boundary += 1;
            }
        }
        assert_eq!(centers.len(), count);
        assert!(raised.positions.iter().any(|p| p[1] > white_top + 0.012));
        assert!(
            registered_finish(Surface::Plastic, 126) && registered_finish(Surface::Plastic, 128)
        );
        compass.insert((k.keys, k.white_keys, count));
        supports.insert(k.x_stand);
    }
    assert_eq!(
        compass,
        BTreeSet::from([(61, 36, 25), (73, 43, 30), (88, 52, 36)])
    );
    assert_eq!(supports.len(), 2);
}

#[test]
fn music_desks_support_readable_mipmapped_parts_and_grounded_feet() {
    let mut variants = BTreeSet::new();
    let mut slots = BTreeSet::new();
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::MusicStand, seed);
        let desk = Stand::sample(&o);
        let mesh = build_object(&o);
        let check = audit::check(&o, &mesh).unwrap();
        assert!(check.batches <= 6 && check.min_y_m.abs() < 0.001);
        variants.insert((desk.open_frame, desk.folding, desk.score_slot.is_some()));
        if let Some(slot) = desk.score_slot {
            slots.insert(slot);
            assert!(registered_finish(Surface::PrintedPaper, slot));
            let paper = &mesh.parts[&(
                Surface::PrintedPaper,
                format!("paper#finish{slot}#uv_local"),
            )];
            let sheets = if desk.spread { 2 } else { 1 };
            assert_eq!(
                paper.positions.len(),
                4 * sheets * if desk.open_frame { 2 } else { 1 }
            );
            let normal = Vec3::from_array(paper.normals[0]);
            assert!(normal.z > 0.84 && normal.y > 0.14);
            assert!(paper.positions[0][1] > desk.lip_height_m && paper.positions[2][1] <= o.size.y);
        }
    }
    assert_eq!(slots.len(), score::END - score::FIRST);
    assert_eq!(variants.len(), 6);
}
