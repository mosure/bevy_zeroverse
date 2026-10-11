use super::super::{program, program::replay_helpers::same_bits, Surface};
use super::*;

#[test]
fn dye_program_is_recorded_seedable_and_legacy_recipes_remain_plain() {
    let mut patterned = 0;
    let mut checks = 0;
    let mut programmes = std::collections::BTreeSet::new();
    for seed in 0..512 {
        let t = TextileRecipe::sample(seed);
        assert_eq!(t, TextileRecipe::sample(seed));
        t.validate().unwrap();
        let value = serde_json::to_value(&t).unwrap();
        assert_eq!(t, serde_json::from_value::<TextileRecipe>(value).unwrap());
        if let Some(p) = &t.dye_program {
            patterned += 1;
            checks += usize::from(p.strength[1] > 0.);
            programmes.insert(serde_json::to_string(p).unwrap());
        }
    }
    assert!((220..370).contains(&patterned));
    assert!(checks > 120 && programmes.len() == patterned);
    let mut value = serde_json::to_value(TextileRecipe::sample(202)).unwrap();
    value.as_object_mut().unwrap().remove("dye_program");
    let legacy: TextileRecipe = serde_json::from_value(value).unwrap();
    assert!(legacy.dye_program.is_none());
    assert_eq!(legacy.yarn_dye(0, 4), [1.; 3]);
}

#[test]
fn dyed_yarn_changes_pigment_not_surface_shape_and_replays_bounded_preparation() {
    let mut r = program::sample(202)[Surface::Fabric as usize].clone();
    let t = r.textile.as_mut().unwrap();
    t.dye_program = Some(YarnDyeProgram {
        runs: [1, 2],
        duty: [0.55, 0.50],
        phase: [0., 0.2],
        strength: [0.95, 0.90],
        secondary_tint: [0.38, 0.48, 0.63],
    });
    let mut plain = r.clone();
    plain.textile.as_mut().unwrap().dye_program = None;
    let mut darkened = 0;
    for budget in [0, 32, 1024, prepared::TABLE_BUDGET_BYTES] {
        let p = PreparedTextile::with_budget(&r, budget);
        assert!(p.bytes() <= budget);
        for y in 0..32 {
            for x in 0..32 {
                let uv = [x as f32 / 32., y as f32 / 32.];
                let dyed = r.textile.as_ref().unwrap().evaluate(&r, uv);
                let uncolored = plain.textile.as_ref().unwrap().evaluate(&plain, uv);
                same_bits(&dyed, &p.evaluate(&r, uv));
                assert_eq!(dyed.height, uncolored.height);
                assert_eq!(dyed.roughness, uncolored.roughness);
                assert_eq!(dyed.occlusion, uncolored.occlusion);
                assert!(dyed.color.into_iter().all(|c| (0.0..=1.).contains(&c)));
                darkened += usize::from(uncolored.color[0] - dyed.color[0] > 0.12);
            }
        }
    }
    assert!(
        darkened > 500,
        "pattern never materially changed dyed yarns: {darkened}"
    );
}

#[test]
fn malformed_yarn_dye_is_rejected_before_division_or_indexing() {
    let valid = YarnDyeProgram::sample(202);
    for malformed in 0..5 {
        let mut p = valid.clone();
        match malformed {
            0 => p.runs[0] = 0,
            1 => p.runs[1] = u32::MAX,
            2 => p.duty[0] = f32::NAN,
            3 => p.phase[1] = f32::INFINITY,
            _ => p.secondary_tint[0] = -0.1,
        }
        assert!(p.validate().is_err());
        let mut textile = TextileRecipe::sample(4);
        textile.dye_program = Some(p);
        assert!(textile.validate().is_err());
    }
}
