use super::*;
use crate::scene::procedural_indoor::objects::{self, audit, kitchen};

#[test]
fn physical_size_ranges_replay_and_build_within_annotation_envelopes() {
    use ObjectKind::*;
    let mut report = serde_json::Map::new();
    for kind in [
        Mug,
        CoffeeCup,
        WaterBottle,
        SodaCan,
        PenHolder,
        Pencil,
        Notepad,
        Notebook,
        Phone,
        Keyboard,
        Mouse,
        Speaker,
        Headphones,
        Microphone,
        Stapler,
        TapeDispenser,
        StorageBox,
        Books,
        Bowl,
        Dishware,
        SoapDispenser,
        Sponge,
        DryingRack,
        KnifeBlock,
        UtensilHolder,
        PaperTowel,
        KitchenTowel,
        CuttingBoard,
        FruitBowl,
        Bread,
    ] {
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::ZERO;
        for seed in 0..128 {
            let mut o = audit::fixture(kind, seed);
            let sample = || {
                if kind.is_kitchen() {
                    kitchen::dimensions(kind, seed)
                } else {
                    loose(kind, &mut stream(seed, 123)).unwrap()
                }
            };
            o.size = sample();
            assert_eq!(o.size, sample());
            assert!(o.size.is_finite() && o.size.cmpgt(Vec3::splat(0.005)).all());
            if matches!(kind, CoffeeCup | WaterBottle | SodaCan | Bowl) {
                assert_eq!(o.size.x, o.size.z, "round cross-section must stay round");
            }
            let a = objects::build_object(&o);
            audit::check(&o, &a)
                .unwrap_or_else(|e| panic!("{kind:?} seed={seed} size={:?}: {e}", o.size));
            min = min.min(o.size);
            max = max.max(o.size);
        }
        assert!(
            (max / min).cmpgt(Vec3::splat(1.25)).all(),
            "{kind:?}: {min:?}..{max:?}"
        );
        report.insert(
            format!("{kind:?}"),
            serde_json::json!({"min_m":min.to_array(),"max_m":max.to_array()}),
        );
    }
    println!(
        "loose item size audit: {}",
        serde_json::Value::Object(report)
    );
}
