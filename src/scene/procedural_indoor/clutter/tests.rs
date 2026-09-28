use super::super::{
    layout::{IndoorLayout, IndoorManifest},
    objects::build_object,
    validation,
};
use super::*;
use std::collections::BTreeSet;

#[test]
fn tabletop_meshes_fit_support_envelopes_and_keep_semantics() {
    let mut scene =
        IndoorManifest::generate_with_humans(10, IndoorLayout::Conference, 0.6, 0, 0.).unwrap();
    let mut template = scene.objects[0].clone();
    template.position = Vec3::ZERO;
    template.yaw = 0.;
    let cases = [
        (ObjectKind::Mug, Vec3::new(0.12, 0.10, 0.09)),
        (ObjectKind::CoffeeCup, Vec3::new(0.09, 0.14, 0.09)),
        (ObjectKind::WaterBottle, Vec3::new(0.075, 0.24, 0.075)),
        (ObjectKind::SodaCan, Vec3::new(0.065, 0.12, 0.065)),
        (ObjectKind::Notepad, Vec3::new(0.16, 0.012, 0.22)),
        (ObjectKind::Pencil, Vec3::new(0.008, 0.009, 0.17)),
        (ObjectKind::Microphone, Vec3::new(0.13, 0.24, 0.16)),
        (ObjectKind::Phone, Vec3::new(0.075, 0.009, 0.15)),
    ];
    scene.objects.clear();
    for seed in 0..48 {
        for (kind, size) in cases {
            let mut o = template.clone();
            o.kind = kind;
            o.size = size;
            o.seed = seed;
            let assembly = build_object(&o);
            let (lo, hi) = assembly.bounds();
            assert!(
                lo.cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(0.0006))
                    .all()
                    && hi
                        .cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.0006))
                        .all(),
                "{kind:?} seed={seed}: {lo:?}..{hi:?}, envelope={size:?}"
            );
            for (_, label) in assembly.parts.keys() {
                assert_eq!(super::super::objects::part_label(label), kind.class_name());
            }
            o.id = scene.objects.len();
            scene.objects.push(o);
        }
    }
    validation::validate_geometry(&scene).unwrap();
}

#[test]
fn new_props_are_sampled_on_valid_supports_across_room_programs() {
    let mut observed = BTreeSet::new();
    for seed in 0..64 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.85, 0, 0.).unwrap();
        validation::validate_layout(&scene).unwrap();
        for o in &scene.objects {
            if matches!(
                o.kind,
                ObjectKind::CoffeeCup
                    | ObjectKind::SodaCan
                    | ObjectKind::Notepad
                    | ObjectKind::Pencil
                    | ObjectKind::Microphone
                    | ObjectKind::Phone
            ) {
                observed.insert(format!("{:?}", o.kind));
                let support = &scene.objects[o.support.expect("tabletop support")];
                assert!(scene.prop_clear(o, support, 0.010));
                assert!((o.position.y - support.position.y - support.size.y).abs() < 1e-5);
            }
        }
    }
    assert_eq!(observed.len(), 6);
}

#[test]
fn clock_hands_share_fractional_time_and_screens_do_not_repeat_one_dashboard() {
    use super::super::{materials::screens, objects::clocks::hand_angles};
    let a = hand_angles(3 * 3600 + 15 * 60 + 30);
    assert!((a[0].to_degrees() - 97.75).abs() < 1e-4);
    assert!((a[1].to_degrees() - 93.).abs() < 1e-4);
    assert!((a[2].to_degrees() - 180.).abs() < 1e-4);
    let mut layouts = BTreeSet::new();
    let mut times = BTreeSet::new();
    let mut images = BTreeSet::new();
    let mut mobile_images = BTreeSet::new();
    for seed in 0..64 {
        let p = screens::parameters(seed);
        layouts.insert(p.layout);
        times.insert(p.minutes);
        if p.layout != 7 {
            images.insert(screens::pixels(seed));
            let mobile = screens::mobile_pixels(seed);
            assert_ne!(mobile, screens::pixels(seed));
            mobile_images.insert(mobile);
        }
    }
    assert_eq!(layouts.len(), 8);
    assert!(times.len() > 55);
    assert!(images.len() > 50);
    assert!(mobile_images.len() > 50);
    assert_eq!(screens::pixels(10), screens::pixels(10));
}
