use super::super::{
    layout::{IndoorLayout, IndoorManifest},
    validation,
};
use super::*;

#[test]
fn upholstery_shelves_and_wall_hardware_fit_declared_envelopes() {
    let mut scene =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Lounge, 0.7, 0, 0.).unwrap();
    let template = scene.objects[0].clone();
    scene.objects.clear();
    for seed in 0..32 {
        let mut cases = vec![
            (ObjectKind::Sofa, Vec3::new(2.8, 0.92, 1.85), 0),
            (ObjectKind::Sofa, Vec3::new(1.55, 0.84, 0.90), 0),
            (ObjectKind::Bookcase, Vec3::new(0.80, 1.65, 0.32), 0),
            (
                ObjectKind::WallOutlet,
                Vec3::new(0.225, 0.115, 0.018),
                (seed % 4) as u32,
            ),
            (
                ObjectKind::LightSwitch,
                Vec3::new(0.15, 0.115, 0.018),
                (seed % 4) as u32,
            ),
        ];
        cases.extend(
            (0..chairs::FAMILIES).map(|v| (ObjectKind::Chair, Vec3::new(0.68, 1.40, 0.68), v)),
        );
        for (kind, size, variant) in cases {
            let mut o = template.clone();
            o.kind = kind;
            o.size = size;
            o.variant = variant;
            o.seed = seed;
            o.position = Vec3::ZERO;
            o.yaw = 0.;
            o.id = scene.objects.len();
            if kind == ObjectKind::Chair {
                let p = chairs::parameters(&o);
                assert!(
                    p.back_construction != 1 || variant == 3,
                    "spindles require a wood frame"
                );
                assert!(
                    variant != 9 || p.back_construction >= 2,
                    "executive back must support its upholstery"
                );
                if variant == 8 {
                    assert_eq!(seating::parameters(&o).modules, 1);
                }
            }
            let a = build_object(&o);
            let (lo, hi) = a.bounds();
            assert!(
                lo.cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(0.003))
                    .all()
                    && hi
                        .cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.003))
                        .all(),
                "seed={seed} {kind:?}/{variant}: {lo:?}..{hi:?} size={size:?}"
            );
            for (_, label) in a.parts.keys() {
                assert!(SemanticLabel::from_label(part_label(label)).is_some());
            }
            scene.objects.push(o);
        }
    }
    validation::validate_geometry(&scene).unwrap();
}

#[test]
fn activity_profiles_replay_continuous_mixtures_and_keep_functional_layouts() {
    let mut blends = std::collections::BTreeSet::new();
    for layout in IndoorLayout::PROFILES {
        for seed in 0..16 {
            let scene = IndoorManifest::generate_with_humans(seed, layout, 0.65, 2, 0.).unwrap();
            validation::validate_layout(&scene).unwrap();
            let replay: IndoorManifest =
                serde_json::from_str(&serde_json::to_string(&scene).unwrap()).unwrap();
            assert_eq!(scene, replay);
            for zone in &scene.program.unwrap().zones {
                let mix = zone.composition.as_ref().unwrap();
                mix.validate().unwrap();
                blends.insert(mix.weights.map(|v| (v * 10000.) as u32));
            }
        }
    }
    assert!(blends.len() > 200);
    let mut invalid =
        super::super::program::activity::ActivityMix::sample(0, 0, IndoorLayout::Mixed);
    invalid.weights[0] = f32::NAN;
    assert!(invalid.validate().is_err());
}

#[test]
fn marker_programs_have_deterministic_nonrepeating_content() {
    let images: std::collections::BTreeSet<_> = (0..64)
        .map(super::super::materials::boards::pixels)
        .collect();
    assert!(images.len() > 55);
    assert!(images.iter().all(|i| i.len() == 256 * 256 * 4));
    assert_eq!(
        super::super::materials::boards::pixels(312),
        super::super::materials::boards::pixels(312)
    );
}

#[test]
fn continuous_table_tops_remain_convex_and_support_real_mounts() {
    let mut scene =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Mixed, 0.5, 0, 0.).unwrap();
    let mut o = scene.objects[0].clone();
    scene.objects.clear();
    let mut shapes = std::collections::BTreeSet::new();
    for seed in 0..128 {
        for (kind, size) in [
            (ObjectKind::CoffeeTable, Vec3::new(0.80, 0.36, 0.60)),
            (ObjectKind::Table, Vec3::new(1.1, 0.74, 3.4)),
            (ObjectKind::Desk, Vec3::new(1.4, 0.74, 0.70)),
        ] {
            o.kind = kind;
            o.size = size;
            o.seed = seed;
            o.id = scene.objects.len();
            let outline = tables::outline(&o);
            for i in 0..outline.len() {
                let a = outline[i];
                let e = outline[(i + 1) % outline.len()] - a;
                assert!(
                    outline.iter().all(|p| e.perp_dot(*p - a) >= -1e-6),
                    "convex support seed={seed}/{kind:?}"
                );
            }
            assert!(tables::supports(
                &o,
                Vec2::splat(-0.06),
                Vec2::splat(0.06),
                0.
            ));
            assert_eq!(outline, tables::outline(&o));
            shapes.insert(
                outline
                    .iter()
                    .map(|v| v.to_array().map(f32::to_bits))
                    .collect::<Vec<_>>(),
            );
            let a = build_object(&o);
            let (lo, hi) = a.bounds();
            assert!(lo
                .cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(0.003))
                .all());
            assert!(hi
                .cmple(Vec3::new(size.x * 0.5, size.y + 0.004, size.z * 0.5) + Vec3::splat(0.003))
                .all());
            assert!(lo.y <= 0.002, "supported feet {kind:?} seed={seed}: {lo:?}");
            // Every mount/crossbar lies below the actual top, and no frame tube
            // folds its inner bend (validated through triangle/normal agreement).
            scene.objects.push(o.clone());
        }
    }
    assert_eq!(shapes.len(), 384);
    validation::validate_geometry(&scene).unwrap();
}
