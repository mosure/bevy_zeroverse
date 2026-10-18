use super::*;
use crate::scene::procedural_indoor::layout::IndoorLayout;

#[test]
fn ceiling_program_replays_diversifies_and_contains_every_mesh() {
    let mut systems = std::collections::BTreeSet::new();
    let mut exposed = 0;
    let mut modules = 0;
    for seed in 0..96 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let plan = scene.ceiling_installation.as_ref().unwrap();
        assert_eq!(*plan, CeilingInstallation::sample(&scene));
        plan.validate(&scene)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        exposed += usize::from(plan.exposed_slab);
        for m in &plan.modules {
            modules += 1;
            systems.insert(format!("{:?}", m.system));
            if !plan.exposed_slab && m.system == CeilingSystem::Ventilation {
                let roof = sampling::support_heights(&scene, m.center.xz(), m.size.xz())
                    .into_iter()
                    .fold(f32::INFINITY, f32::min);
                assert!(roof - m.body_bounds().1.y <= 0.02);
            }
            let mut a = Assembly::default();
            construction::build(&mut a, &scene, m);
            let (lo, hi) = m.bounds();
            for ((_, label), g) in &a.parts {
                assert_eq!(label.split('#').next(), Some("ceiling"));
                for p in &g.positions {
                    let p = Vec3::from_array(*p);
                    assert!(
                        p.is_finite()
                            && p.cmpge(lo - Vec3::splat(0.003)).all()
                            && p.cmple(hi + Vec3::splat(0.003)).all(),
                        "seed {seed} {:?}: {p} outside {lo}..{hi}",
                        m.system
                    );
                }
            }
        }
    }
    assert_eq!(systems.len(), 13, "missing ceiling systems: {systems:?}");
    assert!(exposed > 12 && exposed < 80);
    assert!(modules > 160, "sparse ceiling sampling: {modules}");
}

#[test]
fn ceiling_validation_rejects_overlaps_and_nonfinite_supports() {
    let scene = IndoorManifest::generate_with_humans(2, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut plan = scene.ceiling_installation.clone().unwrap();
    assert!(!plan.modules.is_empty());
    plan.modules.push(plan.modules[0].clone());
    assert!(plan.validate(&scene).is_err());
    plan.modules.pop();
    plan.modules[0].support_top_m = f32::NAN;
    assert!(plan.validate(&scene).is_err());
}

#[test]
fn supported_props_cannot_enter_ceiling_installations() {
    use crate::scene::procedural_indoor::layout::ObjectKind;
    let mut scene =
        IndoorManifest::generate_with_humans(2, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let m = scene.ceiling_installation.as_ref().unwrap().modules[0].clone();
    let mut support = scene.objects[0].clone();
    support.id = 0;
    support.kind = ObjectKind::Cabinet;
    support.position = m.center.with_y(scene.floor_height(m.center.xz()));
    support.size = Vec3::new(
        0.5,
        m.center.y - m.size.y * 0.5 - support.position.y - 0.05,
        0.5,
    );
    support.yaw = 0.;
    support.neighbor = false;
    scene.objects = vec![support.clone()];
    let mut prop = support.clone();
    prop.id = 1;
    prop.kind = ObjectKind::Plant;
    prop.support = Some(0);
    prop.solid = false;
    prop.position = support.position + Vec3::Y * support.size.y;
    prop.size = Vec3::splat(0.12);
    assert!(!scene.prop_clear(&prop, &support, 0.012));
    prop.size.y = 0.01;
    assert!(scene.prop_clear(&prop, &support, 0.012));
}

#[test]
fn raised_floor_inside_a_module_footprint_counts_toward_headroom() {
    let mut scene =
        IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let module = scene.ceiling_installation.as_ref().unwrap().modules[0].clone();
    let lo = module.body_bounds().0;
    let patch = super::super::super::envelope::FloorPatch {
        min: module.center.xz() - Vec2::splat(0.03),
        max: module.center.xz() + Vec2::splat(0.03),
        height: lo.y - 2.30,
    };
    scene.envelope.as_mut().unwrap().floor_patches.push(patch);
    assert!((module.headroom(&scene) - 2.30).abs() < 0.001);
    assert!(!sampling::admissible(
        &scene,
        &module,
        &sampling::Reservations::new(&scene)
    ));
}

#[test]
fn ceiling_shapes_have_finite_wound_faces_and_bounded_complexity() {
    let mut scene =
        IndoorManifest::generate_with_humans(2, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let envelope = scene.envelope.as_mut().unwrap();
    envelope.ceiling_drop = Vec2::ZERO;
    envelope.soffits.clear();
    let systems = [
        CeilingSystem::AcousticCloud,
        CeilingSystem::Baffles,
        CeilingSystem::TimberSlats,
        CeilingSystem::RoundDuct,
        CeilingSystem::RectangularDuct,
        CeilingSystem::SprinklerMain,
        CeilingSystem::Truss,
        CeilingSystem::SuspendedTiles,
        CeilingSystem::OpenCell,
        CeilingSystem::Conduit,
        CeilingSystem::CableTray,
        CeilingSystem::Coffers,
        CeilingSystem::Ventilation,
    ];
    let mut winding_failures = Vec::new();
    for seed in 0..48 {
        let mut rng = stream(seed, 32);
        for system in systems {
            let width = if matches!(
                system,
                CeilingSystem::SprinklerMain | CeilingSystem::Conduit | CeilingSystem::CableTray
            ) {
                0.28
            } else {
                1.4
            };
            let mut size = Vec3::new(
                if seed % 2 == 0 { 0.65 } else { 4.2 },
                rng.random_range(0.10..0.88),
                width,
            );
            if seed % 3 == 0 {
                size = Vec3::new(size.z, size.y, size.x);
            }
            let m = CeilingModule {
                system,
                center: Vec3::new(0., scene.room_size.y - 0.5 - size.y * 0.5, 0.),
                size,
                support_top_m: scene.room_size.y - 0.001,
                pitch_m: rng.random_range(0.16..0.65),
                thickness_m: rng.random_range(0.012..0.055),
                seed,
                detail: Some(CeilingDetail::sample(system, &mut rng, 0)),
            };
            assert!(m.detail.as_ref().unwrap().valid(system));
            let decoded: CeilingModule =
                serde_json::from_value(serde_json::to_value(&m).unwrap()).unwrap();
            assert_eq!(m, decoded);
            let a = m.geometry(&scene);
            let triangles: usize = a.parts.values().map(|g| g.indices.len() / 3).sum();
            assert!(
                triangles > 50 && triangles < 30_000,
                "{system:?}: {triangles} triangles"
            );
            let (lo, hi) = m.bounds();
            for g in a.parts.values() {
                assert_eq!(g.positions.len(), g.normals.len());
                assert_eq!(g.positions.len(), g.uvs.len());
                for (&p, (&n, &uv)) in g.positions.iter().zip(g.normals.iter().zip(&g.uvs)) {
                    let p = Vec3::from_array(p);
                    assert!(
                        p.is_finite()
                            && p.cmpge(lo - Vec3::splat(0.003)).all()
                            && p.cmple(hi + Vec3::splat(0.003)).all(),
                        "seed {seed} {system:?}: {p} outside {lo}..{hi}"
                    );
                    assert!(
                        (Vec3::from_array(n).length() - 1.).abs() < 0.001,
                        "{system:?}: invalid normal"
                    );
                    assert!(Vec2::from_array(uv).is_finite());
                }
                for tri in g.indices.as_chunks::<3>().0 {
                    let p =
                        [tri[0], tri[1], tri[2]].map(|i| Vec3::from_array(g.positions[i as usize]));
                    let face = (p[1] - p[0]).cross(p[2] - p[0]);
                    let normal: Vec3 = tri
                        .iter()
                        .map(|&i| Vec3::from_array(g.normals[i as usize]))
                        .sum();
                    assert!(
                        face.length_squared() > 1e-16,
                        "seed {seed} {system:?}: degenerate triangle"
                    );
                    if face.dot(normal) <= -1e-8 && winding_failures.len() < 20 {
                        winding_failures.push(format!(
                            "seed {seed} {system:?}: reversed normals {p:?} dot={} module={m:?}",
                            face.dot(normal)
                        ));
                    }
                }
            }
        }
    }
    assert!(
        winding_failures.is_empty(),
        "{}",
        winding_failures.join("\n")
    );
}

#[test]
fn ceiling_detail_validation_and_legacy_manifest_replay() {
    let scene = IndoorManifest::generate_with_humans(2, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut m = scene.ceiling_installation.as_ref().unwrap().modules[0].clone();
    m.system = CeilingSystem::RoundDuct;
    m.detail = None;
    let mut json = serde_json::to_value(&m).unwrap();
    json.as_object_mut().unwrap().remove("detail");
    let old: CeilingModule = serde_json::from_value(json).unwrap();
    let mut legacy = Assembly::default();
    super::legacy::build(&mut legacy, &scene, &m);
    let decoded = old.geometry(&scene);
    for (key, g) in legacy.parts {
        assert_eq!(g.positions, decoded.parts[&key].positions);
        assert_eq!(g.indices, decoded.parts[&key].indices);
    }
    let mut rng = stream(11, 32);
    let mut detail = CeilingDetail::sample(CeilingSystem::Conduit, &mut rng, 0);
    assert!(detail.valid(CeilingSystem::Conduit));
    assert!(!detail.valid(CeilingSystem::RoundDuct));
    if let CeilingShape::Conduit { ref mut rise, .. } = detail.shape {
        *rise = f32::NAN;
    }
    assert!(!detail.valid(CeilingSystem::Conduit));
}

#[test]
fn interior_stair_treads_and_window_coverings_keep_clearance() {
    use crate::scene::procedural_indoor::envelope::{polygon, FloorPatch, Mezzanine};
    let mut scene =
        IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let m = scene.ceiling_installation.as_ref().unwrap().modules[0].clone();
    let (lo, hi) = m.body_bounds();
    let middle = m.center.xz();
    // A flight wholly inside the module footprint, missed by corner sampling.
    scene.envelope.as_mut().unwrap().mezzanine = Some(Mezzanine {
        deck: FloorPatch {
            min: middle - Vec2::splat(0.04),
            max: middle + Vec2::splat(0.04),
            height: lo.y - 2.20,
        },
        thickness: 0.20,
        stair_min: middle - Vec2::splat(0.025),
        stair_max: middle + Vec2::splat(0.025),
        stair_axis: 0,
        steps: 5,
        rail_height: 1.1,
    });
    assert!((sampling::highest_floor(&scene, lo.xz(), hi.xz()) - (lo.y - 2.20)).abs() < 0.001);
    assert!((m.headroom(&scene) - 2.20).abs() < 0.001);
    for seed in [22, 43, 60, 89, 114, 126, 160, 192, 221] {
        let s =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        for m in &s.ceiling_installation.as_ref().unwrap().modules {
            let (lo, hi) = m.bounds();
            assert!(
                polygon::box_inside(
                    &s.envelope.as_ref().unwrap().footprint,
                    lo.xz(),
                    hi.xz(),
                    0.24
                ),
                "seed {seed}: window covering clearance"
            );
        }
    }
}
