use super::*;
use crate::scene::procedural_indoor::{
    items::sizing,
    objects::{audit, build_object, containers::bins},
};
use std::collections::{BTreeMap, BTreeSet};

fn metrics(kind: ObjectKind, seed: u64) -> IndoorObject {
    let mut o = audit::fixture(kind, seed);
    let mut r = stream(seed, 0x53544f5241474554);
    o.size = match kind {
        ObjectKind::CoatRack => racks::dimensions(&mut r),
        ObjectKind::TrashCan => bins::dimensions(&mut r),
        _ => sizing::loose(kind, &mut r).unwrap(),
    };
    o
}

#[test]
fn storage_and_rack_extremes_fit_replay_and_remain_grounded() {
    for (kind, sizes) in [
        (
            ObjectKind::StorageBox,
            [[0.17, 0.075, 0.12], [0.55, 0.40, 0.38], [0.28, 0.24, 0.45]],
        ),
        (
            ObjectKind::TrashCan,
            [[0.24, 0.25, 0.18], [0.46, 0.82, 0.55], [0.40, 0.38, 0.27]],
        ),
        (
            ObjectKind::CoatRack,
            [[0.43, 1.42, 0.40], [1.28, 1.90, 0.43], [0.73, 1.96, 0.80]],
        ),
    ] {
        for seed in 0..96 {
            for size in sizes {
                let mut o = audit::fixture(kind, seed);
                o.size = Vec3::from_array(size);
                let a = build_object(&o);
                let q = audit::check(&o, &a)
                    .unwrap_or_else(|e| panic!("{kind:?}/{seed}/{size:?}: {e}"));
                assert!(q.max_envelope_overrun_m < 0.0005, "{kind:?}/{seed}: {q:?}");
                assert!(
                    a.bounds().0.y.abs() < 0.0001,
                    "{kind:?}/{seed}: {:?}",
                    a.bounds()
                );
                assert!(q.triangles < 10_000, "{kind:?}/{seed}: {q:?}");
                let b = build_object(&o);
                assert_eq!(a.parts.len(), b.parts.len());
                for (k, g) in &a.parts {
                    let h = &b.parts[k];
                    assert_eq!(g.positions, h.positions);
                    assert_eq!(g.normals, h.normals);
                    assert_eq!(g.uvs, h.uvs);
                    assert_eq!(g.indices, h.indices);
                }
            }
        }
    }
}

// Independent vertical triangle intersection: a dark plane or closed solid
// cannot masquerade as an open cavity in this check.
fn top_at(a: &Assembly, q: Vec2) -> f32 {
    let mut y = f32::NEG_INFINITY;
    for g in a.parts.values() {
        for ids in g.indices.as_chunks::<3>().0 {
            let [a, b, c] = ids.map(|i| Vec3::from_array(g.positions[i as usize]));
            let e = b.xz() - a.xz();
            let f = c.xz() - a.xz();
            let v = q - a.xz();
            let det = e.perp_dot(f);
            if det.abs() < 1e-10 {
                continue;
            }
            let u = v.perp_dot(f) / det;
            let w = e.perp_dot(v) / det;
            if u >= -1e-5 && w >= -1e-5 && u + w <= 1.00001 {
                y = y.max(a.y + u * (b.y - a.y) + w * (c.y - a.y));
            }
        }
    }
    y
}

#[test]
fn open_containers_have_real_cavities_and_closed_lids_have_seams() {
    let mut open = 0;
    let mut bin_modes = BTreeSet::new();
    for seed in 0..256 {
        let o = metrics(ObjectKind::StorageBox, seed);
        let p = boxes::parameters(&o);
        let a = build_object(&o);
        let hit = top_at(&a, Vec2::ZERO);
        if p.lid == 0. {
            open += 1;
            assert!((hit - p.floor).abs() < 1e-5, "{seed}: false cavity {hit}");
        } else {
            assert!((hit - o.size.y).abs() < 1e-5);
            assert!(p.seam > o.size.y * 0.003);
        }
        for side in [-1., 1.] {
            let path = p.handle_path(side);
            for v in [path[0], *path.last().unwrap()] {
                let h = p.half_at(v.y);
                let equation =
                    (v.x / h.x).abs().powf(p.roundness) + (v.z / h.y).abs().powf(p.roundness);
                assert!((equation - 1.).abs() < 1e-4, "handle detached");
            }
        }
        let o = metrics(ObjectKind::TrashCan, seed);
        let p = bins::parameters(&o);
        bin_modes.insert(p.lid);
        if p.lid_height == 0. {
            let a = build_object(&o);
            let x = p.body.x * if p.divider_x >= 0. { -0.20 } else { 0.20 };
            let hit = top_at(&a, Vec2::new(x, 0.));
            assert!(
                (hit - p.floor).abs() < 1e-5,
                "{seed}: bin cavity obscured {hit}, {p:?}"
            );
        }
    }
    assert!((60..145).contains(&open), "open boxes {open}/256");
    assert_eq!(bin_modes.len(), 5);
    println!("storage cavity audit: open_boxes={open}/256, bin_modes={bin_modes:?}");
}

#[test]
fn rack_support_hook_and_rail_programs_are_independent_and_clear() {
    let mut forms = [0; 2];
    let mut hooks = BTreeSet::new();
    let mut feet = BTreeSet::new();
    let mut occupancy = BTreeSet::new();
    let mut phases = BTreeSet::new();
    for seed in 0..256 {
        let o = metrics(ObjectKind::CoatRack, seed);
        let p = racks::parameters(&o);
        forms[p.rail as usize] += 1;
        feet.insert(p.feet);
        hooks.insert(p.hooks.len());
        occupancy.insert(p.hanger_x.len());
        assert!(p.foot_span.max_element() + p.upright * 2. < o.size.x.max(o.size.z) * 0.5);
        if p.rail {
            for (i, x) in p.hanger_x.iter().enumerate() {
                assert!(x.abs() < p.rail_width * 0.41);
                for other in &p.hanger_x[i + 1..] {
                    assert!((x - other).abs() > p.upright * 5.);
                }
            }
        } else {
            phases.insert(((p.hooks[0].angle - p.foot_phase).rem_euclid(TAU) * 100.) as u32);
            for h in &p.hooks {
                assert!(h.root_height + h.rise + p.upright * 1.05 < p.height);
                assert!(h.reach + p.upright * 1.05 < p.radius);
            }
            // Actual ground contacts must enclose a nonzero support area.
            let a = build_object(&o);
            let points: Vec<_> = a
                .parts
                .values()
                .flat_map(|g| &g.positions)
                .map(|p| Vec3::from_array(*p))
                .filter(|p| p.y.abs() < 1e-5)
                .map(|p| p.xz())
                .collect();
            let lo = points
                .iter()
                .fold(Vec2::splat(f32::INFINITY), |a, &b| a.min(b));
            let hi = points
                .iter()
                .fold(Vec2::splat(f32::NEG_INFINITY), |a, &b| a.max(b));
            assert!((hi - lo).min_element() > o.size.x.min(o.size.z) * 0.35);
        }
    }
    assert!(forms.iter().all(|v| *v > 70));
    assert!(hooks.len() >= 5);
    assert_eq!(feet.len(), 3);
    assert!(occupancy.len() >= 4);
    assert!(phases.len() > 70);
    println!("rack audit: forms={forms:?}, hook_counts={hooks:?}, feet={feet:?}, hanger_occupancy={occupancy:?}");
}

#[test]
fn storage_finish_seeds_preserve_geometry_and_semantic_ownership() {
    for kind in [
        ObjectKind::StorageBox,
        ObjectKind::TrashCan,
        ObjectKind::CoatRack,
    ] {
        let mut changed = 0;
        for seed in 0..64 {
            let o = metrics(kind, seed);
            let a = build_object(&o);
            let mut o2 = o.clone();
            o2.finish_seed = Some(seed ^ 0xcafe1234);
            let b = build_object(&o2);
            let shape = |a: &Assembly| {
                let mut v: Vec<_> = a
                    .parts
                    .values()
                    .flat_map(|g| g.positions.iter().map(|p| p.map(f32::to_bits)))
                    .collect();
                v.sort_unstable();
                v
            };
            assert_eq!(shape(&a), shape(&b));
            assert!(a
                .parts
                .keys()
                .all(|(_, l)| part_label(l) == kind.class_name()));
            changed += usize::from(
                a.parts.keys().collect::<Vec<_>>() != b.parts.keys().collect::<Vec<_>>(),
            );
        }
        assert!(changed > 20, "{kind:?} finish stream collapsed");
    }
}

#[test]
fn storage_metric_sampling_and_gallery() {
    let output = std::env::var_os("ZEROVERSE_OBJECT_REPORT").map(std::path::PathBuf::from);
    let mut image = output
        .as_ref()
        .map(|_| image::RgbImage::new(8 * 192, 3 * 192));
    let mut report = BTreeMap::new();
    for (row, kind) in [
        ObjectKind::StorageBox,
        ObjectKind::TrashCan,
        ObjectKind::CoatRack,
    ]
    .into_iter()
    .enumerate()
    {
        let mut samples = Vec::new();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::ZERO;
        for seed in 0..128 {
            let o = metrics(kind, seed);
            lo = lo.min(o.size);
            hi = hi.max(o.size);
            let a = build_object(&o);
            let q = audit::check(&o, &a).unwrap_or_else(|e| panic!("{kind:?}/{seed}: {e}"));
            samples.push(serde_json::json!({"seed":seed,"size":o.size,"program":audit::program(&o),"mesh":q}));
            if let Some(image) = image.as_mut().filter(|_| seed < 8) {
                let (span, height) = match kind {
                    ObjectKind::StorageBox => (0.73, 0.18),
                    ObjectKind::TrashCan => (1.1, 0.40),
                    _ => (2.7, 0.95),
                };
                let frame = crate::scene::procedural_indoor::geometry::preview::orthographic(
                    &a,
                    Vec3::Y * height,
                    span,
                    192,
                    Vec3::new(1.1, 1.25, 1.6),
                );
                image::imageops::replace(image, &frame, seed as i64 * 192, row as i64 * 192);
            }
        }
        assert!(
            (hi / lo).cmpgt(Vec3::splat(1.25)).all(),
            "{kind:?}: {lo:?}..{hi:?}"
        );
        report.insert(
            format!("{kind:?}"),
            serde_json::json!({"samples":samples,"min_m":lo,"max_m":hi}),
        );
    }
    if let Some(path) = output {
        std::fs::create_dir_all(&path).unwrap();
        image
            .unwrap()
            .save(path.join("storage-metric.png"))
            .unwrap();
        std::fs::write(
            path.join("storage-metric.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
    }
}

#[test]
fn storage_programs_survive_real_room_admission() {
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest},
        validation,
    };
    let mut counts = BTreeMap::<String, usize>::new();
    let mut items = Vec::new();
    for profile in [
        IndoorLayout::Reception,
        IndoorLayout::Workshop,
        IndoorLayout::Coworking,
        IndoorLayout::OpenOffice,
    ] {
        for seed in 2..34 {
            let scene = IndoorManifest::generate_with_humans(seed, profile, 0.65, 0, 0.).unwrap();
            validation::validate_layout(&scene)
                .unwrap_or_else(|e| panic!("{profile:?}/{seed}: {e}"));
            for o in scene.objects.iter().filter(|o| {
                !o.neighbor
                    && matches!(
                        o.kind,
                        ObjectKind::StorageBox | ObjectKind::TrashCan | ObjectKind::CoatRack
                    )
            }) {
                let a = build_object(o);
                audit::check(o, &a).unwrap();
                assert!(a.bounds().0.y.abs() < 0.0001);
                assert!(a.bounds().1.y <= o.size.y + 0.0005);
                *counts.entry(format!("{:?}", o.kind)).or_default() += 1;
                items.push(serde_json::json!({"layout":profile,"room_seed":seed,"kind":o.kind,"seed":o.seed,"size":o.size,"position":o.position,"yaw":o.yaw,"support":o.support,"program":audit::program(o),"mesh_bounds":a.bounds()}));
            }
        }
    }
    assert_eq!(counts.len(), 3, "missing admitted category: {counts:?}");
    assert!(
        counts.values().all(|v| *v >= 4),
        "too few admitted cases: {counts:?}"
    );
    println!("storage accepted room audit: {counts:?}");
    if let Some(path) = std::env::var_os("ZEROVERSE_OBJECT_REPORT").map(std::path::PathBuf::from) {
        std::fs::create_dir_all(&path).unwrap();
        std::fs::write(path.join("storage-admission.json"),serde_json::to_vec_pretty(&serde_json::json!({"rooms":128,"profiles":["Reception","Workshop","Coworking","OpenOffice"],"seeds":"2..34","density":0.65,"cameras":0,"humans":0,"layout_and_changed_meshes_validated":true,"counts":counts,"items":items})).unwrap()).unwrap();
    }
}
