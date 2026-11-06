//! Realized geometry/support checks and an opt-in local catalog receipt.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::preview,
    layout::{IndoorLayout, IndoorManifest},
    validation,
};
use serde_json::json;
use std::collections::BTreeSet;

fn sample(category: usize, seed: u64) -> IndoorObject {
    let mut r = stream(seed, 0x4449564552534954);
    let kind = [
        ObjectKind::TrashCan,
        ObjectKind::TrashCan,
        ObjectKind::TeachingBoard,
        ObjectKind::Sofa,
        ObjectKind::Chair,
        ObjectKind::Rug,
    ][category];
    let mut o = fixture(kind, seed);
    match category {
        0 | 1 => {
            o.size = containers::bins::dimensions(&mut r);
            if category == 1 {
                o.variant = containers::bins::WASTE_STATION
                    | if seed.is_multiple_of(2) {
                        containers::bins::RECYCLING
                    } else {
                        0
                    };
            }
        }
        3 => o.size = seating::dimensions(&mut r),
        4 => {
            o.variant = [0, 1, 5, 9][seed as usize % 4];
            o.size = Vec3::new(
                r.random_range(0.60..0.78),
                r.random_range(0.92..1.45),
                r.random_range(0.60..0.78),
            );
        }
        5 => o.size = Vec3::new(r.random_range(0.65..6.5), 0.007, r.random_range(0.5..4.5)),
        _ => (),
    }
    o
}

#[test]
fn furnishing_diversity_meshes_fit_metric_envelopes_and_replay() {
    for category in 0..6 {
        for seed in 0..64 {
            let o = sample(category, seed);
            let a = build_object(&o);
            check(&o, &a)
                .unwrap_or_else(|e| panic!("category {category}/{seed}: {e}; {}", program(&o)));
            if category == 4 {
                let boxes = chairs::clearance_boxes(&o);
                for point in a.parts.values().flat_map(|g| &g.positions) {
                    let point = Vec3::from_array(*point);
                    assert!(
                        boxes
                            .iter()
                            .any(|(lo, hi)| point.cmpge(*lo - Vec3::splat(0.003)).all()
                                && point.cmple(*hi + Vec3::splat(0.003)).all()),
                        "chair {seed}: {point}"
                    );
                }
            }
            if seed < 4 {
                let b = build_object(&o);
                for (key, g) in &a.parts {
                    assert_eq!(g.positions, b.parts[key].positions);
                    assert_eq!(g.normals, b.parts[key].normals);
                    assert_eq!(g.indices, b.parts[key].indices);
                }
            }
        }
    }
}

#[test]
fn furnishing_diversity_has_distinct_supports_proportions_and_real_waste_apertures() {
    let mut post_spans = BTreeSet::new();
    let mut unaligned_backs = 0;
    let mut double_returns = 0;
    let mut bins_lo = Vec3::splat(f32::INFINITY);
    let mut bins_hi = Vec3::ZERO;
    for seed in 0..256 {
        let bin = sample(1, seed);
        bins_lo = bins_lo.min(bin.size);
        bins_hi = bins_hi.max(bin.size);
        let board = public::furnishing::parameters(&sample(2, seed));
        post_spans.insert((board.post_span * 10.) as u32);
        let sofa = seating::parameters(&sample(3, seed));
        unaligned_backs += usize::from(sofa.back_modules != sofa.modules);
        double_returns += usize::from(sofa.double_return);
        if seed < 16 {
            let a = build_object(&bin);
            let label = if seed.is_multiple_of(2) {
                "trash_can#finish5"
            } else {
                "trash_can#finish1"
            };
            let lid = &a.parts[&(Surface::Plastic, label.to_owned())];
            let p = containers::bin_parameters(&bin);
            // A vertical ray through the aperture cannot hit a hidden disk cap.
            for tri in lid.indices.as_chunks::<3>().0 {
                // Material batching may merge matching body/hinge finishes;
                // test the cap above the rim, not the intentional bin floor.
                if tri
                    .iter()
                    .any(|&i| lid.positions[i as usize][1] < p.body.y + p.seam - 1e-5)
                {
                    continue;
                }
                let [a, b, c] = [tri[0], tri[1], tri[2]]
                    .map(|i| Vec3::from_array(lid.positions[i as usize]).xz());
                let d = (b - a).perp_dot(c - a);
                if d.abs() < 1e-10 {
                    continue;
                }
                let u = b.perp_dot(c) / d;
                let v = c.perp_dot(a) / d;
                let w = a.perp_dot(b) / d;
                assert!(
                    u < -1e-5 || v < -1e-5 || w < -1e-5,
                    "bin {seed}: capped aperture"
                );
            }
        }
    }
    assert!(post_spans.len() >= 6 && unaligned_backs > 120 && double_returns > 1);
    assert!((bins_hi / bins_lo).cmpgt(Vec3::new(2.4, 3.0, 3.0)).all());
    println!(
        "{}",
        json!({"samples":256,"post_span_bins":post_spans,"independent_sofa_backs":unaligned_backs,
        "double_returns":double_returns,"bin_min_m":bins_lo,"bin_max_m":bins_hi})
    );
}

#[test]
#[ignore = "local mesh gallery and matched room diversity qualification"]
fn furnishing_diversity_qualification() {
    let dir = std::path::PathBuf::from("out/furnishing_diversity_20261106");
    std::fs::create_dir_all(&dir).unwrap();
    let names = [
        "bins",
        "waste_stations",
        "mobile_whiteboards",
        "sofas",
        "office_chairs",
        "rugs",
    ];
    let n = 192;
    let mut sheet = image::RgbImage::new(n * 8, n * 6);
    let mut objects = Vec::new();
    for (row, name) in names.iter().enumerate() {
        let mut gallery = image::RgbImage::new(n * 8, n);
        for seed in 0..64 {
            let o = sample(row, seed);
            let started = std::time::Instant::now();
            let a = build_object(&o);
            let elapsed = started.elapsed().as_secs_f64() * 1000.;
            let audit = check(&o, &a).unwrap();
            objects.push(json!({"category":name,"seed":seed,"size_m":o.size,"program":program(&o),"mesh":audit,"build_ms":elapsed}));
            if seed < 8 {
                let image = preview::orthographic(
                    &a,
                    Vec3::Y * o.size.y * 0.5,
                    o.size.length() * 1.05,
                    n,
                    Vec3::new(1.2, 0.8, -2.),
                );
                image::imageops::replace(&mut gallery, &image, seed as i64 * n as i64, 0);
                image::imageops::replace(
                    &mut sheet,
                    &image,
                    seed as i64 * n as i64,
                    row as i64 * n as i64,
                );
            }
        }
        gallery.save(dir.join(format!("{name}.png"))).unwrap();
    }
    sheet.save(dir.join("geometry.png")).unwrap();
    let mut rooms = Vec::new();
    let mut errors = Vec::new();
    for seed in 0..128 {
        let profile = [
            IndoorLayout::Mixed,
            IndoorLayout::Lounge,
            IndoorLayout::Conference,
            IndoorLayout::Cafe,
            IndoorLayout::Cafeteria,
            IndoorLayout::Classroom,
            IndoorLayout::Reception,
            IndoorLayout::OpenOffice,
        ][seed as usize % 8];
        let started = std::time::Instant::now();
        let scene = match IndoorManifest::generate_with_humans(seed, profile, 0.65, 3, 0.35) {
            Ok(scene) => scene,
            Err(e) => {
                errors.push(format!("{profile:?}/{seed}: {e}"));
                continue;
            }
        };
        let ms = started.elapsed().as_secs_f64() * 1000.;
        if let Err(e) = validation::validate_layout(&scene) {
            errors.push(format!("{profile:?}/{seed}: {e}"));
        }
        if seed < 32 {
            if let Err(e) = validation::validate_geometry(&scene) {
                errors.push(format!("mesh {profile:?}/{seed}: {e}"));
            }
        }
        let bins: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::TrashCan)
            .collect();
        for partner in bins
            .iter()
            .filter(|o| o.variant & containers::bins::RECYCLING != 0)
        {
            let source = &scene.objects[partner.interaction_target.unwrap()];
            assert_eq!(source.kind, ObjectKind::TrashCan);
            assert_eq!(source.size, partner.size);
            assert!(!crate::scene::procedural_indoor::layout::furniture_overlap(
                source, partner, 0.03
            ));
        }
        rooms.push(json!({"seed":seed,"profile":profile,"generation_ms":ms,"objects":scene.objects.len(),
            "people":scene.humans.len(),"waste_pairs":bins.iter().filter(|o| o.variant & containers::bins::RECYCLING != 0).count(),
            "rugs":scene.objects.iter().filter(|o| o.kind == ObjectKind::Rug).map(|o| json!({"size_m":o.size,"yaw":o.yaw})).collect::<Vec<_>>()}));
    }
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&json!({
        "generator":crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
        "rendering":"CPU flat mesh previews, not PBR", "objects":objects,"rooms":rooms,"errors":errors})).unwrap()).unwrap();
    assert!(errors.is_empty(), "{}", errors.join("\n"));
    assert!(
        rooms
            .iter()
            .filter(|r| r["waste_pairs"].as_u64().unwrap() > 0)
            .count()
            >= 8
    );
}
