//! CPU-only correctness and bounded population receipts, safe beside training.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::preview, layout::IndoorLayout, materials, validation,
};
use image::{Rgb, RgbImage};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{collections::BTreeSet, time::Instant};

fn check(a: &Assembly) -> usize {
    let mut triangles = 0;
    for ((surface, label), g) in &a.parts {
        assert!(
            crate::render::semantic::SemanticLabel::from_label(label.split('#').next().unwrap())
                .is_some(),
            "{label}"
        );
        if let Some(slot) = materials::finish_slot(*surface, label) {
            assert!(
                materials::registered_finish(*surface, slot),
                "{surface:?} {label}"
            );
        }
        assert_eq!(g.positions.len(), g.normals.len());
        assert_eq!(g.positions.len(), g.uvs.len());
        for (p, n) in g.positions.iter().zip(&g.normals) {
            assert!(Vec3::from_array(*p).is_finite());
            assert!((Vec3::from_array(*n).length() - 1.).abs() < 1e-3);
        }
        assert!(g.uvs.iter().flatten().all(|v| v.is_finite()));
        assert_eq!(g.indices.len() % 3, 0);
        for t in g.indices.as_chunks::<3>().0 {
            let [v0, v1, v2] = t.map(|i| Vec3::from_array(g.positions[i as usize]));
            let cross = (v1 - v0).cross(v2 - v0);
            assert!(cross.length_squared() > 1e-14, "degenerate {label}");
            let normal: Vec3 = t
                .iter()
                .map(|&i| Vec3::from_array(g.normals[i as usize]))
                .sum();
            assert!(cross.dot(normal) > 0., "winding {label}");
            triangles += 1;
        }
    }
    triangles
}

fn fingerprint(a: &Assembly) -> String {
    let mut hash = Sha256::new();
    for ((surface, label), g) in &a.parts {
        hash.update((*surface as u32).to_le_bytes());
        hash.update(label.as_bytes());
        for v in g
            .positions
            .iter()
            .chain(&g.normals)
            .flatten()
            .chain(g.uvs.iter().flatten())
        {
            hash.update(v.to_le_bytes());
        }
        for i in &g.indices {
            hash.update(i.to_le_bytes());
        }
    }
    format!("{:x}", hash.finalize())
}

#[test]
fn building_programs_are_bounded_replayable_and_structurally_varied() {
    let mut roofs = BTreeSet::new();
    let mut finishes = BTreeSet::new();
    let mut hashes = BTreeSet::new();
    let mut wings = 0;
    for seed in 0..256 {
        let p = Parcel {
            center: Vec2::new(11., -9.),
            size: Vec2::new(
                2.4 + (seed % 19) as f32 * 0.35,
                2.3 + (seed % 13) as f32 * 0.40,
            ),
            yaw: seed as f32 * 0.41,
            kind: 0,
            seed,
        };
        let b = Building::sample(&p, 30.);
        let mut a = Assembly::default();
        buildings::build(&mut a, &p, &b);
        let n = check(&a);
        assert!(n < 12_000, "unbounded exterior facade {n}");
        let inverse = Transform::from_translation(Vec3::new(p.center.x, 0., p.center.y))
            .with_rotation(Quat::from_rotation_y(p.yaw))
            .compute_affine()
            .inverse();
        for v in a.parts.values().flat_map(|g| &g.positions) {
            let world = Vec3::from_array(*v);
            let local = inverse.transform_point3(world);
            assert!(world.length() < 30.);
            assert!(
                local.x.abs() <= p.size.x * 0.5 + 0.30 && local.z.abs() <= p.size.y * 0.5 + 0.30
            );
        }
        let mut replay = Assembly::default();
        buildings::build(&mut replay, &p, &Building::sample(&p, 30.));
        assert_eq!(fingerprint(&a), fingerprint(&replay));
        hashes.insert(fingerprint(&a));
        roofs.insert(b.roof);
        finishes.insert(b.wall);
        wings += usize::from(b.split < 1.);
    }
    assert_eq!(roofs.len(), 3);
    assert_eq!(finishes.len(), 5);
    assert_eq!(hashes.len(), 256);
    assert!((20..120).contains(&wings), "realized wings {wings}");
}

#[test]
fn parapets_and_grounded_plants_have_continuous_support() {
    let scene = IndoorManifest::generate_with_humans(6, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    for seed in 0..32 {
        let p = Parcel {
            center: Vec2::ZERO,
            size: Vec2::new(6., 4.),
            yaw: 0.,
            kind: 1,
            seed,
        };
        let mut b = Building::sample(&p, 30.);
        b.roof = 0;
        b.split = 1.;
        b.terrace_level = b.levels;
        let mut a = Assembly::default();
        buildings::build(&mut a, &p, &b);
        // Sweep a vertical line through a parapet and every slab beneath it.
        // Signed events union overlapping closed components, exposing real air
        // gaps without mistaking deliberately embedded supports for collisions.
        let point = Vec2::new(p.size.x * 0.5 - 0.0317, 0.1173);
        let mut events = Vec::new();
        for g in a.parts.values() {
            for tri in g.indices.as_chunks::<3>().0 {
                let [v0, v1, v2] = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                let (u, v, q) = ((v1 - v0).xz(), (v2 - v0).xz(), point - v0.xz());
                let det = u.perp_dot(v);
                if det.abs() < 1e-9 {
                    continue;
                }
                let (x, z) = (q.perp_dot(v) / det, u.perp_dot(q) / det);
                if x < 0. || z < 0. || x + z > 1. {
                    continue;
                }
                let height = v0.y + x * (v1.y - v0.y) + z * (v2.y - v0.y);
                events.push((height, if det > 0. { 1 } else { -1 }));
            }
        }
        events.sort_by(|a, b| a.0.total_cmp(&b.0));
        let mut winding = 0;
        for pair in events.windows(2) {
            winding += pair[0].1;
            assert!(
                winding > 0 || pair[1].0 - pair[0].0 < 1e-5,
                "seed {seed}: unsupported parapet/slab interval {pair:?}"
            );
        }
        assert_eq!(winding + events.last().unwrap().1, 0);
        let mut plants = Assembly::default();
        garden(&mut plants, &scene, &p);
        let root = plants
            .parts
            .iter()
            .filter(|((s, _), _)| *s == Surface::Bark)
            .flat_map(|(_, g)| g.positions.iter().map(|p| p[1]))
            .fold(f32::INFINITY, f32::min);
        assert!((0.024..=0.030).contains(&root), "floating root {root}");
    }
}

#[test]
#[ignore = "64-room exterior mesh/distribution audit and CPU previews; writes out/exterior_diversity"]
fn exterior_population_qualification() {
    let dir = std::path::Path::new("out/exterior_diversity");
    std::fs::create_dir_all(dir).unwrap();
    let mut samples = Vec::new();
    let mut hashes = BTreeSet::new();
    let mut sheet = RgbImage::from_pixel(1600, 1600, Rgb([241, 243, 246]));
    let mut building_sheet = sheet.clone();
    for seed in 0..64 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
        validation::validate_layout(&scene).unwrap();
        let start = Instant::now();
        let mut a = Assembly::default();
        build(&mut a, &scene);
        let build_ms = start.elapsed().as_secs_f64() * 1000.;
        let triangles = check(&a);
        let hash = fingerprint(&a);
        hashes.insert(hash.clone());
        let sites = parcels(&scene);
        let mut programs = Vec::new();
        for p in sites.iter().filter(|p| p.kind == 0) {
            programs.push(json!({"parcel":p,"program":Building::sample(p,super::super::background::radius(&scene))}));
        }
        let image = preview::orthographic(
            &a,
            Vec3::Y * 3.,
            super::super::background::radius(&scene) * 1.8,
            200,
            Vec3::new(0.6, 0.9, 1.),
        );
        image::imageops::replace(
            &mut sheet,
            &image,
            (seed % 8) as i64 * 200,
            (seed / 8) as i64 * 200,
        );
        // Fixed metric footprint strips away size and neighborhood variation.
        let p = Parcel {
            center: Vec2::ZERO,
            size: Vec2::new(7.5, 5.8),
            yaw: 0.,
            kind: 0,
            seed,
        };
        let b = Building::sample(&p, 30.);
        let mut sample = Assembly::default();
        buildings::build(&mut sample, &p, &b);
        let height = b.levels as f32 * b.pitch + b.roof_rise;
        let image = preview::orthographic(
            &sample,
            Vec3::Y * height * 0.5,
            20.,
            200,
            Vec3::new(0.65, 0.4, 1.),
        );
        image::imageops::replace(
            &mut building_sheet,
            &image,
            (seed % 8) as i64 * 200,
            (seed / 8) as i64 * 200,
        );
        let full = validation::validate_geometry(&scene).unwrap();
        assert_eq!(full.degenerate_triangles, 0);
        let tagged = super::super::architecture(&scene);
        assert!(tagged.parts.keys().any(|(_, l)| l == "sky#exterior"));
        for (s, label) in a.parts.keys() {
            assert!(tagged
                .parts
                .contains_key(&(*s, format!("{label}#exterior"))));
        }
        samples.push(json!({"seed":seed,"layout":scene.layout,"district":District::sample(seed),
            "parcel_counts":counts(&scene),"window_normal_context_fraction":window_context_fraction(&scene),
            "construction_ms":build_ms,"triangles":triangles,"batches":a.parts.len(),"mesh_sha256":hash,
            "building_programs":programs,"measurements":measurements(&scene),"validated_total_triangles":full.triangles}));
    }
    assert_eq!(hashes.len(), 64);
    sheet.save(dir.join("neighborhoods.png")).unwrap();
    building_sheet.save(dir.join("buildings.png")).unwrap();
    std::fs::write(dir.join("report.json"), serde_json::to_vec_pretty(&json!({
        "generator_revision":super::super::super::layout::GENERATOR_VERSION,
        "qualification":"CPU geometry, replay, annotation ownership, distribution and flat shaded previews; no GPU/PBR render or training evaluation",
        "samples":samples,"unique_meshes":hashes.len()
    })).unwrap()).unwrap();
}
