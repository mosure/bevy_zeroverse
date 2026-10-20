use super::*;
use crate::scene::procedural_indoor::{objects, validation};

#[test]
fn panel_groups_replay_have_positive_sizes_and_separated_frames() {
    let mut scene =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut unequal = 0;
    let mut staggered = 0;
    let mut aligned = 0;
    for seed in 0..256 {
        scene.seed = seed;
        let p = WallComposition::sample(&scene);
        assert_eq!(p, WallComposition::sample(&scene));
        assert!((p.weights.iter().sum::<f32>() - 1.).abs() < 1e-5);
        let mut rng = stream(seed, 9810);
        for family in 0..3 {
            for span in [0.35, 0.8, 1.5, 3., 8., 16.] {
                let mut replay = rng.clone();
                let panels = p.panels(family, span, &mut rng);
                assert_eq!(panels, p.panels(family, span, &mut replay));
                for (i, (_, pos, size)) in panels.iter().enumerate() {
                    assert!(size.cmpgt(Vec3::ZERO).all());
                    assert!(pos.x.abs() + size.x * 0.5 <= span * 0.5 - 0.10 + 1e-5);
                    for (_, other, s) in &panels[..i] {
                        assert!(
                            (pos.x - other.x).abs() >= (size.x + s.x) * 0.5 + 0.01
                                || (pos.y + size.y * 0.5 - other.y - s.y * 0.5).abs()
                                    >= (size.y + s.y) * 0.5 + 0.01
                        );
                    }
                }
                if family == 0 && panels.len() > 1 {
                    let centers: Vec<_> = panels.iter().map(|(_, p, s)| p.y + s.y * 0.5).collect();
                    let heights = centers.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                        - centers.iter().copied().fold(f32::INFINITY, f32::min);
                    let areas: Vec<_> = panels.iter().map(|(_, _, s)| s.x * s.y).collect();
                    let ratio = areas.iter().copied().fold(f32::NEG_INFINITY, f32::max)
                        / areas.iter().copied().fold(f32::INFINITY, f32::min);
                    staggered += usize::from(heights > 0.20);
                    unequal += usize::from(ratio > 1.5);
                    aligned += usize::from(heights < 1e-5 && ratio < 1.001);
                }
            }
        }
    }
    assert!(staggered > 150 && unequal > 150);
    assert!(
        aligned > 20,
        "intentional matched series must remain possible"
    );
}

#[test]
fn wall_statistics_distinguish_parallel_walls_and_ignore_non_wall_art() {
    let mut scene =
        IndoorManifest::generate_with_humans(17, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    let mut base = objects::audit::fixture(ObjectKind::WallArt, 7);
    base.position = Vec3::new(0., 1., 0.);
    base.yaw = 0.;
    scene.objects.push(base.clone());
    let mut upper = base.clone();
    upper.position.y = 2.;
    scene.objects.push(upper);
    let mut parallel = base.clone();
    parallel.position.z = 3.;
    scene.objects.push(parallel);
    let mut neighbor = base.clone();
    neighbor.neighbor = true;
    scene.objects.push(neighbor);
    base.support = Some(0);
    scene.objects.push(base);
    let walls = mounted_art_walls(&scene);
    assert_eq!(walls.len(), 2);
    let stats = ArtWallStats::measure(&walls[0]);
    assert_eq!(stats.count, 2);
    assert_eq!(stats.pairs, 1);
    assert_eq!(stats.aligned_pairs, 0);
    assert_eq!(stats.matching_size_pairs, 1);
    assert_eq!(stats.stacked_pairs, 1);
    assert!((stats.center_height_span_m - 1.).abs() < 1e-6);
}

#[test]
fn installed_art_retains_size_height_diversity_after_clearance_and_mesh_validation() {
    let mut rows = Vec::new();
    let mut elevations = Vec::new();
    let mut rooms_with_art = [0; 2];
    let mut multi_walls = 0;
    let mut staggered = 0;
    let mut unequal = 0;
    let mut stacked = 0;
    let mut checked_meshes = 0;
    let started = std::time::Instant::now();
    for (profile, layout) in [IndoorLayout::Gallery, IndoorLayout::Mixed]
        .into_iter()
        .enumerate()
    {
        for seed in 0..64 {
            let density = [0.40, 0.65, 0.90][seed as usize % 3];
            let scene =
                IndoorManifest::generate_with_humans(seed, layout, density, 3, 0.25).unwrap();
            validation::validate_layout(&scene)
                .unwrap_or_else(|e| panic!("{layout:?}/{seed}: {e}"));
            if seed < 4 {
                let replay =
                    IndoorManifest::generate_with_humans(seed, layout, density, 3, 0.25).unwrap();
                assert_eq!(
                    serde_json::to_vec(&scene).unwrap(),
                    serde_json::to_vec(&replay).unwrap()
                );
            }
            let walls = mounted_art_walls(&scene);
            rooms_with_art[profile] += usize::from(!walls.is_empty());
            for wall in walls {
                for o in &wall {
                    assert!(scene.wall_attachment_clear(o));
                    let mesh = objects::build_object(o);
                    objects::audit::check(o, &mesh)
                        .unwrap_or_else(|e| panic!("{layout:?}/{seed}/{}: {e}", o.id));
                    let (lo, hi) = o.bounds();
                    for other in scene
                        .objects
                        .iter()
                        .filter(|other| other.id != o.id && !other.neighbor)
                    {
                        // A hanging cable's outer box contains empty space.
                        // Use the same segment/rotor bounds as scene clearance.
                        for (a, b) in other.obstacle_bounds() {
                            assert!(
                                !(lo.cmplt(b - Vec3::splat(0.001)).all()
                                    && hi.cmpgt(a + Vec3::splat(0.001)).all()),
                                "{layout:?}/{seed}: art {} intersects {} {:?}",
                                o.id,
                                other.id,
                                other.kind
                            );
                        }
                    }
                    checked_meshes += 1;
                }
                let stats = ArtWallStats::measure(&wall);
                if stats.pairs > 0 {
                    multi_walls += 1;
                    staggered += usize::from(stats.center_height_span_m > 0.20);
                    unequal += usize::from(stats.area_ratio > 1.5);
                    stacked += usize::from(stats.stacked_pairs > 0);
                    if profile == 0 && elevations.len() < 12 {
                        elevations.push(elevation(seed, &scene, &wall, elevations.len()));
                    }
                }
                rows.push(serde_json::json!({
                    "requested_layout": format!("{layout:?}"), "seed": seed,
                    "sampled_layout": format!("{:?}", scene.layout),
                    "density": density, "stats": stats
                }));
            }
        }
    }
    let output = std::path::Path::new("out/wall_art_review");
    std::fs::create_dir_all(output).unwrap();
    std::fs::write(
        output.join("report.json"),
        serde_json::to_vec_pretty(&serde_json::json!({
            "rooms": 128, "rooms_per_requested_layout": 64, "rooms_with_art": rooms_with_art,
            "checked_art_meshes": checked_meshes,
            "multi_picture_walls": multi_walls, "walls_with_center_span_above_20cm": staggered,
            "walls_with_area_ratio_above_1_5": unequal, "walls_with_stacks": stacked,
            "cpu_generation_and_validation_seconds": started.elapsed().as_secs_f64(),
            "seed_range": [0, 63], "densities": [0.40, 0.65, 0.90],
            "camera_count": 3, "human_density": 0.25, "gpu_rendered": false,
            "gallery_selection": "first 12 accepted multi-picture walls in Gallery seeds 0..63",
            "walls": rows
        }))
        .unwrap(),
    )
    .unwrap();
    std::fs::write(output.join("wall-elevations.svg"), format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"1080\" height=\"860\" viewBox=\"0 0 1080 860\">\
         <rect width=\"1080\" height=\"860\" fill=\"#f3f1eb\"/>\
         <g font-family=\"sans-serif\" fill=\"#242e3b\">\
         <text x=\"24\" y=\"30\" font-size=\"19\">Sampled wall art — accepted placements</text>\
         <text x=\"24\" y=\"52\" font-size=\"12\">Wall elevations of frame bounds, not RGB renders. Heights in metres; each card scales to fit.</text>\
         {}</g></svg>", elevations.join("")
    )).unwrap();
    assert!(
        rooms_with_art.iter().all(|&n| n >= 20),
        "{rooms_with_art:?}"
    );
    assert!(
        multi_walls >= 30,
        "only {multi_walls} multi-picture walls admitted"
    );
    assert!(
        staggered * 2 > multi_walls,
        "{staggered}/{multi_walls} varied heights"
    );
    assert!(
        unequal * 2 > multi_walls,
        "{unequal}/{multi_walls} varied sizes"
    );
    assert!(
        stacked >= 8,
        "stacked arrangements rejected too often: {stacked}"
    );
}

fn elevation(seed: u64, scene: &IndoorManifest, wall: &[&IndoorObject], index: usize) -> String {
    use std::fmt::Write;
    let tangent = Quat::from_rotation_y(wall[0].yaw) * Vec3::X;
    let x = |o: &IndoorObject| tangent.dot(o.position);
    let lo = wall
        .iter()
        .map(|o| x(o) - o.size.x * 0.5)
        .fold(f32::INFINITY, f32::min)
        - 0.25;
    let hi = wall
        .iter()
        .map(|o| x(o) + o.size.x * 0.5)
        .fold(f32::NEG_INFINITY, f32::max)
        + 0.25;
    let scale = (304. / (hi - lo)).min(125. / scene.room_size.y);
    let stats = ArtWallStats::measure(wall);
    let mut svg = format!(
        "<g transform=\"translate({}, {})\"><rect width=\"336\" height=\"188\" rx=\"8\" fill=\"#fff\"/>\
         <text x=\"14\" y=\"21\" font-size=\"13\">Seed {seed} · {} works</text>\
         <text x=\"14\" y=\"39\" font-size=\"11\">Center span {:.2} m · area ratio {:.1}×</text>",
        24 + index % 3 * 352, 70 + index / 3 * 196,
        wall.len(), stats.center_height_span_m, stats.area_ratio
    );
    for o in wall {
        let left = 16. + (x(o) - o.size.x * 0.5 - lo) * scale;
        let top = 174. - (o.position.y + o.size.y) * scale;
        let color = ["#548d91", "#bd835d", "#8880a5", "#708561", "#bb9e5b"][o.seed as usize % 5];
        write!(
            svg,
            "<rect x=\"{left:.2}\" y=\"{top:.2}\" width=\"{:.2}\" height=\"{:.2}\" \
               fill=\"{color}\" stroke=\"#353c44\" stroke-width=\"1.5\"/>",
            o.size.x * scale,
            o.size.y * scale
        )
        .unwrap();
    }
    svg.push_str("<path d=\"M16 175H320\" stroke=\"#b4b9bc\"/></g>");
    svg
}
