use super::{geometry::Geometry, layout::*, validation::*};
use bevy::prelude::*;

#[test]
fn sampled_niches_keep_wall_mounts_on_solid_wall_and_reject_invalid_fields() {
    let mut niche_widths = std::collections::BTreeSet::new();
    for seed in 0..256 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.0).unwrap();
        validate_layout(&scene).unwrap();
        if scene.architecture_style != ArchitectureStyle::Classic {
            continue;
        }
        let (lo, hi) = super::architecture::details::niche_region(&scene);
        niche_widths.insert(((hi.x - lo.x) * 100.0) as i32);
        for o in &scene.objects {
            if !o.neighbor
                && o.support.is_none()
                && matches!(
                    o.kind,
                    ObjectKind::Display
                        | ObjectKind::Whiteboard
                        | ObjectKind::WallArt
                        | ObjectKind::Clock
                )
            {
                let (a, b) = o.bounds();
                assert!(
                    !super::architecture::details::overlaps_niche(&scene, a, b),
                    "seed {seed}: unsupported wall mount"
                );
            }
        }
    }
    assert!(niche_widths.len() > 15);
    let mut scene =
        IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.65, 0, 0.0).unwrap();
    scene.program.as_mut().unwrap().zones[0]
        .furnishing
        .as_mut()
        .unwrap()
        .curvature = f32::NAN;
    assert!(validate_layout(&scene).is_err());
}

#[test]
fn legacy_config_json_does_not_require_new_indoor_fields() {
    let mut json = serde_json::to_value(crate::app::BevyZeroverseConfig::default()).unwrap();
    let object = json.as_object_mut().unwrap();
    for key in [
        "indoor_seed",
        "indoor_layout",
        "indoor_density",
        "indoor_human_density",
        "indoor_quality",
        "indoor_gi_rays",
    ] {
        object.remove(key);
    }
    let config: crate::app::BevyZeroverseConfig = serde_json::from_value(json).unwrap();
    assert_eq!(config.indoor_seed, None);
    assert_eq!(config.indoor_layout, IndoorLayout::Mixed);
    assert_eq!(config.indoor_density, 0.65);
    assert_eq!(config.indoor_human_density, 0.25);
    assert_eq!(config.indoor_gi_rays, 256);
}

#[test]
fn seeded_replay_and_independent_camera_streams() {
    let a = IndoorManifest::generate(501, IndoorLayout::Mixed, 0.65, 4).unwrap();
    let b = IndoorManifest::generate(501, IndoorLayout::Mixed, 0.65, 4).unwrap();
    assert_eq!(a, b);
    let c = IndoorManifest::generate(501, IndoorLayout::Mixed, 0.65, 8).unwrap();
    assert_eq!(a.objects, c.objects);
    assert_eq!(a.cameras, c.cameras[..4]);
    assert_ne!(
        a.objects,
        IndoorManifest::generate(502, IndoorLayout::Mixed, 0.65, 4)
            .unwrap()
            .objects
    );
    let json = serde_json::to_string(&a).unwrap();
    assert_eq!(a, serde_json::from_str::<IndoorManifest>(&json).unwrap());
    validate_layout(&IndoorManifest::generate(u64::MAX, IndoorLayout::Mixed, 1.0, 8).unwrap())
        .unwrap();
}

#[test]
fn procedural_texture_uploads_have_complete_mips_and_correct_color_spaces() {
    use bevy::render::render_resource::TextureFormat;
    let scene = IndoorManifest::generate(3, IndoorLayout::Mixed, 0.5, 1).unwrap();
    let mut images = Assets::<Image>::default();
    let mut materials = Assets::<StandardMaterial>::default();
    let generated = super::materials::IndoorMaterials::build(&scene, &mut images, &mut materials);
    let wood = materials
        .get(&generated.get(super::materials::Surface::Wood))
        .unwrap();
    let albedo = images
        .get(wood.base_color_texture.as_ref().unwrap())
        .unwrap();
    let normal = images
        .get(wood.normal_map_texture.as_ref().unwrap())
        .unwrap();
    let roughness = images
        .get(wood.metallic_roughness_texture.as_ref().unwrap())
        .unwrap();
    assert_eq!(
        albedo.texture_descriptor.format,
        TextureFormat::Rgba8UnormSrgb
    );
    assert_eq!(normal.texture_descriptor.format, TextureFormat::Rgba8Unorm);
    assert_eq!(
        roughness.texture_descriptor.format,
        TextureFormat::Rgba8Unorm
    );
    for (id, image) in images.iter() {
        let size = image.width();
        let descriptor = &image.texture_descriptor;
        let diffuse_cube = generated.environment.diffuse_map.id() == id;
        assert_eq!(
            descriptor.mip_level_count,
            if diffuse_cube { 1 } else { size.ilog2() + 1 }
        );
        let pixels: u32 = (0..descriptor.mip_level_count)
            .map(|level| (size >> level).pow(2))
            .sum();
        assert_eq!(
            image.data.as_ref().unwrap().len(),
            (pixels
                * descriptor.size.depth_or_array_layers
                * if descriptor.format == TextureFormat::Rgba16Float {
                    8
                } else {
                    4
                }) as usize
        );
    }
    assert_eq!(
        wood.perceptual_roughness, 1.0,
        "roughness texture must not be multiplied twice"
    );
    let chrome = materials
        .get(&generated.get(super::materials::Surface::Chrome))
        .unwrap();
    assert_eq!(chrome.metallic, 1.);
    assert!(
        chrome.normal_map_texture.is_none(),
        "micron slopes must not become quantized visible bumps"
    );
    assert_eq!(
        generated.environment.intensity, 1.,
        "HDR radiance is already in cd/m²"
    );
    let specular = images.get(&generated.environment.specular_map).unwrap();
    assert_eq!(
        specular.texture_descriptor.format,
        TextureFormat::Rgba16Float
    );
    let values: Vec<_> = specular
        .data
        .as_ref()
        .unwrap()
        .as_chunks::<8>()
        .0
        .iter()
        .flat_map(|p| {
            p[..6]
                .as_chunks::<2>()
                .0
                .iter()
                .map(|b| half::f16::from_le_bytes(*b).to_f32())
        })
        .collect();
    assert!(values.iter().all(|v| v.is_finite() && *v >= 0.));
    assert!(
        values.iter().any(|v| *v > 1.),
        "room radiance was clipped to LDR"
    );
}

#[test]
fn distribution_is_valid_and_covers_all_families() {
    let report = audit(0, 1024, 8, 0.65);
    assert!(
        report.invalid_seeds.is_empty(),
        "{:?}",
        &report.invalid_seeds[..report.invalid_seeds.len().min(10)]
    );
    assert_eq!(report.layout_counts.len(), IndoorLayout::PROFILES.len());
    assert_eq!(report.palette_counts.len(), 6);
    assert_eq!(report.floor_counts.len(), 3);
    assert_eq!(report.lighting_counts.len(), 3);
    assert_eq!(report.architecture_counts.len(), 4);
    assert_eq!(report.plant_species_counts.len(), 6);
    for count in report.layout_counts.values() {
        assert!(
            (60..=145).contains(count),
            "layout distribution collapsed: {report:?}"
        );
    }
    assert!(report.camera_height_range[0] < 0.85 && report.camera_height_range[1] > 2.1);
    assert!(report.fov_range[0] < 49.0 && report.fov_range[1] > 73.0);
    let specific = audit_layout(0, 32, 2, 0.5, IndoorLayout::Conference);
    assert!(
        specific.invalid_seeds.is_empty(),
        "{:?}",
        specific.invalid_seeds
    );
    assert_eq!(specific.layout_counts.len(), 1);
    assert_eq!(specific.layout_counts["Conference"], 32);
}

#[test]
fn visibility_sampling_handles_high_back_seats_and_glazed_rooms() {
    for seed in [100075, 102210, 105756, 105940, 107796] {
        if let Err(error) = IndoorManifest::generate(seed, IndoorLayout::Mixed, 0.65, 2) {
            let scene = IndoorManifest::generate(seed, IndoorLayout::Mixed, 0.65, 0).unwrap();
            let heads: Vec<_> = scene
                .humans
                .iter()
                .filter(|h| !h.neighbor)
                .map(|h| {
                    (
                        h.joints[4].y,
                        h.chair
                            .and_then(|id| scene.objects.iter().find(|o| o.id == id))
                            .map(|o| o.size.y),
                    )
                })
                .collect();
            panic!("{error}; seated head/seat envelope heights: {heads:?}");
        }
    }
}

#[test]
fn all_grammars_and_density_extremes_remain_valid() {
    for layout in IndoorLayout::PROFILES {
        for density in [0.0, 1.0] {
            for seed in 0..32 {
                let scene = IndoorManifest::generate(seed, layout, density, 8)
                    .unwrap_or_else(|e| panic!("{layout:?}, density {density}, seed {seed}: {e}"));
                validate_layout(&scene).unwrap();
            }
        }
    }
    assert!(IndoorManifest::generate(0, IndoorLayout::Mixed, f32::NAN, 4).is_err());
    assert!(IndoorManifest::generate(0, IndoorLayout::Mixed, -0.1, 4).is_err());
    assert!(IndoorManifest::generate(0, IndoorLayout::Mixed, 0.5, 257).is_err());
}

#[test]
fn procedural_meshes_have_valid_topology_normals_and_tangents() {
    for seed in 0..12 {
        let scene = IndoorManifest::generate(seed, IndoorLayout::Mixed, 0.65, 2).unwrap();
        let stats = validate_geometry(&scene).unwrap();
        assert!(stats.triangles > 10000);
        assert!(stats.semantic_triangles.len() >= 12);
        for object in &scene.objects {
            let assembly = super::objects::build_object(object);
            let (lo, hi) = assembly.bounds();
            assert!(
                lo.y >= -0.001,
                "floating support origin: {:?} {lo:?}",
                object.kind
            );
            assert!(
                hi.y <= object.size.y + 0.05,
                "height envelope: {:?} {hi:?}",
                object.kind
            );
            assert!(
                lo.x >= -object.size.x * 0.5 - 0.05 && hi.x <= object.size.x * 0.5 + 0.05,
                "width envelope: {:?} {lo:?} {hi:?}",
                object.kind
            );
            assert!(
                lo.z >= -object.size.z * 0.5 - 0.06 && hi.z <= object.size.z * 0.5 + 0.06,
                "depth envelope: {:?} {lo:?} {hi:?}",
                object.kind
            );
            for (_, geometry) in assembly.parts {
                let mesh = geometry.into_mesh();
                assert!(mesh.attribute(Mesh::ATTRIBUTE_TANGENT).is_some());
            }
        }
    }
}

#[test]
fn annotation_gate_detects_a_wrong_frame_or_camera() {
    let n = 64u32;
    let mut view = crate::sample::View {
        world_from_view: Mat4::IDENTITY.to_cols_array_2d(),
        fovy: std::f32::consts::FRAC_PI_2,
        ..default()
    };
    for y in 0..n {
        for x in 0..n {
            let point = Vec3::new(
                ((x as f32 + 0.5) / n as f32 * 2.0 - 1.0) * 2.0,
                (1.0 - (y as f32 + 0.5) / n as f32 * 2.0) * 2.0,
                -2.0,
            );
            let encoded = (point + Vec3::splat(4.0)) / 8.0;
            view.position.extend(bytemuck::cast_slice(&[
                encoded.x, encoded.y, encoded.z, 1.0,
            ]));
            view.depth
                .extend(bytemuck::cast_slice(&[2.0_f32, 2.0, 2.0, 1.0]));
            view.normal
                .extend(bytemuck::cast_slice(&[0.5_f32, 0.5, 1.0, 1.0]));
        }
    }
    let aabb = [[-4.0; 3], [4.0; 3]];
    validate_annotations(&view, aabb, n, n).unwrap();
    view.world_from_view[3][0] += 1.0;
    assert!(validate_annotations(&view, aabb, n, n).is_err());
}

#[test]
fn single_camera_stream_still_covers_camera_heights() {
    let report = audit(0, 128, 1, 0.5);
    assert!(report.camera_height_range[0] < 0.85 && report.camera_height_range[1] > 2.1);
}

#[test]
fn continuous_camera_sweeps_detect_thin_obstacles() {
    let lo = Vec3::new(-0.001, -1.0, -1.0);
    let hi = Vec3::new(0.001, 1.0, 1.0);
    assert!(segment_hits_box(-Vec3::X, Vec3::X, lo, hi));
    assert!(segment_hits_box(Vec3::ZERO, Vec3::ZERO, lo, hi));
    assert!(!segment_hits_box(
        Vec3::Y * 2.0,
        Vec3::X + Vec3::Y * 2.0,
        lo,
        hi
    ));
}

#[test]
fn metric_uvs_do_not_stretch_with_box_size() {
    let mut geometry = Geometry::default();
    geometry.cuboid(Vec3::new(3.0, 0.1, 1.0), 0.0, Transform::IDENTITY);
    let max_u = geometry.uvs.iter().map(|uv| uv[0]).fold(0.0, f32::max);
    assert_eq!(max_u, 3.0);
}

#[test]
fn tabletop_wood_grain_follows_the_long_axis() {
    use super::materials::Surface;
    let scene = IndoorManifest::generate(0, IndoorLayout::Conference, 0.65, 0).unwrap();
    let mut object = scene
        .objects
        .iter()
        .find(|o| matches!(o.kind, ObjectKind::Table | ObjectKind::Desk))
        .unwrap()
        .clone();
    while super::objects::tables::parameters(&object).top_surface != Surface::Wood {
        object.seed = object.seed.wrapping_add(1);
    }
    for size in [Vec3::new(1.5, 0.75, 3.5), Vec3::new(1.5, 0.75, 0.7)] {
        object.size = size;
        let assembly = super::objects::build_object(&object);
        let top = assembly
            .parts
            .iter()
            .find(|((surface, label), _)| {
                *surface == Surface::Wood
                    && super::objects::part_label(label) == object.kind.class_name()
            })
            .unwrap()
            .1;
        let is_top = |index: usize| top.normals[index][1] > 0.99;
        let top_indices: Vec<_> = (0..top.positions.len()).filter(|&i| is_top(i)).collect();
        let along = if size.x > size.z { 0 } else { 2 };
        let across = 2 - along;
        let i = top_indices[0];
        for j in top_indices {
            assert!(
                ((top.uvs[j][0] - top.uvs[i][0])
                    - (top.positions[j][across] - top.positions[i][across]))
                    .abs()
                    < 1e-5,
                "grain crosses the long edge"
            );
            let sign = if size.x > size.z { -1. } else { 1. };
            assert!(
                ((top.uvs[j][1] - top.uvs[i][1])
                    - sign * (top.positions[j][along] - top.positions[i][along]))
                    .abs()
                    < 1e-5,
                "wood UVs lost physical scale"
            );
        }
    }
}

#[test]
fn floor_furniture_has_real_geometry_at_the_contact_plane() {
    for seed in 0..24 {
        let scene = IndoorManifest::generate(seed, IndoorLayout::Mixed, 1.0, 0).unwrap();
        for object in scene.objects.iter().filter(|o| o.solid) {
            let (lo, _) = super::objects::build_object(object).bounds();
            assert!(
                (-0.001..=0.002).contains(&lo.y),
                "seed {seed}: {:?}/{} floats or sinks: lowest point {} m",
                object.kind,
                object.variant,
                lo.y
            );
        }
    }
}

#[test]
fn neighboring_room_furniture_and_conference_seats_fit() {
    for seed in 0..512 {
        let scene = IndoorManifest::generate(seed, IndoorLayout::Conference, 1.0, 0).unwrap();
        let neighbor: Vec<_> = scene
            .objects
            .iter()
            .filter(|o| o.neighbor && o.solid)
            .collect();
        for (i, object) in neighbor.iter().enumerate() {
            let (lo, hi) = object.bounds();
            assert!(lo.x >= -scene.room_size.x * 0.5 + 0.299);
            assert!(hi.x <= scene.room_size.x * 0.5 - 0.299);
            assert!(lo.z >= scene.room_size.z * 0.5 + 0.299);
            assert!(hi.z <= scene.room_size.z * 0.5 + NEIGHBOR_DEPTH - 0.299);
            for other in neighbor.iter().skip(i + 1) {
                let (a, b) = other.bounds();
                assert!(
                    !(lo.x < b.x && hi.x > a.x && lo.z < b.z && hi.z > a.z),
                    "seed {seed}: neighboring {:?} overlaps {:?}",
                    object.kind,
                    other.kind
                );
            }
        }
        // Programs sample occupancy, dimensions and multiple work zones; a fixed
        // legacy table-length -> eight seats formula is no longer the contract.
        validate_layout(&scene).unwrap();
        let table = scene
            .objects
            .iter()
            .find(|o| matches!(o.kind, ObjectKind::Table | ObjectKind::Desk) && !o.neighbor)
            .unwrap();
        for prop in scene.objects.iter().filter(|o| o.support.is_some()) {
            assert!(scene.prop_clear(prop, &scene.objects[prop.support.unwrap()], 0.0));
        }
        for laptop in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Laptop && o.support == Some(table.id))
        {
            let facing = Quat::from_rotation_y(laptop.yaw) * Vec3::Z;
            let Some(seat_id) = laptop.interaction_target else {
                continue;
            };
            let seat = &scene.objects[seat_id];
            assert_eq!(seat.interaction_target, laptop.support);
            let direction = (seat.position - laptop.position).with_y(0.0).normalize();
            assert!(
                facing.dot(direction) > 0.95,
                "screen does not face its seated user"
            );
        }
    }
}

#[test]
fn loaded_manifests_reject_pillars_neighbor_overlaps_and_bad_supports() {
    let baseline = IndoorManifest::generate(3, IndoorLayout::Conference, 0.65, 0).unwrap();
    let mut scene = baseline.clone();
    let object = scene
        .objects
        .iter_mut()
        .find(|o| o.kind == ObjectKind::Plant && !o.neighbor)
        .unwrap();
    scene.column_width = 0.42;
    object.size = Vec3::new(0.08, 1.0, 0.08);
    object.yaw = 0.0;
    object.position = Vec3::new(
        -scene.room_size.x * 0.5 + 0.35,
        0.0,
        -scene.room_size.z * 0.5 + 0.35,
    );
    assert!(
        validate_layout(&scene).is_err(),
        "loaded furniture intersects a pillar"
    );
    let mut scene = baseline.clone();
    let ids: Vec<_> = scene
        .objects
        .iter()
        .filter(|o| o.solid && o.neighbor)
        .map(|o| o.id)
        .collect();
    scene.objects[ids[1]].position = scene.objects[ids[0]].position;
    assert!(
        validate_layout(&scene).is_err(),
        "loaded neighboring furniture overlaps"
    );
    let mut scene = baseline.clone();
    scene.objects[ids[0]].position.z = scene.room_size.z * 0.5 + NEIGHBOR_DEPTH;
    assert!(
        validate_layout(&scene).is_err(),
        "loaded neighboring furniture leaves the room"
    );
    let mut scene = baseline;
    let prop = scene
        .objects
        .iter_mut()
        .find(|o| o.support.is_some())
        .unwrap();
    prop.neighbor = !prop.neighbor;
    assert!(
        validate_layout(&scene).is_err(),
        "support room flag mismatch went undetected"
    );
}

#[test]
fn trajectory_rejects_supported_props_and_midpath_view_obstruction() {
    let mut scene = IndoorManifest::generate(0, IndoorLayout::Conference, 0.5, 0).unwrap();
    scene.humans.clear();
    scene.program = None;
    scene.floor_plan = super::floorplan::FloorPlan::OpenHall;
    let mut obstacle = scene.objects[0].clone();
    obstacle.id = 0;
    obstacle.kind = ObjectKind::Monitor;
    obstacle.position = Vec3::new(0.0, 0.75, 0.0);
    obstacle.size = Vec3::new(0.4, 0.5, 0.1);
    obstacle.solid = false;
    obstacle.support = Some(1);
    scene.objects = vec![obstacle];
    assert!(
        !scene.camera_clear(Vec3::new(0.0, 1.2, 0.0)),
        "supported monitors must block cameras"
    );

    scene.objects[0].position = Vec3::new(0.0, 1.05, -0.75);
    scene.objects[0].size = Vec3::new(0.05, 0.3, 0.05);
    let start = Vec3::new(-0.6, 1.2, 0.0);
    let end = Vec3::new(0.6, 1.2, 0.0);
    let target = Vec3::new(0.0, 1.2, -3.0);
    assert!(scene.camera_path_clear(start, end));
    assert!(scene.camera_view_clear(start, target));
    assert!(scene.camera_view_clear(end, target));
    assert!(!scene.camera_trajectory_clear(start, end, target));
}

#[test]
fn trajectory_validation_matches_runtime_poses_and_covers_metric_baselines() {
    let mut range = [f32::INFINITY, 0.0_f32];
    for seed in 0..96 {
        let scene = IndoorManifest::generate(seed, IndoorLayout::Mixed, 0.8, 4).unwrap();
        for camera in &scene.cameras {
            let length = camera.start.distance(camera.end);
            range[0] = range[0].min(length);
            range[1] = range[1].max(length);
            let mut runtime = camera.runtime_trajectory();
            for step in 0..=16 {
                let t = step as f32 / 16.0;
                let actual = runtime.sample(t);
                let checked = camera.transform_at(t);
                assert!(actual.translation.distance(checked.translation) < 1e-6);
                assert!(actual.rotation.dot(checked.rotation).abs() > 1.0 - 1e-6);
                assert!(scene.camera_clear(actual.translation));
                assert!(scene.camera_view_clear(
                    actual.translation,
                    actual.translation + actual.rotation * Vec3::NEG_Z * 2.0
                ));
                assert!(
                    (actual.rotation * Vec3::X).y.abs() < 0.14001,
                    "camera roll exceeds sampled handheld bounds"
                );
            }
        }
    }
    assert!(
        range[0] < 0.21 && range[1] > 1.1,
        "trajectory baseline distribution collapsed: {range:?}"
    );
}

#[test]
fn books_expose_the_paper_edges_between_the_covers() {
    use super::materials::Surface;
    let scene = IndoorManifest::generate(0, IndoorLayout::Lounge, 1.0, 0).unwrap();
    for object in scene
        .objects
        .iter()
        .filter(|o| matches!(o.kind, ObjectKind::Books | ObjectKind::Notebook))
    {
        let assembly = super::objects::build_object(object);
        let height = object.size.y
            / if object.kind == ObjectKind::Books {
                3.0
            } else {
                1.0
            };
        let origin = Vec3::new(object.size.x, height * 0.5, 0.0);
        let direction = -Vec3::X;
        let mut hits = Vec::new();
        for ((surface, _), geometry) in assembly.parts {
            for tri in geometry.indices.as_chunks::<3>().0.iter() {
                let a = Vec3::from_array(geometry.positions[tri[0] as usize]);
                let b = Vec3::from_array(geometry.positions[tri[1] as usize]);
                let c = Vec3::from_array(geometry.positions[tri[2] as usize]);
                let edge1 = b - a;
                let edge2 = c - a;
                let p = direction.cross(edge2);
                let determinant = edge1.dot(p);
                if determinant.abs() < 1e-9 {
                    continue;
                }
                let t = origin - a;
                let u = t.dot(p) / determinant;
                let q = t.cross(edge1);
                let v = direction.dot(q) / determinant;
                let distance = edge2.dot(q) / determinant;
                if u >= 0.0 && v >= 0.0 && u + v <= 1.0 && distance > 0.0 {
                    hits.push((distance, surface));
                }
            }
        }
        hits.sort_by(|a, b| a.0.total_cmp(&b.0));
        assert_eq!(
            hits[0].1,
            Surface::Paper,
            "page edges hidden inside solid book cover"
        );
    }
}

#[test]
#[ignore = "bounded 40,000-scene distribution qualification; run explicitly for releases"]
fn broad_density_sweep_is_valid() {
    for density in [0.0, 0.35, 0.65, 1.0] {
        let report = audit(0, 10_000, 4, density);
        assert!(
            report.invalid_seeds.is_empty(),
            "density {density}: {:?}",
            report.invalid_seeds
        );
        assert_eq!(report.layout_counts.values().sum::<usize>(), 10_000);
        println!(
            "density={density}: 10000 seeds, 40000 cameras, zero invalid scenes; layouts={:?}",
            report.layout_counts
        );
    }
}

#[test]
fn dataset_metrics_preserve_denominators_heatmap_mass_and_camera_calibration() {
    use crate::camera::PerspectiveSampler;
    use bevy::camera::CameraProjection;
    let directory =
        std::env::temp_dir().join(format!("zeroverse-indoor-metrics-{}", std::process::id()));
    let scenes = 40;
    let cameras = 3;
    let (width, height) = (641, 479);
    let report = super::metrics::export_metrics(
        0,
        scenes,
        cameras,
        0.65,
        IndoorLayout::Mixed,
        width,
        height,
        &directory,
    )
    .unwrap();
    assert_eq!(report.object_counts_per_scene.len(), 74);
    for counts in report.object_counts_per_scene.values() {
        assert_eq!(
            counts.values().sum::<usize>(),
            scenes,
            "absent kinds must count as zero scenes"
        );
    }
    assert_eq!(report.object_counts_per_scene["neighbor/Sofa"][&0], scenes);
    assert_eq!(report.object_counts_per_scene["neighbor/Chair"][&1], scenes);
    for (key, counts) in &report.object_counts_by_layout {
        let layout = key.split('/').next().unwrap();
        assert_eq!(
            counts.values().sum::<usize>(),
            report.categories["layout"][layout],
            "by-layout count denominator"
        );
    }
    for (key, cells) in &report.placement_heatmaps {
        assert_eq!(cells.len(), report.heatmap_grid_size.pow(2));
        let expected = match key.as_str() {
            "camera_start" | "camera_end" => scenes * cameras,
            "camera_path" => scenes * cameras * 33,
            _ => report.object_counts_per_scene[key]
                .iter()
                .map(|(count, scenes)| count * scenes)
                .sum(),
        };
        assert_eq!(
            cells.iter().sum::<usize>(),
            expected,
            "lost heatmap mass for {key}"
        );
    }
    assert_eq!(report.numeric["room_width_m"].count, scenes);
    assert_eq!(report.numeric["fy_pixels"].count, scenes * cameras);
    for distribution in report.numeric.values() {
        assert_eq!(
            distribution.bin_counts.iter().sum::<usize>(),
            distribution.count
        );
    }
    let csv = std::fs::read_to_string(directory.join("cameras.csv")).unwrap();
    assert_eq!(csv.lines().count(), scenes * cameras + 1);
    for line in csv.lines().skip(1) {
        let fields: Vec<_> = line.split(',').collect();
        assert_eq!(fields.len(), 23);
        let number = |i: usize| fields[i].parse::<f32>().unwrap();
        let mut projection = PerspectiveSampler::exact(number(12)).sample();
        projection.update(width as f32, height as f32);
        let clip = projection.get_clip_from_view();
        let fx = clip.x_axis.x * width as f32 * 0.5;
        let fy = clip.y_axis.y * height as f32 * 0.5;
        assert!((number(14) - fx).abs() < 0.001);
        assert!((number(15) - fy).abs() < 0.001);
        assert_eq!(number(16), width as f32 * 0.5);
        assert_eq!(number(17), height as f32 * 0.5);
        assert_eq!(number(18), projection.near);
        assert_eq!(number(19), projection.far);
    }
    for (seeds, cameras, density, width, height) in [
        (0, 3, 0.65, 641, 479),
        (1, 257, 0.65, 641, 479),
        (1, 3, f32::NAN, 641, 479),
        (1, 3, 1.1, 641, 479),
        (1, 3, 0.65, 0, 479),
        (1, 3, 0.65, 641, 0),
    ] {
        assert!(super::metrics::export_metrics(
            0,
            seeds,
            cameras,
            density,
            IndoorLayout::Mixed,
            width,
            height,
            &directory
        )
        .is_err());
    }
    std::fs::remove_dir_all(directory).unwrap();
}

fn point_triangle_distance_squared(point: Vec3, a: Vec3, b: Vec3, c: Vec3) -> f32 {
    let ab = b - a;
    let ac = c - a;
    let n = ab.cross(ac);
    let projected = point - n * ((point - a).dot(n) / n.length_squared());
    let ap = projected - a;
    let denominator = ab.dot(ab) * ac.dot(ac) - ab.dot(ac).powi(2);
    if denominator > 1e-16 {
        let u = (ac.dot(ac) * ap.dot(ab) - ab.dot(ac) * ap.dot(ac)) / denominator;
        let v = (ab.dot(ab) * ap.dot(ac) - ab.dot(ac) * ap.dot(ab)) / denominator;
        if u >= 0.0 && v >= 0.0 && u + v <= 1.0 {
            return point.distance_squared(projected);
        }
    }
    [(a, b), (b, c), (c, a)]
        .into_iter()
        .map(|(start, end)| {
            let delta = end - start;
            let t = ((point - start).dot(delta) / delta.length_squared()).clamp(0.0, 1.0);
            point.distance_squared(start + delta * t)
        })
        .fold(f32::INFINITY, f32::min)
}

#[test]
fn actual_architecture_preserves_door_opening_and_camera_clearance() {
    for seed in 0..32 {
        let scene = IndoorManifest::generate(seed, IndoorLayout::Mixed, 1.0, 4).unwrap();
        let architecture = super::architecture::architecture(&scene);
        let triangles: Vec<_> = architecture
            .parts
            .values()
            .flat_map(|geometry| {
                geometry.indices.as_chunks::<3>().0.iter().map(|indices| {
                    let a = Vec3::from_array(geometry.positions[indices[0] as usize]);
                    let b = Vec3::from_array(geometry.positions[indices[1] as usize]);
                    let c = Vec3::from_array(geometry.positions[indices[2] as usize]);
                    (a, b, c, a.min(b).min(c), a.max(b).max(c))
                })
            })
            .collect();
        let clear = |point: Vec3, radius: f32| {
            triangles.iter().all(|&(a, b, c, lo, hi)| {
                let nearest = point.clamp(lo, hi);
                nearest.distance_squared(point) >= radius * radius
                    || point_triangle_distance_squared(point, a, b, c) >= radius * radius
            })
        };
        for camera in &scene.cameras {
            for step in 0..=16 {
                let position = camera.transform_at(step as f32 / 16.0).translation;
                let nearest = || {
                    architecture
                        .parts
                        .iter()
                        .flat_map(|((surface, label), g)| {
                            g.indices.as_chunks::<3>().0.iter().map(move |t| {
                                let p =
                                    [t[0], t[1], t[2]].map(|i| Vec3::from(g.positions[i as usize]));
                                (
                                    point_triangle_distance_squared(position, p[0], p[1], p[2]),
                                    surface,
                                    label,
                                )
                            })
                        })
                        .min_by(|a, b| a.0.total_cmp(&b.0))
                };
                assert!(
                    clear(position, CAMERA_CLEARANCE - 0.001),
                    "seed {seed}: actual architectural geometry violates camera clearance at {position:?}: nearest {:?}", nearest()
                );
            }
        }
        for x in [-0.4, 0.0, 0.4] {
            for y in [0.30, 1.1, 2.1] {
                for z in [-0.25, 0.0, 0.25, 0.9] {
                    let point = Vec3::new(scene.door_x + x, y, scene.room_size.z * 0.5 + z);
                    assert!(
                        clear(point, 0.05),
                        "seed {seed}: obstructed actual doorway at {point:?}"
                    );
                }
            }
        }
        let luminaire = super::architecture::fixture_positions(&scene)[0];
        let unsafe_position = luminaire - Vec3::Y * 0.15;
        assert!(!clear(unsafe_position, CAMERA_CLEARANCE));
        assert!(
            !scene.camera_clear(unsafe_position),
            "nominal camera checks missed a suspended lamp"
        );
    }
}

#[test]
fn people_are_deterministic_diverse_supported_and_inside_collision_envelopes() {
    use super::humans::{build_human, HumanPoseKind, HUMAN_BONE_NAMES};
    use std::collections::BTreeSet;
    let mut poses = BTreeSet::new();
    let mut outfits = BTreeSet::new();
    let mut skin = BTreeSet::new();
    let mut hair = BTreeSet::new();
    let mut counts = [0_usize; 3];
    let mut neighbor_count = 0;
    for seed in 0..128 {
        let empty =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 2, 0.0).unwrap();
        assert!(empty.humans.is_empty());
        for (index, density) in [0.0, 0.25, 1.0].into_iter().enumerate() {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 2, density)
                    .unwrap();
            assert_eq!(
                scene.objects, empty.objects,
                "human occupancy must not perturb furnishings"
            );
            validate_layout(&scene).unwrap();
            counts[index] += scene.humans.len();
            for human in &scene.humans {
                assert_eq!(human.joints.len(), HUMAN_BONE_NAMES.len());
                poses.insert(format!("{:?}", human.pose));
                outfits.insert(format!("{:?}", human.outfit));
                skin.insert(human.skin_tone);
                hair.insert(human.hairstyle);
                neighbor_count += usize::from(human.neighbor);
                assert_eq!(human.pose.seated(), human.chair.is_some());
                let (lo, hi) = build_human(human).bounds();
                assert!(
                    lo.cmpge(human.bounds_min - Vec3::splat(0.001)).all()
                        && hi.cmple(human.bounds_max + Vec3::splat(0.001)).all(),
                    "human actual geometry exceeds collision envelope seed={seed} pose={:?} actual={lo:?} {hi:?} planned={:?} {:?}",
                    human.pose,
                    human.bounds_min,
                    human.bounds_max
                );
                assert!(
                    lo.y >= -0.001 && lo.y <= 0.02,
                    "human shoes must contact floor: {lo:?}"
                );
            }
        }
    }
    assert!(
        counts[1] > 128 && counts[2] > counts[1],
        "human population collapsed: {counts:?}"
    );
    assert_eq!(poses.len(), 8, "{poses:?}");
    assert_eq!(outfits.len(), 6);
    assert_eq!(skin.len(), 8);
    assert_eq!(hair.len(), super::humans::hair::HairStyle::ALL.len());
    assert!(neighbor_count > 50, "neighbor chair occupancy disappeared");
    assert!(poses.contains(&format!("{:?}", HumanPoseKind::SeatedWorking)));
    eprintln!(
        "128-seed human counts at densities 0/.25/1: {counts:?}; neighbor instances={neighbor_count}"
    );
    let a = IndoorManifest::generate_with_humans(78, IndoorLayout::Mixed, 0.65, 2, 1.0).unwrap();
    let b = IndoorManifest::generate_with_humans(78, IndoorLayout::Mixed, 0.65, 8, 1.0).unwrap();
    assert_eq!(a.humans, b.humans);
    assert_eq!(a.cameras, b.cameras[..2]);
    assert!(
        IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.5, 2, f32::NAN).is_err()
    );
}

#[test]
#[ignore = "bounded 4096-scene full-occupancy geometry qualification"]
fn broad_full_human_occupancy_preserves_geometry_and_layout() {
    use super::humans::build_human;
    use std::collections::{BTreeMap, BTreeSet};
    let mut poses = BTreeSet::new();
    let mut outfits = BTreeSet::new();
    let mut skin = BTreeSet::new();
    let mut hair = BTreeSet::new();
    let mut total_vertices = 0_usize;
    let mut total_triangles = 0_usize;
    for density in [0.0, 0.35, 0.65, 1.0] {
        let mut layouts = BTreeMap::<String, usize>::new();
        let mut people = 0;
        let mut neighbors = 0;
        let mut min_people = usize::MAX;
        let mut max_people = 0;
        for seed in 0..1024 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, density, 4, 1.0)
                    .unwrap();
            validate_layout(&scene).unwrap_or_else(|error| {
                panic!("full human occupancy seed={seed} furniture_density={density}: {error}")
            });
            *layouts.entry(format!("{:?}", scene.layout)).or_default() += 1;
            people += scene.humans.len();
            min_people = min_people.min(scene.humans.len());
            max_people = max_people.max(scene.humans.len());
            for human in &scene.humans {
                poses.insert(format!("{:?}", human.pose));
                outfits.insert(format!("{:?}", human.outfit));
                skin.insert(human.skin_tone);
                hair.insert(human.hairstyle);
                neighbors += usize::from(human.neighbor);
                let assembly = build_human(human);
                let (lo, hi) = assembly.bounds();
                assert!(
                    lo.cmpge(human.bounds_min - Vec3::splat(0.001)).all()
                        && hi.cmple(human.bounds_max + Vec3::splat(0.001)).all(),
                    "human geometry outside placement envelope seed={seed} id={}",
                    human.id
                );
                assert!(
                    (-0.001..=0.02).contains(&lo.y),
                    "human floor contact seed={seed} id={} min={lo:?}",
                    human.id
                );
                for geometry in assembly.parts.values() {
                    assert_eq!(geometry.positions.len(), geometry.normals.len());
                    assert_eq!(geometry.positions.len(), geometry.uvs.len());
                    assert_eq!(geometry.indices.len() % 3, 0);
                    assert!(geometry
                        .indices
                        .iter()
                        .all(|i| (*i as usize) < geometry.positions.len()));
                    assert!(geometry
                        .positions
                        .iter()
                        .all(|p| Vec3::from_array(*p).is_finite()));
                    assert!(geometry.normals.iter().all(|n| {
                        let n = Vec3::from_array(*n);
                        n.is_finite() && (n.length_squared() - 1.0).abs() < 0.001
                    }));
                    total_vertices += geometry.positions.len();
                    total_triangles += geometry.indices.len() / 3;
                }
            }
        }
        assert_eq!(layouts.len(), IndoorLayout::PROFILES.len());
        assert!(
            people > 4096 && neighbors > 128,
            "full occupancy collapsed: people={people} neighboring={neighbors}"
        );
        println!(
            "full human density=1 furniture_density={density}: scenes=1024 cameras=4096 people={people} neighbor_people={neighbors} per_scene_min={min_people} per_scene_max={max_people} layouts={layouts:?}"
        );
    }
    assert_eq!(poses.len(), 8);
    assert_eq!(outfits.len(), 3);
    assert_eq!(skin.len(), 8);
    assert_eq!(hair.len(), 8);
    println!(
        "full occupancy geometry: vertices={total_vertices} triangles={total_triangles}; poses={poses:?}, outfits={outfits:?}, skin_tones={skin:?}, hairstyles={hair:?}"
    );
}

#[test]
fn furniture_programs_have_bounded_geometry_and_varied_parameters() {
    use std::collections::BTreeSet;
    let mut plans = BTreeSet::new();
    let mut designs = BTreeSet::new();
    let mut chairs = BTreeSet::new();
    let mut laptops = BTreeSet::new();
    let mut angles = Vec::new();
    for seed in 0..128 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.8, 2, 0.0).unwrap();
        validate_layout(&scene).unwrap();
        let program = scene.program.as_ref().unwrap();
        plans.insert(program.partitions.len());
        designs.insert((
            (program.fixture_size.x * 10.0) as u32,
            (program.fixture_size.y * 10.0) as u32,
        ));
        for o in scene
            .objects
            .iter()
            .filter(|o| matches!(o.kind, ObjectKind::Chair | ObjectKind::Laptop))
        {
            if o.kind == ObjectKind::Chair {
                chairs.insert(o.variant);
            } else {
                laptops.insert(o.variant);
                angles.push(super::objects::computers::lid_angle(o));
            }
            let (lo, hi) = super::objects::build_object(o).bounds();
            assert!(
                lo.x >= -o.size.x * 0.5 - 0.005 && hi.x <= o.size.x * 0.5 + 0.005,
                "width envelope seed={seed} kind={:?} variant={} {lo:?} {hi:?}",
                o.kind,
                o.variant
            );
            assert!(
                lo.z >= -o.size.z * 0.5 - 0.005 && hi.z <= o.size.z * 0.5 + 0.005,
                "depth envelope seed={seed} kind={:?} variant={} {lo:?} {hi:?}",
                o.kind,
                o.variant
            );
            assert!(lo.y >= -0.001 && hi.y <= o.size.y + 0.025);
        }
    }
    assert!(plans.len() >= 4);
    assert!(designs.len() >= 24);
    assert_eq!(chairs.len(), super::objects::chairs::FAMILIES as usize);
    assert_eq!(laptops.len(), 4);
    assert!(angles.iter().copied().fold(f32::INFINITY, f32::min) < 1.6);
    assert!(angles.iter().copied().fold(f32::NEG_INFINITY, f32::max) > 2.15);
}

#[test]
fn dressed_staged_humans_have_finite_geometry_and_material_parts() {
    for seed in [0, 1, 13, 34] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.35, 0, 0.7).unwrap();
        crate::human_motion::planning::prepare_scene(
            &mut scene,
            &crate::human_motion::HumanMotionConfig {
                fraction: 1.0,
                locomotion_fraction: 1.0,
                max_actors: 16,
                ..default()
            },
        )
        .unwrap();
        for h in &scene.humans {
            let assembly = super::humans::build_human(h);
            for (surface, g) in assembly.parts {
                for (index, (p, n)) in g.positions.iter().zip(&g.normals).enumerate() {
                    assert!(Vec3::from_array(*p).is_finite() && Vec3::from_array(*n).is_finite()
                        && Vec3::from_array(*n).length_squared() > 0.99,
                        "seed={seed}, human={}, surface={surface:?}, vertex={index}, p={p:?}, n={n:?}", h.id);
                }
            }
        }
    }
}

#[cfg(feature = "viewer")]
#[test]
fn editor_intrinsics_survive_room_regeneration() {
    use super::position_editor;
    use crate::camera::{EditorCameraMarker, ProcessedEditorCameraMarker};
    use crate::{app::BevyZeroverseConfig, scene::ZeroverseSceneType};
    use bevy::ecs::system::RunSystemOnce;
    let mut app = App::new();
    app.insert_resource(BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        ..default()
    });
    let camera = app
        .world_mut()
        .spawn((
            EditorCameraMarker::default(),
            ProcessedEditorCameraMarker,
            bevy_panorbit_camera::PanOrbitCamera::default(),
            Transform::IDENTITY,
            Projection::Perspective(PerspectiveProjection {
                fov: 0.92,
                near: 0.13,
                far: 131.0,
                aspect_ratio: 1.71,
                ..default()
            }),
        ))
        .id();
    for seed in [0, 13, 42] {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.4, 1, 0.0).unwrap();
        app.insert_resource(scene);
        app.world_mut().run_system_once(position_editor).unwrap();
        let Projection::Perspective(p) = app.world().get::<Projection>(camera).unwrap() else {
            panic!()
        };
        assert_eq!(
            (p.fov, p.near, p.far, p.aspect_ratio),
            (0.92, 0.13, 131.0, 1.71)
        );
        assert_ne!(
            app.world().get::<Transform>(camera).unwrap().translation,
            Vec3::ZERO
        );
    }
}

#[test]
fn backless_stools_survive_placement_and_have_no_back_geometry() {
    let mut seats = 0;
    let mut stools = 0;
    let mut rooms = 0;
    for seed in 0..128 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.0).unwrap();
        let mut found = false;
        for chair in scene
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Chair && !o.neighbor)
        {
            seats += 1;
            if super::objects::chairs::is_backless(chair) {
                stools += 1;
                found = true;
                let (_, hi) = super::objects::build_object(chair).bounds();
                assert!(hi.y <= 0.481, "stool acquired a back: {hi:?}");
                assert!(hi.y >= 0.469, "missing stool seat: {hi:?}");
            }
        }
        rooms += usize::from(found);
    }
    println!("128 primary rooms: {stools}/{seats} backless seats in {rooms} rooms");
    assert!(stools as f32 / seats as f32 > 0.10);
    assert!(rooms > 40);
}
