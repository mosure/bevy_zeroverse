use super::*;

#[test]
fn standalone_selected_materials_include_camera_ready_cubemaps() {
    use super::super::{layout::ObjectKind, objects};
    use bevy::render::render_resource::TextureViewDimension;
    let mut scene =
        IndoorManifest::generate_with_humans(0, Default::default(), 0.5, 0, 0.).unwrap();
    let o = objects::audit::fixture(ObjectKind::Sponge, 0);
    let parts: Vec<_> = objects::build_object(&o).parts.into_keys().collect();
    scene.objects = vec![o];
    scene.humans.clear();
    let (mut images, mut materials) = (Assets::default(), Assets::default());
    let set = IndoorMaterials::build_for_parts(
        &scene,
        &mut images,
        &mut materials,
        parts
            .iter()
            .map(|(surface, label)| (*surface, label.as_str())),
    );
    for handle in [&set.environment.diffuse_map, &set.environment.specular_map] {
        let image = images
            .get(handle)
            .expect("standalone environment map missing");
        assert_eq!(image.texture_descriptor.size.depth_or_array_layers, 6);
        assert_eq!(
            image
                .texture_view_descriptor
                .as_ref()
                .and_then(|d| d.dimension),
            Some(TextureViewDimension::Cube)
        );
        assert!(image.data.as_ref().is_some_and(|data| !data.is_empty()));
    }
}

#[test]
fn substrate_fields_consume_the_upper_seed_bits() {
    for (x, y) in [(3, 7), (79, 19), (12, 53)] {
        assert_ne!(hash(x, y, 73), hash(x, y, 73 + (1u64 << 32)));
    }
}

#[test]
fn leaf_maps_clamp_at_atlas_edges_and_bind_absolute_color() {
    let scene = IndoorManifest::generate_with_humans(
        31,
        super::super::layout::IndoorLayout::Mixed,
        0.65,
        0,
        0.,
    )
    .unwrap();
    let (mut images, mut materials) = (Assets::default(), Assets::default());
    let set = IndoorMaterials::build(&scene, &mut images, &mut materials);
    for s in [Surface::Leaf, Surface::LeafLight, Surface::LeafVariegated] {
        let m = materials.get(&set.get(s)).unwrap();
        assert_eq!(m.base_color, Color::WHITE);
        assert_eq!(m.uv_transform, bevy::math::Affine2::IDENTITY);
        assert!(m.diffuse_transmission > 0. && m.diffuse_transmission < 0.5);
        for h in [
            &m.base_color_texture,
            &m.normal_map_texture,
            &m.metallic_roughness_texture,
        ] {
            let ImageSampler::Descriptor(d) = &images.get(h.as_ref().unwrap()).unwrap().sampler
            else {
                panic!("leaf sampler")
            };
            assert_eq!(d.address_mode_u, ImageAddressMode::ClampToEdge);
            assert_eq!(d.address_mode_v, ImageAddressMode::ClampToEdge);
        }
    }
}

#[test]
fn byte_color_lookup_preserves_every_original_transfer_value() {
    for (byte, &cached) in srgb8_table().iter().enumerate() {
        let original = (Vec3::splat(byte as f32) / 255.).map(srgb_to_linear).x;
        assert_eq!(cached.to_bits(), original.to_bits(), "byte {byte}");
    }
}

#[test]
fn used_finish_palette_preserves_materials_and_texture_pixels() {
    let scene = IndoorManifest::generate_with_humans(
        44,
        super::super::layout::IndoorLayout::Mixed,
        0.65,
        0,
        0.0,
    )
    .unwrap();
    let mut assemblies = vec![super::super::architecture::architecture(&scene)];
    assemblies.extend(
        scene
            .objects
            .iter()
            .map(super::super::objects::build_object),
    );
    let keys: std::collections::BTreeSet<_> = assemblies
        .iter()
        .flat_map(|a| a.parts.keys())
        .filter_map(|(surface, label)| {
            label
                .rsplit_once("#finish")
                .and_then(|(_, s)| s.split('#').next()?.parse::<usize>().ok())
                .map(|i| (*surface, i))
        })
        .collect();
    let selection = MaterialSelection {
        surfaces: assemblies
            .iter()
            .flat_map(|a| a.parts.keys().map(|(surface, _)| *surface))
            .collect(),
        finishes: keys.clone(),
        direct_surfaces: assemblies
            .iter()
            .flat_map(|a| a.parts.keys())
            .filter_map(|(surface, label)| {
                let variant = finish_slot(*surface, label)
                    .is_some_and(|slot| registered_finish(*surface, slot));
                (!variant).then_some(*surface)
            })
            .collect(),
    };
    let (mut all_images, mut all_materials) = (Assets::default(), Assets::default());
    let all = IndoorMaterials::build(&scene, &mut all_images, &mut all_materials);
    let (mut images, mut materials) = (Assets::default(), Assets::default());
    let used = IndoorMaterials::build_with_selection(
        &scene,
        super::super::IndoorQuality::Auto,
        &mut images,
        &mut materials,
        Some(&selection),
        false,
    );
    assert!(!used.variants.is_empty());
    assert!(used.variants.len() < all.variants.len());
    assert!(images.len() < all_images.len());
    for key in keys.into_iter().filter(|k| variants::supports(k.0)) {
        let a = all_materials.get(&all.variants[&key]).unwrap();
        let b = materials.get(&used.variants[&key]).unwrap();
        assert_eq!(a.base_color, b.base_color);
        assert_eq!(a.emissive, b.emissive);
        assert_eq!(a.perceptual_roughness, b.perceptual_roughness);
        assert_eq!(a.metallic, b.metallic);
        assert_eq!(a.uv_transform, b.uv_transform);
        assert_eq!(a.clearcoat, b.clearcoat);
        for (ah, bh) in [
            (&a.base_color_texture, &b.base_color_texture),
            (&a.emissive_texture, &b.emissive_texture),
            (&a.normal_map_texture, &b.normal_map_texture),
            (&a.clearcoat_normal_texture, &b.clearcoat_normal_texture),
            (&a.metallic_roughness_texture, &b.metallic_roughness_texture),
        ] {
            assert_eq!(ah.is_some(), bh.is_some());
            if let (Some(ah), Some(bh)) = (ah, bh) {
                let ai = all_images.get(ah).unwrap();
                let bi = images.get(bh).unwrap();
                assert_eq!(ai.texture_descriptor, bi.texture_descriptor);
                assert_eq!(ai.data, bi.data, "finish {key:?}");
            }
        }
    }
}

fn assert_same_material(
    mut a: StandardMaterial,
    mut b: StandardMaterial,
    all_images: &Assets<Image>,
    images: &Assets<Image>,
) {
    // Compare every PBR field after checking the full image contents and
    // clearing only allocator-specific handles, including packed AO reuse.
    macro_rules! image {
        ($field:ident) => {
            let ah = a.$field.take();
            let bh = b.$field.take();
            assert_eq!(ah.is_some(), bh.is_some(), stringify!($field));
            if let (Some(ah), Some(bh)) = (ah, bh) {
                let ai = all_images.get(&ah).unwrap();
                let bi = images.get(&bh).unwrap();
                assert_eq!(ai.texture_descriptor, bi.texture_descriptor);
                assert_eq!(ai.data, bi.data, stringify!($field));
                assert_eq!(format!("{:?}", ai.sampler), format!("{:?}", bi.sampler));
                assert_eq!(ai.asset_usage, bi.asset_usage);
            }
        };
    }
    image!(base_color_texture);
    image!(emissive_texture);
    image!(metallic_roughness_texture);
    image!(normal_map_texture);
    image!(clearcoat_normal_texture);
    image!(clearcoat_roughness_texture);
    image!(clearcoat_texture);
    image!(occlusion_texture);
    image!(depth_map);
    assert_eq!(format!("{a:?}"), format!("{b:?}"));
}

#[test]
fn geometry_selection_preserves_all_referenced_pbr_and_human_maps() {
    use super::super::{
        humans,
        layout::{IndoorLayout, ObjectKind},
        preparation::SceneGeometry,
    };
    for (seed, human_density, legacy, layout) in [
        (44, 0., false, IndoorLayout::Mixed),
        (207, 1., false, IndoorLayout::Mixed),
        (7, 0., true, IndoorLayout::Mixed),
        (43_084_482, 0., false, IndoorLayout::Mixed),
        (13, 0., false, IndoorLayout::Kitchen),
    ] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, layout, 0.65, 0, human_density).unwrap();
        if legacy {
            scene.program = None;
        }
        if layout.is_kitchen() {
            for kind in [ObjectKind::FruitBowl, ObjectKind::Bread] {
                let mut o = super::super::objects::audit::fixture(kind, 1);
                o.id = scene.objects.len();
                scene.objects.push(o);
            }
        }
        let geometry = SceneGeometry {
            architecture: super::super::architecture::architecture(&scene),
            objects: scene
                .objects
                .iter()
                .map(super::super::objects::build_object)
                .collect(),
            humans: scene.humans.iter().map(humans::build_human).collect(),
        };
        let selection = geometry.material_selection(&scene);
        let (mut all_images, mut all_materials) = (Assets::default(), Assets::default());
        let all = IndoorMaterials::build(&scene, &mut all_images, &mut all_materials);
        let (mut images, mut materials) = (Assets::default(), Assets::default());
        let used = IndoorMaterials::build_with_selection(
            &scene,
            super::super::IndoorQuality::Auto,
            &mut images,
            &mut materials,
            Some(&selection),
            false,
        );
        assert!(images.len() < all_images.len());
        assert!(selection.surfaces.len() < definitions(&scene).len());
        for assembly in std::iter::once(&geometry.architecture).chain(&geometry.objects) {
            for ((surface, label), part) in &assembly.parts {
                if part.indices.is_empty() {
                    continue;
                }
                assert!(selection.surfaces.contains(surface));
                assert_same_material(
                    all_materials
                        .get(&all.for_part(*surface, label))
                        .unwrap()
                        .clone(),
                    materials
                        .get(&used.for_part(*surface, label))
                        .unwrap()
                        .clone(),
                    &all_images,
                    &images,
                );
            }
        }
        if scene.humans.is_empty() {
            // No human-only template has been synthesized; aliases are
            // intentionally inaccessible from any scene geometry.
            assert_eq!(used.cloth, used.get(Surface::Fabric));
            assert_eq!(used.skin, used.cloth);
            assert_eq!(used.hair, used.cloth);
        } else {
            for (person, assembly) in scene.humans.iter().zip(&geometry.humans) {
                for &surface in assembly.parts.keys() {
                    assert_same_material(
                        humans::person_material(person, surface, &all, &all_materials),
                        humans::person_material(person, surface, &used, &materials),
                        &all_images,
                        &images,
                    );
                }
            }
        }
        // All surfaces retain valid handles for GI's stable surface indices.
        for surface in program::SURFACES {
            assert!(materials.get(&used.get(surface)).is_some());
            if !selection.needs_maps(&scene, surface) {
                let mat = materials.get(&used.get(surface)).unwrap();
                assert!(mat.base_color_texture.is_none());
                assert!(mat.normal_map_texture.is_none());
                assert!(mat.emissive_texture.is_none());
                if selection.omits_parent_maps(&scene, surface) {
                    assert_eq!(mat.perceptual_roughness, 1.0);
                    assert!(mat.metallic_roughness_texture.is_none());
                    assert!(mat.occlusion_texture.is_none());
                }
            }
        }
        assert_same_gi_part_rays(
            &scene,
            &geometry,
            &all,
            &used,
            &all_materials,
            &materials,
            &all_images,
            &images,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn assert_same_gi_part_rays(
    scene: &IndoorManifest,
    geometry: &super::super::preparation::SceneGeometry,
    all: &IndoorMaterials,
    used: &IndoorMaterials,
    all_materials: &Assets<StandardMaterial>,
    materials: &Assets<StandardMaterial>,
    all_images: &Assets<Image>,
    images: &Assets<Image>,
) {
    use super::super::gi::BakeScene;
    let complete = BakeScene::from_geometry(scene, all, all_materials, all_images, &[], geometry);
    let selected = BakeScene::from_geometry(scene, used, materials, images, &[], geometry);
    let check = |part: &super::super::geometry::Geometry, transform: Transform| {
        let triangles = part.indices.as_chunks::<3>().0;
        if triangles.is_empty() {
            return;
        }
        // Surface-origin rays exercise actual closest-hit diffuse/emission
        // sampling, including each referenced finish and the human proxy.
        for triangle in [
            triangles[0],
            triangles[triangles.len() / 2],
            triangles[triangles.len() - 1],
        ] {
            let [a, b, c] = triangle
                .map(|i| transform.transform_point(Vec3::from_array(part.positions[i as usize])));
            let normal = (b - a).cross(c - a).normalize_or_zero();
            if normal.length_squared() < 0.5 {
                continue;
            }
            let origin = (a + b + c) / 3. + normal * 0.0005;
            let (ad, ac) = complete.reflection_radiance(origin, -normal, 0.37);
            let (bd, bc) = selected.reflection_radiance(origin, -normal, 0.37);
            assert_eq!(ad.to_bits(), bd.to_bits());
            assert_eq!(
                ac.to_array().map(f32::to_bits),
                bc.to_array().map(f32::to_bits),
                "seed {} GI ray {:?}",
                scene.seed,
                origin
            );
        }
    };
    for part in geometry.architecture.parts.values() {
        check(part, Transform::IDENTITY);
    }
    for (object, assembly) in scene.objects.iter().zip(&geometry.objects) {
        for part in assembly.parts.values() {
            check(part, object.transform());
        }
    }
    for (human, assembly) in scene.humans.iter().zip(&geometry.humans) {
        for part in assembly.parts.values() {
            check(part, human.transform());
        }
    }
}

#[test]
fn only_nonzero_replacement_structures_can_omit_parent_maps() {
    use std::collections::BTreeSet;
    let mut scene =
        IndoorManifest::generate(207, super::super::layout::IndoorLayout::Mixed, 0.65, 0).unwrap();
    let mut selection = MaterialSelection {
        surfaces: BTreeSet::from([Surface::Fabric]),
        finishes: BTreeSet::from([(Surface::Fabric, 1), (Surface::Fabric, 2)]),
        direct_surfaces: BTreeSet::new(),
    };
    assert!(selection.omits_parent_maps(&scene, Surface::Fabric));
    selection.direct_surfaces.insert(Surface::Fabric);
    assert!(selection.needs_maps(&scene, Surface::Fabric));
    selection.direct_surfaces.clear();
    for slot in [0, 3, variants::COUNT, usize::MAX] {
        selection.finishes.insert((Surface::Fabric, slot));
        assert!(selection.needs_maps(&scene, Surface::Fabric));
        selection.finishes.remove(&(Surface::Fabric, slot));
    }
    scene.program = None;
    assert!(selection.needs_maps(&scene, Surface::Fabric));
    selection.finishes.clear();
    assert!(selection.needs_maps(&scene, Surface::Fabric));
}

#[test]
fn replaced_parent_maps_preserve_every_finish_pbr_and_gi_hit() {
    use super::super::{layout::IndoorLayout, objects::Assembly, preparation::SceneGeometry};
    let mut scene =
        IndoorManifest::generate_with_humans(200, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    scene.objects.clear();
    let mut architecture = Assembly::default();
    let mut surfaces = Vec::new();
    for surface in variants::SURFACES {
        if variants::structure_count(surface) <= 1 {
            continue;
        }
        let x = surfaces.len() as f32 * 0.25 - 1.;
        architecture.box_part(
            surface,
            "chair#finish1",
            Vec3::new(x, 1., 0.),
            Vec3::splat(0.2),
            0.,
        );
        surfaces.push(surface);
    }
    let geometry = SceneGeometry {
        architecture,
        objects: Vec::new(),
        humans: Vec::new(),
    };
    let selection = geometry.material_selection(&scene);
    assert_eq!(surfaces.len(), 13);
    assert!(selection.direct_surfaces.is_empty());
    let (mut all_images, mut all_materials) = (Assets::default(), Assets::default());
    let all = IndoorMaterials::build(&scene, &mut all_images, &mut all_materials);
    let (mut images, mut materials) = (Assets::default(), Assets::default());
    let used = IndoorMaterials::build_with_selection(
        &scene,
        super::super::IndoorQuality::Auto,
        &mut images,
        &mut materials,
        Some(&selection),
        false,
    );
    for surface in surfaces {
        assert!(selection.omits_parent_maps(&scene, surface));
        assert!(materials
            .get(&used.get(surface))
            .unwrap()
            .base_color_texture
            .is_none());
        assert_same_material(
            all_materials
                .get(&all.for_part(surface, "chair#finish1"))
                .unwrap()
                .clone(),
            materials
                .get(&used.for_part(surface, "chair#finish1"))
                .unwrap()
                .clone(),
            &all_images,
            &images,
        );
    }
    assert_same_gi_part_rays(
        &scene,
        &geometry,
        &all,
        &used,
        &all_materials,
        &materials,
        &all_images,
        &images,
    );
}

#[test]
#[ignore = "count-only seed coverage diagnostic; no textures or GPU work"]
fn variant_parent_map_counts_200_through_231() {
    use super::super::{humans, layout::IndoorLayout, preparation::SceneGeometry};
    let mut rows = Vec::new();
    let mut total = 0;
    for seed in 200..232 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
        let geometry = SceneGeometry {
            architecture: super::super::architecture::architecture(&scene),
            objects: scene
                .objects
                .iter()
                .map(super::super::objects::build_object)
                .collect(),
            humans: scene.humans.iter().map(humans::build_human).collect(),
        };
        let selection = geometry.material_selection(&scene);
        let omitted: Vec<_> = program::SURFACES
            .into_iter()
            .filter(|&s| selection.omits_parent_maps(&scene, s))
            .map(|s| format!("{s:?}"))
            .collect();
        total += omitted.len();
        rows.push(serde_json::json!({"seed":seed,"humans":scene.humans.len(),"omitted_parent_triplets":omitted,"selected_base_surfaces_before":selection.surfaces.len()}));
    }
    eprintln!(
        "{}",
        serde_json::json!({"scope":"exact assembly references; count-only, no capture speedup claim","rooms":rows,"omitted_parent_triplets":total,"omitted_images":total*3})
    );
}

#[test]
fn ceramic_items_share_three_glazes_with_independent_pigment() {
    let scene = IndoorManifest::generate_with_humans(
        81,
        super::super::layout::IndoorLayout::Conference,
        0.5,
        0,
        0.,
    )
    .unwrap();
    let mut images = Assets::default();
    let mut materials = Assets::default();
    let set = IndoorMaterials::build(&scene, &mut images, &mut materials);
    let mut structures = std::collections::BTreeSet::new();
    let mut pigments = std::collections::BTreeSet::new();
    for slot in 0..variants::COUNT {
        let mat = materials
            .get(&set.variants[&(Surface::Ceramic, slot)])
            .unwrap();
        let r = scene.program.as_ref().unwrap().materials[Surface::Ceramic as usize].variant(slot);
        let c = r.coating.as_ref().unwrap();
        assert_eq!(mat.clearcoat, c.clearcoat);
        assert_eq!(mat.clearcoat_perceptual_roughness, c.coat_roughness);
        assert_eq!(mat.metallic, 0.);
        assert_eq!(mat.occlusion_texture, mat.metallic_roughness_texture);
        structures.insert(mat.normal_map_texture.as_ref().unwrap().id());
        pigments.insert(mat.base_color.to_srgba().to_u8_array());
    }
    assert_eq!(structures.len(), 3);
    assert_eq!(pigments.len(), variants::COUNT);
    let size = images.len();
    for slot in 0..1000 {
        set.for_part(
            Surface::Ceramic,
            &format!("mug#finish{}", slot % variants::COUNT),
        );
    }
    assert_eq!(images.len(), size);
}

#[test]
fn wardrobe_reuses_bounded_structure_maps_with_independent_tint() {
    use super::super::humans::{cloth_finish, person_material, HumanSurface};
    let scene = IndoorManifest::generate_with_humans(
        44,
        super::super::layout::IndoorLayout::Mixed,
        0.65,
        0,
        0.8,
    )
    .unwrap();
    assert!(!scene.humans.is_empty());
    let mut images = Assets::default();
    let mut materials = Assets::default();
    let indoor = IndoorMaterials::build(&scene, &mut images, &mut materials);
    let before = images.len();
    let mut structures = std::collections::BTreeSet::new();
    for person in &scene.humans {
        for surface in [
            HumanSurface::Shoes,
            HumanSurface::Sole,
            HumanSurface::ShoeDetail,
        ] {
            let material = person_material(person, surface, &indoor, &materials);
            assert_eq!(material.base_color, person.material_color(surface));
            assert!(material.base_color_texture.is_some());
            assert!(material.normal_map_texture.is_some());
            assert!(material.metallic_roughness_texture.is_some());
            assert_eq!(material.metallic, 0.);
            assert_eq!(material.clearcoat, 0.);
        }
        for surface in [HumanSurface::Top, HumanSurface::Trousers] {
            let m = person_material(person, surface, &indoor, &materials);
            let base = if person.outfit.knitted() && surface == HumanSurface::Top {
                materials.get(&indoor.knit).unwrap()
            } else {
                materials
                    .get(&indoor.variants[&cloth_finish(person, surface)])
                    .unwrap()
            };
            assert_eq!(m.base_color, person.material_color(surface));
            assert_eq!(m.base_color_texture, base.base_color_texture);
            assert_eq!(m.normal_map_texture, base.normal_map_texture);
            assert_eq!(
                m.metallic_roughness_texture,
                base.metallic_roughness_texture
            );
            assert_eq!(m.occlusion_texture, m.metallic_roughness_texture);
            assert_eq!(m.metallic, 0.);
            assert_eq!(m.clearcoat, 0.);
            structures.insert(m.normal_map_texture.unwrap().id());
            if let Some(a) = &person.appearance {
                assert!(
                    (m.anisotropy_rotation - base.anisotropy_rotation + a.weave_rotation).abs()
                        < 1e-5
                );
            }
        }
    }
    assert!(structures.len() <= 7);
    assert_eq!(images.len(), before, "wardrobe allocated per-person maps");
}

#[test]
fn hard_surface_maps_vary_normals_and_roughness_without_losing_metalness() {
    let recipes = program::sample(81);
    for surface in [
        Surface::Metal,
        Surface::Chrome,
        Surface::Plastic,
        Surface::Rubber,
        Surface::Paper,
    ] {
        let recipe = &recipes[surface as usize];
        let (_, normals, data) = texture_maps(surface, 0, 81, recipe.roughness, Some(recipe));
        assert!(
            normals
                .as_chunks::<4>()
                .0
                .iter()
                .any(|p| p[0] != 128 || p[1] != 128),
            "{surface:?} has no resolved relief"
        );
        let roughness: std::collections::BTreeSet<_> =
            data.as_chunks::<4>().0.iter().map(|p| p[1]).collect();
        assert!(roughness.len() > 8, "{surface:?} has flat roughness");
        assert!(data
            .as_chunks::<4>()
            .0
            .iter()
            .all(|p| p[2] == if surface == Surface::Chrome { 255 } else { 0 }));
    }
}

#[test]
fn pbr_maps_have_correct_encodings_mips_and_finite_unit_normals() {
    for surface in [
        Surface::Wood,
        Surface::Floor,
        Surface::Fabric,
        Surface::Leaf,
        Surface::Concrete,
    ] {
        let (color, normals, data) = texture_maps(surface, 0, 11, 0.7, None);
        assert_ne!(color, texture_maps(surface, 0, 12, 0.7, None).0);
        for pixel in normals.as_chunks::<4>().0.iter() {
            let normal =
                Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32) / 127.5 - Vec3::ONE;
            assert!((normal.length() - 1.0).abs() < 0.018);
            assert!(normal.z > 0.0);
        }
        for pixel in data.as_chunks::<4>().0.iter() {
            assert_eq!(pixel[2], 0, "dielectrics must not become metallic");
        }
        let color_image = mip_image(color, 256, MapType::Color);
        let normal_image = mip_image(normals, 256, MapType::Normal);
        let data_image = mip_image(data, 256, MapType::Data);
        assert_eq!(
            color_image.texture_descriptor.format,
            TextureFormat::Rgba8UnormSrgb
        );
        assert_eq!(
            normal_image.texture_descriptor.format,
            TextureFormat::Rgba8Unorm
        );
        assert_eq!(
            data_image.texture_descriptor.format,
            TextureFormat::Rgba8Unorm
        );
        assert_eq!(color_image.texture_descriptor.mip_level_count, 9);
        assert_eq!(
            color_image.data.as_ref().unwrap().len(),
            (0..9)
                .map(|m| ((256 >> m) as usize).pow(2) * 4)
                .sum::<usize>()
        );
    }
}

#[test]
fn sampled_metric_recipes_produce_valid_pbr_maps() {
    for seed in [0, 6, 115] {
        let recipes = program::sample(seed);
        for surface in [
            Surface::Paint,
            Surface::Wood,
            Surface::Fabric,
            Surface::Floor,
        ] {
            let recipe = &recipes[surface as usize];
            let (colors, normals, data) = texture_maps(
                surface,
                seed as u32 % 3,
                seed,
                recipe.roughness,
                Some(recipe),
            );
            let size = recipe.map_size(seed as u32 % 3) as usize;
            assert_eq!(colors.len(), size * size * 4);
            for pixel in normals.as_chunks::<4>().0.iter() {
                let n = Vec3::new(pixel[0] as f32, pixel[1] as f32, pixel[2] as f32) / 127.5
                    - Vec3::ONE;
                assert!((n.length() - 1.0).abs() < 0.018 && n.z > 0.0);
            }
            assert!(data
                .as_chunks::<4>()
                .0
                .iter()
                .all(|p| p[1] >= 30 && p[2] == 0));
        }
    }
}

#[test]
fn portable_glazing_avoids_screen_space_refraction() {
    let scene =
        IndoorManifest::generate(0, super::super::layout::IndoorLayout::Mixed, 0.65, 1).unwrap();
    let mut images = Assets::default();
    let mut materials = Assets::default();
    let set = IndoorMaterials::build_with_quality(
        &scene,
        super::super::IndoorQuality::Portable,
        &mut images,
        &mut materials,
    );
    let glass = materials.get(&set.get(Surface::Glass)).unwrap();
    assert_eq!(glass.alpha_mode, AlphaMode::Blend);
    assert_eq!(glass.specular_transmission, 0.0);
}

#[test]
fn surface_ordinals_wire_names_and_default_recipe_slots_remain_aligned() {
    let names: Vec<String> = serde_json::from_str(include_str!(
        "../../../../tests/fixtures/indoor_surface_catalog.json"
    ))
    .unwrap();
    assert_eq!(names.len(), Surface::COUNT);
    assert_eq!(program::SURFACES, Surface::ALL);
    for (id, (surface, name)) in Surface::ALL.into_iter().zip(names).enumerate() {
        assert_eq!(surface as usize, id);
        assert_eq!(serde_json::to_value(surface).unwrap(), name);
        assert_eq!(
            serde_json::from_value::<Surface>(serde_json::Value::String(name)).unwrap(),
            surface
        );
    }
    for seed in [0, 1, 400, u64::MAX] {
        let scene = IndoorManifest::generate_with_humans(
            seed,
            super::super::layout::IndoorLayout::Mixed,
            0.65,
            0,
            0.,
        )
        .unwrap();
        for (index, definition) in definitions(&scene).into_iter().enumerate() {
            assert_eq!(definition.0 as usize, index);
        }
        for (index, recipe) in program::sample(seed).into_iter().enumerate() {
            assert_eq!(recipe.surface as usize, index);
        }
    }
}
