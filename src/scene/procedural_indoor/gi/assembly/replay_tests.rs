//! Independent copies of the accepted assembly/binding and serial median
//! builder. They must not call the new split preparation implementation.
use super::*;
use crate::scene::procedural_indoor::photometry::source_color;
use crate::scene::procedural_indoor::{
    geometry::Geometry,
    humans::{HumanAssembly, HumanOutfit, HumanSurface},
    layout::IndoorLayout,
    materials,
    preparation::{self, SceneGeometry},
};
fn original_from_geometry(
    scene: &IndoorManifest,
    set: &IndoorMaterials,
    materials: &impl preparation::AssetStore<StandardMaterial>,
    images: &impl preparation::AssetStore<Image>,
    moving_humans: &[usize],
    geometry: &preparation::SceneGeometry,
) -> BakeScene {
    let started = Instant::now();
    let mut result = BakeScene {
        triangles: Vec::new(),
        nodes: Vec::new(),
        #[cfg(not(target_arch = "wasm32"))]
        transport_tree: default(),
        materials: Vec::new(),
        lights: Vec::new(),
        sun_direction: architecture::sun_direction(scene),
        sun: linear(architecture::sun_color(scene)) * architecture::sun_illuminance(scene),
        sky: scene.sky_radiance(),
        sky_field: set.sky_field.clone(),
        bounds_min: Vec3::new(
            -scene.room_size.x * 0.5 - 0.10,
            scene.envelope.as_ref().map_or(0., |e| e.minimum_floor()) - 0.10,
            -scene.room_size.z * 0.5 - 0.10,
        ),
        bounds_max: Vec3::new(
            scene.room_size.x * 0.5 + 0.10,
            scene.room_size.y + 0.10,
            scene.room_size.z * 0.5 + scene.adjoining_space().max_depth() + 0.10,
        ),
        preparation_ms: 0.0,
        world_rotation: Quat::from_rotation_y(scene.world_yaw),
        probe_domain: ProbeDomain::from_manifest(scene),
    };
    let mut finish_indices = std::collections::BTreeMap::new();
    let handles = materials::program::SURFACES
        .into_iter()
        .map(|surface| (surface, None, set.get(surface)))
        .chain(
            set.variants
                .iter()
                .map(|(&(surface, slot), handle)| (surface, Some(slot), handle.clone())),
        );
    for (surface, slot, handle) in handles {
        if let Some(slot) = slot {
            finish_indices.insert((surface, slot), result.materials.len());
        }
        let mat = materials.get(&handle).expect("indoor material exists");
        result
            .materials
            .push(DiffuseMaterial::from_standard(mat, images, &set.gpu_bakes));
    }
    for object in &scene.objects {
        if objects::equipment::signals::strings::practical(object)
            || crate::scene::procedural_indoor::photometry::decorative_emission(object).is_none()
        {
            continue;
        }
        for channel in 0..crate::scene::procedural_indoor::photometry::emission_channels(object) {
            let label = format!(
                "{}#source{}#bulb{}",
                object.kind.class_name(),
                object.id,
                channel
            );
            let mat = materials
                .get(&set.for_part(Surface::Light, &label))
                .expect("decorative emitter");
            finish_indices.insert(
                (Surface::Light, decorative_emitter_slot(&label).unwrap()),
                result.materials.len(),
            );
            result
                .materials
                .push(DiffuseMaterial::from_standard(mat, images, &set.gpu_bakes));
        }
    }
    original_add_assembly(
        &mut result,
        &geometry.architecture,
        Transform::IDENTITY,
        &finish_indices,
    );
    for (object, assembly) in scene.objects.iter().zip(&geometry.objects) {
        original_add_assembly(&mut result, assembly, object.transform(), &finish_indices);
    }
    // Knit Top/Seam parts use the room's neutral knit atlas in rendering.
    // Resolving their unused furniture finish made diffuse transport depend
    // on whether the caller had built a complete or a selected palette.
    // Prepare this shared proxy once, rather than once per garment part.
    let knit = scene
        .humans
        .iter()
        .any(|person| person.outfit.knitted() && !moving_humans.contains(&person.id))
        .then(|| {
            DiffuseMaterial::from_standard(
                materials
                    .get(&set.knit)
                    .expect("indoor knit material exists"),
                images,
                &set.gpu_bakes,
            )
        });
    for (person, assembly) in scene
        .humans
        .iter()
        .zip(&geometry.humans)
        .filter(|(p, _)| !moving_humans.contains(&p.id))
    {
        for (&surface, geometry) in &assembly.parts {
            use humans::HumanSurface;
            // The diffuse proxy has no thin-lens transmission model.
            // Clear spectacles must not become opaque eye shadow casters.
            if surface == HumanSurface::Lens {
                continue;
            }
            let cloth = matches!(
                surface,
                HumanSurface::Top
                    | HumanSurface::Trousers
                    | HumanSurface::Shirt
                    | HumanSurface::Seam
            );
            let mapped_eye = surface == HumanSurface::Eye && person.eye_optics().is_some();
            let mut material = if mapped_eye {
                let eye = humans::person_material(person, surface, set, materials);
                DiffuseMaterial::from_standard(&eye, images, &set.gpu_bakes)
            } else if person.outfit.knitted()
                && matches!(surface, HumanSurface::Top | HumanSurface::Seam)
            {
                knit.as_ref().expect("static knit proxy exists").clone()
            } else if cloth {
                let key = humans::cloth_finish(person, surface);
                let index = finish_indices.get(&key).copied().unwrap_or(key.0 as usize);
                result.materials[index].clone()
            } else {
                DiffuseMaterial {
                    albedo: Vec3::ONE,
                    emission: Vec3::ZERO,
                    uv_transform: Affine2::IDENTITY,
                    base_map: None,
                    emission_map: EmissionMap::Uniform,
                }
            };
            if cloth {
                material.uv_transform =
                    humans::person_material(person, surface, set, materials).uv_transform;
            }
            if !mapped_eye {
                material.albedo = linear(person.material_color(surface));
            }
            let index = result.materials.len();
            result.materials.push(material);
            original_add_geometry(&mut result, geometry, person.transform(), index);
        }
    }
    for (i, p) in architecture::fixture_positions(scene)
        .into_iter()
        .enumerate()
    {
        let (c, lumens) = architecture::fixture_photometry(scene, i);
        let (inner, outer) = architecture::fixtures::beam_angles(scene, i, p);
        result.lights.push(LocalLight {
            direction: architecture::fixtures::direction(scene, p),
            position: architecture::fixtures::emitter(scene, p),
            color: linear(source_color(scene, c)),
            candela: architecture::spot_intensity_for_lumens(lumens, inner, outer) / (4.0 * PI),
            range: 13.0,
            spot: true,
            inner_cos: inner.cos(),
            outer_cos: outer.cos(),
        });
    }
    for lamp in scene
        .objects
        .iter()
        .filter(|o| o.kind == ObjectKind::FloorLamp)
    {
        result.lights.push(LocalLight {
            direction: Vec3::NEG_Y,
            position: crate::scene::procedural_indoor::objects::standing_lamp::emitter(lamp),
            color: {
                let c = crate::scene::procedural_indoor::photometry::lamp(scene, lamp).color;
                linear(source_color(scene, c))
            },
            candela: crate::scene::procedural_indoor::photometry::lamp(scene, lamp).lumens
                / (4.0 * PI),
            range: 5.0,
            spot: false,
            inner_cos: 1.0,
            outer_cos: 0.0,
        });
    }
    for lamp in scene.objects.iter().filter(|o| {
        matches!(
            o.kind,
            ObjectKind::DeskLamp | ObjectKind::WallSconce | ObjectKind::CeilingPendant
        )
    }) {
        let (beam, power) = architecture::object_fixture_photometry(scene, lamp);
        result.lights.push(LocalLight {
            direction: beam.direction,
            position: beam.position,
            color: linear(source_color(scene, power.color)),
            candela: architecture::spot_intensity_for_lumens(power.lumens, beam.inner, beam.outer)
                / (4.0 * PI),
            range: 3.0,
            spot: true,
            inner_cos: beam.inner.cos(),
            outer_cos: beam.outer.cos(),
        });
    }
    result.lights.extend(
        objects::equipment::signals::strings::practical_lights(scene)
            .map(|(_, _, l)| LocalLight::from(l)),
    );
    // The scalar oracle enumerates installed sources. A zero-candela source
    // contributes nothing at any hit and need not occupy a transport slot.
    result.lights.retain(|light| light.candela > 0.);
    result.nodes = original_median(&mut result.triangles);
    result.preparation_ms = started.elapsed().as_secs_f64() * 1000.0;
    result
}

fn original_add_assembly(
    scene: &mut BakeScene,
    assembly: &Assembly,
    transform: Transform,
    finishes: &std::collections::BTreeMap<(Surface, usize), usize>,
) {
    for ((surface, label), geometry) in &assembly.parts {
        let surface = *surface;
        // Match NotShadowCaster on transparent glazing and analytic-light
        // emitters. No double-counting emissive luminaire geometry + lights.
        if matches!(
            surface,
            Surface::Glass | Surface::GlassInterior | Surface::ContainerGlass | Surface::Liquid
        ) || (surface == Surface::Light && decorative_emitter_slot(label).is_none())
        {
            continue;
        }
        let material = decorative_emitter_slot(label)
            .or_else(|| crate::scene::procedural_indoor::materials::finish_slot(surface, label))
            .and_then(|slot| finishes.get(&(surface, slot)))
            .copied()
            .unwrap_or(surface as usize);
        original_add_geometry(scene, geometry, transform, material);
    }
}

fn original_add_geometry(
    scene: &mut BakeScene,
    geometry: &Geometry,
    transform: Transform,
    material: usize,
) {
    if geometry.indices.is_empty() {
        return;
    }
    // Indexed vertices are commonly shared by several triangles. Reuse
    // the identical world transform result without changing triangle order,
    // normal construction, UVs or intersection arithmetic.
    let positions: Vec<_> = geometry
        .positions
        .iter()
        .map(|p| transform.transform_point(Vec3::from_array(*p)))
        .collect();
    scene.triangles.reserve(geometry.indices.len() / 3);
    for i in geometry.indices.as_chunks::<3>().0 {
        let a = positions[i[0] as usize];
        let b = positions[i[1] as usize];
        let c = positions[i[2] as usize];
        let normal = (b - a).cross(c - a).normalize_or_zero();
        if normal.length_squared() < 0.5 {
            continue;
        }
        scene.triangles.push(Triangle {
            a,
            ab: b - a,
            ac: c - a,
            uv: [
                Vec2::from_array(geometry.uvs[i[0] as usize]),
                Vec2::from_array(geometry.uvs[i[1] as usize]),
                Vec2::from_array(geometry.uvs[i[2] as usize]),
            ],
            normal,
            material,
        });
    }
}

fn original_median(triangles: &mut [Triangle]) -> Vec<Node> {
    fn split(triangles: &mut [Triangle], start: usize, nodes: &mut Vec<Node>) {
        let index = nodes.len();
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for t in triangles.iter() {
            let (a, b) = t.bounds();
            lo = lo.min(a);
            hi = hi.max(b);
        }
        nodes.push(Node {
            lo: lo - Vec3::splat(1e-5),
            hi: hi + Vec3::splat(1e-5),
            start,
            count: triangles.len(),
            right: 0,
            axis: 0,
        });
        if triangles.len() > 8 {
            let size = hi - lo;
            let axis = if size.x > size.y && size.x > size.z {
                0
            } else if size.y > size.z {
                1
            } else {
                2
            };
            let middle = triangles.len() / 2;
            triangles.select_nth_unstable_by(middle, |a, b| {
                (a.a + (a.ab + a.ac) / 3.0)[axis].total_cmp(&(b.a + (b.ab + b.ac) / 3.0)[axis])
            });
            let (left, right) = triangles.split_at_mut(middle);
            split(left, start, nodes);
            nodes[index].right = nodes.len();
            split(right, start + middle, nodes);
            nodes[index].axis = axis;
            nodes[index].count = 0;
        }
    }
    let mut nodes = Vec::new();
    if !triangles.is_empty() {
        split(triangles, 0, &mut nodes);
    }
    nodes
}

fn vec3_bits(v: Vec3) -> [u32; 3] {
    v.to_array().map(f32::to_bits)
}

fn same_triangles(a: &[Triangle], b: &[Triangle]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        for (a, b) in [(a.a, b.a), (a.ab, b.ab), (a.ac, b.ac), (a.normal, b.normal)] {
            assert_eq!(vec3_bits(a), vec3_bits(b));
        }
        for (a, b) in a.uv.iter().zip(b.uv) {
            assert_eq!(
                a.to_array().map(f32::to_bits),
                b.to_array().map(f32::to_bits)
            );
        }
        assert_eq!(a.material, b.material);
    }
}

fn same_nodes(a: &[Node], b: &[Node]) {
    assert_eq!(a.len(), b.len());
    for (a, b) in a.iter().zip(b) {
        assert_eq!(vec3_bits(a.lo), vec3_bits(b.lo));
        assert_eq!(vec3_bits(a.hi), vec3_bits(b.hi));
        assert_eq!(
            (a.start, a.count, a.right, a.axis),
            (b.start, b.count, b.right, b.axis)
        );
    }
}

fn same_scene(a: &BakeScene, b: &BakeScene) {
    same_triangles(&a.triangles, &b.triangles);
    same_nodes(&a.nodes, &b.nodes);
    assert_eq!(a.materials.len(), b.materials.len());
    for (a, b) in a.materials.iter().zip(&b.materials) {
        assert_eq!(vec3_bits(a.albedo), vec3_bits(b.albedo));
        assert_eq!(vec3_bits(a.emission), vec3_bits(b.emission));
        assert_eq!(
            a.uv_transform.to_cols_array().map(f32::to_bits),
            b.uv_transform.to_cols_array().map(f32::to_bits)
        );
        assert_eq!(a.base_map, b.base_map);
        assert_eq!(a.emission_map, b.emission_map);
        for uv in [Vec2::ZERO, Vec2::new(-0.19, 0.47), Vec2::new(0.95, 1.17)] {
            let (diffuse_a, emitted_a) = a.response(uv);
            let (diffuse_b, emitted_b) = b.response(uv);
            assert_eq!(vec3_bits(diffuse_a), vec3_bits(diffuse_b));
            assert_eq!(vec3_bits(emitted_a), vec3_bits(emitted_b));
        }
    }
    for (a, b) in [
        (a.sun_direction, b.sun_direction),
        (a.sun, b.sun),
        (a.sky, b.sky),
        (a.bounds_min, b.bounds_min),
        (a.bounds_max, b.bounds_max),
    ] {
        assert_eq!(vec3_bits(a), vec3_bits(b));
    }
    assert_eq!(
        a.world_rotation.to_array().map(f32::to_bits),
        b.world_rotation.to_array().map(f32::to_bits)
    );
    assert_eq!(a.lights.len(), b.lights.len());
    for (a, b) in a.lights.iter().zip(&b.lights) {
        assert_eq!(vec3_bits(a.position), vec3_bits(b.position));
        assert_eq!(vec3_bits(a.direction), vec3_bits(b.direction));
        assert_eq!(vec3_bits(a.color), vec3_bits(b.color));
        assert_eq!(a.spot, b.spot);
        for (a, b) in [
            (a.candela, b.candela),
            (a.range, b.range),
            (a.inner_cos, b.inner_cos),
            (a.outer_cos, b.outer_cos),
        ] {
            assert_eq!(a.to_bits(), b.to_bits());
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let a_tree = a.gpu_transport();
        let b_tree = b.gpu_transport();
        same_triangles(
            &a_tree.surfaces(a).copied().collect::<Vec<_>>(),
            &b_tree.surfaces(b).copied().collect::<Vec<_>>(),
        );
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&a_tree.nodes),
            bytemuck::cast_slice::<_, u8>(&b_tree.nodes)
        );
    }
}

fn original_inside_solid(scene: &BakeScene, point: Vec3) -> bool {
    let mut backfaces = 0;
    for direction in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
        if let Some((index, _, _)) = scene.hit(point, direction, 1000.0, false) {
            if scene.triangles[index].normal.dot(direction) > 0.0 {
                backfaces += 1;
            }
        }
    }
    backfaces >= 4
}

// Copy of the accepted GPU origin preparation, including nearest-valid tie
// order. Compare against the independent complete-six-ray classification.
fn origins(scene: &BakeScene, original: bool) -> (Vec<[u32; 4]>, usize) {
    let resolution = ((scene.bounds_max - scene.bounds_min) / 1.7)
        .ceil()
        .as_uvec3()
        .max(UVec3::splat(2));
    let count = (resolution.x * resolution.y * resolution.z) as usize;
    let points: Vec<_> = (0..count as u32)
        .map(|i| {
            let xyz = UVec3::new(
                i % resolution.x,
                (i / resolution.x) % resolution.y,
                i / (resolution.x * resolution.y),
            );
            scene.bounds_min
                + (xyz.as_vec3() + Vec3::splat(0.5)) / resolution.as_vec3()
                    * (scene.bounds_max - scene.bounds_min)
        })
        .collect();
    let valid: Vec<_> = points
        .iter()
        .map(|p| {
            !if original {
                original_inside_solid(scene, *p)
            } else {
                scene.inside_solid(*p)
            }
        })
        .collect();
    let mut relocated = 0;
    let origins = points
        .iter()
        .enumerate()
        .map(|(i, point)| {
            let index = if valid[i] {
                i
            } else {
                relocated += 1;
                (0..count)
                    .filter(|&j| valid[j])
                    .min_by(|&a, &b| {
                        points[a]
                            .distance_squared(*point)
                            .total_cmp(&points[b].distance_squared(*point))
                    })
                    .unwrap_or(i)
            };
            points[index]
                .extend(index as f32)
                .to_array()
                .map(f32::to_bits)
        })
        .collect();
    (origins, relocated)
}

fn human_part(offset: f32) -> Geometry {
    let mut assembly = Assembly::default();
    assembly.box_part(
        Surface::Fabric,
        "proxy",
        Vec3::new(offset, 0.9, -0.01),
        Vec3::new(0.12, 0.18, 0.11),
        0.37,
    );
    assembly.parts.into_values().next().unwrap()
}

fn fixture(seed: u64, occupied: bool) -> (IndoorManifest, SceneGeometry) {
    let mut scene = IndoorManifest::generate_with_humans(
        seed,
        IndoorLayout::Mixed,
        0.65,
        0,
        if occupied { 1. } else { 0. },
    )
    .unwrap();
    scene.world_yaw = if occupied { 1.375 } else { -0. };
    scene.objects.truncate(3);
    scene.humans.truncate(2);
    let mut architecture = architecture::architecture(&scene);
    for (i, surface) in [
        Surface::Wood,
        Surface::Fabric,
        Surface::Leaf,
        Surface::Glass,
        Surface::GlassInterior,
        Surface::ContainerGlass,
        Surface::Liquid,
        Surface::Light,
    ]
    .into_iter()
    .enumerate()
    {
        architecture.box_part(
            surface,
            "test#finish1",
            Vec3::new(i as f32 * 0.25, 0.7, 0.41),
            Vec3::splat(0.17),
            0.21,
        );
    }
    architecture.box_part(
        Surface::Wood,
        "missing#finish999",
        Vec3::new(0.3, 0.8, 0.6),
        Vec3::splat(0.13),
        -0.7,
    );
    architecture.box_part(
        Surface::Paint,
        "malformed#finishbad",
        Vec3::new(0.1, 0.4, 0.3),
        Vec3::splat(0.23),
        0.13,
    );
    let mut humans = Vec::new();
    for (i, person) in scene.humans.iter_mut().enumerate() {
        person.id = 7 + i * 11;
        person.outfit = if i == 0 {
            HumanOutfit::Knitwear
        } else {
            HumanOutfit::Cardigan
        };
        if let Some(appearance) = &mut person.appearance {
            appearance.weave_scale = 1.73 + i as f32 * 0.58;
        }
        let mut human = HumanAssembly::default();
        for (j, surface) in [
            HumanSurface::Top,
            HumanSurface::Seam,
            HumanSurface::Trousers,
            HumanSurface::Shirt,
            HumanSurface::Skin,
            HumanSurface::Hair,
            HumanSurface::Lens,
            HumanSurface::Eyewear,
        ]
        .into_iter()
        .enumerate()
        {
            human.parts.insert(surface, human_part(j as f32 * 0.02));
        }
        // The original emits a proxy even for an empty human part.
        human.parts.insert(HumanSurface::Brow, Geometry::default());
        humans.push(human);
    }
    // Exercise all decorative color channels through both transport builders.
    for kind in [
        ObjectKind::StringLights,
        ObjectKind::FireAlarm,
        ObjectKind::FireDetector,
    ] {
        let mut o = objects::audit::fixture(kind, seed);
        o.id = scene.objects.len();
        o.position = Vec3::new(1., 2., 0.);
        scene.objects.push(o);
    }
    let objects = scene.objects.iter().map(objects::build_object).collect();
    (
        scene,
        SceneGeometry {
            architecture,
            objects,
            humans,
        },
    )
}

#[test]
fn split_transport_binding_matches_original_geometry_materials_and_probe_origins() {
    for (seed, occupied) in [(7, false), (207, true), (43_084_584, true)] {
        let (scene, geometry) = fixture(seed, occupied);
        if occupied {
            assert!(
                !scene.humans.is_empty(),
                "occupied oracle needs a real appearance"
            );
        }
        let (mut images, mut materials) = (Assets::default(), Assets::default());
        let set = IndoorMaterials::build_with_quality(
            &scene,
            super::super::super::IndoorQuality::Portable,
            &mut images,
            &mut materials,
        );
        let mut moving_sets = vec![Vec::new()];
        if let Some(person) = scene.humans.last() {
            moving_sets.push(vec![person.id]);
            moving_sets.push(scene.humans.iter().map(|p| p.id).collect());
        }
        for moving in moving_sets {
            let original =
                original_from_geometry(&scene, &set, &materials, &images, &moving, &geometry);
            for warm_transport in [false, cfg!(not(target_arch = "wasm32"))] {
                let unbound = GeometryTransport::build(&scene, &moving, &geometry, warm_transport);
                assert!(unbound.scene.materials.is_empty());
                let actual = unbound.bind(&scene, &set, &materials, &images, &moving);
                same_scene(&original, &actual);
                assert_eq!(origins(&original, true), origins(&actual, false));
                for point in [
                    Vec3::ZERO,
                    Vec3::Y * 1.45,
                    Vec3::new(-0.02, scene.room_size.y - 0.05, 0.1),
                ] {
                    for direction in [
                        Vec3::X,
                        Vec3::Y,
                        Vec3::Z,
                        Vec3::new(0.3, -0.2, 0.7).normalize(),
                    ] {
                        let a = original.reflection_radiance(point, direction, 45.);
                        let b = actual.reflection_radiance(point, direction, 45.);
                        assert_eq!(
                            (a.0.to_bits(), vec3_bits(a.1)),
                            (b.0.to_bits(), vec3_bits(b.1))
                        );
                    }
                }
            }
        }
    }
}

#[test]
#[cfg(not(target_arch = "wasm32"))]
fn preparation_scope_nested_jobs_preserve_order_and_join() {
    let pool = preparation::workers::pool();
    let ready = pool.scope(|scope| {
        for stage in 0..2 {
            scope.spawn(async move {
                pool.scope(|scope| {
                    for job in 0..17 {
                        scope.spawn(async move { (stage, job, job * job + stage) });
                    }
                })
            });
        }
    });
    for (stage, jobs) in ready.into_iter().enumerate() {
        for (job, actual) in jobs.into_iter().enumerate() {
            assert_eq!(actual, (stage, job, job * job + stage));
        }
    }
}

#[test]
fn complete_room_probe_trees_preserve_scalar_reference() {
    for seed in [600, 602] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 0, 0.25).unwrap();
        scene
            .resample_cameras_typed(3, Default::default(), 1.0)
            .unwrap();
        let (geometry, _) = bevy::tasks::block_on(SceneGeometry::build(&scene));
        let input = GeometryTransport::emit(&scene, &[], &geometry);
        let mut reference = input.0.scene.triangles.clone();
        let expected_nodes = original_median(&mut reference);
        let actual = input.finish(true);
        eprintln!(
            "complete transport oracle seed={seed} triangles={}",
            reference.len()
        );
        same_triangles(&reference, &actual.scene.triangles);
        same_nodes(&expected_nodes, &actual.scene.nodes);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let expected = super::super::bvh::build_transport_scalar_reference(&reference);
            let tree = actual.scene.gpu_transport();
            same_triangles(
                &expected.0,
                &tree.surfaces(&actual.scene).copied().collect::<Vec<_>>(),
            );
            assert_eq!(expected.1.len(), tree.nodes.len());
            for (reference, packed) in expected.1.iter().zip(tree.nodes.iter()) {
                assert_eq!(vec3_bits(reference.lo), vec3_bits(packed.lo));
                assert_eq!(vec3_bits(reference.hi), vec3_bits(packed.hi));
                if reference.count == 0 {
                    assert_eq!(packed.tag as usize, reference.axis);
                    assert_eq!(packed.index as usize, reference.right);
                } else {
                    assert_eq!(packed.tag >> 31, 1);
                    assert_eq!((packed.tag & 0x7fff_ffff) as usize, reference.count);
                    assert_eq!(packed.index as usize, reference.start);
                }
            }
        }
    }
}

#[test]
fn rest_pose_irradiance_does_not_leave_a_light_behind_after_the_lamp_moves() {
    use crate::scene::procedural_indoor::{
        motion::{IndoorMotionReport, MechanismReport, MotionKind},
        objects::audit,
    };
    let mut room =
        IndoorManifest::generate_with_humans(42, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
    let mut lamp = audit::fixture(ObjectKind::DeskLamp, 42);
    lamp.id = 0;
    lamp.position = Vec3::new(0.2, 0.65, -0.45);
    let source = objects::accessories::lamps::beam(&lamp).position;
    room.objects = vec![lamp];
    room.humans.clear();
    let geometry = SceneGeometry {
        architecture: default(),
        objects: vec![default()],
        humans: vec![],
    };
    let baked = |room: &IndoorManifest| {
        GeometryTransport::emit(room, &[], &geometry)
            .0
            .scene
            .lights
            .iter()
            .any(|l| l.position.distance(source) < 1e-6)
    };
    assert!(baked(&room));
    let mut report = IndoorMotionReport::new(default());
    report.mechanisms.push(MechanismReport {
        operation_cycle: None,
        object_id: Some(0),
        carried_object_ids: vec![],
        semantic_labels: vec!["lamp".into()],
        joint: 0,
        linked_slides: vec![],
        kind: MotionKind::Prop,
        interval: [0., 1.],
        rejection: None,
        swept_bounds: vec![],
        translation_m: Vec3::X * 0.12,
        angle_radians: 0.,
        pivot: Vec3::ZERO,
        axis: Vec3::Y,
        attempts: 1,
        adaptation: None,
        lift_m: 0.,
        requested_lift_m: 0.,
        projection_screen_travel: None,
        requested_translation_m: Vec3::X * 0.12,
        requested_angle_radians: 0.,
    });
    room.indoor_motion = Some(report);
    assert!(!baked(&room));
    let m = &mut room.indoor_motion.as_mut().unwrap().mechanisms[0];
    m.object_id = Some(1);
    m.carried_object_ids.push(0);
    m.kind = MotionKind::Cart;
    assert!(!baked(&room));
    room.indoor_motion.as_mut().unwrap().mechanisms[0].rejection = Some("obstructed".into());
    assert!(baked(&room));
}
