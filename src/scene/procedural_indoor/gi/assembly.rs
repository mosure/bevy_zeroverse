//! Geometry-only transport preparation, followed by ordered material binding.
//! The BVH partitions use geometry alone. Symbolic indices let that CPU work
//! overlap map synthesis without copying assemblies or changing triangle order.
use super::*;
use crate::scene::procedural_indoor::humans::{self, HumanSurface};
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum MaterialKey {
    Surface(Surface),
    Finish(Surface, usize),
    Human(usize, HumanSurface),
}

/// No radiance query can use this intermediate: material indices are symbolic
/// until `bind` consumes it. It owns only the final trees and a small key table.
pub(crate) struct GeometryTransport {
    scene: BakeScene,
    keys: Vec<MaterialKey>,
    pub(crate) timings: super::super::preparation::TransportTimings,
}

/// Owns emitted triangles only. Releasing the assembly borrow lets tangent/mesh
/// realization proceed while the spatial trees are built, without scene clones.
pub(crate) struct GeometryInput(GeometryTransport);

impl GeometryInput {
    pub(crate) fn finish(self, gpu_transport: bool) -> GeometryTransport {
        let mut result = self.0;
        let started = Instant::now();
        result.scene.build_bvh();
        result.timings.probe_tree_seconds = started.elapsed().as_secs_f64();
        result.timings.probe_nodes = result.scene.nodes.len();
        result.scene.preparation_ms += result.timings.probe_tree_seconds * 1000.;
        #[cfg(not(target_arch = "wasm32"))]
        if gpu_transport {
            let started = Instant::now();
            result.scene.prepare_gpu_transport();
            result.timings.shadow_tree_seconds = started.elapsed().as_secs_f64();
        }
        #[cfg(target_arch = "wasm32")]
        let _ = gpu_transport;
        result
    }
}

impl GeometryTransport {
    pub(crate) fn build(
        scene: &IndoorManifest,
        moving_humans: &[usize],
        geometry: &super::super::preparation::SceneGeometry,
        gpu_transport: bool,
    ) -> Self {
        Self::emit(scene, moving_humans, geometry).finish(gpu_transport)
    }

    pub(crate) fn emit(
        scene: &IndoorManifest,
        moving_humans: &[usize],
        geometry: &super::super::preparation::SceneGeometry,
    ) -> GeometryInput {
        let started = Instant::now();
        let mut result = Self {
            scene: BakeScene {
                triangles: Vec::new(),
                nodes: Vec::new(),
                #[cfg(not(target_arch = "wasm32"))]
                transport_tree: default(),
                materials: Vec::new(),
                lights: Vec::new(),
                sun_direction: architecture::sun_direction(scene),
                sun: linear(architecture::sun_color(scene)) * architecture::sun_illuminance(scene),
                sky: scene.sky_radiance(),
                bounds_min: Vec3::new(
                    -scene.room_size.x * 0.5 - 0.10,
                    scene.envelope.as_ref().map_or(0., |e| e.minimum_floor()) - 0.10,
                    -scene.room_size.z * 0.5 - 0.10,
                ),
                bounds_max: Vec3::new(
                    scene.room_size.x * 0.5 + 0.10,
                    scene.room_size.y + 0.10,
                    scene.room_size.z * 0.5 + super::super::layout::NEIGHBOR_DEPTH + 0.10,
                ),
                preparation_ms: 0.0,
                world_rotation: Quat::from_rotation_y(scene.world_yaw),
            },
            keys: super::super::materials::program::SURFACES
                .into_iter()
                .map(MaterialKey::Surface)
                .collect(),
            timings: default(),
        };
        let mut symbols: BTreeMap<_, _> = result
            .keys
            .iter()
            .copied()
            .enumerate()
            .map(|(i, key)| (key, i))
            .collect();
        result.add_assembly(&geometry.architecture, Transform::IDENTITY, &mut symbols);
        for (object, assembly) in scene.objects.iter().zip(&geometry.objects) {
            result.add_assembly(assembly, object.transform(), &mut symbols);
        }
        for (person_index, (person, assembly)) in scene
            .humans
            .iter()
            .zip(&geometry.humans)
            .enumerate()
            .filter(|(_, (p, _))| !moving_humans.contains(&p.id))
        {
            for (&surface, geometry) in &assembly.parts {
                // Clear spectacles are not opaque diffuse shadow casters.
                if surface == HumanSurface::Lens {
                    continue;
                }
                let index = result.keys.len();
                result.keys.push(MaterialKey::Human(person_index, surface));
                result
                    .scene
                    .add_geometry(geometry, person.transform(), index);
            }
        }
        let fixture_positions = architecture::fixture_positions(scene);
        for (i, p) in fixture_positions.iter().copied().enumerate() {
            let (c, lumens) = architecture::fixture_photometry_at(scene, i, &fixture_positions);
            let (inner, outer) = architecture::fixture_angles(scene, i);
            result.scene.lights.push(LocalLight {
                direction: architecture::fixtures::direction(scene, p),
                position: architecture::fixtures::emitter(scene, p),
                color: linear(Color::srgb(c.x, c.y, c.z)),
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
            result.scene.lights.push(LocalLight {
                direction: Vec3::NEG_Y,
                position: crate::scene::procedural_indoor::objects::standing_lamp::emitter(lamp),
                color: {
                    let c = crate::scene::procedural_indoor::photometry::lamp(scene, lamp).color;
                    linear(Color::srgb(c.x, c.y, c.z))
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
            let (position, color, lumens, inner, outer) =
                architecture::object_fixture_photometry(scene, lamp);
            result.scene.lights.push(LocalLight {
                direction: Vec3::NEG_Y,
                position,
                color: linear(Color::srgb(color.x, color.y, color.z)),
                candela: architecture::spot_intensity_for_lumens(lumens, inner, outer) / (4.0 * PI),
                range: 3.0,
                spot: true,
                inner_cos: inner.cos(),
                outer_cos: outer.cos(),
            });
        }
        result.timings.emission_seconds = started.elapsed().as_secs_f64();
        result.timings.triangles = result.scene.triangles.len();
        result.scene.preparation_ms = result.timings.emission_seconds * 1000.;
        GeometryInput(result)
    }

    fn add_assembly(
        &mut self,
        assembly: &Assembly,
        transform: Transform,
        symbols: &mut BTreeMap<MaterialKey, usize>,
    ) {
        for ((surface, label), geometry) in &assembly.parts {
            if !assembly.joints.is_empty()
                && super::super::motion::tag(label)
                    .and_then(|id| assembly.joints.get(id))
                    .is_some_and(|j| j.accepted)
            {
                continue;
            }
            let surface = *surface;
            if matches!(
                surface,
                Surface::Glass | Surface::GlassInterior | Surface::ContainerGlass | Surface::Liquid
            ) || (surface == Surface::Light && decorative_emitter_slot(label).is_none())
            {
                continue;
            }
            // An unsupported/missing finish resolves to its original base only
            // during binding, just as the completed material palette does.
            let key = decorative_emitter_slot(label)
                .or_else(|| crate::scene::procedural_indoor::materials::finish_slot(surface, label))
                .map_or(MaterialKey::Surface(surface), |slot| {
                    MaterialKey::Finish(surface, slot)
                });
            let index = *symbols.entry(key).or_insert_with(|| {
                let index = self.keys.len();
                self.keys.push(key);
                index
            });
            self.scene.add_geometry(geometry, transform, index);
        }
    }

    pub(crate) fn bind(
        mut self,
        scene: &IndoorManifest,
        set: &IndoorMaterials,
        materials: &impl super::super::preparation::AssetStore<StandardMaterial>,
        images: &impl super::super::preparation::AssetStore<Image>,
        moving_humans: &[usize],
    ) -> BakeScene {
        let started = Instant::now();
        let mut finish_indices = BTreeMap::new();
        let handles = super::super::materials::program::SURFACES
            .into_iter()
            .map(|surface| (surface, None, set.get(surface)))
            .chain(
                set.variants
                    .iter()
                    .map(|(&(surface, slot), handle)| (surface, Some(slot), handle.clone())),
            );
        for (surface, slot, handle) in handles {
            if let Some(slot) = slot {
                finish_indices.insert((surface, slot), self.scene.materials.len());
            }
            let mat = materials.get(&handle).expect("indoor material exists");
            self.scene
                .materials
                .push(DiffuseMaterial::from_standard(mat, images, &set.gpu_bakes));
        }
        for object in &scene.objects {
            if super::super::photometry::decorative_emission(object).is_none() {
                continue;
            }
            for channel in 0..crate::scene::procedural_indoor::photometry::emission_channels(object)
            {
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
                    self.scene.materials.len(),
                );
                self.scene.materials.push(DiffuseMaterial::from_standard(
                    mat,
                    images,
                    &set.gpu_bakes,
                ));
            }
        }
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
        let remap: Vec<_> = self
            .keys
            .iter()
            .map(|key| match *key {
                MaterialKey::Surface(surface) => surface as usize,
                MaterialKey::Finish(surface, slot) => finish_indices
                    .get(&(surface, slot))
                    .copied()
                    .unwrap_or(surface as usize),
                MaterialKey::Human(person_index, surface) => {
                    let person = &scene.humans[person_index];
                    let cloth = matches!(
                        surface,
                        HumanSurface::Top
                            | HumanSurface::Trousers
                            | HumanSurface::Shirt
                            | HumanSurface::Seam
                    );
                    let mut material = if person.outfit.knitted()
                        && matches!(surface, HumanSurface::Top | HumanSurface::Seam)
                    {
                        let mut material = knit.as_ref().expect("static knit proxy exists").clone();
                        material.uv_scale *=
                            person.appearance.as_ref().map_or(1., |a| a.weave_scale);
                        material
                    } else if cloth {
                        let key = humans::cloth_finish(person, surface);
                        let index = finish_indices.get(&key).copied().unwrap_or(key.0 as usize);
                        let mut material = self.scene.materials[index].clone();
                        material.uv_scale *=
                            2. * person.appearance.as_ref().map_or(1., |a| a.weave_scale);
                        material
                    } else {
                        DiffuseMaterial {
                            albedo: Vec3::ONE,
                            emission: Vec3::ZERO,
                            uv_scale: Vec2::ONE,
                            texture: None,
                            textured_emission: false,
                        }
                    };
                    material.albedo = linear(person.material_color(surface));
                    let index = self.scene.materials.len();
                    self.scene.materials.push(material);
                    index
                }
            })
            .collect();
        // The shadow tree refers back to this final median order. Bind attributes
        // once: its compact geometry needs no material remap or duplicate scan.
        for triangle in &mut self.scene.triangles {
            triangle.material = remap[triangle.material];
        }
        self.scene.preparation_ms += started.elapsed().as_secs_f64() * 1000.0;
        self.scene
    }
}

#[cfg(test)]
mod replay_tests;
