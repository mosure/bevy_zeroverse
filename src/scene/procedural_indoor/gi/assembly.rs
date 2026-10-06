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
}

impl GeometryTransport {
    pub(crate) fn build(
        scene: &IndoorManifest,
        moving_humans: &[usize],
        geometry: &super::super::preparation::SceneGeometry,
        gpu_transport: bool,
    ) -> Self {
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
        for (i, p) in architecture::fixture_positions(scene)
            .into_iter()
            .enumerate()
        {
            let (c, lumens) = architecture::fixture_photometry(scene, i);
            let (inner, outer) = architecture::fixture_angles(scene, i);
            result.scene.lights.push(LocalLight {
                position: p - Vec3::Y * 0.06,
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
                position: lamp.position + Vec3::Y * (lamp.size.y - 0.22),
                color: linear(Color::srgb(1.0, 0.78, 0.57)),
                candela: architecture::floor_lamp_lumens(scene, lamp.seed) / (4.0 * PI),
                range: 5.0,
                spot: false,
                inner_cos: 1.0,
                outer_cos: 0.0,
            });
        }
        result.scene.build_bvh();
        result.scene.preparation_ms = started.elapsed().as_secs_f64() * 1000.0;
        #[cfg(not(target_arch = "wasm32"))]
        if gpu_transport {
            result.scene.prepare_gpu_transport();
        }
        #[cfg(target_arch = "wasm32")]
        let _ = gpu_transport;
        result
    }

    fn add_assembly(
        &mut self,
        assembly: &Assembly,
        transform: Transform,
        symbols: &mut BTreeMap<MaterialKey, usize>,
    ) {
        for ((surface, label), geometry) in &assembly.parts {
            let surface = *surface;
            if matches!(
                surface,
                Surface::Glass
                    | Surface::GlassInterior
                    | Surface::ContainerGlass
                    | Surface::Liquid
                    | Surface::Light
            ) {
                continue;
            }
            // An unsupported/missing finish resolves to its original base only
            // during binding, just as the completed material palette does.
            let key = label
                .rsplit_once("#finish")
                .and_then(|(_, slot)| slot.parse::<usize>().ok())
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
                .push(DiffuseMaterial::from_standard(mat, images));
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
        // Partitioning never reads material indices. Bind both final orders in
        // place; neither tree, geometry attributes nor traversal changes.
        for triangle in &mut self.scene.triangles {
            triangle.material = remap[triangle.material];
        }
        #[cfg(not(target_arch = "wasm32"))]
        if let Some(transport) = self.scene.transport_tree.get_mut() {
            for triangle in &mut transport.triangles {
                triangle.material = remap[triangle.material];
            }
        }
        self.scene.preparation_ms += started.elapsed().as_secs_f64() * 1000.0;
        self.scene
    }
}

#[cfg(test)]
mod replay_tests;
