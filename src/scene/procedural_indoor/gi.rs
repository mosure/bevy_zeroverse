//! Deterministic diffuse light transport for generated static rooms.
//!
//! The baker traces the same triangle assemblies that are rendered. Next-event
//! estimation uses the renderer's photometric lights; cosine-weighted paths
//! transport reflected light for a bounded number of bounces. Six cosine
//! convolutions are stored as Bevy ambient cubes, in cd/m² (irradiance / π).
//! This is diffuse baked GI, not a replacement for a full specular path tracer.
mod bvh;
#[cfg(not(target_arch = "wasm32"))]
pub mod gpu;
use super::{
    architecture,
    layout::{IndoorManifest, ObjectKind},
    materials::{IndoorMaterials, Surface},
    objects::{self, Assembly},
};
use bevy::{
    asset::RenderAssetUsages,
    image::ImageSampler,
    prelude::*,
    render::render_resource::{Extent3d, TextureDimension, TextureFormat},
};
use std::{
    f32::consts::{PI, TAU},
    time::Instant,
};

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct BakeSettings {
    pub spacing: f32,
    pub rays_per_probe: u32,
    pub diffuse_bounces: u32,
    pub workers: usize,
}

impl Default for BakeSettings {
    fn default() -> Self {
        Self {
            spacing: 0.85,
            rays_per_probe: 256,
            diffuse_bounces: 3,
            workers: 8,
        }
    }
}

/// Runtime override for controlled experiments or a larger offline bake budget.
/// Portable and Wasm still omit irradiance volumes regardless of this resource.
#[derive(Resource, Clone, Copy, Debug, serde::Serialize, serde::Deserialize)]
pub struct IndoorGiSettings {
    pub enabled: bool,
    pub bake: BakeSettings,
    /// Keep the CPU path available as a correctness oracle and explicit fallback.
    pub gpu: bool,
}

impl Default for IndoorGiSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            bake: BakeSettings::default(),
            gpu: true,
        }
    }
}

/// Every input affecting transport is part of the cache identity. Float bit
/// patterns avoid approximate matches between distinct dataset configurations.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefetchKey {
    pub seed: u64,
    pub layout: super::layout::IndoorLayout,
    pub density_bits: u32,
    pub human_density_bits: u32,
    pub cameras: usize,
    pub rotation_augmentation: bool,
    pub settings: BakeSettings,
}

/// At most one CPU preparation task exists. It owns no GPU assets and is never
/// detached or replaced while running, including randomly indexed workloads.
#[cfg(not(target_arch = "wasm32"))]
#[derive(Resource, Default)]
pub struct GiPrefetch {
    task: Option<(PrefetchKey, bevy::tasks::Task<Result<ProbeData, String>>)>,
}

#[cfg(not(target_arch = "wasm32"))]
impl GiPrefetch {
    pub fn take(&mut self, key: &PrefetchKey) -> Option<ProbeData> {
        if !self
            .task
            .as_ref()
            .is_some_and(|(pending, _)| pending == key)
        {
            return None;
        }
        let (_, task) = self.task.take().unwrap();
        let started = Instant::now();
        match futures_lite::future::block_on(task) {
            Ok(mut data) => {
                data.statistics.prefetch_hit = true;
                data.statistics.prefetch_wait_ms = started.elapsed().as_secs_f64() * 1000.0;
                Some(data)
            }
            Err(error) => {
                warn!("indoor GI prefetch rejected: {error}");
                None
            }
        }
    }

    pub fn prepare(&mut self, key: PrefetchKey) {
        if self
            .task
            .as_ref()
            .is_some_and(|(_, task)| !task.is_finished())
        {
            return;
        }
        self.task = None;
        let input = key.clone();
        let task = bevy::tasks::AsyncComputeTaskPool::get().spawn(async move {
            use rand::Rng as _;
            let mut scene = IndoorManifest::generate_with_humans(
                input.seed,
                input.layout,
                f32::from_bits(input.density_bits),
                input.cameras,
                f32::from_bits(input.human_density_bits),
            )?;
            super::validation::validate_layout(&scene)?;
            if input.rotation_augmentation {
                scene.world_yaw = super::layout::stream(input.seed, 11).random_range(0.0..TAU);
            }
            let transport = {
                let mut images = Assets::default();
                let mut materials = Assets::default();
                let set = IndoorMaterials::build(&scene, &mut images, &mut materials);
                BakeScene::from_manifest(&scene, &set, &materials, &images)
            };
            Ok(transport.bake(input.settings, input.seed))
        });
        self.task = Some((key, task));
    }
}

#[derive(Clone, Debug, Default, Resource, serde::Serialize, serde::Deserialize)]
pub struct BakeStatistics {
    pub backend: String,
    pub triangles: usize,
    pub probes: usize,
    pub relocated_probes: usize,
    pub primary_rays: u64,
    pub diffuse_bounces: u32,
    pub texture_bytes: usize,
    pub preparation_ms: f64,
    pub bake_ms: Option<f64>,
    pub mean_radiance: Option<[f32; 3]>,
    pub transport_bytes: usize,
    pub prefetch_hit: bool,
    pub prefetch_wait_ms: f64,
}

/// CPU-only result: can be prepared on a bounded background worker and uploaded
/// only when the matching scene is ready. No GPU handles or ECS world are held.
pub struct ProbeData {
    pub resolution: UVec3,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    pub values: Vec<[Vec3; 6]>,
    pub statistics: BakeStatistics,
}

impl ProbeData {
    pub fn image(&self) -> Image {
        let n = self.resolution;
        let extent = Extent3d {
            width: n.x,
            height: n.y * 2,
            depth_or_array_layers: n.z * 3,
        };
        let mut pixels = vec![[0u16; 4]; (n.x * n.y * n.z * 6) as usize];
        for z in 0..n.z {
            for y in 0..n.y {
                for x in 0..n.x {
                    let p = &self.values[((z * n.y + y) * n.x + x) as usize];
                    for (face, value) in p.iter().enumerate() {
                        // Shader ordering, not the contradictory prose in Bevy's
                        // module docs: positive lobes first, negative at y + Ry.
                        let yy = y + (face as u32 % 2) * n.y;
                        let zz = z + (face as u32 / 2) * n.z;
                        pixels[((zz * n.y * 2 + yy) * n.x + x) as usize] = [
                            half::f16::from_f32(value.x).to_bits(),
                            half::f16::from_f32(value.y).to_bits(),
                            half::f16::from_f32(value.z).to_bits(),
                            half::f16::ONE.to_bits(),
                        ];
                    }
                }
            }
        }
        let mut image = Image::new(
            extent,
            TextureDimension::D3,
            bytemuck::cast_slice(&pixels).to_vec(),
            TextureFormat::Rgba16Float,
            RenderAssetUsages::default(),
        );
        image.sampler = ImageSampler::linear();
        image
    }

    pub fn transform(&self) -> Transform {
        Transform::from_translation((self.bounds_min + self.bounds_max) * 0.5)
            .with_scale(self.bounds_max - self.bounds_min)
    }

    pub fn sample(&self, point: Vec3, normal: Vec3) -> Vec3 {
        let n = self.resolution;
        let p = ((point - self.bounds_min) / (self.bounds_max - self.bounds_min) * n.as_vec3()
            - Vec3::splat(0.5))
        .clamp(Vec3::ZERO, n.as_vec3() - Vec3::ONE);
        let lo = p.floor().as_uvec3();
        let hi = (lo + UVec3::ONE).min(n - UVec3::ONE);
        let f = p.fract();
        let mut result = Vec3::ZERO;
        for dz in 0..2 {
            for dy in 0..2 {
                for dx in 0..2 {
                    let x = if dx == 0 { lo.x } else { hi.x };
                    let y = if dy == 0 { lo.y } else { hi.y };
                    let z = if dz == 0 { lo.z } else { hi.z };
                    let weight = if dx == 0 { 1.0 - f.x } else { f.x }
                        * if dy == 0 { 1.0 - f.y } else { f.y }
                        * if dz == 0 { 1.0 - f.z } else { f.z };
                    let v = &self.values[((z * n.y + y) * n.x + x) as usize];
                    result += weight
                        * (v[usize::from(normal.x < 0.0)] * normal.x.powi(2)
                            + v[2 + usize::from(normal.y < 0.0)] * normal.y.powi(2)
                            + v[4 + usize::from(normal.z < 0.0)] * normal.z.powi(2));
                }
            }
        }
        result
    }
}

#[derive(Clone)]
struct DiffuseMaterial {
    albedo: Vec3,
    emission: Vec3,
    uv_scale: Vec2,
    texture: Option<(u32, Vec<Vec3>)>,
    textured_emission: bool,
}

impl DiffuseMaterial {
    fn albedo(&self, uv: Vec2) -> Vec3 {
        let Some((n, pixels)) = &self.texture else {
            return self.albedo;
        };
        let uv = (uv * self.uv_scale).rem_euclid(Vec2::ONE);
        let x = (uv.x * *n as f32) as usize % *n as usize;
        let y = (uv.y * *n as f32) as usize % *n as usize;
        self.albedo * pixels[y * *n as usize + x]
    }
}

#[derive(Clone, Copy)]
struct Triangle {
    a: Vec3,
    ab: Vec3,
    ac: Vec3,
    uv: [Vec2; 3],
    normal: Vec3,
    material: usize,
}

impl Triangle {
    fn bounds(&self) -> (Vec3, Vec3) {
        (
            self.a.min(self.a + self.ab).min(self.a + self.ac),
            self.a.max(self.a + self.ab).max(self.a + self.ac),
        )
    }
    fn hit(&self, origin: Vec3, direction: Vec3, max: f32) -> Option<(f32, Vec2)> {
        let p = direction.cross(self.ac);
        let det = self.ab.dot(p);
        if det.abs() < 1e-9 {
            return None;
        }
        let inverse = det.recip();
        let t = origin - self.a;
        let u = t.dot(p) * inverse;
        if !(0.0..=1.0).contains(&u) {
            return None;
        }
        let q = t.cross(self.ab);
        let v = direction.dot(q) * inverse;
        if v < 0.0 || u + v > 1.0 {
            return None;
        }
        let distance = self.ac.dot(q) * inverse;
        (distance > 0.0002 && distance < max).then_some((distance, Vec2::new(u, v)))
    }
}

struct Node {
    lo: Vec3,
    hi: Vec3,
    start: usize,
    count: usize,
    right: usize,
    axis: usize,
}
impl Node {
    fn intersects(&self, origin: Vec3, inverse: Vec3, max: f32) -> bool {
        let a = (self.lo - origin) * inverse;
        let b = (self.hi - origin) * inverse;
        let near = a.min(b).max_element().max(0.0);
        let far = a.max(b).min_element().min(max);
        near <= far
    }
}

#[derive(Clone, Copy)]
struct LocalLight {
    position: Vec3,
    color: Vec3,
    candela: f32,
    range: f32,
    spot: bool,
    inner_cos: f32,
    outer_cos: f32,
}

/// Owns only immutable CPU data; safe to send to a preparation worker.
pub struct BakeScene {
    triangles: Vec<Triangle>,
    nodes: Vec<Node>,
    materials: Vec<DiffuseMaterial>,
    lights: Vec<LocalLight>,
    sun_direction: Vec3,
    sun: Vec3,
    sky: Vec3,
    pub bounds_min: Vec3,
    pub bounds_max: Vec3,
    preparation_ms: f64,
    world_rotation: Quat,
}

fn linear(color: Color) -> Vec3 {
    let c = color.to_linear();
    Vec3::new(c.red, c.green, c.blue)
}

impl BakeScene {
    pub fn from_manifest(
        scene: &IndoorManifest,
        set: &IndoorMaterials,
        materials: &impl super::preparation::AssetStore<StandardMaterial>,
        images: &impl super::preparation::AssetStore<Image>,
    ) -> Self {
        Self::from_manifest_excluding_humans(scene, set, materials, images, &[])
    }

    pub(crate) fn from_manifest_excluding_humans(
        scene: &IndoorManifest,
        set: &IndoorMaterials,
        materials: &impl super::preparation::AssetStore<StandardMaterial>,
        images: &impl super::preparation::AssetStore<Image>,
        moving_humans: &[usize],
    ) -> Self {
        let geometry = super::preparation::SceneGeometry {
            architecture: architecture::architecture(scene),
            objects: scene.objects.iter().map(objects::build_object).collect(),
            humans: scene
                .humans
                .iter()
                .map(super::humans::build_human)
                .collect(),
        };
        Self::from_geometry(scene, set, materials, images, moving_humans, &geometry)
    }

    pub(crate) fn from_geometry(
        scene: &IndoorManifest,
        set: &IndoorMaterials,
        materials: &impl super::preparation::AssetStore<StandardMaterial>,
        images: &impl super::preparation::AssetStore<Image>,
        moving_humans: &[usize],
        geometry: &super::preparation::SceneGeometry,
    ) -> Self {
        let started = Instant::now();
        let mut result = Self {
            triangles: Vec::new(),
            nodes: Vec::new(),
            materials: Vec::new(),
            lights: Vec::new(),
            sun_direction: architecture::sun_direction(scene),
            sun: linear(architecture::sun_color(scene)) * architecture::sun_illuminance(scene),
            // Isotropic hemispherical sky luminance. Sun is a separate analytic
            // source; its disk must not be integrated a second time here.
            sky: scene.sky_radiance(),
            bounds_min: Vec3::new(
                -scene.room_size.x * 0.5 - 0.10,
                scene.envelope.as_ref().map_or(0., |e| e.minimum_floor()) - 0.10,
                -scene.room_size.z * 0.5 - 0.10,
            ),
            bounds_max: Vec3::new(
                scene.room_size.x * 0.5 + 0.10,
                scene.room_size.y + 0.10,
                scene.room_size.z * 0.5 + super::layout::NEIGHBOR_DEPTH + 0.10,
            ),
            preparation_ms: 0.0,
            world_rotation: Quat::from_rotation_y(scene.world_yaw),
        };
        let mut finish_indices = std::collections::BTreeMap::new();
        let handles = super::materials::program::SURFACES
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
            let texture = mat
                .base_color_texture
                .as_ref()
                .and_then(|h| images.get(h))
                .and_then(|image| {
                    image.data.as_ref().map(|data| {
                        // Downsample once, not per path; only low-frequency diffuse
                        // reflectance participates in this GI approximation.
                        let width = image.width() as usize;
                        let n = 32usize;
                        let mut pixels = Vec::with_capacity(n * n);
                        for y in 0..n {
                            for x in 0..n {
                                let index = ((y * width / n) * width + x * width / n) * 4;
                                pixels.push(linear(Color::srgb(
                                    data[index] as f32 / 255.0,
                                    data[index + 1] as f32 / 255.0,
                                    data[index + 2] as f32 / 255.0,
                                )));
                            }
                        }
                        (n as u32, pixels)
                    })
                });
            result.materials.push(DiffuseMaterial {
                albedo: (linear(mat.base_color) * (1.0 - mat.metallic))
                    .clamp(Vec3::ZERO, Vec3::splat(0.95)),
                emission: Vec3::new(mat.emissive.red, mat.emissive.green, mat.emissive.blue),
                uv_scale: mat.uv_transform.matrix2 * Vec2::ONE,
                texture,
                textured_emission: mat.emissive_texture.is_some(),
            });
        }
        result.add_assembly(&geometry.architecture, Transform::IDENTITY, &finish_indices);
        for (object, assembly) in scene.objects.iter().zip(&geometry.objects) {
            result.add_assembly(assembly, object.transform(), &finish_indices);
        }
        for (person, assembly) in scene
            .humans
            .iter()
            .zip(&geometry.humans)
            .filter(|(p, _)| !moving_humans.contains(&p.id))
        {
            for (&surface, geometry) in &assembly.parts {
                use super::humans::HumanSurface;
                // The diffuse proxy has no thin-lens transmission model.
                // Clear spectacles must not become opaque eye shadow casters.
                if surface == HumanSurface::Lens {
                    continue;
                }
                let cloth = matches!(
                    surface,
                    HumanSurface::Top | HumanSurface::Trousers | HumanSurface::Shirt
                );
                let mut material = if cloth {
                    result.materials[Surface::Fabric as usize].clone()
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
                let index = result.materials.len();
                result.materials.push(material);
                result.add_geometry(geometry, person.transform(), index);
            }
        }
        for (i, p) in architecture::fixture_positions(scene)
            .into_iter()
            .enumerate()
        {
            let (c, lumens) = architecture::fixture_photometry(scene, i);
            let (inner, outer) = architecture::fixture_angles(scene, i);
            result.lights.push(LocalLight {
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
            result.lights.push(LocalLight {
                position: lamp.position + Vec3::Y * (lamp.size.y - 0.22),
                color: linear(Color::srgb(1.0, 0.78, 0.57)),
                candela: architecture::floor_lamp_lumens(scene, lamp.seed) / (4.0 * PI),
                range: 5.0,
                spot: false,
                inner_cos: 1.0,
                outer_cos: 0.0,
            });
        }
        result.build_bvh();
        result.preparation_ms = started.elapsed().as_secs_f64() * 1000.0;
        result
    }

    fn add_assembly(
        &mut self,
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
                Surface::Glass
                    | Surface::GlassInterior
                    | Surface::ContainerGlass
                    | Surface::Liquid
                    | Surface::Light
            ) {
                continue;
            }
            let material = label
                .rsplit_once("#finish")
                .and_then(|(_, slot)| slot.parse::<usize>().ok())
                .and_then(|slot| finishes.get(&(surface, slot)))
                .copied()
                .unwrap_or(surface as usize);
            self.add_geometry(geometry, transform, material);
        }
    }

    fn add_geometry(
        &mut self,
        geometry: &super::geometry::Geometry,
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
        self.triangles.reserve(geometry.indices.len() / 3);
        for i in geometry.indices.as_chunks::<3>().0 {
            let a = positions[i[0] as usize];
            let b = positions[i[1] as usize];
            let c = positions[i[2] as usize];
            let normal = (b - a).cross(c - a).normalize_or_zero();
            if normal.length_squared() < 0.5 {
                continue;
            }
            self.triangles.push(Triangle {
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

    fn build_bvh(&mut self) {
        self.nodes = bvh::build_probe_tree(&mut self.triangles);
    }

    fn hit(
        &self,
        origin: Vec3,
        direction: Vec3,
        mut max: f32,
        any: bool,
    ) -> Option<(usize, f32, Vec2)> {
        if self.nodes.is_empty() {
            return None;
        }
        let inverse = direction.recip();
        let mut stack = [0usize; 64];
        let mut len = 1;
        let mut result = None;
        while len > 0 {
            len -= 1;
            let index = stack[len];
            let node = &self.nodes[index];
            if !node.intersects(origin, inverse, max) {
                continue;
            }
            if node.count > 0 {
                for (offset, triangle) in self.triangles[node.start..node.start + node.count]
                    .iter()
                    .enumerate()
                {
                    let i = node.start + offset;
                    if let Some((distance, uv)) = triangle.hit(origin, direction, max) {
                        if any {
                            return Some((i, distance, uv));
                        }
                        max = distance;
                        result = Some((i, distance, uv));
                    }
                }
            } else {
                // Closest near hits clip traversal of the far half; shadow
                // queries can return as soon as any occluder is encountered.
                let (near, far) = if direction[node.axis] >= 0.0 {
                    (index + 1, node.right)
                } else {
                    (node.right, index + 1)
                };
                stack[len] = far;
                stack[len + 1] = near;
                len += 2;
            }
        }
        result
    }

    fn direct(&self, position: Vec3, normal: Vec3) -> Vec3 {
        let mut light = Vec3::ZERO;
        let origin = position + normal * 0.003;
        let sun_cosine = normal.dot(self.sun_direction).max(0.0);
        if sun_cosine > 0.0 && self.hit(origin, self.sun_direction, 1000.0, true).is_none() {
            light += self.sun * sun_cosine;
        }
        for source in &self.lights {
            let delta = source.position - position;
            let d2 = delta.length_squared();
            let d = d2.sqrt();
            let direction = delta / d;
            let cosine = normal.dot(direction).max(0.0);
            if cosine <= 0.0 || d >= source.range {
                continue;
            }
            let spot = if source.spot {
                ((direction.y - source.outer_cos) / (source.inner_cos - source.outer_cos))
                    .clamp(0.0, 1.0)
                    .powi(2)
            } else {
                1.0
            };
            if spot <= 0.0
                || self
                    .hit(origin, direction, (d - 0.01).max(0.0), true)
                    .is_some()
            {
                continue;
            }
            let attenuation =
                (1.0 - (d2 / source.range.powi(2)).powi(2)).max(0.0).powi(2) / d2.max(0.0001);
            light += source.color * (source.candela * cosine * spot * attenuation);
        }
        light
    }

    fn radiance(&self, mut origin: Vec3, mut direction: Vec3, bounces: u32, rng: &mut Rng) -> Vec3 {
        let mut throughput = Vec3::ONE;
        let mut light = Vec3::ZERO;
        for _ in 0..bounces {
            let Some((index, distance, bary)) = self.hit(origin, direction, 1000.0, false) else {
                // Only upper-hemisphere sky is an emitter. Geometry (including
                // the generated exterior ground/facades) handles the lower half.
                if direction.y > 0.0 {
                    light += throughput * self.sky;
                }
                break;
            };
            let triangle = &self.triangles[index];
            let normal = if triangle.normal.dot(direction) > 0.0 {
                -triangle.normal
            } else {
                triangle.normal
            };
            let position = origin + direction * distance;
            let mat = &self.materials[triangle.material];
            let uv = triangle.uv[0] * (1.0 - bary.x - bary.y)
                + triangle.uv[1] * bary.x
                + triangle.uv[2] * bary.y;
            let albedo = mat.albedo(uv);
            let emission = if mat.textured_emission {
                mat.emission * albedo
            } else {
                mat.emission
            };
            light += throughput * (emission + albedo * self.direct(position, normal) / PI);
            throughput *= albedo;
            if throughput.max_element() < 0.001 {
                break;
            }
            origin = position + normal * 0.003;
            direction = cosine_direction(normal, rng.next(), rng.next());
        }
        light
    }

    /// Irradiance / π for six cardinal normals. One spherical sample contributes
    /// to three positive cosine lobes, preserving spectral energy without a fit.
    pub fn integrate(&self, position: Vec3, samples: u32, bounces: u32, seed: u64) -> [Vec3; 6] {
        let mut result = [Vec3::ZERO; 6];
        let mut rng = Rng(seed ^ 0x9e3779b97f4a7c15);
        let phase = rng.next() * TAU;
        for i in 0..samples {
            let y = 1.0 - 2.0 * (i as f32 + 0.5) / samples as f32;
            let phi = i as f32 * 2.399_963_1 + phase;
            let r = (1.0 - y * y).sqrt();
            let direction = Vec3::new(r * phi.cos(), y, r * phi.sin());
            let light =
                self.radiance(position, direction, bounces, &mut rng) * (4.0 / samples as f32);
            // Bevy transforms probe positions, but its lobe weights use world
            // normals. Store world-oriented lobes even when the grid rotates.
            let direction = self.world_rotation * direction;
            for axis in 0..3 {
                result[axis * 2 + usize::from(direction[axis] < 0.0)] +=
                    light * direction[axis].abs();
            }
        }
        result
    }

    fn inside_solid(&self, point: Vec3) -> bool {
        let mut backfaces = 0;
        for direction in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z] {
            if let Some((index, _, _)) = self.hit(point, direction, 1000.0, false) {
                if self.triangles[index].normal.dot(direction) > 0.0 {
                    backfaces += 1;
                }
            }
        }
        backfaces >= 4
    }

    pub fn bake(&self, settings: BakeSettings, seed: u64) -> ProbeData {
        assert!(settings.spacing.is_finite() && settings.spacing >= 0.2);
        assert!(settings.rays_per_probe >= 16 && (1..=32).contains(&settings.diffuse_bounces));
        let started = Instant::now();
        let resolution = ((self.bounds_max - self.bounds_min) / settings.spacing)
            .ceil()
            .as_uvec3()
            .max(UVec3::splat(2));
        let count = (resolution.x * resolution.y * resolution.z) as usize;
        assert!(count <= 100_000, "irradiance probe budget exceeded");
        let point = |i: usize| {
            let i = i as u32;
            let xyz = UVec3::new(
                i % resolution.x,
                (i / resolution.x) % resolution.y,
                i / (resolution.x * resolution.y),
            );
            self.bounds_min
                + (xyz.as_vec3() + Vec3::splat(0.5)) / resolution.as_vec3()
                    * (self.bounds_max - self.bounds_min)
        };
        let mut values = vec![[Vec3::ZERO; 6]; count];
        let valid: Vec<bool> = (0..count).map(|i| !self.inside_solid(point(i))).collect();
        let calculate = |i: usize| {
            if valid[i] {
                self.integrate(
                    point(i),
                    settings.rays_per_probe,
                    settings.diffuse_bounces,
                    seed.wrapping_add(i as u64 * 7919),
                )
            } else {
                [Vec3::ZERO; 6]
            }
        };
        #[cfg(not(target_arch = "wasm32"))]
        {
            let workers = settings
                .workers
                .max(1)
                .min(std::thread::available_parallelism().map_or(1, usize::from))
                .min(count);
            std::thread::scope(|scope| {
                for (chunk, output) in values.chunks_mut(count.div_ceil(workers)).enumerate() {
                    let calculate = &calculate;
                    scope.spawn(move || {
                        for (j, value) in output.iter_mut().enumerate() {
                            *value = calculate(chunk * count.div_ceil(workers) + j);
                        }
                    });
                }
            });
        }
        #[cfg(target_arch = "wasm32")]
        for (i, value) in values.iter_mut().enumerate() {
            *value = calculate(i);
        }
        let mut relocated = 0;
        // Interior probes are never allowed to darken/emit from inside a solid.
        // A nearest valid sample is used at the original grid location; this is
        // bounded spatial extrapolation, not visibility-aware probe interpolation.
        for i in 0..count {
            if !valid[i] {
                if let Some(nearest) = (0..count).filter(|&j| valid[j]).min_by(|&a, &b| {
                    point(a)
                        .distance_squared(point(i))
                        .total_cmp(&point(b).distance_squared(point(i)))
                }) {
                    values[i] = values[nearest];
                    relocated += 1;
                }
            }
        }
        let mean = values.iter().flatten().copied().sum::<Vec3>() / (count * 6) as f32;
        ProbeData {
            resolution,
            bounds_min: self.bounds_min,
            bounds_max: self.bounds_max,
            values,
            statistics: BakeStatistics {
                backend: "cpu_bvh".into(),
                triangles: self.triangles.len(),
                probes: count,
                relocated_probes: relocated,
                primary_rays: ((count - relocated) as u64) * settings.rays_per_probe as u64,
                diffuse_bounces: settings.diffuse_bounces,
                texture_bytes: count * 6 * 8,
                preparation_ms: self.preparation_ms,
                bake_ms: Some(started.elapsed().as_secs_f64() * 1000.0),
                mean_radiance: Some(mean.to_array()),
                transport_bytes: self.triangles.len() * std::mem::size_of::<Triangle>()
                    + self.nodes.len() * std::mem::size_of::<Node>(),
                prefetch_hit: false,
                prefetch_wait_ms: 0.0,
            },
        }
    }
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(2685821657736338717) >> 40) as f32 / (1u32 << 24) as f32
    }
}

fn cosine_direction(normal: Vec3, u: f32, v: f32) -> Vec3 {
    let tangent = normal.any_orthonormal_vector();
    let bitangent = normal.cross(tangent);
    let radius = u.sqrt();
    let angle = v * TAU;
    tangent * (radius * angle.cos())
        + bitangent * (radius * angle.sin())
        + normal * (1.0 - u).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motion_candidates_do_not_leave_static_gi_casters() {
        let scene = IndoorManifest::generate_with_humans(
            0,
            super::super::layout::IndoorLayout::Mixed,
            0.35,
            0,
            0.7,
        )
        .unwrap();
        assert!(scene.humans.len() > 1);
        let mut images = Assets::default();
        let mut materials = Assets::default();
        let set = IndoorMaterials::build(&scene, &mut images, &mut materials);
        let fixed = BakeScene::from_manifest(&scene, &set, &materials, &images);
        let moving = BakeScene::from_manifest_excluding_humans(
            &scene,
            &set,
            &materials,
            &images,
            &[scene.humans[0].id],
        );
        let all_ids: Vec<_> = scene.humans.iter().map(|h| h.id).collect();
        let empty =
            BakeScene::from_manifest_excluding_humans(&scene, &set, &materials, &images, &all_ids);
        assert!(empty.triangles.len() < moving.triangles.len());
        assert!(moving.triangles.len() < fixed.triangles.len());
        assert_eq!(moving.lights.len(), fixed.lights.len());
        assert_eq!(moving.bounds_min, fixed.bounds_min);
        assert_eq!(moving.bounds_max, fixed.bounds_max);
    }

    fn empty() -> BakeScene {
        BakeScene {
            triangles: Vec::new(),
            nodes: Vec::new(),
            materials: Vec::new(),
            lights: Vec::new(),
            sun_direction: Vec3::Y,
            sun: Vec3::ZERO,
            sky: Vec3::ONE,
            bounds_min: Vec3::splat(-1.0),
            bounds_max: Vec3::ONE,
            preparation_ms: 0.0,
            world_rotation: Quat::IDENTITY,
        }
    }

    #[test]
    fn hemispherical_sky_matches_analytic_cosine_integrals() {
        let scene = empty();
        let values = scene.integrate(Vec3::ZERO, 16384, 4, 1);
        for (value, expected) in values.iter().zip([0.5, 0.5, 1.0, 0.0, 0.5, 0.5]) {
            assert!(
                (*value - Vec3::splat(expected)).abs().max_element() < 0.002,
                "{value:?} != {expected}"
            );
        }
    }

    #[test]
    fn diffuse_cavity_matches_independent_geometric_series_energy() {
        let mut scene = empty();
        scene.materials.push(DiffuseMaterial {
            albedo: Vec3::new(0.5, 0.25, 0.0),
            emission: Vec3::ONE,
            uv_scale: Vec2::ONE,
            texture: None,
            textured_emission: false,
        });
        let mut walls = Assembly::default();
        walls.box_part(Surface::Paint, "wall", Vec3::ZERO, Vec3::splat(4.0), 0.0);
        scene.add_assembly(&walls, Transform::IDENTITY, &Default::default());
        scene.build_bvh();
        for bounces in [1, 2, 4, 8] {
            let values = scene.integrate(Vec3::ZERO, 2048, bounces, 4);
            let expected = Vec3::new(
                (1.0 - 0.5_f32.powi(bounces as i32)) / 0.5,
                (1.0 - 0.25_f32.powi(bounces as i32)) / 0.75,
                1.0,
            );
            for value in values {
                assert!(
                    (value - expected).abs().max_element() < 0.012,
                    "{bounces} bounces: {value:?} != {expected:?}"
                );
            }
        }
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn bvh_matches_brute_force_triangle_intersections_and_shadows() {
        let mut scene = empty();
        for x in -2..3 {
            for z in -2..3 {
                let mut assembly = Assembly::default();
                assembly.box_part(
                    Surface::Paint,
                    "box",
                    Vec3::new(x as f32, 0.0, z as f32),
                    Vec3::splat(0.6),
                    0.0,
                );
                scene.add_assembly(&assembly, Transform::IDENTITY, &Default::default());
            }
        }
        for build in [bvh::build_probe_tree, bvh::build] {
            scene.nodes = build(&mut scene.triangles);
            let mut rng = Rng(917);
            for _ in 0..1024 {
                let origin = Vec3::new((rng.next() - 0.5) * 8.0, 2.0, (rng.next() - 0.5) * 8.0);
                let direction = Vec3::new(rng.next() - 0.5, -1.0, rng.next() - 0.5).normalize();
                let brute = scene
                    .triangles
                    .iter()
                    .filter_map(|t| t.hit(origin, direction, 100.0))
                    .map(|h| h.0)
                    .min_by(f32::total_cmp);
                let accelerated = scene.hit(origin, direction, 100.0, false).map(|h| h.1);
                assert_eq!(brute.is_some(), accelerated.is_some());
                assert_eq!(
                    brute.is_some(),
                    scene.hit(origin, direction, 100.0, true).is_some()
                );
                if let (Some(a), Some(b)) = (brute, accelerated) {
                    assert!((a - b).abs() < 1e-5);
                }
            }
            assert!(scene.inside_solid(Vec3::ZERO));
            assert!(!scene.inside_solid(Vec3::Y));
        }
    }

    #[test]
    fn rgba16_volume_packing_matches_bevy_shader_lobes() {
        let values = std::array::from_fn(|i| Vec3::splat((i + 1) as f32));
        let data = ProbeData {
            resolution: UVec3::splat(2),
            bounds_min: Vec3::ZERO,
            bounds_max: Vec3::ONE,
            values: vec![values; 8],
            statistics: default(),
        };
        let image = data.image();
        let bytes = image.data.unwrap();
        for (face, normal) in [Vec3::X, -Vec3::X, Vec3::Y, -Vec3::Y, Vec3::Z, -Vec3::Z]
            .iter()
            .enumerate()
        {
            let offset = ((face / 2 * 2) * 4 * 2 + (face % 2 * 2) * 2) * 8;
            let value =
                half::f16::from_bits(u16::from_le_bytes([bytes[offset], bytes[offset + 1]]))
                    .to_f32();
            assert_eq!(value, (face + 1) as f32);
            assert_eq!(data.sample(Vec3::splat(0.5), *normal), Vec3::splat(value));
        }
    }

    #[test]
    fn shadow_visibility_and_rotated_lobes_follow_geometry() {
        let mut scene = empty();
        scene.sky = Vec3::ZERO;
        scene.sun = Vec3::ONE;
        assert_eq!(scene.direct(Vec3::ZERO, Vec3::Y), Vec3::ONE);
        let mut blocker = Assembly::default();
        blocker.box_part(
            Surface::Paint,
            "blocker",
            Vec3::Y,
            Vec3::new(4.0, 0.1, 4.0),
            0.0,
        );
        scene.add_assembly(&blocker, Transform::IDENTITY, &Default::default());
        scene.build_bvh();
        assert_eq!(scene.direct(Vec3::ZERO, Vec3::Y), Vec3::ZERO);

        scene.triangles.clear();
        scene.nodes.clear();
        scene.sun = Vec3::ZERO;
        scene.materials.push(DiffuseMaterial {
            albedo: Vec3::ZERO,
            emission: Vec3::X,
            uv_scale: Vec2::ONE,
            texture: None,
            textured_emission: false,
        });
        let mut panel = Assembly::default();
        panel.box_part(
            Surface::Paint,
            "emitter",
            Vec3::X * 2.0,
            Vec3::new(0.1, 4.0, 4.0),
            0.0,
        );
        scene.add_assembly(&panel, Transform::IDENTITY, &Default::default());
        scene.build_bvh();
        let local = scene.integrate(Vec3::ZERO, 8192, 1, 7);
        scene.world_rotation = Quat::from_rotation_y(PI * 0.5);
        let world = scene.integrate(Vec3::ZERO, 8192, 1, 7);
        assert!((local[0] - world[5]).length() < 0.001);
        assert!((local[1] - world[4]).length() < 0.001);
        assert!((local[2] - world[2]).length() < 0.001);
        assert!(world[5].x > 0.4 && world[4].x < 0.001);
    }

    #[test]
    #[ignore = "bounded CPU transport benchmark and high-sample comparison artifact"]
    fn export_indoor_gi_reference() {
        use super::super::layout::IndoorLayout;
        let scene = IndoorManifest::generate(6, IndoorLayout::OpenOffice, 0.6, 2).unwrap();
        let mut images = Assets::default();
        let mut materials = Assets::default();
        let set = IndoorMaterials::build(&scene, &mut images, &mut materials);
        let transport = BakeScene::from_manifest(&scene, &set, &materials, &images);
        let probes = transport.bake(BakeSettings::default(), scene.seed);
        let mut errors = Vec::new();
        for point in [
            Vec3::new(0.0, 0.35, 0.0),
            Vec3::new(0.0, 1.5, 0.0),
            Vec3::new(0.0, scene.room_size.y - 0.2, 0.0),
            Vec3::new(-scene.room_size.x * 0.4, 1.5, 0.0),
        ] {
            let low = transport.integrate(point, 256, 3, 61);
            let reference = transport.integrate(point, 16384, 8, 9741);
            let mean_error = low
                .iter()
                .zip(reference)
                .map(|(a, b)| (*a - b).abs().element_sum())
                .sum::<f32>()
                / 18.0;
            let mean_reference = reference.iter().map(|a| a.element_sum()).sum::<f32>() / 18.0;
            errors.push(
                serde_json::json!({"point":point.to_array(),"low":low.map(|v|v.to_array()),
                "reference":reference.map(|v|v.to_array()),"mean_absolute_error_cd_m2":mean_error,
                "relative_mean_absolute_error":mean_error/mean_reference.max(0.001)}),
            );
        }
        std::fs::create_dir_all("out/indoor_gi").unwrap();
        std::fs::write("out/indoor_gi/reference.json",serde_json::to_vec_pretty(&serde_json::json!({
            "seed":scene.seed,"settings":BakeSettings::default(),"statistics":probes.statistics,
            "reference_note":"Same triangle/light transport integrator, independent sequence, 16384 rays and 8 bounces; analytical controls are separate unit tests.",
            "points":errors})).unwrap()).unwrap();
        eprintln!("GI statistics: {:?}", probes.statistics);
    }
}
