use std::collections::HashSet;

use bevy::{
    prelude::*,
    // asset::{load_internal_asset, weak_handle},
    pbr::{ExtendedMaterial, MaterialExtension},
    render::{
        mesh::{
            Indices,
            Mesh,
            VertexAttributeValues,
        },
        // render_asset::RenderAssetUsages,
        render_resource::*,
        // storage::ShaderStorageBuffer,
    },
};
use bevy_gaussian_splatting::{
    CloudSettings,
    Gaussian3d,
    GaussianSplattingPlugin,
    gaussian::f32::{
        PositionVisibility,
        ScaleOpacity,
    },
    material::spherical_harmonics::SH_COEFF_COUNT,
    PlanarGaussian3d,
    PlanarGaussian3dHandle,
    SphericalHarmonicCoefficients,
};
use bytemuck::{
    Pod,
    Zeroable,
};
use rand::{
    prelude::{Rng, SliceRandom, thread_rng},
    rngs::SmallRng,
    SeedableRng,
};

use rayon::prelude::*;



#[derive(Component, Clone, Debug, Default, Reflect)]
#[reflect(Component, Default)]
pub struct SdfRoot;


#[derive(Resource, Reflect, Debug, Clone)]
#[reflect(Resource, Default)]
pub struct SdfConfig {
    pub per_m2: f32,
    pub band: f32,
    pub jitter: f32,
    pub min_pts: u32,
    pub max_pts: u32,
    pub near_z: f32,
    pub far_z: f32,
}

impl Default for SdfConfig {
    fn default() -> Self {
        Self {
            per_m2: 10.0,
            band: 0.01,
            jitter: 0.05,
            min_pts: 256,
            max_pts: 1024,
            near_z: 0.1,
            far_z: 25.0,
        }
    }
}


#[derive(Clone, Copy, Debug, Default, Reflect, ShaderType, Pod, Zeroable)]
#[repr(C)]
struct GpuPt {
    pos_dist: Vec4,
}

#[derive(Clone, Copy, Debug, Default, Reflect, ShaderType, Pod, Zeroable)]
#[repr(C)]
struct LodParams {
    counts: UVec4,
    range: Vec4,
}

// #[derive(Clone, Copy, Debug, Default, Reflect, ShaderType, Pod, Zeroable)]
// #[repr(C)]
// pub struct KdNodeGpu {
//     min_split: Vec4,
//     max_pad: Vec4,
//     axis: u32,
//     left: u32,
//     right: u32,
//     _pad: u32,
// }

// #[derive(Resource, Default, Clone)]
// pub struct SdfBuffers {
//     pub pts: Handle<ShaderStorageBuffer>,
//     pub kd: Handle<ShaderStorageBuffer>,
// }


// pub const SDF_SHADER_HANDLE: Handle<Shader> = weak_handle!("01b0a4af-31b4-4a71-a6bc-045e8778e6d4");

pub type SdfMaterial = ExtendedMaterial<StandardMaterial, SdfExtension>;


#[derive(Asset, AsBindGroup, TypePath, Debug, Clone)]
pub struct SdfExtension {
//     #[storage(14, read_only)] pts: Handle<ShaderStorageBuffer>,
//     #[storage(15, read_only)] kd: Handle<ShaderStorageBuffer>,
    #[uniform(16)] lod: LodParams,
}

impl Default for SdfExtension {
    fn default() -> Self {
        Self {
            // pts: Handle::default(),
            // kd: Handle::default(),
            lod: LodParams::default(),
        }
    }
}

impl MaterialExtension for SdfExtension {
    // fn fragment_shader() -> ShaderRef { SDF_SHADER_HANDLE.into() }
}


#[derive(Component, Debug, Clone, Default, Reflect, Eq, PartialEq)]
#[reflect(Component, Default)]
pub struct Sdf;


#[derive(Component, Debug, Clone, Default, Reflect, Eq, PartialEq)]
#[reflect(Component, Default)]
pub struct ComputeSdf;


// TODO: cache across mesh instances?
#[derive(Component, Debug, Default, Reflect)]
#[reflect(Component)]
struct MeshSdfCache {
    pts: Vec<GpuPt>,
    dirty: bool,
}


#[derive(Resource, Default)]
struct AggregateSdf {
    pts: Vec<GpuPt>,
    // kd: Vec<KdNodeGpu>,
    dirty: bool,
}


pub struct SdfPlugin;
impl Plugin for SdfPlugin {
    fn build(&self, app: &mut App) {
        // load_internal_asset!(
        //     app,
        //     SDF_SHADER_HANDLE,
        //     "sdf.wgsl",
        //     Shader::from_wgsl,
        // );

        app
            .register_type::<MeshSdfCache>()
            .register_type::<ComputeSdf>()
            .register_type::<Sdf>()
            .register_type::<SdfRoot>()
            .register_type::<SdfConfig>()
            .init_resource::<SdfConfig>()
            // .init_resource::<SdfBuffers>()
            .init_resource::<AggregateSdf>()
            .add_plugins((
                MaterialPlugin::<SdfMaterial>::default(),
            ))
            .add_plugins(GaussianSplattingPlugin)
            .add_systems(PreUpdate, propagate_sdf_tags)
            .add_systems(Update, cache_mesh_sdf)
            .add_systems(PostUpdate, aggregate_sdf_system)
            .add_systems(PreUpdate, upload_sdf)
            .add_systems(Update, apply_sdf_material);
    }
}



pub fn propagate_sdf_tags(
    mut commands: Commands,
    roots: Query<Entity, With<SdfRoot>>,
    has_sdf: Query<Entity, With<Sdf>>,
    parents: Query<&ChildOf>,
) {
    let root_set: HashSet<Entity> = roots.iter().collect();

    for entity in has_sdf.iter() {
        let is_descendant = parents
            .iter_ancestors(entity)
            .any(|ancestor| root_set.contains(&ancestor));

        if is_descendant {
            if let Ok(mut valid_mesh) = commands.get_entity(entity) {
                valid_mesh.try_insert(ComputeSdf);
            }
        }
    }
}


fn cache_mesh_sdf(
    mut commands: Commands,
    cfg: Res<SdfConfig>,
    meshes: Res<Assets<Mesh>>,
    mut q: Query<
        (
            Entity,
            &Mesh3d,
            &GlobalTransform,
            Option<&mut MeshSdfCache>,
        ),
        (
            With<ComputeSdf>,
            // Changed<Mesh3d>,
            // Changed<GlobalTransform>,
            Without<MeshSdfCache>,
        ),
    >,
    // time: Res<Time>,
    mut agg: ResMut<AggregateSdf>,
    mut rng: Local<Option<SmallRng>>,
) {
    if rng.is_none() { *rng = SmallRng::from_entropy().into(); }
    let rng = rng.as_mut().unwrap();

    if cfg.is_changed() {
        // TODO: clear sdf caches
    }

    for (e, mesh_handle, tf, maybe_cache) in &mut q {
        let Some(mesh) = meshes.get(mesh_handle) else { continue };

        let pts = compute_sparse_sdf_for_mesh(
            mesh,
            tf,
            &cfg,
            &mut *rng,
        );

        match maybe_cache {
            Some(mut cache) => {
                info!("updating cached {} SDF points for mesh {:?}", pts.len(), e);
                cache.pts = pts;
                cache.dirty = true;
            }
            None => {
                info!("cached {} SDF points for mesh {:?}", pts.len(), e);
                commands
                    .entity(e)
                    .insert(MeshSdfCache {
                        pts,
                        dirty: true,
                    });
            }
        }

        agg.dirty = true;
    }
}


fn apply_sdf_material(
    mut commands: Commands,
    tagged: Query<
        Entity,
        (
            With<ComputeSdf>,
            Without<MeshMaterial3d<SdfMaterial>>,
        ),
    >,
    mut removed: RemovedComponents<ComputeSdf>,
    mut mats: ResMut<Assets<SdfMaterial>>,
    // bufs: Res<SdfBuffers>,
    cfg: Res<SdfConfig>,
) {
    for e in removed.read() {
        if let Ok(mut ent) = commands.get_entity(e) {
            ent.remove::<MeshMaterial3d<SdfMaterial>>();
        }
    }

    let lod = LodParams {
        counts: UVec4::new(cfg.max_pts, cfg.min_pts, 0, 0),
        range : Vec4::new(cfg.near_z , cfg.far_z, 0.0, 0.0),
    };

    for e in &tagged {
        let mat = mats.add(SdfMaterial {
            base: StandardMaterial {
                alpha_mode: AlphaMode::Blend,
                base_color: Color::srgba(0.4, 0.4, 0.4, 0.05),
                cull_mode: None,
                unlit: true,
                ..default()
            },
            extension: SdfExtension {
                // pts: bufs.pts.clone(),
                // kd: bufs.kd.clone(),
                lod,
            },
        });

        commands
            .entity(e)
            .insert(MeshMaterial3d(mat));
    }
}


fn point_budget(area_sum: f32, cfg: &SdfConfig) -> (usize, u32) {
    let raw = (area_sum * cfg.per_m2 * 2.0).round() as u32;
    let want = raw.clamp(cfg.min_pts, cfg.max_pts);
    let per_tri = (want as f32 / (area_sum.max(1.0e-6) * 2.0)).ceil() as u32;
    (want as usize, per_tri)
}


#[inline]
pub fn compute_sparse_sdf_for_mesh(
    mesh: &Mesh,
    tf: &GlobalTransform,
    cfg: &SdfConfig,
    rng: &mut SmallRng,
) -> Vec<GpuPt> {
    let mat = tf.compute_matrix();
    let verts: Vec<Vec3> = match mesh.attribute(Mesh::ATTRIBUTE_POSITION) {
        Some(VertexAttributeValues::Float32x3(v)) =>
            v.iter().map(|&[x, y, z]| mat.transform_point3(Vec3::new(x, y, z))).collect(),
        Some(VertexAttributeValues::Float32x2(v)) =>
            v.iter().map(|&[x, y]| mat.transform_point3(Vec3::new(x, y, 0.0))).collect(),
        _ => return vec![],
    };

    let tris: Vec<[usize; 3]> = match mesh.indices() {
        Some(Indices::U32(ix)) => ix.chunks_exact(3).map(|t| [t[0] as usize, t[1] as usize, t[2] as usize]).collect(),
        Some(Indices::U16(ix)) => ix.chunks_exact(3).map(|t| [t[0] as usize, t[1] as usize, t[2] as usize]).collect(),
        None => (0..verts.len()).step_by(3).map(|i| [i, i + 1, i + 2]).collect(),
    };
    if tris.is_empty() { return vec![]; }

    #[inline] fn tri_area_normal(v0: Vec3, v1: Vec3, v2: Vec3) -> (f32, Vec3) {
        let n = (v1 - v0).cross(v2 - v0);
        (0.5 * n.length(), n.normalize())
    }

    let per_tri: Vec<(f32, Vec3)> = tris.par_iter()
        .map(|&[i0, i1, i2]| tri_area_normal(verts[i0], verts[i1], verts[i2]))
        .collect();

    let area_sum: f32 = per_tri.par_iter().map(|(a, _)| *a).sum();
    let (k_target, _) = point_budget(area_sum, cfg);
    if k_target == 0 { return vec![]; }

    let per_tri_samples: Vec<u32> = per_tri
        .iter()
        .map(|(a, _)| ((a / area_sum) * k_target as f32).ceil() as u32)
        .collect();

    let mut band: Vec<(Vec3, f32)> = tris.par_iter()
        .zip(per_tri.par_iter())
        .zip(per_tri_samples.par_iter())
        .flat_map_iter(|((&[i0, i1, i2], &(_area, n)), &n_samples)| {
            let mut thread_rng = SmallRng::from_rng(thread_rng()).unwrap();
            let mut local = Vec::with_capacity(n_samples as usize * 2);
            for _ in 0..n_samples {
                let (u, v): (f32, f32) = (thread_rng.gen(), thread_rng.gen());
                let b  = (1.0 - u).sqrt();
                let p = verts[i0] * (1.0 - b)
                       + verts[i1] * b * (1.0 - v)
                       + verts[i2] * b * v;

                let jitter = cfg.jitter * (thread_rng.gen::<f32>() * 2.0 - 1.0);
                let band_r = cfg.band + jitter;

                let sign  = if thread_rng.gen_bool(0.5) { 1.0 } else { -1.0 };
                let dist  = thread_rng.gen::<f32>() * band_r;
                let offset = n * dist * sign;
                local.push((p + offset, dist * sign));
            }
            local
        })
        .collect();

    if band.is_empty() { return vec![]; }

    band[..].shuffle(rng);

    let mut selected = Vec::<(Vec3, f32)>::with_capacity(k_target);
    let mut remaining = band;
    let seed = remaining.swap_remove(rng.gen_range(0..remaining.len()));
    let (seed_p, seed_d) = seed;
    selected.push((seed_p, seed_d));

    let mut dists: Vec<f32> = remaining.iter()
        .map(|(q, _)| (*q - seed_p).length_squared())
        .collect();

    while selected.len() < k_target && !remaining.is_empty() {
        let best = dists
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let (p_new, d_new) = remaining.swap_remove(best);
        selected.push((p_new, d_new));
        dists.swap_remove(best);

        for (di, (q, _)) in remaining.iter().enumerate() {
            let new_d2 = (*q - p_new).length_squared();
            if new_d2 < dists[di] { dists[di] = new_d2; }
        }
    }

    selected.into_iter()
        .map(|(p, d)| GpuPt { pos_dist: p.extend(d) })
        .collect()
}


fn aggregate_sdf_system(
    mut agg: ResMut<AggregateSdf>,
    mut caches: Query<
        &mut MeshSdfCache,
    >,
) {
    let any_dirty = caches.iter().any(|c| c.dirty);
    if !any_dirty && !agg.dirty {
        return;
    }

    info!("aggregating SDF points from {} caches", caches.iter().count());

    agg.pts.clear();
    for mut c in &mut caches {
        agg.pts.extend(&c.pts);
        c.dirty = false;
    }

    { // note: for gaussian representation, no KD tree is needed
        // let kd = {
        //     let mut order = (0..agg.pts.len()).collect::<Vec<_>>();
        //     let mut kd = Vec::<KdNodeGpu>::new();
        //     agg.kd.clear();
        //     build_kd_nodes(&agg.pts, &mut order, &mut kd);
        //     kd
        // };

        // agg.kd = kd;
    }

    agg.dirty = true;
}


fn upload_sdf(
    mut commands: Commands,
    mut planar_gaussian_3d: ResMut<Assets<PlanarGaussian3d>>,
    // mut ssbos: ResMut<Assets<ShaderStorageBuffer>>,
    mut agg: ResMut<AggregateSdf>,
    // mut bufs: ResMut<SdfBuffers>,
    scene_root: Query<
        Entity,
        With<SdfRoot>,
    >,
) {
    if agg.pts.is_empty() || !agg.dirty {
        return;
    }

    info!("uploading {} SDF points to PlanarGaussian3d", agg.pts.len());

    // TODO: write gaussians in compute shader
    { // map sdf to gaussians
        const MIN_SCALE: f32 = 0.005;
        const MAX_SCALE: f32 = 0.02;

        let max_abs = agg.pts
            .iter()
            .map(|p| p.pos_dist.w.abs())
            .fold(1e-5f32, f32::max);

        let gaussians: Vec<Gaussian3d> = agg.pts
            .iter()
            .map(|pt| {
                let dist = pt.pos_dist.w;
                let signed_n = (dist / max_abs).clamp(-1.0, 1.0);
                let mag_n = signed_n.abs().sqrt();

                let s = MAX_SCALE - mag_n * (MAX_SCALE - MIN_SCALE);
                let opacity = 1.0 - mag_n;

                Gaussian3d {
                    rotation: [1.0, 0.0, 0.0, 0.0].into(),
                    position_visibility: PositionVisibility {
                        position: pt.pos_dist.truncate().into(),
                        visibility: 1.0,
                    },
                    scale_opacity: ScaleOpacity {
                        scale: [s, s, s].into(),
                        opacity,
                    },
                    spherical_harmonic: SphericalHarmonicCoefficients {
                        coefficients: {
                            let mut coefficients = [0.0; SH_COEFF_COUNT];
                            let r = dist.is_sign_positive() as u32 as f32;
                            coefficients[0] = r;
                            coefficients[1] = 0.3;
                            coefficients[2] = 1.0 - r;
                            coefficients
                        },
                    },
                }
            })
            .collect();
        let gaussians_handle = planar_gaussian_3d.add(PlanarGaussian3d::from(gaussians));

        commands
            .entity(scene_root.single().expect("SdfRoot required to attach gaussians"))
            .insert((
                CloudSettings::default(),
                PlanarGaussian3dHandle(gaussians_handle),
            ));

        // TODO: remove gaussians from scene root when leaving RenderMode::Sdf?
    }

    // {
    //     debug_assert!(
    //         !agg.kd.is_empty(),
    //         "aggregate_sdf: kd-tree is empty while pts are not"
    //     );

    //     debug_assert_eq!(
    //         agg.kd.iter().filter(|n| n.axis == 3).count(),
    //         agg.pts.len(),
    //         "kd leaves must equal point count"
    //     );

    //     for (i, n) in agg.kd.iter().enumerate() {
    //         debug_assert!(
    //             n.axis <= 3,
    //             "kd node {i}: axis out of range ({})",
    //             n.axis
    //         );

    //         if n.axis < 3 {
    //             debug_assert!(
    //                 (n.left == u32::MAX || (n.left as usize) < agg.kd.len())
    //                     && (n.right == u32::MAX || (n.right as usize) < agg.kd.len()),
    //                 "kd node {i}: child index out of bounds"
    //             );
    //         } else {
    //             let idx = n.min_split.w as usize;
    //             debug_assert!(
    //                 idx < agg.pts.len(),
    //                 "kd leaf {i}: point index {} out of bounds",
    //                 idx
    //             );
    //         }
    //     }
    // }

    // {
    //     let pts_buf = if let Some(buf) = ssbos.get_mut(&bufs.pts) {
    //         buf
    //     } else {
    //         bufs.pts = ssbos.add(ShaderStorageBuffer::new(&[], RenderAssetUsages::RENDER_WORLD));
    //         ssbos.get_mut(&bufs.pts).unwrap()
    //     };
    //     pts_buf.set_data(&agg.pts);
    // }

    // {
    //     let kd_buf = if let Some(buf) = ssbos.get_mut(&bufs.kd) {
    //         buf
    //     } else {
    //         bufs.kd = ssbos.add(ShaderStorageBuffer::new(&[], RenderAssetUsages::RENDER_WORLD));
    //         ssbos.get_mut(&bufs.kd).unwrap()
    //     };
    //     kd_buf.set_data(&agg.kd);
    // }

    agg.dirty = false;
}



// trait HasPos {
//     fn pos(&self) -> Vec3;
// }

// impl HasPos for GpuPt {
//     #[inline] fn pos(&self) -> Vec3 {
//         self.pos_dist.truncate()
//     }
// }

// impl HasPos for (Vec3, f32) {
//     #[inline] fn pos(&self) -> Vec3 {
//         self.0
//     }
// }


// fn build_kd_nodes<T: HasPos>(
//     pts: &[T],
//     idxs: &mut [usize],
//     out: &mut Vec<KdNodeGpu>,
// ) -> u32 {
//     match idxs.len() {
//         0 => return u32::MAX,
//         1 => {
//             let p = pts[idxs[0]].pos();
//             let me = out.len() as u32;
//             out.push(KdNodeGpu {
//                 min_split: p.extend(idxs[0] as f32),
//                 max_pad: p.extend(0.0),
//                 axis: 3,
//                 left: u32::MAX,
//                 right: u32::MAX,
//                 _pad: u32::MAX,
//             });
//             return me;
//         }
//         _ => {}
//     }

//     let (mut min, mut max) = {
//         let p0 = pts[idxs[0]].pos();
//         (p0, p0)
//     };

//     for &i in idxs.iter().skip(1) {
//         let p = pts[i].pos();
//         min = min.min(p);
//         max = max.max(p);
//     }

//     let delta = max - min;
//     let axis = if delta.x >= delta.y && delta.x >= delta.z { 0 }
//                 else if delta.y >= delta.z { 1 } else { 2 };

//     idxs.sort_by(|&a, &b| pts[a]
//         .pos()[axis]
//         .partial_cmp(&pts[b].pos()[axis])
//         .unwrap()
//     );
//     let mid = idxs.len() / 2;

//     let me = out.len() as u32;
//     out.push(KdNodeGpu {
//         min_split: min.extend(pts[idxs[mid]].pos()[axis]),
//         max_pad: max.extend(0.0),
//         axis: axis as u32,
//         left: u32::MAX,
//         right: u32::MAX,
//         _pad: u32::MAX,
//     });

//     let (left, right) = idxs.split_at_mut(mid);
//     if !left.is_empty() { out[me as usize].left = build_kd_nodes(pts, left,  out); }
//     if !right.is_empty() { out[me as usize].right = build_kd_nodes(pts, right, out); }
//     me
// }
