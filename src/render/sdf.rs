use std::collections::{HashMap, HashSet};

use bevy::{
    prelude::*,
    pbr::{ExtendedMaterial, MaterialExtension},
    render::{
        mesh::{
            Indices,
            Mesh,
            VertexAttributeValues,
        },
        render_resource::*,
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


// TODO: python api
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
    pub gaussian_visualization: bool,
    pub dense_sampling: bool,
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
            gaussian_visualization: true,
            dense_sampling: false,
        }
    }
}


#[derive(Clone, Copy, Debug, Default, Reflect, ShaderType, Pod, Zeroable)]
#[repr(C)]
pub struct GpuPt {
    pub pos_dist: Vec4,
}

#[derive(Clone, Copy, Debug, Default, Reflect, ShaderType, Pod, Zeroable)]
#[repr(C)]
struct LodParams {
    counts: UVec4,
    range: Vec4,
}

pub type SdfMaterial = ExtendedMaterial<StandardMaterial, SdfExtension>;


#[derive(Asset, AsBindGroup, TypePath, Debug, Default, Clone)]
pub struct SdfExtension {
    #[uniform(16)] lod: LodParams,
}

impl MaterialExtension for SdfExtension { }


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
    uniform_start: usize,
    dirty: bool,
}


pub struct SdfPlugin;
impl Plugin for SdfPlugin {
    fn build(&self, app: &mut App) {
        app
            .register_type::<MeshSdfCache>()
            .register_type::<ComputeSdf>()
            .register_type::<Sdf>()
            .register_type::<SdfRoot>()
            .register_type::<SdfConfig>()
            .init_resource::<SdfConfig>()
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


#[allow(clippy::type_complexity)]
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
    caches: Query<Entity, With<MeshSdfCache>>,
    mut agg: ResMut<AggregateSdf>,
    mut rng: Local<Option<SmallRng>>,
) {
    if rng.is_none() { *rng = SmallRng::from_entropy().into(); }
    let rng = rng.as_mut().unwrap();

    if cfg.is_changed() {
        for e in caches.iter() {
            if let Ok(mut ent) = commands.get_entity(e) {
                ent.remove::<MeshSdfCache>();
            }
        }
        return;
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


// TODO: resolve non-manifold ambiguities
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


pub fn aggregate_sdf_system(
    mut agg: ResMut<AggregateSdf>,
    mut caches: Query<&mut MeshSdfCache>,
    config: Res<SdfConfig>,
) {
    if !caches.iter().any(|c| c.dirty) {
        return;
    }

    agg.pts.clear();
    for mut c in &mut caches {
        agg.pts.extend(&c.pts);
        c.dirty = false;
    }
    agg.dirty = true;
    agg.uniform_start = agg.pts.len();

    info!("aggregated {} SDF points from mesh caches", agg.pts.len());

    if !config.dense_sampling {
        return;
    }

    if agg.pts.is_empty() {
        return;
    }

    let (mut min, mut max) =
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY));

    for p in &agg.pts {
        let pos = p.pos_dist.truncate();
        min = min.min(pos);
        max = max.max(pos);
    }

    const CELLS_PER_AXIS: f32 = 64.0;
    let cell_size = (max - min).max_element() / CELLS_PER_AXIS;
    if cell_size == 0.0 {
        return;
    }
    let inv_cell = 1.0 / cell_size;
    let dims = ((max - min) * inv_cell).ceil().as_ivec3();

    #[derive(Default, Clone, Copy)]
    struct Acc {
        sum_pos: Vec3,
        sum_dist: f32,
        count: u32,
        has_neg: bool,
    }
    let mut acc: HashMap<IVec3, Acc> = HashMap::with_capacity(agg.pts.len());

    const NB6: [IVec3; 6] = [
        IVec3::new( 1, 0, 0),  IVec3::new(-1, 0, 0),
        IVec3::new( 0, 1, 0),  IVec3::new( 0,-1, 0),
        IVec3::new( 0, 0, 1),  IVec3::new( 0, 0,-1),
    ];

    for p in &agg.pts {
        let pos   = p.pos_dist.truncate();
        let key   = ((pos - min) * inv_cell).floor().as_ivec3();
        let dist  = p.pos_dist.w;
        let isneg = dist.is_sign_negative();

        let mut bump = |k: IVec3| {
            let e = acc.entry(k).or_default();
            e.sum_pos  += pos;
            e.sum_dist += dist;
            e.count    += 1;
            e.has_neg  |= isneg;
        };

        bump(key);

        for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if dx == 0 && dy == 0 && dz == 0 {
                        continue;
                    }
                    bump(key + IVec3::new(dx, dy, dz));
                }
            }
        }
    }

    #[derive(Clone, Copy)]
    struct Cell { pos: Vec3, dist: f32, has_neg: bool }

    let occ: HashMap<IVec3, Cell> = acc
        .into_iter()
        .map(|(k, a)| {
            let inv = 1.0 / a.count as f32;
            (
                k,
                Cell {
                    pos:      a.sum_pos * inv,
                    dist:     a.sum_dist * inv,
                    has_neg:  a.has_neg,
                },
            )
        })
        .collect();

    let mut outside = HashSet::<IVec3>::new();
    let mut stack   = Vec::<IVec3>::new();

    for x in 0..=dims.x {
        for y in 0..=dims.y {
            for z in 0..=dims.z {
                if x == 0 || y == 0 || z == 0 ||
                   x == dims.x || y == dims.y || z == dims.z
                {
                    let k = IVec3::new(x, y, z);
                    if !occ.contains_key(&k) && outside.insert(k) {
                        stack.push(k);
                    }
                }
            }
        }
    }

    let is_solid = |k: &IVec3| occ.contains_key(k);

    while let Some(c) = stack.pop() {
        for d in NB6 {
            let n = c + d;
            if n.cmpge(IVec3::ZERO).all() &&
               n.cmple(dims).all() &&
               !outside.contains(&n) &&
               !is_solid(&n)
            {
                outside.insert(n);
                stack.push(n);
            }
        }
    }

    let offsets_26 = || {
        (-1..=1)
            .flat_map(|dx| (-1..=1).flat_map(move |dy| (-1..=1).map(move |dz| (dx, dy, dz))))
            .filter(|&(dx, dy, dz)| dx != 0 || dy != 0 || dz != 0)
            .map(|(dx, dy, dz)| IVec3::new(dx, dy, dz))
    };

    for x in 0..=dims.x {
        for y in 0..=dims.y {
            for z in 0..=dims.z {
                let k = IVec3::new(x, y, z);
                if occ.contains_key(&k) {
                    continue;
                }

                let centre = min + (k.as_vec3() + Vec3::splat(0.5)) * cell_size;

                let want_positive = outside.contains(&k);

                let mut best_mag: Option<f32> = None;
                for off in offsets_26() {
                    if let Some(c) = occ.get(&(k + off)) {
                        if c.dist.is_sign_positive() != want_positive {
                            continue;
                        }

                        let cand = c.dist.abs() + (centre - c.pos).length();
                        best_mag = Some(match best_mag {
                            Some(b) => b.min(cand),
                            None => cand,
                        });
                    }
                }

                let magnitude = best_mag.unwrap_or(0.5 * cell_size);
                let signed = if want_positive { magnitude } else { -magnitude };

                agg.pts.push(GpuPt {
                    pos_dist: centre.extend(signed),
                });
            }
        }
    }

    info!("SDF aggregate: {} points after uniform sampling", agg.pts.len());
}


fn upload_sdf(
    mut commands: Commands,
    mut planar_gaussian_3d: ResMut<Assets<PlanarGaussian3d>>,
    mut agg: ResMut<AggregateSdf>,
    scene_root: Query<
        Entity,
        With<SdfRoot>,
    >,
    config: Res<SdfConfig>,
) {
    if !config.gaussian_visualization {
        return;
    }

    if agg.pts.is_empty() || !agg.dirty {
        return;
    }

    info!("uploading {} SDF points to PlanarGaussian3d", agg.pts.len());

    // TODO: write gaussians in compute shader
    { // map sdf to gaussians
        const MIN_SCALE: f32 = 0.005;
        const MAX_SCALE: f32 = 0.02;

        let max_abs = agg.pts[..agg.uniform_start]
            .iter()
            .map(|p| p.pos_dist.w.abs())
            .fold(1e-5f32, f32::max);

        let gaussians: Vec<Gaussian3d> = agg.pts//[..agg.uniform_start]
            .iter()
            .map(|pt| {
                let dist = pt.pos_dist.w;
                let signed_n = (dist / max_abs).clamp(-1.0, 1.0);
                let mag_n = signed_n.abs().sqrt();

                let s = MAX_SCALE - mag_n * (MAX_SCALE - MIN_SCALE);
                let opacity = 1.0;// - mag_n;

                Gaussian3d {
                    rotation: [1.0, 0.0, 0.0, 0.0].into(),
                    position_visibility: PositionVisibility {
                        position: pt.pos_dist.truncate().into(),
                        visibility: 1.0,
                    },
                    scale_opacity: ScaleOpacity {
                        scale: [s, s, s],
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

        // TODO: redraw on PlanarGaussian3dHandle override (upstream issue)
        commands
            .entity(scene_root.single().expect("SdfRoot required to attach gaussians"))
            .insert((
                CloudSettings::default(),
                PlanarGaussian3dHandle(gaussians_handle),
            ));

        // TODO: remove gaussians from scene root when leaving RenderMode::Sdf?
    }

    agg.dirty = false;
}
