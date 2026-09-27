//! Prepare dressed Anny geometry once, then update only bone transforms. Baked
//! capture uses the identical four-weight deformation as the interactive GPU.
use super::{planning, validation, MotionPlan};
use crate::scene::procedural_indoor::{
    geometry::Geometry,
    humans::{self, HumanSurface, IndoorHuman},
    layout::IndoorManifest,
};
use bevy::{mesh::VertexAttributeValues, prelude::*};
use burn_human::motion::AnnyMotionBinding;
use burn_human_motion::MotionClip;
use std::{
    collections::{BTreeMap, HashMap},
    sync::Arc,
};

#[derive(Clone)]
pub struct SkinVertices {
    pub positions: Vec<Vec3>,
    pub normals: Vec<Vec3>,
    pub tangents: Vec<[f32; 4]>,
    pub indices: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
}
pub struct DeformedVertices {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub tangents: Vec<[f32; 4]>,
}
impl SkinVertices {
    pub fn deform(&self, matrices: &[Mat4]) -> DeformedVertices {
        let mut p = Vec::with_capacity(self.positions.len());
        let mut n = Vec::with_capacity(p.capacity());
        let mut tangents = Vec::with_capacity(self.tangents.len());
        for i in 0..self.positions.len() {
            let m = blend(matrices, self.indices[i], self.weights[i]);
            p.push(m.transform_point3(self.positions[i]).to_array());
            n.push(
                (Mat3::from_mat4(m).inverse().transpose() * self.normals[i])
                    .normalize_or(Vec3::Y)
                    .to_array(),
            );
            // Match Bevy's skin_normals and mesh_tangent_local_to_world, including
            // the inverse transpose of the *blended* matrix, not each bone.
            if let Some(&t) = self.tangents.get(i) {
                tangents.push(
                    m.transform_vector3(Vec3::new(t[0], t[1], t[2]))
                        .normalize_or_zero()
                        .extend(t[3])
                        .to_array(),
                );
            }
        }
        DeformedVertices {
            positions: p,
            normals: n,
            tangents,
        }
    }
    pub fn attributes(&self, mesh: &mut Mesh) {
        mesh.insert_attribute(
            Mesh::ATTRIBUTE_JOINT_INDEX,
            VertexAttributeValues::Uint16x4(self.indices.clone()),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_JOINT_WEIGHT, self.weights.clone());
    }
}
pub fn blend(matrices: &[Mat4], ids: [u16; 4], weights: [f32; 4]) -> Mat4 {
    matrices[ids[0] as usize] * weights[0]
        + matrices[ids[1] as usize] * weights[1]
        + matrices[ids[2] as usize] * weights[2]
        + matrices[ids[3] as usize] * weights[3]
}

#[derive(Clone)]
pub struct MotionFrame {
    pub bones: Vec<Transform>,
    pub joints: Vec<Vec3>,
    pub bounds: (Vec3, Vec3),
}
pub struct MotionRig {
    pub parents: Vec<Option<usize>>,
    pub annotation_indices: [usize; 21],
    pub stature: f32,
    /// Bounds of all render vertices influenced by each bone, in bind space.
    pub local_bounds: Vec<Option<(Vec3, Vec3)>>,
}
impl MotionRig {
    fn bounds(&self, bones: &[Transform]) -> Option<(Vec3, Vec3)> {
        let mut result: Option<(Vec3, Vec3)> = None;
        for (bone, bounds) in bones.iter().zip(&self.local_bounds) {
            let Some((lo, hi)) = bounds else { continue };
            let center = bone.transform_point((lo + hi) * 0.5);
            let matrix = Mat3::from_quat(bone.rotation) * Mat3::from_diagonal(bone.scale);
            let half = (hi - lo) * 0.5;
            let extent = matrix.x_axis.abs() * half.x
                + matrix.y_axis.abs() * half.y
                + matrix.z_axis.abs() * half.z;
            let next = (center - extent, center + extent);
            result = Some(result.map_or(next, |old| (old.0.min(next.0), old.1.max(next.1))));
        }
        result
    }

    fn joints(&self, bones: &[Transform]) -> Vec<Vec3> {
        let mut joints: Vec<_> = self
            .annotation_indices
            .iter()
            .map(|&i| bones[i].translation)
            .collect();
        joints[0] = (joints[13] + joints[17]) * 0.5;
        joints[2] = (joints[5] + joints[9]) * 0.5;
        joints[1] = joints[0].lerp(joints[2], 0.48);
        let offset = (joints[4] - joints[3]).normalize_or(Vec3::Y) * 0.055 * self.stature / 1.75;
        joints[4] += offset;
        joints
    }
}
impl MotionFrame {
    pub fn interpolate(&self, next: &Self, t: f32, rig: &MotionRig) -> Self {
        let mut bones: Vec<Transform> = Vec::with_capacity(self.bones.len());
        for (i, (a, b)) in self.bones.iter().zip(&next.bones).enumerate() {
            let transform = if let Some(parent) = rig.parents[i] {
                let pa = self.bones[parent];
                let pb = next.bones[parent];
                let local_a = pa.rotation.inverse() * a.rotation;
                let local_b = pb.rotation.inverse() * b.rotation;
                let offset = pa.rotation.inverse() * (a.translation - pa.translation);
                Transform {
                    translation: bones[parent].translation + bones[parent].rotation * offset,
                    rotation: (bones[parent].rotation * local_a.slerp(local_b, t)).normalize(),
                    scale: a.scale,
                }
            } else {
                Transform {
                    translation: a.translation.lerp(b.translation, t),
                    rotation: a.rotation.slerp(b.rotation, t),
                    scale: a.scale,
                }
            };
            bones.push(transform);
        }
        Self {
            joints: rig.joints(&bones),
            bounds: rig.bounds(&bones).unwrap_or((
                self.bounds.0.min(next.bounds.0),
                self.bounds.1.max(next.bounds.1),
            )),
            bones,
        }
    }
}
pub struct PreparedActor {
    pub rig: Arc<MotionRig>,
    pub plan: MotionPlan,
    pub parts: BTreeMap<HumanSurface, (Geometry, SkinVertices)>,
    pub inverse_bind: Vec<Mat4>,
    pub frames: Vec<MotionFrame>,
    pub clip: MotionClip,
}

pub fn prepare(
    scene: &IndoorManifest,
    person: &IndoorHuman,
    plan: MotionPlan,
    clip: MotionClip,
) -> Result<PreparedActor, String> {
    clip.validate().map_err(|e| e.to_string())?;
    if clip.frames.len() != plan.request.frames {
        return Err("motion frame count mismatch".into());
    }
    validation::validate_behavior(&plan, &clip)?;
    let (assembly, rest) = humans::body::build_rest(person);
    let body = humans::body::installed().ok_or("Anny reference unavailable")?;
    let binding =
        AnnyMotionBinding::new(&body, &rest.phenotype, &clip.rig).map_err(|e| e.to_string())?;
    let inverse_bind: Vec<_> = rest.bones.iter().map(|b| b.inverse()).collect();
    let labels = [
        "root",
        "spine04",
        "spine01",
        "neck01",
        "head",
        "upperarm01.L",
        "lowerarm01.L",
        "wrist.L",
        "finger3-1.L",
        "upperarm01.R",
        "lowerarm01.R",
        "wrist.R",
        "finger3-1.R",
        "upperleg01.L",
        "lowerleg01.L",
        "foot.L",
        "toe3-1.L",
        "upperleg01.R",
        "lowerleg01.R",
        "foot.R",
        "toe3-1.R",
    ];
    let joint_indices = labels.map(|n| binding.target.index(n).expect("Anny joint"));
    let mut rig = MotionRig {
        parents: binding.target.joints.iter().map(|j| j.parent).collect(),
        annotation_indices: joint_indices,
        stature: person.stature,
        local_bounds: Vec::new(),
    };
    let basis = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
    let mut frames = Vec::with_capacity(clip.frames.len());
    let obstacles = planning::obstacles(scene, person.id, plan.support_chair);
    let support = plan
        .support_chair
        .and_then(|id| scene.objects.iter().find(|o| o.id == id))
        .map(|chair| {
            super::contact::SupportSurface::new(
                &crate::scene::procedural_indoor::objects::build_object(chair),
                chair.transform(),
            )
        });
    let validate = |vertices: &[Vec3], bounds| {
        validation::validate_positions(scene, vertices, bounds, &obstacles)?;
        if support
            .as_ref()
            .is_some_and(|s| vertices.iter().any(|&p| s.penetrates(p)))
        {
            return Err("body penetrates generated support chair geometry".to_string());
        }
        Ok(())
    };
    let vertex_positions = |bones: &[Transform]| -> Vec<Vec3> {
        let matrices: Vec<_> = bones
            .iter()
            .zip(&inverse_bind)
            .map(|(b, i)| b.to_matrix() * *i)
            .collect();
        rest.positions
            .iter()
            .enumerate()
            .map(|(i, &p)| blend(&matrices, rest.indices[i], rest.weights[i]).transform_point3(p))
            .collect()
    };
    for (index, source) in clip.frames.iter().enumerate() {
        let expected = validation::expected_position(&plan, index).with_y(0.0);
        if source.root_translation.with_y(0.0).distance(expected) > 0.75 {
            return Err(format!(
                "frame {index}: generated root misses its timed waypoint corridor"
            ));
        }

        if validation::path_distance(source.root_translation, &plan) > 0.60 {
            return Err(
                "generated trajectory deviates more than 0.6 m from navigation corridor".into(),
            );
        }
        let mut source = source.clone();
        source.root_translation.y *= binding.root_height_scale;
        let pose = binding
            .mapping
            .retarget(&clip.rig, &binding.target, &source)
            .map_err(|e| e.to_string())?;
        let (positions, rotations) = binding.target.forward(&pose).map_err(|e| e.to_string())?;
        let offset = Vec3::new(source.root_translation.x, 0.0, source.root_translation.z)
            * (1.0 - rest.scale);
        let mut bones: Vec<_> = positions
            .iter()
            .zip(rotations)
            .map(|(p, q)| Transform {
                translation: basis * *p * rest.scale + offset,
                rotation: basis * q,
                scale: Vec3::splat(rest.scale),
            })
            .collect();
        let mut vertices = vertex_positions(&bones);
        let floor = vertices.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
        if (source.foot_contacts.iter().any(|c| *c) && floor.abs() < 0.12)
            || (-0.06..0.0).contains(&floor)
        {
            for bone in &mut bones {
                bone.translation.y -= floor;
            }
            for p in &mut vertices {
                p.y -= floor;
            }
        }
        let frame_bounds = bounds(&vertices);
        validate(&vertices, frame_bounds).map_err(|e| format!("frame {index}: {e}"))?;
        let joints = rig.joints(&bones);
        let frame = MotionFrame {
            bones,
            joints,
            bounds: frame_bounds,
        };
        if let Some(previous) = frames.last() {
            let previous: &MotionFrame = previous;
            let max_angle = previous
                .bones
                .iter()
                .zip(&frame.bones)
                .map(|(a, b)| a.rotation.angle_between(b.rotation))
                .fold(0.0, f32::max);
            let max_step = previous
                .bones
                .iter()
                .zip(&frame.bones)
                .map(|(a, b)| a.translation.distance(b.translation))
                .fold(0.0, f32::max);
            let steps = ((max_angle / 0.15).max(max_step / 0.08).ceil() as usize).max(2);
            if steps > 8 {
                return Err(format!("frame {index}: discontinuous generated pose"));
            }
            for k in 1..steps {
                let between = previous.interpolate(&frame, k as f32 / steps as f32, &rig);
                let p = vertex_positions(&between.bones);
                validate(&p, bounds(&p))
                    .map_err(|e| format!("between frames {} and {index}: {e}", index - 1))?;
            }
        }
        let time = index as f32 / (clip.frames.len() - 1) as f32;
        if scene
            .cameras
            .iter()
            .any(|c| validation::point_inside(c.transform_at(time).translation, frame_bounds, 0.10))
        {
            return Err(format!("frame {index}: camera intersects moving person"));
        }
        frames.push(frame);
    }
    // Transfer the original Anny weights to garment seam vertices and attached
    // details using a rest-space spatial grid; no nearest-neighbour search runs
    // during playback. Identical seam positions receive identical weights.
    let cell = |v: Vec3| {
        (
            (v.x / 0.04).floor() as i32,
            (v.y / 0.04).floor() as i32,
            (v.z / 0.04).floor() as i32,
        )
    };
    let mut grid: HashMap<_, Vec<usize>> = HashMap::new();
    for (i, &p) in rest.positions.iter().enumerate() {
        grid.entry(cell(p)).or_default().push(i);
    }
    let nearest = |p: Vec3| {
        let (x, y, z) = cell(p);
        let mut best = (f32::INFINITY, 0);
        for dx in -2..=2 {
            for dy in -2..=2 {
                for dz in -2..=2 {
                    if let Some(ids) = grid.get(&(x + dx, y + dy, z + dz)) {
                        for &i in ids {
                            let d = p.distance_squared(rest.positions[i]);
                            if d < best.0 {
                                best = (d, i);
                            }
                        }
                    }
                }
            }
        }
        if !best.0.is_finite() {
            for (i, &v) in rest.positions.iter().enumerate() {
                let d = p.distance_squared(v);
                if d < best.0 {
                    best = (d, i);
                }
            }
        }
        best.1
    };
    let parts: BTreeMap<_, _> = assembly
        .parts
        .into_iter()
        .filter(|(_, geometry)| !geometry.indices.is_empty())
        .map(|(surface, g)| {
            let positions: Vec<_> = g.positions.iter().copied().map(Vec3::from_array).collect();
            let ids: Vec<_> = positions.iter().map(|&p| nearest(p)).collect();
            let skin = SkinVertices {
                positions,
                normals: g.normals.iter().copied().map(Vec3::from_array).collect(),
                tangents: Vec::new(),
                indices: ids.iter().map(|&i| rest.indices[i]).collect(),
                weights: ids.iter().map(|&i| rest.weights[i]).collect(),
            };
            (surface, (g, skin))
        })
        .collect();
    // Linear blend skinning is a convex combination of transformed vertices.
    // The union of their per-bone bounds encloses clothing, hair and accessories
    // at arbitrary interpolated times, without a CPU mesh pass in the viewer.
    rig.local_bounds = vec![None; inverse_bind.len()];
    for (_, skin) in parts.values() {
        for (i, &p) in skin.positions.iter().enumerate() {
            for (&bone, &weight) in skin.indices[i].iter().zip(&skin.weights[i]) {
                if weight <= 0.0 {
                    continue;
                }
                let bone = bone as usize;
                let q = inverse_bind[bone].transform_point3(p);
                let old = rig.local_bounds[bone];
                rig.local_bounds[bone] =
                    Some(old.map_or((q, q), |(lo, hi)| (lo.min(q), hi.max(q))));
            }
        }
    }
    for frame in &mut frames {
        frame.bounds = rig
            .bounds(&frame.bones)
            .ok_or("motion actor has no render vertices")?;
    }
    Ok(PreparedActor {
        rig: Arc::new(rig),
        plan,
        parts,
        inverse_bind,
        frames,
        clip,
    })
}

fn bounds(vertices: &[Vec3]) -> (Vec3, Vec3) {
    (
        vertices
            .iter()
            .copied()
            .fold(Vec3::splat(f32::INFINITY), Vec3::min),
        vertices
            .iter()
            .copied()
            .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn baked_normals_and_tangents_match_bevy_skinning_convention() {
        let source = SkinVertices {
            positions: vec![Vec3::X],
            normals: vec![Vec3::Y],
            tangents: vec![[1.0, 0.0, 0.0, -1.0]],
            indices: vec![[0, 1, 0, 0]],
            weights: vec![[0.35, 0.65, 0.0, 0.0]],
        };
        let matrices = [
            Mat4::IDENTITY,
            Mat4::from_scale_rotation_translation(
                Vec3::new(1.0, 1.8, 0.6),
                Quat::from_rotation_z(1.1),
                Vec3::Z,
            ),
        ];
        let deformed = source.deform(&matrices);
        let m = blend(&matrices, source.indices[0], source.weights[0]);
        assert!(
            Vec3::from_array(deformed.positions[0]).distance(m.transform_point3(Vec3::X)) < 1e-6
        );
        let n = Vec3::from_array(deformed.normals[0]);
        let tangent = Vec4::from_array(deformed.tangents[0]);
        assert!(n.dot(tangent.truncate()).abs() < 1e-6);
        assert!((n.length() - 1.0).abs() < 1e-6);
        assert_eq!(tangent.w, -1.0);
    }
    #[test]
    fn interpolation_preserves_bone_lengths_and_encloses_deformed_render_vertices() {
        let rig = MotionRig {
            parents: vec![None, Some(0)],
            annotation_indices: [1; 21],
            stature: 1.75,
            local_bounds: vec![None, Some((Vec3::ZERO, Vec3::Y * 0.2))],
        };
        let frame = |angle| {
            let rotation = Quat::from_rotation_y(angle);
            MotionFrame {
                bones: vec![
                    Transform::from_rotation(rotation),
                    Transform::from_translation(rotation * Vec3::X).with_rotation(rotation),
                ],
                joints: vec![],
                bounds: (Vec3::ZERO, Vec3::ZERO),
            }
        };
        let a = frame(0.0);
        let b = frame(2.7);
        for i in 0..=100 {
            let f = a.interpolate(&b, i as f32 / 100.0, &rig);
            assert!((f.bones[0].translation.distance(f.bones[1].translation) - 1.0).abs() < 1e-5);
            let p = f.bones[1].transform_point(Vec3::Y * 0.2);
            assert!(p.cmpge(f.bounds.0 - Vec3::splat(1e-5)).all());
            assert!(p.cmple(f.bounds.1 + Vec3::splat(1e-5)).all());
        }
    }
}
