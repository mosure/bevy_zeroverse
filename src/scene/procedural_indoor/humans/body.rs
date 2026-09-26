//! AnnyBody surface, retargeted with its full skinning weights. Garments are
//! offset shells of that surface; no substitute primitive body is rendered.
use super::{HumanAssembly, HumanSurface, IndoorHuman};
use crate::scene::procedural_indoor::layout::stream;
use bevy::prelude::*;
use burn_human::{AnnyBody, AnnyInput};
use rand::Rng;
use std::sync::{Arc, Mutex, OnceLock};

static REFERENCE: OnceLock<Arc<AnnyBody>> = OnceLock::new();
type CachedPerson = (IndoorHuman, HumanAssembly);
static CACHE: Mutex<std::collections::VecDeque<CachedPerson>> =
    Mutex::new(std::collections::VecDeque::new());
const CACHE_CAPACITY: usize = 24;

pub(crate) fn installed() -> Option<Arc<AnnyBody>> {
    REFERENCE.get().cloned()
}

pub(crate) fn install(reference: Arc<AnnyBody>) {
    let _ = REFERENCE.set(reference);
}

fn reference() -> &'static Arc<AnnyBody> {
    REFERENCE.get_or_init(|| {
        #[cfg(not(target_family = "wasm"))]
        {
            // Offline validators/reference exporters use the same model. The viewer
            // installs its asynchronously loaded asset before starting preparation.
            let root = crate::asset::asset_root().join("burn_human");
            Arc::new(
                AnnyBody::from_reference_paths(
                    root.join("fullbody_default.safetensors"),
                    root.join("fullbody_default.meta.json"),
                )
                .expect("AnnyBody reference required for indoor people"),
            )
        }
        #[cfg(target_family = "wasm")]
        panic!("wait for the AnnyBody asset before building indoor people")
    })
}

pub(super) fn build(h: &IndoorHuman) -> HumanAssembly {
    if let Some((_, mesh)) = CACHE.lock().unwrap().iter().find(|(person, _)| person == h) {
        return mesh.clone();
    }
    let mesh = build_uncached(h, reference());
    let mut cache = CACHE.lock().unwrap();
    if cache.len() == CACHE_CAPACITY {
        cache.pop_front();
    }
    cache.push_back((h.clone(), mesh.clone()));
    mesh
}

fn build_uncached(h: &IndoorHuman, body: &AnnyBody) -> HumanAssembly {
    let mut rng = stream(h.seed, 43);
    let phenotype: Vec<_> = body
        .metadata()
        .metadata
        .phenotype_labels
        .iter()
        .map(|name| match name.as_str() {
            "gender" => rng.random_range(0.0..1.0),
            "age" => rng.random_range(0.60..0.95), // adult reference anchors only
            "muscle" => rng.random_range(0.25..0.70),
            "weight" => ((h.build as f64 - 0.82) / 0.40).mul_add(0.42, 0.30),
            "height" => ((h.stature as f64 - 1.50) / 0.45).clamp(0.0, 1.0),
            _ => rng.random_range(0.30..0.70),
        })
        .collect();
    let output = body
        .forward(AnnyInput {
            phenotype_inputs: Some(&phenotype),
            ..Default::default()
        })
        .expect("valid AnnyBody phenotype");
    // Anny is Z-up, faces -Y. Indoor people are Y-up, facing -Z; this is a
    // proper rotation, including the left/right mapping (not a reflection).
    let rotate = |p: &[f64]| Vec3::new(-p[0] as f32, p[2] as f32, p[1] as f32);
    let vertices: Vec<_> = output
        .rest_vertices
        .data
        .as_chunks::<3>()
        .0
        .iter()
        .map(|p| rotate(p))
        .collect();
    let (labels, parents) = body.bone_hierarchy();
    let heads: Vec<_> = output
        .rest_bone_poses
        .data
        .as_chunks::<16>()
        .0
        .iter()
        .map(|m| rotate(&[m[3], m[7], m[11]]))
        .collect();
    let index = |name: &str| {
        labels
            .iter()
            .position(|n| n == name)
            .expect("Anny default rig bone")
    };
    let head = |name: &str| heads[index(name)];
    let lo = vertices
        .iter()
        .copied()
        .fold(Vec3::splat(f32::INFINITY), Vec3::min);
    let hi = vertices
        .iter()
        .copied()
        .fold(Vec3::splat(f32::NEG_INFINITY), Vec3::max);
    let scale = h.stature / (hi.y - lo.y);
    let p = &h.joints;
    let rest_pelvis = (head("upperleg01.L") + head("upperleg01.R")) * 0.5;
    // The program's chest is the shoulder girdle, not the spine01 bone head.
    // Mapping spine01 here elongated the torso above it and bunched the shoulders.
    let rest_chest = (head("upperarm01.L") + head("upperarm01.R")) * 0.5;
    let torso = segment(rest_pelvis, rest_chest, p[0], p[2], scale);
    let head_rotation = cranial_rotation(p, h.head_yaw);
    // Rest neck bones tilt forward anatomically. Aligning that vector with an
    // upright target rotates the whole skull backwards. Preserve the rest skull
    // orientation in the program's torso frame, and anchor its actual rig pivot.
    let neck = Mat4::from_scale_rotation_translation(
        Vec3::splat(scale),
        head_rotation,
        p[4] - Vec3::Y * 0.055 * scale,
    ) * Mat4::from_translation(-head("head"));
    let mut transforms = vec![torso; labels.len()];
    for (i, name) in labels.iter().enumerate() {
        transforms[i] = if parents[i] >= 0 {
            transforms[parents[i] as usize]
        } else {
            torso
        };
        if name.starts_with("neck") || name == "head" {
            transforms[i] = neck;
        }
        for (side, arm, leg) in [("L", 5, 13), ("R", 9, 17)] {
            let bone = |base: &str| head(&format!("{base}.{side}"));
            if !name.ends_with(&format!(".{side}")) {
                continue;
            }
            if name.starts_with("upperarm") {
                transforms[i] = segment(
                    bone("upperarm01"),
                    bone("lowerarm01"),
                    p[arm],
                    p[arm + 1],
                    scale,
                );
            } else if name.starts_with("lowerarm") {
                transforms[i] = segment(
                    bone("lowerarm01"),
                    bone("wrist"),
                    p[arm + 1],
                    p[arm + 2],
                    scale,
                );
            } else if name.starts_with("wrist") {
                transforms[i] = segment(
                    bone("wrist"),
                    bone("finger3-1"),
                    p[arm + 2],
                    p[arm + 3],
                    scale,
                );
            } else if name.starts_with("upperleg") {
                transforms[i] = segment(
                    bone("upperleg01"),
                    bone("lowerleg01"),
                    p[leg],
                    p[leg + 1],
                    scale,
                );
            } else if name.starts_with("lowerleg") {
                transforms[i] = segment(
                    bone("lowerleg01"),
                    bone("foot"),
                    p[leg + 1],
                    p[leg + 2],
                    scale,
                );
            } else if name.starts_with("foot") {
                transforms[i] =
                    segment(bone("foot"), bone("toe2-1"), p[leg + 2], p[leg + 3], scale);
            }
        }
    }
    let (bone_ids, weights) = body.skinning_bindings();
    let influences = *bone_ids.shape.last().unwrap();
    let mut posed = Vec::with_capacity(vertices.len());
    let mut dominant = Vec::with_capacity(vertices.len());
    for (i, &v) in vertices.iter().enumerate() {
        let mut q = Vec3::ZERO;
        let mut total = 0.0;
        let mut strongest = (0.0, 0);
        for influence in 0..influences {
            let at = i * influences + influence;
            let w = weights.data[at] as f32;
            let bone = bone_ids.data[at];
            if bone < 0 || w <= 0.0 {
                continue;
            }
            q += transforms[bone as usize].transform_point3(v) * w;
            total += w;
            if w > strongest.0 {
                strongest = (w, bone as usize);
            }
        }
        posed.push(if total > 0.0 {
            q / total
        } else {
            torso.transform_point3(v)
        });
        dominant.push(strongest.1);
    }
    // Ground the deformed feet exactly. The skeleton is anchored to footwear in
    // placement; a small sole offset avoids a floating shell at the floor.
    let floor = posed.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    for p in &mut posed {
        p.y -= floor;
    }
    let faces = &body.faces_quads().data;
    let mut normals = vec![Vec3::ZERO; posed.len()];
    for q in faces.as_chunks::<4>().0 {
        for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let [a, b, c] = tri.map(|i| i as usize);
            let normal = (posed[b] - posed[a]).cross(posed[c] - posed[a]);
            for i in [a, b, c] {
                normals[i] += normal;
            }
        }
    }
    for n in &mut normals {
        *n = n.normalize_or(Vec3::Y);
    }
    let uv = &body.metadata().static_data.texture_coordinates.data;
    let mut mesh = HumanAssembly {
        local_joints: h.joints.iter().map(|p| *p - Vec3::Y * floor).collect(),
        ..Default::default()
    };
    let torso_vertices: Vec<_> = vertices
        .iter()
        .filter(|v| v.y > rest_pelvis.y && v.y < rest_chest.y && v.x.abs() < 0.08)
        .collect();
    let front = torso_vertices
        .iter()
        .map(|v| v.z)
        .fold(f32::INFINITY, f32::min);
    let back = torso_vertices
        .iter()
        .map(|v| v.z)
        .fold(f32::NEG_INFINITY, f32::max);
    let cut = super::garments::GarmentCut {
        waist: rest_pelvis.y + (rest_chest.y - rest_pelvis.y) * 0.14,
        neck: head("neck01").y,
        chest: rest_chest.y - 0.03,
        torso_half_width: head("upperarm01.L")
            .x
            .abs()
            .max(head("upperarm01.R").x.abs())
            * 0.94,
        depth_center: (front + back) * 0.5,
        torso_half_depth: (back - front) * 0.5,
        shoe_top: (head("foot.L").y + head("foot.R").y) * 0.5 + 0.015,
        cuffs: ["L", "R"].map(|side| {
            let elbow = head(&format!("lowerarm01.{side}"));
            let wrist = head(&format!("wrist.{side}"));
            let short = h.outfit == super::HumanOutfit::Knitwear && h.seed.is_multiple_of(3);
            (
                elbow.lerp(wrist, if short { 0.12 } else { 0.90 }),
                (wrist - elbow).normalize(),
            )
        }),
    };
    let mut offsets = vec![0.0; posed.len()];
    let mut incident_faces = vec![0_u32; posed.len()];
    for q in faces.as_chunks::<4>().0 {
        let bone = &labels[dominant[q[0] as usize]];
        let center = q.iter().map(|&i| vertices[i as usize]).sum::<Vec3>() * 0.25;
        let surface = cut.surface(h, center, bone);
        let ease = match surface {
            HumanSurface::Top | HumanSurface::Shirt => match h.outfit {
                super::HumanOutfit::Knitwear => 0.006,
                super::HumanOutfit::Shirt => 0.012,
                super::HumanOutfit::Blazer => 0.021,
            },
            HumanSurface::Trousers => 0.008,
            HumanSurface::Shoes => 0.005,
            _ => 0.0,
        };
        let ease = if matches!(
            surface,
            HumanSurface::Top | HumanSurface::Shirt | HumanSurface::Trousers
        ) {
            h.appearance.as_ref().map_or(ease, |a| {
                a.garment_ease
                    * if surface == HumanSurface::Trousers {
                        0.65
                    } else {
                        1.0
                    }
            })
        } else {
            ease
        };
        for &i in q {
            offsets[i as usize] += ease;
            incident_faces[i as usize] += 1;
        }
    }
    // All material parts must share boundary positions. Offsetting each face's
    // material independently opens visible cracks around collars and cuffs.
    for (offset, count) in offsets.iter_mut().zip(incident_faces) {
        *offset /= count.max(1) as f32;
    }
    for (i, offset) in offsets.iter_mut().enumerate() {
        if *offset > 0.006 {
            if let Some(a) = &h.appearance {
                let v = vertices[i] * scale;
                let wave = (v.y * a.fold_frequency + (v.x * 11.0).sin() * 1.8 + v.z * 7.0).sin();
                // Compression folds are attenuated at garment seams by the shared
                // offset field. One displaced position is used by all incident faces.
                *offset += a.fold_amplitude * wave * (*offset / a.garment_ease).clamp(0.0, 1.0);
            }
        }
    }
    let garment_positions: Vec<_> = posed
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let surface = cut.surface(h, vertices[i], &labels[dominant[i]]);
            let rest_delta = cut.ease(vertices[i], surface);
            let mut delta = Vec3::ZERO;
            let mut total = 0.0;
            for k in 0..influences {
                let at = i * influences + k;
                let id = bone_ids.data[at];
                let w = weights.data[at] as f32;
                if id >= 0 && w > 0.0 {
                    delta += transforms[id as usize].transform_vector3(rest_delta) * w;
                    total += w;
                }
            }
            let q = *p + normals[i] * offsets[i] + delta / total.max(1e-6);
            q.with_y(q.y.max(0.0))
        })
        .collect();
    normals.fill(Vec3::ZERO);
    for q in faces.as_chunks::<4>().0 {
        for tri in [[q[0], q[1], q[2]], [q[0], q[2], q[3]]] {
            let [a, b, c] = tri.map(|i| i as usize);
            let n = (garment_positions[b] - garment_positions[a])
                .cross(garment_positions[c] - garment_positions[a]);
            for i in [a, b, c] {
                normals[i] += n;
            }
        }
    }
    for n in &mut normals {
        *n = n.normalize_or(Vec3::Y);
    }
    let uv_faces = &body
        .metadata()
        .static_data
        .face_texture_coordinate_indices
        .data;
    for (q, texture_indices) in faces
        .as_chunks::<4>()
        .0
        .iter()
        .zip(uv_faces.as_chunks::<4>().0)
    {
        let corners: [_; 4] = std::array::from_fn(|k| {
            let i = q[k] as usize;
            let t = texture_indices[k] as usize;
            super::garments::GarmentVertex {
                rest: vertices[i],
                position: garment_positions[i],
                normal: normals[i],
                // Anny has separate vertex and face-corner UV indices at seams.
                uv: Vec2::new(uv[t * 2] as f32, uv[t * 2 + 1] as f32),
            }
        });
        for tri in [[0, 1, 2], [0, 2, 3]] {
            cut.append(
                h,
                &labels[dominant[q[0] as usize]],
                tri.map(|i| corners[i]),
                &mut mesh,
            );
        }
    }
    // Eye and eyewear details attach to Anny's facial rig, following head turns.
    let face_rotation = head_rotation;
    let forward = face_rotation * Vec3::NEG_Z;
    let mut eyes = Vec::new();
    for name in ["eye.L", "eye.R"] {
        let i = index(name);
        let eye = transforms[i].transform_point3(heads[i]) - Vec3::Y * floor;
        eyes.push(eye);
        let front = posed
            .iter()
            .zip(&dominant)
            .filter(|(_, bone)| **bone == i)
            .map(|(v, _)| (*v - eye).dot(forward))
            .reduce(f32::max)
            .unwrap_or(0.011 * scale);
        mesh.part(HumanSurface::Iris).ellipsoid(
            Vec3::new(0.0055, 0.0055, 0.001) * scale,
            Transform::from_translation(eye + forward * (front + 0.0004 * scale))
                .with_rotation(face_rotation),
        );
        mesh.part(HumanSurface::Detail).ellipsoid(
            Vec3::new(0.0021, 0.0021, 0.0005) * scale,
            Transform::from_translation(eye + forward * (front + 0.0015 * scale))
                .with_rotation(face_rotation),
        );
        // Eyebrows follow the facial frame, set above the eye rim.
        for k in 0..5 {
            let local = |t: f32| {
                Vec3::new(
                    (t - 0.5) * 0.032,
                    0.021 + (t * std::f32::consts::PI).sin() * 0.003,
                    -front - 0.001,
                )
            };
            mesh.part(HumanSurface::Hair).rod(
                eye + face_rotation * local(k as f32 / 5.0),
                eye + face_rotation * local((k + 1) as f32 / 5.0),
                0.0014,
            );
        }
        if h.glasses {
            let points = [
                Vec3::new(-0.019, -0.012, -0.018),
                Vec3::new(0.019, -0.012, -0.018),
                Vec3::new(0.019, 0.012, -0.018),
                Vec3::new(-0.019, 0.012, -0.018),
            ];
            for k in 0..4 {
                mesh.part(HumanSurface::Detail).rod(
                    eye + face_rotation * points[k] * scale,
                    eye + face_rotation * points[(k + 1) % 4] * scale,
                    0.0012 * scale,
                );
            }
        }
    }
    if h.glasses {
        mesh.part(HumanSurface::Detail).rod(
            eyes[0] + forward * 0.018 * scale,
            eyes[1] + forward * 0.018 * scale,
            0.001 * scale,
        );
    }
    if h.outfit != super::HumanOutfit::Knitwear {
        for row in 0..6 {
            let y = cut.waist + (cut.chest - cut.waist) * (row as f32 + 0.5) / 6.0;
            let at = vertices
                .iter()
                .enumerate()
                .filter(|(_, v)| v.x.abs() < 0.025 && v.z < cut.depth_center)
                .min_by(|(_, a), (_, b)| {
                    ((a.y - y).abs() + a.x.abs()).total_cmp(&((b.y - y).abs() + b.x.abs()))
                })
                .map(|(i, _)| i);
            if let Some(i) = at {
                mesh.part(HumanSurface::Detail).ellipsoid(
                    Vec3::new(0.0028, 0.0028, 0.0012),
                    Transform::from_translation(garment_positions[i] + normals[i] * 0.002)
                        .with_rotation(Quat::from_rotation_arc(Vec3::Z, normals[i])),
                );
            }
        }
    }
    super::hair::append(
        h,
        &vertices,
        &posed,
        &normals,
        faces,
        &dominant,
        labels,
        head_rotation,
        mesh.part(HumanSurface::Hair),
    );
    // The deformed surface contains sharp creases at bent joints. Split their
    // shading normals instead of interpolating a normal through the back face.
    for g in mesh.parts.values_mut() {
        for triangle in g.indices.as_chunks_mut::<3>().0 {
            let [a, b, c] = [triangle[0], triangle[1], triangle[2]]
                .map(|i| Vec3::from_array(g.positions[i as usize]));
            let n = (b - a).cross(c - a).normalize_or(Vec3::Y);
            if triangle
                .iter()
                .any(|i| n.dot(Vec3::from_array(g.normals[*i as usize])) < 0.0)
            {
                for i in triangle {
                    let source = *i as usize;
                    *i = g.positions.len() as u32;
                    g.positions.push(g.positions[source]);
                    g.uvs.push(g.uvs[source]);
                    g.normals.push(n.to_array());
                }
            }
        }
    }
    mesh
}

fn cranial_rotation(joints: &[Vec3], yaw: f32) -> Quat {
    let up = (joints[2] - joints[0]).normalize_or(Vec3::Y);
    let shoulder = (joints[9] - joints[5]).normalize_or(Vec3::X);
    let back = shoulder.cross(up).normalize_or(Vec3::Z);
    let right = up.cross(back).normalize_or(Vec3::X);
    Quat::from_mat3(&Mat3::from_cols(right, up, back)) * Quat::from_rotation_y(yaw)
}

/// Match joint endpoints while retaining transverse anatomical dimensions.
fn segment(a: Vec3, b: Vec3, target_a: Vec3, target_b: Vec3, scale: f32) -> Mat4 {
    let source = b - a;
    let target = target_b - target_a;
    let axis = source.normalize_or(Vec3::Y);
    let stretch = target.length() / source.length().max(0.0001);
    let radial = Mat3::IDENTITY * scale;
    let along = Mat3::from_cols(axis * axis.x, axis * axis.y, axis * axis.z) * (stretch - scale);
    let linear = Mat3::from_quat(Quat::from_rotation_arc(axis, target.normalize_or(Vec3::Y)))
        * (radial + along);
    Mat4::from_cols(
        linear.x_axis.extend(0.0),
        linear.y_axis.extend(0.0),
        linear.z_axis.extend(0.0),
        (target_a - linear * a).extend(1.0),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest};
    use std::collections::HashSet;

    #[test]
    fn upright_program_does_not_pitch_the_cranium_backwards() {
        for seed in 0..128 {
            let program = super::super::poses::PoseProgram::sample(
                seed,
                super::super::HumanPoseKind::StandingRelaxed,
            );
            let joints = program.solve(1.75, 1.0, 0.43, false);
            let rotation = cranial_rotation(&joints, 0.0);
            let up = (joints[2] - joints[0]).normalize();
            assert!((rotation * Vec3::Y).distance(up) < 1e-5);
            assert!((rotation * Vec3::NEG_Z).y.abs() < 0.15);
        }
    }

    #[test]
    fn clothing_material_boundaries_keep_the_anny_surface_closed() {
        let scene = IndoorManifest::generate(31, IndoorLayout::Mixed, 0.65, 1).unwrap();
        let body = reference();
        let original_vertices: HashSet<_> = body.faces_quads().data.iter().copied().collect();
        let data = &body.metadata().static_data;
        let expected_uvs: HashSet<_> = data
            .face_texture_coordinate_indices
            .data
            .iter()
            .map(|&i| {
                let i = i as usize * 2;
                [
                    data.texture_coordinates.data[i] as f32,
                    data.texture_coordinates.data[i + 1] as f32,
                ]
                .map(f32::to_bits)
            })
            .collect();
        assert!(!scene.humans.is_empty());
        for human in &scene.humans {
            let mesh = build(human);
            let vertices: HashSet<_> = mesh
                .parts
                .iter()
                .filter(|(surface, _)| {
                    !matches!(
                        surface,
                        HumanSurface::Hair
                            | HumanSurface::Detail
                            | HumanSurface::Seam
                            | HumanSurface::Iris
                    )
                })
                .flat_map(|(_, g)| &g.positions)
                .map(|p| p.map(f32::to_bits))
                .collect();
            // Clipping retains the complete Anny body and adds boundary vertices.
            // Most vertices remain unchanged; this is not a substitute body.
            assert!(vertices.len() >= original_vertices.len());
            assert!(vertices.len() < original_vertices.len() * 2);
            let uvs: HashSet<_> = mesh
                .parts
                .iter()
                .filter(|(s, _)| {
                    !matches!(
                        s,
                        HumanSurface::Hair
                            | HumanSurface::Detail
                            | HumanSurface::Seam
                            | HumanSurface::Iris
                    )
                })
                .flat_map(|(_, g)| &g.uvs)
                .map(|uv| uv.map(f32::to_bits))
                .collect();
            assert!(
                expected_uvs.is_subset(&uvs),
                "Anny face-corner UV seams must survive material splitting"
            );
        }
    }
}
