//! AnnyBody surface, retargeted with its full skinning weights. Garments are
//! offset shells of that surface; no substitute primitive body is rendered.
use super::rig::{cranial_rotation, segment, torso_segment};
use super::{HumanAssembly, HumanSurface, IndoorHuman};
use bevy::prelude::*;
use burn_human::{AnnyBody, AnnyInput};
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
    build_impl(h, body, false).0
}

/// Motion-only rest preparation; static people take the original cached path.
#[cfg(feature = "human_motion")]
pub(crate) fn build_rest(h: &IndoorHuman) -> (HumanAssembly, RestSkin) {
    let (mesh, rest) = build_impl(h, reference(), true);
    (mesh, rest.expect("requested rest skin"))
}

#[cfg_attr(not(feature = "human_motion"), allow(dead_code))]
pub(crate) struct RestSkin {
    pub phenotype: Vec<f64>,
    pub bones: Vec<Mat4>,
    pub positions: Vec<Vec3>,
    pub indices: Vec<[u16; 4]>,
    pub weights: Vec<[f32; 4]>,
    pub scale: f32,
}

fn build_impl(
    h: &IndoorHuman,
    body: &AnnyBody,
    motion_rest: bool,
) -> (HumanAssembly, Option<RestSkin>) {
    let shape = super::morphology::phenotype(h);
    let phenotype: Vec<_> = body
        .metadata()
        .metadata
        .phenotype_labels
        .iter()
        .map(|name| shape.value(name))
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
    let mut resolved_joints = h.joints.clone();
    let make_transforms = |p: &[Vec3]| {
        let rest_pelvis = (head("upperleg01.L") + head("upperleg01.R")) * 0.5;
        // The program's chest is the shoulder girdle, not the spine01 bone head.
        // Mapping spine01 here elongated the torso above it and bunched the shoulders.
        let rest_chest = (head("upperarm01.L") + head("upperarm01.R")) * 0.5;
        let torso = torso_segment(
            rest_pelvis,
            rest_chest,
            [head("upperarm01.L"), head("upperarm01.R")],
            p,
            scale,
        );
        let head_rotation = if motion_rest {
            Quat::IDENTITY
        } else {
            cranial_rotation(p, h.head_yaw)
        };
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
                    transforms[i] = if !motion_rest
                        && h.worktop_contact
                            .as_ref()
                            .is_some_and(|c| c.palms[(arm - 5) / 4])
                    {
                        super::rig::palm_segment(
                            bone("wrist"),
                            bone("finger3-1"),
                            (bone("finger5-1") - bone("finger2-1"))
                                * if side == "L" { 1. } else { -1. },
                            p[arm + 2],
                            p[arm + 3],
                            scale,
                        )
                    } else {
                        segment(
                            bone("wrist"),
                            bone("finger3-1"),
                            p[arm + 2],
                            p[arm + 3],
                            scale,
                        )
                    };
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
        if !motion_rest {
            if let Some(contact) = &h.worktop_contact {
                for (arm, side) in ["L", "R"].into_iter().enumerate() {
                    if !contact.palms[arm] {
                        continue;
                    }
                    let wrist = transforms[index(&format!("wrist.{side}"))];
                    for finger in 1..=5 {
                        for joint in 1..=3 {
                            let i = index(&format!("finger{finger}-{joint}.{side}"));
                            let a = heads[i];
                            let b = if joint < 3 {
                                head(&format!("finger{finger}-{}.{side}", joint + 1))
                            } else {
                                a + (a - head(&format!("finger{finger}-2.{side}"))) * 0.65
                            };
                            let origin = transforms[parents[i] as usize].transform_point3(a);
                            transforms[i] = super::rig::supported_finger(wrist, a, b, origin);
                        }
                    }
                }
            }
        }
        if motion_rest {
            transforms.fill(Mat4::from_scale(Vec3::splat(scale)));
        }
        transforms
    };
    let mut transforms = make_transforms(&resolved_joints);
    let (bone_ids, weights) = body.skinning_bindings();
    let influences = *bone_ids.shape.last().unwrap();
    let skin = |transforms: &[Mat4]| {
        let torso = transforms[0];
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
        (posed, dominant)
    };
    let (mut posed, dominant) = skin(&transforms);
    let floor = posed.iter().map(|p| p.y).fold(f32::INFINITY, f32::min);
    if !motion_rest {
        if let Some(contact) = &h.worktop_contact {
            // Only the two small arm chains are re-solved; body morphology is
            // evaluated once. Grounding comes from unchanged feet. Fit the
            // actual Anny palm, not a spherical hand proxy or a nominal height.
            for _ in 0..6 {
                let mut max_error = 0.0_f32;
                for (arm, side) in ["L", "R"].into_iter().enumerate() {
                    if !contact.palms[arm] {
                        continue;
                    }
                    let suffix = format!(".{side}");
                    let min = posed
                        .iter()
                        .enumerate()
                        .filter(|(i, p)| {
                            let name = &labels[dominant[*i]];
                            name.ends_with(&suffix)
                                && (name.starts_with("wrist") || name.starts_with("metacarpal"))
                                && super::contact::palm_region(
                                    &resolved_joints,
                                    arm,
                                    **p,
                                    h.stature / 1.75,
                                )
                        })
                        .map(|(_, p)| p.y - floor)
                        .fold(f32::INFINITY, f32::min);
                    let error = contact.height + 0.0005 - min;
                    max_error = max_error.max(error.abs());
                    let wrist = resolved_joints[7 + arm * 4] + Vec3::Y * error;
                    contact.arm(&mut resolved_joints, arm, wrist, h.stature / 1.75);
                }
                transforms = make_transforms(&resolved_joints);
                posed = skin(&transforms).0;
                if max_error < 0.0002 {
                    break;
                }
            }
        }
    }
    // Ground the deformed feet exactly. The skeleton is anchored to footwear in
    // placement; a small sole offset avoids a floating shell at the floor.
    for p in &mut posed {
        p.y -= floor;
    }
    let p = &resolved_joints;
    let rest_pelvis = (head("upperleg01.L") + head("upperleg01.R")) * 0.5;
    let rest_chest = (head("upperarm01.L") + head("upperarm01.R")) * 0.5;
    let torso = torso_segment(
        rest_pelvis,
        rest_chest,
        [head("upperarm01.L"), head("upperarm01.R")],
        p,
        scale,
    );
    let head_rotation = if motion_rest {
        Quat::IDENTITY
    } else {
        cranial_rotation(p, h.head_yaw)
    };
    let neck = Mat4::from_scale_rotation_translation(
        Vec3::splat(scale),
        head_rotation,
        p[4] - Vec3::Y * 0.055 * scale,
    ) * Mat4::from_translation(-head("head"));
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
        local_joints: resolved_joints
            .iter()
            .map(|p| *p - Vec3::Y * floor)
            .collect(),
        ..Default::default()
    };
    let torso_vertices: Vec<_> = vertices
        .iter()
        .enumerate()
        .filter(|(i, v)| {
            let bone = &labels[dominant[*i]];
            v.y > rest_pelvis.y - 0.06
                && v.y < rest_chest.y + 0.05
                && (bone.starts_with("spine") || bone.starts_with("breast"))
        })
        .map(|(_, v)| v)
        .collect();
    let front = torso_vertices
        .iter()
        .map(|v| v.z)
        .fold(f32::INFINITY, f32::min);
    let back = torso_vertices
        .iter()
        .map(|v| v.z)
        .fold(f32::NEG_INFINITY, f32::max);
    let left = torso_vertices
        .iter()
        .map(|v| v.x)
        .fold(f32::INFINITY, f32::min);
    let right = torso_vertices
        .iter()
        .map(|v| v.x)
        .fold(f32::NEG_INFINITY, f32::max);
    mesh.body_measurements = super::morphology::BodyMeasurements {
        rest_stature_metres: (hi.y - lo.y) * scale,
        rest_shoulder_bone_span_metres: head("upperarm01.L").distance(head("upperarm01.R")) * scale,
        rest_torso_width_metres: (right - left) * scale,
        rest_torso_depth_metres: (back - front) * scale,
        shoulder_attachment_offset_metres: if motion_rest {
            0.0
        } else {
            [("L", 5), ("R", 9)]
                .into_iter()
                .map(|(side, joint)| {
                    torso
                        .transform_point3(head(&format!("upperarm01.{side}")))
                        .distance(p[joint])
                })
                .fold(0.0, f32::max)
        },
    };
    let waist = rest_pelvis.y
        + (rest_chest.y - rest_pelvis.y) * h.appearance.as_ref().map_or(0.14, |a| a.hem_fraction);
    let mut garment = h
        .appearance
        .as_ref()
        .map_or_else(Default::default, |a| a.garment.clone());
    if h.outfit.collared() {
        garment.neckline_depth *= 0.18;
    }
    let cut = super::garments::GarmentCut {
        waist,
        neck: head("neck01").y,
        chest: rest_chest.y - 0.03,
        depth_center: (front + back) * 0.5,
        fit: super::garments::fit::TorsoFit::new(&torso_vertices, waist, rest_chest.y)
            .with_program(&garment),
        legs: ["L", "R"].map(|side| {
            let suffix = format!(".{side}");
            let leg: Vec<_> = vertices
                .iter()
                .enumerate()
                .filter_map(|(i, &p)| {
                    let name = &labels[dominant[i]];
                    (name.ends_with(&suffix)
                        && (name.starts_with("upperleg") || name.starts_with("lowerleg")))
                    .then_some(p)
                })
                .collect();
            super::garments::legs::LegFit::new(
                &leg,
                head(&format!("upperleg01.{side}")),
                head(&format!("foot.{side}")),
                &garment,
            )
        }),
        shoe_top: (head("foot.L").y + head("foot.R").y) * 0.5 + 0.015,
        cuffs: ["L", "R"].map(|side| {
            let elbow = head(&format!("lowerarm01.{side}"));
            let wrist = head(&format!("wrist.{side}"));
            let coverage = h.appearance.as_ref().map_or(0.95, |a| a.sleeve_coverage);
            let length = if h.outfit.open_front() {
                0.86 + coverage * 0.09
            } else if matches!(h.outfit, super::HumanOutfit::Tee | super::HumanOutfit::Polo) {
                -0.55 + coverage * 0.5
            } else {
                -0.40 + coverage * 1.35
            };
            (elbow.lerp(wrist, length), (wrist - elbow).normalize())
        }),
        leg_cuffs: ["L", "R"].map(|side| {
            let knee = head(&format!("lowerleg01.{side}"));
            let ankle = head(&format!("foot.{side}"));
            (
                knee.lerp(ankle, garment.trouser_coverage),
                (ankle - knee).normalize(),
            )
        }),
        program: garment,
    };
    let mut offsets = vec![0.0; posed.len()];
    let mut incident_faces = vec![0_u32; posed.len()];
    for q in faces.as_chunks::<4>().0 {
        let bone = &labels[dominant[q[0] as usize]];
        let center = q.iter().map(|&i| vertices[i as usize]).sum::<Vec3>() * 0.25;
        let surface = cut.surface(h, center, bone);
        let ease = match surface {
            HumanSurface::Top | HumanSurface::Shirt => match h.outfit {
                super::HumanOutfit::Knitwear | super::HumanOutfit::Tee => 0.006,
                super::HumanOutfit::Shirt | super::HumanOutfit::Polo => 0.012,
                super::HumanOutfit::Blazer | super::HumanOutfit::Cardigan => 0.021,
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
                        a.garment.trouser_ease / a.garment_ease.max(0.001)
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
    let fold_joints = [
        "lowerarm01.L",
        "lowerarm01.R",
        "lowerleg01.L",
        "lowerleg01.R",
    ]
    .map(head);
    let drape = super::garments::drape::Drape::new(h, waist, cut.chest, [front, back], fold_joints);
    for (i, offset) in offsets.iter_mut().enumerate() {
        let surface = cut.surface(h, vertices[i], &labels[dominant[i]]);
        if matches!(surface, HumanSurface::Skin | HumanSurface::Eye) {
            *offset = 0.0;
            continue;
        }
        if *offset > 0.006 {
            *offset = (*offset + drape.offset(vertices[i]) * scale).max(0.002);
        }
    }
    let mut garment_positions: Vec<_> = posed
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
    if !motion_rest {
        if let Some(contact) = &h.worktop_contact {
            for (arm, side) in ["L", "R"].into_iter().enumerate() {
                if !contact.palms[arm] {
                    continue;
                }
                let suffix = format!(".{side}");
                let mut report = super::contact::ContactReport {
                    arm,
                    minimum_gap_metres: f32::INFINITY,
                    palm_gap_metres: f32::INFINITY,
                    ..Default::default()
                };
                for (i, q) in garment_positions.iter_mut().enumerate() {
                    let name = &labels[dominant[i]];
                    if !name.ends_with(&suffix)
                        || !(name.starts_with("lowerarm")
                            || name.starts_with("wrist")
                            || name.starts_with("finger")
                            || name.starts_with("metacarpal"))
                        || !contact.contains(*q, 0.)
                    {
                        continue;
                    }
                    // Cloth can compress against a rigid top. This acts only on
                    // the supported limb; no geometry or semantic parts vanish.
                    let compression = (contact.height + 0.0005 - q.y).max(0.);
                    if cut.surface(h, vertices[i], name) == HumanSurface::Skin {
                        report.skin_compression_metres =
                            report.skin_compression_metres.max(compression);
                    } else {
                        report.cloth_compression_metres =
                            report.cloth_compression_metres.max(compression);
                    }
                    q.y += compression;
                    report.minimum_gap_metres = report.minimum_gap_metres.min(q.y - contact.height);
                    if super::contact::palm_region(&mesh.local_joints, arm, *q, h.stature / 1.75) {
                        report.palm_gap_metres = report.palm_gap_metres.min(q.y - contact.height);
                    }
                    report.contact_vertices += usize::from(q.y - contact.height < 0.003);
                }
                mesh.contacts.push(report);
            }
        }
    }
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
    for (face_index, (q, texture_indices)) in faces
        .as_chunks::<4>()
        .0
        .iter()
        .zip(uv_faces.as_chunks::<4>().0)
        .enumerate()
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
                skin_field: cut.skin_field(vertices[i], &labels[dominant[i]]),
            }
        });
        for tri in [[0, 1, 2], [0, 2, 3]] {
            let triangle = tri.map(|i| corners[i]);
            if super::anatomy::lip_face(face_index) {
                super::garments::emit(&triangle, HumanSurface::Lip, &mut mesh);
            } else if h.appearance.as_ref().is_some_and(|a| a.face.stubble > 0.0)
                && triangle.iter().all(|v| v.rest.y > cut.neck)
                && !labels[dominant[q[0] as usize]].starts_with("eye.")
            {
                super::face::stubble(triangle, (head("eye.L") + head("eye.R")) * 0.5, &mut mesh);
            } else {
                cut.append(h, &labels[dominant[q[0] as usize]], triangle, &mut mesh);
            }
        }
    }
    super::garments::details::append(
        h,
        &cut,
        &vertices,
        faces,
        &garment_positions,
        &normals,
        &mut mesh,
    );
    if !motion_rest {
        if let Some(contact) = &h.worktop_contact {
            contact.support_stitching(&mut mesh, h.stature / 1.75);
        }
    }
    // Replace recolored anatomical feet with a padded last and separate sole.
    // The rest body remains available for weight transfer and collision checks.
    mesh.parts.remove(&HumanSurface::Shoes);
    let footwear = h
        .appearance
        .as_ref()
        .map_or_else(Default::default, |a| a.footwear.clone());
    for side in ["L", "R"] {
        let suffix = format!(".{side}");
        let points: Vec<_> = vertices
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| {
                (p.y <= cut.shoe_top + footwear.collar_raise + 0.018
                    && labels[dominant[i]].ends_with(&suffix))
                .then_some(p)
            })
            .collect();
        super::footwear::append(
            &footwear,
            &points,
            super::footwear::FootFrame {
                ankle: head(&format!("foot.{side}")),
                toe: head(&format!("toe2-1.{side}")),
                posed_from_rest: transforms[index(&format!("foot.{side}"))],
                floor,
                shoe_top: cut.shoe_top,
            },
            &mut mesh,
        );
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
    }
    super::face::append(h, &posed, faces, &eyes, face_rotation, scale, &mut mesh);
    if h.outfit.buttoned() {
        for row in 0..6 {
            let y = if h.outfit == super::HumanOutfit::Polo {
                if row > 2 {
                    continue;
                }
                cut.neck - 0.035 - row as f32 * 0.027
            } else {
                cut.waist + (cut.chest - cut.waist) * (row as f32 + 0.5) / 6.0
            };
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
        super::hair::HairFrame {
            head: Mat4::from_translation(-Vec3::Y * floor) * neck,
            torso: Mat4::from_translation(-Vec3::Y * floor) * torso,
            head_origin: head("head"),
        },
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
    let rest = motion_rest.then(|| {
        let basis = Mat4::from_cols(
            Vec3::NEG_X.extend(0.0),
            Vec3::Z.extend(0.0),
            Vec3::Y.extend(0.0),
            Vec4::W,
        );
        let basis =
            Mat4::from_translation(-Vec3::Y * floor) * Mat4::from_scale(Vec3::splat(scale)) * basis;
        let bones = output
            .rest_bone_poses
            .data
            .as_chunks::<16>()
            .0
            .iter()
            .map(|m| {
                basis
                    * Mat4::from_cols_array(&std::array::from_fn(|i| m[(i % 4) * 4 + i / 4] as f32))
            })
            .collect();
        let mut indices = Vec::with_capacity(vertices.len());
        let mut sparse_weights = Vec::with_capacity(vertices.len());
        for i in 0..vertices.len() {
            let mut values: Vec<_> = (0..influences)
                .filter_map(|k| {
                    let at = i * influences + k;
                    (bone_ids.data[at] >= 0 && weights.data[at] > 0.0)
                        .then_some((bone_ids.data[at] as u16, weights.data[at] as f32))
                })
                .collect();
            values.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
            values.truncate(4);
            values.resize(4, (0, 0.0));
            let sum: f32 = values.iter().map(|v| v.1).sum();
            indices.push(std::array::from_fn(|k| values[k].0));
            sparse_weights.push(std::array::from_fn(|k| values[k].1 / sum.max(1e-8)));
        }
        RestSkin {
            phenotype,
            bones,
            positions: garment_positions,
            indices,
            weights: sparse_weights,
            scale,
        }
    });
    (mesh, rest)
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
    fn scalp_and_fibres_have_finite_orthogonal_tangent_frames() {
        use bevy::mesh::VertexAttributeValues;
        let human = super::super::sample_person(
            8646405506642607100,
            0,
            Vec3::ZERO,
            0.0,
            super::super::HumanPoseKind::StandingRelaxed,
            None,
            false,
        );
        let geometry = build(&human).parts.remove(&HumanSurface::Hair).unwrap();
        let mesh = geometry.into_mesh();
        let Some(VertexAttributeValues::Float32x4(tangents)) =
            mesh.attribute(Mesh::ATTRIBUTE_TANGENT)
        else {
            panic!("missing hair tangents")
        };
        let Some(VertexAttributeValues::Float32x3(normals)) =
            mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            panic!("missing hair normals")
        };
        for (t, n) in tangents.iter().zip(normals) {
            let t = Vec4::from_array(*t);
            let n = Vec3::from_array(*n);
            assert!(
                t.is_finite() && (t.truncate().length() - 1.0).abs() < 0.001,
                "invalid tangent {t}"
            );
            assert!(
                t.truncate().dot(n).abs() < 0.001,
                "nonorthogonal tangent frame {t} / {n}"
            );
        }
    }

    #[test]
    fn garments_preserve_anny_body_and_uv_seams_above_constructed_footwear() {
        let scene = IndoorManifest::generate(31, IndoorLayout::Mixed, 0.65, 1).unwrap();
        let body = reference();
        let original_vertices: HashSet<_> = body.faces_quads().data.iter().copied().collect();
        let data = &body.metadata().static_data;
        let (bone_ids, weights) = body.skinning_bindings();
        let influences = *bone_ids.shape.last().unwrap();
        let (labels, _) = body.bone_hierarchy();
        let retained = |vertex: i64| {
            let base = vertex as usize * influences;
            let k = (0..influences)
                .max_by(|&a, &b| weights.data[base + a].total_cmp(&weights.data[base + b]))
                .unwrap();
            let name = &labels[bone_ids.data[base + k] as usize];
            !["lowerleg", "foot", "toe"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
        };
        let expected_uvs: HashSet<_> = data
            .face_texture_coordinate_indices
            .data
            .as_chunks::<4>()
            .0
            .iter()
            .zip(body.faces_quads().data.as_chunks::<4>().0)
            .filter(|(_, q)| q.iter().all(|&i| retained(i)))
            .flat_map(|(uv, _)| uv)
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
            // Only feet are replaced by constructed shoe shells; the body and
            // its face-corner UV atlas remain Anny, including split cloth cuts.
            assert!(vertices.len() >= original_vertices.len() * 9 / 10);
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
