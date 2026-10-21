//! Admission against static triangle surfaces and conservative continuous swept
//! bounds. Reservations cover *all* progress values, so reversal and arbitrary
//! capture jumps do not weaken clearance or require runtime collision work.
use super::super::{layout::IndoorManifest, preparation::SceneGeometry};
use super::*;
mod hull;
use hull::TriangleHull;

#[derive(Clone)]
struct Triangle {
    p: [Vec3; 3],
    owner: Option<usize>,
    joint: Option<usize>,
    attachment: Option<String>,
    floor: bool,
}
struct Node {
    bounds: Bounds,
    range: std::ops::Range<usize>,
    children: Option<[usize; 2]>,
}

#[derive(Clone, Copy)]
struct Sweep {
    local: Bounds,
    frame: Transform,
    hull: Option<TriangleHull>,
}
impl Sweep {
    fn world(self) -> Bounds {
        transformed(self.local, self.frame)
    }
}
struct Tree {
    triangles: Vec<Triangle>,
    nodes: Vec<Node>,
}
fn bounds(points: impl Iterator<Item = Vec3>) -> Bounds {
    points.fold(
        (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
        |(a, b), p| (a.min(p), b.max(p)),
    )
}
fn union(a: Bounds, b: Bounds) -> Bounds {
    (a.0.min(b.0), a.1.max(b.1))
}
fn overlaps(a: Bounds, b: Bounds) -> bool {
    a.0.cmplt(b.1).all() && a.1.cmpgt(b.0).all()
}
fn transformed(b: Bounds, t: Transform) -> Bounds {
    bounds(corners(b).into_iter().map(|p| t.transform_point(p)))
}
fn corners((lo, hi): Bounds) -> [Vec3; 8] {
    std::array::from_fn(|i| {
        Vec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        )
    })
}
impl Tree {
    fn new(scene: &IndoorManifest, g: &SceneGeometry) -> Self {
        let mut triangles = vec![];
        for (owner, tf, a) in std::iter::once((None, Transform::IDENTITY, &g.architecture)).chain(
            scene
                .objects
                .iter()
                .zip(&g.objects)
                .map(|(o, a)| (Some(o.id), o.transform(), a)),
        ) {
            for ((_, label), g) in &a.parts {
                let joint = tag(label);
                let attachment = label
                    .split('#')
                    .find(|s| s.starts_with("portal") || s.starts_with("contact_"))
                    .map(str::to_owned);
                let floor = matches!(
                    super::super::objects::part_label(label),
                    "floor" | "floormat"
                );
                for tri in g.indices.as_chunks::<3>().0 {
                    triangles.push(Triangle {
                        p: std::array::from_fn(|i| {
                            tf.transform_point(Vec3::from_array(g.positions[tri[i] as usize]))
                        }),
                        owner,
                        joint,
                        attachment: attachment.clone(),
                        floor,
                    });
                }
            }
        }
        let mut result = Self {
            triangles,
            nodes: vec![],
        };
        if !result.triangles.is_empty() {
            result.build(0..result.triangles.len());
        }
        result
    }
    fn build(&mut self, range: std::ops::Range<usize>) -> usize {
        let bb = bounds(self.triangles[range.clone()].iter().flat_map(|t| t.p));
        let id = self.nodes.len();
        self.nodes.push(Node {
            bounds: bb,
            range: range.clone(),
            children: None,
        });
        if range.len() > 12 {
            let d = bb.1 - bb.0;
            let axis = if d.x > d.y && d.x > d.z {
                0
            } else if d.y > d.z {
                1
            } else {
                2
            };
            let mid = range.start + range.len() / 2;
            self.triangles[range.clone()].select_nth_unstable_by(mid - range.start, |a, b| {
                let center = |t: &Triangle| t.p.iter().map(|p| p[axis]).sum::<f32>();
                center(a).total_cmp(&center(b))
            });
            let left = self.build(range.start..mid);
            let right = self.build(mid..range.end);
            self.nodes[id].children = Some([left, right]);
        }
        id
    }
    #[cfg(test)]
    fn hits(
        &self,
        sweep: Sweep,
        owner: Option<usize>,
        ids: &[usize],
        attachment: Option<&str>,
        floor: Option<f32>,
    ) -> bool {
        self.hits_context(sweep, owner, ids, attachment, floor, None, &[])
    }
    #[allow(clippy::too_many_arguments)]
    fn hits_context(
        &self,
        sweep: Sweep,
        owner: Option<usize>,
        ids: &[usize],
        attachment: Option<&str>,
        floor: Option<f32>,
        support: Option<(usize, f32)>,
        carried: &[usize],
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let bb = sweep.world();
        let local_from_world = sweep.frame.compute_affine().inverse();
        // Median-split BVH depth is bounded by the number of bits in usize.
        // Keep the traversal stack local: fallback searches visit this hot path
        // thousands of times and must not allocate once per swept block.
        let mut stack = [0usize; usize::BITS as usize];
        let mut pending = 1;
        while pending > 0 {
            pending -= 1;
            let id = stack[pending];
            let node = &self.nodes[id];
            if !overlaps(bb, node.bounds) {
                continue;
            }
            if let Some(children) = node.children {
                stack[pending..pending + 2].copy_from_slice(&children);
                pending += 2;
                continue;
            }
            for t in &self.triangles[node.range.clone()] {
                if t.owner.is_some_and(|o| carried.contains(&o))
                    || support.is_some_and(|(id, height)| {
                        t.owner == Some(id) && t.p.iter().all(|p| (p.y - height).abs() < 0.0002)
                    })
                    || (t.owner == owner && t.joint.is_some_and(|j| ids.contains(&j)))
                    || (floor.is_some_and(|height| {
                        t.floor && t.p.iter().all(|p| (p.y - height).abs() < 0.002)
                    }))
                    || (owner == t.owner
                        && t.joint.is_none()
                        && attachment.is_some()
                        && t.attachment.as_deref() == attachment)
                {
                    continue;
                }
                let p = t.p.map(|p| local_from_world.transform_point3(p));
                if triangle_box(p, sweep.local) && sweep.hull.is_none_or(|hull| hull.intersects(p))
                {
                    return true;
                }
            }
        }
        false
    }
}
/// Separating axis theorem, including triangle edges and normal. A triangle's
/// large AABB alone must not reject a chair beside a diagonal wall/floor.
fn triangle_box(p: [Vec3; 3], (lo, hi): Bounds) -> bool {
    let center = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    let p = p.map(|p| p - center);
    let edges = [p[1] - p[0], p[2] - p[1], p[0] - p[2]];
    let separated = |axis: Vec3| {
        let projected = p.map(|p| axis.dot(p));
        let r = half.dot(axis.abs());
        projected.iter().copied().fold(f32::INFINITY, f32::min) > r
            || projected.iter().copied().fold(f32::NEG_INFINITY, f32::max) < -r
    };
    ![Vec3::X, Vec3::Y, Vec3::Z, edges[0].cross(edges[1])]
        .into_iter()
        .chain(
            edges
                .into_iter()
                .flat_map(|e| [e.cross(Vec3::X), e.cross(Vec3::Y), e.cross(Vec3::Z)]),
        )
        .any(separated)
}

fn sweeps(a: &Assembly, joint: &Joint, tf: Transform) -> Vec<Sweep> {
    let mut result = vec![];
    if joint.kind == MotionKind::Fan {
        // Bound the complete rotor at any speed, including many revolutions.
        // AABB vertices at finitely sampled angles cannot certify a fast rotor.
        let points: Vec<_> = joint
            .parts
            .iter()
            .flat_map(|k| &a.parts[k].positions)
            .map(|p| Vec3::from_array(*p))
            .collect();
        let b = bounds(points.iter().copied());
        let radius = points
            .iter()
            .map(|p| (*p - joint.pivot).xz().length())
            .fold(0., f32::max)
            + 0.0001;
        return vec![Sweep {
            local: (
                Vec3::new(
                    joint.pivot.x - radius,
                    b.0.y - 0.0001,
                    joint.pivot.z - radius,
                ),
                Vec3::new(
                    joint.pivot.x + radius,
                    b.1.y + 0.0001,
                    joint.pivot.z + radius,
                ),
            ),
            frame: tf,
            hull: None,
        }];
    }
    let local_from_assembly = joint.frame.compute_affine().inverse();
    for key in &joint.parts {
        let g = &a.parts[key];
        // Local triangle blocks keep handles, hinge barrels and separate slats
        // from turning into a fictitious solid slab around the whole mechanism.
        // Thin articulated frames have nearby stationary carcasses. Large
        // triangle batches can enclose empty space and falsely reject every
        // possible angle. Bound each face for these compact mechanisms.
        let block = if matches!(joint.kind, MotionKind::Cabinet | MotionKind::Appliance)
            || (joint.kind == MotionKind::Prop && joint.contact == Contact::Free)
        {
            3
        } else {
            96
        };
        for indices in g.indices.chunks(block) {
            if let Some(end) = joint.morph.get(key) {
                if indices
                    .iter()
                    .all(|&i| g.positions[i as usize] == end.positions[i as usize])
                {
                    continue;
                }
                let b = bounds(
                    indices
                        .iter()
                        .flat_map(|&i| {
                            let start = Vec3::from_array(g.positions[i as usize]);
                            [
                                start,
                                start.lerp(
                                    Vec3::from_array(end.positions[i as usize]),
                                    joint.adjustment.morph,
                                ),
                            ]
                        })
                        .map(|p| local_from_assembly.transform_point3(p)),
                );
                result.push(Sweep {
                    local: b,
                    frame: tf.mul_transform(joint.frame),
                    hull: None,
                });
            } else {
                if joint.effective_translation() == Vec3::ZERO && joint.effective_angle() == 0. {
                    continue;
                }
                let b = bounds(indices.iter().map(|&i| {
                    local_from_assembly.transform_point3(Vec3::from_array(g.positions[i as usize]))
                }));
                let radius = corners(b)
                    .iter()
                    .map(|&p| joint.frame.transform_point(p).distance(joint.pivot))
                    .fold(0., f32::max);
                let steps = (joint.effective_angle().abs() / 0.04).ceil().clamp(1., 96.) as usize;
                let step = 1. / steps as f32;
                let pad = Vec3::splat(
                    radius * (1. - (joint.effective_angle().abs() * step * 0.5).cos()) + 0.0001,
                );
                for i in 0..steps {
                    let frame = tf
                        .mul_transform(joint.transform((i as f32 + 0.5) * step))
                        .mul_transform(joint.frame);
                    let inverse = frame.compute_affine().inverse();
                    let local = bounds(
                        [i as f32 * step, (i + 1) as f32 * step]
                            .into_iter()
                            .flat_map(|t| {
                                let pose = tf
                                    .mul_transform(joint.transform(t))
                                    .mul_transform(joint.frame);
                                corners(b)
                                    .map(|p| inverse.transform_point3(pose.transform_point(p)))
                            }),
                    );
                    result.push(Sweep {
                        local: (local.0 - pad, local.1 + pad),
                        frame,
                        hull: (block == 3).then(|| TriangleHull {
                            points: std::array::from_fn(|k| {
                                let amount = (i + k / 3) as f32 * step;
                                let pose = tf.mul_transform(joint.transform(amount));
                                inverse.transform_point3(pose.transform_point(Vec3::from_array(
                                    g.positions[indices[k % 3] as usize],
                                )))
                            }),
                            error: pad.x,
                        }),
                    });
                }
            }
        }
    }
    result
}
#[cfg(test)]
fn sweep(a: &Assembly, joint: &Joint, tf: Transform) -> Vec<Bounds> {
    sweeps(a, joint, tf).into_iter().map(Sweep::world).collect()
}

/// Tight continuous path hulls, including cubic curvature between samples and
/// every original waypoint segment (short corners cannot disappear in sampling).
fn camera_reservations(scene: &IndoorManifest) -> Vec<Bounds> {
    use crate::camera::TrajectorySampler;
    let mut result = vec![];
    for c in &scene.cameras {
        if let Some(mount) = &c.ceiling_mount {
            result.push(mount.bounds());
        }
        let mut path = c.runtime_trajectory();
        let clearance = super::super::layout::CAMERA_CLEARANCE.max(
            super::super::cameras::intrinsics::near_clearance(
                c.fov_degrees,
                scene.camera_aspect_ratio,
                c.principal_point_offset
                    .unwrap_or(scene.camera_settings.principal_point_jitter),
            ),
        );
        let (points, error) = match &path {
            TrajectorySampler::CubicBezier { positions, .. } => {
                let curvature = 6.
                    * (positions[0] - 2. * positions[1] + positions[2])
                        .length()
                        .max((positions[1] - 2. * positions[2] + positions[3]).length());
                (
                    (0..=32)
                        .map(|i| path.sample(i as f32 / 32.).translation)
                        .collect::<Vec<_>>(),
                    curvature / (8. * 32. * 32.),
                )
            }
            TrajectorySampler::WaypointPath { positions, .. } => (positions.clone(), 0.),
            _ => (vec![c.start, c.end], 0.),
        };
        let pad = Vec3::splat(clearance + error);
        result.extend(
            points
                .windows(2)
                .map(|p| (p[0].min(p[1]) - pad, p[0].max(p[1]) + pad)),
        );
    }
    result
}

/// Certify the full continuous footprint, including turns, on the existing
/// support. The normal layout gate understands round/cutout tabletops and islands.
fn support_clear(
    scene: &IndoorManifest,
    o: &super::super::layout::IndoorObject,
    shapes: &[Sweep],
) -> bool {
    let Some(support) = o.support.and_then(|id| scene.objects.get(id)) else {
        return false;
    };
    let inverse = support.transform().compute_affine().inverse();
    let (lo, hi) = bounds(shapes.iter().flat_map(|s| {
        corners(s.local).map(|p| inverse.transform_point3(s.frame.transform_point(p)))
    }));
    let mut proxy = o.clone();
    proxy.yaw = support.yaw;
    proxy.size = (hi - lo).with_y(o.size.y);
    proxy.position = support
        .transform()
        .transform_point((lo + hi) * 0.5)
        .with_y(o.position.y);
    scene.prop_clear(&proxy, support, 0.001)
}

pub(crate) fn prepare(scene: &mut IndoorManifest, geometry: &mut SceneGeometry) {
    let Some(report) = &scene.indoor_motion else {
        return;
    };
    if geometry.architecture.joints.is_empty()
        && geometry.objects.iter().all(|a| a.joints.is_empty())
    {
        return;
    }
    let limit = report.policy.max_mechanisms;
    let cargo = mobility::cargo(scene, geometry);
    let carried: std::collections::HashSet<_> = cargo.iter().flatten().copied().collect();
    // Cargo proxies are constructed once, not once per fallback or mesh part.
    let cargo_geometry: Vec<_> = cargo
        .iter()
        .enumerate()
        .map(|(id, children)| {
            let mut a = Assembly::default();
            if children.is_empty() {
                return a;
            }
            let inverse = scene.objects[id].transform().compute_affine().inverse();
            for &child in children {
                let (lo, hi) = geometry.objects[child].bounds();
                let relative = Transform::from_matrix(
                    Mat4::from(inverse) * scene.objects[child].transform().to_matrix(),
                );
                a.part(Surface::Wood, "cargo").cuboid(
                    hi - lo,
                    0.,
                    relative.mul_transform(Transform::from_translation((lo + hi) * 0.5)),
                );
            }
            a
        })
        .collect();
    let tree = Tree::new(scene, geometry);
    let people: Vec<_> = scene.humans.iter().map(|p| p.bounds()).collect();
    let camera_bounds = camera_reservations(scene);
    let mut reports = vec![];
    let mut reserved: Vec<(Bounds, Vec<Bounds>)> = vec![];
    let mut accepted = 0;
    let mut order = vec![];
    for (owner, a) in std::iter::once((None, &geometry.architecture)).chain(
        geometry
            .objects
            .iter()
            .enumerate()
            .map(|(i, a)| (Some(i), a)),
    ) {
        if owner.is_some_and(|id| carried.contains(&id)) {
            continue;
        }
        let mut index = 0;
        while index < a.joints.len() {
            let j = &a.joints[index];
            let count = if j.kind == MotionKind::Chair || j.coupled {
                a.joints[index..]
                    .iter()
                    .take_while(|other| {
                        other.kind == j.kind
                            && other.coupled == j.coupled
                            && other.attachment == j.attachment
                    })
                    .count()
            } else {
                1
            };
            // Stable per-mechanism priority prevents architecture from consuming
            // every slot before chairs/fans.
            let priority: u64 = stream(
                scene.seed ^ owner.map_or(u64::MAX, |i| i as u64),
                0x41444d49 ^ index as u64,
            )
            .random();
            order.push((priority, owner, index, count));
            index += count;
        }
    }
    order.sort_by_key(|&(priority, owner, index, _)| (priority, owner, index));
    for (_, owner, index, count) in order {
        let tf = owner.map_or(Transform::IDENTITY, |i| scene.objects[i].transform());
        let a = match owner {
            None => &mut geometry.architecture,
            Some(i) => &mut geometry.objects[i],
        };
        let ids: Vec<_> = (index..index + count).collect();
        for &id in &ids {
            a.joints[id].adjustment = default();
            a.joints[id].accepted = false;
        }
        let j = &a.joints[index];
        let children = owner.map_or(&[][..], |id| cargo[id].as_slice());
        let grounded = j.contact != Contact::Free || j.kind == MotionKind::Chair;
        let connected = grounded
            && owner.is_some_and(|id| {
                scene.objects.iter().any(|o| {
                    o.kind == super::super::layout::ObjectKind::Cable
                        && [o.cable_target, o.interaction_target]
                            .into_iter()
                            .flatten()
                            .any(|target| target == id || children.contains(&target))
                })
            });
        let occupied = owner.is_some_and(|id| {
            scene.humans.iter().any(|h| h.chair == Some(id))
                || (grounded
                    && j.kind != MotionKind::Cart
                    && scene.objects.iter().any(|o| o.support == Some(id)))
        });
        let mut reason = if accepted >= limit {
            Some("mechanism_limit")
        } else if connected {
            Some("connected_cable")
        } else if occupied {
            Some("occupied_support")
        } else {
            None
        };
        let mut attempts = 0;
        let mut adaptation = None;
        let mut compact = vec![];
        if reason.is_none() {
            for (adjustment, name) in proposals::candidates(&a.joints[index..index + count]) {
                attempts += 1;
                for &id in &ids {
                    a.joints[id].adjustment = adjustment;
                }
                let joint = &a.joints[index];
                let mut shapes: Vec<_> = ids
                    .iter()
                    .flat_map(|&i| sweeps(a, &a.joints[i], tf))
                    .collect();
                if let Some(id) = owner.filter(|_| !children.is_empty()) {
                    let mut proxy = joint.clone();
                    proxy.parts = cargo_geometry[id].parts.keys().cloned().collect();
                    shapes.extend(sweeps(&cargo_geometry[id], &proxy, tf));
                }
                let swept: Vec<_> = shapes.iter().map(|s| s.world()).collect();
                reason = if swept.is_empty() {
                    Some("no_travel")
                } else if ids.iter().any(|&id| {
                    a.joints[id]
                        .morph
                        .iter()
                        .any(|(k, g)| !validate_morph_to(&a.parts[k], g, adjustment.morph))
                }) {
                    Some("deformation_topology")
                } else if swept
                    .iter()
                    .any(|&b| people.iter().any(|&p| overlaps(b, p)))
                {
                    Some("static_person")
                } else if swept
                    .iter()
                    .any(|&b| camera_bounds.iter().any(|&p| overlaps(b, p)))
                {
                    Some("camera_path")
                } else if swept.iter().any(|&b| {
                    reserved.iter().any(|(all, parts)| {
                        overlaps(b, *all) && parts.iter().any(|&p| overlaps(b, p))
                    })
                }) {
                    Some("other_motion")
                } else if shapes.iter().any(|&shape| {
                    tree.hits_context(
                        shape,
                        owner,
                        &ids,
                        joint.attachment.as_deref(),
                        (joint.kind == MotionKind::Chair || joint.contact == Contact::Floor)
                            .then_some(tf.translation.y),
                        owner
                            .filter(|_| joint.contact == Contact::Surface)
                            .and_then(|id| {
                                scene.objects[id].support.map(|s| (s, tf.translation.y))
                            }),
                        children,
                    )
                }) {
                    Some("static_geometry")
                } else if (joint.kind == MotionKind::Chair || joint.contact == Contact::Floor)
                    && swept
                        .iter()
                        .any(|&b| !scene.floor_support_clear(b.0.with_y(tf.translation.y), b.1))
                {
                    Some("uneven_floor")
                } else if owner.is_some_and(|id| {
                    joint.contact == Contact::Surface
                        && !support_clear(scene, &scene.objects[id], &shapes)
                }) {
                    Some("support_edge")
                } else {
                    None
                };
                if reason.is_none() {
                    compact = swept
                        .chunks(swept.len().div_ceil(32).max(1))
                        .map(|s| s.iter().copied().reduce(union).unwrap())
                        .collect();
                    adaptation = (name != "requested").then(|| name.to_owned());
                    break;
                }
            }
        }
        if reason.is_none() {
            accepted += 1;
            // Most independently moving items are far apart. Reject disjoint
            // mechanism bounds before scanning their detailed reservations.
            // This is exactly the same predicate as the flat scan.
            reserved.push((
                compact.iter().copied().reduce(union).unwrap(),
                compact.clone(),
            ));
        }
        for &id in &ids {
            a.joints[id].accepted = reason.is_none();
        }
        let j = &a.joints[index];
        reports.push(MechanismReport {
            object_id: owner,
            carried_object_ids: children.to_vec(),
            semantic_labels: ids
                .iter()
                .flat_map(|&i| a.joints[i].parts.iter())
                .map(|(_, label)| super::super::objects::part_label(label).to_owned())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            joint: index,
            kind: j.kind,
            interval: if j.kind == MotionKind::Fan {
                [0., 1.]
            } else {
                j.interval
            },
            rejection: reason.map(str::to_owned),
            swept_bounds: compact,
            translation_m: tf.rotation * j.effective_translation(),
            angle_radians: j.effective_angle(),
            pivot: tf.transform_point(j.pivot),
            axis: tf.rotation * j.axis,
            attempts,
            adaptation,
            requested_translation_m: tf.rotation * j.translation,
            requested_angle_radians: j.angle,
        });
    }
    for report in &mut reports {
        for &child in &report.carried_object_ids {
            report.semantic_labels.extend(
                geometry.objects[child]
                    .parts
                    .keys()
                    .map(|(_, label)| super::super::objects::part_label(label).to_owned()),
            );
        }
        report.semantic_labels.sort();
        report.semantic_labels.dedup();
    }
    mobility::carry(scene, geometry, &cargo);
    reports.sort_by_key(|r| (r.object_id, r.joint));
    scene.indoor_motion.as_mut().unwrap().mechanisms = reports;
}

/// A linear vertex morph gives a quadratic area vector. Its projection onto
/// the original normal has an analytic minimum; positive area certifies that
/// no time between the endpoints contains a collapsed/inverted triangle.
#[cfg(test)]
pub(crate) fn validate_morph(a: &Geometry, b: &Geometry) -> bool {
    validate_morph_to(a, b, 1.)
}
fn validate_morph_to(a: &Geometry, b: &Geometry, amount: f32) -> bool {
    if a.indices != b.indices || a.positions.len() != b.positions.len() {
        return false;
    }
    a.indices.as_chunks::<3>().0.iter().all(|ids| {
        let p: [Vec3; 3] = std::array::from_fn(|i| Vec3::from_array(a.positions[ids[i] as usize]));
        let d: [Vec3; 3] = std::array::from_fn(|i| {
            (Vec3::from_array(b.positions[ids[i] as usize]) - p[i]) * amount
        });
        let u = p[1] - p[0];
        let v = p[2] - p[0];
        let du = d[1] - d[0];
        let dv = d[2] - d[0];
        let normal = u.cross(v);
        let area = normal.length();
        if area < 1e-12 {
            return false;
        }
        let normal = normal / area;
        let linear = (du.cross(v) + u.cross(dv)).dot(normal);
        let quadratic = du.cross(dv).dot(normal);
        let mut minimum = area.min(area + linear + quadratic);
        if quadratic > 0. {
            let t = (-linear / (2. * quadratic)).clamp(0., 1.);
            minimum = minimum.min(area + linear * t + quadratic * t * t);
        }
        minimum > area * 1e-4 && minimum.is_finite()
    })
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod mobility_tests;
