//! Admission against static triangle surfaces and conservative continuous swept
//! bounds. Reservations cover *all* progress values, so reversal and arbitrary
//! capture jumps do not weaken clearance or require runtime collision work.
use super::super::{layout::IndoorManifest, preparation::SceneGeometry};
use super::*;

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
                    .find(|s| s.starts_with("portal"))
                    .map(str::to_owned);
                let floor = super::super::objects::part_label(label) == "floor";
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
    fn hits(
        &self,
        sweep: Sweep,
        owner: Option<usize>,
        ids: &[usize],
        attachment: Option<&str>,
        floor: Option<f32>,
    ) -> bool {
        if self.nodes.is_empty() {
            return false;
        }
        let bb = sweep.world();
        let local_from_world = sweep.frame.compute_affine().inverse();
        let mut stack = vec![0];
        while let Some(id) = stack.pop() {
            let node = &self.nodes[id];
            if !overlaps(bb, node.bounds) {
                continue;
            }
            if let Some(children) = node.children {
                stack.extend(children);
                continue;
            }
            for t in &self.triangles[node.range.clone()] {
                if (t.owner == owner && t.joint.is_some_and(|j| ids.contains(&j)))
                    || (floor.is_some_and(|height| {
                        t.floor && t.p.iter().all(|p| (p.y - height).abs() < 0.002)
                    }))
                    || (owner.is_none()
                        && t.owner.is_none()
                        && t.joint.is_none()
                        && attachment.is_some()
                        && t.attachment.as_deref() == attachment)
                {
                    continue;
                }
                if triangle_box(
                    t.p.map(|p| local_from_world.transform_point3(p)),
                    sweep.local,
                ) {
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
    let local_from_assembly = joint.frame.compute_affine().inverse();
    for key in &joint.parts {
        let g = &a.parts[key];
        // Local triangle blocks keep handles, hinge barrels and separate slats
        // from turning into a fictitious solid slab around the whole mechanism.
        for indices in g.indices.chunks(96) {
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
                        .flat_map(|&i| [g.positions[i as usize], end.positions[i as usize]])
                        .map(|p| local_from_assembly.transform_point3(Vec3::from_array(p))),
                );
                result.push(Sweep {
                    local: b,
                    frame: tf.mul_transform(joint.frame),
                });
            } else {
                let b = bounds(indices.iter().map(|&i| {
                    local_from_assembly.transform_point3(Vec3::from_array(g.positions[i as usize]))
                }));
                let radius = corners(b)
                    .iter()
                    .map(|&p| joint.frame.transform_point(p).distance(joint.pivot))
                    .fold(0., f32::max);
                let pad = Vec3::splat(radius * (1. - (joint.angle.abs() / 128.).cos()) + 0.0001);
                for i in 0..64 {
                    let frame = tf
                        .mul_transform(joint.transform((i as f32 + 0.5) / 64.))
                        .mul_transform(joint.frame);
                    let inverse = frame.compute_affine().inverse();
                    let local = bounds(
                        [i as f32 / 64., (i + 1) as f32 / 64.]
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
    let tree = Tree::new(scene, geometry);
    let people: Vec<_> = scene.humans.iter().map(|p| p.bounds()).collect();
    let camera_bounds: Vec<_> = scene
        .cameras
        .iter()
        .map(|c| {
            // Bezier control hull and polyline vertices bound the complete trajectory.
            let mut points = vec![c.start, c.end];
            if let Some(m) = &c.motion {
                points.extend(m.control);
                points.extend(m.route.iter().copied());
            }
            let b = bounds(points.into_iter());
            (b.0 - Vec3::splat(0.18), b.1 + Vec3::splat(0.18))
        })
        .collect();
    let mut reports = vec![];
    let mut reserved: Vec<Bounds> = vec![];
    let mut accepted = 0;
    for (owner, tf, a) in std::iter::once((None, Transform::IDENTITY, &mut geometry.architecture))
        .chain(
            scene
                .objects
                .iter()
                .zip(&mut geometry.objects)
                .map(|(o, a)| (Some(o.id), o.transform(), a)),
        )
    {
        let mut index = 0;
        while index < a.joints.len() {
            let joint = &a.joints[index];
            // Seat and caster base move as one mechanism; never accept half.
            let count = if joint.kind == MotionKind::Chair || joint.coupled {
                a.joints
                    .iter()
                    .skip(index)
                    .take_while(|j| {
                        j.kind == joint.kind
                            && j.coupled == joint.coupled
                            && j.attachment == joint.attachment
                    })
                    .count()
            } else {
                1
            };
            let ids: Vec<_> = (index..index + count).collect();
            let shapes: Vec<_> = if accepted < limit {
                ids.iter()
                    .flat_map(|&i| sweeps(a, &a.joints[i], tf))
                    .collect()
            } else {
                vec![]
            };
            let swept: Vec<_> = shapes.iter().map(|s| s.world()).collect();
            let reason = if accepted >= limit {
                Some("mechanism_limit")
            } else if swept.is_empty() {
                Some("no_travel")
            } else if joint
                .morph
                .iter()
                .any(|(k, g)| !validate_morph(&a.parts[k], g))
            {
                Some("deformation_topology")
            } else if owner.is_some_and(|id| {
                scene.humans.iter().any(|h| h.chair == Some(id))
                    || (joint.kind == MotionKind::Chair
                        && scene.objects.iter().any(|o| o.support == Some(id)))
            }) {
                Some("occupied_support")
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
            } else if swept
                .iter()
                .any(|&b| reserved.iter().any(|&p| overlaps(b, p)))
            {
                Some("other_motion")
            } else if shapes.iter().any(|&shape| {
                tree.hits(
                    shape,
                    owner,
                    &ids,
                    joint.attachment.as_deref(),
                    (joint.kind == MotionKind::Chair).then_some(tf.translation.y),
                )
            }) {
                Some("static_geometry")
            } else if joint.kind == MotionKind::Chair
                && swept.iter().any(|&b| {
                    scene
                        .envelope
                        .as_ref()
                        .is_some_and(|e| !e.support_clear(b.0.with_y(tf.translation.y), b.1))
                })
            {
                Some("uneven_floor")
            } else {
                None
            };
            let compact: Vec<_> = swept
                .chunks(swept.len().div_ceil(32).max(1))
                .map(|s| s.iter().copied().reduce(union).unwrap())
                .collect();
            if reason.is_none() {
                accepted += 1;
                reserved.extend(compact.iter().copied());
            }
            for &id in &ids {
                a.joints[id].accepted = reason.is_none();
            }
            let j = &a.joints[index];
            reports.push(MechanismReport {
                object_id: owner,
                joint: index,
                kind: j.kind,
                interval: j.interval,
                rejection: reason.map(str::to_owned),
                swept_bounds: if reason.is_none() { compact } else { vec![] },
                translation_m: tf.rotation * j.translation,
                angle_radians: j.angle,
                pivot: tf.transform_point(j.pivot),
                axis: tf.rotation * j.axis,
            });
            index += count;
        }
    }
    scene.indoor_motion.as_mut().unwrap().mechanisms = reports;
}

/// A linear vertex morph gives a quadratic area vector. Its projection onto
/// the original normal has an analytic minimum; positive area certifies that
/// no time between the endpoints contains a collapsed/inverted triangle.
pub(crate) fn validate_morph(a: &Geometry, b: &Geometry) -> bool {
    if a.indices != b.indices || a.positions.len() != b.positions.len() {
        return false;
    }
    a.indices.as_chunks::<3>().0.iter().all(|ids| {
        let p: [Vec3; 3] = std::array::from_fn(|i| Vec3::from_array(a.positions[ids[i] as usize]));
        let d: [Vec3; 3] =
            std::array::from_fn(|i| Vec3::from_array(b.positions[ids[i] as usize]) - p[i]);
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
mod tests {
    use super::*;
    fn empty() -> (IndoorManifest, SceneGeometry) {
        let mut scene = IndoorManifest::generate_with_humans(0, default(), 0.65, 0, 0.).unwrap();
        scene.objects.clear();
        scene.humans.clear();
        scene.cameras.clear();
        scene.indoor_motion = Some(IndoorMotionReport::new(default()));
        (
            scene,
            SceneGeometry {
                architecture: default(),
                objects: vec![],
                humans: vec![],
            },
        )
    }
    fn block(center: Vec3, size: Vec3) -> Assembly {
        let mut a = Assembly::default();
        a.box_part(Surface::Wood, "door", center, size, 0.);
        a
    }
    #[test]
    fn full_arc_clearance_catches_obstacles_between_clear_endpoints() {
        let (mut scene, mut geometry) = empty();
        let a = &mut geometry.architecture;
        a.append_assembly(block(Vec3::new(0.71, 1., -0.71), Vec3::splat(0.10)));
        a.articulated(
            block(Vec3::new(1., 1., 0.), Vec3::splat(0.04)),
            construction::rigid(
                MotionKind::Door,
                1,
                Vec3::new(0., 1., 0.),
                Vec3::Y,
                std::f32::consts::FRAC_PI_2,
                Vec3::ZERO,
            ),
        );
        prepare(&mut scene, &mut geometry);
        assert_eq!(
            scene.indoor_motion.unwrap().mechanisms[0]
                .rejection
                .as_deref(),
            Some("static_geometry")
        );
    }
    #[test]
    fn crossing_mechanisms_cannot_both_be_accepted() {
        let (mut scene, mut geometry) = empty();
        for (p, delta) in [
            (Vec3::new(-1., 1., 0.), Vec3::X * 2.),
            (Vec3::new(0., 1., -1.), Vec3::Z * 2.),
        ] {
            geometry.architecture.articulated(
                block(p, Vec3::splat(0.08)),
                construction::rigid(MotionKind::Door, 5, Vec3::ZERO, Vec3::Y, 0., delta),
            );
        }
        prepare(&mut scene, &mut geometry);
        let report = scene.indoor_motion.unwrap();
        assert!(report.mechanisms[0].rejection.is_none());
        assert_eq!(
            report.mechanisms[1].rejection.as_deref(),
            Some("other_motion")
        );
    }
    #[test]
    fn coupled_rotor_leaves_are_admitted_atomically() {
        let (mut scene, mut geometry) = empty();
        for i in 0..4 {
            let p =
                Quat::from_rotation_y(i as f32 * std::f32::consts::FRAC_PI_2) * Vec3::X + Vec3::Y;
            let mut joint =
                construction::rigid(MotionKind::Door, 5, Vec3::Y, Vec3::Y, 2.0, Vec3::ZERO);
            joint.coupled = true;
            joint.attachment = Some("portal0".into());
            geometry
                .architecture
                .articulated(block(p, Vec3::splat(0.05)), joint);
        }
        prepare(&mut scene, &mut geometry);
        assert_eq!(scene.indoor_motion.unwrap().mechanisms.len(), 1);
        assert!(geometry.architecture.joints.iter().all(|j| j.accepted));
    }
    #[test]
    fn arc_enclosures_contain_every_intermediate_vertex() {
        let mut a = Assembly::default();
        let j = construction::rigid(
            MotionKind::Door,
            1,
            Vec3::new(0.15, 0.5, -0.2),
            Vec3::Y,
            2.1,
            Vec3::X * 0.4,
        );
        a.articulated(
            block(Vec3::new(0.6, 0.8, 0.1), Vec3::new(0.9, 1.2, 0.06)),
            j,
        );
        let j = &a.joints[0];
        let envelopes = sweep(&a, j, Transform::IDENTITY);
        for i in 0..=8192 {
            let tf = j.transform(i as f32 / 8192.);
            for g in a.parts.values() {
                for p in &g.positions {
                    let p = tf.transform_point(Vec3::from_array(*p));
                    assert!(envelopes
                        .iter()
                        .any(|(a, b)| p.cmpge(*a).all() && p.cmple(*b).all()));
                }
            }
        }
    }
    #[test]
    fn collapsed_mid_morph_is_rejected_even_when_endpoints_are_valid() {
        let a = Geometry {
            positions: vec![[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]],
            indices: vec![0, 1, 2],
            ..default()
        };
        let mut b = a.clone();
        b.positions[1] = [-1., 0., 0.];
        b.positions[2] = [0., -1., 0.];
        assert!(!validate_morph(&a, &b));
        assert!(validate_morph(&a, &a));
    }
}
