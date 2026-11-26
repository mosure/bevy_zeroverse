//! Admission against static triangle surfaces and conservative continuous swept
//! bounds. Reservations cover *all* progress values, so reversal and arbitrary
//! capture jumps do not weaken clearance or require runtime collision work.
use super::super::{layout::IndoorManifest, preparation::SceneGeometry};
use super::*;
mod board;
mod hull;
mod rotor;
use hull::TriangleHull;

const ROTATION_STEP_RADIANS: f32 = 0.04;

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
struct RestingContact {
    support: usize,
    height: f32,
    slides_on_shelf: bool,
}

#[derive(Clone, Copy)]
struct Sweep {
    local: Bounds,
    frame: Transform,
    shape: SweepShape,
}
#[derive(Clone, Copy)]
enum SweepShape {
    Box,
    Hull(TriangleHull),
    Rotor,
}
impl Sweep {
    fn world(self) -> Bounds {
        if matches!(self.shape, SweepShape::Rotor) {
            rotor::world_bounds(self.local, self.frame)
        } else {
            transformed(self.local, self.frame)
        }
    }
    fn intersects(self, triangle: [Vec3; 3]) -> bool {
        match self.shape {
            SweepShape::Box => triangle_box(triangle, self.local),
            SweepShape::Hull(h) => triangle_box(triangle, self.local) && h.intersects(triangle),
            SweepShape::Rotor => rotor::intersects(triangle, self.local),
        }
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
        Self::with_region(scene, g, None)
    }
    fn for_admission(scene: &IndoorManifest, g: &SceneGeometry) -> Self {
        if scene.generator_version < ROTOR_CARGO_VERSION
            || std::iter::once(&g.architecture)
                .chain(&g.objects)
                .flat_map(|a| &a.joints)
                .any(|j| {
                    !matches!(j.kind, MotionKind::Fan | MotionKind::TeachingBoard)
                        && !matches!(j.cloth_motion, Some(ClothMotion::ProjectionScreen { .. }))
                })
        {
            return Self::new(scene, g);
        }
        let assemblies = std::iter::once((Transform::IDENTITY, &g.architecture)).chain(
            scene
                .objects
                .iter()
                .zip(&g.objects)
                .filter(|(_, a)| !a.joints.is_empty())
                .map(|(o, a)| (o.transform(), a)),
        );
        // Bound all retries, including re-admission of a shortened operation.
        let region = assemblies
            .flat_map(|(tf, a)| {
                a.joints.iter().filter_map(move |j| {
                    if j.kind == MotionKind::TeachingBoard {
                        return board::search_bounds(a, j, tf);
                    }
                    let shapes = match j.cloth_motion {
                        Some(ClothMotion::ProjectionScreen { feed, .. }) => {
                            screen_sweeps(a, j, tf, feed, 1.)
                        }
                        _ => sweeps(a, j, tf),
                    };
                    shapes.into_iter().map(Sweep::world).reduce(union)
                })
            })
            .reduce(union);
        Self::with_region(scene, g, region)
    }
    fn with_region(scene: &IndoorManifest, g: &SceneGeometry, region: Option<Bounds>) -> Self {
        let mut triangles = vec![];
        for (owner, tf, a) in std::iter::once((None, Transform::IDENTITY, &g.architecture)).chain(
            scene
                .objects
                .iter()
                .zip(&g.objects)
                .map(|(o, a)| (Some(o.id), o.transform(), a)),
        ) {
            let region = region.map(|bb| {
                let inverse = tf.compute_affine().inverse();
                let (lo, hi) = bounds(corners(bb).into_iter().map(|p| inverse.transform_point3(p)));
                // Leave room for roundoff between the local broad phase and
                // the world-to-sweep transform in the triangle narrow phase.
                (lo - 0.0001, hi + 0.0001)
            });
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
                    let p = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                    if region.is_some_and(|bb| {
                        !overlaps((p[0].min(p[1]).min(p[2]), p[0].max(p[1]).max(p[2])), bb)
                    }) {
                        continue;
                    }
                    triangles.push(Triangle {
                        p: p.map(|p| tf.transform_point(p)),
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
        support: Option<RestingContact>,
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
                    || support.is_some_and(|contact| {
                        // Slats have exposed edges under an object. A horizontal
                        // slide can touch their bevels without crossing them.
                        // Tray lips, posts and the tabletop still obstruct it.
                        t.owner == Some(contact.support)
                            && t.p.iter().all(|p| {
                                (p.y - contact.height).abs() < 0.0002
                                    || (contact.slides_on_shelf && p.y <= contact.height)
                            })
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
                if sweep.intersects(p) {
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

/// Fabric points move monotonically on vertical lines. The stock, weighted bar
/// and pull each get their own swept box so the narrow loop does not reserve
/// empty space under the whole rail.
fn screen_sweeps(
    a: &Assembly,
    joint: &Joint,
    tf: Transform,
    feed: RollerFeed,
    fraction: f32,
) -> Vec<Sweep> {
    joint
        .parts
        .iter()
        .map(|key| {
            let extent = |g: &Geometry| bounds(g.positions.iter().map(|p| Vec3::from_array(*p)));
            let (lo, hi) = extent(&a.parts[key]);
            let (end_lo, end_hi) = if let Some(g) = joint.morph.get(key) {
                let (a, b) = extent(g);
                (
                    feed.interpolate(lo, a, fraction),
                    feed.interpolate(hi, b, fraction),
                )
            } else {
                (
                    lo + joint.translation * fraction,
                    hi + joint.translation * fraction,
                )
            };
            Sweep {
                local: (lo.min(end_lo) - 0.0001, hi.max(end_hi) + 0.0001),
                frame: tf,
                shape: SweepShape::Box,
            }
        })
        .collect()
}

fn sweeps(a: &Assembly, joint: &Joint, tf: Transform) -> Vec<Sweep> {
    if let Some(ClothMotion::ProjectionScreen { feed, .. }) = joint.cloth_motion {
        return screen_sweeps(a, joint, tf, feed, joint.adjustment.morph);
    }
    let mut result = vec![];
    if joint.kind == MotionKind::Fan {
        if joint.axis != Vec3::Y {
            return rotor::sweep(a, joint, tf).into_iter().collect();
        }
        // Bound the complete rotor at any speed, including many revolutions.
        // AABB vertices at finitely sampled angles cannot certify a fast rotor.
        let mut radius_squared = 0.0_f32;
        let (mut bottom, mut top) = (f32::INFINITY, f32::NEG_INFINITY);
        for vertex in joint.parts.iter().flat_map(|key| &a.parts[key].positions) {
            let point = Vec3::from_array(*vertex);
            radius_squared = radius_squared.max((point - joint.pivot).xz().length_squared());
            bottom = bottom.min(point.y);
            top = top.max(point.y);
        }
        if bottom > top {
            return result;
        }
        let radius = radius_squared.sqrt() + 0.0001;
        return vec![Sweep {
            local: (
                Vec3::new(
                    joint.pivot.x - radius,
                    bottom - 0.0001,
                    joint.pivot.z - radius,
                ),
                Vec3::new(joint.pivot.x + radius, top + 0.0001, joint.pivot.z + radius),
            ),
            frame: tf,
            shape: SweepShape::Box,
        }];
    }
    let local_from_assembly = joint.frame.compute_affine().inverse();
    for key in &joint.parts {
        // Both projections of a pleat facet are monotone in the opening's
        // frame. Its endpoint bounds contain even a partially admitted draw.
        let cloth = joint
            .cloth_motion
            .filter(|law| law.applies(&key.1))
            .map(|law| law.at(joint.adjustment.morph));
        let g = &a.parts[key];
        // Local triangle blocks keep handles, hinge barrels and separate slats
        // from turning into a fictitious solid slab around the whole mechanism.
        // Thin articulated frames have nearby stationary carcasses. Large
        // triangle batches can enclose empty space and falsely reject every
        // possible angle. Bound each face for these compact mechanisms.
        let block = if matches!(
            joint.kind,
            MotionKind::Cabinet | MotionKind::Appliance | MotionKind::TeachingBoard
        ) || (joint.kind == MotionKind::Prop && joint.contact == Contact::Free)
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
                                if let Some(sample) = cloth {
                                    sample.interpolate(
                                        start,
                                        Vec3::from_array(end.positions[i as usize]),
                                        joint.adjustment.morph,
                                    )
                                } else {
                                    start.lerp(
                                        Vec3::from_array(end.positions[i as usize]),
                                        joint.adjustment.morph,
                                    )
                                },
                            ]
                        })
                        .map(|p| local_from_assembly.transform_point3(p)),
                );
                result.push(Sweep {
                    local: b,
                    frame: tf.mul_transform(joint.frame),
                    shape: SweepShape::Box,
                });
            } else {
                let lift = joint.effective_lift();
                if joint.effective_translation() == Vec3::ZERO
                    && joint.effective_angle() == 0.
                    && lift == 0.
                {
                    continue;
                }
                let b = bounds(indices.iter().map(|&i| {
                    local_from_assembly.transform_point3(Vec3::from_array(g.positions[i as usize]))
                }));
                if lift > 0.
                    && joint.effective_angle() == 0.
                    && joint.effective_translation() == Vec3::ZERO
                {
                    // A vertical pickup retraces the same segment. Its entire swept box
                    // is the rest box extruded to the apex, independent of easing.
                    let delta = local_from_assembly.transform_vector3(Vec3::Y * lift);
                    result.push(Sweep {
                        local: (
                            b.0 + delta.min(Vec3::ZERO) - 0.0001,
                            b.1 + delta.max(Vec3::ZERO) + 0.0001,
                        ),
                        frame: tf.mul_transform(joint.frame),
                        shape: SweepShape::Box,
                    });
                    continue;
                }
                let radius = corners(b)
                    .iter()
                    .map(|&p| joint.frame.transform_point(p).distance(joint.pivot))
                    .fold(0., f32::max);
                let steps = (joint.effective_angle().abs() / ROTATION_STEP_RADIANS)
                    .ceil()
                    .clamp(if lift > 0. { 6. } else { 1. }, 96.)
                    as usize;
                let step = 1. / steps as f32;
                let pad = Vec3::splat(
                    radius * (1. - (joint.effective_angle().abs() * step * 0.5).cos()) + 0.0001,
                );
                for i in 0..steps {
                    let frame = tf
                        .mul_transform(joint.transform((i as f32 + 0.5) * step))
                        .mul_transform(joint.frame);
                    let inverse = frame.compute_affine().inverse();
                    // Maximum rise of 4 h t(1-t) above its endpoint chord on
                    // this subinterval, expressed in the local sweep frame.
                    let rise = if lift > 0. {
                        inverse.transform_vector3(tf.rotation * Vec3::Y * (lift * step * step))
                    } else {
                        Vec3::ZERO
                    };
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
                        local: (
                            local.0 - pad + rise.min(Vec3::ZERO),
                            local.1 + pad + rise.max(Vec3::ZERO),
                        ),
                        frame,
                        shape: if block == 3 {
                            SweepShape::Hull(TriangleHull {
                                points: std::array::from_fn(|k| {
                                    let amount = (i + k / 3) as f32 * step;
                                    let pose = tf.mul_transform(joint.transform(amount));
                                    inverse.transform_point3(pose.transform_point(
                                        Vec3::from_array(g.positions[indices[k % 3] as usize]),
                                    ))
                                }),
                                error: pad.x + lift * step * step,
                            })
                        } else {
                            SweepShape::Box
                        },
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
        if let Some(bounds) = c.ceiling_sweep_bounds() {
            result.push(bounds);
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
    let limit = report.policy.max_mechanisms;
    let return_fraction = report.policy.operation_return_fraction;
    let independent_cargo = scene.generator_version >= ROTOR_CARGO_VERSION;
    let appearance = AppearanceTimeline::sample(scene, geometry, &report.policy);
    scene.indoor_motion.as_mut().unwrap().appearance = appearance;
    if geometry.architecture.joints.is_empty()
        && geometry.objects.iter().all(|a| a.joints.is_empty())
    {
        scene.indoor_motion.as_mut().unwrap().mechanisms.clear();
        return;
    }
    let cargo = mobility::cargo(scene, geometry);
    // Cargo proxies are constructed once, not once per fallback or mesh part.
    let cargo_geometry: Vec<_> = cargo
        .children
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
    let tree = Tree::for_admission(scene, geometry);
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
        if !independent_cargo && owner.and_then(|id| cargo.cart_for[id]).is_some() {
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
    // Resolve cart travel with all ordinary candidates first, preserving their
    // seeded relative priority. A parked cart then behaves like any other shelf.
    order.sort_by_key(|&(priority, owner, index, _)| {
        (
            owner.and_then(|id| cargo.cart_for[id]).is_some(),
            priority,
            owner,
            index,
        )
    });
    for (_, owner, index, count) in order {
        if owner.and_then(|id| cargo.cart_for[id]).is_some_and(|cart| {
            geometry.objects[cart]
                .joints
                .iter()
                .any(|j| j.kind == MotionKind::Cart && j.accepted)
        }) {
            continue;
        }
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
        let children = owner.map_or(&[][..], |id| cargo.children[id].as_slice());
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
            let resting = owner
                .filter(|_| j.contact == Contact::Surface)
                .and_then(|id| {
                    let o = &scene.objects[id];
                    Some(RestingContact {
                        support: o.support?,
                        height: tf.translation.y,
                        slides_on_shelf: j.translation.y == 0.
                            && j.axis.xz() == Vec2::ZERO
                            && scene.support_plane(o)
                                == Some(super::super::layout::SupportPlane::LowerShelf),
                    })
                });
            let choices =
                proposals::candidates(&a.joints[index..index + count], scene.generator_version);
            for (adjustment, name) in choices {
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
                        resting,
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
        let cycle = if reason.is_none()
            && scene.generator_version >= OPERATION_CYCLE_VERSION
            && return_fraction > 0.
        {
            OperationCycle::sample(&a.joints[index], return_fraction)
        } else {
            None
        };
        for &id in &ids {
            a.joints[id].accepted = reason.is_none();
            a.joints[id].operation_cycle = cycle;
        }
        let j = &a.joints[index];
        reports.push(MechanismReport {
            operation_cycle: cycle,
            object_id: owner,
            carried_object_ids: if reason.is_none() || !independent_cargo {
                children.to_vec()
            } else {
                vec![]
            },
            semantic_labels: ids
                .iter()
                .flat_map(|&i| a.joints[i].parts.iter())
                .map(|(_, label)| super::super::objects::part_label(label).to_owned())
                .collect::<std::collections::BTreeSet<_>>()
                .into_iter()
                .collect(),
            joint: index,
            linked_slides: if j.coupled
                && j.kind == MotionKind::Door
                && j.angle.abs() <= 1e-6
                && j.translation.length_squared() > 1e-8
            {
                ids[1..]
                    .iter()
                    .map(|&id| LinkedSlide {
                        joint: id,
                        translation_m: tf.rotation * a.joints[id].effective_translation(),
                        requested_translation_m: tf.rotation * a.joints[id].translation,
                    })
                    .collect()
            } else {
                Vec::new()
            },
            kind: j.kind,
            interval: if let Some(cycle) = cycle {
                [cycle.starts_at, cycle.returns_at]
            } else if j.kind == MotionKind::Fan {
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
            lift_m: if j.accepted { j.effective_lift() } else { 0. },
            requested_lift_m: j.lift_m,
            projection_screen_travel: match j.cloth_motion {
                Some(ClothMotion::ProjectionScreen { start_drop_m, .. }) => {
                    Some(ProjectionScreenTravel {
                        start_drop_m,
                        requested_end_drop_m: start_drop_m - j.translation.y,
                        end_drop_m: if j.accepted {
                            start_drop_m - j.effective_translation().y
                        } else {
                            start_drop_m
                        },
                    })
                }
                _ => None,
            },
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
    mobility::carry(scene, geometry, &cargo.children);
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
