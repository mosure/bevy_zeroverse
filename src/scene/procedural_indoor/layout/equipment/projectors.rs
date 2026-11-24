//! Fit standby projectors to a physical, fully exposed screen. The optical
//! axis is checked against the room and existing furnishings before placement.
use super::*;
use crate::scene::procedural_indoor::objects::presentation::projection;

struct Target {
    id: usize,
    center: Vec3,
    normal: Vec3,
    width: f32,
}
struct Obstacles {
    structure: Vec<(Vec3, Vec3)>,
    solid: Vec<(usize, (Vec3, Vec3))>,
}
impl Obstacles {
    fn sample(scene: &IndoorManifest) -> Self {
        Self {
            structure: scene.camera_obstacles(),
            solid: scene
                .objects
                .iter()
                .filter(|o| o.solid)
                .map(|o| (o.id, o.bounds()))
                .collect(),
        }
    }
    fn clear(&self, scene: &IndoorManifest, o: &IndoorObject, target: &Target) -> bool {
        let Some((origin, end)) = target.axis(scene, o, &self.structure) else {
            return false;
        };
        !self.solid.iter().any(|&(id, (a, b))| {
            id != target.id
                && id != o.id
                && Some(id) != o.support
                && segment_hits_box(origin, end, a, b)
        })
    }
}
impl Target {
    fn from_screen(o: &IndoorObject) -> Option<Self> {
        let p = projection::parameters(o);
        let tf = o.transform();
        Some(Self {
            id: o.id,
            center: tf.transform_point(p.projection_target_local()?),
            normal: tf.rotation * Vec3::Z,
            width: p.image_width_m(),
        })
    }
    fn axis(
        &self,
        scene: &IndoorManifest,
        projector: &IndoorObject,
        structure: &[(Vec3, Vec3)],
    ) -> Option<(Vec3, Vec3)> {
        let optics = equipment::projector::parameters(projector);
        let origin = projector
            .transform()
            .transform_point(optics.optical_origin() + Vec3::Z * 0.015);
        let target = self.center;
        let planar = (target - origin).with_y(0.);
        let throw = planar.length();
        if !(0.9..=9.0).contains(&throw) || !(0.65..=3.5).contains(&(throw / self.width)) {
            return None;
        }
        let normal = self.normal;
        let forward = projector.transform().rotation * Vec3::Z;
        if normal.dot(-planar.normalize()) < 0.55
            || forward.dot(planar.normalize()) < 0.98
            || (target.y - origin.y).abs() > throw * 0.65
            || scene.envelope.as_ref().is_some_and(|e| {
                // Camera segment queries deliberately reserve a 20 cm wall
                // rail. An optical ray must be able to reach a wall-mounted
                // screen: test its physical first surface instead of applying
                // camera-body clearance to the near-wall endpoint.
                let delta = target - origin;
                [origin, target].into_iter().any(|point| {
                    !crate::scene::procedural_indoor::envelope::polygon::contains(
                        &e.footprint,
                        point.xz(),
                        0.,
                    ) || point.y <= e.floor_height(point.xz())
                        || point.y >= e.ceiling_height(scene.room_size, point.xz())
                }) || e
                    .ray_hit(
                        scene.room_size,
                        scene.door_x,
                        scene.entry_width(),
                        scene.entry_height(),
                        origin,
                        delta.normalize(),
                    )
                    .is_some_and(|(distance, _)| distance < delta.length() - 0.001)
            })
            || structure
                .iter()
                .any(|&(a, b)| segment_hits_box(origin, target, a, b))
        {
            return None;
        }
        Some((origin, target))
    }
}

impl IndoorManifest {
    pub(super) fn projector_equipment(&mut self, surfaces: &[IndoorObject]) {
        let screens: Vec<_> = self
            .objects
            .iter()
            .filter(|o| !o.neighbor && o.kind == ObjectKind::ProjectionScreen)
            .cloned()
            .collect();
        let mut obstacles = None;
        for screen in screens {
            let Some(target) = Target::from_screen(&screen) else {
                continue;
            };
            let mut rng = stream(screen.seed, 5785);
            if !rng.random_bool(0.60 + self.density as f64 * 0.35) {
                continue;
            }
            let mut supports: Vec<_> = surfaces
                .iter()
                .filter(|surface| {
                    matches!(
                        surface.kind,
                        ObjectKind::Desk
                            | ObjectKind::Table
                            | ObjectKind::MobileCart
                            | ObjectKind::Cabinet
                    ) && (0.60..=1.22).contains(&(surface.position.y + surface.size.y))
                })
                .collect();
            if supports.is_empty() {
                continue;
            }
            // Projectors added by this pass are non-solid; every trial can
            // share the same immutable architectural and furniture bounds.
            let obstacles = obstacles.get_or_insert_with(|| Obstacles::sample(self));
            supports.shuffle(&mut rng);
            let mut installed = false;
            for support in supports.into_iter().take(12) {
                // One physical product per support; retries change placement,
                // not its dimensions. A centered first proposal used to win
                // almost every time, suppressing the sampled offsets entirely.
                let size = Vec3::new(
                    rng.random_range(0.23..0.36),
                    rng.random_range(0.09..0.15),
                    rng.random_range(0.21..0.29),
                );
                for _ in 0..8 {
                    let offset = Vec3::new(
                        rng.random_range(-0.28..0.28) * support.size.x,
                        0.,
                        rng.random_range(-0.26..0.26) * support.size.z,
                    );
                    let position = support
                        .transform()
                        .transform_point(offset + Vec3::Y * support.size.y);
                    let direction = (target.center - position).with_y(0.);
                    let yaw = direction.x.atan2(direction.z);
                    let mut projector =
                        self.candidate(ObjectKind::Projector, position, size, yaw, &mut rng);
                    projector.solid = false;
                    projector.support = Some(support.id);
                    if !self.prop_clear(&projector, support, 0.012)
                        || !obstacles.clear(self, &projector, &target)
                    {
                        self.rejected_placements += 1;
                        continue;
                    }
                    self.objects.push(projector);
                    installed = true;
                    break;
                }
                if installed {
                    break;
                }
            }
        }
    }
}

#[cfg(test)]
pub(super) fn projector_axis(
    scene: &IndoorManifest,
    o: &IndoorObject,
    screen: &IndoorObject,
) -> Option<(Vec3, Vec3)> {
    Target::from_screen(screen)?.axis(scene, o, &scene.camera_obstacles())
}
#[cfg(test)]
pub(super) fn projector_axis_clear(
    scene: &IndoorManifest,
    o: &IndoorObject,
    screen: &IndoorObject,
) -> bool {
    Target::from_screen(screen).is_some_and(|t| Obstacles::sample(scene).clear(scene, o, &t))
}

/// Independently qualify the final room after chairs and other movable items
/// have settled. An optical ray can pass through empty space in a chair's AABB.
#[cfg(test)]
pub(crate) fn rendered_projector_axis_clear(
    scene: &IndoorManifest,
    o: &IndoorObject,
    screen: &IndoorObject,
) -> bool {
    let Some((start, end)) = projector_axis(scene, o, screen) else {
        return false;
    };
    !scene
        .objects
        .iter()
        .filter(|b| b.solid && b.id != screen.id && b.id != o.id && Some(b.id) != o.support)
        .any(|b| {
            let (lo, hi) = b.bounds();
            if !segment_hits_box(start, end, lo, hi) {
                return false;
            }
            let inv = b.transform().compute_affine().inverse();
            let origin = inv.transform_point3(start);
            // Unnormalised direction puts the screen at ray parameter one.
            let ray = inv.transform_vector3(end - start);
            crate::scene::procedural_indoor::objects::build_object_at_version(
                b,
                None,
                scene.generator_version,
            )
            .parts
            .values()
            .any(|g| {
                g.indices.as_chunks::<3>().0.iter().any(|t| {
                    crate::scene::procedural_indoor::envelope::ray_triangle(
                        origin,
                        ray,
                        t.map(|i| Vec3::from_array(g.positions[i as usize])),
                    )
                    .is_some_and(|f| f < 1.)
                })
            })
        })
}
