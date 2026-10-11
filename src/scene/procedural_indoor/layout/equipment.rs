//! Equipment is fitted to existing rooms and work surfaces, then wires are routed
//! between actual ports after all supported objects have been accepted.
use super::super::architecture::facade::FacadeSide;
use super::super::objects::equipment::{self, wires};
use super::*;
use rand::seq::SliceRandom;
use std::f32::consts::PI;

impl IndoorManifest {
    /// Call before ordinary supported clutter, reserving useful presentation and
    /// electrical equipment rather than hoping they fit as late random props.
    pub(super) fn furnish_equipment(&mut self) {
        let mut rng = stream(self.seed, 5780);
        let count = (self.density * rng.random_range(2.0..6.0)).round() as usize;
        for _ in 0..count {
            let kind = [
                ObjectKind::WallShelf,
                ObjectKind::NoticeBoard,
                ObjectKind::ProjectionScreen,
                ObjectKind::WallSconce,
            ][rng.random_range(0..4)];
            let size = match kind {
                ObjectKind::WallShelf => Vec3::new(
                    rng.random_range(0.60..1.40),
                    rng.random_range(0.17..0.29),
                    rng.random_range(0.20..0.32),
                ),
                ObjectKind::NoticeBoard => Vec3::new(
                    rng.random_range(0.65..1.55),
                    rng.random_range(0.55..1.10),
                    rng.random_range(0.020..0.034),
                ),
                ObjectKind::ProjectionScreen => Vec3::new(
                    rng.random_range(1.20..2.10),
                    rng.random_range(0.80..1.25),
                    rng.random_range(0.06..0.09),
                ),
                _ => Vec3::new(
                    rng.random_range(0.12..0.27),
                    rng.random_range(0.19..0.35),
                    rng.random_range(0.08..0.15),
                ),
            };
            for _ in 0..8 {
                let (tf, span) = if let Some(e) = &self.envelope {
                    let wall = &e.walls[rng.random_range(0..e.walls.len())];
                    let a = e.footprint[wall.edge];
                    let b = e.footprint[(wall.edge + 1) % e.footprint.len()];
                    (e.wall_transform(wall.edge), a.distance(b))
                } else {
                    let side = FacadeSide::ALL[rng.random_range(0..FacadeSide::ALL.len())];
                    (side.transform(self.room_size), side.span(self.room_size))
                };
                if span < size.x + 0.25 {
                    continue;
                }
                let x =
                    rng.random_range((-span + size.x) * 0.5 + 0.10..(span - size.x) * 0.5 - 0.10);
                let y = match kind {
                    ObjectKind::WallShelf => rng.random_range(1.12..1.65),
                    ObjectKind::WallSconce => rng.random_range(1.65..2.15),
                    _ => rng.random_range(0.90..1.35),
                };
                let position = tf.transform_point(Vec3::new(x, y, size.z * 0.5 + 0.001));
                let yaw = tf.rotation.to_euler(EulerRot::YXZ).0;
                let mut object = self.candidate(kind, position, size, yaw, &mut rng);
                object.solid = false;
                let (lo, hi) = object.bounds();
                if !self.wall_attachment_clear(&object)
                    || self
                        .objects
                        .iter()
                        .map(IndoorObject::bounds)
                        .chain(self.columns())
                        .any(|(a, b)| {
                            lo.cmplt(b + Vec3::splat(0.025)).all()
                                && hi.cmpgt(a - Vec3::splat(0.025)).all()
                        })
                {
                    continue;
                }
                self.objects.push(object);
                break;
            }
        }
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && matches!(
                        o.kind,
                        ObjectKind::Desk
                            | ObjectKind::Table
                            | ObjectKind::MobileCart
                            | ObjectKind::Cabinet
                    )
            })
            .cloned()
            .collect();
        // Presentation hardware is a paired installation, not an unrelated
        // late clutter lottery. Reserve its real footprint before power strips.
        self.projector_equipment(&surfaces);
        for surface in &surfaces {
            let mut station = stream(surface.seed, 5790);
            if station.random_bool(0.35 + self.density as f64 * 0.55) {
                let x = (surface.size.x * 0.31).min(surface.size.x * 0.5 - 0.18);
                self.prop(
                    surface,
                    ObjectKind::PowerStrip,
                    Vec3::new(x, 0., -surface.size.z * 0.31),
                    Vec3::new(
                        station.random_range(0.22..0.34),
                        0.030,
                        station.random_range(0.043..0.061),
                    ),
                    station.random_range(-0.07..0.07),
                    &mut station,
                );
                if let Some(strip) = self
                    .objects
                    .last_mut()
                    .filter(|o| o.kind == ObjectKind::PowerStrip && o.support == Some(surface.id))
                {
                    // Match wall-hardware electrical convention throughout the scene.
                    strip.variant = (self.seed % 4) as u32;
                }
            }
        }
        self.pendant_equipment(&mut rng);
    }

    fn projector_equipment(&mut self, surfaces: &[IndoorObject]) {
        let screens: Vec<_> = self
            .objects
            .iter()
            .filter(|o| !o.neighbor && o.kind == ObjectKind::ProjectionScreen)
            .cloned()
            .collect();
        for screen in screens {
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
            supports.shuffle(&mut rng);
            let mut installed = false;
            for support in supports.into_iter().take(12) {
                for proposal in 0..8 {
                    let offset = if proposal == 0 {
                        Vec3::ZERO
                    } else {
                        Vec3::new(
                            rng.random_range(-0.28..0.28) * support.size.x,
                            0.,
                            rng.random_range(-0.26..0.26) * support.size.z,
                        )
                    };
                    let position = support
                        .transform()
                        .transform_point(offset + Vec3::Y * support.size.y);
                    let direction = (screen.position - position).with_y(0.);
                    let yaw = direction.x.atan2(direction.z);
                    let mut projector = self.candidate(
                        ObjectKind::Projector,
                        position,
                        Vec3::new(
                            rng.random_range(0.23..0.36),
                            rng.random_range(0.09..0.15),
                            rng.random_range(0.21..0.29),
                        ),
                        yaw,
                        &mut rng,
                    );
                    projector.solid = false;
                    projector.support = Some(support.id);
                    if !self.prop_clear(&projector, support, 0.012)
                        || !projector_axis_clear(self, &projector, &screen)
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

    fn pendant_equipment(&mut self, rng: &mut ChaCha8Rng) {
        let tables: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && matches!(
                        o.kind,
                        ObjectKind::Table | ObjectKind::CoffeeTable | ObjectKind::SideTable
                    )
            })
            .cloned()
            .collect();
        for table in tables {
            if !rng.random_bool(0.23 + self.density as f64 * 0.25) {
                continue;
            }
            let size = Vec3::new(
                rng.random_range(0.22..0.43),
                rng.random_range(0.30..0.61),
                rng.random_range(0.22..0.43),
            );
            let ceiling = self.ceiling_height(table.position.xz());
            let position = Vec3::new(table.position.x, ceiling - size.y - 0.004, table.position.z);
            if position.y < (table.position.y + table.size.y + 0.95).max(2.05) {
                continue;
            }
            let mut o = self.candidate(ObjectKind::CeilingPendant, position, size, 0., rng);
            o.solid = false;
            if !ceiling_fixture_clear(self, &o) {
                continue;
            }
            self.objects.push(o);
        }
    }

    /// Call after all supported decoration and occupied-seat settlement, before
    /// camera planning. Screen rotations have then stopped changing port frames.
    /// Routes avoid existing supported clutter and people, and stay on the top.
    pub(super) fn connect_equipment(&mut self) {
        let sources: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && o.support.is_some()
                    && matches!(
                        o.kind,
                        ObjectKind::Laptop | ObjectKind::Monitor | ObjectKind::Projector
                    )
            })
            .cloned()
            .collect();
        for source in sources {
            let Some(parent) = self.objects.get(source.support.unwrap()).cloned() else {
                continue;
            };
            let targets: Vec<_> = self
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::PowerStrip && o.support == source.support)
                .cloned()
                .collect();
            for target in targets {
                let mut rng = stream(source.seed ^ target.seed, 5800);
                if !rng.random_bool(0.55 + self.density as f64 * 0.35) {
                    continue;
                }
                for _ in 0..10 {
                    let mut o = self.candidate(
                        ObjectKind::Cable,
                        Vec3::ZERO,
                        Vec3::ONE,
                        parent.yaw,
                        &mut rng,
                    );
                    o.solid = false;
                    o.support = Some(parent.id);
                    o.interaction_target = Some(source.id);
                    o.cable_target = Some(target.id);
                    let radius = wires::bound_radius(&o);
                    let inv = parent.transform().compute_affine().inverse();
                    let low = source.kind == ObjectKind::Laptop;
                    let start = inv.transform_point3(equipment::power_port(&source, low).unwrap());
                    let end = inv.transform_point3(equipment::power_port(&target, low).unwrap());
                    let source_dir = inv
                        .transform_vector3(equipment::power_port_normal(&source, low).unwrap())
                        .normalize();
                    let target_dir = inv
                        .transform_vector3(equipment::power_port_normal(&target, low).unwrap())
                        .normalize();
                    let mut points = vec![start, start + source_dir * 0.039];
                    let b = end + target_dir * 0.039;
                    if source.kind == ObjectKind::Monitor {
                        // Leave the rear housing, then traverse behind the
                        // complete display before turning around its side. The
                        // checker still rejects every source crossing after the
                        // initial short connector segment.
                        let target_local = source
                            .transform()
                            .compute_affine()
                            .inverse()
                            .transform_point3(equipment::power_port(&target, low).unwrap());
                        let side = if rng.random_bool(0.80) {
                            if target_local.x >= 0. {
                                1.
                            } else {
                                -1.
                            }
                        } else if rng.random_bool(0.5) {
                            1.
                        } else {
                            -1.
                        };
                        let port_local =
                            super::super::objects::computers::monitor_power_port_local(&source).0;
                        let rail = Vec3::new(
                            side * (source.size.x * 0.5 + radius + 0.022),
                            port_local.y,
                            port_local.z - 0.039,
                        );
                        points.push(inv.transform_point3(source.transform().transform_point(rail)));
                    }
                    let a = *points.last().unwrap();
                    let bend = rng.random_range(-0.055..0.055);
                    let sag = rng.random_range(0.004..0.013);
                    let perpendicular = Vec3::new(-(b.z - a.z), 0., b.x - a.x).normalize_or_zero();
                    for i in 1..=9 {
                        let t = i as f32 / 10.;
                        let mut p = a.lerp(b, t) + perpendicular * (PI * t).sin() * bend;
                        p.y = (p.y - (PI * t).sin() * sag).max(parent.size.y + radius + 0.001);
                        points.push(p);
                    }
                    points.extend([b, end]);
                    let Some(points) = wires::fillet_path(&points, wires::radius(&o)) else {
                        continue;
                    };
                    let lo = points
                        .iter()
                        .fold(Vec3::splat(f32::INFINITY), |a, &b| a.min(b))
                        - Vec3::splat(radius);
                    let hi = points
                        .iter()
                        .fold(Vec3::splat(f32::NEG_INFINITY), |a, &b| a.max(b))
                        + Vec3::splat(radius);
                    let origin = Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
                    o.position = parent.transform().transform_point(origin);
                    o.size = hi - lo;
                    o.cable_path = Some(points.into_iter().map(|p| p - origin).collect());
                    if cable_route_clear(self, &o) {
                        self.objects.push(o);
                        break;
                    }
                }
                // A source has one power lead, regardless of nearby strip count.
                if self
                    .objects
                    .iter()
                    .any(|o| o.kind == ObjectKind::Cable && o.interaction_target == Some(source.id))
                {
                    break;
                }
            }
        }
    }
}

/// Plausible tabletop projection axis at installation time. The equipment is
/// powered off: later occupants may cross this axis without an imaginary beam
/// or projected image being fabricated. Architecture is never traversed.
fn projector_axis_clear(
    scene: &IndoorManifest,
    projector: &IndoorObject,
    screen: &IndoorObject,
) -> bool {
    let Some((origin, target)) = projector_axis(scene, projector, screen) else {
        return false;
    };
    !scene
        .objects
        .iter()
        .filter(|o| {
            o.solid && o.id != screen.id && o.id != projector.id && Some(o.id) != projector.support
        })
        .any(|o| {
            let (a, b) = o.bounds();
            segment_hits_box(origin, target, a, b)
        })
}

fn projector_axis(
    scene: &IndoorManifest,
    projector: &IndoorObject,
    screen: &IndoorObject,
) -> Option<(Vec3, Vec3)> {
    let p = equipment::parameters(projector);
    let foot = (projector.size.y * 0.09).min(0.013);
    let origin = projector.transform().transform_point(Vec3::new(
        projector.size.x * p.lens_offset,
        foot + (projector.size.y - foot) * 0.49,
        projector.size.z * 0.5 + 0.015,
    ));
    let target = screen.transform().transform_point(Vec3::new(
        0.,
        screen.size.y * 0.5,
        screen.size.z * 0.2 + 0.0032,
    ));
    let planar = (target - origin).with_y(0.);
    let throw = planar.length();
    if !(0.9..=9.0).contains(&throw) || !(0.65..=3.5).contains(&(throw / screen.size.x)) {
        return None;
    }
    let normal = screen.transform().rotation * Vec3::Z;
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
                !super::super::envelope::polygon::contains(&e.footprint, point.xz(), 0.)
                    || point.y <= e.floor_height(point.xz())
                    || point.y >= e.ceiling_height(scene.room_size, point.xz())
            }) || e
                .ray_hit(scene.room_size, scene.door_x, origin, delta.normalize())
                .is_some_and(|(distance, _)| distance < delta.length() - 0.001)
        })
        || scene
            .camera_obstacles()
            .into_iter()
            .any(|(a, b)| segment_hits_box(origin, target, a, b))
    {
        return None;
    }
    Some((origin, target))
}
/// Real mounted volume below the locally sampled ceiling, with conservative
/// existing geometry clearance. A sloping ceiling cannot intersect the canopy.
pub(crate) fn ceiling_fixture_clear(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    if o.kind != ObjectKind::CeilingPendant || o.solid || o.support.is_some() || o.neighbor {
        return false;
    }
    let (lo, hi) = o.bounds();
    let corners = [
        lo.xz(),
        hi.xz(),
        Vec2::new(lo.x, hi.z),
        Vec2::new(hi.x, lo.z),
    ];
    let center_height = scene.ceiling_height(o.position.xz());
    if (hi.y + 0.004 - center_height).abs() > 0.001
        || lo.y < 2.05
        || corners
            .into_iter()
            .any(|p| hi.y > scene.ceiling_height(p) - 0.001)
        || scene
            .columns()
            .iter()
            .any(|&(a, b)| lo.cmplt(b).all() && hi.cmpgt(a).all())
    {
        return false;
    }
    let fixture_half = super::super::architecture::fixture_size(scene) * 0.5;
    if super::super::architecture::fixture_positions(scene)
        .into_iter()
        .any(|p| {
            // Reserve the native fixture's body plus its suspension anchors, not
            // merely the glowing panel. New canopies cannot occupy those mounts.
            let a = p - fixture_half - Vec3::Y * 0.008;
            let b = Vec3::new(
                p.x + fixture_half.x,
                scene.ceiling_height(p.xz()),
                p.z + fixture_half.z,
            );
            lo.cmplt(b + Vec3::splat(0.015)).all() && hi.cmpgt(a - Vec3::splat(0.015)).all()
        })
    {
        return false;
    }
    !scene
        .objects
        .iter()
        .filter(|other| other.id != o.id)
        .any(|other| {
            let (a, b) = other.bounds();
            lo.cmplt(b + Vec3::splat(0.025)).all() && hi.cmpgt(a - Vec3::splat(0.025)).all()
        })
}

/// Test a proposed prop against existing wire segments, including edits to an
/// already wired endpoint. A cable's enclosing box is not occupied tabletop
/// space; only its jacket and the validated port contact reserve space.
pub(crate) fn cables_clear_proposed_prop(scene: &IndoorManifest, prop: &IndoorObject) -> bool {
    for cable in scene.objects.iter().filter(|o| o.kind == ObjectKind::Cable) {
        let Some(points) = &cable.cable_path else {
            return false;
        };
        if !(2..=wires::MAX_PATH_POINTS).contains(&points.len())
            || points.iter().any(|p| !p.is_finite())
        {
            return false;
        }
        let tf = cable.transform();
        let is_source = cable.interaction_target == Some(prop.id);
        let is_target = cable.cable_target == Some(prop.id);
        if is_source || is_target {
            let Some(source) = cable
                .interaction_target
                .and_then(|id| scene.objects.get(id))
            else {
                return false;
            };
            if prop.support != cable.support
                || prop.neighbor
                || if is_source {
                    prop.kind != source.kind
                        || !matches!(
                            prop.kind,
                            ObjectKind::Laptop | ObjectKind::Monitor | ObjectKind::Projector
                        )
                } else {
                    prop.kind != ObjectKind::PowerStrip
                }
            {
                return false;
            }
            let low_voltage = source.kind == ObjectKind::Laptop;
            let Some(port) = equipment::power_port(prop, low_voltage) else {
                return false;
            };
            let (end, next) = if is_source {
                (points[0], points[1])
            } else {
                (points[points.len() - 1], points[points.len() - 2])
            };
            let Some(normal) = equipment::power_port_normal(prop, low_voltage) else {
                return false;
            };
            let end = tf.transform_point(end);
            let next = tf.transform_point(next);
            if end.distance(port) > 0.001
                || next.distance(end) > 0.04
                || (next - end).normalize_or_zero().dot(normal) < 0.98
            {
                return false;
            }
        }
        let hits_prop = cable_prop_intersector(prop, wires::bound_radius(cable));
        for (i, pair) in points.windows(2).enumerate() {
            if (is_source && i == 0) || (is_target && i == points.len() - 2) {
                continue;
            }
            if hits_prop(tf.transform_point(pair[0]), tf.transform_point(pair[1])) {
                return false;
            }
        }
    }
    true
}

/// Generation and later prop edits share the same expanded, rotated box. Its
/// broad phase encloses the transformed local padding as well as the housing;
/// expanding an unpadded world AABB by just `radius` underbounds rotated corners.
fn cable_prop_intersector(prop: &IndoorObject, radius: f32) -> impl Fn(Vec3, Vec3) -> bool {
    let transform = prop.transform().compute_affine();
    let inverse = transform.inverse();
    let padding = Vec3::splat(radius);
    let lo = Vec3::new(-prop.size.x * 0.5, 0., -prop.size.z * 0.5) - padding;
    let hi = Vec3::new(prop.size.x * 0.5, prop.size.y, prop.size.z * 0.5) + padding;
    let mut world_lo = Vec3::splat(f32::INFINITY);
    let mut world_hi = Vec3::splat(f32::NEG_INFINITY);
    for x in [lo.x, hi.x] {
        for y in [lo.y, hi.y] {
            for z in [lo.z, hi.z] {
                let p = transform.transform_point3(Vec3::new(x, y, z));
                world_lo = world_lo.min(p);
                world_hi = world_hi.max(p);
            }
        }
    }
    move |start, end| {
        if start.min(end).cmpgt(world_hi).any() || start.max(end).cmplt(world_lo).any() {
            return false;
        }
        segment_hits_box(
            inverse.transform_point3(start),
            inverse.transform_point3(end),
            lo,
            hi,
        )
    }
}

/// Validate exact archived routes and their electrical connection graph. There
/// is no blanket overlap exception: only the first/last short segment may touch
/// its respective plug's housing, and every other peer is checked by segment.
pub(crate) fn cable_route_clear(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    if o.kind != ObjectKind::Cable || o.solid || o.neighbor {
        return false;
    }
    let Some(points) = &o.cable_path else {
        return false;
    };
    if !(2..=wires::MAX_PATH_POINTS).contains(&points.len())
        || points.iter().any(|p| !p.is_finite())
    {
        return false;
    }
    if !wires::path_qualified(o) {
        return false;
    }
    // The circular sweep uses each interior point's centered tangent. Distinct
    // adjacent points alone do not exclude a P,Q,P reversal with a zero tangent.
    if points
        .windows(3)
        .any(|triple| triple[0].distance_squared(triple[2]) < 1e-10)
    {
        return false;
    }
    let (Some(parent), Some(source), Some(target)) = (
        o.support.and_then(|id| scene.objects.get(id)),
        o.interaction_target.and_then(|id| scene.objects.get(id)),
        o.cable_target.and_then(|id| scene.objects.get(id)),
    ) else {
        return false;
    };
    if !parent.kind.is_surface()
        || source.id >= o.id
        || target.id >= o.id
        || source.support != Some(parent.id)
        || target.support != Some(parent.id)
        || source.neighbor
        || target.neighbor
        || parent.neighbor
        || target.kind != ObjectKind::PowerStrip
        || !matches!(
            source.kind,
            ObjectKind::Laptop | ObjectKind::Monitor | ObjectKind::Projector
        )
    {
        return false;
    }
    let world: Vec<_> = points
        .iter()
        .map(|&p| o.transform().transform_point(p))
        .collect();
    let low = source.kind == ObjectKind::Laptop;
    if world[0].distance(equipment::power_port(source, low).unwrap()) > 0.001
        || world
            .last()
            .unwrap()
            .distance(equipment::power_port(target, low).unwrap())
            > 0.001
    {
        return false;
    }
    let source_normal = equipment::power_port_normal(source, low).unwrap();
    let target_normal = equipment::power_port_normal(target, low).unwrap();
    if (world[1] - world[0]).normalize_or_zero().dot(source_normal) < 0.98
        || (world[world.len() - 2] - world[world.len() - 1])
            .normalize_or_zero()
            .dot(target_normal)
            < 0.98
    {
        return false;
    }
    let radius = wires::bound_radius(o);
    let lo = points
        .iter()
        .fold(Vec3::splat(f32::INFINITY), |a, &b| a.min(b))
        - Vec3::splat(radius);
    let hi = points
        .iter()
        .fold(Vec3::splat(f32::NEG_INFINITY), |a, &b| a.max(b))
        + Vec3::splat(radius);
    if lo.distance(Vec3::new(-o.size.x * 0.5, 0., -o.size.z * 0.5)) > 0.0001
        || hi.distance(Vec3::new(o.size.x * 0.5, o.size.y, o.size.z * 0.5)) > 0.0001
    {
        return false;
    }
    let inv = parent.transform().compute_affine().inverse();
    // Prepare each rigid peer once for the complete sampled route, keeping the
    // exact rotated broad phase inexpensive for long curved wires.
    let peers: Vec<_> = scene
        .objects
        .iter()
        .filter(|other| other.id != parent.id && other.id != o.id)
        .map(|other| {
            (
                other,
                (other.kind != ObjectKind::Cable).then(|| cable_prop_intersector(other, radius)),
            )
        })
        .collect();
    let mut length = 0.;
    for (i, pair) in world.windows(2).enumerate() {
        let distance = pair[0].distance(pair[1]);
        if distance < 0.0001 || distance > 0.5 {
            return false;
        }
        length += distance;
        if (i == 0 || i == world.len() - 2) && distance > 0.04 {
            return false;
        }
        for step in 0..=4 {
            let q = inv.transform_point3(pair[0].lerp(pair[1], step as f32 / 4.));
            let half = parent.size.xz() * 0.5;
            if q.y < parent.size.y + wires::radius(o) - 0.0003
                || q.xz().abs().cmpgt(half - Vec2::splat(radius + 0.002)).any()
                || parent.kind.has_table_outline()
                    && !super::super::objects::tables::supports(
                        parent,
                        q.xz() - Vec2::splat(radius),
                        q.xz() + Vec2::splat(radius),
                        0.001,
                    )
            {
                return false;
            }
        }
        if peers.iter().any(|(other, hits_prop)| {
            if (other.id == source.id && i == 0) || (other.id == target.id && i == world.len() - 2)
            {
                return false;
            }
            if other.kind == ObjectKind::Cable {
                let (a, b) = other.bounds();
                let padding = Vec3::splat(radius);
                if pair[0].min(pair[1]).cmpgt(b + padding).any()
                    || pair[0].max(pair[1]).cmplt(a - padding).any()
                {
                    return false;
                }
                // Symmetric capsule clearance prevents accepting a new
                // lead whose large enclosing rectangle would invalidate a
                // previously accepted, physically separate lead on replay.
                let Some(other_points) = &other.cable_path else {
                    return true;
                };
                let tf = other.transform();
                let clearance = (radius + wires::bound_radius(other)) as f64;
                return other_points.windows(2).any(|segment| {
                    wires::segment_distance_squared(
                        pair[0],
                        pair[1],
                        tf.transform_point(segment[0]),
                        tf.transform_point(segment[1]),
                    ) <= clearance * clearance
                });
            }
            hits_prop.as_ref().unwrap()(pair[0], pair[1])
        }) {
            return false;
        }
        if scene.columns().iter().any(|&(a, b)| {
            segment_hits_box(
                pair[0],
                pair[1],
                a - Vec3::splat(radius),
                b + Vec3::splat(radius),
            )
        }) {
            return false;
        }
        if scene.humans.iter().any(|h| {
            let (a, b) = h.bounds();
            segment_hits_box(
                pair[0],
                pair[1],
                a - Vec3::splat(radius),
                b + Vec3::splat(radius),
            )
        }) {
            return false;
        }
    }
    (0.05..=2.0).contains(&length)
}

#[cfg(test)]
mod tests;
