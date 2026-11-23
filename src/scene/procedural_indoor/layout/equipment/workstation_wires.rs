//! Guided USB and display leads, routed after chairs, people and sockets settle.
use super::*;
use crate::scene::procedural_indoor::objects::desktop::ports;

pub(super) fn host(scene: &IndoorManifest, o: &IndoorObject) -> Option<usize> {
    o.support.or_else(|| {
        if o.kind != ObjectKind::PcTower {
            return None;
        }
        let screen = scene.objects.get(o.interaction_target?)?;
        (screen.kind == ObjectKind::Monitor)
            .then_some(screen.support)
            .flatten()
    })
}
pub(super) fn leaves_top(source: &IndoorObject, target: &IndoorObject) -> bool {
    source.support.is_none() || target.support.is_none()
}
pub(super) fn hosts_match(
    scene: &IndoorManifest,
    parent: Option<usize>,
    source: &IndoorObject,
    target: &IndoorObject,
) -> bool {
    if parent.is_none() || source.neighbor || target.neighbor || host(scene, source) != parent {
        return false;
    }
    if matches!(
        target.kind,
        ObjectKind::FloorOutlet | ObjectKind::WallOutlet
    ) {
        return target.support.is_none();
    }
    if host(scene, target) != parent {
        return false;
    }
    let same_station = |screen_id: Option<usize>, tower: &IndoorObject| {
        let (Some(screen), Some(main)) = (
            screen_id.and_then(|id| scene.objects.get(id)),
            tower
                .interaction_target
                .and_then(|id| scene.objects.get(id)),
        ) else {
            return false;
        };
        screen.kind == ObjectKind::Monitor
            && main.kind == ObjectKind::Monitor
            && screen.support == main.support
            && screen.interaction_target == main.interaction_target
    };
    match (source.kind, target.kind) {
        (ObjectKind::Laptop, ObjectKind::Monitor) => {
            source.interaction_target == target.interaction_target
        }
        (ObjectKind::Keyboard | ObjectKind::Mouse, ObjectKind::Monitor) => {
            source.interaction_target == Some(target.id)
        }
        (ObjectKind::Keyboard | ObjectKind::Mouse, ObjectKind::PcTower) => {
            same_station(source.interaction_target, target)
        }
        (ObjectKind::Monitor, ObjectKind::PcTower) => same_station(Some(source.id), target),
        _ => true,
    }
}

impl IndoorManifest {
    pub(super) fn connect_workstation_data(&mut self) {
        if !matches!(
            self.layout,
            IndoorLayout::OpenOffice | IndoorLayout::Coworking
        ) {
            return;
        }
        // Build a compact index once. Only installed, usable workstations offer endpoints.
        let mut towers = std::collections::BTreeMap::new();
        let mut monitors = std::collections::BTreeMap::new();
        let mut powered_laptops = std::collections::BTreeSet::new();
        for o in self.objects.iter().filter(|o| !o.neighbor) {
            match o.kind {
                ObjectKind::PcTower => {
                    if let Some(main) = o.interaction_target.and_then(|id| self.objects.get(id)) {
                        towers.insert((main.support, main.interaction_target), o.id);
                    }
                }
                ObjectKind::Monitor => {
                    monitors
                        .entry((o.support, o.interaction_target))
                        .or_insert(o.id);
                }
                ObjectKind::Cable => {
                    if let Some(source) = o.interaction_target.and_then(|id| self.objects.get(id)) {
                        if source.kind == ObjectKind::Laptop {
                            powered_laptops.insert(source.id);
                        }
                    }
                }
                _ => {}
            }
        }
        let count = self.objects.len();
        for i in 0..count {
            let source = &self.objects[i];
            if source.neighbor
                || !matches!(
                    source.kind,
                    ObjectKind::Monitor
                        | ObjectKind::Mouse
                        | ObjectKind::Keyboard
                        | ObjectKind::Laptop
                )
            {
                continue;
            }
            let Some(parent) = source.support else {
                continue;
            };
            let mut rng = stream(source.seed, 0x4445534b44415441);
            if source.kind == ObjectKind::Laptop {
                if !powered_laptops.contains(&i) {
                    if let Some(&display) = monitors.get(&(Some(parent), source.interaction_target))
                    {
                        self.try_data_connection(parent, i, display, &mut rng);
                    }
                }
                continue;
            }
            let main = if source.kind == ObjectKind::Monitor {
                i
            } else {
                let Some(id) = source.interaction_target else {
                    continue;
                };
                id
            };
            if self.objects[main].kind != ObjectKind::Monitor {
                continue;
            }
            let tower = towers
                .get(&(Some(parent), self.objects[main].interaction_target))
                .copied();
            let mut targets = [tower, None];
            if source.kind != ObjectKind::Monitor {
                if !rng.random_bool(if source.kind == ObjectKind::Mouse {
                    0.66
                } else {
                    0.81
                }) {
                    continue;
                }
                targets[1] = Some(main);
                if rng.random_bool(0.51) {
                    targets.swap(0, 1);
                }
            }
            for target in targets.into_iter().flatten() {
                if self.try_data_connection(parent, i, target, &mut rng) {
                    break;
                }
            }
        }
    }
    fn try_data_connection(
        &mut self,
        desk_id: usize,
        src: usize,
        dst: usize,
        r: &mut ChaCha8Rng,
    ) -> bool {
        let (source, target, desk) = (
            &self.objects[src],
            &self.objects[dst],
            &self.objects[desk_id],
        );
        let Some(variant) = wires::data_variant(source.kind, target.kind) else {
            return false;
        };
        let Some([(start, n), (end, m)]) = ports::data_ports(source, target) else {
            return false;
        };
        let tf = desk.transform();
        let local = |p: Vec3| tf.rotation.inverse() * (p - tf.translation);
        let t = target.transform();
        let front = t.rotation * Vec3::Z;
        let right = t.rotation * Vec3::X;
        let sign = if (start - end).dot(right) >= 0. {
            1.
        } else {
            -1.
        };
        let floor = target.support.is_none();
        let stand = (target.kind == ObjectKind::Monitor)
            .then(|| super::super::super::objects::computers::low_mount_clearance(target));
        for attempt in 0..5 {
            let mut o = self.candidate(ObjectKind::Cable, Vec3::ZERO, Vec3::ONE, 0., r);
            o.variant = variant;
            o.solid = false;
            o.support = Some(desk_id);
            o.interaction_target = Some(src);
            o.cable_target = Some(dst);
            let radius = wires::bound_radius(&o);
            let a = start + n * 0.036;
            let b = end + m * 0.036;
            let y = desk.position.y + desk.size.y + radius + 0.002;
            let mut path = vec![start, a];
            if floor {
                let local_target = local(end);
                let side = local_target.x.signum();
                let z = (local(a).z * 0.43 + local_target.z * 0.57 + r.random_range(-0.085..0.085))
                    .clamp(-desk.size.z * 0.38, desk.size.z * 0.38);
                let edge = tf.transform_point(Vec3::new(
                    side * (desk.size.x * 0.5 + 0.035 + radius),
                    desk.size.y + radius + 0.008,
                    z,
                ));
                if let Some(curve) = wires::hanging_curve(
                    edge,
                    b,
                    self.floor_height(target.position.xz()) + radius + 0.003,
                    r.random_range(0.16..0.49),
                ) {
                    path.extend(curve)
                } else {
                    continue;
                }
            } else {
                let hand = if attempt % 2 == 0 { sign } else { -sign };
                let half = stand
                    .filter(|&(_, h)| y + radius < target.position.y + h)
                    .map_or(target.size.x * 0.5, |(width, _)| width);
                let width = half + r.random_range(0.027..0.071);
                if attempt < 4 {
                    // Along the desk, around the real stand/case, into a rear-facing connector.
                    let side = target.position + right * (hand * width);
                    let rear = (b - target.position).dot(front) - r.random_range(0.016..0.075);
                    let approach = ((a - target.position).dot(front) * r.random_range(0.37..0.78))
                        .clamp(rear + 0.06, target.size.z * 0.5 + 0.11);
                    path.push(
                        (side + right * hand * r.random_range(-0.014..0.047) + front * approach)
                            .with_y(y),
                    );
                    path.push((side + front * rear).with_y(y));
                } else {
                    path.push((a.lerp(b, 0.44) + right * r.random_range(-0.06..0.06)).with_y(y));
                }
                path.push(b);
            }
            path.push(end);
            if wires::archive_world_path(&mut o, &path, 0.32) && cable_route_clear(self, &o) {
                self.objects.push(o);
                return true;
            }
        }
        false
    }
}
