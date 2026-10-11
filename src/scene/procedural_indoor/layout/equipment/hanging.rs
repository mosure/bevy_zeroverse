//! Bounded wall-power routing. Paths are archived after smoothing and checked
//! against the same ports and physical obstacles used by replay validation.
use super::*;

impl IndoorManifest {
    pub(super) fn connect_wall_power(&mut self) {
        let sources: Vec<_> =
            self.objects
                .iter()
                .filter(|o| {
                    !o.neighbor
                        && o.support.is_some()
                        && matches!(
                            o.kind,
                            ObjectKind::PowerStrip | ObjectKind::Monitor | ObjectKind::Projector
                        )
                        && !self.objects.iter().any(|c| {
                            c.kind == ObjectKind::Cable && c.interaction_target == Some(o.id)
                        })
                })
                .cloned()
                .collect();
        for source in sources {
            let parent = self.objects[source.support.unwrap()].clone();
            let mut targets: Vec<_> = self
                .objects
                .iter()
                .filter(|o| {
                    o.kind == ObjectKind::WallOutlet
                        && !o.neighbor
                        && o.position.distance(source.position) < 2.8
                })
                .cloned()
                .collect();
            targets.sort_by(|a, b| {
                a.position
                    .distance_squared(source.position)
                    .total_cmp(&b.position.distance_squared(source.position))
            });
            let mut installed = false;
            for target in targets.into_iter().take(3) {
                let mut rng = stream(source.seed ^ target.seed, 5801);
                // One physical receptacle cannot receive multiple mains plugs.
                if self
                    .objects
                    .iter()
                    .any(|c| c.kind == ObjectKind::Cable && c.cable_target == Some(target.id))
                {
                    continue;
                }
                for side in 0..4 {
                    let mut o =
                        self.candidate(ObjectKind::Cable, Vec3::ZERO, Vec3::ONE, 0., &mut rng);
                    o.solid = false;
                    o.support = Some(parent.id);
                    o.interaction_target = Some(source.id);
                    o.cable_target = Some(target.id);
                    let radius = wires::bound_radius(&o);
                    let start = equipment::source_power_port(&source).unwrap();
                    let end = equipment::power_port(&target, false).unwrap();
                    let a = start + equipment::source_power_normal(&source).unwrap() * 0.039;
                    let b = end + equipment::power_port_normal(&target, false).unwrap() * 0.039;
                    let local = parent
                        .transform()
                        .compute_affine()
                        .inverse()
                        .transform_point3(a);
                    let half = parent.size.xz() * 0.5;
                    let gap = radius + rng.random_range(0.045..0.085);
                    let exit = match side {
                        0 => Vec2::new(local.x.clamp(-half.x + 0.04, half.x - 0.04), -half.y - gap),
                        1 => Vec2::new(-half.x - gap, local.z.clamp(-half.y + 0.04, half.y - 0.04)),
                        2 => Vec2::new(half.x + gap, local.z.clamp(-half.y + 0.04, half.y - 0.04)),
                        _ => Vec2::new(local.x.clamp(-half.x + 0.04, half.x - 0.04), half.y + gap),
                    };
                    let edge = parent.transform().transform_point(Vec3::new(
                        exit.x,
                        if source.kind == ObjectKind::PowerStrip {
                            parent.size.y + radius + 0.003
                        } else {
                            local.y.max(parent.size.y + radius + 0.003)
                        },
                        exit.y,
                    ));
                    let floor = (0..=8)
                        .map(|i| self.floor_height(edge.lerp(b, i as f32 / 8.).xz()))
                        // A raised step must not create an imaginary support
                        // plane over the lower floor. The final route check
                        // rejects any curve that intersects the actual steps.
                        .fold(f32::INFINITY, f32::min)
                        + radius
                        + 0.002;
                    let Some(hanging) =
                        wires::hanging_curve(edge, b, floor, rng.random_range(0.12..0.55))
                    else {
                        continue;
                    };
                    let mut path = vec![start, a];
                    path.extend(hanging);
                    path.push(end);
                    let Some(path) = wires::fillet_path(&path, wires::radius(&o)) else {
                        continue;
                    };
                    // Long straight segments need intermediate samples too;
                    // filleting only inserts samples at curved bends.
                    let mut sampled = vec![path[0]];
                    for pair in path.windows(2) {
                        let steps = (pair[0].distance(pair[1]) / 0.30).ceil().max(1.) as usize;
                        sampled.extend(
                            (1..=steps).map(|i| pair[0].lerp(pair[1], i as f32 / steps as f32)),
                        );
                    }
                    let lo = sampled
                        .iter()
                        .fold(Vec3::splat(f32::INFINITY), |a, &b| a.min(b))
                        - Vec3::splat(radius);
                    let hi = sampled
                        .iter()
                        .fold(Vec3::splat(f32::NEG_INFINITY), |a, &b| a.max(b))
                        + Vec3::splat(radius);
                    o.position = Vec3::new((lo.x + hi.x) * 0.5, lo.y, (lo.z + hi.z) * 0.5);
                    o.size = hi - lo;
                    o.cable_path = Some(sampled.into_iter().map(|p| p - o.position).collect());
                    if cable_route_clear(self, &o) {
                        self.objects.push(o);
                        installed = true;
                        break;
                    }
                }
                if installed {
                    break;
                }
            }
        }
    }
}
