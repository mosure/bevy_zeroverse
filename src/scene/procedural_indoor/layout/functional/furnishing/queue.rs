//! Service queues follow the actual admitted counter, before dining placement.
use super::*;

impl IndoorManifest {
    pub(super) fn serving_queue(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        if p.activity == IndoorLayout::Cafe && !r.random_bool(0.55) {
            return;
        }
        let center = (p.min.x + p.max.x) * 0.5;
        let aisle = p.parameters.aisle_metres * 0.5 + 0.06;
        let counters: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                o.kind == ObjectKind::ServiceCounter
                    && o.position.x >= p.min.x
                    && o.position.x <= p.max.x
                    && o.position.z >= p.min.y
                    && o.position.z <= p.max.y
            })
            .cloned()
            .collect();
        for counter in counters {
            let front = counter.bounds().1.z;
            let bands = if counter.position.x < center {
                [(p.min.x, center - aisle), (center + aisle, p.max.x)]
            } else {
                [(center + aisle, p.max.x), (p.min.x, center - aisle)]
            };
            for (lo, hi) in bands {
                let length = (counter.size.x * r.random_range(0.70..1.05))
                    .min(hi - lo - 0.10)
                    .min(2.5);
                if length < 0.90 {
                    continue;
                }
                let x = counter
                    .position
                    .x
                    .clamp(lo + length * 0.5 + 0.04, hi - length * 0.5 - 0.04);
                let min = Vec2::new(x - length * 0.5, front + r.random_range(0.12..0.24));
                let max = min + Vec2::new(length, r.random_range(0.85..1.15));
                let queue = Reservation {
                    role: ReservationRole::Queue,
                    min,
                    max,
                };
                if !min.cmpge(p.min).all()
                    || !max.cmple(p.max).all()
                    || p.reservations
                        .iter()
                        .any(|v| v.min.cmplt(max).all() && v.max.cmpgt(min).all())
                    || self.objects.iter().filter(|o| o.solid).any(|o| {
                        let (a, b) = o.bounds();
                        queue.overlaps(a, b)
                    })
                {
                    continue;
                }
                self.assembly_diagnostics.programs[index]
                    .reservations
                    .push(queue);
                let mut size = geometry::dimensions(ObjectKind::QueueBarrier, r.random());
                size.x = length * r.random_range(0.76..0.96);
                let admitted = self.public_fit(
                    index,
                    ObjectKind::QueueBarrier,
                    Vec3::new(x, 0., max.y + size.z * 0.5 + 0.045),
                    size,
                    0.,
                    r,
                );
                if let Some(id) = admitted {
                    if self.generator_version >= geometry::barrier::FITTED_VERSION {
                        self.align_queue_barrier(index, id, min, max);
                    }
                }
                break;
            }
        }
    }

    fn align_queue_barrier(&mut self, zone: usize, id: usize, lane_min: Vec2, lane_max: Vec2) {
        let original = &self.objects[id];
        let mut rng = stream(original.seed, u64::from_le_bytes(*b"QueueYaw"));
        // Deliberately straight installations coexist with hand-set stanchions.
        if rng.random_bool(0.16) {
            return;
        }
        let angle = (rng.random::<f32>() - rng.random::<f32>()) * 0.32;
        let slide =
            (lane_max.x - lane_min.x - original.size.x).max(0.) * rng.random_range(-0.46..0.46);
        let gap = rng.random_range(0.018..0.090);
        let domain = &self.assembly_diagnostics.programs[zone];
        let mut candidate = original.clone();
        for fraction in [1., 0.45] {
            candidate.yaw = fraction * angle;
            let half = crate::scene::procedural_indoor::placement::rotated_half(
                candidate.size.xz(),
                candidate.yaw,
            );
            let (left, right) = (lane_min.x + half.x, lane_max.x - half.x);
            if right < left {
                continue;
            }
            candidate.position.x = (original.position.x + fraction * slide).clamp(left, right);
            candidate.position.z = original.position.z.max(lane_max.y + half.y + gap);
            candidate.position.y = self.floor_height(candidate.position.xz());
            let (lo, hi) = candidate.bounds();
            if lo.xz().cmpge(domain.min).all()
                && hi.xz().cmple(domain.max).all()
                && self.placement_clear(&candidate, 0.035)
            {
                self.objects[id] = candidate;
                return;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn service_queues_admit_multiple_barrier_constructions_before_seating() {
        let mut count = 0;
        let mut styles = std::collections::BTreeSet::new();
        let mut angles = [0; 3];
        for seed in 0..32 {
            let s =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Cafeteria, 0.65, 0, 0.)
                    .unwrap();
            crate::scene::procedural_indoor::validation::validate_layout(&s).unwrap();
            for o in s
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::QueueBarrier)
            {
                count += 1;
                angles[if o.yaw < -0.02 {
                    0
                } else if o.yaw > 0.02 {
                    2
                } else {
                    1
                }] += 1;
                if o.yaw != 0. {
                    let (lo, hi) = o.bounds();
                    assert!(s
                        .functional_programs()
                        .iter()
                        .flat_map(|p| &p.reservations)
                        .any(|q| q.role == ReservationRole::Queue
                            && lo.x >= q.min.x - 1e-5
                            && hi.x <= q.max.x + 1e-5
                            && lo.z >= q.max.y));
                }
                styles.insert(geometry::barrier::parameters(o).construction);
                let mesh = crate::scene::procedural_indoor::objects::build_object(o);
                crate::scene::procedural_indoor::objects::audit::check(o, &mesh).unwrap();
                assert!(s.functional_programs().iter().any(|p| p
                    .reservations
                    .iter()
                    .any(|v| v.role == ReservationRole::Queue)));
            }
        }
        assert!(
            count >= 8,
            "only {count}/32 service layouts admitted a queue barrier"
        );
        assert!(styles.len() >= 3, "{styles:?}");
        assert!(angles.iter().all(|&n| n >= 2), "{angles:?}");
    }
}
