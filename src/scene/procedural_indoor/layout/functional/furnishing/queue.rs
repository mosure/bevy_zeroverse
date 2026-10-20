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
                self.public_fit(
                    index,
                    ObjectKind::QueueBarrier,
                    Vec3::new(x, 0., max.y + size.z * 0.5 + 0.045),
                    size,
                    0.,
                    r,
                );
                break;
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
    }
}
