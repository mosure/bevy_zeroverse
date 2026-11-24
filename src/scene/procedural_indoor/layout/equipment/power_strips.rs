//! Surface-local distribution hardware, admitted before clutter and wiring.
use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, SurfacePattern};

impl IndoorManifest {
    pub(super) fn place_power_strip(&mut self, support: &IndoorObject, r: &mut ChaCha8Rng) {
        if self
            .domain()
            .is_some_and(|d| !r.random_bool((0.12 + 0.88 * d.clutter) as f64))
            || !self.permits_table_item(support, ObjectKind::PowerStrip)
        {
            return;
        }
        // Sample once per product. Failed clearance must not select smaller or
        // more axis-aligned products through repeated shape resampling.
        let size = equipment::power_strip::dimensions(r);
        let seed = r.random();
        let variant = if self.generator_version >= 198 {
            crate::scene::procedural_indoor::objects::wall::installed_variant(
                super::super::decor::electrical_standard(self.seed),
            )
        } else {
            sampling::Domain::PowerStandard
                .rng(self.seed)
                .random_range(0..4)
        };
        let pattern = SurfacePattern::sample(r);
        let edge = r.random_bool(0.72).then(|| r.random_range(0..4));
        // Reversing a strip preserves edge alignment but changes the inlet end.
        // Without this, aligned products always point along +X or +Z locally.
        let reverse = if r.random_bool(0.5) { PI } else { 0. };
        for _ in 0..8 {
            let yaw = edge.map_or_else(
                || pattern.yaw(r),
                |side| reverse + if side < 2 { 0. } else { PI * 0.5 },
            );
            let yaw = yaw + r.random_range(-0.35..0.35);
            let bounds = (-support.size.xz() * 0.5, support.size.xz() * 0.5);
            let Some(mut proposal) = pattern.propose(bounds, size.xz(), yaw, 0.025, r) else {
                continue;
            };
            if let Some(side) = edge {
                let half = rotated_half(size.xz(), yaw);
                let lo = bounds.0 + half + Vec2::splat(0.025);
                let hi = bounds.1 - half - Vec2::splat(0.025);
                let t = r.random_range(0.03..0.25);
                match side {
                    0 => proposal.center.y = lo.y + (hi.y - lo.y) * t,
                    1 => proposal.center.y = hi.y - (hi.y - lo.y) * t,
                    2 => proposal.center.x = lo.x + (hi.x - lo.x) * t,
                    _ => proposal.center.x = hi.x - (hi.x - lo.x) * t,
                }
            }
            let position = support.transform().transform_point(Vec3::new(
                proposal.center.x,
                support.size.y,
                proposal.center.y,
            ));
            let mut o =
                self.candidate(ObjectKind::PowerStrip, position, size, support.yaw + yaw, r);
            o.seed = seed;
            o.variant = variant;
            o.solid = false;
            o.support = Some(support.id);
            o.neighbor = support.neighbor;
            if !self.prop_clear(&o, support, 0.012) {
                continue;
            }
            self.objects.push(o);
            return;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{layout::IndoorLayout, validation};
    #[test]
    fn indexed_strip_leads_seat_in_the_reserved_well() {
        let mut checked_rooms = 0;
        for seed in 0..64 {
            if super::super::super::decor::electrical_standard(seed) != 1 {
                continue;
            }
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::OpenOffice, 0.75, 0, 0.)
                    .unwrap();
            let Some(cable) = scene.objects.iter().find(|c| {
                c.cable_target.is_some_and(|id| {
                    let target = &scene.objects[id];
                    target.kind == ObjectKind::PowerStrip
                        && equipment::receptacles::recessed_connection(
                            target,
                            c.cable_target_socket,
                        )
                })
            }) else {
                continue;
            };
            let target = &scene.objects[cable.cable_target.unwrap()];
            let port =
                equipment::receptacles::target_port(target, false, cable.cable_target_socket)
                    .unwrap();
            let tip = cable
                .transform()
                .transform_point(*cable.cable_path.as_ref().unwrap().last().unwrap());
            assert!(port.distance(tip) < 1e-5);
            assert_eq!(cable.variant, wires::RECESSED_MAINS);
            assert!(cable_route_clear(&scene, cable));
            let mut invalid = cable.clone();
            invalid.variant = 0;
            assert!(!cable_route_clear(&scene, &invalid));
            checked_rooms += 1;
            if checked_rooms == 2 {
                break;
            }
        }
        assert_eq!(checked_rooms, 2, "missing occupied round-well power strips");
    }

    #[test]
    fn accepted_strips_span_support_quadrants_headings_and_metric_sizes() {
        let mut quadrants = std::collections::BTreeSet::new();
        let mut headings = std::collections::BTreeSet::new();
        let mut aligned_heading_counts = [0; 4];
        let mut heights = std::collections::BTreeSet::new();
        let mut positions = std::collections::BTreeSet::new();
        let mut count = 0;
        for seed in 0..64 {
            let s =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::OpenOffice, 0.7, 0, 0.)
                    .unwrap();
            validation::validate_layout(&s).unwrap();
            for o in s
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::PowerStrip)
            {
                let standard = super::super::super::decor::electrical_parameters(&s).standard;
                assert_eq!(
                    o.variant,
                    crate::scene::procedural_indoor::objects::wall::installed_variant(standard)
                );
                for cable in s.objects.iter().filter(|c| c.cable_target == Some(o.id)) {
                    assert_eq!(
                        cable.variant == wires::RECESSED_MAINS,
                        standard == 1 && cable.cable_target_socket.is_some()
                    );
                }
                let support = &s.objects[o.support.unwrap()];
                assert!(s.prop_clear(o, support, 0.011));
                let local = support
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(o.position);
                assert!((local.y - support.size.y).abs() < 1e-5);
                quadrants.insert((local.x > 0., local.z > 0.));
                positions.insert((
                    (local.x / support.size.x * 100.) as i32,
                    (local.z / support.size.z * 100.) as i32,
                ));
                headings.insert(((o.yaw - support.yaw) * 10.) as i32);
                let heading = o.yaw - support.yaw;
                let quarter = (heading / (PI * 0.5)).round();
                if (heading - quarter * PI * 0.5).abs() < 0.36 {
                    aligned_heading_counts[(quarter as i32).rem_euclid(4) as usize] += 1;
                }
                heights.insert((o.size.y * 1000.) as i32);
                count += 1;
            }
        }
        assert!(count > 50, "{count}");
        assert_eq!(quadrants.len(), 4);
        assert!(positions.len() > 45, "{}", positions.len());
        assert!(headings.len() > 20, "{}", headings.len());
        assert!(
            aligned_heading_counts.iter().all(|&n| n > count / 12),
            "aligned inlet directions collapsed: {aligned_heading_counts:?} / {count}"
        );
        assert!(heights.len() > 12, "{}", heights.len());
    }
}
