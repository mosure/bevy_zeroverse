//! Place a bounded collection in the reading room, sharing browsing aisles between stacks.
use super::*;
use crate::scene::procedural_indoor::{objects::storage::library_stack, program::Zone};

pub(super) const INTRODUCED: u32 = 296;

impl IndoorManifest {
    pub(super) fn furnish_library_stacks(&mut self, zones: &[Zone]) {
        let mut r = stream(self.seed, u64::from_le_bytes(*b"LibAisle"));
        let mut rooms: Vec<_> = zones
            .iter()
            .filter(|zone| !zone.activity.is_kitchen())
            .collect();
        // Small reading-only libraries still occur. In suites the largest reading
        // room usually holds the collection, with an occasional secondary section.
        if r.random_bool(0.08) {
            return;
        }
        rooms.sort_by(|a, b| {
            (b.max - b.min)
                .element_product()
                .total_cmp(&(a.max - a.min).element_product())
        });
        let mut remaining = r.random_range(1..=3);
        for zone in rooms.into_iter().take(2) {
            let span = zone.max - zone.min;
            if remaining == 0 || span.element_product() < 13. {
                continue;
            }
            let along_x = (span.x > span.y) ^ r.random_bool(0.19);
            let yaw = if along_x {
                0.
            } else {
                std::f32::consts::FRAC_PI_2
            } + r.random_range(-0.16..0.16);
            let across = (Quat::from_rotation_y(yaw) * Vec3::Z).xz();
            let finish = r.random::<u64>();
            let mut last: Option<Vec2> = None;
            let side = if r.random_bool(0.5) { 1. } else { -1. };
            let pitch = r.random_range(1.85..2.48);
            let structure = PlacementStructure::new();
            let mut installed = 0;
            for _ in 0..remaining {
                let size = library_stack::dimensions(r.random());
                let mut stack =
                    self.floor_candidate(ObjectKind::LibraryStack, Vec3::ZERO, size, yaw, &mut r);
                stack.finish_seed = Some(finish);
                let offset = super::super::super::placement::rotated_half(
                    super::access::envelope(&stack).unwrap().size.xz(),
                    yaw,
                ) + Vec2::splat(0.08);
                let min = zone.min + offset;
                let max = zone.max - offset;
                if min.cmpge(max).any() {
                    continue;
                }
                // Sample the first stack over the usable floor. Subsequent proposals
                // prefer its parallel browsing aisle before trying other valid sites.
                for trial in 0..12 {
                    let site = if let Some(p) = last.filter(|_| trial < 2) {
                        p + across * pitch * if trial == 0 { side } else { -side }
                    } else {
                        min + (max - min) * Vec2::new(r.random(), r.random())
                    };
                    if site.cmplt(min).any() || site.cmpgt(max).any() {
                        continue;
                    }
                    stack.position = Vec3::new(site.x, self.floor_height(site), site.y);
                    if self.placement_clear_with_structure(&stack, 0.06, false, &structure) {
                        self.objects.push(stack);
                        installed += 1;
                        last = Some(site);
                        break;
                    }
                }
            }
            remaining -= installed;
            if remaining > 0 && !r.random_bool(0.55) {
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn library_rug_at_the_edge_of_a_browsing_aisle_preserves_shelf_support() {
        let room =
            IndoorManifest::generate_with_humans(83, IndoorLayout::Library, 0.65, 0, 0.25).unwrap();
        let crosses_aisle = room
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::LibraryStack)
            .any(|o| {
                let (lo, hi) = o.bounds();
                let (a, b) = super::super::access::envelope(o).unwrap().bounds();
                room.floor_support_clear(lo, hi)
                    && !room.floor_support_clear(a, b)
                    && room.placement_clear(o, 0.)
            });
        assert!(crosses_aisle);
        crate::scene::procedural_indoor::validation::validate_layout(&room).unwrap();
    }

    #[test]
    fn library_browsing_survives_furnishing_and_reading_only_rooms_still_occur() {
        let mut counts = [0; 4];
        let mut paired = false;
        let mut mixed_use = 0;
        for seed in 0..96 {
            let room =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Library, 0.65, 0, 0.15)
                    .unwrap();
            crate::scene::procedural_indoor::validation::validate_layout(&room).unwrap();
            let stacks: Vec<_> = room
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::LibraryStack)
                .collect();
            counts[stacks.len()] += 1;
            if !stacks.is_empty()
                && room
                    .objects
                    .iter()
                    .any(|o| !o.neighbor && matches!(o.kind, ObjectKind::Chair | ObjectKind::Sofa))
            {
                mixed_use += 1;
            }
            for stack in stacks {
                assert!(room.placement_clear(stack, 0.));
                let open = super::super::access::envelope(stack).unwrap();
                let mut obstacle = stack.clone();
                obstacle.kind = ObjectKind::Desk;
                obstacle.size = Vec3::splat(0.32);
                for side in [-1., 1.] {
                    obstacle.position = stack.transform().transform_point(Vec3::new(
                        0.,
                        0.,
                        side * (stack.size.z + open.size.z) * 0.25,
                    ));
                    assert!(
                        super::super::access::blocks(stack, &obstacle)
                            && super::super::access::blocks(&obstacle, stack)
                    );
                }
                if room
                    .objects
                    .iter()
                    .any(|o| o.id != stack.id && o.kind == stack.kind && o.yaw == stack.yaw)
                {
                    paired = true;
                }
            }
            if seed < 4 {
                let replay = IndoorManifest::generate_with_humans(
                    seed,
                    IndoorLayout::Library,
                    0.65,
                    0,
                    0.15,
                )
                .unwrap();
                assert_eq!(room, replay);
            }
        }
        assert!(
            counts[0] > 0
                && counts[0] < 48
                && counts[2] + counts[3] > 8
                && paired
                && mixed_use > 40,
            "counts={counts:?}, mixed={mixed_use}"
        );
    }
}
