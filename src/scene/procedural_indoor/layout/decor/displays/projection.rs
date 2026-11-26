//! Projection screens serve a viewing area. Mount proposals reuse the same
//! structural clearance and furniture sightline checks as installed televisions.
use super::*;
use crate::scene::procedural_indoor::objects::presentation::projection as screen;

fn demand(scene: &IndoorManifest) -> usize {
    use IndoorLayout::*;
    if scene.density <= 0. {
        return 0;
    }
    let chance: f32 = match scene.layout {
        LectureHall => 0.94,
        Training => 0.85,
        Classroom => 0.70,
        Conference => 0.65,
        Studio | Showroom | PublicSuite => 0.30,
        Library | Workshop | Coworking => 0.25,
        OpenOffice | Gallery | Cafeteria => 0.16,
        Lounge | Breakroom | Cafe | Reception | Mixed => 0.085,
        Kitchen | Kitchenette | ElevatorLanding | Restaurant | FitnessRoom | CyclingStudio
        | Gymnasium | GroceryStore | ClothingStore | Hallway | StairHall | Courtyard => 0.012,
    };
    let mut rng = sampling::Domain::ProjectionScreens.rng(scene.seed);
    if !rng.random_bool(f64::from(
        (chance * (scene.density / 0.65).sqrt()).min(0.98),
    )) {
        return 0;
    }
    let region = scene.primary_region();
    let area = (region.max - region.min).element_product();
    let large_venue = matches!(scene.layout, LectureHall | Training | Conference)
        && area > 75.
        && scene
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && o.kind == ObjectKind::Chair
                    && region.contains_projection(o.position.xz(), 0.)
            })
            .take(12)
            .count()
            == 12;
    1 + usize::from(
        large_venue && rng.random_bool(f64::from(((area - 75.) / 160.).clamp(0., 1.) * 0.47)),
    )
}

impl IndoorManifest {
    pub(in crate::scene::procedural_indoor::layout) fn furnish_projection_screens(&mut self) {
        let demand = demand(self);
        if demand == 0 {
            return;
        }
        let audience = Audience::sample(self);
        if audience.eyes.is_empty() {
            return;
        }
        let mut space = MountSpace::sample(self);
        let (lo, hi) = self.primary_room_bounds();
        let long_room = (hi - lo).max_element();
        let mut first: Option<IndoorObject> = None;
        for ordinal in 0..demand {
            let mut r = sampling::Domain::ProjectionScreens.proposal(self.seed, ordinal);
            let mut o = self.candidate(
                ObjectKind::ProjectionScreen,
                Vec3::ZERO,
                Vec3::ONE,
                0.,
                &mut r,
            );
            o.solid = false;
            o.variant |= screen::INSTALLED;
            // A twin installation uses comparable stock on the same wall. Give
            // matching a finite budget independent of all placement trials.
            let width = if let Some(prev) = &first {
                let a = screen::parameters(prev).installation.unwrap();
                if !(0..12).any(|_| {
                    o.seed = r.random();
                    let b = screen::parameters(&o).installation.unwrap();
                    a.format == b.format && a.mechanism == b.mechanism
                }) {
                    break;
                }
                o.finish_seed = Some(prev.finish_seed.unwrap_or(prev.seed));
                prev.size.x * r.random_range(0.96..1.04)
            } else {
                let common = r.random_range(1.35_f32..2.35);
                (common * (long_room / 7.).clamp(0.72, 1.65)).min(3.70)
            };
            let depth = r.random_range(0.057..0.118);
            o.size = screen::dimensions(o.seed, width, depth);
            let p = screen::parameters(&o);
            // Paired cassettes share a mounting line, even on a stepped floor
            // or with different amounts of cloth exposed.
            let twin_top = first
                .as_ref()
                .map(|f| f.position.y + f.size.y + r.random_range(-0.025..0.025));
            let mut best: Option<(f32, Vec3, f32)> = None;
            let region = self.primary_region();
            for _ in 0..24 {
                let (wall, span) = self.attachment_wall(&mut r);
                let slack = (span - o.size.x) * 0.5 - 0.12;
                if slack <= 0. {
                    continue;
                }
                o.yaw = wall.rotation.to_euler(EulerRot::YXZ).0;
                o.position = wall.transform_point(Vec3::new(
                    r.random_range(-slack..slack),
                    0.,
                    o.size.z * 0.5 + 0.001,
                ));
                if !region.contains_projection(o.position.xz(), 0.) {
                    continue;
                }
                // Sample bottom clearance from the local floor, including
                // raised teaching platforms, and allow tall products to fit.
                let floor = self.floor_height(o.position.xz());
                let high =
                    (self.ceiling_height(o.position.xz()) - floor - o.size.y - 0.12).min(1.48);
                if high < 0.64 {
                    continue;
                }
                o.position.y = if let Some(top) = twin_top {
                    let bottom = top - o.size.y;
                    if !(floor + 0.64..=floor + high).contains(&bottom) {
                        continue;
                    }
                    bottom
                } else {
                    floor + r.random_range(0.64..=high)
                };
                let front = o.transform().rotation * Vec3::Z;
                if first.as_ref().is_some_and(|f| {
                    let offset = o.position - f.position;
                    front.dot(f.transform().rotation * Vec3::Z) < 0.97
                        || offset.dot(front).abs() > 0.16
                        || offset.xz().length() < (o.size.x + f.size.x) * 0.5 + 0.38
                }) || !self.attachment_clear(wall, span, &o)
                    || !space.clear(&o, None)
                {
                    continue;
                }
                // Sight the intended image even when the roll is stored.
                let target = o.transform().transform_point(p.viewing_target_local()) + front * 0.01;
                let Some(sight) = audience.score_at(self, &o, &space, None, target) else {
                    continue;
                };
                let score = sight + r.random_range(0.0..0.48);
                if best.is_none_or(|(s, _, _)| score > s) {
                    best = Some((score, o.position, o.yaw));
                }
            }
            let Some((_, pos, yaw)) = best else {
                break;
            };
            o.position = pos;
            o.yaw = yaw;
            space.objects.push((o.id, o.bounds()));
            if first.is_none() {
                first = Some(o.clone());
            }
            self.objects.push(o);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn twin_screens_share_a_mounting_line_and_leave_space_between_cases() {
        let mut pairs = 0;
        for seed in 93400..93496 {
            let room =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::LectureHall, 0.65, 0, 0.)
                    .unwrap();
            let mut screens = room
                .objects
                .iter()
                .filter(|o| o.kind == ObjectKind::ProjectionScreen && !o.neighbor);
            let (Some(a), Some(b)) = (screens.next(), screens.next()) else {
                continue;
            };
            assert!(screens.next().is_none());
            assert!(((a.position.y + a.size.y) - (b.position.y + b.size.y)).abs() <= 0.0251);
            let delta = b.position - a.position;
            let front = a.transform().rotation * Vec3::Z;
            assert!(front.dot(b.transform().rotation * Vec3::Z) >= 0.97);
            assert!(delta.dot(front).abs() <= 0.16);
            assert!(delta.xz().length() >= (a.size.x + b.size.x) * 0.5 + 0.38);
            pairs += 1;
        }
        assert!(
            pairs > 0,
            "lecture-hall cohort must exercise paired screens"
        );
    }

    #[test]
    fn projection_demand_depends_on_room_program_and_density_and_accepts_empty_rooms() {
        let mut room =
            IndoorManifest::generate_with_humans(202, IndoorLayout::Conference, 0.65, 0, 0.)
                .unwrap();
        for (profile, low, high) in [
            (IndoorLayout::LectureHall, 440, 512),
            (IndoorLayout::Conference, 280, 375),
            (IndoorLayout::Kitchenette, 0, 17),
        ] {
            room.layout = profile;
            let count = (0..512)
                .filter(|&seed| {
                    room.seed = seed;
                    demand(&room) > 0
                })
                .count();
            assert!((low..=high).contains(&count), "{profile:?} {count}");
        }
        room.layout = IndoorLayout::Training;
        room.density = 0.;
        assert_eq!(demand(&room), 0);
    }
}
