//! Variable service constellations, fitted beside actual accepted furniture.
//! Rejection never moves anchors, removes furniture or relaxes circulation checks.
use super::*;
impl IndoorManifest {
    pub(super) fn furnish_amenities(&mut self) {
        let mut rng = stream(self.seed, 5600);
        use rand::seq::SliceRandom;
        let mut anchors: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                !o.neighbor
                    && matches!(
                        o.kind,
                        ObjectKind::Desk
                            | ObjectKind::Table
                            | ObjectKind::CoffeeTable
                            | ObjectKind::Sofa
                            | ObjectKind::Cabinet
                    )
            })
            .cloned()
            .collect();
        anchors.shuffle(&mut rng);
        let budget = ((self.room_size.x * self.room_size.z).sqrt()
            * self.density
            * rng.random_range(0.45..0.90))
        .round()
        .clamp(1., 12.) as usize;
        let mut accepted = 0;
        for anchor in anchors {
            if accepted >= budget {
                break;
            }
            let personal = matches!(anchor.kind, ObjectKind::Desk | ObjectKind::Table);
            let lounge = matches!(anchor.kind, ObjectKind::CoffeeTable | ObjectKind::Sofa);
            // Independent probabilities let a station have any subset, rather
            // than selecting one of a few complete predefined room templates.
            let choices = [
                (ObjectKind::DrawerUnit, if personal { 0.62 } else { 0.12 }),
                (ObjectKind::SideTable, if lounge { 0.72 } else { 0.22 }),
                (
                    ObjectKind::MobileCart,
                    if self.layout == IndoorLayout::Breakroom {
                        0.62
                    } else {
                        0.32
                    },
                ),
                (ObjectKind::Ottoman, if lounge { 0.54 } else { 0.08 }),
                (ObjectKind::Bench, 0.23),
                (ObjectKind::RoomDivider, if personal { 0.30 } else { 0.12 }),
            ];
            for (kind, probability) in choices {
                if accepted >= budget {
                    break;
                }
                if !rng.random_bool(probability * (0.35 + 0.65 * self.density as f64)) {
                    continue;
                }
                let size = match kind {
                    ObjectKind::DrawerUnit => Vec3::new(
                        rng.random_range(0.35..0.53),
                        rng.random_range(0.51..0.70),
                        rng.random_range(0.37..0.54),
                    ),
                    ObjectKind::SideTable => Vec3::new(
                        rng.random_range(0.42..0.72),
                        rng.random_range(0.42..0.66),
                        rng.random_range(0.39..0.62),
                    ),
                    ObjectKind::MobileCart => Vec3::new(
                        rng.random_range(0.60..0.91),
                        rng.random_range(0.74..1.02),
                        rng.random_range(0.42..0.59),
                    ),
                    ObjectKind::Ottoman => {
                        super::super::objects::services::ottoman::dimensions(&mut rng)
                    }
                    ObjectKind::Bench => Vec3::new(
                        rng.random_range(0.95..1.70),
                        rng.random_range(0.46..0.87),
                        rng.random_range(0.39..0.54),
                    ),
                    ObjectKind::RoomDivider => Vec3::new(
                        rng.random_range(0.90..1.60),
                        rng.random_range(1.05..1.70),
                        rng.random_range(0.35..0.49),
                    ),
                    _ => unreachable!(),
                };
                for _ in 0..12 {
                    let side = if rng.random_bool(0.5) { -1. } else { 1. };
                    let local = Vec3::new(
                        side * (anchor.size.x + size.x) * 0.5 + side * rng.random_range(0.18..0.50),
                        0.,
                        rng.random_range(-0.55..0.55) * anchor.size.z,
                    );
                    let position =
                        (anchor.position + Quat::from_rotation_y(anchor.yaw) * local).with_y(0.);
                    let yaw = anchor.yaw + rng.random_range(-0.24..0.24);
                    if self.add(kind, position, size, yaw, &mut rng).is_some() {
                        accepted += 1;
                        break;
                    }
                }
            }
        }
    }
}
