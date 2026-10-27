//! Surface dressing keeps density decisions separate from physical admission.
use super::super::items::{self, composition};
use super::*;
use std::f32::consts::{FRAC_PI_2, PI};

impl IndoorManifest {
    pub(super) fn scatter_clutter(&mut self) {
        let surfaces: Vec<_> = self
            .objects
            .iter()
            .filter(|o| {
                matches!(
                    o.kind,
                    ObjectKind::Desk
                        | ObjectKind::Table
                        | ObjectKind::CoffeeTable
                        | ObjectKind::SideTable
                        | ObjectKind::Cabinet
                        | ObjectKind::MobileCart
                        | ObjectKind::DrawerUnit
                        | ObjectKind::WallShelf
                )
            })
            .cloned()
            .collect();
        for support in surfaces {
            self.dress_loose_surface(&support);
        }
    }

    fn dress_loose_surface(&mut self, support: &IndoorObject) {
        let program = items::parameters(self, support);
        let composition = composition::parameters(self, support);
        let mut rng = stream(support.seed, 5510);
        // Collect only this top once; don't repeatedly scan the whole room for
        // singleton counts or occupied area. The physical gate remains shared.
        if composition.target_items == 0 {
            return;
        }
        let mut area = 0.;
        let mut present: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.support == Some(support.id) && o.kind != ObjectKind::Cable)
            .map(|o| {
                area += o.size.x * o.size.z;
                o.kind
            })
            .collect();
        let limit = composition.target_coverage * support.size.x * support.size.z;
        for _ in 0..composition.max_attempts {
            if present.len() >= composition.target_items || area >= limit {
                break;
            }
            let cluster = composition.cluster(&mut rng);
            let mut task = program.clone();
            // Related belongings dominate each cluster without fixing its exact
            // inventory. Different clusters on one table can serve different uses.
            task.task_weights.iter_mut().for_each(|w| *w *= 0.35);
            task.task_weights[cluster.task] += 0.65;
            let (mut kind, mut size) = task.item(support, &mut rng);
            let dining = matches!(
                self.layout,
                IndoorLayout::Kitchen
                    | IndoorLayout::Kitchenette
                    | IndoorLayout::Cafe
                    | IndoorLayout::Cafeteria
                    | IndoorLayout::Breakroom
            ) && support.kind == ObjectKind::Table
                && !support.neighbor;
            if dining
                && !matches!(
                    kind,
                    ObjectKind::Mug
                        | ObjectKind::CoffeeCup
                        | ObjectKind::WaterBottle
                        | ObjectKind::SodaCan
                        | ObjectKind::Bowl
                        | ObjectKind::Vase
                        | ObjectKind::Plant
                        | ObjectKind::Phone
                        | ObjectKind::TissueBox
                        | ObjectKind::Notebook
                        | ObjectKind::Bag
                )
            {
                (kind, size) = decor::drink(&mut rng);
            }
            let singleton = matches!(
                kind,
                ObjectKind::CoffeeMaker
                    | ObjectKind::Kettle
                    | ObjectKind::Printer
                    | ObjectKind::DeskLamp
                    | ObjectKind::DeskFan
                    | ObjectKind::Bag
                    | ObjectKind::TissueBox
                    | ObjectKind::Plant
                    | ObjectKind::Vase
                    | ObjectKind::DeskOrganizer
                    | ObjectKind::DocumentTray
            );
            if singleton && present.contains(&kind) {
                continue;
            }
            let Some((offset, yaw)) =
                composition.proposal(cluster, support, size, program.alignment, &mut rng)
            else {
                continue;
            };
            let count = self.objects.len();
            // Density was already sampled into the target. Re-thinning here
            // would square the clutter control and disproportionately empty tops.
            self.supported_prop(support, kind, offset, size, yaw, &mut rng);
            if self.objects.len() > count {
                present.push(kind);
                area += size.x * size.z;
            }
        }
    }

    pub(super) fn conference_props(&mut self, surface: &IndoorObject, rng: &mut ChaCha8Rng) {
        if !self.clutter.active(surface.seed) {
            return;
        }
        let inverse = surface.transform().compute_affine().inverse();
        let seats: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(surface.id))
            .map(|o| inverse.transform_point3(o.position))
            .collect();
        let program = items::parameters(self, surface);
        let activity =
            (self.domain().map_or(0.5, |d| d.clutter) * self.density * program.occupancy)
                .clamp(0., 1.);
        let has_seats = !seats.is_empty();
        for seat in seats {
            // Keep each user's belongings near that seat. No whole-table
            // mirroring, fixed centre pair, or identical inventory per attendee.
            if !rng.random_bool((0.12 + 0.84 * activity) as f64) {
                continue;
            }
            let side = seat.x.abs() / surface.size.x > seat.z.abs() / surface.size.z;
            let yaw = if side {
                seat.x.signum() * FRAC_PI_2
            } else if seat.z > 0. {
                0.
            } else {
                PI
            };
            let orientation = Quat::from_rotation_y(yaw);
            let inset = rng.random_range(0.22..0.34);
            let limit = (surface.size.xz() * 0.5 - Vec2::splat(inset)).max(Vec2::ZERO);
            let center = seat.xz().clamp(-limit, limit);
            let center = Vec3::new(center.x, 0., center.y)
                + orientation * Vec3::new(rng.random_range(-0.075..0.075), 0., 0.);
            let place =
                |scene: &mut Self, kind, size, local: Vec3, rotation: f32, rng: &mut ChaCha8Rng| {
                    let offset = center + orientation * local;
                    scene.supported_prop(surface, kind, offset, size, yaw + rotation, rng);
                };
            if !matches!(self.layout, IndoorLayout::Breakroom)
                && rng.random_bool((0.20 + 0.42 * program.task_weights[0]) as f64)
            {
                place(
                    self,
                    ObjectKind::Laptop,
                    Vec3::new(
                        rng.random_range(0.27..0.42),
                        rng.random_range(0.23..0.32),
                        rng.random_range(0.30..0.38),
                    ),
                    Vec3::ZERO,
                    rng.random_range(-0.16..0.16),
                    rng,
                );
            }
            let handed = if rng.random_bool(0.5) { -1. } else { 1. };
            if rng.random_bool(0.70) {
                let kind = if rng.random_bool(0.5) {
                    ObjectKind::Notepad
                } else {
                    ObjectKind::Notebook
                };
                let size = items::sizing::loose(kind, rng).unwrap();
                place(
                    self,
                    kind,
                    size,
                    Vec3::new(
                        -handed * rng.random_range(0.18..0.33),
                        0.,
                        rng.random_range(-0.08..0.06),
                    ),
                    rng.random_range(-0.65..0.65),
                    rng,
                );
            }
            if rng.random_bool(0.72) {
                let (kind, size) = decor::drink(rng);
                place(
                    self,
                    kind,
                    size,
                    Vec3::new(
                        handed * rng.random_range(0.14..0.34),
                        0.,
                        rng.random_range(-0.16..0.03),
                    ),
                    rng.random_range(-PI..PI),
                    rng,
                );
            }
            if rng.random_bool(0.42) {
                let kind = if rng.random_bool(0.56) {
                    ObjectKind::Pencil
                } else {
                    ObjectKind::Phone
                };
                let size = items::sizing::loose(kind, rng).unwrap();
                place(
                    self,
                    kind,
                    size,
                    Vec3::new(
                        rng.random_range(-0.27..0.27),
                        0.,
                        rng.random_range(-0.13..0.10),
                    ),
                    rng.random_range(-1.3..1.3),
                    rng,
                );
            }
        }
        if has_seats && rng.random_bool((0.12 + activity * 0.38) as f64) {
            let size = items::sizing::loose(ObjectKind::Microphone, rng).unwrap();
            self.supported_prop(
                surface,
                ObjectKind::Microphone,
                Vec3::new(
                    rng.random_range(-0.15..0.15) * surface.size.x,
                    0.,
                    rng.random_range(-0.20..0.20) * surface.size.z,
                ),
                size,
                rng.random_range(-PI..PI),
                rng,
            );
        }
    }
}
#[cfg(test)]
mod tests;
