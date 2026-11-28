//! Activity tables assemble by their long edges; each actual seat owns a work
//! position on one half. Fit complete groups before replacing conventional rows.
use super::*;
use crate::scene::procedural_indoor::{items::sizing, objects::tables};
use rand::seq::SliceRandom;

#[derive(Clone, Copy)]
struct Bay {
    min: Vec2,
    max: Vec2,
}

/// A classroom can set out fewer chairs than its tables could accommodate.
struct SeatOrder {
    single: usize,
    paired: [usize; 2],
}

impl SeatOrder {
    fn sample(density: f32, r: &mut ChaCha8Rng) -> Self {
        let probability = f64::from(0.12 + 0.82 * density);
        let mut additional = || usize::from(r.random_bool(probability));
        let single = 2 + additional() + additional();
        let mut paired = [2 + additional(), 1 + additional() + additional()];
        paired.shuffle(r);
        Self { single, paired }
    }
}

struct Group {
    zone: usize,
    bay: Bay,
    size: Vec3,
    product: u64,
    structure: PlacementStructure,
    order: SeatOrder,
}

impl Group {
    fn admit(&self, s: &mut IndoorManifest, o: IndoorObject) -> Option<usize> {
        let (lo, hi) = o.bounds();
        if !lo.xz().cmpge(self.bay.min).all()
            || !hi.xz().cmple(self.bay.max).all()
            || !s.placement_clear_with_structure(&o, 0.035, false, &self.structure)
        {
            s.assembly_diagnostics.programs[self.zone].rejected_candidates += 1;
            return None;
        }
        let id = o.id;
        s.objects.push(o);
        Some(id)
    }

    fn place(
        &self,
        s: &mut IndoorManifest,
        center: Vec3,
        yaw: f32,
        paired: bool,
        stock: &variation::Collection,
        r: &mut ChaCha8Rng,
    ) -> bool {
        let first = s.objects.len();
        let halves = if paired { 2 } else { 1 };
        let offset = if paired {
            (self.size.z + r.random_range(0.005..0.022)) * 0.5
        } else {
            0.
        };
        let mut seats = [0; 2];
        // Both supporting objects must exist before judging the seats: the
        // second half still obstructs a chair associated with the first half.
        for half in 0..halves {
            let heading = yaw + half as f32 * PI;
            let mut table = s.floor_candidate(
                ObjectKind::Table,
                center + Quat::from_rotation_y(heading) * Vec3::Z * offset,
                self.size,
                heading,
                r,
            );
            table.variant = tables::CLASSROOM_VARIANT;
            table.finish_seed = Some(self.product);
            if self.admit(s, table).is_none() {
                s.objects.truncate(first);
                return false;
            }
        }
        for (half, count) in seats.iter_mut().enumerate().take(halves) {
            let table = s.objects[first + half].clone();
            let corners = tables::classroom_table(&table)
                .unwrap()
                .corners(table.size.xz() * 0.5);
            let mut edges = [0, 1, 2, 3];
            edges.shuffle(r);
            let quota = if paired {
                self.order.paired[half]
            } else {
                self.order.single
            };
            for edge in edges {
                if *count == quota {
                    break;
                }
                if paired && edge == 0 {
                    continue;
                }
                let a = corners[edge];
                let b = corners[(edge + 1) % 4];
                let outward = -(b - a).normalize().perp();
                let size = stock.chair(r);
                let target = a.lerp(b, r.random_range(0.39..0.61));
                let local = target + outward * (size.z * 0.5 + r.random_range(0.075..0.17));
                let pos = table
                    .transform()
                    .transform_point(Vec3::new(local.x, 0., local.y))
                    .with_y(0.);
                let heading = table.yaw + outward.x.atan2(outward.y) + r.random_range(-0.19..0.19);
                let mut chair = s.floor_candidate(ObjectKind::Chair, pos, size, heading, r);
                chair.variant = 3;
                chair.finish_seed = Some(self.product);
                chair.interaction_target = Some(table.id);
                *count += usize::from(self.admit(s, chair).is_some());
            }
        }
        let enough = if paired {
            seats[0] > 0 && seats[1] > 0 && seats[0] + seats[1] >= 3
        } else {
            seats[0] >= 2
        };
        if !enough {
            s.objects.truncate(first);
        }
        enough
    }
}

impl IndoorManifest {
    /// Per-bank design capacity. Zero leaves the existing individual desk or
    /// tablet-chair program in charge of the entire bank.
    pub(super) fn classroom_groups(
        &mut self,
        zone: usize,
        stock: &variation::Collection,
    ) -> [usize; 2] {
        let p = &self.assembly_diagnostics.programs[zone];
        let mut r = stream(
            self.seed,
            u64::from_le_bytes(*b"ClassPod").wrapping_add(zone as u64),
        );
        let fraction = p.parameters.collaborative_fraction;
        if self.generator_version < tables::CLASSROOM_VERSION
            || !r.random_bool((0.22 + 0.66 * fraction) as f64)
        {
            return [0, 0];
        }
        let aisle = p.parameters.aisle_metres * 0.5;
        let (lo, hi) = (p.min, p.max);
        let axis = (lo.x + hi.x) * 0.5;
        let product = r.random();
        let width = r.random_range(1.18..1.66);
        let size = Vec3::new(
            width,
            r.random_range(0.69..0.79),
            width * r.random_range(0.43..0.53),
        );
        let pair_probability = r.random_range(0.44..0.91);
        let hybrid = r
            .random_bool((0.52 - 0.27 * fraction) as f64)
            .then(|| r.random_range(0..2));
        let mut capacity = [0; 2];
        for (side, (left, right)) in [(lo.x, axis - aisle), (axis + aisle, hi.x)]
            .into_iter()
            .enumerate()
        {
            if hybrid == Some(side) {
                continue;
            }
            let bay = Bay {
                min: Vec2::new(left + 0.04, lo.y + 2.65),
                max: Vec2::new(right - 0.04, hi.y - 0.04),
            };
            let span = bay.max - bay.min;
            if span.min_element() < 1.66 {
                continue;
            }
            let pitch = r.random_range(2.68..3.21_f32) + 0.82 * (1. - self.density);
            let count = (span.y / pitch).floor().clamp(1., 4.) as usize;
            for row in 0..count {
                let depth = span.y / count as f32;
                let band = Bay {
                    min: bay.min + Vec2::Y * (depth * row as f32),
                    max: Vec2::new(bay.max.x, bay.min.y + depth * (row + 1) as f32),
                };
                let mut demand = stream(product, 0x434c4153_53454154 + (side * 4 + row) as u64);
                let order = SeatOrder::sample(self.density, &mut demand);
                let group = Group {
                    zone,
                    bay: band,
                    size,
                    product,
                    structure: PlacementStructure::new(),
                    order,
                };
                let prefer_pair = r.random_bool(pair_probability);
                let orientation = r.random_range(-PI..PI);
                for attempt in 0..4 {
                    let paired = prefer_pair && attempt < 3;
                    // Sample in the free part of the bay. Reorient the kit as a
                    // unit when a column, doorway or side wall prevents a fit.
                    let slack =
                        ((band.max - band.min) * 0.5 - Vec2::splat(1.05)).max(Vec2::splat(0.035));
                    let c = (band.max + band.min) * 0.5
                        + slack
                            * Vec2::new(r.random_range(-0.77..0.77), r.random_range(-0.81..0.81));
                    let heading = orientation + attempt as f32 * 1.20;
                    if group.place(
                        self,
                        Vec3::new(c.x, 0., c.y),
                        heading,
                        paired,
                        stock,
                        &mut r,
                    ) {
                        capacity[side] += if paired {
                            group.order.paired.iter().sum::<usize>()
                        } else {
                            group.order.single
                        };
                        break;
                    }
                }
            }
        }
        capacity
    }

    pub(super) fn dress_classroom_table(&mut self, table: &IndoorObject, r: &mut ChaCha8Rng) {
        let clutter = self.clutter.sample(table.seed);
        if clutter <= 0. {
            return;
        }
        let rotation = Quat::from_rotation_y(-table.yaw);
        let seats: Vec<_> = self
            .objects
            .iter()
            .filter(|o| o.kind == ObjectKind::Chair && o.interaction_target == Some(table.id))
            .map(|o| {
                (
                    (rotation * (o.position - table.position)).xz(),
                    o.yaw - table.yaw,
                )
            })
            .collect();
        for (position, yaw) in seats {
            let outward = Vec2::new(yaw.sin(), yaw.cos());
            let work = position - outward * r.random_range(0.64..0.78);
            let right = Vec2::new(yaw.cos(), -yaw.sin());
            let handedness = if r.random_bool(0.76) { 1. } else { -1. };
            let book = if r.random_bool(0.46) {
                ObjectKind::Notebook
            } else {
                ObjectKind::Notepad
            };
            let personal = if r.random_bool(0.62) {
                ObjectKind::WaterBottle
            } else {
                ObjectKind::Phone
            };
            for (kind, offset, inset, probability) in [
                (book, -0.032, 0., 0.40 + 0.58 * clutter),
                (ObjectKind::Pencil, 0.115, 0., 0.18 + 0.67 * clutter),
                (personal, -0.16, 0.09, 0.10 + 0.30 * clutter),
            ] {
                if !r.random_bool(probability as f64) {
                    continue;
                }
                let size = sizing::loose(kind, r).unwrap();
                let at = work - outward * inset
                    + right * (handedness * offset + r.random_range(-0.023..0.023));
                let heading = yaw + r.random_range(-0.38..0.38);
                self.program_prop(table, kind, Vec3::new(at.x, 0., at.y), size, heading, r);
            }
        }
    }
}
