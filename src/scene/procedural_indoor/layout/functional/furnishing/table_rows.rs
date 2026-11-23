//! Shared tops with several seats per side. Pair relationships exist before
//! collision admission; a failed seating group leaves no orphan table/receipt.
use super::*;
use crate::scene::procedural_indoor::objects::tables::SHARED_ROW_VARIANT;
use rand::seq::SliceRandom;

struct Row {
    min: Vec2,
    max: Vec2,
    double_sided: bool,
    product: u64,
}

/// An installation may have a smaller second order of replacement furniture.
/// Assign the stock to the floor location before retries or individual seat placement.
struct RowOrder {
    products: [u64; 2],
    replacement_share: f64,
    assignment: u64,
}

impl RowOrder {
    fn sample(seed: u64, zone: usize) -> Self {
        let mut r = stream(
            seed,
            u64::from_le_bytes(*b"RowOrder").wrapping_add(zone as u64),
        );
        Self {
            products: r.random(),
            replacement_share: if r.random_bool(0.48) {
                r.random_range(0.09..0.40)
            } else {
                0.
            },
            assignment: r.random(),
        }
    }

    fn product(&self, bank: usize) -> u64 {
        let secondary = stream(self.assignment, bank as u64).random_bool(self.replacement_share);
        self.products[usize::from(secondary)]
    }
}
impl Row {
    fn contains(&self, o: &IndoorObject) -> bool {
        let (a, b) = o.bounds();
        a.xz().cmpge(self.min + Vec2::splat(0.04)).all()
            && b.xz().cmple(self.max - Vec2::splat(0.04)).all()
    }
}

impl IndoorManifest {
    pub(super) fn table_row_program(
        &mut self,
        index: usize,
        double_sided: bool,
        r: &mut ChaCha8Rng,
    ) -> usize {
        let p = self.assembly_diagnostics.programs[index].clone();
        let policy = &p.parameters.table_rows;
        let center = (p.min + p.max) * 0.5;
        let half_aisle = p.parameters.aisle_metres * 0.5;
        let rows = if double_sided {
            let minimum = policy.dining_depth_metres + 1.94 + policy.aisle_metres;
            p.parameters.lecture_seating.rows_with_clearance(
                p.min.y + 3.2,
                p.max.y - 0.25,
                minimum + r.random_range(0.12..0.40),
                minimum,
            )
        } else {
            p.parameters.lecture_rows(p.min.y + 3., p.max.y - 1.)
        };
        let mut tables = 0;
        let order = RowOrder::sample(self.seed, index);
        for (course, [front, back]) in rows.into_iter().enumerate() {
            for (side, (lo, hi)) in [
                (p.min.x, center.x - half_aisle),
                (center.x + half_aisle, p.max.x),
            ]
            .into_iter()
            .enumerate()
            {
                let route = Reservation {
                    role: ReservationRole::Circulation,
                    min: Vec2::new(lo, back - policy.aisle_metres),
                    max: Vec2::new(hi, back),
                };
                if self.objects.iter().filter(|o| o.solid).any(|o| {
                    let (a, b) = o.bounds();
                    route.overlaps(a, b)
                }) {
                    continue;
                }
                let row = Row {
                    min: Vec2::new(lo, front),
                    max: Vec2::new(hi, route.min.y),
                    double_sided,
                    product: order.product(course * 2 + side),
                };
                let before = self.assembly_diagnostics.programs[index].requested_capacity;
                let mut planned = 0;
                // A column, door approach or unhelpful table/leg sample can lose
                // an otherwise viable bank. Retry the complete transaction, not
                // just a chair after leaving its incompatible table in place.
                let fitted = (0..3).any(|_| {
                    self.assembly_diagnostics.programs[index].requested_capacity = before;
                    let fitted = self.furnish_table_row(index, &row, r);
                    planned = planned
                        .max(self.assembly_diagnostics.programs[index].requested_capacity - before);
                    fitted
                });
                self.assembly_diagnostics.programs[index].requested_capacity = before + planned;
                if fitted {
                    self.assembly_diagnostics.programs[index]
                        .reservations
                        .push(route);
                    tables += 1;
                }
            }
        }
        tables
    }

    fn furnish_table_row(&mut self, index: usize, row: &Row, r: &mut ChaCha8Rng) -> bool {
        let parameters = self.assembly_diagnostics.programs[index].parameters.clone();
        let policy = &parameters.table_rows;
        let collection = variation::Collection::sample(self.seed, index);
        let span = (row.max.x - row.min.x - 0.22)
            * (policy.length_fraction + r.random_range(-0.04..0.04)).min(1.);
        let seats = (span / parameters.seat_pitch_metres).floor() as usize;
        if seats < 2 {
            return false;
        }
        let depth = if row.double_sided {
            policy.dining_depth_metres
        } else {
            policy.lecture_depth_metres
        };
        let mut size = collection.table(span, r);
        size.x = span;
        size.z = depth;
        let yaw = r.random_range(-0.045_f32..0.045);
        let extent = (yaw.cos().abs() * depth + yaw.sin().abs() * span) * 0.5;
        let pos = Vec3::new(
            (row.min.x + row.max.x) * 0.5 + r.random_range(-0.06..0.06),
            0.,
            if row.double_sided {
                (row.min.y + row.max.y) * 0.5
            } else {
                row.min.y + extent + r.random_range(0.10..0.17)
            },
        );
        let start = self.objects.len();
        let kind = if row.double_sided {
            ObjectKind::Table
        } else {
            ObjectKind::Desk
        };
        let Some(id) = self.public_fit_with(
            index,
            FloorProposal {
                kind,
                position: pos,
                size,
                yaw,
            },
            r,
            |o| {
                o.variant = SHARED_ROW_VARIANT;
                o.finish_seed = Some(row.product);
                row.contains(o)
            },
        ) else {
            return false;
        };
        let table = self.objects[id].clone();
        let mut sides = [1., -1.];
        if row.double_sided {
            sides.shuffle(r);
        }
        let sides = &sides[..if row.double_sided { 2 } else { 1 }];
        self.assembly_diagnostics.programs[index].requested_capacity += seats * sides.len();
        let regular = parameters.lecture_seating.regular;
        // A dining row needs usable seating on both sides, not a compulsory pair
        // on each edge. Per-table occupancy and independent vacancies also allow
        // realistic unbalanced counts, while lecture rows keep their shared desks.
        let minimum = if row.double_sided { 1 } else { 2 };
        let occupancy = parameters.occupancy
            * if row.double_sided {
                r.random_range(0.40..1.0)
            } else {
                1.
            };
        for &side in sides {
            let mut accepted = 0;
            let mut stations: Vec<_> = (0..seats).collect();
            stations.shuffle(r);
            for (attempt, seat) in stations.into_iter().enumerate() {
                // Preserve the minimum potential stations, then sample vacancies.
                // Random order avoids consistently filling one end of every top.
                if attempt >= minimum && !r.random_bool(occupancy as f64) {
                    continue;
                }
                let size = collection.chair(r);
                let turn = if regular {
                    0.
                } else {
                    r.random_range(-policy.chair_yaw_radians..policy.chair_yaw_radians)
                };
                let x = span * ((seat as f32 + 0.5) / seats as f32 - 0.5)
                    + if regular {
                        0.
                    } else {
                        r.random_range(-0.075..0.075)
                    };
                let chair_extent = (turn.cos().abs() * size.z + turn.sin().abs() * size.x) * 0.5;
                let offset = Vec3::new(
                    x,
                    0.,
                    side * (depth * 0.5 + chair_extent + r.random_range(0.09..0.17)),
                );
                let position = table.position + Quat::from_rotation_y(table.yaw) * offset;
                let facing = table.yaw + if side < 0. { PI } else { 0. } + turn;
                let variant = if regular {
                    11
                } else if r.random_bool(0.5) {
                    3
                } else {
                    4
                };
                if self
                    .public_fit_with(
                        index,
                        FloorProposal {
                            kind: ObjectKind::Chair,
                            position,
                            size,
                            yaw: facing,
                        },
                        r,
                        |o| {
                            o.variant = variant;
                            o.interaction_target = Some(id);
                            // Local recovery must not migrate a chair to the opposite side
                            // or put it on another tier. Shared clearance checks test legs.
                            let local = table
                                .transform()
                                .compute_affine()
                                .inverse()
                                .transform_point3(o.position);
                            row.contains(o)
                                && local.z * side > depth * 0.5
                                && (local.x - x).abs() < 0.18
                                && (o.position.y - table.position.y).abs() < 0.015
                        },
                    )
                    .is_some()
                {
                    accepted += 1;
                }
            }
            if accepted < minimum {
                self.objects.truncate(start);
                return false;
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_and_two_lot_installations_replay_without_an_aisle_side_rule() {
        let mut mixed = 0;
        let mut secondary = [0; 12];
        for seed in 0..256 {
            let order = RowOrder::sample(seed, 1);
            let replay = RowOrder::sample(seed, 1);
            assert_ne!(order.products, RowOrder::sample(seed, 2).products);
            let mut observed = std::collections::BTreeSet::new();
            for (bank, count) in secondary.iter_mut().enumerate() {
                let lot = order.product(bank);
                assert_eq!(lot, replay.product(bank));
                observed.insert(lot);
                *count += usize::from(lot == order.products[1]);
            }
            mixed += usize::from(observed.len() == 2);
        }
        assert!((65..155).contains(&mixed));
        assert!(
            secondary.iter().all(|n| (12..56).contains(n)),
            "{secondary:?}"
        );
    }
}
