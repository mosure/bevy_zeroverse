//! Shared tops with several seats per side. Pair relationships exist before
//! collision admission; a failed seating group leaves no orphan table/receipt.
use super::*;
use rand::seq::SliceRandom;

struct Row {
    min: Vec2,
    max: Vec2,
    double_sided: bool,
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
        for [front, back] in rows {
            for (lo, hi) in [
                (p.min.x, center.x - half_aisle),
                (center.x + half_aisle, p.max.x),
            ] {
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
                o.variant = super::super::super::super::objects::tables::SHARED_ROW_VARIANT;
                row.contains(o)
            },
        ) else {
            return false;
        };
        let table = self.objects[id].clone();
        let sides: &[f32] = if row.double_sided { &[-1., 1.] } else { &[1.] };
        self.assembly_diagnostics.programs[index].requested_capacity += seats * sides.len();
        let regular = parameters.lecture_seating.regular;
        for &side in sides {
            let mut accepted = 0;
            let mut stations: Vec<_> = (0..seats).collect();
            stations.shuffle(r);
            for (attempt, seat) in stations.into_iter().enumerate() {
                // Preserve two potential stations per side, then sample vacancies.
                // Random order avoids consistently filling one end of every top.
                if attempt > 1 && !r.random_bool(parameters.occupancy as f64) {
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
            if accepted < 2 {
                self.objects.truncate(start);
                return false;
            }
        }
        true
    }
}
