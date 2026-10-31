use super::super::super::assemblies::SeatPath;
use super::*;
use rand::seq::SliceRandom;
#[cfg(test)]
mod tests;
impl IndoorManifest {
    pub(super) fn dining_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        let communal = p.activity == IndoorLayout::Cafeteria;
        // Service circulation is a first-class relationship. Reserve it before
        // seating; adding it afterward almost always loses to occupied furniture.
        self.serving_queue(index, r);
        let table_rows = communal
            && p.parameters.table_rows.cafeteria_rows
            && self.table_row_program(index, true, r) > 0;
        if !table_rows {
            self.dining_clusters(index, communal, r);
        }
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
            let required = if communal {
                ObjectKind::DrinkDispenser
            } else {
                ObjectKind::CoffeeMaker
            };
            self.dress_refreshment_counter(&counter, Some(required));
        }
    }

    fn dining_clusters(&mut self, index: usize, communal: bool, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        let center = (p.min + p.max) * 0.5;
        let pitch = r.random_range(2.55..3.15);
        let stagger = r.random_range(-0.45..0.45);
        let setback = r.random_range(3.15..3.65);
        let disorder = r.random_range(0.04..0.22);
        let rows = ((p.max.y - p.min.y - 3.2) / pitch).floor().max(1.) as usize;
        for row in 0..rows {
            let row_shift = r.random_range(-0.22..0.22);
            for side in [-1., 1.] {
                let x = if side < 0. {
                    (p.min.x + center.x) * 0.5 - 0.18
                } else {
                    (p.max.x + center.x) * 0.5 + 0.18
                } + row_shift
                    + r.random_range(-disorder..disorder);
                let z = p.min.y
                    + setback
                    + row as f32 * pitch
                    + side * stagger
                    + r.random_range(-disorder..disorder);
                self.dining_table(
                    index,
                    Vec3::new(x, 0., z),
                    if communal {
                        r.random_range(1.35..1.95)
                    } else {
                        r.random_range(0.78..1.2)
                    },
                    p.parameters.group_rotation_radians + r.random_range(-0.24..0.24),
                    r,
                );
            }
        }
        if !communal {
            let mut size = geometry::dimensions(ObjectKind::Banquette, r.random());
            size.x = size.x.min(p.max.y - p.min.y - 3.5);
            self.public_fit(
                index,
                ObjectKind::Banquette,
                Vec3::new(p.min.x + 0.36, 0., center.y + 1.1),
                size,
                FRAC_PI_2,
                r,
            );
        }
    }
    pub(super) fn dining_table(
        &mut self,
        index: usize,
        pos: Vec3,
        width: f32,
        yaw: f32,
        r: &mut ChaCha8Rng,
    ) {
        let collection = variation::Collection::sample(self.seed, index);
        let table_size = collection.table(width, r);
        let start = self.objects.len();
        let Some(id) = self.public_fit(index, ObjectKind::Table, pos, table_size, yaw, r) else {
            self.assembly_diagnostics.programs[index].requested_capacity += 2;
            return;
        };
        let table = self.objects[id].clone();
        // Capacity follows usable tabletop perimeter, not the longer offset
        // chair path. This keeps small cafe tops from becoming six-seat tables.
        let perimeter = SeatPath::new(&table, 0.).length;
        let occupancy = self.assembly_diagnostics.programs[index]
            .parameters
            .occupancy;
        let capacity =
            (perimeter / r.random_range(0.70..0.92) * occupancy * r.random_range(0.65..1.0))
                .round()
                .clamp(2., 8.) as usize;
        self.assembly_diagnostics.programs[index].requested_capacity += capacity;
        let bench_side = if self.assembly_diagnostics.programs[index].activity == IndoorLayout::Cafe
            && r.random_bool(0.48)
        {
            self.assembly_diagnostics.programs[index].requested_capacity += 1;
            let side = if r.random_bool(0.5) { -1. } else { 1. };
            let depth = r.random_range(0.63..0.76);
            let offset = Vec3::new(
                r.random_range(-0.10..0.10),
                0.,
                side * (table.size.z * 0.5 + depth * 0.5 + r.random_range(0.14..0.26)),
            );
            self.public_fit_with(
                index,
                FloorProposal {
                    kind: ObjectKind::Banquette,
                    position: table.transform().transform_point(offset),
                    size: Vec3::new(table.size.x.max(1.15), r.random_range(0.96..1.17), depth),
                    yaw: yaw + if side > 0. { PI } else { 0. },
                },
                r,
                |o| station_clear(&table, o, offset.xz()),
            )
            .map(|_| side)
        } else {
            None
        };
        let path = SeatPath::new(&table, r.random_range(0.46..0.57));
        let phase = r.random_range(0.0..1.0);
        let mut stations: Vec<_> = (0..capacity).collect();
        // Neither a world axis nor the outline's first edge gets first refusal
        // when a doorway, neighboring table or bench blocks some of the seats.
        stations.shuffle(r);
        for station in stations {
            let fraction = phase + (station as f32 + r.random_range(-0.14..0.14)) / capacity as f32;
            let (point, facing) = path.sample(fraction);
            if bench_side.is_some_and(|side| point.y * side > table.size.z * 0.25) {
                continue;
            }
            let size = collection.chair(r);
            let variant = if r.random_bool(0.5) { 3 } else { 4 };
            self.public_fit_with(
                index,
                FloorProposal {
                    kind: ObjectKind::Chair,
                    position: table
                        .transform()
                        .transform_point(Vec3::new(point.x, 0., point.y)),
                    size,
                    yaw: table.yaw + facing + r.random_range(-0.22..0.22),
                },
                r,
                |o| {
                    // Admit the final chair geometry and pairing, so leg/top
                    // clearance is checked before this station is accepted.
                    o.variant = variant;
                    o.interaction_target = Some(id);
                    station_clear(&table, o, point)
                },
            );
        }
        if self.objects.len() == start + 1 {
            self.objects.truncate(start);
        }
    }
}

/// Local recovery may move a seat slightly, but not across the tabletop or onto
/// another floor level. The shared placement gate checks all physical clearance.
fn station_clear(table: &IndoorObject, seat: &IndoorObject, station: Vec2) -> bool {
    let local = table
        .transform()
        .compute_affine()
        .inverse()
        .transform_point3(seat.position);
    local.xz().distance_squared(station) < 0.40_f32.powi(2)
        && local.xz().dot(station) > 0.
        && (table.position.y - seat.position.y).abs() < 0.015
}
