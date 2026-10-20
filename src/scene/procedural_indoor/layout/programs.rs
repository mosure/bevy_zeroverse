//! Compose sized furniture groups into the leaves of the spatial program.
use super::*;
mod recovery;
#[cfg(test)]
mod tests;
impl IndoorManifest {
    pub(super) fn furnish_program(&mut self, rng: &mut ChaCha8Rng) {
        let zones = self.program.as_ref().unwrap().zones.clone();
        let mut dimension_rng = stream(self.seed, 0x5245_5041_4952_5349);
        for (zone_index, zone) in zones.into_iter().enumerate() {
            // Placement refinements must not consume the activity/composition
            // stream and accidentally change the next zone's function.
            // Dedicated namespace: 5810/5811 belong to wall composition.
            let mut transform_rng = stream(self.seed, 0x5a4f_4e45_504f_5345 + zone_index as u64);
            let extent = zone.max - zone.min;
            if zone.activity.is_kitchen() {
                self.furnish_kitchen(&zone, zone_index);
                self.furnish_kitchen_dining(&zone, rng);
                continue;
            }
            let activity = zone
                .composition
                .as_ref()
                .map_or(zone.activity, |mix| mix.choose(rng));
            let field = zone.furnishing.as_ref();
            let center =
                (zone.min + zone.max) * 0.5 + field.map_or(Vec2::ZERO, |f| f.offset * extent);
            let tf = Transform::from_xyz(center.x, 0.0, center.y)
                .with_rotation(Quat::from_rotation_y(zone.orientation));
            // Shrink the local envelope for the continuously sampled rotation,
            // rather than placing axis-aligned layouts through partition walls.
            let (s, c) = zone.orientation.sin_cos();
            let quarter = c.abs() < s.abs();
            let usable = (extent - Vec2::splat(0.85)) / (s.abs() + c.abs());
            let (w, d) = if quarter {
                (usable.y, usable.x)
            } else {
                (usable.x, usable.y)
            };
            let point = |x, y, z| tf.transform_point(Vec3::new(x, y, z));
            let add_chair = |scene: &mut Self, x, z, yaw, rng: &mut ChaCha8Rng| {
                scene.chair(point(x, 0.0, z), zone.orientation + yaw, rng);
            };
            if activity == IndoorLayout::Lounge && w > 2.7 && d > 3.0 {
                let sofa_width = (w * rng.random_range(0.58..0.88)).clamp(1.6, 3.4);
                let sofa_depth = if w > 3.4 && d > 4.0 && rng.random_bool(0.52) {
                    rng.random_range(1.45..1.95)
                } else {
                    rng.random_range(0.80..1.05)
                };
                let sofa_z = -d * 0.5 + sofa_depth * 0.5 + 0.08;
                self.add(
                    ObjectKind::Sofa,
                    point(0.0, 0.0, sofa_z),
                    Vec3::new(sofa_width, rng.random_range(0.78..1.06), sofa_depth),
                    zone.orientation + std::f32::consts::PI,
                    rng,
                );
                self.add(
                    ObjectKind::CoffeeTable,
                    point(
                        rng.random_range(-0.20..0.20),
                        0.0,
                        sofa_z + sofa_depth * 0.5 + 0.72,
                    ),
                    Vec3::new(
                        rng.random_range(0.85..1.55),
                        rng.random_range(0.34..0.46),
                        rng.random_range(0.5..0.75),
                    ),
                    zone.orientation + rng.random_range(-0.15..0.15),
                    rng,
                );
                for side in [-1.0, 1.0] {
                    add_chair(
                        self,
                        side * (w * 0.32).min(1.45),
                        sofa_z + sofa_depth * 0.5 + 1.55,
                        side * 0.42,
                        rng,
                    );
                }
            } else if activity == IndoorLayout::Conference && w > 3.0 && d > 3.0 {
                self.meeting_group(&zone, zone_index, Vec2::new(w, d), center);
            } else {
                let pitch_x = zone.desk_width + zone.aisle * rng.random_range(0.65..1.0);
                let pitch_z = zone.desk_depth + 1.25 + zone.aisle * 0.5;
                let cols = ((w + zone.aisle * 0.5) / pitch_x).floor().max(1.0) as usize;
                let rows = ((d + zone.aisle * 0.5) / pitch_z).floor().max(1.0) as usize;
                let stagger = rng.random_range(-0.15..0.15);
                let opposing = rng.random_bool(0.5);
                for row in 0..rows {
                    for col in 0..cols {
                        let (station, fan, occupancy) = field.map_or(
                            (
                                Vec2::new(
                                    (col as f32 - (cols - 1) as f32 * 0.5) * pitch_x
                                        + (row as f32 % 2.0 - 0.5) * stagger,
                                    (row as f32 - (rows - 1) as f32 * 0.5) * pitch_z - 0.45,
                                ),
                                0.0,
                                1.0,
                            ),
                            |f| f.station([col, row], [cols, rows], Vec2::new(pitch_x, pitch_z)),
                        );
                        if row + col > 0
                            && !rng.random_bool((zone.occupancy * occupancy).min(1.0) as f64)
                        {
                            continue;
                        }
                        let jitter = field.map_or(0.0, |f| f.jitter);
                        let x = station.x + rng.random_range(-1.0..1.0) * jitter;
                        let z = station.y + rng.random_range(-1.0..1.0) * jitter;
                        let opposing = field.map_or(opposing && col % 2 == 1, |f| {
                            rng.random_bool(f.opposing_probability as f64)
                        });
                        let turn = fan + if opposing { std::f32::consts::PI } else { 0.0 };
                        let depth = zone.desk_depth * rng.random_range(0.94..1.02);
                        self.workstation(
                            ObjectKind::Desk,
                            point(x, 0.0, z),
                            Vec3::new(
                                zone.desk_width * transform_rng.random_range(0.93..1.05),
                                rng.random_range(0.71..0.79),
                                depth,
                            ),
                            zone.orientation + turn,
                            rng,
                        );
                    }
                }
            }
            self.fill_functional_zone(&zone, zone_index, rng);
            // Sample perimeter service objects by free space, not fixed global
            // corners. Plants, storage and bins compete for valid placements.
            let service_count = ((extent.x + extent.y)
                * rng.random_range(0.16..0.36)
                * self.domain().map_or(1.0, |d| d.service_density))
                as usize;
            for _ in 0..service_count {
                let side = rng.random_range(0..4);
                let t = rng.random_range(0.14..0.86);
                let setback = transform_rng.random_range(0.48..0.92);
                let (x, z, yaw) = match side {
                    0 => (
                        zone.min.x + setback,
                        zone.min.y + t * extent.y,
                        -std::f32::consts::FRAC_PI_2,
                    ),
                    1 => (
                        zone.max.x - setback,
                        zone.min.y + t * extent.y,
                        std::f32::consts::FRAC_PI_2,
                    ),
                    2 => (zone.min.x + t * extent.x, zone.min.y + setback, 0.0),
                    _ => (
                        zone.min.x + t * extent.x,
                        zone.max.y - setback,
                        std::f32::consts::PI,
                    ),
                };
                let service = if rng.random_bool(
                    zone.composition
                        .as_ref()
                        .map_or(0.16, |mix| mix.storage_bias) as f64,
                ) {
                    3
                } else {
                    rng.random_range(0..8)
                };
                let (kind, size) = match service {
                    0 | 1 => (
                        ObjectKind::Plant,
                        Vec3::new(
                            rng.random_range(0.55..0.9),
                            rng.random_range(0.65..1.85),
                            rng.random_range(0.55..0.9),
                        ),
                    ),
                    2 => (
                        ObjectKind::Cabinet,
                        Vec3::new(
                            rng.random_range(0.75..1.55),
                            rng.random_range(0.72..1.3),
                            transform_rng.random_range(0.34..0.52),
                        ),
                    ),
                    3 => (
                        ObjectKind::Bookcase,
                        Vec3::new(
                            rng.random_range(0.70..1.4),
                            rng.random_range(1.35..1.95),
                            transform_rng.random_range(0.26..0.42),
                        ),
                    ),
                    4 => (ObjectKind::CoatRack, {
                        let base = transform_rng.random_range(0.48..0.76);
                        Vec3::new(base, rng.random_range(1.45..1.95), base)
                    }),
                    5 => (
                        ObjectKind::Bag,
                        Vec3::new(
                            rng.random_range(0.32..0.53),
                            rng.random_range(0.28..0.46),
                            transform_rng.random_range(0.16..0.29),
                        ),
                    ),
                    6 => (
                        ObjectKind::StorageBox,
                        Vec3::new(
                            rng.random_range(0.3..0.6),
                            rng.random_range(0.2..0.45),
                            transform_rng.random_range(0.26..0.45),
                        ),
                    ),
                    _ => {
                        let height = rng.random_range(0.4_f32..0.65);
                        let mut shape_rng = stream(self.seed ^ height.to_bits() as u64, 6330);
                        (
                            ObjectKind::TrashCan,
                            Vec3::new(
                                shape_rng.random_range(0.27..0.46),
                                height,
                                shape_rng.random_range(0.26..0.42),
                            ),
                        )
                    }
                };
                let yaw = yaw + transform_rng.random_range(-0.20..0.20);
                self.add(kind, Vec3::new(x, 0.0, z), size, yaw, rng);
            }
        }
        // A usable upper floor gets its own work area; base-floor proposals do
        // not get lifted onto the deck merely because their footprints overlap.
        if let Some(m) = self.envelope.as_ref().and_then(|e| e.mezzanine.clone()) {
            for _ in 0..48 {
                let count = self
                    .objects
                    .iter()
                    .filter(|o| {
                        !o.neighbor && o.solid && (o.position.y - m.deck.height).abs() < 0.01
                    })
                    .count();
                if count >= 4 {
                    break;
                }
                let center = (m.deck.min + m.deck.max) * 0.5;
                let extent = m.deck.max - m.deck.min;
                let p = center
                    + extent
                        * Vec2::new(rng.random_range(-0.25..0.25), rng.random_range(-0.25..0.25));
                self.workstation(
                    ObjectKind::Desk,
                    Vec3::new(p.x, m.deck.height, p.y),
                    Vec3::new(
                        dimension_rng.random_range(0.96..1.30),
                        dimension_rng.random_range(0.71..0.79),
                        dimension_rng.random_range(0.55..0.72),
                    ),
                    rng.random_range(0..4) as f32 * std::f32::consts::FRAC_PI_2
                        + dimension_rng.random_range(-0.24..0.24),
                    rng,
                );
            }
        }
        // Constraint rejection can remove a complete group near a portal. Repair
        // underfilled rooms with supported workstations before adding service pieces.
        for _ in 0..384 {
            if self
                .objects
                .iter()
                .filter(|o| o.solid && !o.neighbor)
                .count()
                >= self.minimum_main_objects()
                && self.objects.iter().any(|o| {
                    !o.neighbor
                        && (matches!(o.kind, ObjectKind::Desk | ObjectKind::Table)
                            || self.layout.is_kitchen() && o.kind == ObjectKind::KitchenCounter
                            || self.layout.prefers_soft_seating()
                                && o.kind == ObjectKind::CoffeeTable)
                })
            {
                break;
            }
            let x = rng.random_range(-0.40..0.40) * self.room_size.x;
            let z = rng.random_range(-0.40..0.40) * self.room_size.z;
            let yaw = rng.random_range(0..4) as f32 * std::f32::consts::FRAC_PI_2
                + rng.random_range(-0.12..0.12);
            self.workstation(
                if self.layout == IndoorLayout::Conference || self.layout.is_kitchen() {
                    ObjectKind::Table
                } else if self.layout.prefers_soft_seating() {
                    ObjectKind::CoffeeTable
                } else {
                    ObjectKind::Desk
                },
                Vec3::new(x, 0.0, z),
                Vec3::new(
                    dimension_rng.random_range(1.0..1.34),
                    if self.layout.prefers_soft_seating() {
                        dimension_rng.random_range(0.34..0.48)
                    } else {
                        dimension_rng.random_range(0.71..0.79)
                    },
                    dimension_rng.random_range(0.56..0.74),
                ),
                yaw,
                rng,
            );
        }
        self.complete_lounge_groups();
    }

    /// A workstation is transactional: failed seating must not leave another
    /// unusable desk that consumes the space needed to repair the room.
    fn workstation(
        &mut self,
        kind: ObjectKind,
        position: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) -> bool {
        self.assembly_diagnostics.proposed_workstations += 1;
        let before = self.objects.len();
        if self.add(kind, position, size, yaw, rng).is_none() {
            self.assembly_diagnostics.rejected_surfaces += 1;
            return false;
        }
        let rotation = Quat::from_rotation_y(yaw);
        let sideways = rng.random_range(-0.10..0.10);
        let mut spacing = stream(self.objects[before].seed, 0x5345_4154_4741_5053);
        let initial_gap = spacing.random_range(0.48..0.62);
        let step = spacing.random_range(0.11..0.17);
        for gap in [initial_gap, initial_gap + step, initial_gap + step * 2.] {
            self.chair(
                position + rotation * Vec3::new(sideways, 0.0, size.z * 0.5 + gap),
                yaw,
                rng,
            );
            if self.objects.len() > before + 1 {
                self.objects[before + 1].interaction_target = Some(before);
                self.assembly_diagnostics
                    .accepted
                    .push(assemblies::FunctionalAssembly {
                        role: assemblies::AssemblyRole::Workstation,
                        surface: before,
                        seats: vec![before + 1],
                        requested_capacity: 1,
                        attempts: 1,
                    });
                return true;
            }
        }
        self.assembly_diagnostics.rejected_seating += 1;
        self.objects.truncate(before);
        false
    }

    fn fill_functional_zone(
        &mut self,
        zone: &super::super::program::Zone,
        zone_index: usize,
        rng: &mut ChaCha8Rng,
    ) {
        let mut dimension_rng = stream(self.seed, 0x4649_4c4c_5349_5a45 + zone_index as u64);
        let extent = zone.max - zone.min;
        let area = extent.x * extent.y;
        let area_per_seat = if zone.activity == IndoorLayout::Lounge {
            14.0
        } else {
            9.0
        };
        let target = (area * (0.30 + 0.70 * self.density) / area_per_seat)
            .ceil()
            .clamp(1.0, 32.0) as usize;
        for _ in 0..128 {
            let seats: usize = self
                .objects
                .iter()
                .filter(|o| {
                    !o.neighbor
                        && o.position.x > zone.min.x
                        && o.position.x < zone.max.x
                        && o.position.z > zone.min.y
                        && o.position.z < zone.max.y
                })
                .map(|o| match o.kind {
                    ObjectKind::Chair => 1,
                    ObjectKind::Sofa => (o.size.x / 0.65).floor() as usize,
                    _ => 0,
                })
                .sum();
            if seats >= target {
                break;
            }
            let p = Vec3::new(
                rng.random_range(zone.min.x + 0.7..zone.max.x - 0.7),
                0.0,
                rng.random_range(zone.min.y + 0.7..zone.max.y - 0.7),
            );
            let yaw = zone.orientation + rng.random_range(-0.65..0.65);
            let lounge = zone
                .composition
                .as_ref()
                .map_or(zone.activity, |mix| mix.choose(rng))
                == IndoorLayout::Lounge;
            if lounge && rng.random_bool(0.55) {
                self.add(
                    ObjectKind::Sofa,
                    p,
                    Vec3::new(
                        rng.random_range(1.30..2.25),
                        rng.random_range(0.78..1.02),
                        rng.random_range(0.80..1.10),
                    ),
                    yaw,
                    rng,
                );
            }
            self.workstation(
                if lounge {
                    ObjectKind::CoffeeTable
                } else {
                    ObjectKind::Desk
                },
                p,
                Vec3::new(
                    rng.random_range(0.95..1.55),
                    if lounge {
                        dimension_rng.random_range(0.34..0.48)
                    } else {
                        dimension_rng.random_range(0.71..0.79)
                    },
                    rng.random_range(0.56..0.76),
                ),
                yaw,
                rng,
            );
        }
    }
}
