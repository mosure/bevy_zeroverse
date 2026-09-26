//! Compose sized furniture groups into the leaves of the spatial program.
use super::*;
impl IndoorManifest {
    pub(super) fn furnish_program(&mut self, rng: &mut ChaCha8Rng) {
        let zones = self.program.as_ref().unwrap().zones.clone();
        for zone in zones {
            let extent = zone.max - zone.min;
            let center = (zone.min + zone.max) * 0.5;
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
            if zone.activity == IndoorLayout::Lounge && w > 2.7 && d > 3.0 {
                let sofa_width = (w * rng.random_range(0.58..0.88)).clamp(1.6, 3.4);
                let sofa_z = -d * 0.5 + 0.55;
                self.add(
                    ObjectKind::Sofa,
                    point(0.0, 0.0, sofa_z),
                    Vec3::new(sofa_width, rng.random_range(0.75..1.05), 0.90),
                    zone.orientation + std::f32::consts::PI,
                    rng,
                );
                self.add(
                    ObjectKind::CoffeeTable,
                    point(rng.random_range(-0.20..0.20), 0.0, sofa_z + 1.3),
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
                        sofa_z + 2.25,
                        side * 0.42,
                        rng,
                    );
                }
            } else if zone.activity == IndoorLayout::Conference && w > 3.15 && d > 3.25 {
                let width = rng.random_range(1.05..1.55f32).min(w - 2.0);
                let length = (d - 2.1).min(rng.random_range(2.0..5.8));
                let offset = rng.random_range(-0.08..0.08);
                if self
                    .add(
                        ObjectKind::Table,
                        point(offset, 0.0, 0.0),
                        Vec3::new(width, rng.random_range(0.72..0.78), length),
                        zone.orientation,
                        rng,
                    )
                    .is_some()
                {
                    let rows = ((length - 0.35) / zone.seat_pitch).floor().max(1.0) as usize;
                    for side in [-1.0, 1.0] {
                        for i in 0..rows {
                            if rng.random_bool(zone.occupancy as f64) {
                                let z = (i as f32 - (rows - 1) as f32 * 0.5) * zone.seat_pitch;
                                add_chair(
                                    self,
                                    offset + side * (width * 0.5 + rng.random_range(0.48..0.64)),
                                    z,
                                    side * std::f32::consts::FRAC_PI_2,
                                    rng,
                                );
                            }
                        }
                    }
                    for side in [-1.0, 1.0] {
                        add_chair(
                            self,
                            offset,
                            side * (length * 0.5 + 0.59),
                            if side > 0.0 {
                                0.0
                            } else {
                                std::f32::consts::PI
                            },
                            rng,
                        );
                    }
                }
            } else {
                let pitch_x = zone.desk_width + zone.aisle * rng.random_range(0.65..1.0);
                let pitch_z = zone.desk_depth + 1.25 + zone.aisle * 0.5;
                let cols = ((w + zone.aisle * 0.5) / pitch_x).floor().max(1.0) as usize;
                let rows = ((d + zone.aisle * 0.5) / pitch_z).floor().max(1.0) as usize;
                let stagger = rng.random_range(-0.15..0.15);
                let opposing = rng.random_bool(0.5);
                for row in 0..rows {
                    for col in 0..cols {
                        if row + col > 0 && !rng.random_bool(zone.occupancy as f64) {
                            continue;
                        }
                        let x = (col as f32 - (cols - 1) as f32 * 0.5) * pitch_x
                            + (row as f32 % 2.0 - 0.5) * stagger;
                        let z = (row as f32 - (rows - 1) as f32 * 0.5) * pitch_z - 0.45;
                        let turn = if opposing && col % 2 == 1 {
                            std::f32::consts::PI
                        } else {
                            0.0
                        };
                        let depth = zone.desk_depth * rng.random_range(0.94..1.02);
                        if self
                            .add(
                                ObjectKind::Desk,
                                point(x, 0.0, z),
                                Vec3::new(zone.desk_width, rng.random_range(0.71..0.78), depth),
                                zone.orientation + turn,
                                rng,
                            )
                            .is_some()
                        {
                            add_chair(
                                self,
                                x + rng.random_range(-0.12..0.12),
                                z + turn.cos() * (depth * 0.5 + rng.random_range(0.48..0.62)),
                                turn,
                                rng,
                            );
                        }
                    }
                }
            }
            // Sample perimeter service objects by free space, not fixed global
            // corners. Plants, storage and bins compete for valid placements.
            let service_count = ((extent.x + extent.y)
                * rng.random_range(0.16..0.36)
                * self.domain().map_or(1.0, |d| d.service_density))
                as usize;
            for _ in 0..service_count {
                let side = rng.random_range(0..4);
                let t = rng.random_range(0.14..0.86);
                let (x, z, yaw) = match side {
                    0 => (
                        zone.min.x + 0.65,
                        zone.min.y + t * extent.y,
                        -std::f32::consts::FRAC_PI_2,
                    ),
                    1 => (
                        zone.max.x - 0.65,
                        zone.min.y + t * extent.y,
                        std::f32::consts::FRAC_PI_2,
                    ),
                    2 => (zone.min.x + t * extent.x, zone.min.y + 0.65, 0.0),
                    _ => (
                        zone.min.x + t * extent.x,
                        zone.max.y - 0.65,
                        std::f32::consts::PI,
                    ),
                };
                let (kind, size) = match rng.random_range(0..8) {
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
                            0.42,
                        ),
                    ),
                    3 => (
                        ObjectKind::Bookcase,
                        Vec3::new(
                            rng.random_range(0.70..1.4),
                            rng.random_range(1.35..1.95),
                            0.34,
                        ),
                    ),
                    4 => (
                        ObjectKind::CoatRack,
                        Vec3::new(0.65, rng.random_range(1.45..1.95), 0.65),
                    ),
                    5 => (
                        ObjectKind::Bag,
                        Vec3::new(
                            rng.random_range(0.32..0.53),
                            rng.random_range(0.28..0.46),
                            0.22,
                        ),
                    ),
                    6 => (
                        ObjectKind::StorageBox,
                        Vec3::new(
                            rng.random_range(0.3..0.6),
                            rng.random_range(0.2..0.45),
                            0.38,
                        ),
                    ),
                    _ => (
                        ObjectKind::TrashCan,
                        Vec3::new(0.35, rng.random_range(0.4..0.65), 0.35),
                    ),
                };
                self.add(kind, Vec3::new(x, 0.0, z), size, yaw, rng);
            }
        }
        // Constraint rejection can remove a complete group near a portal. Repair
        // underfilled rooms with supported workstations before adding service pieces.
        for _ in 0..96 {
            if self
                .objects
                .iter()
                .filter(|o| o.solid && !o.neighbor)
                .count()
                >= self.minimum_main_objects()
                && self.objects.iter().any(|o| {
                    !o.neighbor
                        && (matches!(o.kind, ObjectKind::Desk | ObjectKind::Table)
                            || self.layout == IndoorLayout::Lounge
                                && o.kind == ObjectKind::CoffeeTable)
                })
            {
                break;
            }
            let x = rng.random_range(-0.35..0.35) * self.room_size.x;
            let z = rng.random_range(-0.35..0.35) * self.room_size.z;
            let yaw = rng.random_range(0..4) as f32 * std::f32::consts::FRAC_PI_2;
            if self
                .add(
                    if self.layout == IndoorLayout::Conference {
                        ObjectKind::Table
                    } else if self.layout == IndoorLayout::Lounge {
                        ObjectKind::CoffeeTable
                    } else {
                        ObjectKind::Desk
                    },
                    Vec3::new(x, 0.0, z),
                    Vec3::new(
                        1.15,
                        if self.layout == IndoorLayout::Lounge {
                            0.42
                        } else {
                            0.74
                        },
                        0.64,
                    ),
                    yaw,
                    rng,
                )
                .is_some()
            {
                self.chair(
                    Vec3::new(x, 0.0, z) + Quat::from_rotation_y(yaw) * Vec3::Z * 0.91,
                    yaw,
                    rng,
                );
            }
        }
    }
}
