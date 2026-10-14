use super::*;
impl IndoorManifest {
    pub(super) fn teaching_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        let center = (p.min + p.max) * 0.5;
        let lecture = p.activity == IndoorLayout::LectureHall;
        let row_pitch = if lecture {
            p.parameters.row_pitch_metres
        } else {
            p.parameters.row_pitch_metres + 1.0
        };
        let rows = ((p.max.y - p.min.y - 4.0) / row_pitch)
            .floor()
            .clamp(1., 7.) as usize;
        let cols = (((p.max.x - p.min.x) - p.parameters.aisle_metres - 0.8)
            / (if lecture {
                p.parameters.seat_pitch_metres
            } else {
                p.parameters.seat_pitch_metres + 0.60
            })
            * 0.5)
            .floor()
            .max(1.) as usize;
        self.assembly_diagnostics.programs[index].requested_capacity = rows * cols * 2;
        for row in 0..rows {
            for side in [-1., 1.] {
                for col in 0..cols {
                    if row + col > 0 && !r.random_bool(p.parameters.occupancy as f64) {
                        continue;
                    }
                    let x = center.x
                        + side
                            * (p.parameters.aisle_metres * 0.5
                                + 0.53
                                + col as f32
                                    * if lecture {
                                        p.parameters.seat_pitch_metres
                                    } else {
                                        p.parameters.seat_pitch_metres + 0.60
                                    });
                    let curve = p.parameters.row_curvature_metres
                        * ((x - center.x) / ((p.max.x - p.min.x) * 0.5)).powi(2);
                    let z = p.min.y + 3.0 + (row as f32 + 0.52) * row_pitch + curve;
                    let yaw = ((x - center.x) * 0.045).clamp(-0.22, 0.22)
                        + if lecture {
                            0.
                        } else {
                            (if (row + col) % 2 == 0 { 1. } else { -1. })
                                * p.parameters.collaborative_fraction
                                * 0.72
                        };
                    let mut candidates = vec![Vec3::new(x, 0., z)];
                    if !lecture && p.parameters.collaborative_fraction > 0.60 {
                        candidates[0].z += if col % 2 == 0 { 0.12 } else { -0.12 };
                    }
                    for pos in candidates {
                        let before = self.objects.len();
                        if let Some(id) = self.public_fit(
                            index,
                            ObjectKind::Chair,
                            pos,
                            Vec3::new(0.72, 1.04, 0.72),
                            yaw,
                            r,
                        ) {
                            self.objects[id].variant = if lecture { 11 } else { 3 };
                            if !lecture {
                                let table_pos = self.objects[id].position
                                    + Quat::from_rotation_y(yaw) * Vec3::new(0., 0., -0.78);
                                if let Some(table) = self.public_fit(
                                    index,
                                    ObjectKind::Desk,
                                    table_pos,
                                    Vec3::new(0.98, 0.74, 0.55),
                                    yaw,
                                    r,
                                ) {
                                    self.objects[id].interaction_target = Some(table);
                                } else {
                                    self.objects.truncate(before);
                                }
                            }
                        }
                    }
                }
            }
        }
        if !lecture {
            let size = geometry::dimensions(ObjectKind::Locker, r.random());
            self.public_fit(
                index,
                ObjectKind::Locker,
                Vec3::new(p.max.x - size.x * 0.5, 0., p.min.y + size.z * 0.5),
                size,
                0.,
                r,
            );
        }
    }
}
