use super::*;
impl IndoorManifest {
    pub(super) fn teaching_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        let center = (p.min + p.max) * 0.5;
        let lecture = p.activity == IndoorLayout::LectureHall;
        if lecture {
            self.lecture_program(index, r);
            return;
        }
        let tablet_fraction =
            stream(self.seed, 0x5441424c4554 + index as u64).random_range(0.15..0.55);
        let collection = variation::Collection::sample(self.seed, index);
        let aisle_setback = r.random_range(0.48..0.61);
        let disorder = r.random_range(0.02..0.14);
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
                // Collaborative clusters share an orientation, without encoding
                // row/column parity in every seat. Tiered rows retain sightlines.
                let cluster_turn = if lecture {
                    0.
                } else {
                    r.random_range(-0.72..0.72) * p.parameters.collaborative_fraction
                };
                let row_shift = if lecture {
                    0.
                } else {
                    r.random_range(-0.18..0.18)
                };
                for col in 0..cols {
                    if row + col > 0 && !r.random_bool(p.parameters.occupancy as f64) {
                        continue;
                    }
                    let x = center.x
                        + side
                            * (p.parameters.aisle_metres * 0.5
                                + aisle_setback
                                + col as f32
                                    * if lecture {
                                        p.parameters.seat_pitch_metres
                                    } else {
                                        p.parameters.seat_pitch_metres + 0.60
                                    })
                        + if lecture {
                            0.
                        } else {
                            r.random_range(-disorder..disorder)
                        };
                    let curve = p.parameters.row_curvature_metres
                        * ((x - center.x) / ((p.max.x - p.min.x) * 0.5)).powi(2);
                    let z = p.min.y
                        + 3.0
                        + (row as f32 + 0.52) * row_pitch
                        + curve
                        + row_shift
                        + if lecture {
                            0.
                        } else {
                            r.random_range(-disorder..disorder)
                        };
                    let yaw = ((x - center.x) * 0.045).clamp(-0.22, 0.22)
                        + if lecture {
                            0.
                        } else {
                            cluster_turn + r.random_range(-0.12..0.12)
                        };
                    {
                        let pos = Vec3::new(x, 0., z);
                        let tablet = !lecture && r.random_bool(tablet_fraction);
                        let chair_size = collection.chair(r);
                        let before = self.objects.len();
                        if let Some(id) =
                            self.public_fit(index, ObjectKind::Chair, pos, chair_size, yaw, r)
                        {
                            self.objects[id].variant = if lecture {
                                11
                            } else if tablet {
                                10
                            } else {
                                3
                            };
                            if !lecture && !tablet {
                                let desk_size = collection.desk(r);
                                let table_pos = self.objects[id].position
                                    + Quat::from_rotation_y(yaw)
                                        * Vec3::new(
                                            r.random_range(-0.055..0.055),
                                            0.,
                                            -(chair_size.z * 0.5
                                                + desk_size.z * 0.5
                                                + r.random_range(0.12..0.22)),
                                        );
                                if let Some(table) = self.public_fit(
                                    index,
                                    ObjectKind::Desk,
                                    table_pos,
                                    desk_size,
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
