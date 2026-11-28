use super::*;
impl IndoorManifest {
    pub(super) fn teaching_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        if self.assembly_diagnostics.programs[index].activity == IndoorLayout::LectureHall {
            self.lecture_program(index, r);
            return;
        }
        let tablet_fraction =
            stream(self.seed, 0x5441424c4554 + index as u64).random_range(0.15..0.55);
        let collection = variation::Collection::sample(self.seed, index);
        let groups = self.classroom_groups(index, &collection);
        let p = &self.assembly_diagnostics.programs[index];
        let center = (p.min + p.max) * 0.5;
        let (min, max, parameters) = (p.min, p.max, p.parameters.clone());
        let aisle_setback = r.random_range(0.48..0.61);
        let disorder = r.random_range(0.02..0.14);
        let row_pitch = parameters.row_pitch_metres + 1.;
        let seat_pitch = parameters.seat_pitch_metres + 0.60;
        let rows = ((max.y - min.y - 4.0) / row_pitch).floor().clamp(1., 7.) as usize;
        let cols = (((max.x - min.x) - parameters.aisle_metres - 0.8) / seat_pitch * 0.5)
            .floor()
            .max(1.) as usize;
        self.assembly_diagnostics.programs[index].requested_capacity = groups
            .iter()
            .map(|&capacity| if capacity == 0 { rows * cols } else { capacity })
            .sum();
        for row in 0..rows {
            for (bank, side) in [-1., 1.].into_iter().enumerate() {
                if groups[bank] > 0 {
                    continue;
                }
                let cluster_turn = r.random_range(-0.72..0.72) * parameters.collaborative_fraction;
                let row_shift = r.random_range(-0.18..0.18);
                for col in 0..cols {
                    if row + col > 0 && !r.random_bool(parameters.occupancy as f64) {
                        continue;
                    }
                    let x = center.x
                        + side
                            * (parameters.aisle_metres * 0.5
                                + aisle_setback
                                + col as f32 * seat_pitch)
                        + r.random_range(-disorder..disorder);
                    let curve = parameters.row_curvature_metres
                        * ((x - center.x) / ((max.x - min.x) * 0.5)).powi(2);
                    let z = min.y
                        + 3.0
                        + (row as f32 + 0.52) * row_pitch
                        + curve
                        + row_shift
                        + r.random_range(-disorder..disorder);
                    let yaw = ((x - center.x) * 0.045).clamp(-0.22, 0.22)
                        + (cluster_turn + r.random_range(-0.12..0.12));
                    let pos = Vec3::new(x, 0., z);
                    let tablet = r.random_bool(tablet_fraction);
                    let chair_size = collection.chair(r);
                    let before = self.objects.len();
                    if let Some(id) =
                        self.public_fit(index, ObjectKind::Chair, pos, chair_size, yaw, r)
                    {
                        self.objects[id].variant = if tablet { 10 } else { 3 };
                        if !tablet {
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
        let size = geometry::dimensions(ObjectKind::Locker, r.random());
        self.fit_locker_bank(index, size, index as u64, &PlacementStructure::new(), r);
    }
}
