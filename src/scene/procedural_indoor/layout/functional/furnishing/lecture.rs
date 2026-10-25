//! Tier-aware seating banks. Loose chairs have their own spacings and poses;
//! linked banks keep common seat pitch. All candidates use the regular gate.
use super::*;
impl IndoorManifest {
    pub(super) fn lecture_program(&mut self, index: usize, r: &mut ChaCha8Rng) {
        let p = self.assembly_diagnostics.programs[index].clone();
        if p.parameters.table_rows.lecture_tables {
            self.table_row_program(index, false, r);
            return;
        }
        let center = (p.min + p.max) * 0.5;
        let policy = &p.parameters.lecture_seating;
        let collection = variation::Collection::sample(self.seed, index);
        let aisle_setback = r.random_range(0.48..0.61);
        let rows = p.parameters.lecture_rows(p.min.y + 3., p.max.y - 1.);
        let cols = (((p.max.x - p.min.x - p.parameters.aisle_metres - 0.8)
            / p.parameters.seat_pitch_metres)
            * 0.5)
            .floor()
            .max(1.) as usize;
        self.assembly_diagnostics.programs[index].requested_capacity = rows.len() * cols * 2;
        for (row, &[front, back]) in rows.iter().enumerate() {
            for side in [-1., 1.] {
                let mut bank = policy.bank(row, side);
                // Offset away from the aisle; an irregular bank cannot narrow it.
                let stagger = bank.shift().abs();
                let mut distance = 0.;
                for col in 0..cols {
                    if col > 0 {
                        distance += bank.increment(p.parameters.seat_pitch_metres);
                    }
                    let [dx, dz, dyaw] = bank.pose();
                    if row + col > 0 && !r.random_bool(p.parameters.occupancy as f64) {
                        continue;
                    }
                    let x = center.x
                        + side
                            * (p.parameters.aisle_metres * 0.5
                                + aisle_setback
                                + stagger
                                + distance)
                        + dx;
                    let curve = p.parameters.row_curvature_metres
                        * ((x - center.x) / ((p.max.x - p.min.x) * 0.5)).powi(2);
                    let yaw = ((x - center.x) / (front - p.min.y + 1.)).atan()
                        * policy.facing_strength
                        + dyaw;
                    let size = collection.chair(r);
                    let half_depth = (yaw.cos().abs() * size.z + yaw.sin().abs() * size.x) * 0.5;
                    let slack = ((back - front) * 0.5 - half_depth - 0.065).max(0.);
                    // Smoothly compress requested displacement within the usable
                    // tier instead of piling candidates on a clamped step edge.
                    let offset = if slack > 0. {
                        slack * ((curve + dz) / slack).tanh()
                    } else {
                        0.
                    };
                    let z = (front + back) * 0.5 + offset;
                    if let Some(id) =
                        self.public_fit(index, ObjectKind::Chair, Vec3::new(x, 0., z), size, yaw, r)
                    {
                        self.objects[id].variant = if bank.linked { 11 } else { 10 };
                    }
                }
            }
        }
    }
}
