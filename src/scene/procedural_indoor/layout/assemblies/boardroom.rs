use super::*;
use crate::scene::procedural_indoor::program::boardroom::BoardroomPlan;
use crate::scene::procedural_indoor::{envelope::polygon, objects::tables};

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BoardroomSummary {
    pub table_id: usize,
    pub seat_count: usize,
    /// Nominal major and minor dimensions of the installed tabletop.
    pub table_length_m: f32,
    pub table_width_m: f32,
    pub tabletop_area_m2: f32,
    /// Table-aligned bounding rectangle of the tabletop and all installed seats.
    pub seating_envelope_area_m2: f32,
    pub room_floor_area_m2: f32,
}

impl IndoorManifest {
    /// Measure the installed furniture after chair placement and tucking.
    pub fn executive_boardroom(&self) -> Option<BoardroomSummary> {
        self.program.as_ref()?.executive_boardroom?;
        let g = self
            .assembly_diagnostics
            .accepted
            .iter()
            .find(|g| g.role == AssemblyRole::Meeting)?;
        let table = self.objects.get(g.surface)?;
        let outline = tables::outline(table);
        let mut lo = Vec2::splat(f32::INFINITY);
        let mut hi = -lo;
        for &p in &outline {
            lo = lo.min(p);
            hi = hi.max(p);
        }
        let inverse = table.transform().compute_affine().inverse();
        for &seat in &g.seats {
            let o = self.objects.get(seat)?;
            let tf = inverse * o.transform().compute_affine();
            for x in [-0.5, 0.5] {
                for z in [-0.5, 0.5] {
                    let p = tf.transform_point3(o.size * Vec3::new(x, 0., z)).xz();
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        Some(BoardroomSummary {
            table_id: table.id,
            seat_count: g.seats.len(),
            table_length_m: table.size.x.max(table.size.z),
            table_width_m: table.size.x.min(table.size.z),
            tabletop_area_m2: polygon::area(&outline),
            seating_envelope_area_m2: (hi - lo).element_product(),
            room_floor_area_m2: polygon::area(&self.envelope.as_ref()?.footprint),
        })
    }

    pub(in super::super) fn furnish_boardroom(&mut self, plan: BoardroomPlan) {
        let mut rng = stream(self.seed, 0x4558_4543_5345_4154);
        let base_yaw = if plan.long_axis_x {
            std::f32::consts::FRAC_PI_2
        } else {
            0.
        };
        let family = [0, 1, 2, 4, 5, 9][rng.random_range(0..6)];
        let pitch = plan.seat_pitch_m * (1.15 - 0.15 * self.density);
        let rows = (plan.table_size_m.y / pitch).floor().max(3.) as usize;
        let head_seats = if plan.table_size_m.x > 3.6 && rng.random_bool(0.70) {
            2
        } else {
            1
        };
        let checkpoint = self.objects.len();
        self.assembly_diagnostics.proposed_meetings += 1;
        for attempt in 0..8 {
            // The architecture already reserves this span. Resample only the fine
            // alignment, outline and seat positions if a complete ring cannot fit.
            let yaw = base_yaw
                + rng.random_range(-BoardroomPlan::ALIGNMENT_YAW..BoardroomPlan::ALIGNMENT_YAW);
            let offset = BoardroomPlan::ALIGNMENT_OFFSET_M;
            let origin = plan.center()
                + Vec2::new(
                    rng.random_range(-offset..offset),
                    rng.random_range(-offset..offset),
                );
            let size = Vec3::new(
                plan.table_size_m.x,
                rng.random_range(0.72..0.78),
                plan.table_size_m.y,
            );
            let mut table = self.floor_candidate(
                ObjectKind::Table,
                Vec3::new(origin.x, 0., origin.y),
                size,
                yaw,
                &mut rng,
            );
            table.variant = tables::BOARDROOM_VARIANT;
            if !self.placement_clear(&table, 0.06) {
                self.assembly_diagnostics.rejected_surfaces += 1;
                self.rejected_placements += 1;
                continue;
            }
            let surface = table.id;
            let tf = table.transform();
            let path = SeatPath::new(&table, rng.random_range(0.49..0.60));
            self.objects.push(table);
            let mut counts = [0; 4];
            let mut capacity = (rows + head_seats) * 2;
            for (side, count) in counts.iter_mut().enumerate() {
                let axis = side / 2;
                let sign = if side % 2 == 0 { -1. } else { 1. };
                let n = if axis == 0 { rows } else { head_seats };
                let vacancy = (axis == 0
                    && n >= 5
                    && rng.random_bool(0.10 + 0.25 * f64::from(1. - self.density)))
                .then(|| rng.random_range(0..n));
                for seat in 0..n {
                    if vacancy == Some(seat) {
                        capacity -= 1;
                        continue;
                    }
                    let along = if axis == 0 {
                        ((seat as f32 + rng.random_range(0.39..0.61)) / n as f32 - 0.5)
                            * size.z
                            * 0.94
                    } else if n == 1 {
                        rng.random_range(-0.15..0.15) * size.x
                    } else {
                        (seat as f32 - 0.5) * size.x * rng.random_range(0.35..0.48)
                    };
                    if let Some((p, facing)) = path.on_side(axis, sign, along) {
                        let before = self.objects.len();
                        self.chair_in_family_at_surface(
                            tf.transform_point(Vec3::new(p.x, 0., p.y)),
                            yaw + facing,
                            Some(family),
                            Some(surface),
                            &mut rng,
                        );
                        *count += self.objects.len() - before;
                    }
                }
            }
            let seats = checkpoint + 1..self.objects.len();
            if counts[0] >= 3
                && counts[1] >= 3
                && counts[2] >= 1
                && counts[3] >= 1
                && seats.len() * 4 >= capacity * 3
            {
                self.assembly_diagnostics.accepted.push(FunctionalAssembly {
                    role: AssemblyRole::Meeting,
                    surface,
                    seats: seats.collect(),
                    requested_capacity: capacity,
                    attempts: attempt + 1,
                });
                return;
            }
            self.assembly_diagnostics.rejected_seating += 1;
            self.objects.truncate(checkpoint);
        }
        self.assembly_diagnostics.rejected_meetings += 1;
    }
}
