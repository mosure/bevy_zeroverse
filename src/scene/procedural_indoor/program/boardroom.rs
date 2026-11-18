//! Table-first planning for single-room executive meetings. Distances are metres.
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BoardroomPlan {
    /// Width and length in the table's local X/Z frame.
    pub table_size_m: Vec2,
    /// Table edge to nominal wall, across and along the table respectively.
    pub table_to_wall_m: Vec2,
    /// Additional clearance on the entry wall, where the door approach is reserved.
    pub entry_extra_m: f32,
    pub long_axis_x: bool,
    pub seat_pitch_m: f32,
}

impl BoardroomPlan {
    const MIN_WIDTH: f32 = 1.38;
    const MIN_LENGTH: f32 = 3.65;
    const SEATED_DEPTH_M: f32 = 1.20;
    pub(crate) const ALIGNMENT_YAW: f32 = 0.032;
    pub(crate) const ALIGNMENT_OFFSET_M: f32 = 0.10;

    pub(crate) fn sample(seed: u64, layout: IndoorLayout) -> Option<Self> {
        if layout != IndoorLayout::Conference {
            return None;
        }
        let mut r = stream(seed, 0x424f_4152_4452_4f4d);
        if !r.random_bool(0.55) {
            return None;
        }
        let scale: f32 = r.random();
        let large_board = scale.powi(4);
        // Seating demand mostly lengthens the side rows. A steep, continuous
        // upper tail retains long modular tables for occasional flagship rooms.
        let length = Self::MIN_LENGTH + 4.15 * scale + 7.45 * scale.powi(9);
        // Depth varies within the same capacity. Even a long board can be
        // narrow enough to converse across; only some large boards grow wide.
        let width = r.random_range(Self::MIN_WIDTH..2.28)
            + 0.50 * scale
            + r.random_range(0.30..1.55) * scale.powi(7);
        let clearance = Vec2::new(r.random_range(1.54..1.83), r.random_range(1.65..1.98))
            - Vec2::splat(0.10 * scale);
        let table_to_wall_m = clearance.lerp(Vec2::new(1.38, 1.53), 0.90 * large_board);
        let entry_slack = r.random_range(0.02..0.30);
        let mut plan = Self {
            table_size_m: Vec2::new(width, length),
            // Large boards grow into their room; chair and entry clearances
            // remain metric rather than increasing in proportion to its area.
            table_to_wall_m,
            entry_extra_m: 0.,
            long_axis_x: r.random_bool(0.5),
            seat_pitch_m: r.random_range(0.88..1.15),
        };
        plan.entry_extra_m = plan.minimum_entry_extra() + entry_slack;
        Some(plan)
    }

    fn room_axes(self, local: Vec2) -> Vec2 {
        if self.long_axis_x {
            local.yx()
        } else {
            local
        }
    }

    fn minimum_entry_extra(self) -> f32 {
        // A doorway may land behind any chair. Reserve the actual approach in
        // addition to the seated back, including the table's alignment tolerance.
        super::super::architecture::doors::ENTRY_APPROACH_DEPTH_M
            + Self::SEATED_DEPTH_M
            + Self::ALIGNMENT_OFFSET_M
            + 0.5 * self.room_axes(self.table_size_m).x * Self::ALIGNMENT_YAW.sin()
            - self.room_axes(self.table_to_wall_m).y
    }

    pub(crate) fn span(self) -> Vec2 {
        self.room_axes(self.table_size_m + self.table_to_wall_m * 2.) + Vec2::Y * self.entry_extra_m
    }

    /// The standalone spatial-program API accepts an existing shell. Retain
    /// at least the seeded circulation widths if a human-scale board fits.
    pub(super) fn fit_existing_floor(mut self, size: Vec3) -> Option<Self> {
        if !size.is_finite() || size.y > 5.8 {
            return None;
        }
        if self.span().abs_diff_eq(size.xz(), 0.001) {
            return Some(self);
        }
        let entry_slack = self.entry_extra_m - self.minimum_entry_extra();
        self.long_axis_x = size.x > size.z;
        self.entry_extra_m = self.minimum_entry_extra() + entry_slack;
        let local = self.room_axes(size.xz() - Vec2::Y * self.entry_extra_m);
        let available = local - self.table_to_wall_m * 2.;
        if !(Self::MIN_WIDTH..=6.6).contains(&available.x)
            || !(Self::MIN_LENGTH..=15.25).contains(&available.y)
            || available.y < 1.6 * available.x
        {
            return None;
        }
        // A pre-existing shell limits the top. Surplus width belongs to the
        // circulation aisles, just as it does around a stock conference table.
        self.table_size_m = self.table_size_m.min(available);
        self.table_to_wall_m = (local - self.table_size_m) * 0.5;
        Some(self)
    }

    pub(crate) fn center(self) -> Vec2 {
        -Vec2::Y * (self.entry_extra_m * 0.5)
    }

    /// Keep the floor level and the architecture outside the table and seated backs.
    pub(crate) fn reserved_half(self) -> Vec2 {
        self.room_axes(self.table_size_m) * 0.5 + Vec2::splat(Self::SEATED_DEPTH_M)
    }

    pub(crate) fn facade_allowance(self) -> Vec2 {
        self.room_axes(self.table_to_wall_m) - Vec2::splat(Self::SEATED_DEPTH_M)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn existing_shell_keeps_metric_aisles_and_leaves_lofts_for_other_programs() {
        let plan = (0..64)
            .find_map(|seed| BoardroomPlan::sample(seed, IndoorLayout::Conference))
            .unwrap();
        let fitted = plan.fit_existing_floor(Vec3::new(12.9, 3.5, 7.)).unwrap();
        assert!(fitted.span().abs_diff_eq(Vec2::new(12.9, 7.), 0.001));
        assert!(fitted.table_size_m.cmple(plan.table_size_m).all());
        assert!(fitted
            .table_to_wall_m
            .cmpge(plan.table_to_wall_m - Vec2::splat(1e-5))
            .all());
        let span = plan.span();
        assert!(plan
            .fit_existing_floor(Vec3::new(span.x, 6.5, span.y))
            .is_none());
        assert!(plan.fit_existing_floor(Vec3::new(4., 3., 4.)).is_none());
    }

    #[test]
    fn executive_table_sizes_have_a_compact_middle_and_an_installable_long_tail() {
        let sizes: Vec<_> = (0..8192)
            .filter_map(|s| BoardroomPlan::sample(s, IndoorLayout::Conference))
            .map(|p| p.table_size_m)
            .collect();
        let n = sizes.len();
        let mut lengths: Vec<_> = sizes.iter().map(|s| s.y).collect();
        let mut widths: Vec<_> = sizes.iter().map(|s| s.x).collect();
        lengths.sort_by(f32::total_cmp);
        widths.sort_by(f32::total_cmp);
        assert!((5.3..6.3).contains(&lengths[n / 2]));
        assert!((1.8..2.4).contains(&widths[n / 2]));
        assert!(lengths[n / 10] < 4.4 && lengths[9 * n / 10] > 9.);
        let over_ten = sizes.iter().filter(|s| s.y > 10.).count();
        assert!((n / 16..n / 6).contains(&over_ten));
        assert!(lengths[n - 1] > 14. && widths[95 * n / 100] < 3.1);
        assert!(sizes.iter().any(|s| s.y > 10. && s.x < 2.6));
        assert!(sizes.iter().any(|s| s.y > 10. && s.x > 3.3));
    }
}
