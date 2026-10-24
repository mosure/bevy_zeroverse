//! Repair an empty office reconstruction room after ordinary furnishing.
//! Furniture in another partition must not hide a completely rejected work area.
use super::*;
use rand::seq::SliceRandom;

impl IndoorManifest {
    pub(super) fn complete_empty_primary_work_area(&mut self) {
        if !IndoorLayout::OFFICE_PROFILES.contains(&self.layout) || self.program.is_none() {
            return;
        }
        let (lo, hi) = self.primary_room_bounds();
        if self.objects.iter().any(|o| {
            !o.neighbor
                && o.solid
                && o.position.xz().cmpge(lo).all()
                && o.position.xz().cmple(hi).all()
        }) {
            return;
        }
        let Some(zone) = self
            .program
            .as_ref()
            .and_then(|p| p.zones.iter().find(|z| z.min == lo && z.max == hi))
        else {
            return;
        };
        if !IndoorLayout::OFFICE_PROFILES.contains(&zone.activity) {
            return;
        }
        let lounge = zone.activity.prefers_soft_seating();
        let kind = if lounge {
            ObjectKind::CoffeeTable
        } else if zone.activity == IndoorLayout::Conference {
            ObjectKind::Table
        } else {
            ObjectKind::Desk
        };
        let yaw = zone.orientation;
        let inset = Vec2::splat(0.7);
        if (lo + inset).cmpge(hi - inset).any() {
            return;
        }
        let rejected = self.rejected_placements;
        let diagnostics = self.assembly_diagnostics.clone();
        let mut rng = stream(self.seed, 0x5052_494d_574f_524b);
        let mut cells: [_; 36] = std::array::from_fn(|i| i);
        cells.shuffle(&mut rng);
        // Normal proposals share the zone's orientation. A step or doorway can
        // block every such desk/chair pair. Try all four facing directions over
        // a jittered spatial grid; retain ordinary human-scale dimensions and all
        // existing floor, access, furniture and ceiling checks.
        for turn in 0..4 {
            for &cell in &cells {
                let fraction = Vec2::new(
                    (cell % 6) as f32 + rng.random_range(0.2..0.8),
                    (cell / 6) as f32 + rng.random_range(0.2..0.8),
                ) / 6.;
                let point = lo + inset + (hi - lo - inset * 2.) * fraction;
                let size = Vec3::new(
                    rng.random_range(0.96..1.32),
                    if lounge {
                        rng.random_range(0.38..0.48)
                    } else {
                        rng.random_range(0.71..0.79)
                    },
                    rng.random_range(0.54..0.70),
                );
                let before = self.objects.len();
                let accepted = self.assembly_diagnostics.accepted.len();
                if self.workstation(
                    kind,
                    Vec3::new(point.x, 0., point.y),
                    size,
                    yaw + turn as f32 * std::f32::consts::FRAC_PI_2 + rng.random_range(-0.24..0.24),
                    &mut rng,
                ) {
                    if self.objects[before..].iter().all(|o| {
                        let (a, b) = o.bounds();
                        a.xz().cmpge(lo).all()
                            && b.xz().cmple(hi).all()
                            && (o.position.y - self.objects[before].position.y).abs() < 1e-4
                    }) {
                        return;
                    }
                    self.objects.truncate(before);
                    self.assembly_diagnostics.accepted.truncate(accepted);
                }
            }
        }
        // A genuinely unusable region remains a bounded, idempotent failure.
        self.rejected_placements = rejected;
        self.assembly_diagnostics = diagnostics;
    }
}
