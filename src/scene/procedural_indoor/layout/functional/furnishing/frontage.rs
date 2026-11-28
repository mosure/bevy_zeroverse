//! Clear working fronts connect floor equipment and shop fixtures to a common cross-aisle.
use super::*;

pub(super) const APPAREL_BROWSING_DEPTH: f32 = 0.88;

pub(super) fn access_depth(kind: ObjectKind) -> Option<f32> {
    match kind {
        ObjectKind::WashingMachine | ObjectKind::TumbleDryer => Some(1.05),
        ObjectKind::LaundryFoldingTable => Some(0.76),
        ObjectKind::GroceryDisplay => Some(0.96),
        ObjectKind::ApparelDisplay => Some(APPAREL_BROWSING_DEPTH),
        ObjectKind::CheckoutCounter => Some(1.04),
        ObjectKind::FittingRoom => Some(super::fitting::APPROACH_DEPTH),
        _ => None,
    }
}

/// Move a proposed straight aisle just far enough to clear structural
/// obstacles below head height. The nearest legal center to the original draw
/// preserves its continuous distribution and room-specific left/right bias.
pub(super) fn aisle_coordinate(
    scene: &IndoorManifest,
    min: Vec2,
    max: Vec2,
    axis: usize,
    proposed: f32,
    half: f32,
) -> f32 {
    let along = 1 - axis;
    let mut intervals: Vec<_> = scene
        .columns()
        .into_iter()
        .filter_map(|(a, b)| {
            let (l, h) = (a.xz(), b.xz());
            (a.y < 1.94 && b.y > 0.001 && l[along] < max[along] && h[along] > min[along])
                .then_some((l[axis] - half - 0.025, h[axis] + half + 0.025))
        })
        .collect();
    intervals.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut first = min[axis] + half + 0.10;
    let last = max[axis] - half - 0.10;
    let mut nearest = None::<f32>;
    let mut consider = |l: f32, h: f32| {
        if l <= h {
            let v = proposed.clamp(l, h);
            if nearest.is_none_or(|n| (v - proposed).abs() < (n - proposed).abs()) {
                nearest = Some(v);
            }
        }
    };
    for (l, h) in intervals {
        consider(first, last.min(l));
        first = first.max(h);
    }
    consider(first, last);
    nearest.unwrap_or(proposed)
}

impl IndoorManifest {
    pub(super) fn clear_front(
        &self,
        p: &ProgramAssembly,
        route: &Reservation,
        floor: f32,
        structure: &[(Vec3, Vec3)],
    ) -> bool {
        let (min, max) = (route.min, route.max);
        let lower = Vec3::new(min.x, floor, min.y);
        let upper = Vec3::new(max.x, floor + 1.9, max.y);
        let blocks = |(l, h): (Vec3, Vec3)| h.y > floor && l.y < upper.y && route.overlaps(l, h);
        min.cmpge(p.min).all()
            && max.cmple(p.max).all()
            && self.floor_support_clear(lower, upper)
            && self.minimum_ceiling_height(min, max) >= upper.y
            && !structure.iter().copied().any(blocks)
            && !self
                .objects
                .iter()
                .filter(|o| o.solid && !o.neighbor)
                .map(crate::scene::procedural_indoor::layout::programs::access::bounds)
                .any(blocks)
    }

    pub(super) fn front_access(
        &self,
        index: usize,
        o: &IndoorObject,
        depth: f32,
        structure: &[(Vec3, Vec3)],
    ) -> Option<Vec<Reservation>> {
        let p = &self.functional_programs()[index];
        let (lo, hi) = o.bounds();
        let (a, b) = (lo.xz(), hi.xz());
        let (sin, cos) = o.yaw.sin_cos();
        let side = usize::from(cos.abs() >= sin.abs());
        let along = 1 - side;
        let direction = if side == 1 { cos } else { sin };
        let staffed = o.kind == ObjectKind::CheckoutCounter
            && geometry::CheckoutForm::for_object(o) != Some(geometry::CheckoutForm::SelfCheckout);
        let mut routes = Vec::with_capacity(1 + usize::from(staffed));
        for staff in [false, true] {
            if staff && !staffed {
                break;
            }
            let mut min = a;
            let mut max = b;
            let positive = (direction > 0.) != staff;
            if positive {
                min[side] = b[side];
                max[side] += depth;
            } else {
                max[side] = a[side];
                min[side] -= depth;
            }
            // Carry the full browsing width to the common spine. Staff only
            // need sideways access to the till and can share a clear cross-aisle.
            if !staff {
                let spine = &p.reservations[along];
                min[along] = min[along].min(spine.max[along]);
                max[along] = max[along].max(spine.min[along]);
            }
            let route = Reservation {
                min,
                max,
                role: if staff {
                    ReservationRole::StaffAccess
                } else if matches!(
                    o.kind,
                    ObjectKind::WashingMachine
                        | ObjectKind::TumbleDryer
                        | ObjectKind::LaundryFoldingTable
                ) {
                    ReservationRole::EquipmentAccess
                } else if o.kind == ObjectKind::FittingRoom {
                    ReservationRole::FittingAccess
                } else {
                    ReservationRole::Browsing
                },
            };
            if !self.clear_front(p, &route, lo.y, structure) {
                return None;
            }
            routes.push(route);
        }
        Some(routes)
    }
}
