//! Optional self-service stations stay near the staffed register. A compact
//! checkout bank competes for the remaining floor after minimum merchandise.
use super::*;
use geometry::CheckoutForm;

impl IndoorManifest {
    pub(in super::super) fn fit_self_checkouts(
        &mut self,
        index: usize,
        structure: &PlacementStructure,
    ) {
        let p = &self.functional_programs()[index];
        let mut r = stream(self.seed, u64::from_le_bytes(*b"TillBank") ^ index as u64);
        if !r.random_bool(if p.activity == IndoorLayout::GroceryStore {
            0.72
        } else {
            0.20
        }) {
            return;
        }
        let Some(counter) = self.objects.iter().rev().find(|o| {
            o.kind == ObjectKind::CheckoutCounter
                && !o.neighbor
                && o.position.xz().cmpge(p.min).all()
                && o.position.xz().cmple(p.max).all()
        }) else {
            return;
        };
        let apparel = p.activity == IndoorLayout::ClothingStore;
        let size = geometry::retail::checkout::self_checkout_size(&mut r);
        let until = if r.random_bool(0.58) { 2 } else { 1 };
        let till = counter.position.xz();
        let facing = counter.yaw;
        let tangent = (counter.transform().rotation * Vec3::X).xz();
        let front = (counter.transform().rotation * Vec3::Z).xz();
        let gap = r.random_range(0.27..0.82);
        let end = (size.x + counter.size.x) * 0.5 + gap;
        let near_sign = if r.random_bool(0.5) { 1. } else { -1. };
        let mut sites = Vec::with_capacity(10);
        for sign in [near_sign, -near_sign] {
            sites.push((
                till + tangent * (sign * end) + front * r.random_range(-0.16..0.40),
                facing,
            ));
        }
        // A second option is the end of a quadrant facing the cross-aisle.
        // Prefer the least walking distance to the attendant; avoid a constant
        // central island and leave the two public routes themselves unoccupied.
        for axis in 0..2 {
            let along = 1 - axis;
            for inward in [-1., 1.] {
                let edge = &p.reservations[axis];
                for side in [-1., 1.] {
                    let cross = &p.reservations[along];
                    let mut q = Vec2::ZERO;
                    q[axis] = if inward > 0. {
                        edge.min[axis]
                    } else {
                        edge.max[axis]
                    } - inward * (0.5 * size.z + r.random_range(0.12..0.46));
                    q[along] = if side > 0. {
                        cross.max[along]
                    } else {
                        cross.min[along]
                    } + side * (0.5 * size.x + r.random_range(0.27..1.38));
                    let yaw = match (axis, inward > 0.) {
                        (0, true) => FRAC_PI_2,
                        (0, false) => -FRAC_PI_2,
                        (_, true) => 0.,
                        _ => PI,
                    };
                    sites.push((q, yaw));
                }
            }
        }
        sites[2..].sort_by(|a, b| {
            a.0.distance_squared(till)
                .total_cmp(&b.0.distance_squared(till))
        });
        let mut place = |s: &mut Self, q: Vec2, yaw: f32| {
            if q.distance_squared(till) > 6.2_f32.powi(2) {
                return None;
            }
            s.public_fit_with_structure(
                index,
                FloorProposal {
                    kind: ObjectKind::CheckoutCounter,
                    position: Vec3::new(q.x, 0., q.y),
                    size,
                    yaw,
                },
                &mut r,
                structure,
                |o| {
                    o.variant = CheckoutForm::self_service_tag(apparel);
                    o.position.xz().distance_squared(till) <= 6.2_f32.powi(2)
                },
            )
        };
        for (q, yaw) in sites {
            if let Some(id) = place(self, q, yaw) {
                if until == 2 {
                    let unit = &self.objects[id];
                    let pos = unit.position.xz();
                    let run = (unit.transform().rotation * Vec3::X).xz() * (size.x + gap);
                    for sign in [near_sign, -near_sign] {
                        if place(self, pos + sign * run, yaw).is_some() {
                            break;
                        }
                    }
                }
                break;
            }
        }
    }
}

/// Both corners of the customer (or cashier) work area must actually be served.
/// Merely counting reservations cannot detect a reversed station.
pub(in super::super::super) fn serves(route: &Reservation, o: &IndoorObject, staff: bool) -> bool {
    let depth = super::super::frontage::access_depth(ObjectKind::CheckoutCounter).unwrap();
    let sign = if staff { -1. } else { 1. };
    let transform = o.transform();
    [-1., 1.].into_iter().all(|side| {
        [0.003, depth - 0.003].into_iter().all(|at| {
            let corner = transform
                .transform_point(Vec3::new(
                    side * (o.size.x * 0.5 - 0.003),
                    0.,
                    sign * (o.size.z * 0.5 + at),
                ))
                .xz();
            corner.cmpge(route.min).all() && corner.cmple(route.max).all()
        })
    })
}
