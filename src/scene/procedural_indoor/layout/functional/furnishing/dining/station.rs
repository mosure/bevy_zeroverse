use super::*;

const STAFF_DEPTH: f32 = 1.05;

/// Service fronts face the room, with staff access behind the installed counter.
/// All installations are parallel to an edge of their usable zone.
pub(in super::super) fn staff_access(o: &IndoorObject) -> Reservation {
    let (lo, hi) = o.bounds();
    let a = lo.xz();
    let b = hi.xz();
    let (sin, cos) = o.yaw.sin_cos();
    let (min, max) = if sin.abs() > cos.abs() {
        if sin > 0. {
            (
                Vec2::new(a.x - STAFF_DEPTH, a.y),
                Vec2::new(a.x - 0.05, b.y),
            )
        } else {
            (
                Vec2::new(b.x + 0.05, a.y),
                Vec2::new(b.x + STAFF_DEPTH, b.y),
            )
        }
    } else if cos > 0. {
        (
            Vec2::new(a.x, a.y - STAFF_DEPTH),
            Vec2::new(b.x, a.y - 0.05),
        )
    } else {
        (
            Vec2::new(a.x, b.y + 0.05),
            Vec2::new(b.x, b.y + STAFF_DEPTH),
        )
    };
    Reservation {
        role: ReservationRole::StaffAccess,
        min,
        max,
    }
}

impl IndoorManifest {
    /// Fit the complete station inside one dining bank, with its back to an
    /// available wall. Try the ordered alternatives only until one is admitted.
    pub(in super::super) fn install_staffed_counter(
        &mut self,
        index: usize,
        kind: ObjectKind,
        stock: Vec3,
        rng: &mut ChaCha8Rng,
    ) {
        let p = &self.functional_programs()[index];
        let activity = p.activity;
        let aisle = &p.reservations[0];
        let mut banks = [
            (p.min.x, aisle.min.x, FRAC_PI_2),
            (aisle.max.x, p.max.x, -FRAC_PI_2),
        ];
        banks.shuffle(rng);
        let near = if p.activity == IndoorLayout::Restaurant {
            p.reservations[1].max.y
        } else {
            p.min.y
        };
        let far = p.max.y;
        for (left, right, side_yaw) in banks {
            let mut edges = [side_yaw, 0., PI];
            edges.shuffle(rng);
            for yaw in edges {
                let side_wall = yaw == side_yaw;
                let (begin, end, depth) = if side_wall {
                    (near, far, right - left)
                } else {
                    (left, right, far - near)
                };
                let width = stock.x.min(end - begin - 0.14);
                if width < 1.10 || depth < stock.z + STAFF_DEPTH + 0.18 {
                    continue;
                }
                let size = Vec3::new(width, stock.y, stock.z);
                let first = rng.random_range(0.12..0.88);
                for t in [first, 1. - first, 0.08, 0.92] {
                    let along = begin + 0.07 + width * 0.5 + (end - begin - width - 0.14) * t;
                    let inset = STAFF_DEPTH + stock.z * 0.5 + 0.09;
                    let (x, z) = if side_wall {
                        (
                            if yaw > 0. {
                                left + inset
                            } else {
                                right - inset
                            },
                            along,
                        )
                    } else {
                        (along, if yaw == 0. { near + inset } else { far - inset })
                    };
                    if let Some(id) =
                        self.public_fit(index, kind, Vec3::new(x, 0., z), size, yaw, rng)
                    {
                        if kind == ObjectKind::CheckoutCounter
                            && self.generator_version >= geometry::retail::checkout::VERSION
                        {
                            let o = &mut self.objects[id];
                            o.variant =
                                geometry::CheckoutForm::staffed(activity, o.size.x, o.seed).tag();
                        }
                        return;
                    }
                }
            }
        }
    }
}
