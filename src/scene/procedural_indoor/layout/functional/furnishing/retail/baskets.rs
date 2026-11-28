//! Pickup beside an entrance or basket return beside a staffed checkout.
//! A shopper stands in a previously admitted aisle; its full width stays clear.
use super::*;

fn aisle_serves(route: &Reservation, o: &IndoorObject) -> bool {
    let standing = o
        .transform()
        .transform_point(Vec3::Z * (o.size.z * 0.5 + 0.46))
        .xz();
    let clearance = Vec2::splat(0.19);
    (standing - clearance).cmpge(route.min).all() && (standing + clearance).cmple(route.max).all()
}

pub(in super::super::super) fn valid_site(
    s: &IndoorManifest,
    p: &ProgramAssembly,
    o: &IndoorObject,
) -> bool {
    let pos = o.position.xz();
    let in_reach = if o.variant == 0 {
        pos.distance(Vec2::new(s.door_x, s.room_size.z * 0.5)) <= 3.8
    } else if o.variant == 1 {
        p.members
            .iter()
            .filter_map(|&id| s.objects.get(id))
            .any(|t| {
                t.kind == ObjectKind::CheckoutCounter
                    && geometry::CheckoutForm::for_object(t)
                        != Some(geometry::CheckoutForm::SelfCheckout)
                    && pos.distance(t.position.xz()) <= 3.2
                    && (pos - t.position.xz()).dot((t.transform().rotation * Vec3::Z).xz()) >= -0.16
            })
    } else {
        false
    };
    in_reach
        && p.reservations.iter().any(|r| {
            matches!(
                r.role,
                ReservationRole::Browsing | ReservationRole::Circulation
            ) && aisle_serves(r, o)
        })
}

impl IndoorManifest {
    pub(in super::super) fn furnish_basket_stacks(&mut self) {
        if self.generator_version < geometry::baskets::VERSION {
            return;
        }
        for index in 0..self.functional_programs().len() {
            let p = &self.functional_programs()[index];
            let rate = match p.activity {
                IndoorLayout::GroceryStore => 0.85,
                IndoorLayout::ClothingStore => 0.24,
                _ => continue,
            };
            let mut r = stream(
                self.seed,
                u64::from_le_bytes(*b"CarryBay") ^ ((p.activity as u64) << 32) ^ index as u64,
            );
            if !r.random_bool(rate) {
                continue;
            }
            let Some(till) = p.members.iter().map(|&i| &self.objects[i]).find(|o| {
                o.kind == ObjectKind::CheckoutCounter
                    && geometry::CheckoutForm::for_object(o)
                        != Some(geometry::CheckoutForm::SelfCheckout)
            }) else {
                continue;
            };
            let entry = Vec2::new(self.door_x, self.room_size.z * 0.5);
            let counter = till
                .transform()
                .transform_point(Vec3::Z * (till.size.z * 0.5 + 0.6))
                .xz();
            let (lo, hi) = (p.min, p.max);
            let checkout_aisle = p.reservations.iter().find(|route| {
                route.role == ReservationRole::Browsing
                    && super::checkout::serves(route, till, false)
            });
            let kit = r.random();
            let size = geometry::dimensions(ObjectKind::BasketStack, kit);
            let mut o = self.floor_candidate(ObjectKind::BasketStack, Vec3::ZERO, size, 0., &mut r);
            o.seed = kit;
            let prefer_entry = r.random_bool(0.59);
            let structure = PlacementStructure::new();
            'sites: for at_entry in [prefer_entry, !prefer_entry] {
                let target = if at_entry { entry } else { counter };
                let mut routes: Vec<_> = p.reservations.iter().take(2).collect();
                if !at_entry {
                    routes.extend(checkout_aisle)
                }
                routes.shuffle(&mut r);
                o.variant = u32::from(!at_entry);
                for route in routes {
                    let mut edges = [(0, -1.), (0, 1.), (1, -1.), (1, 1.)];
                    edges.shuffle(&mut r);
                    for (axis, side) in edges {
                        let tangent = 1 - axis;
                        let edge = if side < 0. {
                            route.min[axis]
                        } else {
                            route.max[axis]
                        };
                        let mut normal = Vec2::ZERO;
                        normal[axis] = -side;
                        o.yaw = normal.x.atan2(normal.y) + r.random_range(-0.36..0.36);
                        let (sin, cos) = o.yaw.sin_cos();
                        let half = (size.x * Vec2::new(cos.abs(), sin.abs())
                            + size.z * Vec2::new(sin.abs(), cos.abs()))
                            * 0.5;
                        let lower = route.min[tangent] + half[tangent] + 0.22;
                        let upper = route.max[tangent] - half[tangent] - 0.22;
                        if lower > upper {
                            continue;
                        }
                        let mut q = Vec2::ZERO;
                        q[axis] = edge + side * (half[axis] + r.random_range(0.045..0.13));
                        for _ in 0..2 {
                            q[tangent] =
                                (target[tangent] + r.random_range(-1.65..1.65)).clamp(lower, upper);
                            o.position = Vec3::new(q.x, self.floor_height(q), q.y);
                            let (l, h) = o.bounds();
                            if l.xz().cmpge(lo).all()
                                && h.xz().cmple(hi).all()
                                && aisle_serves(route, &o)
                                && valid_site(self, p, &o)
                                && self.placement_clear_with_structure(&o, 0.04, false, &structure)
                            {
                                self.assembly_diagnostics.programs[index].members.push(o.id);
                                self.objects.push(o);
                                break 'sites;
                            }
                        }
                    }
                }
            }
        }
    }
}
