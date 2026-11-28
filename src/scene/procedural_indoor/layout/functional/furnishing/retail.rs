//! Shop fronts connect directly to one of two public cross-aisles. Existing
//! walkways may share floor area. Stock, structure and checkouts may not fill it.
pub(in super::super) mod baskets;
pub(in super::super) mod checkout;
use super::*;
use rand::seq::SliceRandom;

impl IndoorManifest {
    /// Only merchandise facing an additional aisle needs a rear approach. All
    /// static furnishings have settled by now, so this cannot displace a rack
    /// or consume random draws from placement or camera admission.
    pub(in crate::scene::procedural_indoor::layout) fn settle_folded_table_fronts(&mut self) {
        use geometry::retail::{wants_opposite_face, FOLDED_APPAREL_VERSION, SINGLE_BROWSING_FACE};
        if self.generator_version < FOLDED_APPAREL_VERSION {
            return;
        }
        let structure = PlacementStructure::new();
        let mut front_only = Vec::new();
        for (index, zone) in self.functional_programs().iter().enumerate() {
            if zone.activity != IndoorLayout::ClothingStore {
                continue;
            }
            for &id in &zone.members {
                let o = &self.objects[id];
                if !wants_opposite_face(o) {
                    continue;
                }
                let mut reverse = o.clone();
                reverse.yaw += PI;
                let depth = super::frontage::APPAREL_BROWSING_DEPTH;
                if self
                    .front_access(
                        index,
                        &reverse,
                        depth,
                        structure.get_or_init(|| self.columns()),
                    )
                    .is_none()
                {
                    front_only.push(id);
                }
            }
        }
        for id in front_only {
            self.objects[id].variant |= SINGLE_BROWSING_FACE;
        }
    }

    pub(super) fn retail_program(&mut self, index: usize, rng: &mut ChaCha8Rng) {
        let p = &self.functional_programs()[index];
        let kind = if p.activity == IndoorLayout::GroceryStore {
            ObjectKind::GroceryDisplay
        } else {
            ObjectKind::ApparelDisplay
        };
        let (spine, cross) = (&p.reservations[0], &p.reservations[1]);
        let (lo, hi) = (p.min, p.max);
        let xs = [(lo.x, spine.min.x), (spine.max.x, hi.x)];
        let zs = [(lo.y, cross.min.y), (cross.max.y, hi.y)];
        let prefer_long = rng.random_bool(0.52);
        let pitch = rng.random_range(2.45..3.35) + (1. - self.density) * 0.7;
        let mut sites = Vec::new();
        for (left, right) in xs {
            for (near, far) in zs {
                let mid = Vec2::new((left + right) * 0.5, (near + far) * 0.5);
                let span = Vec2::new(right - left, far - near);
                let along_z = if span.x > span.y * 1.65 {
                    false
                } else if span.y > span.x * 1.65 {
                    true
                } else {
                    prefer_long
                        ^ rng.random_bool(if kind == ObjectKind::ApparelDisplay {
                            0.40
                        } else {
                            0.20
                        })
                };
                let across = usize::from(!along_z);
                let along = 1 - across;
                let lanes = ((span[across] / pitch).round() as usize).clamp(1, 3);
                let runs = ((span[along] / (pitch + 0.12)).round() as usize).clamp(1, 3);
                let toward_spine = if across == 0 {
                    (spine.min.x + spine.max.x) * 0.5 - mid.x
                } else {
                    (cross.min.y + cross.max.y) * 0.5 - mid.y
                };
                let yaw = match (across, toward_spine > 0.) {
                    (0, true) => FRAC_PI_2,
                    (0, false) => -FRAC_PI_2,
                    (_, true) => 0.,
                    _ => PI,
                };
                for lane in 0..lanes {
                    for run in 0..runs {
                        let mut q = mid - span * 0.5;
                        q[across] += span[across] * (lane as f32 + rng.random_range(0.35..0.65))
                            / lanes as f32;
                        q[along] +=
                            span[along] * (run as f32 + rng.random_range(0.42..0.58)) / runs as f32;
                        let cant = if kind == ObjectKind::ApparelDisplay {
                            0.19
                        } else {
                            0.045
                        };
                        sites.push((
                            q,
                            yaw + rng.random_range(-cant..cant),
                            span[along] / runs as f32,
                        ));
                    }
                }
            }
        }
        sites.shuffle(rng);
        let first = self.objects.len();
        let mut accepted = 0;
        let wants_fitting_rooms =
            self.generator_version >= 287 && kind == ObjectKind::ApparelDisplay;
        let upgraded = self.generator_version >= geometry::retail::checkout::VERSION;
        let structure = PlacementStructure::new();
        let place_stock =
            |scene: &mut IndoorManifest, q: Vec2, size: Vec3, yaw: f32, rng: &mut ChaCha8Rng| {
                let proposal = FloorProposal {
                    kind,
                    position: Vec3::new(q.x, 0., q.y),
                    size,
                    yaw,
                };
                scene
                    .public_fit_with_structure(index, proposal, rng, &structure, |_| true)
                    .is_some()
            };
        for (q, yaw, length) in sites {
            let mut size = geometry::dimensions(kind, rng.random());
            size.x = size.x.min((length - 0.16).max(1.0));
            self.assembly_diagnostics.programs[index].requested_capacity += 1;
            if place_stock(self, q, size, yaw, rng) {
                accepted += 1;
                if accepted == 2 {
                    if wants_fitting_rooms {
                        self.fit_changing_rooms(index, &structure);
                    }
                    if upgraded {
                        self.fit_self_checkouts(index, &structure);
                    }
                }
            }
        }
        // A narrow clipped quadrant can lose its nominal run. Smaller fixtures
        // still have to admit the complete connected customer path.
        if accepted < 2 {
            for uv in variation::recovery_points(5, rng) {
                let q = lo + (hi - lo) * uv;
                let yaw = [0., FRAC_PI_2, PI, -FRAC_PI_2][rng.random_range(0..4)];
                let mut size = geometry::dimensions(kind, rng.random());
                size.x = size.x.min(1.32);
                self.assembly_diagnostics.programs[index].requested_capacity += 1;
                accepted += usize::from(place_stock(self, q, size, yaw, rng));
                if accepted >= 2 {
                    if wants_fitting_rooms {
                        self.fit_changing_rooms(index, &structure);
                    }
                    if upgraded {
                        self.fit_self_checkouts(index, &structure);
                    }
                    break;
                }
            }
        }
        if self.generator_version >= 286 && kind == ObjectKind::GroceryDisplay {
            assign_refrigeration(&mut self.objects[first..], lo, hi, self.seed, index);
        }
        if accepted > 0 {
            let mut look = stream(self.seed, u64::from_le_bytes(*b"ShopLook") ^ index as u64);
            let stock = self.objects[first..]
                .iter()
                .filter(|o| o.kind == kind)
                .nth(look.random_range(0..accepted))
                .expect("accepted sales fixture");
            // A register is a plausible subject near the sales aisles. Membership
            // is restricted to this zone and ordinary camera admission still runs.
            let mut subject = stream(self.seed, u64::from_le_bytes(*b"TillLook") ^ index as u64);
            let display = if upgraded && subject.random_bool(0.24) {
                use rand::seq::IteratorRandom;
                self.objects
                    .iter()
                    .filter(|o| {
                        o.kind == ObjectKind::CheckoutCounter
                            && !o.neighbor
                            && o.position.xz().cmpge(lo).all()
                            && o.position.xz().cmple(hi).all()
                    })
                    .choose(&mut subject)
                    .unwrap_or(stock)
            } else {
                stock
            };
            let local = Vec3::new(
                display.size.x * look.random_range(-0.24..0.24),
                display.size.y * look.random_range(0.58..0.90),
                display.size.z * 0.30,
            );
            self.assembly_diagnostics.programs[index].focus =
                display.transform().transform_point(local);
        }
        self.assembly_diagnostics.programs[index]
            .required
            .push(kind);
    }
}

/// A shop can specialize in ambient goods. In chilled departments, favor
/// cabinets backed toward the perimeter, looking into an admitted sales aisle.
fn assign_refrigeration(displays: &mut [IndoorObject], lo: Vec2, hi: Vec2, seed: u64, zone: usize) {
    let mut shop = stream(seed, u64::from_le_bytes(*b"ColdDept") ^ zone as u64);
    if shop.random_bool(0.17) {
        return;
    }
    let intensity = shop.random_range(0.43..0.95);
    let preferred_axis = shop.random_range(0..2);
    for o in displays
        .iter_mut()
        .filter(|o| o.kind == ObjectKind::GroceryDisplay && o.size.y >= 1.15)
    {
        let (l, h) = o.bounds();
        let (sin, cos) = o.yaw.sin_cos();
        let axis = usize::from(cos.abs() >= sin.abs());
        let front = if axis == 1 { cos } else { sin };
        let distance = if front > 0. {
            l.xz()[axis] - lo[axis]
        } else {
            hi[axis] - h.xz()[axis]
        };
        let perimeter = 1. - (distance / ((hi - lo)[axis] * 0.29).clamp(1.4, 3.2)).clamp(0., 1.);
        let probability =
            intensity * (0.10 + 0.90 * perimeter) * if axis == preferred_axis { 1. } else { 0.74 };
        if stream(o.seed, u64::from_le_bytes(*b"ColdSite")).random_bool(f64::from(probability)) {
            o.variant |= geometry::retail::REFRIGERATED;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use geometry::retail::{folded_apparel, wants_opposite_face, SINGLE_BROWSING_FACE};

    #[test]
    fn folded_merchandise_faces_only_connected_browsing_aisles() {
        let mut open = 0;
        let mut blocked = 0;
        for seed in 0..40 {
            let shop = IndoorManifest::generate_with_humans(
                seed,
                IndoorLayout::ClothingStore,
                0.65,
                0,
                0.25,
            )
            .unwrap();
            let structure = PlacementStructure::new();
            for (index, program) in shop.functional_programs().iter().enumerate() {
                for &id in &program.members {
                    let o = &shop.objects[id];
                    let Some(stock) = folded_apparel(o, shop.generator_version) else {
                        continue;
                    };
                    if !wants_opposite_face(o) {
                        assert_eq!(stock.rows, 1);
                        continue;
                    }
                    let mut rear = o.clone();
                    rear.yaw += PI;
                    let route = shop.front_access(
                        index,
                        &rear,
                        super::super::frontage::APPAREL_BROWSING_DEPTH,
                        structure.get_or_init(|| shop.columns()),
                    );
                    assert_eq!(route.is_none(), o.variant & SINGLE_BROWSING_FACE != 0);
                    if route.is_some() {
                        open += 1;
                        assert_eq!(stock.rows, 2);
                    } else {
                        blocked += 1;
                        assert_eq!(stock.rows, 1);
                        assert!(stock
                            .piles
                            .iter()
                            .flat_map(|p| &p.layers)
                            .all(|l| l.yaw_rad.cos() > 0.9));
                    }
                }
            }
        }
        assert!(open > 0 && blocked > 0, "{open} open, {blocked} blocked");
    }
}
