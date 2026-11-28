use super::*;

#[derive(Clone, Copy)]
struct Placement {
    base: Vec3,
    yaw: f32,
}

fn products(o: &IndoorObject, case: RefrigeratedCase) -> [(Sku, Vec3); 4] {
    let shelf = case.shelf(o.size, 0);
    let pitch = (shelf.half_width - 0.012) * 2. / case.facings as f32;
    assortment(o).map(|sku| {
        let (surface, _) = sku.pack.face();
        let size = sku.pack.size(
            pitch,
            shelf.clearance,
            &mut stream(o.seed, 0x434f_4c44 + surface as u64 * 16 + sku.lot as u64),
        );
        (sku, size)
    })
}

/// Each run fronts one product. A depleted lane can be empty or set back.
/// The maximum rotated footprint determines both lane pitch and depth spacing.
fn place(
    o: &IndoorObject,
    case: RefrigeratedCase,
    sizes: [Vec3; 4],
    mut put: impl FnMut(usize, usize, Placement),
) {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"ColdLoad"));
    let offset = r.random_range(0..4);
    for level in 0..case.levels {
        let shelf = case.shelf(o.size, level);
        let half = shelf.half_width - 0.012;
        let pitch = half * 2. / case.facings as f32;
        let split = r.random_range(1..case.facings);
        for (group, slots) in [(0, 0..split), (1, split..case.facings)] {
            let sku = (offset + level * 2 + group) % 4;
            let size = sizes[sku];
            // abs(sin(yaw)) <= 0.14, cos(yaw) <= 1.
            let width = size.x + 0.14 * size.z;
            let depth = size.z + 0.14 * size.x;
            let step = (width + r.random_range(0.007..0.030)).min(pitch);
            let count = slots.len();
            let spare = pitch * count as f32 - step * (count - 1) as f32 - width;
            let x0 = -half
                + pitch * slots.start as f32
                + spare.max(0.) * r.random_range(0.08..0.92)
                + width * 0.5;
            let front = shelf.front - depth * 0.5 - r.random_range(0.013..0.060);
            let anchor = r.random_range(0..count);
            let gap = r.random_range(0.008..0.026);
            for lane in 0..count {
                if lane != anchor && !r.random_bool(f64::from(case.stock)) {
                    continue;
                }
                let x = x0 + step * lane as f32;
                let mut z = front - r.random_range(0.0..0.023);
                let copies = r.random_range(1..=3);
                for _ in 0..copies {
                    if z - depth * 0.5 < shelf.back + 0.008 {
                        break;
                    }
                    put(
                        sku,
                        level,
                        Placement {
                            base: Vec3::new(x, shelf.top, z),
                            yaw: r.random_range(-0.14..0.14),
                        },
                    );
                    z -= depth + gap;
                }
            }
        }
    }
}

pub(in super::super) fn stock(a: &mut Assembly, o: &IndoorObject, case: RefrigeratedCase) {
    let products = products(o, case);
    let mut lots: [Vec<(Vec3, f32, f32)>; 4] = Default::default();
    place(o, case, products.map(|(_, s)| s), |sku, _, p| {
        let (sin, cos) = p.yaw.sin_cos();
        lots[sku].push((p.base, sin, cos));
    });
    build_lots(a, products, lots);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::audit;
    #[test]
    fn retail_cold_stock_stays_on_its_shelf_clear_of_neighbors_and_the_air_return() {
        let mut levels = std::collections::BTreeSet::new();
        let mut shapes = [false; 3];
        for seed in 0..128 {
            let mut o = audit::fixture(ObjectKind::GroceryDisplay, seed);
            o.variant |= REFRIGERATED;
            for (width, height, depth) in [(1., 1.15, 0.64), (2.68, 1.94, 0.93)] {
                o.size = Vec3::new(width, height, depth);
                let case = refrigerated_case(&o).unwrap();
                levels.insert(case.levels);
                let products = products(&o, case);
                let mut bounds: Vec<(usize, Vec2, Vec2)> = Vec::new();
                let mut seen = [false; 4];
                place(&o, case, products.map(|(_, s)| s), |sku, level, p| {
                    let size = products[sku].1;
                    let (sn, cs) = p.yaw.sin_cos();
                    let half = (size.xz() * cs.abs() + Vec2::new(size.z, size.x) * sn.abs()) * 0.5;
                    let lo = p.base.xz() - half;
                    let hi = p.base.xz() + half;
                    let s = case.shelf(o.size, level);
                    assert!(lo.x >= -s.half_width && hi.x <= s.half_width);
                    assert!(lo.y > s.back && hi.y < s.front);
                    assert_eq!(p.base.y, s.top);
                    assert!(size.y < s.clearance);
                    for &(other_level, l, h) in &bounds {
                        assert!(
                            other_level != level || !lo.cmplt(h).all() || !hi.cmpgt(l).all(),
                            "stock overlaps at {seed}/{sku}"
                        );
                    }
                    bounds.push((level, lo, hi));
                    seen[sku] = true;
                    shapes[match products[sku].0.pack {
                        Pack::Gable => 0,
                        Pack::Tub => 1,
                        Pack::Bottle => 2,
                        _ => unreachable!(),
                    }] = true;
                });
                assert!(
                    seen.into_iter().all(|v| v) && bounds.len() <= case.facings * case.levels * 3
                );
            }
        }
        assert!(levels.len() >= 3 && shapes.into_iter().all(|v| v));
    }
}
