//! Elliptical cutting radii in the rug's own warp and weft directions.
use super::*;

pub(super) fn tightest_bend(radius: Vec2) -> f32 {
    radius.min_element().powi(2) / radius.max_element()
}

pub(super) fn sample(o: &IndoorObject, cut: Cut) -> Option<[Vec2; 4]> {
    if !matches!(cut, Cut::Capsule | Cut::Asymmetric) {
        return None;
    }
    let size = o.size.xz();
    let short = size.min_element();
    let mut rng = stream(o.seed, 0x5255_4752_4144_4949);
    if cut == Cut::Capsule {
        // A continuous straight selvage joins two rounded ends. End depth
        // can differ, while each pair of quarters has the same tangent at the tip.
        let ends = [(); 2].map(|_| short * rng.random_range(0.26..0.82));
        let scale = (0.92 * size.max_element() / (ends[0] + ends[1])).min(1.);
        let [a, b] = ends.map(|end| end * scale);
        return Some(if size.x >= size.y {
            [a, b, b, a].map(|depth| Vec2::new(depth, short * 0.5))
        } else {
            [a, a, b, b].map(|depth| Vec2::new(short * 0.5, depth))
        });
    }

    let shoulder = rng.random_range(0..4);
    let paired = rng.random_bool(0.5);
    let radii: [Vec2; 4] = std::array::from_fn(|i| {
        let rounded = i == shoulder || (paired && i == (shoulder + 2) % 4);
        let radius = short
            * if rounded {
                rng.random_range(0.42..0.91)
            } else {
                rng.random_range(0.045..0.28)
            };
        Vec2::new(
            radius * rng.random_range(0.73..1.34),
            radius * rng.random_range(0.73..1.34),
        )
    });
    // Opposing fillets must fit on each edge. Preserve a short straight join
    // so the offset binding and the support polygon have no reversing edges.
    let used = Vec2::new(
        (radii[0].x + radii[1].x).max(radii[2].x + radii[3].x),
        (radii[1].y + radii[2].y).max(radii[3].y + radii[0].y),
    );
    let fit = (size * 0.94 / used).min(Vec2::ONE);
    Some(radii.map(|r| r * fit))
}
