use super::*;

/// Offer either long edge or either end of the furniture. Longer edges get more
/// proposals; the regular floor, furniture, doorway and human checks admit them.
pub(super) fn beside(anchor: &IndoorObject, rug: Vec2, rng: &mut impl Rng) -> (Vec3, f32) {
    let end_probability = (anchor.size.z / (anchor.size.x + anchor.size.z)).clamp(0.18, 0.65);
    let beside_end = rng.random_bool(f64::from(end_probability));
    let (span, depth, heading) = if beside_end {
        (anchor.size.z, anchor.size.x, std::f32::consts::FRAC_PI_2)
    } else {
        (anchor.size.x, anchor.size.z, 0.)
    };
    let skew: f32 = rng.random_range(-0.25..0.25);
    let (sin, cos) = skew.sin_cos();
    let reach = (rug.x * sin.abs() + rug.y * cos.abs()) * 0.5;
    let across = depth * 0.5 + reach + rng.random_range(0.05..0.35);
    let across = if rng.random_bool(0.5) {
        across
    } else {
        -across
    };
    let along = rng.random_range(-0.45..0.45) * span;
    let offset = if beside_end {
        Vec3::new(across, 0., along)
    } else {
        Vec3::new(along, 0., across)
    };
    (
        anchor.transform().transform_point(offset).with_y(0.),
        anchor.yaw + heading + skew,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::audit;

    #[test]
    fn loose_mats_reach_all_four_sides_and_clear_the_rotated_anchor() {
        let mut anchor = audit::fixture(ObjectKind::Sofa, 14);
        anchor.size = Vec3::new(2.5, 0.89, 0.94);
        anchor.position = Vec3::new(3., 0., -4.);
        anchor.yaw = 0.71;
        let inverse = anchor.transform().compute_affine().inverse();
        let size = Vec2::new(2.1, 0.64);
        let mut sides = [0; 4];
        let mut rng = stream(54, 7330);
        for _ in 0..512 {
            let (p, yaw) = beside(&anchor, size, &mut rng);
            let c = inverse.transform_point3(p);
            let end = c.x.abs() > anchor.size.x * 0.5;
            let (sin, cos) = (yaw - anchor.yaw).sin_cos();
            let clearance = if end {
                c.x.abs() - (anchor.size.x + size.x * cos.abs() + size.y * sin.abs()) * 0.5
            } else {
                c.z.abs() - (anchor.size.z + size.x * sin.abs() + size.y * cos.abs()) * 0.5
            };
            assert!((0.0499..0.3501).contains(&clearance));
            let side = if end {
                usize::from(c.x > 0.)
            } else {
                2 + usize::from(c.z > 0.)
            };
            sides[side] += 1;
        }
        assert!(sides.iter().all(|&n| n > 36), "{sides:?}");
        assert!(sides[2] + sides[3] > sides[0] + sides[1]);
    }
}
