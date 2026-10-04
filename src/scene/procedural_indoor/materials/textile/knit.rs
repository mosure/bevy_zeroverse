//! Stockinette loop crowns with overlapping courses and irregular yarn widths.
//! The same periodic field supplies colour, relief, roughness and occlusion.
use super::*;
use bevy::prelude::Vec2;

pub(super) fn texel(t: &TextileRecipe, r: &MaterialRecipe, u: f32, v: f32) -> Texel {
    let p = Vec2::new(
        u.rem_euclid(1.) * t.yarns[0] as f32,
        v.rem_euclid(1.) * t.yarns[1] as f32,
    );
    let cell = p.floor();
    let q = p - cell - Vec2::splat(0.5);
    let mut closest = (f32::INFINITY, 0.0, 0.0);
    for row in -1..=1 {
        for col in -1..=1 {
            let jitter = hash(
                (cell.x as i32 + col).rem_euclid(t.yarns[0] as i32) as u32,
                (cell.y as i32 + row).rem_euclid(t.yarns[1] as i32) as u32,
                r.seed,
            );
            let width = 0.075 + t.width[0] * 0.095 + t.slub * (jitter - 0.5) * 0.09;
            let local = q - Vec2::new(col as f32 + (jitter - 0.5) * 0.05, row as f32);
            // Joined quadratic legs and crown, discretized below texture scale.
            let controls = [
                [
                    Vec2::new(-0.35, -0.7),
                    Vec2::new(-0.02, -0.17),
                    Vec2::new(-0.23, 0.18),
                ],
                [
                    Vec2::new(-0.23, 0.18),
                    Vec2::new(0., 0.76),
                    Vec2::new(0.23, 0.18),
                ],
                [
                    Vec2::new(0.23, 0.18),
                    Vec2::new(0.02, -0.17),
                    Vec2::new(0.35, -0.7),
                ],
            ];
            for [a, b, c] in controls {
                let curve = |f| a.lerp(b, f).lerp(b.lerp(c, f), f);
                let mut previous = a;
                for i in 1..=4 {
                    let next = curve(i as f32 * 0.25);
                    let d = next - previous;
                    let f = ((local - previous).dot(d) / d.length_squared()).clamp(0., 1.);
                    let distance = local.distance_squared(previous + d * f) / (width * width);
                    if distance < closest.0 {
                        closest = (distance, jitter, local.y);
                    }
                    previous = next;
                }
            }
        }
    }
    let crown = (1. - closest.0).max(0.).sqrt();
    let nap = periodic_noise(u, v, 71, 73, r.seed.wrapping_add(127)) - 0.5;
    let yarn = 0.79 + crown * (0.18 + t.dye_variation * (closest.1 - 0.5)) + t.fuzz * nap * 0.03;
    Texel {
        color: t.yarn_tint[0].map(|c| (c * yarn).clamp(0., 1.)),
        height: r.relief_m * (crown * (0.85 + t.crimp * closest.2 * 0.15) - 0.45),
        roughness: (r.roughness + (1. - crown) * 0.09 + nap * t.fuzz * 0.08).clamp(0.4, 1.),
        occlusion: 0.89 + crown * 0.11,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn loop_relief_is_periodic_with_coupled_pbr_channels() {
        let recipe = super::super::super::program::sample(33)
            [super::super::super::Surface::Fabric as usize]
            .clone();
        let mut textile = recipe.textile.clone().unwrap();
        textile.knit = true;
        let mut heights = std::collections::BTreeSet::new();
        for i in 0..128 {
            let u = i as f32 / 127.0;
            let a = texel(&textile, &recipe, 0.0, u);
            let b = texel(&textile, &recipe, 1.0, u);
            assert_eq!(a.color, b.color);
            assert_eq!(a.height, b.height);
            assert_eq!(a.roughness, b.roughness);
            let c = texel(&textile, &recipe, u, 0.37);
            assert!(c.height.is_finite() && c.height.abs() <= recipe.relief_m);
            assert!((0.4..=1.0).contains(&c.roughness));
            assert!((0.89..=1.0).contains(&c.occlusion));
            heights.insert(c.height.to_bits());
        }
        assert!(heights.len() > 32);
    }
}
