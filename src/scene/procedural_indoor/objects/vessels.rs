//! Continuous hollow vessel profiles with a real base, wall and open mouth.
use super::*;
use crate::scene::procedural_indoor::geometry::{ProfileJoints, RevolvedOutline};

#[cfg(test)]
mod tests;

#[derive(Clone, Debug, serde::Serialize)]
pub struct VesselProgram {
    pub profile: u8,
    pub foot: f32,
    pub belly: f32,
    pub belly_height: f32,
    pub shoulder_height: f32,
    pub neck: f32,
    pub mouth: f32,
    pub flutes: u32,
    pub relief: f32,
    pub wall_fraction: f32,
    pub finish: Surface,
    pub foot_recess_m: f32,
    pub lip_rounding: f32,
    pub bowl_rise: f32,
    pub bowl_mouth: f32,
}
pub fn parameters(o: &IndoorObject) -> VesselProgram {
    let mut r = stream(o.seed, 0x56455353454c);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x5645535346494e);
    VesselProgram {
        profile: r.random_range(0..6),
        foot: r.random_range(0.32..0.72),
        belly: r.random_range(0.70..1.0),
        belly_height: r.random_range(0.24..0.58),
        shoulder_height: r.random_range(0.66..0.85),
        neck: r.random_range(0.19..0.68),
        mouth: r.random_range(0.72..1.30),
        flutes: r.random_range(6..=28),
        relief: if r.random_bool(0.38) {
            r.random_range(0.012..0.042)
        } else {
            0.
        },
        wall_fraction: r.random_range(0.025..0.070),
        finish: [
            Surface::Ceramic,
            Surface::Terracotta,
            Surface::ContainerGlass,
            Surface::Metal,
        ][f.random_range(0..4)],
        foot_recess_m: if r.random_bool(0.76) {
            (o.size.y * r.random_range(0.016..0.073)).min(0.009)
        } else {
            0.
        },
        lip_rounding: r.random_range(0.26..0.90),
        bowl_rise: r.random_range(0.62..2.60),
        bowl_mouth: r.random_range(0.76..1.0),
    }
}
impl VesselProgram {
    pub(crate) fn knots(&self) -> [(f32, f32); 6] {
        let neck = if self.profile == 1 {
            self.belly * 0.94
        } else {
            self.neck
        };
        let belly = if self.profile == 2 {
            neck * 1.10
        } else {
            self.belly
        };
        let waist = if self.profile == 3 {
            neck * 0.85
        } else {
            belly
        };
        [
            (0., self.foot),
            (0.1, self.foot * 1.05),
            (self.belly_height, belly),
            ((self.belly_height + self.shoulder_height) * 0.5, waist),
            (self.shoulder_height, neck),
            (1., (neck * self.mouth).min(0.98)),
        ]
    }
    pub(crate) fn radius_fraction(&self, bowl: bool, knots: &[(f32, f32); 6], t: f32) -> f32 {
        if bowl {
            let rise = ((t - 0.06) / 0.94).clamp(0., 1.);
            self.foot + (self.bowl_mouth - self.foot) * (1. - (1. - rise).powf(self.bowl_rise))
        } else {
            profile_radius(knots, t)
        }
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let bowl = o.kind == ObjectKind::Bowl;
    let r = s.x.min(s.z) * 0.47;
    let knots = p.knots();
    let radius_at = |t| r * p.radius_fraction(bowl, &knots, t);
    let wall = r * p.wall_fraction;
    let lip_rise = (wall * p.lip_rounding).min(0.048 * s.y);
    let wall_top = s.y - lip_rise;
    let inner_floor = (p.foot_recess_m + wall * 0.85).min(0.24 * wall_top);
    let sides = if p.relief > 0. {
        // Sample entire lobes. Two samples per flute alias the rib silhouette.
        ((64 + p.flutes / 2) / p.flutes).max(4) * p.flutes
    } else if p.profile == 4 {
        10
    } else {
        32
    };
    // Spend the angular budget on ribs while retaining enough profile samples
    // between shoulder stations. Both walls use the exact same sampled rings.
    let rows = (1024 / sides).clamp(10, 20);
    let mut profile = Vec::with_capacity(2 * rows as usize + 16);
    let foot = radius_at(0.);
    profile.push((0., p.foot_recess_m));
    if p.foot_recess_m > 0. {
        profile.extend([(foot * 0.68, p.foot_recess_m), (foot * 0.78, 0.)]);
    }
    let mut outer = Vec::with_capacity(rows as usize + 7);
    // Include the inner floor as an outer knot so both surfaces still have
    // positive thickness on a strongly flared foot, including shallow bowls.
    let mut stations: Vec<f32> = (0..=rows).map(|i| i as f32 / rows as f32).collect();
    stations.push(inner_floor / wall_top);
    if !bowl {
        stations.extend(knots[1..5].iter().map(|v| v.0));
    } else {
        stations.push(0.06);
    }
    stations.sort_unstable_by(f32::total_cmp);
    // Closely spaced knots produce sliver triangles. Preserve the floor height
    // exactly by inserting it into the inner wall separately below.
    stations.dedup_by(|a, b| (*a - *b).abs() < 0.002);
    for t in stations {
        outer.push((radius_at(t), wall_top * t));
    }
    profile.extend_from_slice(&outer);
    let mouth = radius_at(1.);
    let half = wall * 0.5;
    for (cos, sin) in [
        (
            std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ),
        (0., 1.),
        (
            -std::f32::consts::FRAC_1_SQRT_2,
            std::f32::consts::FRAC_1_SQRT_2,
        ),
        (-1., 0.),
    ] {
        profile.push((mouth - half + half * cos, wall_top + lip_rise * sin));
    }
    for &(outer_r, y) in outer.iter().rev().skip(1) {
        if y > inner_floor + wall_top * 0.002 {
            profile.push((outer_r - wall, y));
        }
    }
    profile.extend([
        (radius_at(inner_floor / wall_top) - wall, inner_floor),
        (0., inner_floor),
    ]);
    let outline = if p.relief == 0. && p.profile == 4 {
        RevolvedOutline::polygon(sides)
    } else {
        RevolvedOutline::round(sides, p.relief, p.flutes)
    };
    outline.sweep(
        a.part(p.finish, o.kind.class_name()),
        &profile,
        ProfileJoints::Smooth,
        Transform::IDENTITY,
    );
}

// Limit the derivative where the silhouette changes direction. Monotone spans
// of a turned shoulder keep their slope through intermediate control points.
fn profile_radius(knots: &[(f32, f32); 6], t: f32) -> f32 {
    let i = knots.partition_point(|k| k.0 < t).saturating_sub(1).min(4);
    let (x, y) = knots[i];
    let (dx, dy) = (knots[i + 1].0 - x, knots[i + 1].1 - y);
    let secant = |j: usize| (knots[j + 1].1 - knots[j].1) / (knots[j + 1].0 - knots[j].0);
    let slope = |j| {
        if j == 0 {
            return secant(0);
        }
        if j == 5 {
            return secant(4);
        }
        let (a, b) = (secant(j - 1), secant(j));
        if a * b <= 0. {
            0.
        } else {
            2. * a * b / (a + b)
        }
    };
    let u = ((t - x) / dx).clamp(0., 1.);
    y + u * u * (3. - 2. * u) * dy + u * (1. - u) * dx * ((1. - u) * slope(i) - u * slope(i + 1))
}
