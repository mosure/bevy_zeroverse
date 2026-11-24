//! Welt pockets and folded lapels follow the triangles of the cut front.
use super::*;

fn front(pattern: &Pattern, u: f32, v: f32) -> Vec3 {
    let x = (u + 1.) * ((U - 1) as f32 * 0.5);
    let i = (x as usize).min(U - 2);
    let j = V
        .partition_point(|&h| h <= v)
        .saturating_sub(1)
        .min(V.len() - 2);
    let du = x - i as f32;
    let dv = (v - V[j]) / (V[j + 1] - V[j]);
    let point = |i: usize, j: usize| pattern.panel(across(i), V[j], 1.);
    let b = point(i + 1, j);
    let c = point(i, j + 1);
    if du + dv <= 1. {
        let a = point(i, j);
        a + du * (b - a) + dv * (c - a)
    } else {
        let d = point(i + 1, j + 1);
        d + (1. - du) * (c - d) + (1. - dv) * (b - d)
    }
}

/// A folded piece of cloth with finished edges. The edge of a lapel lifts
/// away from the body while its roll line remains stitched close to the coat.
fn folded<const W: usize, const H: usize>(g: &mut Geometry, points: [[(Vec3, f32); W]; H]) {
    let inner = points.map(|row| row.map(|(p, lift)| p + Vec3::X * lift));
    let outer = inner.map(|row| row.map(|p| p + Vec3::X * 0.0018));
    let reverse = (outer[0][1] - outer[0][0])
        .cross(outer[1][0] - outer[0][0])
        .x
        < 0.;
    cloth::patch(g, outer, reverse);
    cloth::patch(g, inner, !reverse);
    let center = (inner[0][0] + inner[H - 1][W - 1]) * 0.5;
    for i in [0, W - 1] {
        let edge = outer.map(|r| r[i]);
        cloth::band(g, edge, inner.map(|r| r[i]), edge[H / 2] - center);
    }
    for j in [0, H - 1] {
        cloth::band(g, outer[j], inner[j], outer[j][W / 2] - center);
    }
}

pub(super) fn sew(g: &mut Geometry, p: &Pattern) {
    let coat = p.0;
    for side in [-1., 1.] {
        let pocket: [[_; 7]; 2] = std::array::from_fn(|j| {
            std::array::from_fn(|i| {
                let t = i as f32 / 6.;
                let slope = 0.028 + (1. + coat.fold_phase.sin()) * 0.045;
                let v = 0.32 + t * slope + (j as f32 - 0.5) * 0.014 / coat.length_m;
                (
                    front(p, side * (0.19 + 0.59 * t), v),
                    0.0015 + 0.0028 * j as f32,
                )
            })
        });
        folded(g, pocket);
        if coat.closure == 1 {
            let lapel: [[_; 4]; 7] = std::array::from_fn(|j| {
                std::array::from_fn(|i| {
                    let t = j as f32 / 6.;
                    let cross = i as f32 / 3.;
                    let roll = 0.055 + 0.25 * t;
                    // Notch at the gorge of the lapel. The point at the bottom has
                    // finite width, which also keeps tiny garments well-conditioned.
                    let opening = 0.035 + 0.29 * t - if j == 5 { 0.14 } else { 0.0 };
                    let v = (0.63 + 0.34 * t) - 0.035 * cross * t;
                    let lift = 0.0018 + cross * (0.002 + 0.003 * t);
                    (front(p, side * (roll + cross * opening), v), lift)
                })
            });
            folded(g, lapel);
        }
    }
}
