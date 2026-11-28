use super::*;
use crate::scene::procedural_indoor::{geometry::housing_outline, materials::worktop_finish};

/// Continuous metal/ceramic sheet, from its underside over the rolled lip down to
/// a circular waste. The centre is recessed and the two strainer bars bridge the bore.
pub(super) fn build(a: &mut Assembly, p: &Fountain, s: Vec3) {
    let outline = housing_outline(p.roundness);
    let circle = housing_outline(2.);
    let c = Fountain::bowl_center(s);
    let half = s.xz() * Vec2::new(0.495, 0.481);
    let t = p.deck_m;
    let low = t - p.basin_depth_m;
    let r = p.throat_radius_m;
    let ring = |profile: &[Vec2; 32], radii: Vec2, y: f32| {
        profile.map(|v| {
            let q = radii * v;
            c + Vec3::new(q.x, y, q.y)
        })
    };
    let throat = ring(&circle, Vec2::splat(r), low - 0.015);
    let sections = [
        throat,
        ring(&outline, half * 0.66, low - 0.015),
        ring(&outline, half * 0.95, t - 0.043),
        ring(&outline, half, t - 0.010),
        ring(&outline, half * 0.988, t),
        ring(&outline, half * Vec2::new(0.78, 0.76), t),
        ring(&outline, half * Vec2::new(0.69, 0.65), t - 0.029),
        ring(&outline, half * 0.36, low + 0.009),
        ring(&circle, Vec2::splat(r), low),
        throat,
    ];
    let g = a.part(p.bowl, worktop_finish::BASIN_LABEL);
    let mut u = [0.; 33];
    for i in 0..32 {
        u[i + 1] = u[i] + (half * (outline[(i + 1) % 32] - outline[i])).length();
    }
    let mut v = [0.; 32];
    for pair in sections.windows(2) {
        let next_v: [f32; 32] = std::array::from_fn(|i| v[i] + pair[1][i].distance(pair[0][i]));
        let flat = pair[1][0].y == pair[0][0].y;
        for i in 0..32 {
            let j = (i + 1) % 32;
            let vertex = |row: usize, k: usize| {
                let point = pair[row][k];
                let tangent = pair[row][(k + 1) % 32] - pair[row][(k + 31) % 32];
                let n = (pair[1][k] - pair[0][k]).cross(tangent).normalize();
                (point, n)
            };
            let corners = [vertex(0, i), vertex(1, i), vertex(1, j), vertex(0, j)];
            g.vertex_quad(
                corners.map(|(v, _)| v),
                corners.map(|(_, n)| n),
                if flat {
                    corners.map(|(v, _)| v.xz().to_array())
                } else {
                    [
                        [u[i], v[i]],
                        [u[i], next_v[i]],
                        [u[i + 1], next_v[j]],
                        [u[i + 1], v[j]],
                    ]
                },
            );
        }
        v = next_v;
    }
    a.part(Surface::Ink, "sink").cylinder(
        r,
        0.002,
        Transform::from_translation(c + Vec3::Y * (low - 0.016)),
    );
    for side in [-1., 1.] {
        a.box_part(
            Surface::Chrome,
            "sink",
            c + Vec3::new(side * r * 0.32, low - 0.003, 0.),
            Vec3::new(0.0025, 0.002, r * 1.96),
            0.0005,
        );
    }
}
