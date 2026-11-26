//! Attach each span to its posts in the same local frame as the admitted item.
use super::*;

pub(super) fn build(a: &mut Assembly, label: &str, p: &BarrierProgram, fit: &Installation) {
    let x = fit.post_half_span_m;
    let h = fit.connection_y_m;
    let r = fit.post_radius_m;
    let web = if p.rail == Surface::Fabric {
        Surface::Fabric
    } else {
        Surface::FabricAlt
    };
    match fit.infill {
        Infill::Webbing {
            width_m,
            gauge_m,
            sag_m,
            reel_radius_m,
            reel_left,
        } => {
            let sign = if reel_left { 1. } else { -1. };
            let case_x = -sign * x;
            let shell = a.part(fit.hardware, label);
            let lid = width_m * 0.5 + 0.009;
            shell.lathe(
                &[
                    (0., -lid),
                    (reel_radius_m * 0.92, -lid),
                    (reel_radius_m, -lid + 0.004),
                    (reel_radius_m, lid - 0.005),
                    (reel_radius_m * 0.90, lid),
                    (0., lid),
                ],
                24,
                Transform::from_xyz(case_x, h, 0.),
            );
            // Two guides straddle the ribbon. Their gap remains visible at the exit.
            let mouth = case_x + sign * reel_radius_m;
            for z in [-1., 1.] {
                a.box_part(
                    Surface::Plastic,
                    label,
                    Vec3::new(mouth, h, z * (gauge_m * 0.5 + 0.0048)),
                    Vec3::new(0.024, width_m + 0.007, 0.006),
                    0.0013,
                );
            }
            let latch = sign * (x - r - 0.011);
            a.box_part(
                fit.hardware,
                label,
                Vec3::new(sign * (x - r * 0.62), h, 0.),
                Vec3::new(r, width_m + 0.025, r * 0.96),
                0.003,
            );
            a.box_part(
                Surface::Plastic,
                label,
                Vec3::new(latch, h, 0.),
                Vec3::new(0.025, width_m + 0.009, 0.014),
                0.002,
            );
            let from = mouth - sign * 0.002;
            let to = latch - sign * 0.008;
            let (left, right) = (from.min(to), from.max(to));
            ribbon(a.part(web, label), left, right, h, width_m, gauge_m, sag_m);
        }
        Infill::Rope {
            radius_m,
            end_reach_m,
            ferrule_m,
            catenary_q,
        } => {
            let half = x - end_reach_m;
            let at = |u: f32| {
                Vec3::new(
                    half * u,
                    h - half / catenary_q * (catenary_q.cosh() - (catenary_q * u).cosh()),
                    0.,
                )
            };
            let points: [Vec3; 17] = std::array::from_fn(|i| at(i as f32 / 8. - 1.));
            a.part(web, label).tube(&points, radius_m, 12);
            let metal = a.part(fit.hardware, label);
            for s in [-1., 1.] {
                let tip = at(s);
                metal.rod(Vec3::new(s * (x - r * 0.58), h, 0.), tip, radius_m * 0.75);
                // Crimp follows the local rope tangent. Its inboard end covers
                // the jacket and its outboard end overlaps the swivel pin.
                let inner = at(s * (1. - ferrule_m / half));
                metal.rod(
                    tip + (tip - inner).normalize() * radius_m * 0.45,
                    inner,
                    radius_m * 1.18,
                );
            }
        }
        Infill::Pickets {
            lower_y_m,
            rail_radius_m,
            picket_radius_m,
            pickets,
            square,
        } => {
            let metal = a.part(p.frame, label);
            for y in [h, lower_y_m] {
                if square {
                    metal.cuboid(
                        Vec3::new(x * 2., rail_radius_m * 1.8, rail_radius_m * 2.0),
                        rail_radius_m * 0.2,
                        Transform::from_xyz(0., y, 0.),
                    );
                } else {
                    metal.rod(Vec3::new(-x, y, 0.), Vec3::new(x, y, 0.), rail_radius_m);
                }
            }
            for i in 1..=pickets {
                let px = (2. * i as f32 / (pickets + 1) as f32 - 1.) * (x - r);
                if square {
                    metal.cuboid(
                        Vec3::new(picket_radius_m * 1.9, h - lower_y_m, picket_radius_m * 1.55),
                        picket_radius_m * 0.12,
                        Transform::from_xyz(px, (h + lower_y_m) * 0.5, 0.),
                    );
                } else {
                    metal.rod(
                        Vec3::new(px, lower_y_m, 0.),
                        Vec3::new(px, h, 0.),
                        picket_radius_m,
                    );
                }
            }
        }
        Infill::Panel {
            bottom_m,
            thickness_m,
            side_gap_m,
            clamp_inset_m,
            jaw_reach_m,
            finish,
        } => {
            let half = x - r - side_gap_m;
            a.box_part(
                finish,
                label,
                Vec3::Y * ((h + bottom_m) * 0.5),
                Vec3::new(half * 2., h - bottom_m, thickness_m),
                thickness_m * 0.28,
            );
            for side in [-1., 1.] {
                for y in [bottom_m + clamp_inset_m, h - clamp_inset_m] {
                    let jaw = (thickness_m * 0.40).clamp(0.003, 0.0055);
                    let bridge = half + jaw * 1.5;
                    let height = jaw_reach_m * 1.35;
                    let metal = a.part(fit.hardware, label);
                    metal.rod(
                        Vec3::new(side * (x - r * 0.5), y, 0.),
                        Vec3::new(side * bridge, y, 0.),
                        height * 0.22,
                    );
                    a.box_part(
                        fit.hardware,
                        label,
                        Vec3::new(side * bridge, y, 0.),
                        Vec3::new(3. * jaw, height * 0.86, thickness_m + jaw * 1.2),
                        jaw * 0.22,
                    );
                    for z in [-1., 1.] {
                        let inner = half - jaw_reach_m;
                        let end = bridge + jaw;
                        a.box_part(
                            fit.hardware,
                            label,
                            Vec3::new(
                                side * (inner + end) * 0.5,
                                y,
                                z * (thickness_m * 0.5 + jaw * 0.5 - 0.0004),
                            ),
                            Vec3::new(end - inner, height, jaw),
                            jaw * 0.23,
                        );
                    }
                }
            }
        }
    }
}

/// Closed stock with continuous metric face UVs along the arc of a tensioned belt.
fn ribbon(g: &mut Geometry, left: f32, right: f32, h: f32, w: f32, gauge: f32, sag: f32) {
    let section = |t: f32| {
        let x = left + (right - left) * t;
        let y = h - 4. * sag * t * (1. - t);
        [
            Vec3::new(x, y - w * 0.5, gauge * 0.5),
            Vec3::new(x, y + w * 0.5, gauge * 0.5),
            Vec3::new(x, y + w * 0.5, -gauge * 0.5),
            Vec3::new(x, y - w * 0.5, -gauge * 0.5),
        ]
    };
    let mut prev = section(0.);
    let mut u = 0.;
    g.metric_quad(prev, Vec3::NEG_X);
    for i in 1..=8 {
        let next = section(i as f32 / 8.);
        let end = u + next[0].distance(prev[0]);
        for (lo, hi, n) in [(0, 1, Vec3::Z), (2, 3, Vec3::NEG_Z)] {
            let (a, b) = if lo == 0 { (0., w) } else { (w, 0.) };
            g.vertex_quad(
                [prev[lo], next[lo], next[hi], prev[hi]],
                [n; 4],
                [[u, a], [end, a], [end, b], [u, b]],
            );
        }
        g.metric_quad([prev[1], next[1], next[2], prev[2]], Vec3::Y);
        g.metric_quad([prev[3], next[3], next[0], prev[0]], Vec3::NEG_Y);
        prev = next;
        u = end;
    }
    g.metric_quad(prev, Vec3::X);
}
