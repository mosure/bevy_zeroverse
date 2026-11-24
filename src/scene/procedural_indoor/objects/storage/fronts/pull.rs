//! Each mounting foot starts inside a flat, solid mounting rail or stile.
//! Pulls are owned by the moving door all the way to the joint endpoint.
use super::{
    hinged::{Leaf, Pull},
    *,
};

pub(super) fn build(a: &mut Assembly, d: &Leaf) {
    if d.pull == Pull::Touch {
        return;
    }
    let base = d.grip_center_m.extend(-0.0005);
    let along = if d.vertical_pull { Vec3::Y } else { Vec3::X };
    let g = a.part(d.pull_finish, "cabinet_hardware");
    let span = d.grip_span_m;
    let r = d.grip_radius_m;
    let reach = d.reach_m;
    match d.pull {
        Pull::Bow => {
            let height = reach - r;
            let bend = (height * 0.68).min(span * 0.23);
            let mid = (1. - std::f32::consts::FRAC_1_SQRT_2) * bend;
            let mut path = [Vec3::ZERO; 8];
            for (i, (side, inset, z)) in [
                (-1., 0., 0.),
                (-1., 0., height - bend),
                (-1., mid, height - mid),
                (-1., bend, height),
                (1., bend, height),
                (1., mid, height - mid),
                (1., 0., height - bend),
                (1., 0., 0.),
            ]
            .into_iter()
            .enumerate()
            {
                path[i] = base + along * (side * (span * 0.5 - inset)) + Vec3::Z * z;
            }
            g.tube(&path, r, 10);
        }
        Pull::Knob => {
            let k = (r * 2.5).min(d.stile_m * 0.25);
            g.lathe(
                &[
                    (0., 0.),
                    (k * 0.56, 0.),
                    (k * 0.52, reach * 0.28),
                    (k * 0.41, reach * 0.46),
                    (k * 0.93, reach * 0.66),
                    (k, reach * 0.82),
                    (k * 0.86, reach * 0.94),
                    (k * 0.50, reach),
                    (0., reach),
                ],
                20,
                Transform::from_translation(base).with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
            );
        }
        Pull::Tab => {
            let across = if d.vertical_pull {
                Vec3::X * d.hand.sign()
            } else {
                Vec3::NEG_Y
            };
            let thickness = (r * 0.59).min(reach * 0.18);
            let lip = reach * 0.42;
            let profile = [
                (0., 0.),
                (thickness, 0.),
                (thickness, reach - thickness),
                (lip, reach - thickness),
                (lip, reach),
                (0., reach),
            ];
            let end = |side: f32| {
                profile.map(|(v, z)| base + along * (side * span * 0.5) + across * v + Vec3::Z * z)
            };
            let (lo, hi) = (end(-1.), end(1.));
            for i in 0..6 {
                let j = (i + 1) % 6;
                let (v, z) = (profile[j].0 - profile[i].0, profile[j].1 - profile[i].1);
                g.metric_quad([lo[i], lo[j], hi[j], hi[i]], across * z - Vec3::Z * v);
            }
            for (points, n) in [(lo, -along), (hi, along)] {
                for q in [[0, 1, 2, 5], [2, 3, 4, 5]] {
                    g.metric_quad(q.map(|i| points[i]), n);
                }
            }
        }
        Pull::Touch => {}
    }
}
