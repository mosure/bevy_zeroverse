//! Lay-in diffusers with folded four-way cones or angled return blades.
use super::mesh::{band, ring};
use super::*;
use crate::scene::procedural_indoor::materials::acoustic_tile::GRID_LABEL;

pub(super) fn build(a: &mut Assembly, air: &AirTerminal, c: Vec2, span: Vec2, h: f32) {
    let g = a.part(Surface::Metal, GRID_LABEL);
    let half = span * 0.5;
    let rim = 0.012;
    let outlet = half - rim;
    let face = h - 0.003;
    let at = |size: Vec2, y: f32| ring(c - size, c + size, y);
    // Turned-up outside flange rests against the runner and carries the inserts.
    let outer = at(half + 0.001, face);
    let mouth = at(outlet, face);
    band(g, outer, mouth);
    band(g, mouth, at(outlet, h + 0.041));
    band(g, at(half + 0.001, h + 0.001), outer);
    match air.pattern {
        AirPattern::FourWaySupply => {
            let n = f32::from(air.blades);
            let step = outlet / (n + 0.46);
            let depth = (step.min_element() * 0.56 * air.angle_rad.tan()).clamp(0.009, 0.049);
            for i in 1..=air.blades {
                let inside = step * (f32::from(i) - 0.22);
                let outside = inside + step * 0.68;
                let low = at(inside, h + 0.004);
                let high = at(outside, h + 0.004 + depth);
                let t = Vec3::Y * 0.0018;
                band(g, high, low);
                band(g, low, low.map(|v| v + t));
                band(g, low.map(|v| v + t), high.map(|v| v + t));
                band(g, high.map(|v| v + t), high);
            }
            // A narrow cross member carries the nested cones. The central cap
            // terminates the stack with a small deflector over the plenum.
            let bar_y = h + 0.005 + depth;
            for (x, z) in [(2. * outlet.x, 0.009), (0.009, 2. * outlet.y)] {
                let y = if x > z { bar_y } else { bar_y + 0.003 };
                g.cuboid(Vec3::new(x, 0.003, z), 0., Transform::from_xyz(c.x, y, c.y));
            }
            g.cuboid(
                Vec3::new(step.x * 1.1, 0.008, step.y * 1.1),
                0.002,
                Transform::from_xyz(c.x, bar_y - 0.005, c.y),
            );
        }
        AirPattern::SlottedReturn => {
            let across = usize::from(!air.across_x);
            let count = (outlet[across] * 2. / (0.027 + f32::from(air.blades) * 0.002))
                .round()
                .clamp(7., 36.) as usize;
            let pitch = 2. * outlet[across] / count as f32;
            let rotation = if air.across_x {
                Quat::from_rotation_z(air.angle_rad)
            } else {
                Quat::from_rotation_x(air.angle_rad)
            };
            let mut dims = Vec3::new(outlet.x * 2., 0.0022, outlet.y * 2.);
            dims[across * 2] = pitch * 0.81;
            for i in 0..count {
                let mut p = Vec3::new(c.x, h + 0.018, c.y);
                p[across * 2] += (i as f32 + 0.5) * pitch - outlet[across];
                g.cuboid(
                    dims,
                    0.,
                    Transform::from_translation(p).with_rotation(rotation),
                );
            }
        }
    }
}
