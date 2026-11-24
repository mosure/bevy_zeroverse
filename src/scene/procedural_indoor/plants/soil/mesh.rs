//! Each visible grain has a buried skirt and a continuous upper surface.
//! Share the rings and omit the underside sealed inside the opaque soil.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(super) fn append(g: &mut Geometry, grain: Grain, medium: Medium, level: f32) {
    let sides = match medium {
        Medium::BarkChips => 5,
        Medium::ExpandedClay | Medium::MineralGravel => 8,
        _ => 6,
    };
    // Draw small shape perturbations without instantiating another RNG per grain.
    let mut state = grain.shape_seed;
    let mut unit = || {
        state = state.wrapping_mul(1664525).wrapping_add(1013904223);
        (state >> 8) as f32 / 16777216.
    };
    let round = matches!(medium, Medium::ExpandedClay | Medium::MineralGravel);
    let shoulder = if round { 0.66 } else { 0.79 };
    let phase = Vec2::from_angle(grain.yaw);
    let step = Vec2::from_angle(TAU / sides as f32);
    let lift = grain.height;
    let top_offset = grain.half * Vec2::new(0.34 * (unit() - 0.5), 0.24 * (unit() - 0.5));
    let mut direction = Vec2::X;
    let base = g.positions.len() as u32;
    let mut point = |xz: Vec2, y: f32, n: Vec3| {
        let p = phase.rotate(xz) + grain.center;
        let nxz = phase.rotate(n.xz());
        g.positions.push([p.x, level + y, p.y]);
        g.normals.push([nxz.x, n.y, nxz.y]);
        // Chart each chip independently in metres, along its major axis.
        g.uvs
            .push((xz + grain.center + Vec2::Y * (0.21 * y)).to_array());
    };
    for _ in 0..sides {
        let outline = direction * grain.half * (0.83 + 0.17 * unit());
        let flank = direction / grain.half;
        point(
            outline,
            -lift * 0.14,
            Vec3::new(flank.x, 0.12 / lift, flank.y).normalize(),
        );
        let y = lift * (0.54 + 0.22 * unit());
        let p = outline * shoulder + top_offset * (1. - shoulder);
        point(
            p,
            y,
            Vec3::new(flank.x * shoulder, y / lift.powi(2), flank.y * shoulder).normalize(),
        );
        direction = step.rotate(direction);
    }
    let peak = base + 2 * sides;
    point(top_offset, lift, Vec3::Y);
    for i in 0..sides {
        let lo = base + 2 * i;
        let next = base + 2 * ((i + 1) % sides);
        g.indices.extend([
            lo,
            lo + 1,
            next + 1,
            lo,
            next + 1,
            next,
            lo + 1,
            peak,
            next + 1,
        ]);
    }
}
