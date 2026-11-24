//! Folded supply blades or two offset layers of return bars in an open boot.
use super::super::{
    register::{RegisterPattern, RegisterStock},
    sheet::{band, ring},
};
use super::*;
use crate::scene::procedural_indoor::materials::acoustic_tile::GRID_LABEL;

/// Keep the rear of the aperture below any part of a bent or reducing trunk
/// crossing its face. Both endpoint sections enclose their intervening loft.
fn clear_height(rings: &[(Vec3, Vec2)], size: Vec2, joint: Vec3) -> f32 {
    let lo = joint.xz() - size * 0.5;
    let hi = joint.xz() + size * 0.5;
    let mut ceiling = joint.y;
    for segment in rings.windows(2) {
        let (p, r) = segment[0];
        let (q, s) = segment[1];
        let seam = 0.004; // The rolled joint stands proud of the sheet metal.
        if p.x <= hi.x + seam
            && q.x >= lo.x - seam
            && (p.z - r.y).min(q.z - s.y) <= hi.y + seam
            && (p.z + r.y).max(q.z + s.y) >= lo.y - seam
        {
            ceiling = ceiling.min((p.y - r.x).min(q.y - s.x) - seam);
        }
    }
    ceiling
}

/// Top is the underside of the supplying tube. The complete face fits above the
/// module's lower reserve, including on shallow horizontal takeoffs.
pub(super) fn build(
    a: &mut Assembly,
    f: &Frame,
    stock: RegisterStock,
    size: Vec2,
    top: Vec3,
    neck: f32,
    rings: &[(Vec3, Vec2)],
) {
    let ceiling = clear_height(rings, size, top);
    let depth = stock.depth.min(ceiling - 0.002);
    let face = ceiling - depth;
    let half = size * 0.5;
    let flange = stock.flange.min(half.min_element() * 0.23);
    let throat = half - flange;
    let rect = |r: Vec2, y: f32| ring(top.xz() - r, top.xz() + r, y);
    let rim = rect(half, face);
    let shoulder = rect(half, face + depth * 0.76);
    let collar = rect(
        (half * 0.86).min(Vec2::splat(neck * 0.60)),
        top.y + neck * 0.24,
    );
    let shell = f.part(a, Surface::Metal);
    shell.metric_quad(collar, Vec3::Y);
    band(shell, collar, shoulder);
    band(shell, shoulder, rim);

    let mouth = rect(throat, face);
    let rear = rect(throat, face + depth * 0.72);
    let front = if stock.coated {
        a.part(Surface::Metal, GRID_LABEL)
    } else {
        f.part(a, Surface::Metal)
    };
    band(front, rim, mouth);
    band(front, mouth, rear);

    blades(front, stock, top.xz(), throat, face, depth, false);
    if stock.pattern == RegisterPattern::CrossedReturn {
        blades(front, stock, top.xz(), throat, face, depth, true);
    }
    a.part(Surface::Ink, "ceiling")
        .metric_quad(rear, Vec3::NEG_Y);
}

fn blades(
    g: &mut Geometry,
    stock: RegisterStock,
    center: Vec2,
    throat: Vec2,
    face: f32,
    depth: f32,
    transverse: bool,
) {
    let crossed = stock.pattern == RegisterPattern::CrossedReturn;
    let (across, span) = if transverse {
        (throat.y, throat.x)
    } else {
        (throat.x, throat.y)
    };
    let count = (2. * across / stock.pitch).round().clamp(2., 20.) as usize;
    let step = 2. * across / count as f32;
    for i in 0..count {
        let x = (i as f32 + 0.5) * step - across;
        let tilt = match stock.pattern {
            RegisterPattern::Directional => stock.tilt,
            RegisterPattern::Opposed => {
                if x < 0. {
                    -stock.tilt
                } else {
                    stock.tilt
                }
            }
            RegisterPattern::CrossedReturn => 0.,
        };
        // Return bars cross in separate layers with an air gap. The lower
        // tips of supply vanes turn back like roll-formed sheet stock.
        let bottom = face + depth * if transverse { 0.44 } else { 0.09 };
        let height = (step * 0.96).min(depth * if crossed { 0.23 } else { 0.52 });
        let run = (height * tilt.tan()).clamp(-step * 0.64, step * 0.64);
        let gauge = (if crossed { 0.0017_f32 } else { 0.0012 }).min(step * 0.12);
        let low = x - run * 0.50;
        let mid = low + run * 0.22;
        let high = x + run * 0.50;
        let profile = [
            (low - gauge * 0.5, bottom),
            (low + gauge * 0.5, bottom),
            (mid + gauge * 0.5, bottom + height * 0.42),
            (high + gauge * 0.5, bottom + height),
            (high - gauge * 0.5, bottom + height),
            (mid - gauge * 0.5, bottom + height * 0.42),
        ];
        // Ends terminate in the reveal. A quarter turn retains the winding.
        let point = |(u, y), v| {
            if transverse {
                Vec3::new(center.x - v, y, center.y + u)
            } else {
                Vec3::new(center.x + u, y, center.y + v)
            }
        };
        for (j, &v) in profile.iter().enumerate() {
            let w = profile[(j + 1) % profile.len()];
            let p = [
                point(v, -span),
                point(w, -span),
                point(w, span),
                point(v, span),
            ];
            g.metric_quad(p, (p[1] - p[0]).cross(p[2] - p[0]));
        }
    }
}

#[cfg(test)]
mod tests;
