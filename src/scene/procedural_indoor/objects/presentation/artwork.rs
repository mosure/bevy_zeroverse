//! One exterior skin for the framed print, cut window mat, floater or stretched
//! canvas. Adjacent materials share perimeter edges, including at the mitres.
use super::*;
use crate::scene::procedural_indoor::materials::art::{frame_parameters, FrameProgram};
mod moulding;

#[derive(Clone, Copy)]
struct Rect {
    lo: Vec2,
    hi: Vec2,
}

impl Rect {
    fn inset(self, by: Vec2) -> Self {
        Self {
            lo: self.lo + by,
            hi: self.hi - by,
        }
    }
    fn shifted(self, by: Vec2) -> Self {
        Self {
            lo: self.lo + by,
            hi: self.hi + by,
        }
    }
    fn ring(self, z: f32) -> [Vec3; 4] {
        [
            self.lo,
            Vec2::new(self.hi.x, self.lo.y),
            self.hi,
            Vec2::new(self.lo.x, self.hi.y),
        ]
        .map(|p| p.extend(z))
    }
    fn image_uv(self, xy: Vec2) -> Vec2 {
        let unit = (xy - self.lo) / (self.hi - self.lo);
        Vec2::new(unit.x, 1. - unit.y)
    }
}

const ALONG: [Vec3; 4] = [Vec3::X, Vec3::Y, Vec3::NEG_X, Vec3::NEG_Y];
fn strip(a: &[Vec3; 4], b: &[Vec3; 4], side: usize) -> [Vec3; 4] {
    let j = (side + 1) % 4;
    [a[side], a[j], b[j], b[side]]
}

/// Flat paper and its knife-cut opening use one sheet-space texture chart.
fn sheet_band(g: &mut Geometry, outer: [Vec3; 4], inner: [Vec3; 4]) {
    for side in 0..4 {
        let p = strip(&outer, &inner, side);
        let n = (p[1] - p[0]).cross(p[3] - p[0]).normalize();
        g.vertex_quad(p, [n; 4], p.map(|v| v.xy().to_array()));
    }
}

/// Gallery-wrapped edges reflect the adjacent strip of the same image at its
/// physical scale. Normalized front UVs continue exactly at all four folds.
fn canvas_edges(a: &mut Assembly, label: &str, rect: Rect, back: f32, front: f32, printed: bool) {
    let near = rect.ring(front);
    let far = rect.ring(back);
    let size = rect.hi - rect.lo;
    let g = a.part(
        if printed {
            Surface::Art
        } else {
            Surface::Paper
        },
        label,
    );
    for (side, &along) in ALONG.iter().enumerate() {
        let p = strip(&far, &near, side);
        let inward = Vec3::Z.cross(along);
        let n = -inward;
        let uv = if printed {
            let depth = (front - back).min(size.dot(inward.xy().abs()) * 0.48);
            p.map(|v| {
                let offset = if v.z == back {
                    inward.xy() * depth
                } else {
                    Vec2::ZERO
                };
                rect.image_uv(v.xy() + offset)
                    .clamp(Vec2::ZERO, Vec2::ONE)
                    .to_array()
            })
        } else {
            p.map(|v| [v.dot(along), v.z - back])
        };
        g.vertex_quad(p, [n; 4], uv);
    }
}

pub(in crate::scene::procedural_indoor::objects) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = frame_parameters(o);
    let label = o.kind.class_name();
    let rear = -0.5 * o.size.z;
    let front = 0.5 * o.size.z;
    let picture_z = front - p.picture_inset;
    let outside = Rect {
        lo: Vec2::new(-0.5 * o.size.x, 0.),
        hi: Vec2::new(0.5 * o.size.x, o.size.y),
    };
    let opening = outside.inset(Vec2::splat(p.border_width));
    let mat = p.mat_scale * p.mat_width;
    let image = opening
        .inset(mat + Vec2::splat(p.shadow_gap))
        .shifted(mat * p.picture_bias);
    let backing = if p.style == 4 {
        Surface::Plastic
    } else {
        Surface::Paper
    };
    // The hidden board closes the back across the aperture. The rail stock
    // itself closes the remaining rear perimeter and bears against the wall.
    a.part(backing, label)
        .metric_quad(opening.ring(rear), Vec3::NEG_Z);
    if let Some(section) = p.moulding {
        let landing = picture_z + p.mat_thickness_m - p.float_depth_m;
        moulding::build(a, o, &p, section, outside, landing);
        if p.mat_thickness_m > 0. {
            let bevel = image.inset(Vec2::splat(-p.mat_bevel_m)).ring(landing);
            let board = a.part(Surface::Paper, label);
            sheet_band(board, opening.ring(landing), bevel);
            sheet_band(board, bevel, image.ring(picture_z));
        } else if p.float_depth_m > 0. {
            sheet_band(
                a.part(backing, label),
                opening.ring(landing),
                image.ring(landing),
            );
            canvas_edges(a, label, image, landing, picture_z, p.image_wrap);
        }
    } else {
        canvas_edges(a, label, image, rear, picture_z, p.image_wrap);
    }
    // Use the unmodified front image domain even on a printed wrap.
    face(
        a,
        Surface::Art,
        label,
        image.lo.extend(picture_z),
        image.hi.extend(picture_z),
    );
}
