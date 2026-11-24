use super::*;
use crate::scene::procedural_indoor::materials::art::{Moulding, MouldingShape};

/// The four stock ends meet on the exact same mitre at every section station.
/// U follows the stock profile in metres and V runs along the individual rail.
struct Rails<'a> {
    g: &'a mut Geometry,
    outer: Rect,
    at: Vec2, // inset and Z
    ring: [Vec3; 4],
    across: f32,
    origins: [Vec2; 4],
}
impl Rails<'_> {
    fn to(&mut self, inset: f32, z: f32, smooth: Option<[Vec2; 2]>) {
        let to = Vec2::new(inset, z);
        let delta = to - self.at;
        let length = delta.length();
        let next = self.outer.inset(Vec2::splat(inset)).ring(z);
        let flat = Vec2::new(-delta.y, delta.x) / length;
        let normals = smooth.unwrap_or([flat; 2]);
        for (i, &along) in ALONG.iter().enumerate() {
            let points = strip(&self.ring, &next, i);
            let inward = Vec3::Z.cross(along);
            let [a, b] = normals.map(|n| inward * n.x + Vec3::Z * n.y);
            let uv = std::array::from_fn(|j| {
                let u = if j < 2 {
                    self.across
                } else {
                    self.across + length
                };
                (self.origins[i] + Vec2::new(u, -points[j].dot(along))).to_array()
            });
            self.g.vertex_quad(points, [a, a, b, b], uv);
        }
        self.across += length;
        self.at = to;
        self.ring = next;
    }
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &FrameProgram,
    section: Moulding,
    outer: Rect,
    landing: f32,
) {
    let surface = match p.style {
        1 | 4 => Surface::Wood,
        2 => Surface::Plastic,
        _ => Surface::Metal,
    };
    let mut cuts = TimberCuts::new(o.seed, 0x4652_414d_4501);
    let origins = std::array::from_fn(|_| cuts.origin());
    let half = 0.5 * o.size.z;
    let width = p.border_width;
    let mut rails = Rails {
        g: a.part(surface, o.kind.class_name()),
        outer,
        at: Vec2::new(width, -half),
        ring: outer.inset(Vec2::splat(width)).ring(-half),
        across: 0.,
        origins,
    };
    rails.to(0., -half, None);
    rails.to(0., half - section.arris_m, None);
    rails.to(section.arris_m, half, None);
    let shoulder = width * p.outer_fraction * if p.stepped { 1. } else { 0.68 };
    rails.to(shoulder, half, None);
    let step = if p.stepped {
        section.rise_m * (0.24 + 0.48 * p.recess_fraction)
    } else {
        0.
    };
    let start = half - step;
    if p.stepped {
        rails.to(shoulder, start, None);
    }
    let lip = half - section.rise_m;
    let run = width - shoulder;
    match section.shape {
        MouldingShape::Flat => {
            rails.to(width - 0.28 * run, start, None);
            rails.to(width, lip, None);
        }
        MouldingShape::Bevel => rails.to(width, lip, None),
        shape => {
            let drop = start - lip;
            let power = section.curvature;
            let slope = |t: f32| {
                let t = if shape == MouldingShape::Cove {
                    1. - t
                } else {
                    t
                };
                Vec2::new(drop / run * power * t.powf(power - 1.), 1.).normalize()
            };
            let mut prev = slope(0.);
            for i in 1..=4 {
                let t = i as f32 * 0.25;
                let f = if shape == MouldingShape::Cove {
                    1. - (1. - t).powf(power)
                } else {
                    t.powf(power)
                };
                let n = slope(t);
                // Snap the final station to the same physical aperture as the mat.
                let (inset, z) = if i == 4 {
                    (width, lip)
                } else {
                    (shoulder + run * t, start - drop * f)
                };
                rails.to(inset, z, Some([prev, n]));
                prev = n;
            }
        }
    }
    rails.to(width, landing, None);
}
