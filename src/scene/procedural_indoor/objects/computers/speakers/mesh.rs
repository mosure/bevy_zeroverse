//! Cut the existing deck panel and close the new boundary with a backed insert.
use super::*;

impl Opening {
    fn outline(self, design: &SpeakerBank, inset: f32) -> [Vec2; 12] {
        let half = self.size_xz_m * 0.5;
        let radius = self.size_xz_m.min_element() * design.edge_radius_fraction;
        palmrest::rounded_outline(
            half - inset,
            (radius - inset).max(radius * 0.35),
            self.deck_center_xz_m,
        )
    }

    fn bevel(self, design: &SpeakerBank) -> f32 {
        (design.recess_m * 0.63).min(self.size_xz_m.min_element() * 0.10)
    }

    pub(in super::super) fn cut(
        &self,
        g: &mut Geometry,
        outer: [Vec2; 4],
        design: &SpeakerBank,
        tf: &Transform,
        y: f32,
    ) {
        let c = self.deck_center_xz_m;
        let h = self.size_xz_m * 0.5 + COLLAR;
        let rect = [
            c + h,
            c + Vec2::new(-h.x, h.y),
            c - h,
            c + Vec2::new(h.x, -h.y),
        ];
        let lip = self.outline(design, 0.);
        let facing = self.outline(design, self.bevel(design));
        let first_corner = match self.panel {
            5 => 1,
            8 => 2,
            11 => 3,
            _ => unreachable!(),
        };
        let base = g.positions.len() as u32;
        let at = |point: Vec2, h| tf.transform_point(Vec3::new(point.x, h, point.y));
        // Keep the planar chart used by the original metric deck quad.
        let u = (outer[2] - outer[3]).normalize();
        let v = Vec2::new(u.y, -u.x);
        for point in outer.iter().chain(&rect).chain(&lip) {
            let world = at(*point, y);
            g.positions.push(world.to_array());
            g.normals.push(Vec3::Y.to_array());
            g.uvs.push([world.xz().dot(u), world.xz().dot(v)]);
        }
        let mut tri = |a: usize, b: usize, c: usize| {
            g.indices
                .extend([base + a as u32, base + c as u32, base + b as u32])
        };
        for i in 0..4 {
            let j = (i + 1) % 4;
            let k = (first_corner + i) % 4;
            let l = (k + 1) % 4;
            let (a, b, c, d) = (outer[i], outer[j], rect[l], rect[k]);
            // The short end of the touchpad can skew this panel far enough
            // to make one of its two possible diagonals exterior.
            if (b - a).perp_dot(c - a) > 0. && (c - a).perp_dot(d - a) > 0. {
                tri(i, j, 4 + l);
                tri(i, 4 + l, 4 + k);
            } else {
                tri(i, j, 4 + k);
                tri(j, 4 + l, 4 + k);
            }
            let arc = 8 + 3 * k;
            let next = 8 + 3 * l;
            tri(4 + k, arc + 1, arc);
            tri(4 + k, arc + 2, arc + 1);
            tri(4 + k, 4 + l, next);
            tri(4 + k, next, arc + 2);
        }
        for i in 0..lip.len() {
            let j = (i + 1) % lip.len();
            g.metric_quad(
                [
                    at(lip[i], y),
                    at(lip[j], y),
                    at(facing[j], y - design.recess_m),
                    at(facing[i], y - design.recess_m),
                ],
                Vec3::Y,
            );
        }
    }
}

impl SpeakerBank {
    pub(in super::super) fn insert_faces(&self, a: &mut Assembly, tf: &Transform, y: f32) {
        let g = a.part(self.facing, self.label());
        for cut in self.openings() {
            let plate = cut.outline(self, cut.bevel(self));
            g.slab_cap(
                &plate,
                1.,
                cut.deck_center_xz_m,
                y - self.recess_m,
                tf,
                |p| (p - cut.deck_center_xz_m) * self.texture_scale,
            );
        }
    }
}
