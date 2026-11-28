use super::*;
use crate::scene::procedural_indoor::envelope::{construction::triangle, polygon};

impl FeaturedBook {
    pub(super) fn build_rest(&self, a: &mut Assembly) {
        let frame = self.frame();
        let at = |x, up, out| frame.transform_point(Vec3::new(x, up, out));
        let shelf = self.back_foot_m - Vec3::Y * self.lift_m;
        let yaw = self.yaw();
        let deck = |x, y, z| shelf + yaw * Vec3::new(x, y, z);
        let heel = heel(self.cover_height_m, self.recline_rad.sin());
        let t = self.thickness_m;
        match self.rest {
            BookRest::BentWire => {
                let g = a.part(Surface::Metal, "bookshelf");
                let r = (t * 0.12).clamp(0.0025, 0.0038);
                let half = self.cover_width_m * 0.31;
                let high = self.cover_height_m * 0.63;
                for sign in [-1., 1.] {
                    let x = sign * half;
                    // The hooks bear the lower edge and back cover. A separate
                    // pair of folding legs pivots on the crossbar just outboard
                    // of the hooks, as on a wire display easel.
                    g.planar_tube(
                        &[
                            at(x, 0.018, t + r),
                            at(x, -r, t + r),
                            at(x, -r, t * 0.5),
                            at(x, -r, -r),
                            at(x, high * 0.73, -r),
                            at(x, high, -r),
                        ],
                        r,
                        8,
                        yaw * Vec3::X,
                    );
                    let outer = sign * (half + 2.05 * r);
                    let pivot = at(outer, high, -r);
                    let toe = deck(outer, r, t * 0.66);
                    let bend = deck(outer, r, -heel + r);
                    let run = 4. * r;
                    g.planar_tube(
                        &[
                            pivot,
                            bend + (pivot - bend).normalize() * run,
                            bend,
                            bend + (toe - bend).normalize() * run,
                            toe,
                        ],
                        r,
                        8,
                        yaw * Vec3::X,
                    );
                }
                let axle = half + 2.3 * r;
                g.tube(&[at(-axle, high, -r), at(axle, high, -r)], r, 8);
            }
            BookRest::GroovedTimber => {
                let g = a.part(Surface::Wood, "bookshelf");
                let half = self.cover_width_m * 0.365;
                // Section along the grain-extruded block. Only the crown of the
                // groove and its rear ridge bear on the book. Broad faces of
                // the block never coincide with the binding.
                let (sin, cos) = self.recline_rad.sin_cos();
                let cut = |up: f32, out: f32| {
                    Vec2::new(out * cos - up * sin, self.lift_m + up * cos + out * sin)
                };
                let back = self.cover_height_m * 0.18;
                let outline = [
                    Vec2::new(-heel, 0.),
                    Vec2::new(t + 0.020, 0.),
                    cut(0.017, t + 0.015),
                    cut(0.019, t + 0.001),
                    cut(-0.0025, t + 0.0025),
                    cut(0., t * 0.40),
                    cut(-0.0025, -0.0025),
                    cut(back, 0.),
                    cut(back + 0.002, -0.010),
                ];
                let point = |x: f32, p: Vec2| deck(x, p.y, p.x);
                for tri in polygon::triangles(&outline) {
                    for sign in [-1., 1.] {
                        triangle(g, tri.map(|p| point(sign * half, p)), yaw * Vec3::X * sign);
                    }
                }
                for (p, q) in polygon::edges(&outline) {
                    let d = q - p;
                    g.stock_quad(
                        [
                            point(-half, p),
                            point(half, p),
                            point(half, q),
                            point(-half, q),
                        ],
                        yaw * Vec3::new(0., -d.x, d.y),
                        yaw * Vec3::X,
                        Some(self.back_foot_m.xz()),
                    );
                }
            }
        }
    }
}
