use super::*;
pub(super) const WEB: f32 = 0.0025;
const CLASS: &str = "tray_return";

impl TrayStation {
    pub(super) fn frame(&self, a: &mut Assembly, o: &IndoorObject, case: &CaseProgram) {
        let s = o.size;
        let t = self.deck_m;
        let post = self.uprights_m;
        let base = self.toe_height_m;
        // The feet carry the entire bottom deck. All panels and uprights bear
        // on that deck and terminate at the underside of the worktop.
        feet(a, o, case, base);
        for top in [base + t, s.y] {
            a.box_part(
                case.finish,
                CLASS,
                Vec3::Y * (top - t * 0.5),
                Vec3::new(s.x, t, s.z),
                t * 0.18,
            );
        }
        let bottom = base + t;
        let height = s.y - t - bottom;
        let mid = bottom + height * 0.5;
        if self.enclosed {
            for side in [-1., 1.] {
                a.box_part(
                    case.finish,
                    CLASS,
                    Vec3::new(side * (s.x - post) * 0.5, mid, 0.),
                    Vec3::new(post, height, s.z),
                    0.002,
                );
            }
            a.box_part(
                case.finish,
                CLASS,
                Vec3::new(0., mid, -(s.z - post) * 0.5),
                Vec3::new(s.x - 2. * post, height, post),
                0.002,
            );
        } else if self.bearing == Bearing::Shelf {
            for side in [-1., 1.] {
                self.posts(a, s, case.frame, side * (s.x - post) * 0.5, mid, height);
            }
        }
        match self.bearing {
            Bearing::Shelf => self.shelves(a, s, case.frame),
            Bearing::RimGuides => {
                let edge = self.ware.size_m.x * 0.5 + self.guide_clearance_m;
                for &bay in &self.bay_centers_m {
                    for side in [-1., 1.] {
                        self.posts(
                            a,
                            s,
                            Surface::Metal,
                            bay + side * (edge + WEB + post * 0.5),
                            mid,
                            height,
                        );
                        for &y in &self.bearings_m {
                            self.runner(a, s, Vec3::new(bay + side * edge, y, 0.), side);
                        }
                    }
                }
            }
        }
    }

    fn posts(&self, a: &mut Assembly, s: Vec3, stock: Surface, x: f32, mid: f32, height: f32) {
        let t = self.uprights_m;
        for side in [-1., 1.] {
            // The back stop/panel can sit behind both guide posts.
            let z = side * (s.z * 0.5 - t * 1.55);
            a.box_part(
                stock,
                CLASS,
                Vec3::new(x, mid, z),
                Vec3::new(t, height, t),
                0.0015,
            );
        }
    }

    fn shelves(&self, a: &mut Assembly, s: Vec3, frame: Surface) {
        let post = self.uprights_m;
        let t = self.deck_m;
        let half = s.x * 0.5 - post;
        let depth = s.z - 2. * post - 0.006;
        for &y in self.bearings_m.iter().skip(1) {
            // Both side cleats reach the frame and bear the shelf on a flat face.
            let ledge = (post * 0.88).min(0.022);
            for side in [-1., 1.] {
                a.box_part(
                    frame,
                    CLASS,
                    Vec3::new(side * (half - ledge * 0.5), y - t - ledge * 0.5, 0.),
                    Vec3::new(ledge, ledge, depth),
                    0.001,
                );
            }
            a.box_part(
                frame,
                CLASS,
                Vec3::Y * (y - t * 0.5),
                Vec3::new(2. * half - 0.003, t, depth),
                0.001,
            );
        }
    }

    pub(super) fn runner(&self, a: &mut Assembly, s: Vec3, edge: Vec3, side: f32) {
        let half_depth = (s.z - 2. * self.uprights_m - 0.004) * 0.5;
        let height = self.uprights_m * 0.72 - WEB;
        let flange = self.guide_clearance_m + self.ware.lip_m * 0.56;
        // Cross-section of a single bent sheet. Each station gives its inside
        // face and the outward direction through the gauge. The flat flange
        // sits at the reported bearing height and the web reaches the posts.
        let radius = WEB * 0.75;
        let diagonal = std::f32::consts::FRAC_1_SQRT_2;
        let profile = [
            (Vec2::new(-flange, 0.), Vec2::NEG_Y),
            (Vec2::new(-radius, 0.), Vec2::NEG_Y),
            (
                Vec2::new(-radius, radius) * (1. - diagonal),
                Vec2::new(diagonal, -diagonal),
            ),
            (Vec2::new(0., radius), Vec2::X),
            (Vec2::new(0., height), Vec2::X),
        ];
        let point = |p: Vec2, z: f32| edge + Vec3::new(side * p.x, p.y, z);
        let strip = profile.map(|(inside, normal)| {
            let outside = inside + normal * WEB;
            [
                point(inside, -half_depth),
                point(inside, half_depth),
                point(outside, half_depth),
                point(outside, -half_depth),
            ]
        });
        let g = a.part(Surface::Metal, CLASS);
        for pair in strip.windows(2) {
            let [p, q] = [pair[0], pair[1]];
            let tangent = q[0] - p[0];
            let inside = Vec3::new(-side * tangent.y, side * tangent.x, 0.);
            g.metric_quad([p[0], q[0], q[1], p[1]], inside);
            g.metric_quad([p[3], p[2], q[2], q[3]], -inside);
            g.metric_quad([p[1], q[1], q[2], p[2]], Vec3::Z);
            g.metric_quad([p[0], p[3], q[3], q[0]], Vec3::NEG_Z);
        }
        g.metric_quad(strip[0], -side * Vec3::X);
        g.metric_quad(strip[strip.len() - 1], Vec3::Y);
    }
}
