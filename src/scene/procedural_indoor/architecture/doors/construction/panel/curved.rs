//! The leaf shoulders, glazing bead, gasket and pane meet on offsets of one
//! contour. Their shape and UVs stay in leaf coordinates throughout articulation.
use super::*;

#[derive(Clone, Copy)]
struct RimPoint {
    xy: Vec2,
    outward: Vec2,
}

impl RimPoint {
    fn inset(self, distance: f32) -> Vec2 {
        self.xy - distance * self.outward
    }
}

struct Contour {
    points: Vec<RimPoint>,
    corners: [Vec2; 4],
    steps: usize,
}

impl Contour {
    fn new(bounds: Rect, radius: f32) -> Self {
        // Limit the chord error to about 0.8 mm on generated stock. Quadrants
        // share the same samples all the way through the thickness of the leaf.
        let steps = (FRAC_PI_2 / (2. * (1. - 0.0008 / radius).clamp(-1., 1.).acos()))
            .ceil()
            .clamp(3., 16.) as usize;
        let quadrant: Vec<_> = (0..=steps)
            .map(|i| match i {
                0 => Vec2::NEG_Y,
                n if n == steps => Vec2::X,
                _ => {
                    let (s, c) = (FRAC_PI_2 * i as f32 / steps as f32).sin_cos();
                    Vec2::new(s, -c)
                }
            })
            .collect();
        let center = bounds.center();
        let half = bounds.size() * 0.5;
        // A metric circle or capsule can lose a fraction of a micron when its
        // bounding rectangle is translated to eye height in single precision.
        let span = (half - radius).map(|v| if v < 1e-6 { 0. } else { v });
        let corners = [
            Vec2::new(1., -1.),
            Vec2::ONE,
            Vec2::new(-1., 1.),
            Vec2::NEG_ONE,
        ];
        let mut points = Vec::with_capacity(4 * (steps + 1));
        for (i, sign) in corners.iter().enumerate() {
            let arc_center = center + *sign * span;
            for &v in &quadrant {
                let outward = match i {
                    0 => v,
                    1 => v.perp(),
                    2 => -v,
                    _ => -v.perp(),
                };
                points.push(RimPoint {
                    xy: arc_center + radius * outward,
                    outward,
                });
            }
        }
        Self {
            points,
            corners: corners.map(|s| center + s * half),
            steps,
        }
    }

    fn shoulders(&self, g: &mut Geometry, tf: Transform, z: f32, side: f32) {
        // Fill only the four corners left by the rectangular machining boundary.
        for (&corner, arc) in self
            .corners
            .iter()
            .zip(self.points.chunks_exact(self.steps + 1))
        {
            let base = g.positions.len() as u32;
            flat_vertex(g, tf, corner, z, side);
            for p in arc {
                flat_vertex(g, tf, p.xy, z, side);
            }
            for i in 0..self.steps as u32 {
                triangle(g, base, base + i + 2, base + i + 1, side);
            }
        }
    }

    fn pane_face(&self, g: &mut Geometry, tf: Transform, xy: Vec2, inset: f32, z: f32, side: f32) {
        let base = g.positions.len() as u32;
        flat_vertex(g, tf, xy, z, side);
        for p in &self.points {
            flat_vertex(g, tf, p.inset(inset), z, side);
        }
        for (i, j) in self.edges() {
            triangle(g, base, base + 1 + i as u32, base + 1 + j as u32, side);
        }
    }

    fn edges(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        (0..self.points.len())
            .map(|i| (i, (i + 1) % self.points.len()))
            // Straight segments disappear at the semicircular ends of a slot
            // or between quadrants of a porthole.
            .filter(|&(i, j)| self.points[i].xy != self.points[j].xy)
    }

    /// Cross section endpoints are (inward inset, local Z). Unwrap the bead
    /// continuously by perimeter distance, with analytic normals at arc joints.
    fn strip(&self, g: &mut Geometry, tf: Transform, from: Vec2, to: Vec2, facing: f32) {
        let base = g.positions.len() as u32;
        let section = to - from;
        let unit = section.normalize() * facing;
        let mid = 0.5 * (from.x + to.x);
        let mut distance = 0.;
        for i in 0..=self.points.len() {
            let p = self.points[i % self.points.len()];
            if i > 0 {
                distance += p.inset(mid).distance(self.points[i - 1].inset(mid));
            }
            let normal = tf.rotation * (p.outward * unit.y).extend(unit.x);
            for (sample, v) in [(from, 0.), (to, section.length())] {
                g.positions.push(
                    tf.transform_point(p.inset(sample.x).extend(sample.y))
                        .to_array(),
                );
                g.normals.push(normal.to_array());
                g.uvs.push([distance, v]);
            }
        }
        for (i, _) in self.edges() {
            let b = base + 2 * i as u32;
            triangle(g, b, b + 2, b + 3, facing);
            triangle(g, b, b + 3, b + 1, facing);
        }
    }
}

fn triangle(g: &mut Geometry, a: u32, b: u32, c: u32, side: f32) {
    g.indices
        .extend(if side > 0. { [a, b, c] } else { [a, c, b] });
}

fn flat_vertex(g: &mut Geometry, tf: Transform, xy: Vec2, z: f32, side: f32) {
    g.positions
        .push(tf.transform_point(xy.extend(z)).to_array());
    g.normals.push((tf.rotation * (Vec3::Z * side)).to_array());
    g.uvs.push([xy.x * side, xy.y]);
}

pub(super) fn build(
    a: &mut Assembly,
    leaf: Leaf,
    hole: Rect,
    v: VisionLight,
    finish: Surface,
    j: &Joinery,
    labels: &Labels,
) {
    let radius = v.radius(hole);
    let rim = Contour::new(hole, radius);
    let tf = leaf.transform;
    let half = 0.5 * leaf.size.z;
    let glass_half = 0.5 * j.glass.metres();
    let bead = j.bead.min(radius * 0.55);
    let clear = bead + j.seal.min(radius * 0.20);
    let pane_edge = clear - 0.001;
    for side in [-1., 1.] {
        let z = j.inset + side * glass_half;
        let rebate = Vec2::new(bead, z + side * 0.0025);
        rim.shoulders(a.part(finish, &labels.leaf), tf, half * side, side);
        let bead_finish = if v.metal_bead { Surface::Metal } else { finish };
        rim.strip(
            a.part(bead_finish, &labels.leaf),
            tf,
            Vec2::new(0., half * side),
            rebate,
            side,
        );
        rim.strip(
            a.part(Surface::Rubber, &labels.seal),
            tf,
            rebate,
            Vec2::new(clear, z),
            side,
        );
        rim.pane_face(
            a.part(Surface::GlassInterior, &labels.glass),
            tf,
            hole.center(),
            pane_edge,
            z,
            side,
        );
    }
    let front = j.inset + glass_half;
    let back = j.inset - glass_half;
    rim.strip(
        a.part(Surface::Rubber, &labels.seal),
        tf,
        Vec2::new(clear, front),
        Vec2::new(clear, back),
        1.,
    );
    rim.strip(
        a.part(Surface::GlassInterior, &labels.glass),
        tf,
        Vec2::new(pane_edge, back),
        Vec2::new(pane_edge, front),
        1.,
    );
}
