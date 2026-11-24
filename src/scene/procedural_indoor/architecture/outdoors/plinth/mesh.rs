//! Sweep the stock section along the exposed frontage, with door and unit returns.
use super::*;
use crate::scene::procedural_indoor::{
    envelope::construction::triangle, geometry::Geometry, materials::exterior::label,
};
impl Course {
    pub(in super::super) fn build(
        &self,
        a: &mut Assembly,
        p: &Parcel,
        b: &Building,
        notch: Option<Vec2>,
        doors: &[Option<streets::Entrance>; 2],
    ) {
        let Some(surface) = self.surface() else {
            return;
        };
        let tf = Transform::from_translation(p.center.extend(0.).xzy())
            .with_rotation(Quat::from_rotation_y(p.yaw));
        // Both wings can have doors. Use the apertures actually emitted by the facade.
        let jambs = doors.map(|door| {
            door.map(|d| {
                (
                    (tf.rotation.inverse() * (d.threshold - p.center).extend(0.).xzy()).xz(),
                    d.width * 0.5 + b.frame_width + JAMB_GAP,
                )
            })
        });
        let (corners, count) = perimeter(p.size, notch);
        let corners = &corners[..count];
        let (profile, start) =
            self.section(self.top(b).unwrap(), self.fitted_projection(b, p.size));
        let profile = &profile[start..];
        // A matching render can continue the wall finish. Separate applied stock
        // can take another finish from the same bounded exterior PBR palette.
        let finish_seed = if surface == b.wall && !p.seed.is_multiple_of(3) {
            p.seed ^ 0x534f_434c_4543_4f4c
        } else {
            p.seed
        };
        let g = a.part(surface, &label(surface, "other_structure", finish_seed));
        for i in 0..count {
            let edge = Edge::new(corners, i);
            let door = jambs.iter().flatten().find_map(|&(center, radius)| {
                let delta = center - edge.start;
                let u = delta.dot(edge.along);
                (edge.normal.y > 0.5
                    && delta.dot(edge.normal).abs() < 0.002
                    && (0.0..edge.length).contains(&u))
                .then_some([u - radius, u + radius])
            });
            if let Some([left, right]) = door {
                self.span(g, &tf, &edge, profile, 0., left);
                self.span(g, &tf, &edge, profile, right, edge.length);
            } else {
                self.span(g, &tf, &edge, profile, 0., edge.length);
            }
        }
    }

    fn span(
        &self,
        g: &mut Geometry,
        tf: &Transform,
        edge: &Edge,
        profile: &[Vec2],
        lo: f32,
        hi: f32,
    ) {
        let length = hi - lo;
        if length <= 0. {
            return;
        }
        // Fit whole units between corners and door reveals, leaving no slivers.
        let count = self
            .module
            .map_or(1, |m| (length / m).round().clamp(1., 32.) as usize);
        for unit in 0..count {
            let start = if unit == 0 {
                lo
            } else {
                lo + length * unit as f32 / count as f32 + self.joint * 0.5
            };
            let end = if unit + 1 == count {
                hi
            } else {
                lo + length * (unit + 1) as f32 / count as f32 - self.joint * 0.5
            };
            let point = |at, section: Vec2| tf.transform_point(edge.point(at, section));
            for pair in profile.windows(2) {
                let (lower, upper) = (pair[0], pair[1]);
                let d = upper - lower;
                let outside = (edge.normal * d.y).extend(-d.x).xzy();
                g.metric_quad(
                    [
                        point(start, lower),
                        point(end, lower),
                        point(end, upper),
                        point(start, upper),
                    ],
                    tf.rotation * outside,
                );
                // Close only exposed ends. Adjacent corner faces share vertices
                // and the supporting wall is the back of each expansion joint.
                for (at, normal) in [(start, -edge.along), (end, edge.along)] {
                    if at == 0. || at == edge.length {
                        continue;
                    }
                    let n = tf.rotation * normal.extend(0.).xzy();
                    let back = point(at, Vec2::new(0., lower.y));
                    let front = [point(at, lower), point(at, upper)];
                    if upper.x == 0. {
                        triangle(g, [back, front[0], front[1]], n);
                    } else {
                        g.metric_quad(
                            [back, front[0], front[1], point(at, Vec2::new(0., upper.y))],
                            n,
                        );
                    }
                }
            }
        }
    }
}

pub(super) fn perimeter(size: Vec2, notch: Option<Vec2>) -> ([Vec2; 6], usize) {
    let h = size * 0.5;
    let (sw, se, nw) = (-h, Vec2::new(h.x, -h.y), Vec2::new(-h.x, h.y));
    if let Some(k) = notch {
        ([sw, se, Vec2::new(h.x, k.y), k, Vec2::new(k.x, h.y), nw], 6)
    } else {
        ([sw, se, h, nw, sw, sw], 4)
    }
}

pub(super) struct Edge {
    start: Vec2,
    end: Vec2,
    along: Vec2,
    normal: Vec2,
    mitres: [Vec2; 2],
    pub(super) length: f32,
}

impl Edge {
    pub(super) fn new(corners: &[Vec2], i: usize) -> Self {
        let n = corners.len();
        let start = corners[i];
        let end = corners[(i + 1) % n];
        let outward = |d: Vec2| -d.normalize().perp();
        let along = (end - start).normalize();
        let normal = -along.perp();
        let mitres = [
            outward(start - corners[(i + n - 1) % n]) + normal,
            normal + outward(corners[(i + 2) % n] - end),
        ];
        Self {
            start,
            end,
            along,
            normal,
            mitres,
            length: start.distance(end),
        }
    }

    pub(super) fn point(&self, at: f32, section: Vec2) -> Vec3 {
        let (p, offset) = if at == 0. {
            (self.start, self.mitres[0])
        } else if at == self.length {
            (self.end, self.mitres[1])
        } else {
            (self.start + self.along * at, self.normal)
        };
        (p + offset * section.x).extend(section.y).xzy()
    }
}
