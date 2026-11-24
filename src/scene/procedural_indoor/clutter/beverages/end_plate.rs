//! Stamped circular or oval sheet around one offset perforation. The bore and both
//! faces share a cut boundary. Callers can keep the punched metal as a hinged flap.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(super) struct PiercedPlate {
    pub points: Vec<(Vec2, Vec2)>,
    offset: Vec2,
    half: Vec2,
}
impl PiercedPlate {
    pub fn new(outer: Vec2, offset: Vec2, half: Vec2) -> Self {
        // Project each outer vertex toward the centre of the port. The annular
        // triangles stay inside their own ray sectors even for an off-centre ellipse.
        // Include the two ends of the long axis on the straight outer polygon edges.
        let mut corners = Vec::with_capacity(34);
        let mut last = Vec2::X * outer.x;
        for (point, _, _) in outline(32).meridians().skip(1) {
            let end = point * outer;
            corners.push(last);
            if (last.y < offset.y) != (end.y < offset.y) {
                let u = (offset.y - last.y) / (end.y - last.y);
                if (0.045..0.955).contains(&u) {
                    corners.push(last.lerp(end, u));
                }
            }
            last = end;
        }
        let points = corners
            .into_iter()
            .map(|corner| {
                let ray = corner - offset;
                (corner, offset + ray / (ray / half).length())
            })
            .collect();
        Self {
            points,
            offset,
            half,
        }
    }
    pub fn faces(&self, g: &mut Geometry, top: f32, bottom: f32, tf: Transform) {
        let start = g.positions.len() as u32;
        let n = self.points.len() as u32;
        let mut previous = None;
        let mut along = 0.;
        for &(corner, inner) in self.points.iter().chain(self.points.first()) {
            if let Some(last) = previous {
                along += inner.distance(last);
            }
            previous = Some(inner);
            let normal = ((self.offset - inner) / (self.half * self.half)).normalize();
            for (p, y, face) in [
                (corner, top, Vec3::Y),
                (inner, top, Vec3::Y),
                (inner, top, Vec3::new(normal.x, 0., normal.y)),
                (inner, bottom, Vec3::new(normal.x, 0., normal.y)),
                (inner, bottom, Vec3::NEG_Y),
                (corner, bottom, Vec3::NEG_Y),
            ] {
                g.positions
                    .push(tf.transform_point(Vec3::new(p.x, y, p.y)).to_array());
                g.normals.push((tf.rotation * face).to_array());
                // The bore has its own metric chart so its gauge retains a tangent.
                g.uvs.push(if face.y == 0. {
                    [along, y - bottom]
                } else {
                    p.to_array()
                });
            }
        }
        for j in 0..n {
            let a = start + j * 6;
            let b = start + (j + 1) * 6;
            for k in [0, 2, 4] {
                g.indices
                    .extend([a + k, a + k + 1, b + k, b + k, a + k + 1, b + k + 1]);
            }
        }
    }
    /// A rolled flange and a recessed finger edge give the stamped ring a reflective
    /// silhouette even when the tab lies flat against a sealed end.
    pub fn embossed_tab(&self, g: &mut Geometry, outer_half: Vec2, gauge: f32, tf: Transform) {
        let start = g.positions.len() as u32;
        let mut previous = self.points[0].0;
        let mut along = 0.;
        for &(outer, inner) in self.points.iter().chain(self.points.first()) {
            along += outer.distance(previous);
            previous = outer;
            let outside = (outer / (outer_half * outer_half)).normalize();
            let bore = ((self.offset - inner) / (self.half * self.half)).normalize();
            let outward = Vec3::new(outside.x, 0., outside.y);
            let inward = Vec3::new(bore.x, 0., bore.y);
            let lip = self.offset + (inner - self.offset) * 1.17;
            let section = [
                (outer * 0.89, 0., Vec3::NEG_Y),
                (outer, gauge * 0.30, outward),
                (outer * 0.95, gauge * 0.86, outward * 0.65 + Vec3::Y),
                (outer * 0.86, gauge, Vec3::Y),
                (lip, gauge, Vec3::Y),
                (inner, gauge * 0.48, inward),
                (inner, gauge * 0.18, inward),
                (lip, 0., Vec3::NEG_Y),
                (outer * 0.89, 0., Vec3::NEG_Y),
            ];
            let mut previous_p = Vec3::new(section[0].0.x, 0., section[0].0.y);
            let mut across = 0.;
            for (p, y, n) in section {
                let local = Vec3::new(p.x, y, p.y);
                across += local.distance(previous_p);
                previous_p = local;
                g.positions.push(tf.transform_point(local).to_array());
                g.normals.push((tf.rotation * n.normalize()).to_array());
                g.uvs.push([along, across]);
            }
        }
        for j in 0..self.points.len() as u32 {
            let a = start + j * 9;
            for k in 0..8 {
                let i = a + k;
                g.indices.extend([i, i + 1, i + 9, i + 9, i + 1, i + 10]);
            }
        }
    }
}
