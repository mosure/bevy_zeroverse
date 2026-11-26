use super::*;

/// Bend every face on the same vertical sections. This keeps the thin bezel
/// and the broad rear shoulder apart even near the ends of an ultrawide panel.
pub(super) struct Bend {
    radius: f32,
    transform: Transform,
    cuts: Vec<f32>,
}

impl Bend {
    pub(super) fn new(d: &DisplayGeometry, sections: &[Rim]) -> Self {
        let radius = d.curvature_radius_m;
        let mut cuts = Vec::new();
        if radius > 0. {
            let angle = (d.width_m * 0.5 / radius).asin();
            // Approximate the arc to 0.18 mm in depth, accounting for the
            // steeper silhouette where the screen curves toward the viewer.
            let max_angle = (8. * 0.00018 / radius * angle.cos()).sqrt();
            let count = ((2. * angle / max_angle).ceil() as usize).clamp(2, 64);
            cuts.reserve(count - 1);
            for i in 1..count {
                let x = radius * (angle * (2. * i as f32 / count as f32 - 1.)).sin();
                // A cut grazing a tooling corner produces hairline triangles.
                if sections.iter().all(|r| {
                    (x.abs() - r.half.x).abs() > 0.00015
                        && (x.abs() - (r.half.x - r.corner)).abs() > 0.00015
                }) {
                    cuts.push(x);
                }
            }
        }
        Self {
            radius,
            transform: d.panel_transform,
            cuts,
        }
    }

    pub(super) fn face(&self, g: &mut Geometry, face: &[Vec3], normal: Vec3, display_uv: bool) {
        let u = if normal.x.abs() < 0.8 {
            Vec3::X
        } else {
            Vec3::Z
        };
        let u = (u - normal * normal.dot(u)).normalize();
        let v = normal.cross(u);
        let flat_normal = self.transform.rotation * normal;
        let emit = |g: &mut Geometry, vertices: &[Vec3]| {
            let base = g.positions.len() as u32;
            for &p in vertices {
                let mut point = p;
                let mut n = flat_normal;
                let mut image_u = p.x;
                if self.radius > 0. {
                    let adjacent = (self.radius * self.radius - p.x * p.x).sqrt();
                    point.z += p.x * p.x / (self.radius + adjacent);
                    n = self.transform.rotation
                        * Vec3::new(normal.x - normal.z * p.x / adjacent, normal.y, normal.z)
                            .normalize();
                    if display_uv {
                        image_u = self.radius * (p.x / self.radius).asin();
                    }
                }
                g.positions
                    .push(self.transform.transform_point(point).to_array());
                g.normals.push(n.to_array());
                g.uvs.push(if display_uv {
                    [image_u, -p.y]
                } else {
                    [p.dot(u), p.dot(v)]
                });
            }
            for i in 1..vertices.len() as u32 - 1 {
                g.indices.extend([base, base + i, base + i + 1]);
            }
        };
        if self.cuts.is_empty() {
            emit(g, face);
            return;
        }
        let (min, max) = face
            .iter()
            .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), p| {
                (lo.min(p.x), hi.max(p.x))
            });
        let first = self.cuts.partition_point(|&x| x <= min);
        let last = self.cuts.partition_point(|&x| x < max);
        if first >= last {
            emit(g, face);
            return;
        }
        let mut left = f32::NEG_INFINITY;
        for right in self.cuts[first..last]
            .iter()
            .copied()
            .chain([f32::INFINITY])
        {
            // Clipping a convex octagon by two parallel planes adds at most
            // four vertices. Always intersect the original shared edge.
            let mut polygon = [Vec3::ZERO; 12];
            let mut len = 0;
            let mut push = |p| {
                polygon[len] = p;
                len += 1;
            };
            for (i, &a) in face.iter().enumerate() {
                let b = face[(i + 1) % face.len()];
                if a.x >= left && a.x <= right {
                    push(a);
                }
                let (lo, hi, planes) = if a.x < b.x {
                    (a, b, [left, right])
                } else {
                    (b, a, [right, left])
                };
                for x in planes {
                    if x > lo.x && x < hi.x {
                        push(lo.lerp(hi, (x - lo.x) / (hi.x - lo.x)).with_x(x));
                    }
                }
            }
            emit(g, &polygon[..len]);
            left = right;
        }
    }
}
