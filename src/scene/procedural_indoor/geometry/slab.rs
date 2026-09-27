use super::*;
impl Geometry {
    /// Closed tabletop/seat with bevelled edges and metric UVs. Outline winds in XZ.
    pub fn profile_slab(&mut self, outline: &[Vec2], height: f32, bevel: f32, tf: Transform) {
        let radius = outline
            .iter()
            .map(|p| p.length())
            .fold(f32::INFINITY, f32::min);
        let bevel = bevel.clamp(0.0001, height * 0.45);
        let inset = 1.0 - bevel / radius.max(bevel * 2.0);
        for (sign, y) in [(-1.0, -height * 0.5), (1.0, height * 0.5)] {
            let center = self.positions.len() as u32;
            self.vertex(Vec3::Y * y, Vec3::Y * sign, Vec2::ZERO, &tf);
            for &p in outline {
                self.vertex(
                    Vec3::new(p.x * inset, y, p.y * inset),
                    Vec3::Y * sign,
                    p * inset,
                    &tf,
                );
            }
            for i in 0..outline.len() as u32 {
                let (a, b) = (center + 1 + i, center + 1 + (i + 1) % outline.len() as u32);
                self.indices.extend(if sign > 0.0 {
                    [center, b, a]
                } else {
                    [center, a, b]
                });
            }
        }
        let rings = [
            (-height * 0.5, inset),
            (-height * 0.5 + bevel, 1.0),
            (height * 0.5 - bevel, 1.0),
            (height * 0.5, inset),
        ];
        let mut along = 0.0;
        for i in 0..outline.len() {
            let a = outline[i];
            let b = outline[(i + 1) % outline.len()];
            let length = a.distance(b);
            for pair in rings.windows(2) {
                let p = Vec3::new(a.x * pair[0].1, pair[0].0, a.y * pair[0].1);
                let q = Vec3::new(a.x * pair[1].1, pair[1].0, a.y * pair[1].1);
                let r = Vec3::new(b.x * pair[1].1, pair[1].0, b.y * pair[1].1);
                let s = Vec3::new(b.x * pair[0].1, pair[0].0, b.y * pair[0].1);
                let n = (q - p).cross(r - p).normalize();
                let start = self.positions.len() as u32;
                for (p, u) in [
                    (p, along),
                    (q, along),
                    (r, along + length),
                    (s, along + length),
                ] {
                    self.vertex(p, n, Vec2::new(u, p.y), &tf);
                }
                self.indices
                    .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
            }
            along += length;
        }
    }
}
