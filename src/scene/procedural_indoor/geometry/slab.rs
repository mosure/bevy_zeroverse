use super::*;
impl Geometry {
    /// Closed tabletop/seat with bevelled edges and metric UVs. Outline winds in XZ.
    pub fn profile_slab(&mut self, outline: &[Vec2], height: f32, bevel: f32, tf: Transform) {
        self.profile_slab_with_top(outline, height, bevel, tf, |g, outline, inset, y, tf| {
            g.slab_cap(outline, inset, Vec2::ZERO, y, tf, |v| v);
        });
    }

    /// A chart seam can split the flat cap without splitting the solid, its
    /// rounded perimeter, or the support surface used for object placement.
    pub(crate) fn profile_slab_with_top(
        &mut self,
        outline: &[Vec2],
        height: f32,
        bevel: f32,
        tf: Transform,
        top: impl FnOnce(&mut Self, &[Vec2], f32, f32, &Transform),
    ) {
        let profile = Self::slab_profile(outline, height, bevel);
        self.slab_with_profile(outline, height, profile, tf, top);
    }

    pub(crate) fn slab_with_profile(
        &mut self,
        outline: &[Vec2],
        height: f32,
        profile: (f32, f32),
        tf: Transform,
        top: impl FnOnce(&mut Self, &[Vec2], f32, f32, &Transform),
    ) {
        self.slab_cap(outline, profile.1, Vec2::ZERO, -height * 0.5, &tf, |v| v);
        top(self, outline, profile.1, height * 0.5, &tf);
        self.slab_edge(outline, height, profile, &tf);
    }

    pub(crate) fn slab_profile(outline: &[Vec2], height: f32, bevel: f32) -> (f32, f32) {
        let radius = outline
            .iter()
            .map(|p| p.length())
            .fold(f32::INFINITY, f32::min);
        // Submillimetre stamped stock can be thinner than the usual bevel
        // floor. Its thickness always limits the rounding on each side.
        let bevel = bevel.max(0.0001).min(height * 0.45);
        (bevel, 1.0 - bevel / radius.max(bevel * 2.0))
    }

    pub(crate) fn slab_edge(
        &mut self,
        outline: &[Vec2],
        height: f32,
        profile: (f32, f32),
        tf: &Transform,
    ) {
        self.slab_edge_with_uv(outline, height, profile, tf, |_, along, rings| {
            rings.map(|(y, _)| Vec2::new(along, y))
        });
    }

    /// Chart each perimeter vertex at the four profile rings, bottom to top.
    /// The closing corner gets the total arc length for strips with a UV seam.
    pub(crate) fn slab_edge_with_uv(
        &mut self,
        outline: &[Vec2],
        height: f32,
        profile: (f32, f32),
        tf: &Transform,
        chart: impl Fn(usize, f32, &[(f32, f32); 4]) -> [Vec2; 4],
    ) {
        self.slab_edge_sections(outline, height, profile, tf, chart, |_| false);
    }

    /// Selected edges supply their own pierced vertical face between the two
    /// bevels. The surrounding rings and their original UV charts stay intact.
    pub(crate) fn slab_edge_sections(
        &mut self,
        outline: &[Vec2],
        height: f32,
        (bevel, inset): (f32, f32),
        tf: &Transform,
        chart: impl Fn(usize, f32, &[(f32, f32); 4]) -> [Vec2; 4],
        pierced: impl Fn(usize) -> bool,
    ) {
        let rings = [
            (-height * 0.5, inset),
            (-height * 0.5 + bevel, 1.0),
            (height * 0.5 - bevel, 1.0),
            (height * 0.5, inset),
        ];
        let mut along = 0.0;
        if outline.is_empty() {
            return;
        }
        let mut a_uv = chart(0, along, &rings);
        for i in 0..outline.len() {
            let a = outline[i];
            let next = (i + 1) % outline.len();
            let b = outline[next];
            let length = a.distance(b);
            let b_uv = chart(next, along + length, &rings);
            for (ring, pair) in rings.windows(2).enumerate() {
                if ring == 1 && pierced(i) {
                    continue;
                }
                let p = Vec3::new(a.x * pair[0].1, pair[0].0, a.y * pair[0].1);
                let q = Vec3::new(a.x * pair[1].1, pair[1].0, a.y * pair[1].1);
                let r = Vec3::new(b.x * pair[1].1, pair[1].0, b.y * pair[1].1);
                let s = Vec3::new(b.x * pair[0].1, pair[0].0, b.y * pair[0].1);
                let n = (q - p).cross(r - p).normalize();
                let start = self.positions.len() as u32;
                for (p, uv) in [
                    (p, a_uv[ring]),
                    (q, a_uv[ring + 1]),
                    (r, b_uv[ring + 1]),
                    (s, b_uv[ring]),
                ] {
                    self.vertex(p, n, uv, tf);
                }
                self.indices
                    .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
            }
            along += length;
            a_uv = b_uv;
        }
    }

    /// Fan a convex cut around a point inside that cut, in local slab space.
    pub(crate) fn slab_cap(
        &mut self,
        outline: &[Vec2],
        scale: f32,
        center: Vec2,
        y: f32,
        tf: &Transform,
        uv: impl Fn(Vec2) -> Vec2,
    ) {
        let start = self.positions.len() as u32;
        let normal = Vec3::Y * y.signum();
        for p in std::iter::once(center).chain(outline.iter().map(|p| *p * scale)) {
            self.vertex(Vec3::new(p.x, y, p.y), normal, uv(p), tf);
        }
        for i in 0..outline.len() as u32 {
            let a = start + 1 + i;
            let b = start + 1 + (i + 1) % outline.len() as u32;
            self.indices
                .extend(if y > 0. { [start, b, a] } else { [start, a, b] });
        }
    }
}
