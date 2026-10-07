//! Conservative solids from the exact top/support program, without mesh allocation.
use super::*;

enum Solid {
    Box(Vec3, Vec3),
    Capsule(Vec3, Vec3, f32),
}
#[derive(Default)]
pub(crate) struct Clearance {
    outline: Vec<Vec2>,
    bottom: f32,
    top: f32,
    solids: Vec<Solid>,
}
impl Clearance {
    pub(crate) fn new(o: &IndoorObject) -> Self {
        let p = parameters(o);
        let outline = outline(o);
        let mut result = Self {
            bottom: o.size.y - p.top_thickness - 0.012,
            top: o.size.y,
            ..default()
        };
        build_structure(&mut result, o, &p, &outline);
        result.outline = outline;
        result
    }
    /// Axis-aligned box in table-local coordinates. Bounding rotated chair parts
    /// here is conservative; empty spaces between furniture remain available.
    pub(crate) fn hits_box(&self, lo: Vec3, hi: Vec3, margin: f32) -> bool {
        let lo = lo - Vec3::splat(margin);
        let hi = hi + Vec3::splat(margin);
        let slab = hi.y >= self.bottom && lo.y <= self.top && {
            let c = (lo + hi).xz() * 0.5;
            let h = (hi - lo).xz() * 0.5;
            let mut axes = [Vec2::X, Vec2::Y].into_iter().chain(
                self.outline
                    .iter()
                    .enumerate()
                    .map(|(i, a)| (self.outline[(i + 1) % self.outline.len()] - *a).perp()),
            );
            axes.all(|n| {
                let (min, max) = self
                    .outline
                    .iter()
                    .map(|p| p.dot(n))
                    .fold((f32::INFINITY, f32::NEG_INFINITY), |(a, b), v| {
                        (a.min(v), b.max(v))
                    });
                let r = h.dot(n.abs());
                c.dot(n) + r >= min && c.dot(n) - r <= max
            })
        };
        slab || self.solids.iter().any(|s| match *s {
            Solid::Box(a, b) => lo.cmple(b).all() && hi.cmpge(a).all(),
            Solid::Capsule(a, b, r) => super::super::super::layout::segment_hits_box(
                a,
                b,
                lo - Vec3::splat(r),
                hi + Vec3::splat(r),
            ),
        })
    }
    fn slab_intersects(&self, a: Vec3, b: Vec3, radius: f32) -> bool {
        let mut interval = (0.0_f32, 1.0_f32);
        let mut clip = |n: Vec3, d: f32| {
            let distance = d + radius * n.length() - n.dot(a);
            let slope = n.dot(b - a);
            if slope.abs() < 1e-8 {
                return distance >= 0.;
            }
            let t = distance / slope;
            if slope > 0. {
                interval.1 = interval.1.min(t);
            } else {
                interval.0 = interval.0.max(t);
            }
            interval.0 <= interval.1
        };
        clip(Vec3::Y, self.top)
            && clip(Vec3::NEG_Y, -self.bottom)
            && self.outline.iter().enumerate().all(|(i, p)| {
                let e = self.outline[(i + 1) % self.outline.len()] - *p;
                let n = Vec3::new(e.y, 0., -e.x);
                clip(n, n.x * p.x + n.z * p.y)
            })
    }
    fn point_slab_distance_squared(&self, p: Vec3) -> f32 {
        let mut inside = true;
        let mut distance = f32::INFINITY;
        for (i, &a) in self.outline.iter().enumerate() {
            let b = self.outline[(i + 1) % self.outline.len()];
            let e = b - a;
            inside &= e.perp_dot(p.xz() - a) >= 0.;
            let q = a + e * ((p.xz() - a).dot(e) / e.length_squared().max(1e-12)).clamp(0., 1.);
            distance = distance.min(q.distance_squared(p.xz()));
        }
        (p.y - p.y.clamp(self.bottom, self.top)).powi(2) + if inside { 0. } else { distance }
    }
    pub(crate) fn hits_capsule(&self, a: Vec3, b: Vec3, radius: f32) -> bool {
        let r2 = radius * radius;
        // Expanded planes are only a broad phase. Exact segment/edge distance
        // avoids treating the rounded capsule at a tabletop edge as a cube.
        let slab = self.slab_intersects(a, b, radius)
            && (self.slab_intersects(a, b, 0.)
                || self.point_slab_distance_squared(a) <= r2
                || self.point_slab_distance_squared(b) <= r2
                || self.outline.iter().enumerate().any(|(i, p)| {
                    let q = self.outline[(i + 1) % self.outline.len()];
                    let lo = Vec3::new(p.x, self.bottom, p.y);
                    let hi = lo.with_y(self.top);
                    let end = Vec3::new(q.x, self.bottom, q.y);
                    [(lo, hi), (lo, end), (hi, end.with_y(self.top))]
                        .into_iter()
                        .any(|(c, d)| segment_distance_squared(a, b, c, d) <= r2)
                }));
        slab || self.hits_structure_capsule(a, b, radius)
    }
    pub(crate) fn hits_structure_capsule(&self, a: Vec3, b: Vec3, radius: f32) -> bool {
        self.solids.iter().any(|s| match *s {
            Solid::Box(lo, hi) => capsule_box(a, b, radius, lo, hi),
            Solid::Capsule(c, d, r) => segment_distance_squared(a, b, c, d) <= (r + radius).powi(2),
        })
    }
}
impl StructureSink for Clearance {
    type Part = Self;
    fn part(&mut self, _: Surface, _: &str) -> &mut Self {
        self
    }
}
impl PartSink for Clearance {
    fn cuboid(&mut self, size: Vec3, _: f32, tf: Transform) {
        let mut lo = Vec3::splat(f32::INFINITY);
        let mut hi = Vec3::splat(f32::NEG_INFINITY);
        for x in [-0.5, 0.5] {
            for y in [-0.5, 0.5] {
                for z in [-0.5, 0.5] {
                    let p = tf.transform_point(size * Vec3::new(x, y, z));
                    lo = lo.min(p);
                    hi = hi.max(p);
                }
            }
        }
        self.solids.push(Solid::Box(lo, hi));
    }
    fn cylinder(&mut self, r: f32, h: f32, tf: Transform) {
        self.cuboid(Vec3::new(2. * r, h, 2. * r), 0., tf);
    }
    fn lathe(&mut self, profile: &[(f32, f32)], _: u32, tf: Transform) {
        let r = profile.iter().map(|p| p.0).fold(0., f32::max);
        let lo = profile.iter().map(|p| p.1).fold(f32::INFINITY, f32::min);
        let hi = profile
            .iter()
            .map(|p| p.1)
            .fold(f32::NEG_INFINITY, f32::max);
        self.cuboid(
            Vec3::new(2. * r, hi - lo, 2. * r),
            0.,
            tf.with_translation(tf.transform_point(Vec3::Y * ((hi + lo) * 0.5))),
        );
    }
    fn rod(&mut self, a: Vec3, b: Vec3, r: f32) {
        self.solids.push(Solid::Capsule(a, b, r));
    }
    fn tube(&mut self, path: &[Vec3], r: f32, _: u32) {
        for p in path.windows(2) {
            self.rod(p[0], p[1], r);
        }
    }
}

fn segment_distance_squared(p: Vec3, q: Vec3, a: Vec3, b: Vec3) -> f32 {
    let u = q - p;
    let v = b - a;
    let w = p - a;
    let uu = u.length_squared();
    let vv = v.length_squared();
    let uv = u.dot(v);
    let uw = u.dot(w);
    let vw = v.dot(w);
    if uu < 1e-12 && vv < 1e-12 {
        return w.length_squared();
    }
    let mut s = if uu < 1e-12 {
        0.
    } else if vv < 1e-12 {
        (-uw / uu).clamp(0., 1.)
    } else {
        let d = uu * vv - uv * uv;
        if d > 1e-12 {
            ((uv * vw - vv * uw) / d).clamp(0., 1.)
        } else {
            0.
        }
    };
    let mut t = if vv > 1e-12 { (uv * s + vw) / vv } else { 0. };
    if t < 0. {
        t = 0.;
        s = if uu > 1e-12 {
            (-uw / uu).clamp(0., 1.)
        } else {
            0.
        };
    } else if t > 1. {
        t = 1.;
        s = if uu > 1e-12 {
            ((uv - uw) / uu).clamp(0., 1.)
        } else {
            0.
        };
    }
    (w + s * u - t * v).length_squared()
}

fn capsule_box(a: Vec3, b: Vec3, r: f32, lo: Vec3, hi: Vec3) -> bool {
    use super::super::super::layout::segment_hits_box;
    if !segment_hits_box(a, b, lo - Vec3::splat(r), hi + Vec3::splat(r)) {
        return false;
    }
    if segment_hits_box(a, b, lo, hi)
        || a.distance_squared(a.clamp(lo, hi)) <= r * r
        || b.distance_squared(b.clamp(lo, hi)) <= r * r
    {
        return true;
    }
    for axis in 0..3 {
        for u in [false, true] {
            for v in [false, true] {
                let mut c = lo;
                c[(axis + 1) % 3] = if u {
                    hi[(axis + 1) % 3]
                } else {
                    lo[(axis + 1) % 3]
                };
                c[(axis + 2) % 3] = if v {
                    hi[(axis + 2) % 3]
                } else {
                    lo[(axis + 2) % 3]
                };
                let mut d = c;
                d[axis] = hi[axis];
                if segment_distance_squared(a, b, c, d) <= r * r {
                    return true;
                }
            }
        }
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rounded_limb_can_clear_a_top_edge_without_waiving_intersection() {
        let c = Clearance {
            outline: vec![
                Vec2::new(-1., -1.),
                Vec2::new(1., -1.),
                Vec2::new(1., 1.),
                Vec2::new(-1., 1.),
            ],
            bottom: 0.7,
            top: 0.75,
            solids: vec![],
        };
        let p = Vec3::new(1.06, 0.64, 0.);
        assert!(!c.hits_capsule(p, p, 0.07));
        let p = Vec3::new(1.04, 0.66, 0.);
        assert!(c.hits_capsule(p, p, 0.07));
        assert!(c.hits_capsule(Vec3::new(-2., 0.72, 0.), Vec3::new(2., 0.72, 0.), 0.01));
        assert!(c.hits_capsule(Vec3::new(-2., 0.69, 0.), Vec3::new(2., 0.69, 0.), 0.011));
        assert!(!c.hits_capsule(Vec3::new(-2., 0.68, 0.), Vec3::new(2., 0.68, 0.), 0.019));
    }
}
