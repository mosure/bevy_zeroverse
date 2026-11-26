//! One open roof with four covered walks. The mesh, light meter and mounting
//! checks use the same metric hole, including the thickness of its reveal.
use super::{ColumnProfile, Pillar};
use crate::scene::procedural_indoor::{
    geometry::Geometry,
    layout::{segment_hits_box, stream, IndoorLayout},
    materials::Surface,
    objects::Assembly,
};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CourtyardRoof {
    pub min: Vec2,
    pub max: Vec2,
    pub slab_m: f32,
    pub soffit: Surface,
    pub fascia: Surface,
}

pub(super) fn corners(lo: Vec2, hi: Vec2) -> [Vec2; 4] {
    [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)]
}

impl CourtyardRoof {
    pub(super) fn sample(seed: u64, size: Vec3) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"OpenWalk"));
        let half = size.xz() * 0.5;
        // Each walk has its own depth. A deeper seating arcade can face a
        // narrow circulation canopy with the opening well off the room axes.
        let mut walk = || r.random_range(1.55..3.20);
        let min = -half + Vec2::new(walk(), walk());
        let max = half - Vec2::new(walk(), r.random_range(2.25..3.45));
        Self {
            min,
            max,
            slab_m: r.random_range(0.18..0.38),
            soffit: [Surface::Wood, Surface::Concrete, Surface::Paint][r.random_range(0..3)],
            fascia: [Surface::Metal, Surface::Concrete, Surface::WoodEdge][r.random_range(0..3)],
        }
    }

    pub fn open_area_m2(&self) -> f32 {
        (self.max - self.min).element_product()
    }

    pub(crate) fn contains(&self, p: Vec2) -> bool {
        p.cmpgt(self.min).all() && p.cmplt(self.max).all()
    }

    pub(crate) fn overlaps(&self, lo: Vec2, hi: Vec2) -> bool {
        lo.cmplt(self.max).all() && hi.cmpgt(self.min).all()
    }

    /// Uniform by supported area, including when the selected primary region
    /// clips a walk. Sampling a point inside the open sky wastes a rig trial.
    pub(crate) fn sample_mount(
        &self,
        lo: Vec2,
        hi: Vec2,
        radius: f32,
        rng: &mut impl Rng,
    ) -> Option<Vec2> {
        if lo.cmpge(hi).any() {
            return None;
        }
        let a = (self.min - Vec2::splat(radius)).clamp(lo, hi);
        let b = (self.max + Vec2::splat(radius)).clamp(lo, hi);
        let strips = [
            (lo, Vec2::new(a.x, hi.y)),
            (Vec2::new(b.x, lo.y), hi),
            (Vec2::new(a.x, lo.y), Vec2::new(b.x, a.y)),
            (Vec2::new(a.x, b.y), Vec2::new(b.x, hi.y)),
        ];
        let areas = strips.map(|(l, h)| (h - l).element_product());
        let total = areas.iter().sum::<f32>();
        if total <= 0. {
            return None;
        }
        let mut pick = rng.random_range(0.0..total);
        for ((l, h), area) in strips.into_iter().zip(areas) {
            if pick < area {
                return Some(Vec2::new(
                    rng.random_range(l.x..h.x),
                    rng.random_range(l.y..h.y),
                ));
            }
            pick -= area;
        }
        None
    }

    pub(super) fn supported_corners(&self, posts: &[Pillar]) -> bool {
        corners(self.min, self.max).into_iter().all(|corner| {
            posts.iter().any(|post| {
                let offset =
                    (post.center - corner) * (corner - self.min.midpoint(self.max)).signum();
                offset.cmpge(Vec2::splat(post.radius)).all()
                    && offset.cmple(Vec2::splat(post.radius + 0.16)).all()
            })
        })
    }

    pub(crate) fn supports_rail(&self, start: Vec2, end: Vec2, radius: f32) -> bool {
        let lo = self.min - Vec2::splat(radius);
        let hi = self.max + Vec2::splat(radius);
        !segment_hits_box(
            Vec3::new(start.x, 0., start.y),
            Vec3::new(end.x, 0., end.y),
            Vec3::new(lo.x, -1., lo.y),
            Vec3::new(hi.x, 1., hi.y),
        )
    }

    pub(super) fn columns(&self, seed: u64) -> Vec<Pillar> {
        let mut r = stream(seed, u64::from_le_bytes(*b"CourtCol"));
        let radius = r.random_range(0.14..0.26);
        let inset = Vec2::splat(radius + r.random_range(0.05..0.15));
        let points = corners(self.min - inset, self.max + inset);
        let pitch = r.random_range(3.5..5.8);
        let mut out = Vec::with_capacity(16);
        for (a, b) in super::polygon::edges(&points) {
            let bays = (a.distance(b) / pitch).ceil() as usize;
            for i in 0..bays {
                out.push(Pillar {
                    center: a.lerp(b, i as f32 / bays as f32),
                    radius,
                    sides: 32,
                    profile: Some(ColumnProfile::for_layout(
                        seed,
                        out.len(),
                        IndoorLayout::Courtyard,
                    )),
                });
            }
        }
        out
    }

    pub(super) fn valid(&self, size: Vec3) -> bool {
        let half = size.xz() * 0.5;
        self.min.is_finite()
            && self.max.is_finite()
            && self.min.cmpge(-half + Vec2::ONE).all()
            && self.max.cmple(half - Vec2::ONE).all()
            && (self.max - self.min).cmpge(Vec2::splat(3.)).all()
            && (0.14..=0.45).contains(&self.slab_m)
            && matches!(
                self.soffit,
                Surface::Paint | Surface::Concrete | Surface::Wood
            )
            && matches!(
                self.fascia,
                Surface::Concrete | Surface::Metal | Surface::WoodEdge
            )
    }

    /// Floor fraction receiving a parallel daylight ray at the metering plane.
    /// Project both lips of the aperture and clip the footprint to the court.
    fn lit_fraction(&self, size: Vec3, light: Vec3) -> f32 {
        if light.y <= 0. {
            return 0.;
        }
        let t = light.xz() / light.y;
        let lower = t * (size.y - 0.75);
        let upper = t * (size.y + self.slab_m - 0.75);
        let half = size.xz() * 0.5;
        let lo = (self.min - lower.min(upper)).max(-half);
        let hi = (self.max - lower.max(upper)).min(half);
        (hi - lo).max(Vec2::ZERO).element_product() / size.xz().element_product()
    }

    pub(crate) fn daylight_factors(&self, size: Vec3, sun: Vec3) -> Vec2 {
        // Midpoint quadrature, uniform in cosine-weighted solid angle. This
        // meter runs during room setup; direct lighting uses the actual mesh.
        let mut sky = 0.;
        for i in 0..4 {
            let mu = ((i as f32 + 0.5) / 4.).sqrt();
            let horizontal = (1. - mu * mu).sqrt();
            for j in 0..12 {
                let dir = Vec2::from_angle((j as f32 + 0.5) * std::f32::consts::TAU / 12.);
                sky +=
                    self.lit_fraction(size, Vec3::new(dir.x * horizontal, mu, dir.y * horizontal));
            }
        }
        Vec2::new(sky / 48., self.lit_fraction(size, sun) * sun.y.max(0.))
    }

    /// Physical roof hit from inside the court, including grazing views of the
    /// aperture reveal. The scene's finite sky mesh closes the open directions.
    pub(super) fn ray_hit(&self, size: Vec3, origin: Vec3, ray: Vec3) -> Option<f32> {
        let mut closest = f32::INFINITY;
        if ray.y.abs() > 1e-8 {
            for y in [size.y, size.y + self.slab_m] {
                let t = (y - origin.y) / ray.y;
                let at = (origin + ray * t).xz();
                if t > 1e-5 && at.abs().cmple(size.xz() * 0.5).all() && !self.contains(at) {
                    closest = closest.min(t);
                }
            }
        }
        for axis in 0..2 {
            let d = ray.xz()[axis];
            if d.abs() <= 1e-8 {
                continue;
            }
            for edge in [self.min[axis], self.max[axis]] {
                let t = (edge - origin.xz()[axis]) / d;
                let at = origin + ray * t;
                if t > 1e-5
                    && (size.y..=size.y + self.slab_m).contains(&at.y)
                    && (self.min[1 - axis]..=self.max[1 - axis]).contains(&at.xz()[1 - axis])
                {
                    closest = closest.min(t);
                }
            }
        }
        closest.is_finite().then_some(closest)
    }

    pub(crate) fn build(&self, a: &mut Assembly, size: Vec3) {
        let outer = corners(-size.xz() * 0.5, size.xz() * 0.5);
        let inner = corners(self.min, self.max);
        let y = size.y;
        let top = y + self.slab_m;
        let at = |p: Vec2, h: f32| Vec3::new(p.x, h, p.y);
        for i in 0..4 {
            let j = (i + 1) % 4;
            let strip = [outer[i], outer[j], inner[j], inner[i]];
            a.part(self.soffit, "ceiling")
                .metric_quad(strip.map(|p| at(p, y)), Vec3::NEG_Y);
            a.part(Surface::Concrete, "ceiling")
                .metric_quad(strip.map(|p| at(p, top)), Vec3::Y);
            let tangent = outer[j] - outer[i];
            let normal = Vec3::new(tangent.y, 0., -tangent.x).normalize();
            let edge = |g: &mut Geometry, p, q, n| {
                g.metric_quad([at(p, y), at(q, y), at(q, top), at(p, top)], n);
            };
            let face = a.part(self.fascia, "ceiling");
            edge(face, outer[i], outer[j], normal);
            edge(face, inner[i], inner[j], -normal);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::envelope::ray_triangle;

    #[test]
    fn courtyard_mounts_sample_supported_area_in_clipped_regions() {
        let size = Vec3::new(12., 3.8, 14.);
        let roof = CourtyardRoof::sample(91, size);
        let r = 0.12;
        let mut rng = stream(21, 19);
        assert!(roof.sample_mount(roof.min, roof.max, r, &mut rng).is_none());
        let lo = -size.xz() * 0.5 + Vec2::splat(0.3);
        let hi = Vec2::new(roof.max.x - 0.3, size.z * 0.5 - 0.3);
        let a = roof.min - Vec2::splat(r);
        let b = roof.max + Vec2::splat(r);
        let left_area = (a.x - lo.x) * (hi.y - lo.y);
        let roof_area = left_area + (hi.x - a.x) * (a.y - lo.y + hi.y - b.y);
        let mut left = 0;
        for _ in 0..8192 {
            let p = roof.sample_mount(lo, hi, r, &mut rng).unwrap();
            assert!(p.cmpge(lo).all() && p.cmplt(hi).all() && roof.supports_rail(p, p, r));
            left += usize::from(p.x < a.x);
        }
        assert!((left as f32 / 8192. - left_area / roof_area).abs() < 0.018);
    }

    #[test]
    fn courtyard_opening_reveal_and_daylight_agree_with_physical_mesh() {
        let size = Vec3::new(12., 3.85, 14.);
        for seed in 0..24 {
            let roof = CourtyardRoof::sample(seed, size);
            assert!(roof.valid(size));
            let mut a = Assembly::default();
            roof.build(&mut a, size);
            let tris: Vec<_> = a
                .parts
                .values()
                .flat_map(|g| {
                    g.indices
                        .as_chunks::<3>()
                        .0
                        .iter()
                        .map(|t| t.map(|i| Vec3::from(g.positions[i as usize])))
                })
                .collect();
            assert_eq!(tris.len(), 32);
            let center = (roof.min + roof.max) * 0.5;
            let sky = Vec3::new(center.x, 1.52, center.y);
            let covered = sky.with_x(roof.max.x + 0.55);
            let lip = Vec3::new(roof.max.x, size.y + 0.5 * roof.slab_m, center.y);
            for (eye, ray) in [
                (sky, Vec3::Y),
                (covered, Vec3::Y),
                (sky, (lip - sky).normalize()),
                (covered.with_y(size.y + 2.), Vec3::NEG_Y),
            ] {
                let mesh = tris
                    .iter()
                    .filter_map(|&t| ray_triangle(eye, ray, t))
                    .min_by(f32::total_cmp);
                let hit = roof.ray_hit(size, eye, ray);
                assert_eq!(mesh.is_some(), hit.is_some());
                if let Some(d) = hit {
                    assert!((d - mesh.unwrap()).abs() < 2e-5);
                }
            }
            let factors = roof.daylight_factors(size, Vec3::Y);
            let open = roof.open_area_m2() / size.xz().element_product();
            assert!((factors.y - open).abs() < 2e-6);
            assert!(factors.x > 0. && factors.x < open);
            assert_eq!(
                roof.daylight_factors(size, Vec3::new(1., 0.01, 0.).normalize())
                    .y,
                0.
            );
            assert_eq!(roof.daylight_factors(size, Vec3::NEG_Y).y, 0.);
        }
    }
}
