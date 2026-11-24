//! Orthographic probes on the posed facial skin and eyes. Index only the facial
//! rig once, so a hand passing the forehead cannot become a brow or nose support.
use bevy::prelude::*;

pub(crate) struct Topology {
    sources: Vec<usize>,
    quads: Vec<[usize; 4]>,
}

impl Topology {
    pub(crate) fn new(faces: &[i64], vertices: usize, belongs: impl Fn(usize) -> bool) -> Self {
        let mut remap = vec![usize::MAX; vertices];
        let mut sources = Vec::new();
        let quads = faces
            .as_chunks::<4>()
            .0
            .iter()
            .filter_map(|q| {
                let q = q.map(|i| i as usize);
                q.iter().all(|&i| belongs(i)).then(|| {
                    q.map(|i| {
                        if remap[i] == usize::MAX {
                            remap[i] = sources.len();
                            sources.push(i);
                        }
                        remap[i]
                    })
                })
            })
            .collect();
        Self { sources, quads }
    }
    pub(in super::super) fn project(
        &self,
        posed: &[Vec3],
        origin: Vec3,
        rotation: Quat,
        scale: f32,
        bounds: Rect,
        eyewear: bool,
    ) -> ProjectedHead {
        let inverse = rotation.inverse();
        let local: Vec<_> = self
            .sources
            .iter()
            .map(|&i| inverse * (posed[i] - origin))
            .collect();
        let mut front = Patch::new(bounds);
        let side_bounds = Rect::from_corners(
            Vec2::new(-0.055, -0.012) * scale,
            Vec2::new(0.076, 0.028) * scale,
        );
        let mut temples = eyewear.then(|| std::array::from_fn(|_| Patch::new(side_bounds)));
        for q in &self.quads {
            if q.iter()
                .any(|&i| local[i].length_squared() >= (0.19 * scale).powi(2))
            {
                continue;
            }
            for [i, j, k] in [[0, 1, 2], [0, 2, 3]] {
                let tri = [i, j, k].map(|i| local[q[i]]);
                front.insert(tri);
                if let Some(sides) = &mut temples {
                    for side in [-1., 1.] {
                        if tri.iter().any(|p| p.x * side > 0.042 * scale) {
                            sides[usize::from(side > 0.)]
                                .insert(tri.map(|p| Vec3::new(p.z, p.y, -side * p.x)));
                        }
                    }
                }
            }
        }
        ProjectedHead { front, temples }
    }
}

pub(crate) struct ProjectedHead {
    pub front: Patch,
    pub temples: Option<[Patch; 2]>,
}
const NX: usize = 8;
const NY: usize = 4;

#[derive(Clone, Copy)]
struct Triangle {
    a: Vec3,
    ab: Vec3,
    ac: Vec3,
    determinant_inverse: f32,
    normal: Vec3,
}
pub(crate) struct Patch {
    bounds: Rect,
    bins_per_metre: Vec2,
    bins: [Vec<Triangle>; NX * NY],
}
impl Patch {
    fn new(bounds: Rect) -> Self {
        Self {
            bounds,
            bins_per_metre: Vec2::new(NX as f32, NY as f32) / bounds.size().max(Vec2::splat(1e-5)),
            bins: Default::default(),
        }
    }
    fn insert(&mut self, [a, b, c]: [Vec3; 3]) {
        let lower = a.min(b).min(c).truncate();
        let upper = a.max(b).max(c).truncate();
        if lower.cmpgt(self.bounds.max).any() || upper.cmplt(self.bounds.min).any() {
            return;
        }
        let ab = b - a;
        let ac = c - a;
        let determinant = ab.truncate().perp_dot(ac.truncate());
        if determinant.abs() < 1e-10 {
            return;
        }
        let n = ab.cross(ac).normalize_or(Vec3::NEG_Z);
        let tri = Triangle {
            a,
            ab,
            ac,
            determinant_inverse: determinant.recip(),
            normal: if n.z > 0. { -n } else { n },
        };
        let from = self.cell(lower - Vec2::splat(1e-5));
        let to = self.cell(upper + Vec2::splat(1e-5));
        for y in from.y..=to.y {
            for x in from.x..=to.x {
                self.bins[y as usize * NX + x as usize].push(tri);
            }
        }
    }
    fn cell(&self, xy: Vec2) -> UVec2 {
        ((xy - self.bounds.min) * self.bins_per_metre)
            .clamp(Vec2::ZERO, Vec2::new((NX - 1) as f32, (NY - 1) as f32))
            .as_uvec2()
    }
    pub(crate) fn front(&self, xy: Vec2) -> Option<(f32, Vec3)> {
        let cell = self.cell(xy);
        self.bins[cell.y as usize * NX + cell.x as usize]
            .iter()
            .filter_map(|t| {
                let p = xy - t.a.truncate();
                let u = p.perp_dot(t.ac.truncate()) * t.determinant_inverse;
                let v = t.ab.truncate().perp_dot(p) * t.determinant_inverse;
                (u >= -1e-4 && v >= -1e-4 && u + v <= 1.0001)
                    .then_some((t.a.z + u * t.ab.z + v * t.ac.z, t.normal))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0))
    }
}
