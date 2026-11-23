//! Metric slab courses fitted to a bounded site. All slabs of a lot share its
//! PBR finish; their texture origins vary as they would across cut stock.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

#[derive(Clone, Copy)]
pub(super) struct Rect {
    pub lo: Vec2,
    pub hi: Vec2,
}

impl Rect {
    pub fn size(self) -> Vec2 {
        self.hi - self.lo
    }

    fn usable(self) -> bool {
        self.size().min_element() > 0.0005
    }

    pub fn inset(self, width: f32) -> Self {
        Self {
            lo: self.lo + width,
            hi: self.hi - width,
        }
    }

    fn intersection(self, other: Self) -> Self {
        Self {
            lo: self.lo.max(other.lo),
            hi: self.hi.min(other.hi),
        }
    }

    /// Nonoverlapping partition of self outside a rectangular planting bed.
    fn without(self, hole: Option<Self>, mut visit: impl FnMut(Self)) {
        let Some(hole) = hole.map(|r| r.intersection(self)).filter(|r| r.usable()) else {
            visit(self);
            return;
        };
        for (lo, hi) in [
            (self.lo, Vec2::new(hole.lo.x, self.hi.y)),
            (Vec2::new(hole.hi.x, self.lo.y), self.hi),
            (
                Vec2::new(hole.lo.x, self.lo.y),
                Vec2::new(hole.hi.x, hole.lo.y),
            ),
            (
                Vec2::new(hole.lo.x, hole.hi.y),
                Vec2::new(hole.hi.x, self.hi.y),
            ),
        ] {
            let part = Self { lo, hi };
            if part.usable() {
                visit(part);
            }
        }
    }
}

#[derive(Clone, Copy)]
pub(super) struct Frame {
    origin: Vec3,
    x: Vec3,
    z: Vec3,
}

impl Frame {
    pub fn new(center: Vec2, yaw: f32) -> Self {
        let rotation = Quat::from_rotation_y(yaw);
        Self {
            origin: Vec3::new(center.x, 0., center.y),
            x: rotation * Vec3::X,
            z: rotation * Vec3::Z,
        }
    }

    pub fn local(self, world: Vec2) -> Vec2 {
        let d = Vec3::new(world.x, 0., world.y) - self.origin;
        Vec2::new(d.dot(self.x), d.dot(self.z))
    }

    fn corners(self, rect: Rect, y: f32) -> [Vec3; 4] {
        let origin = self.origin + Vec3::Y * y;
        let lo = origin + self.x * rect.lo.x + self.z * rect.lo.y;
        let x = self.x * rect.size().x;
        let z = self.z * rect.size().y;
        [lo, lo + z, lo + z + x, lo + x]
    }

    pub fn top(self, g: &mut Geometry, rect: Rect, y: f32, offset: Vec2) {
        let [lo, hi] = [rect.lo + offset, rect.hi + offset];
        g.vertex_quad(
            self.corners(rect, y),
            [Vec3::Y; 4],
            [[lo.x, -lo.y], [lo.x, -hi.y], [hi.x, -hi.y], [hi.x, -lo.y]],
        );
    }

    // Match the existing cuboid top chart on ground outside the installation.
    fn apron_top(self, g: &mut Geometry, r: Rect, y: f32, anchor: Vec2) {
        let lo = r.lo - anchor;
        let hi = r.hi - anchor;
        g.vertex_quad(
            self.corners(r, y),
            [Vec3::Y; 4],
            [[lo.x, lo.y], [lo.x, hi.y], [hi.x, hi.y], [hi.x, lo.y]],
        );
    }

    fn side(self, g: &mut Geometry, r: Rect, low: f32, high: f32, edge: usize, inner: bool) {
        let points = self.corners(r, high);
        let normal = [-self.x, self.z, self.x, -self.z][edge];
        let j = (edge + 1) % 4;
        let (a, b, n) = if inner {
            (edge, j, -normal)
        } else {
            (j, edge, normal)
        };
        let down = Vec3::Y * (high - low);
        let width = if edge.is_multiple_of(2) {
            r.size().y
        } else {
            r.size().x
        };
        g.vertex_quad(
            [points[a], points[b], points[b] - down, points[a] - down],
            [n; 4],
            [[0., high], [width, high], [width, low], [0., low]],
        );
    }

    fn sides(self, g: &mut Geometry, r: Rect, low: f32, high: f32, inner: bool) {
        for edge in 0..4 {
            self.side(g, r, low, high, edge, inner);
        }
    }

    fn stone(
        self,
        g: &mut Geometry,
        cell: Rect,
        bed: Option<Rect>,
        base: f32,
        top: f32,
        offset: Vec2,
    ) {
        let Some(cut) = bed.map(|h| cell.intersection(h)).filter(|h| h.usable()) else {
            self.top(g, cell, top, offset);
            self.sides(g, cell, base, top, false);
            return;
        };
        // Subdivide only the top. Add returns on the actual exposed boundary;
        // the rectangular decomposition is not a physical seam in a cut stone.
        cell.without(Some(cut), |part| {
            self.top(g, part, top, offset);
            let outer = [
                part.lo.x == cell.lo.x,
                part.hi.y == cell.hi.y,
                part.hi.x == cell.hi.x,
                part.lo.y == cell.lo.y,
            ];
            let at_cut = [
                part.lo.x == cut.hi.x,
                part.hi.y == cut.lo.y,
                part.hi.x == cut.lo.x,
                part.lo.y == cut.hi.y,
            ];
            for edge in 0..4 {
                if outer[edge] {
                    self.side(g, part, base, top, edge, false);
                } else if at_cut[edge] {
                    let mut reach = part;
                    let axis = if edge.is_multiple_of(2) { 1 } else { 0 };
                    reach.lo[axis] = reach.lo[axis].max(cut.lo[axis]);
                    reach.hi[axis] = reach.hi[axis].min(cut.hi[axis]);
                    if reach.size()[axis] > 0.0005 {
                        self.side(g, reach, base, top, edge, false);
                    }
                }
            }
        });
    }

    pub fn rim(self, g: &mut Geometry, outer: Rect, inner: Rect, soil: f32, base: f32, crest: f32) {
        outer.without(Some(inner), |r| self.top(g, r, crest, Vec2::ZERO));
        self.sides(g, outer, base, crest, false);
        self.sides(g, inner, soil, crest, true);
    }

    /// The substrate has no hidden top spanning a tree pit.
    pub fn base(self, g: &mut Geometry, r: Rect, low: f32, high: f32, hole: Option<Rect>) {
        g.metric_quad(self.corners(r, low), -Vec3::Y);
        self.sides(g, r, low, high, false);
        r.without(hole, |s| self.top(g, s, high, Vec2::ZERO));
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Bond {
    Stack,
    Running,
    Coursed,
}

#[derive(Clone, Copy)]
pub(super) struct Paving {
    pub bond: Bond,
    pub surface: Surface,
    pub joint: f32,
    pub relief: f32,
    nx: usize,
    nz: usize,
    crosswise: bool,
    seed: u64,
}

impl Paving {
    pub fn sample(seed: u64, rect: Rect) -> Self {
        let mut r = stream(seed, 9561);
        let family: f32 = r.random();
        let crosswise = r.random_bool(0.50);
        let size = if crosswise {
            rect.size().yx()
        } else {
            rect.size()
        };
        let (bond, length, width) = if family < 0.36 {
            // Scored cast slabs may span several metres.
            (
                Bond::Stack,
                r.random_range(0.85..2.35),
                r.random_range(0.68..1.65),
            )
        } else if family < 0.71 {
            (
                Bond::Running,
                r.random_range(0.66..1.48),
                r.random_range(0.36..0.82),
            )
        } else {
            (
                Bond::Coursed,
                r.random_range(0.75..1.82),
                r.random_range(0.40..0.98),
            )
        };
        let nz = (size.y / width).round().clamp(1., 9.) as usize;
        let nx = (size.x / length)
            .round()
            .clamp(1., (90 / nz).min(14) as f32) as usize;

        // Fired-clay flags have a different manufacturing scale. Keep their
        // actual size even in large sites by choosing concrete when a clay lot
        // would exceed the geometric budget.
        let clay_pitch = r.random_range(0.38..0.60);
        let clay_depth = clay_pitch * r.random_range(0.52..1.0);
        let clay_nx = (size.x / clay_pitch).ceil().max(1.) as usize;
        let clay_nz = (size.y / clay_depth).ceil().max(1.) as usize;
        let clay = !matches!(bond, Bond::Coursed)
            && r.random_bool(0.40)
            && clay_nx * clay_nz + clay_nz / 2 <= 96;
        let (surface, nx, nz, joint, relief) = if clay {
            (
                Surface::Terracotta,
                clay_nx,
                clay_nz,
                r.random_range(0.003..0.008),
                r.random_range(0.002..0.005),
            )
        } else {
            (
                Surface::Concrete,
                nx,
                nz,
                r.random_range(0.006..0.015),
                r.random_range(0.004..0.010),
            )
        };
        Self {
            bond,
            surface,
            nx,
            nz,
            crosswise,
            seed,
            joint,
            relief,
        }
    }

    pub fn module(self, rect: Rect) -> Vec2 {
        let size = if self.crosswise {
            rect.size().yx()
        } else {
            rect.size()
        };
        size / Vec2::new(self.nx as f32, self.nz as f32)
    }

    /// Fit entire courses to the site. A running bond has half terminal stones;
    /// cut-flag courses vary their lengths and share an exact row boundary.
    pub fn panels(self, rect: Rect, mut visit: impl FnMut(Rect, Vec2)) {
        let mut rng = stream(self.seed, 9562);
        let mut z0 = 0.;
        for row in 0..self.nz {
            let z1 = if row + 1 == self.nz {
                1.
            } else {
                let skew = if matches!(self.bond, Bond::Coursed) {
                    rng.random_range(-0.20..0.20)
                } else {
                    0.
                };
                (row as f32 + 1. + skew) / self.nz as f32
            };
            let stagger = matches!(self.bond, Bond::Running) && row % 2 == 1;
            let columns = self.nx + usize::from(stagger);
            let mut x0 = 0.;
            for col in 0..columns {
                let x1 = if col + 1 == columns {
                    1.
                } else {
                    let skew = match self.bond {
                        Bond::Running if stagger => -0.50,
                        Bond::Coursed => rng.random_range(-0.27..0.27),
                        _ => 0.,
                    };
                    (col as f32 + 1. + skew) / self.nx as f32
                };
                let map = |v: Vec2| rect.lo + rect.size() * if self.crosswise { v.yx() } else { v };
                let cell = Rect {
                    lo: map(Vec2::new(x0, z0)),
                    hi: map(Vec2::new(x1, z1)),
                };
                visit(
                    cell.inset(self.joint * 0.5),
                    Vec2::new(rng.random_range(-9.0..9.), rng.random_range(-9.0..9.)),
                );
                x0 = x1;
            }
            z0 = z1;
        }
    }

    pub fn lay(self, frame: Frame, g: &mut Geometry, rect: Rect, top: f32, bed: Option<Rect>) {
        self.panels(rect, |cell, offset| {
            frame.stone(g, cell, bed, top - self.relief, top, offset);
        });
    }

    pub fn panel_count(self) -> usize {
        self.nx * self.nz
            + if matches!(self.bond, Bond::Running) {
                self.nz / 2
            } else {
                0
            }
    }
}

pub(super) fn fill_apron(
    frame: Frame,
    g: &mut Geometry,
    whole: Rect,
    field: Rect,
    grade: f32,
    relief: f32,
) {
    // The original grade is continuous up to the lip of the field. Its bedding
    // is a separate, lower top inside that lip, so joint depth survives ray casts.
    g.metric_quad(frame.corners(whole, grade - 0.20), -Vec3::Y);
    frame.sides(g, whole, grade - 0.20, grade, false);
    whole.without(Some(field), |r| frame.apron_top(g, r, grade, whole.lo));
    frame.apron_top(g, field, grade - relief, whole.lo);
    frame.sides(g, field, grade - relief, grade, true);
}
