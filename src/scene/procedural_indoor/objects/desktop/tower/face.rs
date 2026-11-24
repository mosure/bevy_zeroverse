//! Exterior sheet and finite-depth openings. Every face is cut before a
//! terminal, control or intake is installed; no full panel spans an aperture.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

#[derive(Clone, Copy)]
pub(super) enum Plane {
    Front(f32),
    Back(f32),
    Side(f32),
}
impl Plane {
    pub(super) fn normal(self) -> Vec3 {
        match self {
            Self::Front(_) => Vec3::Z,
            Self::Back(_) => Vec3::NEG_Z,
            Self::Side(x) => Vec3::X * x.signum(),
        }
    }
    pub(super) fn point(self, xy: Vec2, inset: f32) -> Vec3 {
        let p = match self {
            Self::Front(z) | Self::Back(z) => xy.extend(z),
            Self::Side(x) => Vec3::new(x, xy.y, xy.x),
        };
        p - self.normal() * inset
    }
    pub(super) fn rect(self, g: &mut Geometry, bounds: Rect, inset: f32) {
        if bounds.width() > 0. && bounds.height() > 0. {
            g.metric_quad(corners(bounds).map(|p| self.point(p, inset)), self.normal());
        }
    }
    /// Partition into strips only at aperture boundaries. Uncut regions have
    /// one quad and adjacent strips share edges, including on the back face.
    pub(super) fn sheet(self, g: &mut Geometry, area: Rect, cuts: &[Rect]) {
        const MAX: usize = 8; // five rear connectors, or up to five controls and two intakes
        debug_assert!(cuts.len() <= MAX);
        if cuts.is_empty() {
            self.rect(g, area, 0.);
            return;
        }
        let mut heights = [0.; 2 + 2 * MAX];
        heights[..2].copy_from_slice(&[area.min.y, area.max.y]);
        for (i, h) in cuts.iter().enumerate() {
            heights[2 + i * 2..4 + i * 2].copy_from_slice(&[h.min.y, h.max.y]);
        }
        let levels = &mut heights[..2 + 2 * cuts.len()];
        levels.sort_unstable_by(f32::total_cmp);
        for row in levels.windows(2) {
            let (lo, hi) = (row[0], row[1]);
            if hi <= lo {
                continue;
            }
            let mut intervals = [Vec2::ZERO; MAX];
            let mut count = 0;
            for h in cuts.iter().filter(|h| h.min.y < hi && h.max.y > lo) {
                intervals[count] = Vec2::new(h.min.x, h.max.x);
                count += 1;
            }
            intervals[..count].sort_unstable_by(|a, b| a.x.total_cmp(&b.x));
            let mut x = area.min.x;
            for h in &intervals[..count] {
                if h.x > x {
                    self.rect(g, Rect::new(x, lo, h.x, hi), 0.);
                }
                x = x.max(h.y);
            }
            if x < area.max.x {
                self.rect(g, Rect::new(x, lo, area.max.x, hi), 0.);
            }
        }
    }
    pub(super) fn walls(self, g: &mut Geometry, bounds: Rect, inset: f32) {
        let pts = corners(bounds);
        for i in 0..4 {
            let j = (i + 1) % 4;
            let inward = self.point(bounds.center(), 0.) - self.point((pts[i] + pts[j]) * 0.5, 0.);
            g.metric_quad(
                [
                    self.point(pts[i], 0.),
                    self.point(pts[j], 0.),
                    self.point(pts[j], inset),
                    self.point(pts[i], inset),
                ],
                inward,
            );
        }
    }
}

pub(super) fn corners(r: Rect) -> [Vec2; 4] {
    [
        r.min,
        Vec2::new(r.max.x, r.min.y),
        r.max,
        Vec2::new(r.min.x, r.max.y),
    ]
}
pub(super) fn clipped(r: Rect, c: f32) -> [Vec2; 8] {
    let [a, b, d, e] = corners(r);
    [
        a + Vec2::X * c,
        b - Vec2::X * c,
        b + Vec2::Y * c,
        d - Vec2::Y * c,
        d - Vec2::X * c,
        e + Vec2::X * c,
        e - Vec2::Y * c,
        a + Vec2::Y * c,
    ]
}
pub(super) fn flat_field(r: Rect, c: f32) -> Rect {
    Rect::from_corners(r.min + Vec2::X * c, r.max - Vec2::X * c)
}

pub(super) fn case_shell(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &CaseProgram,
    f: &Fascia,
    stock: &Housing,
    label: &str,
    grille: &str,
) {
    let outer = Rect::new(-o.size.x * 0.5, p.feet_m, o.size.x * 0.5, o.size.y);
    let rear = outer.inflate(-f.rear_bevel_m);
    let face = outer.inflate(-f.nose_inset_m);
    let (back, front) = (-o.size.z * 0.5, o.size.z * 0.5);
    let levels = [
        (rear, back),
        (outer, back + f.rear_bevel_m),
        (outer, front - p.fascia_depth_m),
        (face, front),
    ];
    for (k, pair) in levels.windows(2).enumerate() {
        let [(r0, z0), (r1, z1)] = [pair[0], pair[1]];
        let lo = clipped(r0, f.corner_m);
        let hi = clipped(r1, f.corner_m);
        for i in 0..8 {
            let j = (i + 1) % 8;
            let side = if i == 2 {
                1
            } else if i == 6 {
                -1
            } else {
                0
            };
            if k == 1 && side != 0 && f.side == side {
                let plane = Plane::Side(lo[i].x);
                let bounds = Rect::new(z0, outer.min.y + f.corner_m, z1, outer.max.y - f.corner_m);
                let span = bounds.size() * f.side_fraction;
                let vent = Rect::from_center_size(
                    bounds.min + span * 0.5 + (bounds.size() - span) * f.side_offset,
                    span,
                );
                plane.sheet(a.part(stock.surface, label), bounds, &[vent]);
                plane.walls(a.part(stock.surface, label), vent, f.intake_recess_m);
                plane.rect(a.part(Surface::Metal, grille), vent, f.intake_recess_m);
                continue;
            }
            let outside = ((lo[i] + lo[j]) * 0.5 - r0.center()).extend(0.);
            a.part(stock.surface, label).metric_quad(
                [
                    lo[i].extend(z0),
                    lo[j].extend(z0),
                    hi[j].extend(z1),
                    hi[i].extend(z1),
                ],
                outside,
            );
        }
    }
    for (r, plane) in [(rear, Plane::Back(back)), (face, Plane::Front(front))] {
        let pts = clipped(r, f.corner_m);
        for ids in [[1, 2, 3, 4], [5, 6, 7, 0]] {
            a.part(stock.surface, label)
                .metric_quad(ids.map(|i| plane.point(pts[i], 0.)), plane.normal());
        }
    }
}
