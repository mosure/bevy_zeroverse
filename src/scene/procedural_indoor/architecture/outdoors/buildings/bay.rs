//! Front wall, masonry returns, rebated sash and glazing in one facade frame.
//! Near openings lead into an enclosed storey. Distant proxy openings have
//! an opaque backing at the rear of the jamb.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;

pub(super) struct Face {
    pub origin: Vec3,
    pub axis: Vec3,
    pub normal: Vec3,
}

impl Face {
    pub(super) fn point(&self, uv: Vec2, depth: f32) -> Vec3 {
        self.origin + self.axis * uv.x + Vec3::Y * uv.y - self.normal * depth
    }
    pub(super) fn panel(&self, g: &mut Geometry, lo: Vec2, hi: Vec2, depth: f32) {
        g.metric_quad(
            [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)].map(|uv| self.point(uv, depth)),
            self.normal,
        );
    }
    fn surround(&self, g: &mut Geometry, lo: Vec2, hi: Vec2, hole: &Aperture, depth: f32) {
        self.panel(g, lo, Vec2::new(hole.lo.x, hi.y), depth);
        self.panel(g, Vec2::new(hole.hi.x, lo.y), hi, depth);
        self.panel(
            g,
            Vec2::new(hole.lo.x, lo.y),
            Vec2::new(hole.hi.x, hole.lo.y),
            depth,
        );
        self.panel(
            g,
            Vec2::new(hole.lo.x, hole.hi.y),
            Vec2::new(hole.hi.x, hi.y),
            depth,
        );
    }
    pub(super) fn block(&self, g: &mut Geometry, at: Vec3, size: Vec3) {
        let tf = Transform::from_translation(self.point(at.xy(), at.z))
            .with_rotation(Quat::from_rotation_y(self.normal.x.atan2(self.normal.z)));
        g.cuboid(size, 0., tf);
    }

    pub(super) fn return_panel(
        &self,
        g: &mut Geometry,
        hole: &Aperture,
        i: usize,
        front: f32,
        rear: f32,
    ) {
        let corners = [
            hole.lo,
            Vec2::new(hole.hi.x, hole.lo.y),
            hole.hi,
            Vec2::new(hole.lo.x, hole.hi.y),
        ];
        let (a, b) = (corners[i], corners[(i + 1) % 4]);
        g.metric_quad(
            [
                self.point(a, front),
                self.point(b, front),
                self.point(b, rear),
                self.point(a, rear),
            ],
            [Vec3::Y, -self.axis, -Vec3::Y, self.axis][i],
        );
    }
    fn returns(&self, g: &mut Geometry, hole: &Aperture, front: f32, rear: f32) {
        for i in 0..4 {
            self.return_panel(g, hole, i, front, rear);
        }
    }
}

pub(super) struct Aperture {
    pub lo: Vec2,
    pub hi: Vec2,
}

pub(super) struct Bays {
    pub columns: usize,
    pub cell: f32,
    width: f32,
    frame: f32,
    corner: f32,
}
impl Bays {
    pub fn new(width: f32, recess: f32, building: &Building) -> Self {
        let columns = (width / building.bay_pitch).floor().clamp(1., 9.) as usize;
        Self {
            columns,
            cell: (width - 0.20) / columns as f32,
            width,
            frame: building.frame_width,
            corner: building.window_installation.corner_pier(recess),
        }
    }
    pub fn offset(&self, column: usize) -> f32 {
        (column as f32 + 0.5 - self.columns as f32 * 0.5) * self.cell
    }
    pub fn available(&self, u: f32) -> f32 {
        let corner_span = 2. * (self.width * 0.5 - u.abs() - self.corner - self.frame);
        corner_span.min(self.cell - 2. * self.frame - 0.025)
    }
    pub fn door_width(&self, column: usize, available: f32) -> Option<f32> {
        (column == self.columns / 2 && self.cell > 0.9 && available >= 0.76)
            .then(|| (self.cell * 0.78).clamp(0.78, 1.6).min(available))
    }
}

impl Aperture {
    pub fn expanded(&self, margin: f32) -> Self {
        Self {
            lo: self.lo - margin,
            hi: self.hi + margin,
        }
    }
    pub fn wall(&self, face: &Face, g: &mut Geometry, bay: &Self, back: f32, sealed: bool) {
        face.surround(g, bay.lo, bay.hi, self, 0.);
        face.returns(g, self, 0., back);
        if sealed {
            face.panel(g, self.lo, self.hi, back);
        }
    }
    pub fn sash(&self, face: &Face, g: &mut Geometry, outer: &Self, recess: f32, rebate: f32) {
        face.surround(g, outer.lo, outer.hi, self, recess);
        face.returns(g, self, recess, recess + rebate);
    }
    pub fn pane(&self, face: &Face, g: &mut Geometry, depth: f32) {
        face.panel(g, self.lo, self.hi, depth);
    }

    /// The shared architectural glass material describes an 8 mm plate. Bury
    /// its lateral faces in the gasket and keep both optical faces distinct.
    pub fn clear_pane(&self, face: &Face, g: &mut Geometry, depth: f32) {
        let span = self.hi - self.lo + Vec2::splat(0.002);
        face.block(
            g,
            ((self.hi + self.lo) * 0.5).extend(depth + 0.004),
            span.extend(0.008),
        );
    }
}
