//! Flush floor service boxes. The raised cartridge, cable ports and moving lid
//! share the aperture of the closed box. All dimensions are in metres.
use super::*;
use crate::scene::procedural_indoor::objects::electrical::{self, Contact, OutletFace, Rect};

const FITTED: u32 = 1 << 12;
const LIP: f32 = 0.0045;
const PIVOT_HEIGHT: f32 = 0.0039;
const JOINT: f32 = 0.002;
const LABEL: &str = "floor_outlet";

pub(crate) fn installed_variant(case: u32, standard: u32) -> u32 {
    FITTED | ((standard % 3) << 4) | (case % 3)
}

pub(crate) fn installed(o: &IndoorObject) -> bool {
    o.kind == ObjectKind::FloorOutlet && o.variant & FITTED != 0
}

pub(crate) fn face(o: &IndoorObject) -> Option<OutletFace> {
    installed(o)
        .then(|| electrical::installed_variant((o.variant >> 4) & 3))
        .and_then(OutletFace::from_variant)
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub enum Housing {
    Round,
    RoundedRectangle,
    ClippedRectangle,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct FloorBox {
    pub housing: Housing,
    pub outlet_finish: Surface,
    pub hatch_finish: Surface,
    pub insert: Surface,
    pub face: OutletFace,
    pub open: bool,
    pub socket_count: usize,
    pub hatch_width_m: f32,
    pub hatch_depth_m: f32,
    pub open_angle_rad: f32,
    pub port_pitch_m: f32,
    pub contact_height_m: f32,
    pub well_depth_m: f32,
    center_x: f32,
    pull_offset: f32,
    pull_width: f32,
    corner: f32,
}

pub fn parameters(o: &IndoorObject) -> Option<FloorBox> {
    let face = face(o)?;
    let mut r = stream(o.seed, u64::from_le_bytes(*b"FloorBay"));
    let s = super::service_size(o);
    let housing = match o.variant & 7 {
        0 => Housing::Round,
        1 => Housing::RoundedRectangle,
        _ => Housing::ClippedRectangle,
    };
    // Even the lowest open reservation (70 mm) accommodates the same hatch
    // that lies flush in the 6 mm closed box. Its depth fits a full-size plug.
    let hatch_width_m = s.x * r.random_range(0.49..0.74);
    let min_depth = (s.z * 0.25).clamp(0.053, 0.057);
    let max_depth = (s.z * 0.38).clamp(0.059, 0.065);
    let hatch_depth_m = r.random_range(min_depth..max_depth);
    let port_pitch_m = r.random_range(0.048..0.059);
    let pair = r.random_bool(0.68) && hatch_width_m >= port_pitch_m + 0.052;
    let outlet_finish = super::finish(o);
    let mut stock = stream(
        o.finish_seed.unwrap_or(o.seed),
        u64::from_le_bytes(*b"HatchPbr"),
    );
    Some(FloorBox {
        housing,
        outlet_finish,
        hatch_finish: if stock.random_bool(0.22) {
            Surface::Plastic
        } else {
            outlet_finish
        },
        insert: if stock.random_bool(0.23) {
            Surface::Ceramic
        } else {
            Surface::Plastic
        },
        face,
        open: is_open(o),
        socket_count: if pair { 2 } else { 1 },
        hatch_width_m,
        hatch_depth_m,
        open_angle_rad: r.random_range(86_f32..116.).to_radians(),
        port_pitch_m,
        contact_height_m: r.random_range(0.013..0.020),
        well_depth_m: if face == OutletFace::RoundWell {
            r.random_range(0.006..0.0095)
        } else {
            0.
        },
        center_x: s.x * r.random_range(-0.040..0.040),
        pull_offset: hatch_width_m * r.random_range(-0.24..0.24),
        pull_width: hatch_width_m * r.random_range(0.19..0.33),
        corner: r.random_range(0.13..0.44),
    })
}

impl FloorBox {
    fn center(&self) -> Vec2 {
        Vec2::X * self.center_x
    }
    fn hatch_size(&self) -> Vec2 {
        Vec2::new(self.hatch_width_m, self.hatch_depth_m)
    }
    fn port_x(&self, i: usize) -> f32 {
        self.center_x + (i as f32 - (self.socket_count - 1) as f32 * 0.5) * self.port_pitch_m
    }
    pub(crate) fn port(&self, i: usize) -> Option<Vec3> {
        (self.open && i < self.socket_count).then(|| {
            Vec3::new(
                self.port_x(i),
                self.contact_height_m - self.well_depth_m,
                0.,
            )
        })
    }
    // Fabricate in the XY plane, +Z up. The hinge sits at the rear of the cut.
    fn hinge(&self) -> Transform {
        let angle = if self.open { self.open_angle_rad } else { 0. };
        Transform::from_xyz(self.center_x, self.hatch_depth_m * 0.5, PIVOT_HEIGHT)
            .with_rotation(Quat::from_rotation_x(-angle))
    }
    fn lid_bounds(&self) -> (Vec3, Vec3) {
        (
            Vec3::new(-self.hatch_width_m * 0.5, -self.hatch_depth_m, -0.0011),
            Vec3::new(self.hatch_width_m * 0.5, -0.0012, 0.0014),
        )
    }
    pub(crate) fn cable_hits(&self, size: Vec3, start: Vec3, end: Vec3, radius: f32) -> bool {
        use crate::scene::procedural_indoor::layout::segment_hits_box;
        let pad = Vec3::splat(radius);
        let base = Vec3::new(
            size.x * 0.5,
            if self.open { LIP } else { 0.006 },
            size.z * 0.5,
        );
        if segment_hits_box(start, end, -base.with_y(0.) - pad, base + pad) {
            return true;
        }
        if !self.open {
            return false;
        }
        let half = self.hatch_size() * 0.5;
        // Also enclose the hinge pin at the back of the deployed cartridge.
        let lo = Vec3::new(self.center_x - half.x - JOINT, 0., -half.y - JOINT);
        let hi = Vec3::new(
            self.center_x + half.x + JOINT,
            self.contact_height_m,
            half.y,
        );
        if segment_hits_box(start, end, lo - pad, hi + pad) {
            return true;
        }
        let hinge = self.hinge();
        let to_lid =
            |p: Vec3| hinge.rotation.inverse() * (Vec3::new(p.x, -p.z, p.y) - hinge.translation);
        let (lo, hi) = self.lid_bounds();
        segment_hits_box(to_lid(start), to_lid(end), lo - pad, hi + pad)
    }
}

/// Split an outer edge at rays through the four aperture corners. The two
/// endpoints of each strip then project onto the same inner edge, even for a
/// wide, off-centre cut in a circular plate.
fn flange_top(g: &mut Geometry, b: Vec2, c: Vec2, aperture: Rect) {
    let center = (aperture.lo + aperture.hi) * 0.5;
    let half = (aperture.hi - aperture.lo) * 0.5;
    let at_cut = |p: Vec2| {
        let ray = p - center;
        center + ray / (ray.abs() / half).max_element()
    };
    let span = c - b;
    let mut cuts = [0., 1., 0., 0., 0., 0.];
    let mut count = 2;
    for corner in aperture.corners() {
        let ray = corner - center;
        let denom = span.perp_dot(ray);
        if denom.abs() < 1e-10 {
            continue;
        }
        let t = (center - b).perp_dot(ray) / denom;
        if (1e-5..1. - 1e-5).contains(&t) && (b + span * t - center).dot(ray) > 0. {
            cuts[count] = t;
            count += 1;
        }
    }
    cuts[..count].sort_unstable_by(f32::total_cmp);
    for pair in cuts[..count].windows(2) {
        let a = b.lerp(c, pair[0]);
        let d = b.lerp(c, pair[1]);
        let top = [a, d, at_cut(d), at_cut(a)];
        g.vertex_quad(
            top.map(|v| v.extend(LIP)),
            [Vec3::Z; 4],
            top.map(|v| v.to_array()),
        );
    }
}

fn flange(a: &mut Assembly, p: &FloorBox, size: Vec2) {
    let aperture = Rect::new(p.center(), p.hatch_size() + Vec2::splat(2. * JOINT));
    let g = a.part(p.outlet_finish, LABEL);
    let dirs = electrical::directions(24);
    let outer = |d: Vec2, q: Vec2| {
        let contour = match p.housing {
            Housing::Round => d,
            Housing::RoundedRectangle => q.lerp(d, p.corner),
            Housing::ClippedRectangle => q * ((2. - p.corner) / q.abs().element_sum()).min(1.),
        };
        contour * size * 0.5
    };
    for (i, &d) in dirs.iter().enumerate() {
        let e = dirs[(i + 1) % dirs.len()];
        let square = |d: Vec2| d / d.abs().max_element();
        let (q, t) = (square(d), square(e));
        let (b, c) = (outer(d, q), outer(e, t));
        let (bt, ct) = (b * 0.973, c * 0.973);
        // The entire stamped flange is cut from one sheet. A chart aligned
        // to each polygon edge would rotate the brushed PBR maps at every seam.
        let bevel = [b.extend(0.), c.extend(0.), ct.extend(LIP), bt.extend(LIP)];
        let normal = (bevel[1] - bevel[0]).cross(bevel[2] - bevel[0]).normalize();
        g.vertex_quad(bevel, [normal; 4], bevel.map(|v| [v.x, v.y]));
        flange_top(g, bt, ct, aperture);
    }
    // The gasket is recessed beneath the lip and meets the hatch on its own
    // boundary. A solid disk at this height would seal the access opening.
    let gasket = a.part(Surface::Rubber, LABEL);
    let seat = Rect::new(p.center(), p.hatch_size() - Vec2::splat(0.001));
    electrical::edge(gasket, aperture, LIP, seat, 0.0024, true);
    let throat = Rect::new(p.center(), p.hatch_size() - Vec2::splat(0.005));
    electrical::edge(gasket, seat, 0.0024, throat, 0.001, true);
}

fn lid(p: &FloorBox) -> Assembly {
    let mut a = Assembly::default();
    let (lo, hi) = p.lid_bounds();
    let outline = Rect {
        lo: lo.truncate(),
        hi: hi.truncate(),
    };
    let face = Rect {
        lo: outline.lo + Vec2::splat(0.0008),
        hi: outline.hi - Vec2::splat(0.0008),
    };
    let g = a.part(p.hatch_finish, LABEL);
    g.metric_quad(outline.corners().map(|v| v.extend(lo.z)), -Vec3::Z);
    electrical::edge(g, outline, lo.z, outline, 0.0006, false);
    electrical::edge(g, outline, 0.0006, face, hi.z, false);
    let pull = Rect::new(
        Vec2::new(p.pull_offset, -p.hatch_depth_m * 0.80),
        Vec2::new(p.pull_width, 0.0055),
    );
    electrical::pierced(g, face, hi.z, &[pull]);
    electrical::edge(g, pull, hi.z, pull, -0.0004, true);
    pull.face(a.part(Surface::Rubber, LABEL), -0.0004);
    a
}

fn cartridge(a: &mut Assembly, p: &FloorBox) {
    // The cartridge lifts above the finished floor when opened. Even the back
    // of a deep socket well stays above the continuous structural floor mesh.
    let skirt = Rect::new(p.center(), p.hatch_size() - Vec2::splat(0.005));
    let deck = Rect::new(p.center(), p.hatch_size() - Vec2::splat(0.008));
    let g = a.part(p.insert, LABEL);
    electrical::edge(g, skirt, 0., skirt, p.contact_height_m - 0.0022, false);
    electrical::edge(
        g,
        skirt,
        p.contact_height_m - 0.0022,
        deck,
        p.contact_height_m,
        false,
    );
    let contact = Contact {
        face: p.face,
        contact_scale: 1.,
        face_z_m: p.contact_height_m,
        well_depth_m: p.well_depth_m,
        insert: p.insert,
        pin_sides: 8,
        well_sides: 16,
    };
    for i in 0..p.socket_count {
        let mut cell = deck;
        if p.socket_count == 2 {
            if i == 0 {
                cell.hi.x = p.center_x;
            } else {
                cell.lo.x = p.center_x;
            }
        }
        electrical::insert(a, &contact, LABEL, cell, &[Vec2::X * p.port_x(i)], None);
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &FloorBox) {
    let mut frame = Assembly::default();
    flange(&mut frame, p, super::service_size(o).xz());
    if p.open {
        cartridge(&mut frame, p);
    }
    let hinge = p.hinge();
    let end = Vec3::X * (p.hatch_width_m * 0.5 + JOINT * 0.9);
    frame
        .part(Surface::Metal, LABEL)
        .rod(hinge.translation - end, hinge.translation + end, 0.0017);
    let up = Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2));
    public::append_at(a, frame, up);
    public::append_at(a, lid(p), up * hinge);
}

#[cfg(test)]
mod tests;
