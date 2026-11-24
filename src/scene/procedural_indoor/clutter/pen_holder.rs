//! Hollow desk cups and writing tools resting on the floor and inner lip.
use super::tabletop::writing;
use super::*;
use crate::scene::procedural_indoor::{
    geometry::{ProfileJoints, RevolvedOutline},
    materials::variants,
};

mod clearance;
mod packing;
mod pocket;
use pocket::{Edge, Pocket};
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, serde::Serialize)]
pub struct PenHolderProgram {
    pub height_fraction: f32,
    pub base_radius_fraction: f32,
    pub mouth_radius_fraction: f32,
    pub belly_height: f32,
    pub wall_fraction: f32,
    pub sides: u32,
    pub finish: Surface,
    pub requested_pens: usize,
    pub body_span_m: Vec2,
    pub floor_m: f32,
    pub lean_heading_rad: f32,
    pub lean_spread_rad: f32,
}

pub fn parameters(o: &IndoorObject) -> PenHolderProgram {
    let mut r = stream(o.seed, 0x50454e435550);
    let height_fraction = r.random_range(0.45..0.78);
    let base_radius_fraction = r.random_range(0.68..1.0);
    let mouth_radius_fraction = r.random_range(0.78..1.0);
    let belly_height = r.random_range(0.25..0.72);
    let gauge = (r.random_range(0.06_f32..0.14) - 0.06) / 0.08;
    let sides = [4, 6, 8, 28, 32, 32][r.random_range(0..6)];
    let finish = [
        Surface::Wood,
        Surface::Metal,
        Surface::Plastic,
        Surface::Ceramic,
    ][r.random_range(0..4)];
    let requested_pens = r.random_range(0..=9);
    let mut form = stream(o.seed, u64::from_le_bytes(*b"CupStock"));
    // The occupied envelope includes a real overhang beyond the rim.
    let span = o.size.xz() * form.random_range(0.72..0.92);
    let body_span_m = span * Vec2::new(1., form.random_range(0.80..1.04));
    let gauge_range = match finish {
        Surface::Metal => 0.0007_f32..0.0017,
        Surface::Wood => 0.0028..0.0052,
        Surface::Ceramic => 0.0024..0.0046,
        _ => 0.0011..0.0028,
    };
    let wall_m = gauge_range.start + gauge * (gauge_range.end - gauge_range.start);
    PenHolderProgram {
        height_fraction,
        base_radius_fraction,
        mouth_radius_fraction,
        belly_height,
        wall_fraction: (wall_m / (body_span_m.min_element() * 0.5)).clamp(0.015, 0.18),
        sides,
        finish,
        requested_pens,
        body_span_m,
        floor_m: (wall_m * form.random_range(1.15..2.15)).min(o.size.y * height_fraction * 0.13),
        lean_heading_rad: form.random_range(0.0..std::f32::consts::TAU),
        lean_spread_rad: form.random_range(0.35..std::f32::consts::PI),
    }
}

struct Cup<'a> {
    p: &'a PenHolderProgram,
    outline: RevolvedOutline,
    radius: f32,
    pocket: Pocket,
}
impl<'a> Cup<'a> {
    fn new(o: &IndoorObject, p: &'a PenHolderProgram) -> Self {
        let radius = p.body_span_m.min_element() * 0.5;
        let outline = RevolvedOutline::superellipse(p.sides, 2., p.body_span_m / (2. * radius));
        let mut edges = Vec::with_capacity(p.sides as usize);
        let mut last = None;
        for (point, _, _) in outline.meridians() {
            let end = point * radius;
            if let Some(start) = last {
                let chord: Vec2 = end - start;
                let normal = Vec2::new(chord.y, -chord.x).normalize();
                edges.push(Edge {
                    start,
                    end,
                    normal,
                    distance: normal.dot(start),
                });
            }
            last = Some(end);
        }
        let height = o.size.y * p.height_fraction;
        Self {
            p,
            outline: if p.sides <= 8 {
                outline.faceted()
            } else {
                outline
            },
            radius,
            pocket: Pocket {
                edges,
                span: p.body_span_m,
                height,
                belly_y: height * p.belly_height,
                floor: p.floor_m,
                base: p.base_radius_fraction,
                mouth: p.mouth_radius_fraction,
                wall: p.wall_fraction,
                count: p.requested_pens,
                heading: p.lean_heading_rad,
                spread: p.lean_spread_rad,
            },
        }
    }

    fn draw(&self, a: &mut Assembly, label: &str) {
        let p = self.p;
        let r = self.radius;
        let h = self.pocket.height;
        let wall = r * p.wall_fraction;
        let base = r * p.base_radius_fraction;
        let mouth = r * p.mouth_radius_fraction;
        let lip = wall.min(h * 0.025);
        // A rolled or eased mouth and foot have their own planar/curved charts.
        // The internal bearing face terminates exactly on the top inner edge.
        self.outline.sweep(
            a.part(p.finish, label),
            &[
                (0., 0.),
                (base - wall * 0.43, 0.),
                (base, wall * 0.43),
                (r, self.pocket.belly_y),
                (mouth, h - lip),
                (mouth + lip * 0.22, h - lip * 0.55),
                (mouth, h),
                (mouth - wall, h),
                (r - wall, self.pocket.belly_y),
                (base - wall, p.floor_m),
                (0., p.floor_m),
            ],
            ProfileJoints::Creased,
            Transform::IDENTITY,
        );
    }
}

#[derive(Debug, serde::Serialize)]
pub struct Pen {
    pub base: Vec3,
    pub top: Vec3,
    pub contact: Vec3,
    pub rotation: Quat,
    pub upper_end: writing::WritingEnd,
    pub tool: writing::WritingProgram,
    pub finish_slot: usize,
    #[serde(skip)]
    bounds: clearance::ToolBounds,
}

impl Pen {
    fn transform(&self) -> Transform {
        let axis_up = (self.rotation * Vec3::Y).y > 0.;
        Transform::from_translation(if axis_up { self.base } else { self.top })
            .with_rotation(self.rotation)
    }

    pub(crate) fn draw(&self, a: &mut Assembly, label: &str, offset: Vec3) {
        let mut tf = self.transform();
        tf.translation += offset;
        let barrel = format!("{label}#finish{}", self.finish_slot);
        self.tool.draw(a, tf, label, &barrel);
    }

    fn clearance_radius(&self) -> f32 {
        self.tool.radius * if self.tool.clip { 1.56 } else { 1. }
    }

    pub fn inclination_rad(&self) -> f32 {
        let d = self.top - self.base;
        d.xz().length().atan2(d.y)
    }
}

pub fn contents(o: &IndoorObject, p: &PenHolderProgram) -> Vec<Pen> {
    if p.requested_pens == 0 {
        return Vec::new();
    }
    packing::settle(o.seed, o.finish_seed, o.size, &Cup::new(o, p).pocket)
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let cup = Cup::new(o, &p);
    let label = o.kind.class_name();
    cup.draw(a, label);
    for pen in packing::settle(o.seed, o.finish_seed, o.size, &cup.pocket) {
        pen.draw(a, label, Vec3::ZERO);
    }
}

/// Settle the same finished tools inside a rectangular organizer compartment.
/// The envelope contains any overhang above the lip and is exclusive to this bay.
pub(crate) fn compartment_tools(
    o: &IndoorObject,
    index: usize,
    envelope: Vec3,
    clear: Vec2,
    floor: f32,
    rim: f32,
) -> Vec<Pen> {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"DeskPens") + index as u64);
    let seed = r.random();
    let pocket = Pocket::rectangle(clear, floor, rim, &mut r);
    packing::settle(seed, o.finish_seed, envelope, &pocket)
}
