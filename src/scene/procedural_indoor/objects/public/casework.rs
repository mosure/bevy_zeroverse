//! Vitrines, tray stations and plinths with shared load-bearing geometry.
use super::*;
pub mod plinth;
mod service_glazing;
mod stocking;
#[cfg(test)]
mod tests;
pub mod tray_return;
pub use service_glazing::{CaseDoor, ServiceGlazing, ShelfEdge};

#[derive(Clone, Debug, serde::Serialize)]
pub struct CaseProgram {
    pub structure: u8,
    pub base_fraction: f32,
    pub wall_fraction: f32,
    pub taper: f32,
    pub foot_span: f32,
    pub levels: Vec<f32>,
    pub contents: bool,
    pub glazed_top: bool,
    pub frame: Surface,
    pub finish: Surface,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub service_glazing: Option<ServiceGlazing>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub shelf_edge: Option<ShelfEdge>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tray_station: Option<tray_return::TrayStation>,
}
pub fn parameters(o: &IndoorObject) -> CaseProgram {
    parameters_at(
        o,
        crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
    )
}

pub(crate) fn parameters_at(o: &IndoorObject, version: u32) -> CaseProgram {
    let mut r = stream(o.seed, 0x43415345574f524b);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x4341534546494e);
    let base = if o.kind == ObjectKind::DisplayCase {
        r.random_range(0.12..0.58)
    } else {
        r.random_range(0.055..0.16)
    };
    let n = r.random_range(1..=4);
    let levels = (0..n)
        .map(|i| base + 0.025 + (0.90 - base) * (i as f32 + r.random_range(0.02..0.24)) / n as f32)
        .collect();
    let structure = r.random_range(0..4);
    let mut p = CaseProgram {
        structure,
        base_fraction: base,
        wall_fraction: r.random_range(0.018..0.045),
        taper: r.random_range(0.60..0.95),
        foot_span: r.random_range(0.65..0.90),
        levels,
        contents: r.random_bool(0.88),
        glazed_top: r.random_bool(0.64),
        frame: [Surface::Metal, Surface::Wood, Surface::Plastic][f.random_range(0..3)],
        finish: [
            Surface::Wood,
            Surface::Metal,
            Surface::Plastic,
            Surface::Concrete,
        ][f.random_range(0..4)],
        service_glazing: (o.kind == ObjectKind::DisplayCase).then(|| ServiceGlazing::sample(o)),
        shelf_edge: if o.kind == ObjectKind::DisplayCase && structure == 1 {
            ShelfEdge::sample(o)
        } else {
            None
        },
        tray_station: None,
    };
    if o.kind == ObjectKind::TrayReturn && version >= tray_return::INTRODUCED {
        let station = tray_return::TrayStation::fit(o, &p);
        p.frame = Surface::Metal;
        p.finish = match f.random_range(0..8) {
            0 if station.enclosed => Surface::Wood,
            0..=2 => Surface::Plastic,
            _ => Surface::Metal,
        };
        p.levels = station.bearings_m.iter().map(|y| y / o.size.y).collect();
        p.contents = !station.loads.is_empty();
        p.glazed_top = false;
        p.tray_station = Some(station);
    }
    p
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if o.kind == ObjectKind::ExhibitPlinth {
        return plinth::build(a, o);
    }
    let p = parameters_at(o, version);
    match o.kind {
        ObjectKind::DisplayCase => display(a, o, &p),
        ObjectKind::TrayReturn => match &p.tray_station {
            Some(station) => station.build(a, o, &p),
            None => legacy_trays(a, o, &p),
        },
        _ => unreachable!(),
    }
}
fn feet(a: &mut Assembly, o: &IndoorObject, p: &CaseProgram, top: f32) {
    let radius = o.size.x.min(o.size.z) * p.wall_fraction * 0.7;
    for x in [-p.foot_span * 0.5, p.foot_span * 0.5] {
        for z in [-p.foot_span * 0.5, p.foot_span * 0.5] {
            a.part(p.frame, o.kind.class_name()).rod(
                Vec3::new(x * o.size.x, radius, z * o.size.z),
                Vec3::new(x * o.size.x * p.taper, top, z * o.size.z * p.taper),
                radius,
            );
            a.part(Surface::Rubber, o.kind.class_name()).cylinder(
                radius * 1.4,
                radius,
                Transform::from_xyz(x * o.size.x, radius * 0.5, z * o.size.z),
            );
        }
    }
}
fn display(a: &mut Assembly, o: &IndoorObject, p: &CaseProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let base = s.y * p.base_fraction;
    let glazing = p
        .service_glazing
        .expect("display cases have service glazing");
    let cabinet = service_glazing::Cabinet::new(s, p, glazing);
    let frame = cabinet.frame;
    if p.structure < 2 {
        a.box_part(
            p.finish,
            l,
            Vec3::Y * (base - frame * 0.5) * 0.5,
            Vec3::new(s.x * 0.96, base - frame * 0.5, s.z * 0.96),
            frame * 0.3,
        );
    } else {
        feet(a, o, p, base - frame * 0.5);
    }
    a.box_part(
        p.frame,
        l,
        Vec3::Y * base,
        Vec3::new(s.x, frame, s.z),
        0.002,
    );
    cabinet.enclosure(a, p, glazing);
    let (shelf_stock, shelf_label) = if p.structure == 1 {
        (
            Surface::ContainerGlass,
            crate::scene::procedural_indoor::materials::glass::display::SheetGauge::Eight.label(),
        )
    } else {
        (p.frame, l)
    };
    for (i, &level) in p.levels.iter().enumerate() {
        let shelf = cabinet.shelf_height(level, s.y);
        let (lo, hi) = cabinet.shelf;
        cabinet.shelf_supports(a, p.frame, shelf, glazing.pane.metres());
        let glass_end = cabinet.shelf_front(a, shelf);
        a.box_part(
            shelf_stock,
            shelf_label,
            Vec3::new(0., shelf, (lo.y + glass_end) * 0.5),
            Vec3::new(hi.x - lo.x, cabinet.shelf_thickness, glass_end - lo.y),
            0.001,
        );
        let support_y = shelf + cabinet.shelf_thickness * 0.5;
        let above = p.levels.get(i + 1).map_or(cabinet.head, |next| {
            cabinet.shelf_height(*next, s.y) - cabinet.shelf_thickness * 0.5
        });
        if p.contents {
            contents(a, o, i, support_y, above - support_y - 0.012, cabinet.shelf);
        }
    }
}
fn contents(
    a: &mut Assembly,
    o: &IndoorObject,
    level: usize,
    y: f32,
    headroom: f32,
    bounds: (Vec2, Vec2),
) {
    use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};
    let mut r = stream(o.seed, 0x43415345434f4e54 + level as u64);
    let pattern = SurfacePattern::sample(&mut r);
    let mut occupied: Vec<Footprint> = Vec::new();
    if o.variant < 16 {
        occupied.extend(stocking::bakery_tray(a, o, level, y, headroom, bounds));
    }
    for _ in 0..r.random_range(1..=5) {
        let mut child = o.clone();
        child.kind = if o.variant >= 16 {
            [ObjectKind::Sculpture, ObjectKind::Vase, ObjectKind::Bowl][r.random_range(0..3)]
        } else {
            [ObjectKind::Bread, ObjectKind::Dishware, ObjectKind::Mug][r.random_range(0..3)]
        };
        child.seed = r.random();
        child.position = Vec3::ZERO;
        child.yaw = 0.;
        child.finish_seed = None;
        child.variant = (child.seed % 8) as u32;
        child.size = hospitality::content_size(child.kind, child.seed, &mut r);
        if child.size.y > headroom {
            continue;
        }
        for _ in 0..16 {
            let yaw = pattern.yaw(&mut r);
            let Some(proposal) = pattern.propose(bounds, child.size.xz(), yaw, 0.013, &mut r)
            else {
                continue;
            };
            if occupied.iter().any(|q| proposal.overlaps(*q, 0.012)) {
                continue;
            }
            occupied.push(proposal);
            append_at(
                a,
                super::super::build_object(&child),
                Transform::from_xyz(proposal.center.x, y, proposal.center.y)
                    .with_rotation(Quat::from_rotation_y(yaw)),
            );
            break;
        }
    }
}
fn legacy_trays(a: &mut Assembly, o: &IndoorObject, p: &CaseProgram) {
    let s = o.size;
    let l = o.kind.class_name();
    let t = (s.x.min(s.z) * p.wall_fraction).max(0.008);
    let bottom = s.y * p.base_fraction;
    feet(a, o, p, bottom);
    if p.structure < 2 {
        for x in [-1., 1.] {
            a.box_part(
                p.finish,
                l,
                Vec3::new(x * (s.x - t) * 0.5, (bottom + s.y) * 0.5, 0.),
                Vec3::new(t, s.y - bottom, s.z),
                0.002,
            );
        }
        a.box_part(
            p.finish,
            l,
            Vec3::new(0., (bottom + s.y) * 0.5, -(s.z - t) * 0.5),
            Vec3::new(s.x - t * 2., s.y - bottom, t),
            0.002,
        );
    } else {
        for x in [-0.47, 0.47] {
            for z in [-0.47, 0.47] {
                rod(
                    a,
                    o,
                    p.frame,
                    l,
                    [x, p.base_fraction, z],
                    [x, 0.99, z],
                    t * 0.5,
                );
            }
        }
    }
    a.box_part(
        p.finish,
        l,
        Vec3::Y * (s.y - t * 0.5),
        Vec3::new(s.x, t, s.z),
        0.002,
    );
    let mut r = stream(o.seed, 0x54524159524f57);
    for &level in &p.levels {
        let y = level * s.y;
        a.box_part(
            p.frame,
            l,
            Vec3::Y * y,
            Vec3::new(s.x - t * 2.2, 0.008, s.z - t * 2.2),
            0.001,
        );
        if p.contents && r.random_bool(0.85) {
            let w = s.x * r.random_range(0.53..0.76);
            let d = s.z * r.random_range(0.58..0.78);
            let yaw = r.random_range(-0.12..0.12);
            let half = super::super::super::placement::rotated_half(Vec2::new(w, d), yaw);
            let free = (s.xz() * 0.5 - Vec2::splat(t * 1.2) - half).max(Vec2::splat(0.001));
            let tf = Transform::from_xyz(
                r.random_range(-free.x..free.x),
                y + 0.008,
                r.random_range(-free.y..free.y),
            )
            .with_rotation(Quat::from_rotation_y(yaw));
            let mut tray = Assembly::default();
            tray.box_part(
                Surface::Plastic,
                "dishware",
                Vec3::ZERO,
                Vec3::new(w, 0.008, d),
                0.002,
            );
            for x in [-1., 1.] {
                tray.box_part(
                    Surface::Plastic,
                    "dishware",
                    Vec3::new(x * (w - 0.006) * 0.5, 0.009, 0.),
                    Vec3::new(0.006, 0.018, d),
                    0.002,
                );
            }
            for z in [-1., 1.] {
                tray.box_part(
                    Surface::Plastic,
                    "dishware",
                    Vec3::new(0., 0.009, z * (d - 0.006) * 0.5),
                    Vec3::new(w - 0.012, 0.018, 0.006),
                    0.002,
                );
            }
            append_at(a, tray, tf);
        }
    }
}
