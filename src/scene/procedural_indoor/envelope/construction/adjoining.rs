//! Exterior wall faces, jamb returns and base mouldings beyond the shared partition.
use super::*;
use architecture::{
    facade::{self, Facade},
    trim::TrimProgram,
};

pub(super) fn perimeter(
    a: &mut Assembly,
    scene: &IndoorManifest,
    e: &EnvelopeProgram,
    outline: &[Vec2],
) {
    let elevation = scene.adjoining_elevation.as_ref();
    let windows = elevation.and_then(|p| p.windows.as_ref());
    let thickness = elevation.map_or(0.18, |_| e.wall_thickness());
    for (edge, (p, q)) in polygon::edges(outline).enumerate().skip(1) {
        let height = Vec2::new(
            e.ceiling_height(scene.room_size, p),
            e.ceiling_height(scene.room_size, q),
        );
        let point = |p: Vec2, y| Vec3::new(p.x, y, p.y);
        if let Some(w) = windows.filter(|w| w.edge == edge) {
            let tf = polygon::wall_frame(p, q);
            let span = p.distance(q);
            let mut joinery = Assembly::default();
            punched_wall(
                a.part(Surface::Paint, "wall#exterior"),
                tf,
                span,
                height,
                thickness,
                &w.facade,
            );
            facade::glazing_at(&mut joinery, tf, &w.facade);
            for ((surface, label), g) in joinery.parts {
                a.part(surface, &format!("{label}#exterior")).append(g);
            }
        } else {
            let normal = Vec3::new(p.y - q.y, 0., q.x - p.x).normalize();
            panel(
                a.part(Surface::Paint, "wall"),
                &[
                    point(p, 0.),
                    point(q, 0.),
                    point(q, height.y),
                    point(p, height.x),
                ],
                normal * thickness,
            );
        }
    }
    if let Some(trim) = elevation
        .filter(|p| p.continue_skirting)
        .and_then(|_| TrimProgram::for_scene(scene))
    {
        if let Some(section) = &trim.base {
            skirting::build(a, outline, trim.joint_m, section, windows, scene.seed);
        }
    }
}

fn punched_wall(
    g: &mut Geometry,
    frame: Transform,
    span: f32,
    height: Vec2,
    depth: f32,
    facade: &Facade,
) {
    let half = span * 0.5;
    let slope = (height.y - height.x) / span;
    let midpoint = height.element_sum() * 0.5;
    let front = frame.rotation * Vec3::Z;
    let at = |v: Vec2, z| frame.transform_point(v.extend(z));
    for (lo, hi) in facade::solid_rectangles(
        Vec2::new(-half, 0.),
        Vec2::new(half, height.max_element()),
        &facade.openings,
    ) {
        let rect = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)];
        let clipped = polygon::clip(&rect, Vec2::new(slope, -1.), -midpoint);
        for (z, n) in [(0., front), (-depth, -front)] {
            let points: Vec<_> = clipped.iter().map(|&v| at(v, z)).collect();
            wall_face(g, &points, n);
        }
    }
    // Cap only physical edges. Creating a closed box for every remaining
    // rectangle would leave duplicate vertical faces at the slab subdivisions.
    let mut returns = |ring: [Vec2; 4], sign: f32| {
        for i in 0..4 {
            let (a, b) = (ring[i], ring[(i + 1) % 4]);
            let d = b - a;
            let normal = frame.rotation * Vec3::new(d.y, -d.x, 0.) * sign;
            wall_face(
                g,
                &[at(a, 0.), at(b, 0.), at(b, -depth), at(a, -depth)],
                normal,
            );
        }
    };
    returns(
        [
            Vec2::new(-half, 0.),
            Vec2::new(half, 0.),
            Vec2::new(half, height.y),
            Vec2::new(-half, height.x),
        ],
        1.,
    );
    for o in &facade.openings {
        returns(
            [
                o.min,
                Vec2::new(o.max.x, o.min.y),
                o.max,
                Vec2::new(o.min.x, o.max.y),
            ],
            -1.,
        );
    }
}

mod skirting;
#[cfg(test)]
mod tests;
