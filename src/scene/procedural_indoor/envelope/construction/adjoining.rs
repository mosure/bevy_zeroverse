//! Exterior wall faces, jamb returns and base mouldings beyond the shared partition.
use super::*;
use architecture::{facade, trim::TrimProgram};

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
        let window = windows.filter(|w| w.edge == edge);
        let section = wall_section::WallSection::new(
            outline,
            edge,
            |v| e.ceiling_height(scene.room_size, v),
            |i| {
                if i == 0 {
                    SHARED_WALL_THICKNESS_M
                } else {
                    thickness
                }
            },
        );
        let label = if window.is_some() {
            "wall#exterior"
        } else {
            "wall"
        };
        let plaster = a.part(Surface::Paint, label);
        section.build(
            window.map_or(&[], |w| w.facade.openings.as_slice()),
            0.,
            |p, n, _| wall_face(plaster, p, n),
        );
        if let Some(w) = window {
            let mut joinery = Assembly::default();
            facade::glazing_at(&mut joinery, polygon::wall_frame(p, q), &w.facade);
            for ((surface, label), g) in joinery.parts {
                a.part(surface, &format!("{label}#exterior")).append(g);
            }
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

mod skirting;
#[cfg(test)]
mod tests;
