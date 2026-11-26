//! Archived underside through generator 247.
use super::*;

pub(super) fn face(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &HoodProgram,
    base: f32,
    center: Vec3,
    chimney: Vec2,
) {
    let s = o.size;
    let label = o.kind.class_name();
    let filter_width = if p.construction == 2 {
        chimney.x
    } else {
        s.x * 0.72
    };
    let filter_depth = if p.construction == 2 {
        chimney.y
    } else {
        s.z * 0.55
    };
    let filter_center = if p.construction == 2 {
        center
    } else {
        Vec3::ZERO
    };
    for i in 0..p.filters {
        let w = filter_width / p.filters as f32;
        let x = filter_center.x + w * (i as f32 + 0.5) - filter_width * 0.5;
        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(x, base - 0.003, filter_center.z),
            Vec3::new(w * 0.94, 0.006, filter_depth),
            0.002,
        );
        for j in 0..6 {
            a.box_part(
                Surface::Rubber,
                label,
                Vec3::new(
                    x + w * (j as f32 - 2.5) * 0.12,
                    base - 0.0065,
                    filter_center.z,
                ),
                Vec3::new(w * 0.035, 0.002, filter_depth * 0.84),
                0.,
            );
        }
    }
    for i in 0..p.lights {
        let x = filter_center.x + filter_width * ((i as f32 + 0.5) / p.lights as f32 - 0.5);
        a.part(Surface::Light, "lamp").cylinder(
            0.018,
            0.004,
            Transform::from_xyz(x, base - 0.003, filter_center.z + filter_depth * 0.36),
        );
    }
}
