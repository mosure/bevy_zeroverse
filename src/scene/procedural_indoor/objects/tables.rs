//! Continuous slab, edge and leg construction within the placement envelope.
use super::*;
#[derive(Debug, Clone, serde::Serialize)]
pub struct TableProgram {
    pub top_thickness: f32,
    pub edge_bevel: f32,
    pub leg_radius: f32,
    pub leg_inset: f32,
    pub leg_rake: f32,
}
pub fn parameters(o: &IndoorObject) -> TableProgram {
    let mut rng = stream(o.seed, 172);
    TableProgram {
        top_thickness: rng.random_range(0.026..0.052),
        edge_bevel: rng.random_range(0.003..0.012),
        leg_radius: rng.random_range(0.018..0.036),
        leg_inset: rng.random_range(0.11..0.18),
        leg_rake: rng.random_range(0.0..0.05),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let s = o.size;
    let p = parameters(o);
    let underside = s.y - p.top_thickness - 0.014;
    let label = o.kind.class_name();
    a.box_part(
        Surface::Wood,
        label,
        Vec3::new(0.0, s.y - p.top_thickness * 0.5, 0.0),
        Vec3::new(s.x, p.top_thickness, s.z),
        p.edge_bevel,
    );
    if s.x > s.z {
        // The material's fibres run along texture V. Align the tabletop grain
        // with its long axis while retaining metre-scaled UVs and edge detail.
        for uv in &mut a.part(Surface::Wood, label).uvs {
            *uv = [uv[1], -uv[0]];
        }
    }
    a.box_part(
        Surface::WoodEdge,
        label,
        Vec3::new(0.0, s.y - p.top_thickness - 0.007, 0.0),
        Vec3::new(s.x - 0.022, 0.014, s.z - 0.022),
        0.004,
    );
    if o.variant == 2 && o.kind == ObjectKind::Table {
        for z in [-s.z * 0.29, s.z * 0.29] {
            a.part(Surface::Metal, label).cylinder(
                0.11,
                underside - 0.02,
                Transform::from_xyz(0.0, (underside - 0.02) * 0.5 + 0.02, z),
            );
            a.box_part(
                Surface::Metal,
                label,
                Vec3::new(0.0, 0.022, z),
                Vec3::new(s.x * 0.72, 0.044, 0.45),
                0.015,
            );
        }
    } else {
        let mat = if o.variant == 1 {
            Surface::WoodEdge
        } else {
            Surface::Metal
        };
        for x in [-1.0, 1.0] {
            for z in [-1.0, 1.0] {
                let top = Vec3::new(
                    x * (s.x * 0.5 - p.leg_inset),
                    underside,
                    z * (s.z * 0.5 - p.leg_inset),
                );
                let bottom = Vec3::new(
                    x * (s.x * 0.5 - p.leg_inset + p.leg_rake),
                    0.022,
                    z * (s.z * 0.5 - p.leg_inset + p.leg_rake),
                );
                a.part(mat, label).rod(bottom, top, p.leg_radius);
                a.part(Surface::Rubber, label).cylinder(
                    p.leg_radius + 0.003,
                    0.012,
                    Transform::from_translation(bottom.with_y(0.006)),
                );
            }
        }
        for x in [-s.x * 0.5 + 0.12, s.x * 0.5 - p.leg_inset] {
            a.box_part(
                mat,
                label,
                Vec3::new(x, underside - 0.0375, 0.0),
                Vec3::new(0.028, 0.075, s.z - 0.26),
                0.004,
            );
        }
        for z in [-s.z * 0.5 + 0.14, s.z * 0.5 - p.leg_inset] {
            a.box_part(
                mat,
                label,
                Vec3::new(0.0, underside - 0.030, z),
                Vec3::new(s.x - 0.22, 0.060, 0.035),
                0.004,
            );
        }
    }
    if o.kind != ObjectKind::CoffeeTable {
        a.box_part(
            Surface::Plastic,
            label,
            Vec3::new(0.0, underside - 0.07, 0.0),
            Vec3::new(s.x * 0.46, 0.045, 0.16),
            0.005,
        );
        // Flush cable grommet, with a separate dark slot.
        a.box_part(
            Surface::Metal,
            label,
            Vec3::new(0.0, s.y + 0.001, -s.z * 0.28),
            Vec3::new(0.16, 0.002, 0.055),
            0.0,
        );
    }
}
