//! Shared seeded construction contract for exit geometry, mounting and emission.
//! `variant` selects support geometry, never the regional legend.
use super::*;
mod graphics;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExitMount {
    Wall,
    CeilingCanopy,
    PendantRods,
    PendantCables,
}
#[derive(Debug, Clone, Copy, serde::Serialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExitLegend {
    EuropeanPictogram,
    AmericanRed,
    AmericanGreen,
}
pub fn mount(o: &IndoorObject) -> ExitMount {
    [
        ExitMount::Wall,
        ExitMount::CeilingCanopy,
        ExitMount::PendantRods,
        ExitMount::PendantCables,
    ][(o.variant % 4) as usize]
}
#[derive(Debug, Clone, serde::Serialize, PartialEq)]
pub struct ExitProgram {
    pub mount: ExitMount,
    pub legend: ExitLegend,
    pub body: Vec3,
    pub center: Vec3,
    pub bevel_fraction: f32,
    pub rail_mounted: bool,
    pub rail_fraction: f32,
    pub legend_fraction: Vec2,
    pub housing: Surface,
    pub housing_finish: usize,
    pub dark_face: bool,
    pub double_sided: bool,
    pub support_spacing: f32,
    pub support_radius: f32,
    pub canopy_depth: f32,
    pub luminance: f32,
}
pub fn parameters(o: &IndoorObject) -> ExitProgram {
    let mut r = stream(o.seed, 0x455849545349474e);
    let mount = mount(o);
    let legend = [
        ExitLegend::EuropeanPictogram,
        ExitLegend::AmericanRed,
        ExitLegend::AmericanGreen,
    ][r.random_range(0..3)];
    let width = o.size.x * r.random_range(0.84..0.99);
    let max_fraction = match mount {
        ExitMount::Wall => 0.94,
        ExitMount::CeilingCanopy => 0.80,
        _ => 0.65,
    };
    let height = (width * r.random_range(0.38..0.56)).min(o.size.y * max_fraction);
    let depth = o.size.z * r.random_range(0.40..0.76);
    ExitProgram {
        mount,
        legend,
        body: Vec3::new(width, height, depth),
        center: Vec3::new(
            0.,
            height * 0.5,
            if mount == ExitMount::Wall {
                (depth - o.size.z) * 0.5
            } else {
                0.
            },
        ),
        bevel_fraction: r.random_range(0.025..0.12),
        rail_mounted: r.random_bool(0.38),
        rail_fraction: r.random_range(0.10..0.19),
        legend_fraction: Vec2::new(r.random_range(0.75..0.90), r.random_range(0.57..0.72)),
        housing: [Surface::Plastic, Surface::Metal, Surface::Chrome][r.random_range(0..3)],
        housing_finish: [301, 303, 304][r.random_range(0..3)],
        dark_face: r.random_bool(0.45),
        double_sided: mount != ExitMount::Wall,
        support_spacing: width * r.random_range(0.42..0.74),
        support_radius: (depth.min(height)
            * r.random_range(0.035..0.065)
            * if mount == ExitMount::PendantCables {
                0.65
            } else {
                1.
            })
        .max(if mount == ExitMount::PendantCables {
            0.0012
        } else {
            0.0015
        })
        .min(depth * 0.2),
        canopy_depth: o.size.z * r.random_range(0.80..0.98),
        luminance: r.random_range(6_f32.ln()..60_f32.ln()).exp(),
    }
}
impl ExitProgram {
    /// Actual ceiling contact pads. Placement checks every corner on the roof.
    pub fn canopy_boxes(&self, size: Vec3) -> Vec<(Vec3, Vec3)> {
        if self.mount == ExitMount::Wall {
            return vec![];
        }
        let thickness = (size.y - self.body.y).min(0.012) * 0.4;
        let width = match self.mount {
            ExitMount::PendantRods => self.support_spacing + self.body.x * 0.16,
            _ => self.body.x * 0.22,
        };
        let xs = if self.mount == ExitMount::PendantCables {
            vec![-self.support_spacing * 0.5, self.support_spacing * 0.5]
        } else {
            vec![0.]
        };
        xs.into_iter()
            .map(|x| {
                (
                    Vec3::new(x, size.y - thickness * 0.5, self.center.z),
                    Vec3::new(width, thickness, self.canopy_depth),
                )
            })
            .collect()
    }
}
/// European white symbols and green panels have independent color/emission
/// channels, shared by both faces. No additional textures or shadow lights.
pub fn appearance(o: &IndoorObject, channel: usize) -> Option<(Color, LinearRgba)> {
    let p = parameters(o);
    let color = match (p.legend, channel) {
        (ExitLegend::EuropeanPictogram, 0) => Color::srgb(0.97, 1., 0.96),
        (ExitLegend::EuropeanPictogram, 1) => Color::srgb(0.015, 0.48, 0.15),
        (ExitLegend::AmericanRed, 0) => Color::srgb(0.92, 0.018, 0.009),
        (ExitLegend::AmericanGreen, 0) => Color::srgb(0.025, 0.68, 0.12),
        _ => return None,
    };
    let c = color.to_linear();
    Some((
        color,
        LinearRgba::new(
            c.red * p.luminance,
            c.green * p.luminance,
            c.blue * p.luminance,
            1.,
        ),
    ))
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let label = o.kind.class_name();
    let housing = if p.housing == Surface::Chrome {
        label.to_string()
    } else {
        format!("{label}#finish{}", p.housing_finish)
    };
    let b = p.body;
    let rail = b.y * p.rail_fraction;
    let (face_height, face_center, face_depth) = if p.rail_mounted {
        a.box_part(
            p.housing,
            &housing,
            p.center + Vec3::Y * (b.y - rail) * 0.5,
            Vec3::new(b.x, rail, b.z),
            rail * p.bevel_fraction,
        );
        let h = b.y - rail;
        let c = p.center - Vec3::Y * rail * 0.5;
        let d = b.z * 0.35;
        a.box_part(
            Surface::Ink,
            label,
            c,
            Vec3::new(b.x * 0.96, h, d),
            h * 0.01,
        );
        (h, c, d)
    } else {
        a.box_part(p.housing, &housing, p.center, b, b.y * p.bevel_fraction);
        (b.y * 0.84, p.center, b.z)
    };
    for rear in [false, true] {
        if rear && !p.double_sided {
            continue;
        }
        let mut face = Assembly::default();
        graphics::build(&mut face, o, &p, Vec2::new(b.x * 0.89, face_height));
        let sign = if rear { -1. } else { 1. };
        let tf =
            Transform::from_translation(face_center + Vec3::Z * sign * (face_depth * 0.5 + 0.001))
                .with_rotation(Quat::from_rotation_y(if rear { PI } else { 0. }));
        super::super::super::public::append_at(a, face, tf);
    }
    if p.mount != ExitMount::Wall {
        let contacts = p.canopy_boxes(o.size);
        for &(center, size) in &contacts {
            a.box_part(p.housing, &housing, center, size, 0.001);
        }
        let xs = if p.mount == ExitMount::CeilingCanopy {
            vec![0.]
        } else {
            vec![-p.support_spacing * 0.5, p.support_spacing * 0.5]
        };
        for x in xs {
            let support_label = if p.mount == ExitMount::PendantCables {
                format!("{label}#finish303") // coated suspension wire
            } else {
                label.to_string()
            };
            a.part(Surface::Metal, &support_label).rod(
                Vec3::new(x, b.y - 0.001, p.center.z),
                Vec3::new(x, contacts[0].0.y, p.center.z),
                p.support_radius,
            );
        }
    }
}
