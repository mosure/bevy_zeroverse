//! A partly served tray exposes its real metal floor between grounded portions.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::{ProfileJoints, RevolvedOutline},
    materials::roast::Stock,
    placement::{rotated_half, Footprint},
};

pub(super) fn build(a: &mut Assembly, pan: &Pan, y: f32) {
    let mut r = stream(pan.seed, u64::from_le_bytes(*b"RoastPan"));
    let carrot = pan.contents == Contents::GlazedCarrots;
    let stock = if carrot { Stock::Carrot } else { Stock::Potato };
    let g = a.part(Surface::Terracotta, stock.label());
    let safe = pan.bottom() * 0.90;
    let count = (4. * safe.element_product() * (0.20 + pan.fill_fraction * 0.67) / 0.0023)
        .round()
        .clamp(2., 36.) as usize;
    let mut footprints: Vec<Footprint> = Vec::with_capacity(count);
    let shape = if carrot {
        RevolvedOutline::polygon(6)
    } else {
        RevolvedOutline::round(9, 0.14, 3)
    };
    // Sample broken-up clusters with some empty floor. Reject overlap in the
    // pan frame. Work is capped even when a heavily stocked narrow pan is full.
    let focus = safe * Vec2::new(r.random_range(-0.46..0.46), r.random_range(-0.46..0.46));
    for _ in 0..count * 6 {
        if footprints.len() == count {
            break;
        }
        let radius = r.random_range(0.015..0.029);
        let size = if carrot {
            Vec2::new(radius * r.random_range(1.25..1.78), radius * 0.53)
        } else {
            Vec2::new(radius * r.random_range(0.88..1.27), radius)
        };
        let yaw = r.random_range(-PI..PI);
        let half = rotated_half(size * 2., yaw);
        let span = safe - half;
        if span.min_element() <= 0. {
            continue;
        }
        let sample = Vec2::new(r.random_range(-1.0..1.), r.random_range(-1.0..1.));
        let center = if r.random_bool(0.45) {
            focus + span * sample * 0.63
        } else {
            span * sample
        };
        let f = Footprint { center, half, yaw };
        if !f.inside(-safe, safe, 0.) || footprints.iter().any(|&v| f.overlaps(v, 0.0007)) {
            continue;
        }
        footprints.push(f);
        let h = if carrot {
            radius * r.random_range(0.60..0.88)
        } else {
            radius * r.random_range(1.05..1.65)
        };
        let at = pan.center + center;
        let tf = Transform::from_xyz(at.x, y, at.y)
            .with_rotation(Quat::from_rotation_y(yaw))
            .with_scale(Vec3::new(size.x / radius, 1., size.y / radius));
        let first = g.uvs.len();
        shape.sweep(
            g,
            &[
                (0., 0.),
                (radius * 0.65, 0.),
                (radius, h * 0.34),
                (radius * 0.83, h * 0.76),
                (radius * 0.45, h * 0.96),
                (0., h),
            ],
            ProfileJoints::Smooth,
            tf,
        );
        let offset = Vec2::new(r.random::<f32>(), r.random::<f32>()) * 0.09;
        for uv in &mut g.uvs[first..] {
            *uv = (Vec2::from_array(*uv) + offset).to_array();
        }
    }
}
