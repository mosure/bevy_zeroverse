//! Bound printed objects with sampled construction, stacks and fitted covers.
use super::*;
mod opened;
#[derive(Debug, Clone, serde::Serialize)]
pub struct BindingProgram {
    pub volumes: usize,
    pub cover_fraction: f32,
    pub stack_yaw: f32,
    pub size_variation: f32,
    pub binding: u8,
    pub strap: bool,
    pub opened: bool,
    pub page_balance: f32,
    pub page_crown: f32,
    pub envelope_fraction: Vec3,
    pub cover: Surface,
    pub corner_fraction: f32,
    pub strap_width: f32,
    pub strap_position: f32,
    pub strap_thickness: f32,
    pub ring_spacing: f32,
}
pub fn parameters(o: &IndoorObject) -> BindingProgram {
    let mut r = stream(o.seed, 7410);
    let mut use_state = stream(o.seed, 7520);
    let mut product = stream(o.seed, 0x424f4f4b50524f44);
    let cover = if product.random_bool(0.035) {
        Surface::Wood
    } else {
        [
            Surface::Plastic,
            Surface::Plastic,
            Surface::Plastic,
            Surface::FabricAlt,
            Surface::FabricAlt,
            Surface::Leather,
        ][product.random_range(0..6)]
    };
    BindingProgram {
        envelope_fraction: Vec3::new(
            product.random_range(0.80..1.),
            product.random_range(0.64..0.98),
            product.random_range(0.80..1.),
        ),
        cover,
        corner_fraction: product.random_range(0.008..0.055),
        strap_width: product.random_range(0.022..0.075),
        strap_position: product.random_range(0.12..0.40),
        strap_thickness: product.random_range(0.0005..0.0013),
        ring_spacing: product.random_range(0.010..0.024),
        opened: o.kind == ObjectKind::Notebook && use_state.random_bool(0.42),
        page_balance: use_state.random_range(0.25..0.75),
        page_crown: use_state.random_range(0.08..0.26),
        volumes: if o.kind == ObjectKind::Books {
            r.random_range(2..=5)
        } else {
            1
        },
        cover_fraction: r.random_range(0.035..0.12),
        stack_yaw: r.random_range(-0.15..0.15),
        size_variation: r.random_range(0.04..0.20),
        binding: r.random_range(0..3),
        strap: r.random_bool(0.35),
    }
}
fn cover_outline(width: f32, depth: f32, corner: f32) -> [Vec2; 8] {
    let half = Vec2::new(width, depth) * 0.5;
    let corner = corner.max(0.0001).min(half.min_element() * 0.4);
    [
        Vec2::new(-half.x + corner, -half.y),
        Vec2::new(half.x - corner, -half.y),
        Vec2::new(half.x, -half.y + corner),
        Vec2::new(half.x, half.y - corner),
        Vec2::new(half.x - corner, half.y),
        Vec2::new(-half.x + corner, half.y),
        Vec2::new(-half.x, half.y - corner),
        Vec2::new(-half.x, -half.y + corner),
    ]
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    if p.opened {
        opened::build(a, o, &p);
        return;
    }
    let mut s = o.size * p.envelope_fraction;
    if p.strap {
        s.y = s.y.min(o.size.y - p.strap_thickness - 0.0002);
    }
    let mut r = stream(o.seed, 7411);
    let weights: Vec<f32> = (0..p.volumes).map(|_| r.random_range(0.6..1.4)).collect();
    let sum: f32 = weights.iter().sum();
    let mut y = 0.;
    for (i, weight) in weights.into_iter().enumerate() {
        let h = s.y * weight / sum;
        let t = (h * p.cover_fraction * if p.binding == 1 { 0.55 } else { 1. })
            .clamp(0.0003, 0.003)
            .min(h * 0.20);
        let angle = if p.volumes > 1 {
            r.random_range(-p.stack_yaw.abs().max(0.001)..p.stack_yaw.abs().max(0.001))
        } else {
            0.
        };
        let width = s.x * (1. - r.random_range(0.0..p.size_variation));
        let depth = s.z * (1. - r.random_range(0.0..p.size_variation));
        // Analytic rotated corner fit, not a hidden post-build bounds rescale.
        let factor = (s.x / (width * angle.cos().abs() + depth * angle.sin().abs()))
            .min(s.z / (depth * angle.cos().abs() + width * angle.sin().abs()))
            .min(1.);
        let width = width * factor;
        let depth = depth * factor;
        let half = super::super::placement::rotated_half(Vec2::new(width, depth), angle);
        let freedom = (o.size.xz() * 0.5 - half)
            .min(s.xz() * 0.025)
            .max(Vec2::ZERO);
        let tf = Transform::from_xyz(
            freedom.x * r.random_range(-1.0..1.0),
            y,
            freedom.y * r.random_range(-1.0..1.0),
        )
        .with_rotation(Quat::from_rotation_y(angle));
        // A mixed book stack must not inherit one cover material/slot merely
        // because it is batched as one room object.
        let cover = if i == 0 || r.random_bool(0.64) {
            p.cover
        } else {
            [Surface::Plastic, Surface::FabricAlt, Surface::Leather][r.random_range(0..3)]
        };
        let slot = super::super::materials::variants::object_slot(
            cover,
            o.seed ^ (i as u64).wrapping_mul(0x9e3779b97f4a7c15),
            None,
        );
        let l = format!("{}#finish{slot}", o.kind.class_name());
        let corner = (width.min(depth) * p.corner_fraction).min(0.007);
        for cy in [t * 0.5, h - t * 0.5] {
            // Thin covers can have appreciable plan-view corner radii without
            // inflating their thickness into a heavily bevelled solid box.
            let outline = cover_outline(width, depth, corner);
            let tf = tf.with_translation(tf.transform_point(Vec3::Y * cy));
            if t >= 0.00025 {
                a.part(cover, &l).profile_slab(&outline, t, t * 0.12, tf);
            } else {
                a.part(cover, &l)
                    .cuboid(Vec3::new(width, t, depth), t * 0.12, tf);
            }
        }
        // The page block shares the clipped outline, so cover corner changes
        // do not expose paper protruding through a closed notebook cover.
        let inset = (t * 1.5).max(0.0004);
        a.part(Surface::Paper, &l).profile_slab(
            &cover_outline(width - inset * 2., depth - inset * 2., corner - inset),
            h - t * 2.,
            t * 0.2,
            tf.with_translation(tf.transform_point(Vec3::Y * h * 0.5)),
        );
        if p.binding != 2 {
            a.part(cover, &l).cuboid(
                Vec3::new(t, h - t * 2., depth - corner * 2.),
                t * 0.2,
                tf.with_translation(tf.transform_point(Vec3::new(
                    -width * 0.5 + t * 0.5,
                    h * 0.5,
                    0.,
                ))),
            );
        } else {
            let wire_radius = (h * 0.025).min(0.0005);
            let ring_width = h * 0.10;
            // The cover was fitted against the rotated annotation envelope.
            // Keep the entire wire cross-section inside that fitted rectangle,
            // including for narrow, thick volumes in a tall stack.
            let ring_center = -width * 0.5 + ring_width + wire_radius;
            let rings = (depth / p.ring_spacing).round().clamp(5., 24.) as usize;
            for j in 0..rings {
                let z = (depth - inset * 2.) * ((j as f32 + 0.5) / rings as f32 - 0.5) * 0.91;
                let points: Vec<_> = (0..=12)
                    .map(|k| {
                        let t = TAU * k as f32 / 12.;
                        tf.transform_point(Vec3::new(
                            ring_center + ring_width * t.cos(),
                            h * (0.5 + 0.47 * t.sin()),
                            z,
                        ))
                    })
                    .collect();
                a.part(Surface::Metal, &l).tube(&points, wire_radius, 6);
            }
        }
        if p.strap && i == p.volumes - 1 {
            a.part(Surface::Rubber, &l).cuboid(
                Vec3::new(width * p.strap_width, p.strap_thickness, depth - inset * 2.),
                p.strap_thickness * 0.15,
                tf.with_translation(tf.transform_point(Vec3::new(
                    width * p.strap_position,
                    h + p.strap_thickness * 0.5 + 0.0001,
                    0.,
                ))),
            );
        }
        y += h;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bindings_fit_narrow_tall_stacks() {
        for seed in 0..256 {
            for size in [
                Vec3::new(0.12, 0.16, 0.174),
                Vec3::new(0.1444325, 0.15422024, 0.20178014),
            ] {
                let mut object = audit::fixture(ObjectKind::Books, seed);
                object.size = size;
                let assembly = build_object(&object);
                audit::check(&object, &assembly)
                    .unwrap_or_else(|e| panic!("seed={seed}, size={size:?}: {e}"));
            }
        }
    }
}
