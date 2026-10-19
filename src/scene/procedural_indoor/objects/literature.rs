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
}
pub fn parameters(o: &IndoorObject) -> BindingProgram {
    let mut r = stream(o.seed, 7410);
    let mut use_state = stream(o.seed, 7520);
    BindingProgram {
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
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    if p.opened {
        opened::build(a, o, &p);
        return;
    }
    let s = o.size;
    let l = o.kind.class_name();
    let mut r = stream(o.seed, 7411);
    let weights: Vec<f32> = (0..p.volumes).map(|_| r.random_range(0.6..1.4)).collect();
    let sum: f32 = weights.iter().sum();
    let mut y = 0.;
    for (i, weight) in weights.into_iter().enumerate() {
        let h = s.y * weight / sum;
        let t = (h * p.cover_fraction).min(0.003);
        let angle = if p.volumes > 1 {
            p.stack_yaw * (i as f32 - (p.volumes - 1) as f32 * 0.5)
        } else {
            0.
        };
        let width = s.x * (1. - p.size_variation * i as f32 / p.volumes as f32);
        let depth = s.z * (1. - p.size_variation * (p.volumes - 1 - i) as f32 / p.volumes as f32);
        // Analytic rotated corner fit, not a hidden post-build bounds rescale.
        let factor = (s.x / (width * angle.cos().abs() + depth * angle.sin().abs()))
            .min(s.z / (depth * angle.cos().abs() + width * angle.sin().abs()))
            .min(1.);
        let width = width * factor;
        let depth = depth * factor;
        let tf = Transform::from_xyz(0., y, 0.).with_rotation(Quat::from_rotation_y(angle));
        let mut part = |surface, pos: Vec3, size: Vec3, bevel| {
            a.part(surface, l)
                .cuboid(size, bevel, tf.with_translation(tf.transform_point(pos)));
        };
        let cover = if p.binding == 1 {
            Surface::FabricAlt
        } else {
            Surface::Plastic
        };
        for cy in [t * 0.5, h - t * 0.5] {
            part(cover, Vec3::Y * cy, Vec3::new(width, t, depth), t * 0.25);
        }
        part(
            Surface::Paper,
            Vec3::new(0., h * 0.5, 0.),
            Vec3::new(width - t * 3., h - t * 2., depth - t * 3.),
            t * 0.2,
        );
        if p.binding != 2 {
            part(
                cover,
                Vec3::new(-width * 0.5 + t * 0.5, h * 0.5, 0.),
                Vec3::new(t, h, depth),
                t * 0.2,
            );
        } else {
            let wire_radius = (h * 0.025).min(0.0005);
            let ring_width = h * 0.10;
            // The cover was fitted against the rotated annotation envelope.
            // Keep the entire wire cross-section inside that fitted rectangle,
            // including for narrow, thick volumes in a tall stack.
            let ring_center = -width * 0.5 + ring_width + wire_radius;
            for j in 0..12 {
                let z = depth * ((j as f32 + 0.5) / 12. - 0.5) * 0.91;
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
                a.part(Surface::Metal, l).tube(&points, wire_radius, 6);
            }
        }
        if p.strap && i == p.volumes - 1 {
            a.part(Surface::Rubber, l).cuboid(
                Vec3::new(width * 0.035, t * 0.2, depth * 0.97),
                t * 0.05,
                tf.with_translation(tf.transform_point(Vec3::new(width * 0.30, h + t * 0.15, 0.))),
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
