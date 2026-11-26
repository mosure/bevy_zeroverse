//! Bound printed objects with sampled construction, stacks and fitted covers.
use super::*;
mod jacket;
pub(crate) mod media;
mod opened;
pub(crate) mod spread;

/// Earlier manifests use variants 0..4 and retain their original mesh and ink.
pub(crate) const SPREAD_BINDING: u32 = 1 << 8;
/// A single closed reading copy in the same fitted binding program as a stack.
pub(crate) const SINGLE_VOLUME: u32 = 1 << 9;
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
    parameters_at(o, super::super::layout::GENERATOR_VERSION)
}

pub(crate) fn parameters_at(o: &IndoorObject, version: u32) -> BindingProgram {
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
    let mut p = BindingProgram {
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
        volumes: if o.kind == ObjectKind::Books && o.variant & SINGLE_VOLUME == 0 {
            r.random_range(2..=5)
        } else {
            1
        },
        cover_fraction: r.random_range(0.035..0.12),
        stack_yaw: r.random_range(-0.15..0.15),
        size_variation: r.random_range(0.04..0.20),
        binding: r.random_range(0..3),
        strap: r.random_bool(0.35),
    };
    if o.kind == ObjectKind::Books && version >= super::super::materials::paper::jacket::INTRODUCED
    {
        let mut stock = stream(o.seed, u64::from_le_bytes(*b"BookCase"));
        // A single field manual can be wire-bound. A stack of reading copies
        // is much more often a mixture of sewn cases and glued paperbacks.
        let single = p.volumes == 1;
        if p.binding == 2 && !stock.random_bool(if single { 0.20 } else { 0.08 }) {
            p.binding = u8::from(stock.random_bool(0.58));
        }
        p.strap &= stock.random_bool(if single { 0.12 } else { 0.035 });
        if stock.random_bool(0.76) {
            p.cover = match p.binding {
                1 => Surface::Paper,
                0 if stock.random_bool(0.63) => Surface::FabricAlt,
                0 => Surface::Paper,
                _ => Surface::Plastic,
            };
        }
    }
    p
}
/// Manufacturing and pose decisions for a single closed volume in a stack.
pub(crate) struct Volume {
    pub binding: u8,
    pub back_up: bool,
    pub reversed: bool,
    edition: u64,
    print: Option<usize>,
}

impl BindingProgram {
    pub(crate) fn volume(&self, o: &IndoorObject, version: u32, index: usize) -> Volume {
        use super::super::materials::paper::jacket;
        let edition = o.seed ^ (index as u64).wrapping_mul(0x9e3779b97f4a7c15);
        let current = version >= jacket::INTRODUCED;
        let binding = if current
            && index > 0
            && self.binding < 2
            && stream(edition, u64::from_le_bytes(*b"BindMix ")).random_bool(0.28)
        {
            1 - self.binding
        } else {
            self.binding
        };
        let (back_up, reversed) = if current {
            let mut resting = stream(edition, u64::from_le_bytes(*b"BookPose"));
            let has_strap = self.strap && index + 1 == self.volumes;
            // A closure protrudes from the front board. Keep it off the support.
            let back_up = !has_strap
                && resting.random_bool(if o.kind == ObjectKind::Notebook {
                    0.09
                } else {
                    0.19
                });
            (back_up, index > 0 && resting.random_bool(0.27))
        } else {
            (false, false)
        };
        Volume {
            edition,
            binding,
            back_up,
            reversed,
            print: jacket::slot(version, edition, o.kind == ObjectKind::Notebook),
        }
    }
}
impl Volume {
    fn pose(&self, mut tf: Transform, height: f32) -> Transform {
        // Exact half-turns preserve the admitted footprint, including slack
        // between volumes, and keep both outer boards on the same support planes.
        if self.reversed {
            tf.rotation *= Quat::from_xyzw(0., 1., 0., 0.);
        }
        if self.back_up {
            tf.translation = tf.transform_point(Vec3::Y * height);
            tf.rotation *= Quat::from_xyzw(0., 0., 1., 0.);
        }
        tf
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
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters_at(o, version);
    if p.opened {
        if o.variant & SPREAD_BINDING == 0 {
            opened::build(a, o, &p);
        } else {
            spread::build(a, o, &p);
        }
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
        let volume = p.volume(o, version, i);
        let binding = volume.binding;
        let h = s.y * weight / sum;
        let t = (h * p.cover_fraction * if binding == 1 { 0.55 } else { 1. })
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
        let tf = volume.pose(tf, h);
        // A mixed book stack must not inherit one cover material/slot merely
        // because it is batched as one room object.
        let cover = if i == 0 || r.random_bool(0.64) {
            p.cover
        } else {
            [Surface::Plastic, Surface::FabricAlt, Surface::Leather][r.random_range(0..3)]
        };
        let cover = if binding != p.binding {
            if binding == 1 {
                Surface::Paper
            } else {
                Surface::FabricAlt
            }
        } else {
            cover
        };
        let slot = super::super::materials::variants::object_slot(cover, volume.edition, None);
        let print = volume
            .print
            .filter(|_| t >= 0.00025)
            .map(|s| format!("{}#finish{s}#uv_local", o.kind.class_name()));
        let l = format!("{}#finish{slot}", o.kind.class_name());
        let corner = (width.min(depth) * p.corner_fraction).min(0.007);
        let outline = cover_outline(width, depth, corner);
        for (top, cy) in [(false, t * 0.5), (true, h - t * 0.5)] {
            // Thin covers can have appreciable plan-view corner radii without
            // inflating their thickness into a heavily bevelled solid box.
            let tf = tf.with_translation(tf.transform_point(Vec3::Y * cy));
            if let Some(print) = &print {
                jacket::board(
                    a,
                    &outline,
                    (cover, &l),
                    print,
                    Vec3::new(width, t, depth),
                    tf,
                    top,
                );
            } else if t >= 0.00025 {
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
        if binding != 2 {
            let size = Vec3::new(t, h - t * 2., depth - corner * 2.);
            let spine = tf.with_translation(tf.transform_point(Vec3::new(
                -width * 0.5 + t * 0.5,
                h * 0.5,
                0.,
            )));
            if let Some(print) = &print {
                jacket::spine(a.part(Surface::PrintedPaper, print), size, t * 0.2, spine);
            } else {
                a.part(cover, &l).cuboid(size, t * 0.2, spine);
            }
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
    fn reading_copies_and_notebooks_have_distinct_binding_distributions() {
        let version = super::super::super::materials::paper::jacket::INTRODUCED;
        for (kind, single) in [
            (ObjectKind::Books, true),
            (ObjectKind::Books, false),
            (ObjectKind::Notebook, true),
        ] {
            let mut wire = 0;
            let mut straps = 0;
            let mut soft = 0;
            let mut old_wire = 0;
            let mut mixed_stacks = 0;
            let mut backs = 0;
            let mut reversals = 0;
            for seed in 0..1024 {
                let mut o = audit::fixture(kind, seed);
                o.variant = if single { SINGLE_VOLUME } else { 0 };
                let now = parameters_at(&o, version);
                let old = parameters_at(&o, version - 1);
                let first = now.volume(&o, version, 0);
                backs += usize::from(first.back_up);
                assert!(!first.reversed);
                assert!(!(now.volumes == 1 && now.strap && first.back_up));
                let legacy = old.volume(&o, version - 1, 0);
                assert!(!legacy.back_up && !legacy.reversed && legacy.print.is_none());
                if (1..now.volumes).any(|i| now.volume(&o, version, i).reversed) {
                    reversals += 1;
                }
                if (1..now.volumes).any(|i| now.volume(&o, version, i).binding != now.binding) {
                    mixed_stacks += 1;
                }
                old_wire += usize::from(old.binding == 2);
                wire += usize::from(now.binding == 2);
                straps += usize::from(now.strap);
                soft += usize::from(now.binding == 1);
                assert_eq!((now.volumes, now.opened), (old.volumes, old.opened));
                if kind == ObjectKind::Notebook {
                    assert_eq!(
                        (now.binding, now.strap, now.cover),
                        (old.binding, old.strap, old.cover)
                    );
                }
            }
            assert!((275..405).contains(&old_wire));
            assert!(
                (if kind == ObjectKind::Notebook {
                    30..100
                } else {
                    125..245
                })
                .contains(&backs),
                "{backs}"
            );
            if !single {
                assert!((360..690).contains(&reversals), "{reversals}");
                assert!((360..690).contains(&mixed_stacks), "{mixed_stacks}");
            }
            if kind == ObjectKind::Books {
                assert!(
                    (if single { 35..100 } else { 8..55 }).contains(&wire),
                    "{wire}"
                );
                assert!(
                    (if single { 15..70 } else { 2..35 }).contains(&straps),
                    "{straps}"
                );
                assert!((410..580).contains(&soft), "{soft}");
            }
        }
    }

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
