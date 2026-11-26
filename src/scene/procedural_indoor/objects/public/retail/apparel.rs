use super::*;
use crate::scene::procedural_indoor::{
    materials::garment_stock,
    objects::utilities::racks::dressing::{Garment, GarmentKind},
};

pub(super) fn rail(
    a: &mut Assembly,
    s: Vec3,
    count: usize,
    collection_length: f32,
    frame: Surface,
    r: &mut ChaCha8Rng,
) {
    let span = s.x - 0.10;
    let bar = s.y - 0.036;
    let radius = 0.016;
    for side in [-1., 1.] {
        let x = side * span * 0.5;
        board(
            a,
            frame,
            FIXTURE,
            Vec3::new(x, 0.017, 0.),
            Vec3::new(0.065, 0.034, s.z * 0.92),
        );
        a.part(frame, FIXTURE)
            .rod(Vec3::new(x, 0.034, 0.), Vec3::new(x, bar, 0.), radius);
    }
    a.part(frame, FIXTURE).rod(
        Vec3::new(-span * 0.5, bar, 0.),
        Vec3::new(span * 0.5, bar, 0.),
        radius,
    );
    let pitch = (span - 0.12) / count as f32;
    let shoulder = (s.z * 0.63).clamp(0.36, 0.46);
    let neck = bar - 0.079;
    let hook = 0.019;
    let mut dye = garment_stock::sample_slot(r);
    let mut surface = Surface::Fabric;
    let mut cut = GarmentKind::Shirt;
    for i in 0..count {
        let x = (i as f32 + r.random_range(0.35..0.65) - count as f32 * 0.5) * pitch;
        if i == 0 || r.random_bool(0.36) {
            dye = garment_stock::sample_slot(r);
            cut = match r.random_range(0..10) {
                0..=4 => GarmentKind::Shirt,
                5..=7 => GarmentKind::Dress,
                _ if collection_length > 0.8 => GarmentKind::Overcoat,
                _ => GarmentKind::Jacket,
            };
            surface = if cut == GarmentKind::Jacket && r.random_bool(0.32) {
                Surface::Leather
            } else if r.random_bool(0.47) {
                Surface::FabricAlt
            } else {
                Surface::Fabric
            };
        }
        // Bend each hook around the actual rail. A triangular timber hanger
        // supports the sewn shoulders at the same slope as the garment cut.
        let wire = a.part(Surface::Chrome, FIXTURE);
        let mut previous = Vec3::new(x, neck, -hook);
        for step in 0..=5 {
            let angle = PI - step as f32 * PI / 5.;
            let q = Vec3::new(x, bar + angle.sin() * hook, angle.cos() * hook);
            wire.rod(previous, q, 0.0025);
            previous = q;
        }
        let apex = Vec3::new(x, neck, -hook);
        let half = shoulder * 0.5;
        let left = apex + Vec3::new(0., -half * 0.51, -half);
        let right = apex + Vec3::new(0., -half * 0.51, half);
        let hanger = a.part(Surface::Wood, FIXTURE);
        hanger.rod(apex, left, 0.006);
        hanger.rod(apex, right, 0.006);
        hanger.rod(left, right, 0.003);
        let (length, sleeve) = match cut {
            GarmentKind::Shirt => (
                collection_length.clamp(0.56, 0.72),
                r.random_range(0.40..0.53),
            ),
            GarmentKind::Dress => (
                (collection_length + 0.19).clamp(0.87, 1.22),
                r.random_range(0.40..0.56),
            ),
            _ => (collection_length, r.random_range(0.62..0.93)),
        };
        let g = Garment {
            kind: cut,
            support_index: i,
            anchor_m: apex,
            support_axis: Vec3::Z,
            support_radius_m: 0.006,
            shoulder_width_m: shoulder,
            width_m: (shoulder + r.random_range(0.075..0.125)).min(s.z - 0.07),
            length_m: (length * r.random_range(0.93..1.08)).min(neck - 0.12),
            half_depth_m: (pitch * 0.25).min(0.046),
            tail_fraction: sleeve,
            fold_phase: r.random_range(-PI..PI),
            fold_count: r.random_range(2..=4),
            closure: r.random_range(0..=1),
            surface,
            finish_slot: dye,
        };
        g.build(a);
    }
}

pub(super) fn folded(
    a: &mut Assembly,
    s: Vec3,
    piles: usize,
    layers: usize,
    frame: Surface,
    r: &mut ChaCha8Rng,
) {
    let t = r.random_range(0.026..0.043);
    let top = s.y - layers as f32 * t;
    table_base(a, s, top, frame);
    let pitch = (s.x - 0.10) / piles as f32;
    for i in 0..piles {
        let label = format!("clothes#finish{}", garment_stock::sample_slot(r));
        let surface = if r.random_bool(0.5) {
            Surface::Fabric
        } else {
            Surface::FabricAlt
        };
        let n = if i == 0 {
            layers
        } else {
            r.random_range(1..=layers)
        };
        let w = (pitch * 0.75).min(0.34);
        let depth = (s.z * r.random_range(0.54..0.72)).min(0.40);
        let x = pitch * (i as f32 + 0.5 - piles as f32 * 0.5);
        let z = s.z * r.random_range(-0.095..0.095);
        let align = r.random_range(-0.08..0.08);
        for j in 0..n {
            let heading = align + r.random_range(-0.024..0.024);
            let rot = Quat::from_rotation_y(heading);
            let size = Vec3::new(w, t, depth);
            let mid = Vec3::new(
                x + r.random_range(-0.009..0.009),
                top + (j as f32 + 0.5) * t,
                z + r.random_range(-0.009..0.009),
            );
            a.part(surface, &label).cuboid(
                size,
                t * 0.38,
                Transform::from_translation(mid).with_rotation(rot),
            );
            // A folded cuff wraps the outside edge; keep the shared contact
            // inside the roll, with a distinct narrow bound hem on its end.
            let hem = mid + rot * Vec3::new(w * 0.35, 0., depth * 0.5 + 0.001);
            a.part(surface, &label).cuboid(
                Vec3::new(w * 0.19, t * 0.42, 0.006),
                0.0018,
                Transform::from_translation(hem).with_rotation(rot),
            );
        }
    }
}
