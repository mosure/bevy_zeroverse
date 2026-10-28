//! Fit exit signs to the real entry header or ceiling above its approach.
//! Bounded alternatives fall back to a wall fixture, never to floating supports.
use super::*;
use signals::exit::{self, ExitMount};

pub(super) fn mounted(scene: &IndoorManifest, o: &IndoorObject) -> bool {
    let Some(d) = &scene.entry_door else {
        return false;
    };
    let (lo, hi) = o.bounds();
    if (o.position.x - d.center.x).abs() + o.size.x * 0.5 >= d.width * 0.5 - 0.03
        || (o.transform().rotation * Vec3::Z).dot(Vec3::NEG_Z) < 0.99999
        || lo.y <= d.center.y + d.height + 0.035
    {
        return false;
    }
    let corners = [
        lo.xz(),
        hi.xz(),
        Vec2::new(lo.x, hi.z),
        Vec2::new(hi.x, lo.z),
    ];
    let p = exit::parameters(o);
    if p.mount == ExitMount::Wall {
        return (o.position.z - (d.center.z - o.size.z * 0.5 - 0.001)).abs() < 0.001
            && corners
                .into_iter()
                .all(|q| hi.y < scene.ceiling_height(q) - 0.06);
    }
    let inward = d.center.z - o.position.z;
    if !(0.20..=0.75).contains(&inward) {
        return false;
    }
    if !corners.into_iter().all(|q| {
        scene.in_primary_room(Vec3::new(q.x, lo.y, q.y), 0.05)
            && hi.y <= scene.ceiling_height(q) + 0.003
            && lo.y > scene.floor_height(q) + 2.12
    }) || lo.y <= architecture::ceiling::highest_floor(scene, lo.xz(), hi.xz()) + 2.12
    {
        return false;
    }
    let tf = o.transform();
    p.canopy_boxes(o.size).iter().all(|&(c, s)| {
        [-1., 1.].into_iter().all(|x| {
            [-1., 1.].into_iter().all(|z| {
                let q = tf.transform_point(c + s * Vec3::new(x, 1., z) * 0.5);
                (q.y - scene.ceiling_height(q.xz()) + 0.001).abs() < 0.004
            })
        })
    })
}

pub(super) fn furnish(scene: &mut IndoorManifest, r: &mut ChaCha8Rng) {
    let Some(door) = scene.entry_door.clone() else {
        return;
    };
    let width = r.random_range(0.28..0.53_f32).min(door.width - 0.12);
    if width < 0.24 {
        return;
    }
    let depth = r.random_range(0.025..0.065);
    let shift = ((door.width - width) * 0.5 - 0.045).max(0.);
    let x = door.center.x + r.random_range(-1_f32..=1.) * shift * 0.65;
    let mut o = scene.candidate(ObjectKind::ExitSign, Vec3::ZERO, Vec3::ONE, PI, r);
    o.solid = false;
    let preferred = r.random_range(0..4);
    // Try the preferred support at two nearby positions; wall fallback remains
    // possible under sloping/obstructed ceilings. No unbounded placement search.
    for attempt in 0..3 {
        o.variant = if attempt < 2 { preferred } else { 0 };
        if o.variant == 0 {
            o.size = Vec3::new(width, r.random_range(0.14..0.23), depth);
            o.position = Vec3::new(
                x,
                door.center.y + door.height + r.random_range(0.055..0.10),
                door.center.z - depth * 0.5 - 0.001,
            );
        } else {
            let z = door.center.z - r.random_range(0.23..0.68);
            let roof = scene.ceiling_height(Vec2::new(x, z));
            let max_drop = roof - (door.center.y + door.height + 0.055);
            let drop: f32 = if o.variant == 1 {
                r.random_range(0.22..0.36)
            } else {
                r.random_range(0.34..0.90)
            };
            let height = drop.min(max_drop);
            if height < 0.20 {
                continue;
            }
            o.size = Vec3::new(width, height, depth);
            o.position = Vec3::new(x, roof - height - 0.001, z);
        }
        if mounted_clear(scene, &o) {
            scene.objects.push(o);
            return;
        }
    }
}
