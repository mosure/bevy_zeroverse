//! Bakery batches sit on a removable board or a rimmed pan. Reserve the whole
//! rotated tray before placing the remaining merchandise on the same shelf.
use super::*;
use crate::scene::procedural_indoor::placement::{Footprint, SurfacePattern};

pub(super) fn bakery_tray(
    a: &mut Assembly,
    o: &IndoorObject,
    level: usize,
    y: f32,
    headroom: f32,
    bounds: (Vec2, Vec2),
) -> Option<Footprint> {
    let mut r = stream(o.seed, 0x5041535452595452 + level as u64);
    if headroom < 0.055 || !r.random_bool(0.74) {
        return None;
    }
    let size = (bounds.1 - bounds.0 - Vec2::splat(0.045)).min(Vec2::new(
        r.random_range(0.26..0.48),
        r.random_range(0.185..0.31),
    ));
    if size.min_element() < 0.165 {
        return None;
    }
    let pattern = SurfacePattern::sample(&mut r);
    let placement = (0..8).find_map(|_| {
        let yaw = r.random_range(-0.26..0.26);
        pattern.propose(bounds, size, yaw, 0.015, &mut r)
    })?;
    let mut batch = Assembly::default();
    let pan = r.random_bool(0.62);
    let t = if pan {
        0.003
    } else {
        r.random_range(0.008..0.014)
    };
    let stock = if pan { Surface::Metal } else { Surface::Wood };
    batch.box_part(
        stock,
        "dishware",
        Vec3::Y * t * 0.5,
        Vec3::new(size.x, t, size.y),
        t * 0.28,
    );
    if pan {
        let rim = r.random_range(0.007..0.014);
        for side in [-1., 1.] {
            batch.box_part(
                stock,
                "dishware",
                Vec3::new(side * (size.x - t) * 0.5, t + rim * 0.5, 0.),
                Vec3::new(t, rim, size.y),
                0.0008,
            );
            batch.box_part(
                stock,
                "dishware",
                Vec3::new(0., t + rim * 0.5, side * (size.y - t) * 0.5),
                Vec3::new(size.x - 2. * t, rim, t),
                0.0008,
            );
        }
    }
    let biscuits = r.random_bool(0.48);
    let radius = if biscuits {
        r.random_range(0.028..0.047)
    } else {
        r.random_range(0.036..0.057)
    };
    let inventory = r.random_range(2..=8);
    let mut placed = Vec::<(Vec2, f32)>::with_capacity(inventory);
    for _ in 0..inventory {
        let radius = radius * r.random_range(0.88..1.09);
        let free = size * 0.5 - Vec2::splat(radius + 0.008);
        let Some(center) = (0..12).find_map(|_| {
            let p = free * Vec2::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0));
            placed
                .iter()
                .all(|(other, rr)| p.distance_squared(*other) > (radius + rr + 0.003).powi(2))
                .then_some(p)
        }) else {
            continue;
        };
        placed.push((center, radius));
        let at = Transform::from_xyz(center.x, t, center.y)
            .with_rotation(Quat::from_rotation_y(r.random_range(-PI..PI)));
        if biscuits {
            let mut biscuit = Assembly::default();
            crate::scene::procedural_indoor::objects::kitchen::biscuit(
                &mut biscuit,
                radius,
                0.,
                headroom - t,
                &mut r,
            );
            append_at(&mut batch, biscuit, at);
        } else {
            let h = (radius * r.random_range(0.73..1.19)).min(headroom - t);
            let tf = at.with_scale(Vec3::new(1., 1., r.random_range(0.68..0.94)));
            batch.part(Surface::Terracotta, "food#finish103").lathe(
                &[
                    (0., 0.),
                    (radius * 0.57, 0.),
                    (radius * 0.89, h * 0.14),
                    (radius, h * 0.43),
                    (radius * 0.88, h * 0.72),
                    (radius * 0.58, h * 0.91),
                    (radius * 0.25, h * 0.985),
                    (0., h),
                ],
                r.random_range(13..=20),
                tf,
            );
        }
    }
    append_at(
        a,
        batch,
        Transform::from_xyz(placement.center.x, y, placement.center.y)
            .with_rotation(Quat::from_rotation_y(placement.yaw)),
    );
    Some(placement)
}
