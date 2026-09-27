//! Bounded A* at lens height; every edge sweeps the actual camera clearance.
use super::*;
use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
};
pub(super) fn route(scene: &IndoorManifest, start: Vec3, end: Vec3) -> Option<Vec<Vec3>> {
    if scene.camera_path_clear(start, end) {
        return Some(vec![start, end]);
    }
    if !scene.camera_clear(end) {
        return None;
    }
    let step = 0.28;
    let point = |(x, z): (i32, i32)| Vec3::new(x as f32 * step, start.y, z as f32 * step);
    let cell = |p: Vec3| ((p.x / step).round() as i32, (p.z / step).round() as i32);
    let (a, b) = (cell(start), cell(end));
    let obstacles: Vec<_> = scene
        .camera_obstacles()
        .into_iter()
        .chain(
            scene
                .objects
                .iter()
                .map(super::super::layout::IndoorObject::bounds),
        )
        .chain(
            scene
                .humans
                .iter()
                .map(super::super::humans::IndoorHuman::bounds),
        )
        .collect();
    let clear = |a: Vec3, b: Vec3| {
        let pad = Vec3::splat(super::super::layout::CAMERA_CLEARANCE);
        (a.x.abs().max(b.x.abs()) < scene.room_size.x * 0.5 - 0.5)
            && (a.z.abs().max(b.z.abs()) < scene.room_size.z * 0.5 - 0.5)
            && (!scene.camera_settings.primary_room
                || (scene.in_primary_room(a, pad.x) && scene.in_primary_room(b, pad.x)))
            && !obstacles
                .iter()
                .any(|(lo, hi)| super::super::layout::segment_hits_box(a, b, *lo - pad, *hi + pad))
    };
    if !clear(start, point(a)) || !clear(point(b), end) {
        return None;
    }
    let heuristic = |p: (i32, i32)| (p.0 - b.0).abs().max((p.1 - b.1).abs()) * 10;
    let mut queue = BinaryHeap::from([Reverse((heuristic(a), 0, a))]);
    let mut cost = HashMap::from([(a, 0)]);
    let mut parents = HashMap::new();
    while let Some(Reverse((_, g, c))) = queue.pop() {
        if c == b {
            let mut path = vec![end, point(b)];
            let mut c = b;
            while c != a {
                c = parents[&c];
                path.push(point(c));
            }
            path.push(start);
            path.reverse();
            let mut simplified = vec![start];
            let mut i = 0;
            while i + 1 < path.len() {
                let j = (i + 1..path.len())
                    .rev()
                    .find(|&j| clear(path[i], path[j]))?;
                if path[j].distance(*simplified.last().unwrap()) > 0.001 {
                    simplified.push(path[j]);
                }
                i = j;
            }
            // Round only corners whose entire swept arc is clear. Unsafe corner
            // cuts retain the original valid vertex instead of clipping furniture.
            let mut rounded = vec![start];
            for triple in simplified.windows(3) {
                let corner = triple[1];
                let trim =
                    (corner.distance(triple[0]).min(corner.distance(triple[2])) * 0.22).min(0.4);
                let before = corner + (triple[0] - corner).normalize_or_zero() * trim;
                let after = corner + (triple[2] - corner).normalize_or_zero() * trim;
                let arc: Vec<_> = (0..=8)
                    .map(|i| {
                        let t = i as f32 / 8.0;
                        before.lerp(corner, t).lerp(corner.lerp(after, t), t)
                    })
                    .collect();
                if arc.windows(2).all(|p| clear(p[0], p[1])) {
                    rounded.extend(arc);
                } else {
                    rounded.push(corner);
                }
            }
            rounded.push(end);
            return Some(rounded);
        }
        if cost.len() > 4096 {
            return None;
        }
        if cost[&c] != g {
            continue;
        }
        for dx in -1..=1 {
            for dz in -1..=1 {
                if dx == 0 && dz == 0 {
                    continue;
                }
                let n = (c.0 + dx, c.1 + dz);
                if !clear(point(c), point(n)) {
                    continue;
                }
                let next = g + if dx == 0 || dz == 0 { 10 } else { 14 };
                if cost.get(&n).is_none_or(|v| next < *v) {
                    cost.insert(n, next);
                    parents.insert(n, c);
                    queue.push(Reverse((next + heuristic(n), next, n)));
                }
            }
        }
    }
    None
}
