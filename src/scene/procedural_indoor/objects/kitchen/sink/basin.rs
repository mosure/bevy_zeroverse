//! Closed basin shell: rectangular worktop interface, rounded/tapered bowl,
//! real open top and a supported inner floor. Fixed 32-side construction cost.
use super::*;
pub(super) fn build(a: &mut Assembly, p: &program::Basin, top: f32, finish: Surface) {
    const N: usize = 32;
    let inner = p.inner_half();
    let floor = p.floor_half();
    let sections = [
        (p.floor_m - p.wall_m, floor + Vec2::splat(p.wall_m), false),
        (top, inner + Vec2::splat(p.wall_m), false),
        (top, p.size * 0.5, true),
        (top + 0.006, p.size * 0.5, true),
        (top + 0.006, inner, false),
        (p.floor_m, floor, false),
    ];
    // All rings sample the same metric rays. Parametric superellipse angles
    // change their polar angle with aspect/roundness and can fold rim quads.
    let directions: [Vec2; N] = std::array::from_fn(|i| {
        let t = TAU * i as f32 / N as f32;
        {
            let clean = |v: f32| if v.abs() < 1e-6 { 0. } else { v };
            Vec2::new(clean(t.cos()), clean(t.sin()))
        }
    });
    let rings: [[Vec3; N]; 6] = std::array::from_fn(|ring| {
        std::array::from_fn(|i| {
            let (y, half, square) = sections[ring];
            let v = directions[i];
            let scaled = v.abs() / half;
            let distance = if square {
                1. / scaled.max_element()
            } else {
                (scaled.x.powf(p.roundness) + scaled.y.powf(p.roundness)).powf(-1. / p.roundness)
            };
            let xy = v * distance + p.center;
            Vec3::new(xy.x, y, xy.y)
        })
    });
    let point = |ring: usize, i: usize| rings[ring][i % N];
    let g = a.part(finish, "sink");
    for ring in 0..sections.len() - 1 {
        for i in 0..N {
            let start = g.positions.len() as u32;
            for (row, col) in [(ring, i), (ring + 1, i), (ring + 1, i + 1), (ring, i + 1)] {
                let pos = point(row, col);
                let tangent = point(row, col + 1) - point(row, col + N - 1);
                let normal = (point(ring + 1, col) - point(ring, col))
                    .cross(tangent)
                    .normalize();
                g.positions.push(pos.to_array());
                g.normals.push(normal.to_array());
                g.uvs.push(if normal.y.abs() > 0.85 {
                    pos.xz().to_array()
                } else {
                    [col as f32 / N as f32 * 2. * (p.size.x + p.size.y), pos.y]
                });
            }
            g.indices
                .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
        }
    }
    for (ring, sign) in [(0, -1.), (sections.len() - 1, 1.)] {
        let start = g.positions.len() as u32;
        let y = sections[ring].0;
        g.positions.push([p.center.x, y, p.center.y]);
        g.normals.push([0., sign, 0.]);
        g.uvs.push(p.center.to_array());
        for i in 0..N {
            let pos = point(ring, i);
            g.positions.push(pos.to_array());
            g.normals.push([0., sign, 0.]);
            g.uvs.push(pos.xz().to_array());
        }
        for i in 0..N as u32 {
            let b = start + 1 + i;
            let c = start + 1 + (i + 1) % N as u32;
            g.indices.extend(if sign > 0. {
                [start, c, b]
            } else {
                [start, b, c]
            });
        }
    }
    // Flush drain insert: no raised plug intersecting dishes on the bowl floor.
    a.part(Surface::Chrome, "sink").cylinder(
        p.drain_radius,
        0.003,
        Transform::from_xyz(p.drain.x, p.floor_m - 0.001, p.drain.y),
    );
    a.part(Surface::Ink, "sink").cylinder(
        p.drain_radius * 0.65,
        0.0004,
        Transform::from_xyz(p.drain.x, p.floor_m + 0.0007, p.drain.y),
    );
}
