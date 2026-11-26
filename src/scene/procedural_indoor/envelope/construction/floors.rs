//! Walk surfaces and front faces are partitioned at actual material joints.
//! An inlaid tread is flush with the support and has no underlying coplanar cap.
use super::*;
mod court;
use crate::scene::procedural_indoor::envelope::steps::{StepInstallation, StepNosing};
use crate::scene::procedural_indoor::program::court::VolleyballCourt;

#[derive(Clone, Copy)]
struct Band {
    along: Vec2,
    span: Vec2,
    height: f32,
    surface: Surface,
}

fn tread_bands(e: &EnvelopeProgram) -> Vec<Band> {
    let Some(s) = &e.step_installation else {
        return vec![];
    };
    let Some(stock) = s.nosing.surface() else {
        return vec![];
    };
    let axis = s.run_axis.index();
    let mut bands = Vec::with_capacity(s.walk_edges.len() * 3);
    for edge in &s.walk_edges {
        let mut p = Vec2::ZERO;
        p[1 - axis] = (edge.span_m.x + edge.span_m.y) * 0.5;
        p[axis] = edge.coordinate_m - 0.001;
        let left = e.floor_height(p);
        p[axis] = edge.coordinate_m + 0.001;
        let right = e.floor_height(p);
        if (left - right).abs() < 0.001 {
            continue;
        }
        let into_tread = (right - left).signum();
        let mut band = |from, to, surface| {
            let a: f32 = edge.coordinate_m + into_tread * from;
            let b: f32 = edge.coordinate_m + into_tread * to;
            bands.push(Band {
                along: Vec2::new(a.min(b), a.max(b)),
                span: edge.span_m,
                height: left.max(right),
                surface,
            });
        };
        if s.nosing == StepNosing::MetalWithInsert {
            band(0., s.insert_margin_m, stock);
            band(
                s.insert_margin_m,
                s.tread_band_m - s.insert_margin_m,
                Surface::Rubber,
            );
            band(s.tread_band_m - s.insert_margin_m, s.tread_band_m, stock);
        } else {
            band(0., s.tread_band_m, stock);
        }
    }
    bands.sort_unstable_by(|a, b| a.along.x.total_cmp(&b.along.x));
    bands
}

fn horizontal_quad(g: &mut Geometry, rect: Rect, y: f32) {
    // The same world X / -Z chart as the clipped structural floor.
    let p = |x, z| Vec3::new(x, y, z);
    g.metric_quad(
        [
            p(rect.min.x, rect.max.y),
            p(rect.max.x, rect.max.y),
            p(rect.max.x, rect.min.y),
            p(rect.min.x, rect.min.y),
        ],
        Vec3::Y,
    );
}

fn floor_top(g: &mut Geometry, footprint: Option<&[[Vec2; 3]]>, rect: Rect, y: f32) {
    let Some(triangles) = footprint else {
        horizontal_quad(g, rect, y);
        return;
    };
    for triangle in triangles {
        let p = polygon::clip_box(triangle, rect.min, rect.max);
        face(
            g,
            &p.iter().map(|p| Vec3::new(p.x, y, p.y)).collect::<Vec<_>>(),
            Vec3::Y,
        );
    }
}

fn build_top(
    a: &mut Assembly,
    footprint: Option<&[[Vec2; 3]]>,
    rect: Rect,
    y: f32,
    finish: Option<&StepInstallation>,
    bands: &[Band],
) {
    let Some(finish) = finish.filter(|_| !bands.is_empty()) else {
        floor_top(a.part(Surface::Floor, "floor"), footprint, rect, y);
        return;
    };
    let k = finish.run_axis.index();
    let cross = rect.center()[1 - k];
    let mut cursor = rect.min[k];
    let end = rect.max[k];
    let interval = |lo: f32, hi: f32| {
        let mut part = rect;
        part.min[k] = lo;
        part.max[k] = hi;
        part
    };
    for b in bands {
        if b.along.x >= end {
            break;
        }
        if b.along.y <= cursor || b.height != y || !(b.span.x..b.span.y).contains(&cross) {
            continue;
        }
        let start = cursor.max(b.along.x);
        if start > cursor {
            floor_top(
                a.part(Surface::Floor, "floor"),
                footprint,
                interval(cursor, start),
                y,
            );
        }
        cursor = end.min(b.along.y);
        let part = interval(start, cursor);
        // Walking edges are inside the footprint, with a larger wall margin
        // than the widest trim. Their bounds share the patch grid exactly.
        let g = a.part(b.surface, "floor");
        let first = g.positions.len();
        horizontal_quad(g, part, y);
        for (uv, pos) in g.uvs[first..].iter_mut().zip(&g.positions[first..]) {
            let xz = Vec2::new(pos[0], pos[2]);
            *uv = [xz[k], xz[1 - k]];
        }
    }
    if cursor < end {
        floor_top(
            a.part(Surface::Floor, "floor"),
            footprint,
            interval(cursor, end),
            y,
        );
    }
}

pub(super) fn build(
    a: &mut Assembly,
    e: &EnvelopeProgram,
    size: Vec3,
    volleyball: Option<VolleyballCourt>,
) {
    let mut xs = vec![-size.x * 0.5, size.x * 0.5];
    let mut zs = vec![-size.z * 0.5, size.z * 0.5];
    for f in &e.floor_patches {
        xs.extend([f.min.x, f.max.x]);
        zs.extend([f.min.y, f.max.y]);
    }
    for v in [&mut xs, &mut zs] {
        v.sort_by(f32::total_cmp);
        v.dedup_by(|a, b| (*a - *b).abs() < 1e-6);
    }
    let triangles = polygon::triangles(&e.footprint);
    if let Some(court) = volleyball {
        court::top(a, court, size);
    } else {
        let bands = tread_bands(e);
        for x in xs.windows(2) {
            for z in zs.windows(2) {
                let rect = Rect::from_corners(Vec2::new(x[0], z[0]), Vec2::new(x[1], z[1]));
                let patch = e.floor_patches.iter().find(|p| p.contains(rect.center()));
                // Patch boundaries are grid cuts. An interior patch cell is a full
                // rectangle inside the envelope, including on a concave floor.
                let finish = e.step_installation.as_ref();
                let clip = if patch.is_some() && finish.is_some() {
                    None
                } else {
                    Some(triangles.as_slice())
                };
                build_top(
                    a,
                    clip,
                    rect,
                    patch.map_or(0., |p| p.height),
                    finish,
                    &bands,
                );
            }
        }
    }
    let g = a.part(Surface::Floor, "floor");
    let bottom = e.minimum_floor() - 0.20;
    for t in &triangles {
        triangle(g, t.map(|p| Vec3::new(p.x, bottom, p.y)), Vec3::NEG_Y);
    }
    for (p, q) in polygon::edges(&e.footprint) {
        quad(
            g,
            [
                Vec3::new(p.x, 0., p.y),
                Vec3::new(q.x, 0., q.y),
                Vec3::new(q.x, bottom, q.y),
                Vec3::new(p.x, bottom, p.y),
            ],
            Vec3::new(q.y - p.y, 0., p.x - q.x),
        );
    }
    // Only actual height discontinuities acquire risers; adjacent tiles do not
    // create duplicate internal faces or hidden O-voxel geometry.
    for axis in 0..2 {
        let (along, cross) = if axis == 0 { (&xs, &zs) } else { (&zs, &xs) };
        for &edge in along.iter().skip(1).take(along.len().saturating_sub(2)) {
            for c in cross.windows(2) {
                let p = if axis == 0 {
                    Vec2::new(edge, (c[0] + c[1]) * 0.5)
                } else {
                    Vec2::new((c[0] + c[1]) * 0.5, edge)
                };
                let n = if axis == 0 { Vec2::X } else { Vec2::Y };
                let left = e.floor_height(p - n * 0.0001);
                let right = e.floor_height(p + n * 0.0001);
                if (left - right).abs() < 1e-5 {
                    continue;
                }
                let (u, v) = if axis == 0 {
                    (Vec2::new(edge, c[0]), Vec2::new(edge, c[1]))
                } else {
                    (Vec2::new(c[0], edge), Vec2::new(c[1], edge))
                };
                let normal = Vec3::new(n.x, 0., n.y) * (left - right).signum();
                let high = left.max(right);
                let low = left.min(right);
                let finish = e.step_installation.as_ref();
                let stock = finish.map_or(Surface::Floor, |s| s.riser_surface);
                let nosing = finish
                    .filter(|s| s.run_axis.index() == axis && s.is_walk_edge(edge, p[1 - axis]))
                    .and_then(|s| Some((s.nosing.surface()?, s.face_return_m.min(high - low))));
                let draw = |a: &mut Assembly, stock: Surface, top: f32, bottom: f32| {
                    // Keep the original face/diagonal for unfitted archived floors.
                    let (h0, h1) = if left > right {
                        (top, bottom)
                    } else {
                        (bottom, top)
                    };
                    let points = [
                        u.extend(h0).xzy(),
                        v.extend(h0).xzy(),
                        v.extend(h1).xzy(),
                        u.extend(h1).xzy(),
                    ];
                    let g = a.part(stock, "floor");
                    let first = g.positions.len();
                    quad(g, points, normal);
                    if stock == Surface::Wood {
                        for (uv, pos) in g.uvs[first..].iter_mut().zip(&g.positions[first..]) {
                            *uv = [pos[1], if axis == 0 { pos[2] } else { pos[0] }];
                        }
                    }
                };
                let cap = nosing.map_or(high, |(_, depth)| high - depth);
                draw(a, stock, cap, low);
                if let Some((metal_or_tread, _)) = nosing {
                    draw(a, metal_or_tread, high, cap);
                }
            }
        }
    }
    exterior_apron(
        a.part(Surface::Concrete, "floor#exterior"),
        &e.footprint,
        size.xz() * 0.5 + Vec2::splat(0.25),
    );
}

/// A concrete apron with vertical faces only on the room perimeter and outer rim.
fn exterior_apron(g: &mut Geometry, footprint: &[Vec2], half_size: Vec2) {
    let top = -0.10;
    let bottom = top - 0.16;
    let at = |p: Vec2, y| Vec3::new(p.x, y, p.y);
    for ring in polygon::outside(footprint, -half_size, half_size) {
        for edge in ring[1..].windows(2) {
            let t = [ring[0], edge[0], edge[1]];
            triangle(g, t.map(|p| at(p, top)), Vec3::Y);
            triangle(g, t.map(|p| at(p, bottom)), Vec3::NEG_Y);
        }
    }
    let outer = [
        -half_size,
        Vec2::new(half_size.x, -half_size.y),
        half_size,
        Vec2::new(-half_size.x, half_size.y),
    ];
    // Footprints are validated counterclockwise. The inner rim faces the room.
    for (a, b) in polygon::edges(&outer).chain(polygon::edges(footprint).map(|(a, b)| (b, a))) {
        quad(
            g,
            [at(a, bottom), at(b, bottom), at(b, top), at(a, top)],
            Vec3::new(b.y - a.y, 0., a.x - b.x),
        );
    }
}

#[cfg(test)]
mod tests;
