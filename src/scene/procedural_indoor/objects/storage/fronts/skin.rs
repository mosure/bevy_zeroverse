//! One closed leaf surface. The frame, bevel and inset field meet at shared
//! edges, keeping the depth, semantic and RGB passes on the same geometry.
use super::{
    hinged::{Face, Leaf},
    *,
};

fn contour(half: Vec2, z: f32) -> [Vec3; 4] {
    [
        Vec2::NEG_ONE,
        Vec2::new(1., -1.),
        Vec2::ONE,
        Vec2::new(-1., 1.),
    ]
    .map(|v| (half * v).extend(z))
}

// A cutter changes the surface normal while the grain stays in the same
// board. Project from the stock face so grain meets across each routed edge.
fn front_quad(g: &mut Geometry, points: [Vec3; 4], grain: Vec3, origin: Vec2) {
    let normal = (points[1] - points[0])
        .cross(points[2] - points[0])
        .normalize();
    debug_assert!(normal.z > 0.);
    let across = Vec3::Z.cross(grain);
    let uv = points.map(|p| (origin + Vec2::new(p.dot(across), -p.dot(grain))).to_array());
    g.vertex_quad(points, [normal; 4], uv);
}

fn band(g: &mut Geometry, outer: [Vec3; 4], inner: [Vec3; 4], cuts: &[Vec2; 4]) {
    for (i, &origin) in cuts.iter().enumerate() {
        let j = (i + 1) % 4;
        front_quad(
            g,
            [outer[i], outer[j], inner[j], inner[i]],
            if i % 2 == 0 { Vec3::X } else { Vec3::Y },
            origin,
        );
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &ShelfProgram, d: &Leaf) {
    let half = d.size_m * 0.5;
    let bevel = (d.thickness_m * 0.10).min(half.min_element() * 0.06);
    let mut cuts = TimberCuts::new(o.seed, 0x444F4F52 + d.bay as u64);
    let field_origin = cuts.origin();
    let grain = if d.horizontal_grain { Vec3::X } else { Vec3::Y };
    let g = a.part(p.fronts, "cabinet#uv_local");
    if matches!(d.face, Face::Fluted | Face::Reeded) {
        flutes_and_reeds(g, d, p, bevel, field_origin);
        return;
    }
    let rear = contour(half, -d.thickness_m);
    let outer = contour(half, -bevel);
    let front = contour(half - Vec2::splat(bevel), 0.);
    g.stock_quad(rear, Vec3::NEG_Z, grain, Some(cuts.origin()));
    for (i, side) in [Vec3::NEG_Y, Vec3::X, Vec3::Y, Vec3::NEG_X]
        .into_iter()
        .enumerate()
    {
        let j = (i + 1) % 4;
        g.metric_quad([rear[i], rear[j], outer[j], outer[i]], side);
    }
    if d.face == Face::Slab {
        field_band(g, outer, front, grain, field_origin);
        front_quad(g, front, grain, field_origin);
        return;
    }
    let rails = std::array::from_fn(|_| cuts.origin());
    band(g, outer, front, &rails);
    let field = half - Vec2::new(d.stile_m, d.rail_m);
    let rim = contour(field, 0.);
    let shoulder = (d.relief_m * 0.8).min(field.min_element() * 0.12);
    let inner = contour(field - Vec2::splat(shoulder), -d.relief_m);
    band(g, front, rim, &rails);
    band(g, rim, inner, &rails);
    let panel = a.part(d.panel_finish, "cabinet#uv_local");
    let cap = if d.face == Face::Raised {
        let land = field - Vec2::splat(shoulder);
        let reveal = contour(
            land - Vec2::splat(land.min_element().min(0.08) * 0.09),
            -d.relief_m,
        );
        let face = contour(
            land - Vec2::splat((d.relief_m * 2.7).min(land.min_element() * 0.32)),
            -d.relief_m * 0.16,
        );
        // Keep the grain continuous over the floating panel's bevel and centre.
        field_band(panel, inner, reveal, grain, field_origin);
        field_band(panel, reveal, face, grain, field_origin);
        face
    } else {
        inner
    };
    front_quad(panel, cap, grain, field_origin);
}

fn field_band(g: &mut Geometry, a: [Vec3; 4], b: [Vec3; 4], grain: Vec3, origin: Vec2) {
    for i in 0..4 {
        let j = (i + 1) % 4;
        front_quad(g, [a[i], a[j], b[j], b[i]], grain, origin);
    }
}

// Routed vertical flutes stop at a flat stile on each side. End grain is cut
// to the same profile; there is no slab face behind separate slat meshes.
fn flutes_and_reeds(g: &mut Geometry, d: &Leaf, p: &ShelfProgram, bevel: f32, origin: Vec2) {
    let half = d.size_m * 0.5;
    let inner = half.x - d.stile_m;
    let count = (2. * inner / p.slat_pitch_m).round().clamp(2., 24.) as usize;
    let pitch = inner * 2. / count as f32;
    let mut last = Vec2::new(-half.x, -bevel);
    let point = |v: Vec2, y: f32| Vec3::new(v.x, y, v.y);
    let mut strip = |x: f32, z: f32| {
        let v = Vec2::new(x, z);
        let rear = -d.thickness_m;
        let strip_face = [
            point(last, -half.y),
            point(v, -half.y),
            point(v, half.y),
            point(last, half.y),
        ];
        front_quad(g, strip_face, Vec3::Y, origin);
        g.stock_quad(
            strip_face.map(|q| q.with_z(rear)),
            Vec3::NEG_Z,
            Vec3::Y,
            Some(origin),
        );
        for (y, n) in [(-half.y, Vec3::NEG_Y), (half.y, Vec3::Y)] {
            let (a, b) = (point(last, y), point(v, y));
            g.metric_quad([a, b, b.with_z(rear), a.with_z(rear)], n);
        }
        last = v;
    };
    strip(-half.x + bevel, 0.);
    strip(-inner, 0.);
    for k in 0..count {
        let x = -inner + k as f32 * pitch;
        if d.face == Face::Reeded {
            let shoulder = p.slat_fill * 0.20;
            strip(x + pitch * shoulder, -d.relief_m);
            strip(x + pitch * 0.21, -d.relief_m * 0.44);
            strip(x + pitch * 0.36, -d.relief_m * 0.10);
            strip(x + pitch * 0.50, 0.);
            strip(x + pitch * 0.64, -d.relief_m * 0.10);
            strip(x + pitch * 0.79, -d.relief_m * 0.44);
            strip(x + pitch * (1. - shoulder), -d.relief_m);
        } else {
            let ramp = (1. - p.slat_fill) * 0.22;
            strip(x + pitch * p.slat_fill * 0.5, 0.);
            strip(x + pitch * (p.slat_fill * 0.5 + ramp), -d.relief_m);
            strip(x + pitch * (1. - p.slat_fill * 0.5 - ramp), -d.relief_m);
            strip(x + pitch * (1. - p.slat_fill * 0.5), 0.);
        }
        // End at the metric boundary, including the last flat stile.
        strip(
            if k + 1 == count { inner } else { x + pitch },
            if d.face == Face::Reeded && k + 1 < count {
                -d.relief_m
            } else {
                0.
            },
        );
    }
    strip(half.x - bevel, 0.);
    strip(half.x, -bevel);
    for sign in [-1., 1.] {
        let x = sign * half.x;
        g.metric_quad(
            [
                Vec3::new(x, -half.y, -bevel),
                Vec3::new(x, half.y, -bevel),
                Vec3::new(x, half.y, -d.thickness_m),
                Vec3::new(x, -half.y, -d.thickness_m),
            ],
            Vec3::X * sign,
        );
    }
}
