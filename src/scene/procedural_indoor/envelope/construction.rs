use super::*;
use crate::scene::procedural_indoor::{
    architecture, geometry::Geometry, layout::NEIGHBOR_DEPTH, materials::Surface, objects::Assembly,
};

/// Flat, metric UVs. Triangle winding and stored normals always agree, including
/// negative floor levels, angled reveals and the underside of a sloping ceiling.
pub fn triangle(g: &mut Geometry, mut p: [Vec3; 3], outward: Vec3) {
    let mut n = (p[1] - p[0]).cross(p[2] - p[0]);
    if n.length_squared() < 1e-14 {
        return;
    }
    if n.dot(outward) < 0. {
        p.swap(1, 2);
        n = -n;
    }
    n = n.normalize();
    let u = if n.y.abs() > 0.6 {
        (Vec3::X - n * n.x).normalize()
    } else {
        Vec3::Y.cross(n).normalize()
    };
    let v = n.cross(u);
    let start = g.positions.len() as u32;
    for p in p {
        g.positions.push(p.to_array());
        g.normals.push(n.to_array());
        g.uvs.push([p.dot(u), p.dot(v)]);
    }
    g.indices.extend([start, start + 1, start + 2]);
}
fn face(g: &mut Geometry, p: &[Vec3], outward: Vec3) {
    for i in 1..p.len().saturating_sub(1) {
        triangle(g, [p[0], p[i], p[i + 1]], outward);
    }
}
fn quad(g: &mut Geometry, p: [Vec3; 4], outward: Vec3) {
    face(g, &p, outward);
}

fn panel(g: &mut Geometry, points: &[Vec3], extrusion: Vec3) {
    if points.len() < 3 {
        return;
    }
    let mut p = points.to_vec();
    if (p[1] - p[0]).cross(p[2] - p[0]).dot(extrusion) < 0. {
        p.reverse();
    }
    face(g, &p, extrusion);
    face(
        g,
        &p.iter().map(|p| *p - extrusion).collect::<Vec<_>>(),
        -extrusion,
    );
    for i in 0..p.len() {
        let a = p[i];
        let b = p[(i + 1) % p.len()];
        quad(
            g,
            [a, b, b - extrusion, a - extrusion],
            (b - a).cross(extrusion),
        );
    }
}

fn slab(g: &mut Geometry, poly: &[Vec2], height: impl Fn(Vec2) -> f32, thickness: f32) {
    solid(g, poly, |p| height(p) - thickness, &height);
}
fn solid(
    g: &mut Geometry,
    poly: &[Vec2],
    bottom: impl Fn(Vec2) -> f32,
    height: impl Fn(Vec2) -> f32,
) {
    for t in polygon::triangles(poly) {
        triangle(g, t.map(|p| Vec3::new(p.x, height(p), p.y)), Vec3::Y);
        triangle(g, t.map(|p| Vec3::new(p.x, bottom(p), p.y)), Vec3::NEG_Y);
    }
    for (a, b) in polygon::edges(poly) {
        quad(
            g,
            [
                Vec3::new(a.x, height(a), a.y),
                Vec3::new(b.x, height(b), b.y),
                Vec3::new(b.x, bottom(b), b.y),
                Vec3::new(a.x, bottom(a), a.y),
            ],
            Vec3::new(b.y - a.y, 0., a.x - b.x),
        );
    }
}

fn floors(a: &mut Assembly, e: &EnvelopeProgram, size: Vec3) {
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
    let g = a.part(Surface::Floor, "floor");
    let triangles = polygon::triangles(&e.footprint);
    for x in xs.windows(2) {
        for z in zs.windows(2) {
            let lo = Vec2::new(x[0], z[0]);
            let hi = Vec2::new(x[1], z[1]);
            let y = e.floor_height((lo + hi) * 0.5);
            for t in &triangles {
                let p = polygon::clip_box(t, lo, hi);
                face(
                    g,
                    &p.iter().map(|p| Vec3::new(p.x, y, p.y)).collect::<Vec<_>>(),
                    Vec3::Y,
                );
            }
        }
    }
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
                quad(
                    g,
                    [
                        Vec3::new(u.x, left, u.y),
                        Vec3::new(v.x, left, v.y),
                        Vec3::new(v.x, right, v.y),
                        Vec3::new(u.x, right, u.y),
                    ],
                    normal,
                );
            }
        }
    }
    for p in polygon::outside(
        &e.footprint,
        -size.xz() * 0.5 - Vec2::splat(0.25),
        size.xz() * 0.5 + Vec2::splat(0.25),
    ) {
        slab(
            a.part(Surface::Concrete, "floor#exterior"),
            &p,
            |_| -0.10,
            0.16,
        );
    }
}

pub(super) fn wall(
    a: &mut Assembly,
    e: &EnvelopeProgram,
    size: Vec3,
    edge: usize,
    holes: &[architecture::facade::WindowOpening],
) {
    let p = e.footprint[edge];
    let q = e.footprint[(edge + 1) % e.footprint.len()];
    let length = p.distance(q);
    let tf = e.wall_transform(edge);
    let hp = e.ceiling_height(size, p);
    let hq = e.ceiling_height(size, q);
    let slope = (hq - hp) / length;
    let midpoint = (hp + hq) * 0.5;
    for (lo, hi) in architecture::facade::solid_rectangles(
        Vec2::new(-length * 0.5, 0.),
        Vec2::new(length * 0.5, hp.max(hq)),
        holes,
    ) {
        let rect = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)];
        let p = polygon::clip(&rect, Vec2::new(slope, -1.), -midpoint);
        panel(
            a.part(Surface::Paint, "wall"),
            &p.iter()
                .map(|p| tf.transform_point(Vec3::new(p.x, p.y, 0.)))
                .collect::<Vec<_>>(),
            tf.rotation * Vec3::Z * 0.18,
        );
    }
    for (lo, hi) in architecture::facade::solid_rectangles(
        Vec2::new(-length * 0.5 + 0.025, 0.01),
        Vec2::new(length * 0.5 - 0.025, 0.12),
        holes,
    ) {
        a.part(Surface::WoodEdge, "wall").cuboid(
            Vec3::new(hi.x - lo.x, hi.y - lo.y, 0.024),
            0.002,
            Transform::from_translation(tf.transform_point(Vec3::new(
                (lo.x + hi.x) * 0.5,
                (lo.y + hi.y) * 0.5,
                0.014,
            )))
            .with_rotation(tf.rotation),
        );
    }
}

pub fn build(scene: &IndoorManifest) -> Assembly {
    let e = scene.envelope.as_ref().unwrap();
    let size = scene.room_size;
    let mut a = Assembly::default();
    floors(&mut a, e, size);
    slab(
        a.part(Surface::Ceiling, "ceiling"),
        &e.footprint,
        |p| e.ceiling_height(size, p) + 0.14,
        0.14,
    );
    for entry in &e.walls {
        let holes = entry
            .facade
            .as_ref()
            .map_or(&[][..], |f| f.openings.as_slice());
        wall(&mut a, e, size, entry.edge, holes);
        if let Some(f) = &entry.facade {
            architecture::facade::glazing_at(&mut a, e.wall_transform(entry.edge), f);
        }
    }
    shared_and_neighbor(&mut a, scene, e);
    // Partition tops are cut by the same analytic roof plane, not scaled: door
    // dimensions and all metric material coordinates are preserved.
    let mut partitions = Assembly::default();
    super::super::floorplan::build(scene, &mut partitions);
    clip_roof(&mut partitions, e, size);
    for (key, g) in partitions.parts {
        a.parts.entry(key).or_default().append(g);
    }
    for p in &e.pillars {
        let g = a.part(Surface::Concrete, "other_structure");
        let outline: Vec<_> = (0..p.sides)
            .map(|i| {
                p.center
                    + Vec2::from_angle(i as f32 / p.sides as f32 * std::f32::consts::TAU) * p.radius
            })
            .collect();
        solid(g, &outline, |_| 0., |v| e.ceiling_height(size, v));
        // Base collars are raised clear of the floor, with a real construction reveal.
        a.part(Surface::Metal, "other_structure").cylinder(
            p.radius + 0.025,
            0.06,
            Transform::from_xyz(p.center.x, 0.032, p.center.y),
        );
    }
    if let Some(m) = &e.mezzanine {
        mezzanine(&mut a, m);
    }
    architecture::fixture_geometry(&mut a, scene);
    let mut exterior = Assembly::default();
    architecture::facade::backdrop(&mut exterior, scene);
    for ((surface, label), g) in exterior.parts {
        a.part(surface, &format!("{label}#exterior")).append(g);
    }
    a
}

pub fn arched_portal(p: &super::super::program::Partition, height: f32, a: &mut Assembly) {
    let normal = if p.axis == 0 { Vec3::X } else { Vec3::Z };
    let mut arc = Vec::new();
    for i in 0..=32 {
        let x = p.door_center - p.door_width * 0.5 + p.door_width * i as f32 / 32.;
        arc.push(p.position(x, p.opening_height(x)));
    }
    for pair in arc.windows(2) {
        let [lo, hi] = [pair[0], pair[1]];
        panel(
            a.part(Surface::Paint, "wall"),
            &[
                lo + normal * p.thickness * 0.5,
                hi + normal * p.thickness * 0.5,
                hi.with_y(height) + normal * p.thickness * 0.5,
                lo.with_y(height) + normal * p.thickness * 0.5,
            ],
            normal * p.thickness,
        );
        for sign in [-1., 1.] {
            let offset = normal * sign * (p.thickness * 0.5 + 0.010);
            a.part(Surface::WoodEdge, "door")
                .rod(lo + offset, hi + offset, 0.018);
        }
    }
    for x in [
        p.door_center - p.door_width * 0.5,
        p.door_center + p.door_width * 0.5,
    ] {
        let h = p.door_height - p.arch_rise - 0.018;
        let size = if p.axis == 0 {
            Vec3::new(p.thickness + 0.02, h, 0.035)
        } else {
            Vec3::new(0.035, h, p.thickness + 0.02)
        };
        a.box_part(
            Surface::WoodEdge,
            "door",
            p.position(x, h * 0.5),
            size,
            0.002,
        );
    }
}

fn shared_and_neighbor(a: &mut Assembly, scene: &IndoorManifest, e: &EnvelopeProgram) {
    use architecture::facade::{Facade, FacadeSide, Shade, WindowOpening};
    let size = scene.room_size;
    let hx = size.x * 0.5;
    let hz = size.z * 0.5;
    let edge = (0..e.footprint.len())
        .find(|&i| e.shared_edge(i, size))
        .unwrap();
    let height = e
        .ceiling_height(size, Vec2::new(-hx, hz))
        .min(e.ceiling_height(size, Vec2::new(hx, hz)));
    let door = -scene.door_x;
    let mut openings = vec![WindowOpening {
        min: Vec2::new(door - 0.55, 0.),
        max: Vec2::new(door + 0.55, 2.27),
        columns: 1,
        transom: 0.,
        operable: false,
    }];
    for (lo, hi) in [(-hx + 0.10, door - 0.60), (door + 0.60, hx - 0.10)] {
        if hi - lo > 0.18 {
            openings.push(WindowOpening {
                min: Vec2::new(lo, 0.05),
                max: Vec2::new(hi, height - 0.15),
                columns: ((hi - lo) / 1.3).ceil().max(1.) as u32,
                transom: 0.,
                operable: false,
            });
        }
    }
    wall(a, e, size, edge, &openings);
    let f = Facade {
        side: FacadeSide::Rear,
        openings: openings[1..].to_vec(),
        frame: Surface::Metal,
        frame_width: 0.035,
        frame_depth: 0.075,
        recess: 0.06,
        sill_projection: 0.,
        shade: Shade::None,
        shade_coverage: 0.,
        shade_tilt: 0.,
    };
    let mut glass = Assembly::default();
    architecture::facade::glazing_at(&mut glass, e.wall_transform(edge), &f);
    for ((surface, label), g) in glass.parts {
        let surface = if surface == Surface::Glass {
            Surface::GlassInterior
        } else {
            surface
        };
        a.part(surface, &label).append(g);
    }
    for x in [scene.door_x - 0.55, scene.door_x + 0.55] {
        a.box_part(
            Surface::WoodEdge,
            "door",
            Vec3::new(x, 1.13, hz - 0.09),
            Vec3::new(0.06, 2.26, 0.20),
            0.003,
        );
    }
    a.box_part(
        Surface::WoodEdge,
        "door",
        Vec3::new(scene.door_x, 2.281, hz - 0.09),
        Vec3::new(1.16, 0.04, 0.20),
        0.003,
    );
    a.box_part(
        Surface::Wood,
        "door",
        Vec3::new(scene.door_x + 0.525, 1.12, hz + 0.56),
        Vec3::new(0.045, 2.24, 1.07),
        0.005,
    );
    // Shared partition is the front edge of the neighboring volume. Exterior
    // walls start beyond it; they never cover the passage or its glass.
    let poly = [
        Vec2::new(-hx, hz),
        Vec2::new(hx, hz),
        Vec2::new(hx, hz + NEIGHBOR_DEPTH),
        Vec2::new(-hx, hz + NEIGHBOR_DEPTH),
    ];
    slab(a.part(Surface::Floor, "floor"), &poly, |_| 0., 0.20);
    slab(
        a.part(Surface::Ceiling, "ceiling"),
        &poly,
        |p| e.ceiling_height(size, p) + 0.14,
        0.14,
    );
    for (p, q) in polygon::edges(&poly).skip(1) {
        let n = Vec3::new(q.y - p.y, 0., p.x - q.x).normalize();
        panel(
            a.part(Surface::Paint, "wall"),
            &[
                Vec3::new(p.x, 0., p.y),
                Vec3::new(q.x, 0., q.y),
                Vec3::new(q.x, e.ceiling_height(size, q), q.y),
                Vec3::new(p.x, e.ceiling_height(size, p), p.y),
            ],
            -n * 0.18,
        );
    }
}

fn mezzanine(a: &mut Assembly, m: &Mezzanine) {
    let lo = Vec3::new(m.deck.min.x, m.deck.height - m.thickness, m.deck.min.y);
    let hi = Vec3::new(m.deck.max.x, m.deck.height, m.deck.max.y);
    a.box_part(Surface::Wood, "floor", (lo + hi) * 0.5, hi - lo, 0.);
    for x in [m.deck.min.x + 0.12, m.deck.max.x - 0.12] {
        for z in [m.deck.min.y + 0.12, m.deck.max.y - 0.12] {
            a.box_part(
                Surface::Metal,
                "other_structure",
                Vec3::new(x, (m.deck.height - m.thickness) * 0.5, z),
                Vec3::new(0.13, m.deck.height - m.thickness, 0.13),
                0.004,
            );
        }
    }
    for (lo, hi) in m.stair_boxes() {
        a.box_part(Surface::Wood, "floor", (lo + hi) * 0.5, hi - lo, 0.);
    }
    for (p, q) in m.rails() {
        let up = Vec3::Y * m.rail_height;
        let normal = (q - p).cross(Vec3::Y).normalize() * 0.01;
        panel(
            a.part(Surface::GlassInterior, "other_structure"),
            &[
                p + Vec3::Y * 0.08,
                q + Vec3::Y * 0.08,
                q + up - Vec3::Y * 0.05,
                p + up - Vec3::Y * 0.05,
            ],
            normal,
        );
        a.part(Surface::Chrome, "other_structure")
            .rod(p + up, q + up, 0.024);
        let count = (p.distance(q) / 1.0).ceil().max(1.) as usize;
        for i in 0..=count {
            let c = p.lerp(q, i as f32 / count as f32);
            a.part(Surface::Metal, "other_structure")
                .rod(c, c + up, 0.018);
        }
    }
}

/// Triangle-plane clipping retains interpolated normals/UVs; no vertex collapse.
pub fn clip_roof(a: &mut Assembly, e: &EnvelopeProgram, size: Vec3) {
    for g in a.parts.values_mut() {
        let original = std::mem::take(g);
        for t in original.indices.as_chunks::<3>().0.iter() {
            let input: Vec<_> = t
                .iter()
                .map(|&i| {
                    (
                        Vec3::from(original.positions[i as usize]),
                        Vec3::from(original.normals[i as usize]),
                        Vec2::from(original.uvs[i as usize]),
                    )
                })
                .collect();
            let mut output = Vec::new();
            for i in 0..3 {
                let a = input[i];
                let b = input[(i + 1) % 3];
                let da = e.ceiling_height(size, a.0.xz()) - a.0.y;
                let db = e.ceiling_height(size, b.0.xz()) - b.0.y;
                if da >= 0. {
                    output.push(a);
                }
                if (da >= 0.) != (db >= 0.) {
                    let t = da / (da - db);
                    output.push((
                        a.0.lerp(b.0, t),
                        a.1.lerp(b.1, t).normalize_or_zero(),
                        a.2.lerp(b.2, t),
                    ));
                }
            }
            for i in 1..output.len().saturating_sub(1) {
                let v = [output[0], output[i], output[i + 1]];
                if (v[1].0 - v[0].0).cross(v[2].0 - v[0].0).length_squared() < 1e-14 {
                    continue;
                }
                let base = g.positions.len() as u32;
                for (p, n, uv) in v {
                    g.positions.push(p.to_array());
                    g.normals.push(n.to_array());
                    g.uvs.push(uv.to_array());
                }
                g.indices.extend([base, base + 1, base + 2]);
            }
        }
    }
}
