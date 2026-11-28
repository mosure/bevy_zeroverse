use super::*;
mod adjoining;
mod ceiling_mesh;
mod column_finish;
mod floors;
mod lining_mesh;
#[cfg(test)]
mod normal_tests;
mod wall_section;
use crate::scene::procedural_indoor::{
    architecture, geometry::Geometry, layout::AdjoiningFloor, materials::Surface, objects::Assembly,
};
pub(crate) const SHARED_WALL_THICKNESS_M: f32 = 0.18;

/// Flat, metric UVs. Triangle winding and stored normals always agree, including
/// negative floor levels, angled reveals and the underside of a sloping ceiling.
pub fn triangle(g: &mut Geometry, p: [Vec3; 3], outward: Vec3) {
    triangle_with_normal(g, p, outward, None);
}

fn triangle_with_normal(
    g: &mut Geometry,
    mut p: [Vec3; 3],
    outward: Vec3,
    planar_normal: Option<Vec3>,
) {
    let mut n = (p[1] - p[0]).cross(p[2] - p[0]);
    if n.length_squared() < 1e-14 {
        return;
    }
    if n.dot(outward) < 0. {
        p.swap(1, 2);
        n = -n;
    }
    n = planar_normal.unwrap_or_else(|| n.normalize());
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

/// Clipped wall fronts share an analytic plane. A valid, micrometre-wide
/// triangle can acquire a very different cross-product normal after its world
/// positions round to f32. Retain every vertex/triangle and its winding, but
/// derive shading and metric UVs from that known plane, not the ill-conditioned
/// sliver. This is restricted to planar wall panels; general/sloping geometry
/// continues to derive its normal from its vertices.
fn wall_face(g: &mut Geometry, points: &[Vec3], outward: Vec3) {
    let normal = outward.normalize();
    for i in 1..points.len().saturating_sub(1) {
        triangle_with_normal(
            g,
            [points[0], points[i], points[i + 1]],
            outward,
            Some(normal),
        );
    }
}

fn face(g: &mut Geometry, p: &[Vec3], outward: Vec3) {
    for i in 1..p.len().saturating_sub(1) {
        triangle(g, [p[0], p[i], p[i + 1]], outward);
    }
}
fn quad(g: &mut Geometry, p: [Vec3; 4], outward: Vec3) {
    face(g, &p, outward);
}

pub(super) fn panel(g: &mut Geometry, points: &[Vec3], extrusion: Vec3) {
    panel_faces(points, extrusion, |p, n, _| face(g, p, n));
}

fn panel_faces(points: &[Vec3], extrusion: Vec3, mut emit: impl FnMut(&[Vec3], Vec3, bool)) {
    if points.len() < 3 {
        return;
    }
    let mut p = points.to_vec();
    if (p[1] - p[0]).cross(p[2] - p[0]).dot(extrusion) < 0. {
        p.reverse();
    }
    emit(&p, extrusion, true);
    emit(
        &p.iter().map(|p| *p - extrusion).collect::<Vec<_>>(),
        -extrusion,
        false,
    );
    for i in 0..p.len() {
        let a = p[i];
        let b = p[(i + 1) % p.len()];
        emit(
            &[a, b, b - extrusion, a - extrusion],
            (b - a).cross(extrusion),
            false,
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

/// Trim end cuts account for adjacent wall angles, including acute chamfers.
fn trim_span(e: &EnvelopeProgram, edge: usize, depth: f32, reveal: f32) -> Vec2 {
    let n = e.footprint.len();
    let p = e.footprint[edge];
    let q = e.footprint[(edge + 1) % n];
    let along = (q - p).normalize();
    let before = (p - e.footprint[(edge + n - 1) % n]).normalize();
    let after = (e.footprint[(edge + 2) % n] - q).normalize();
    let cut =
        |dot: f32| reveal.max(depth * ((1. - dot).max(0.) / (1. + dot).max(1e-6)).sqrt() + 0.004);
    Vec2::new(
        -p.distance(q) * 0.5 + cut(before.dot(along)),
        p.distance(q) * 0.5 - cut(along.dot(after)),
    )
}

#[test]
fn trim_endpoints_clear_acute_and_rectangular_corners() {
    use crate::scene::procedural_indoor::layout::IndoorLayout;
    let scene = IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.5, 0, 0.).unwrap();
    let mut e = scene.envelope.unwrap();
    for footprint in [
        vec![Vec2::ZERO, Vec2::new(4., 0.), Vec2::new(0.3, 2.)],
        vec![
            Vec2::ZERO,
            Vec2::new(4., 0.),
            Vec2::new(4., 3.),
            Vec2::new(0., 3.),
        ],
    ] {
        e.footprint = footprint;
        for edge in 0..e.footprint.len() {
            for depth in [0.01, 0.04, 0.07] {
                let span = trim_span(&e, edge, depth, 0.045);
                for x in [span.x, span.y] {
                    let point = e
                        .wall_transform(edge)
                        .transform_point(Vec3::new(x, 0.1, depth));
                    assert!(
                        polygon::contains(&e.footprint, point.xz(), depth - 1e-5),
                        "edge {edge}, depth {depth}, point {point:?}"
                    );
                }
            }
        }
    }
}

pub(super) fn wall(
    a: &mut Assembly,
    scene: &IndoorManifest,
    edge: usize,
    holes: &[architecture::facade::WindowOpening],
    trim: Option<&architecture::trim::TrimProgram>,
    surface: Surface,
) {
    let e = scene.envelope.as_ref().unwrap();
    let size = scene.room_size;
    let p = e.footprint[edge];
    let q = e.footprint[(edge + 1) % e.footprint.len()];
    let length = p.distance(q);
    let tf = e.wall_transform(edge);
    let hp = e.ceiling_height(size, p);
    let hq = e.ceiling_height(size, q);
    let lining = e
        .walls
        .iter()
        .find(|w| w.edge == edge)
        .and_then(|w| w.lining.as_ref());
    let finish_line = lining.map_or(0., |s| s.installed_height(hp, hq));
    if let Some(s) = lining {
        lining_mesh::build(a, tf, s, length, finish_line, holes);
    }
    let section = wall_section::WallSection::new(
        &e.footprint,
        edge,
        |p| e.ceiling_height(size, p),
        |i| {
            if e.shared_edge(i, size) {
                SHARED_WALL_THICKNESS_M
            } else {
                e.wall_thickness()
            }
        },
    );
    section.build(holes, finish_line, |p, n, front| {
        wall_face(
            a.part(if front { surface } else { Surface::Paint }, "wall"),
            p,
            n,
        );
    });
    if e.courtyard.is_some() {
        return; // Masonry plinths run to the paving on these outdoor elevations.
    }
    if let Some(trim) = trim {
        if let Some(section) = &trim.base {
            let cutouts = e
                .walls
                .iter()
                .find(|w| w.edge == edge)
                .and_then(|w| w.facade.as_ref())
                .map(architecture::facade::Facade::trim_openings);
            let cutouts = architecture::trim::fitted_openings(
                scene,
                tf,
                section,
                cutouts.as_deref().unwrap_or(holes),
            );
            if trim.miter_joint_m.is_some() {
                architecture::trim::perimeter::build(
                    a, scene, edge, trim, section, &cutouts, false,
                );
                return;
            }
            architecture::trim::build(
                a,
                tf,
                trim_span(e, edge, section.depth_m, trim.corner_reveal_m),
                Vec2::splat(trim.joint_m),
                section,
                &cutouts,
                false,
            );
        }
        return;
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

/// Roof-following perimeter moulding, including its cuts at glazed openings.
fn cornice(
    a: &mut Assembly,
    scene: &IndoorManifest,
    edge: usize,
    holes: &[architecture::facade::WindowOpening],
) {
    use rand::Rng;
    let e = scene.envelope.as_ref().unwrap();
    if e.courtyard.is_some() {
        return;
    }
    if let Some(trim) = architecture::trim::TrimProgram::for_scene(scene) {
        if let Some(section) = &trim.crown {
            if trim.miter_joint_m.is_some() {
                architecture::trim::perimeter::build(a, scene, edge, trim, section, holes, true);
                return;
            }
            let tf = e.wall_transform(edge);
            let span = trim_span(e, edge, section.depth_m, trim.corner_reveal_m);
            let at = |x| {
                scene
                    .ceiling_height(tf.transform_point(Vec3::new(x, 0., section.depth_m)).xz())
                    .min(scene.ceiling_height(tf.transform_point(Vec3::new(x, 0., 0.)).xz()))
                    - trim.joint_m
                    - section.height_m
            };
            architecture::trim::build(
                a,
                tf,
                span,
                Vec2::new(at(span.x), at(span.y)),
                section,
                holes,
                true,
            );
        }
        return;
    }
    let mut rng = super::super::layout::stream(scene.seed, 163);
    if rng.random_bool(0.12) {
        return;
    }
    let height = rng.random_range(0.055..0.135);
    let depth = rng.random_range(0.022..0.055);
    let tf = e.wall_transform(edge);
    let length = e.footprint[edge].distance(e.footprint[(edge + 1) % e.footprint.len()]);
    if length < 0.20 {
        return;
    }
    let at = |x| scene.ceiling_height(tf.transform_point(Vec3::new(x, 0., depth)).xz()) - 0.004;
    let hp = at(-length * 0.5);
    let hq = at(length * 0.5);
    let mid = (hp + hq) * 0.5;
    let slope = (hq - hp) / length;
    let surface = if scene.architecture_style == super::super::layout::ArchitectureStyle::Timber {
        Surface::WoodEdge
    } else {
        Surface::Paint
    };
    for (lo, hi) in architecture::facade::solid_rectangles(
        Vec2::new(-length * 0.5 + 0.065, hp.min(hq) - height),
        Vec2::new(length * 0.5 - 0.065, hp.max(hq)),
        holes,
    ) {
        let rect = [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)];
        let top = polygon::clip(&rect, Vec2::new(slope, -1.), -mid);
        let band = polygon::clip(&top, Vec2::new(-slope, 1.), mid - height);
        panel(
            a.part(surface, "ceiling"),
            &band
                .iter()
                .map(|p| tf.transform_point(Vec3::new(p.x, p.y, depth)))
                .collect::<Vec<_>>(),
            tf.rotation * Vec3::Z * (depth + 0.002),
        );
    }
}

pub fn build(scene: &IndoorManifest) -> Assembly {
    bevy::tasks::block_on(build_with_yields(scene, || std::future::ready(())))
}

/// The native builder runs on a preparation thread. The single-threaded browser
/// can give rendering and input time between the same ordered construction steps.
pub(crate) async fn build_with_yields<F: std::future::Future<Output = ()>>(
    scene: &IndoorManifest,
    yield_if_due: impl Fn() -> F,
) -> Assembly {
    let e = scene.envelope.as_ref().unwrap();
    let size = scene.room_size;
    let mut a = Assembly {
        motion_policy: scene.indoor_motion.as_ref().map(|m| m.policy.clone()),
        ..default()
    };
    floors::build(&mut a, e, scene);
    yield_if_due().await;
    let mut guard_posts = Vec::new();
    for (start, end) in e.tier_guards() {
        if let Some(g) = &e.tier_guard {
            guardrail::build(
                &mut a,
                &g.section,
                guardrail::GuardRun {
                    start,
                    end,
                    height: g.height,
                    stair_rise: 0.,
                },
                |p| p.y,
                &mut guard_posts,
            );
            continue;
        }
        let rail = a.part(Surface::Metal, "other_structure");
        rail.rod(start + Vec3::Y * 1.03, end + Vec3::Y * 1.03, 0.022);
        rail.rod(start + Vec3::Y * 0.53, end + Vec3::Y * 0.53, 0.015);
        let count = (start.distance(end) / 0.9).ceil().max(1.) as usize;
        for i in 0..=count {
            let p = start.lerp(end, i as f32 / count as f32);
            rail.rod(p, p + Vec3::Y * 1.03, 0.018);
        }
    }
    let exposed = scene
        .ceiling_installation
        .as_ref()
        .is_some_and(|c| c.exposed_slab);
    if let Some(roof) = &e.courtyard {
        roof.build(&mut a, size);
    } else if let Some(grid) = architecture::ceiling::CeilingGrid::for_scene(scene) {
        architecture::ceiling::build_room_grid(&mut a, scene, grid, &yield_if_due).await;
    } else {
        ceiling_mesh::build(
            a.part(
                if exposed {
                    scene
                        .ceiling_installation
                        .as_ref()
                        .and_then(|c| c.framing.as_ref())
                        .map_or(Surface::Concrete, |f| f.roof_material)
                } else {
                    Surface::Ceiling
                },
                "ceiling",
            ),
            e,
            size,
        );
    }
    let feature_edges = super::super::materials::wall::feature_edges(scene);
    let courtyard_finish = e.courtyard.map(|_| {
        use rand::Rng;
        let mut finish = super::super::layout::stream(scene.material_seed(), 0x5941524457414C4C);
        if finish.random_bool(0.40) {
            Surface::Concrete
        } else {
            Surface::Paint
        }
    });
    for entry in &e.walls {
        yield_if_due().await;
        let holes = entry
            .facade
            .as_ref()
            .map_or(&[][..], |f| f.openings.as_slice());
        wall(
            &mut a,
            scene,
            entry.edge,
            holes,
            architecture::trim::TrimProgram::for_scene(scene),
            if let Some(finish) = courtyard_finish {
                finish
            } else if feature_edges.contains(&entry.edge) {
                Surface::Accent
            } else {
                Surface::Paint
            },
        );
        cornice(&mut a, scene, entry.edge, holes);
        if let Some(f) = &entry.facade {
            architecture::facade::glazing_at(&mut a, e.wall_transform(entry.edge), f);
        }
    }
    yield_if_due().await;
    shared_and_neighbor(&mut a, scene, e);
    // Partition tops are cut by the same analytic roof plane, not scaled: door
    // dimensions and all metric material coordinates are preserved.
    let mut partitions = Assembly {
        motion_policy: a.motion_policy.clone(),
        ..default()
    };
    super::super::floorplan::build(scene, &mut partitions);
    clip_roof(&mut partitions, e, size);
    a.append_assembly(partitions);
    for p in &e.pillars {
        yield_if_due().await;
        if p.profile.is_some() {
            column(&mut a, p, e, size);
            continue;
        }
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
        mezzanine::build(&mut a, m);
    }
    architecture::fixture_geometry(&mut a, scene);
    yield_if_due().await;
    architecture::ceiling::build(&mut a, scene);
    if let Some(frame) = super::super::materials::flooring::laying::mapping(scene) {
        // Set UVs before Bevy generates tangents so normal/roughness/color
        // all follow the actual installation, including across slab cuts.
        super::super::materials::flooring::laying::apply(a.part(Surface::Floor, "floor"), frame);
    }
    let mut exterior = Assembly::default();
    yield_if_due().await;
    architecture::facade::backdrop(&mut exterior, scene);
    for ((surface, label), g) in exterior.parts {
        a.part(surface, &format!("{label}#exterior")).append(g);
    }
    yield_if_due().await;
    architecture::background::append(&mut a, scene);
    a
}

/// Joined cross-section rings, including the base reveal. No duplicate interior
/// discs between the collar and shaft; the roof joint follows the actual plane.
fn column(a: &mut Assembly, p: &Pillar, e: &EnvelopeProgram, size: Vec3) {
    let shape = p.profile.as_ref().unwrap();
    let section: Vec<_> = (0..p.sides)
        .map(|i| shape.section_vertex(i, p.sides))
        .collect();
    let h = shape.collar_height;
    if let Some(stock) = shape.construction {
        column_finish::build(a, p, shape, stock, &section, |v| e.ceiling_height(size, v));
        return;
    }
    let rings: Vec<Vec<Vec3>> = [
        (p.radius + 0.019, 0.),
        (p.radius + 0.025, 0.008),
        (p.radius + 0.025, h - 0.008),
        (p.radius + 0.019, h),
        (p.radius, h),
        (p.radius * shape.taper, f32::INFINITY),
    ]
    .into_iter()
    .map(|(radius, y)| {
        section
            .iter()
            .map(|v| {
                let v = p.center + *v * radius;
                Vec3::new(
                    v.x,
                    if y.is_infinite() {
                        e.ceiling_height(size, v)
                    } else {
                        y
                    },
                    v.y,
                )
            })
            .collect()
    })
    .collect();
    for (index, pair) in rings.windows(2).enumerate() {
        let g = a.part(
            if index < 4 {
                Surface::Metal
            } else {
                Surface::Concrete
            },
            "other_structure",
        );
        for i in 0..section.len() {
            let j = (i + 1) % section.len();
            let radial = (section[i] + section[j]) * 0.5;
            let normal = if index == 3 {
                Vec3::Y
            } else {
                Vec3::new(radial.x, 0., radial.y)
            };
            quad(g, [pair[0][i], pair[0][j], pair[1][j], pair[1][i]], normal);
        }
    }
    face(
        a.part(Surface::Metal, "other_structure"),
        &rings[0],
        Vec3::NEG_Y,
    );
    face(
        a.part(Surface::Concrete, "other_structure"),
        rings.last().unwrap(),
        Vec3::Y,
    );
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
            a.part(Surface::WoodEdge, "door_frame")
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
            "door_frame",
            p.position(x, h * 0.5),
            size,
            0.002,
        );
    }
}

fn shared_and_neighbor(a: &mut Assembly, scene: &IndoorManifest, e: &EnvelopeProgram) {
    use architecture::facade::WindowOpening;
    let size = scene.room_size;
    let hz = size.z * 0.5;
    let edge = (0..e.footprint.len())
        .find(|&i| e.shared_edge(i, size))
        .unwrap();
    let height = architecture::interior_glazing::partition_height(scene);
    let door = -scene.door_x;
    let mut openings = vec![WindowOpening {
        glazing: None,
        min: Vec2::new(door - scene.entry_width() * 0.5, 0.),
        max: Vec2::new(door + scene.entry_width() * 0.5, scene.entry_height()),
        columns: 1,
        transoms: Vec::new(),
        transom: 0.,
        operable: false,
    }];
    let f = architecture::interior_glazing::shared_facade(scene, height);
    openings.extend(f.openings.iter().cloned());
    wall(
        a,
        scene,
        edge,
        &openings,
        architecture::trim::TrimProgram::for_scene(scene),
        Surface::Paint,
    );
    if architecture::trim::TrimProgram::for_scene(scene).is_some_and(|p| p.miter_joint_m.is_some())
    {
        cornice(a, scene, edge, &openings);
    }
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
    if let Some(door) = &scene.entry_door {
        door.build(a, true);
    } else {
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
    }
    // Shared partition is the front edge of the neighboring volume. Exterior
    // walls start beyond it; they never cover the passage or its glass.
    let adjacent = scene.adjoining_space();
    let poly = adjacent.footprint(size);
    let floor = match adjacent.floor {
        AdjoiningFloor::Continuous => Surface::Floor,
        AdjoiningFloor::Concrete => Surface::Concrete,
    };
    slab(a.part(floor, "floor"), &poly, |_| 0., 0.20);
    slab(
        a.part(Surface::Ceiling, "ceiling"),
        &poly,
        |p| e.ceiling_height(size, p) + 0.14,
        0.14,
    );
    adjoining::perimeter(a, scene, e, &poly);
}

/// Cut partition tops to the same analytic roof used for placement and cameras.
pub fn clip_roof(a: &mut Assembly, e: &EnvelopeProgram, size: Vec3) {
    for g in a.parts.values_mut() {
        g.clip_half_space(|p| f64::from(e.ceiling_height(size, p.xz())) - f64::from(p.y));
    }
}
