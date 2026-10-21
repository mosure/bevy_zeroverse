//! Metric floor furnishings. Rug fields share edges with their bindings (no
//! overlaid coplanar decal); floor sockets expose the same port used by routing.
use super::*;

#[derive(Debug, Clone, serde::Serialize)]
pub struct FloorProgram {
    pub lid_width_fraction: f32,
    pub lid_height_fraction: f32,
    pub rug_exponent: f32,
    pub rug_taper: f32,
    pub clipped_corners: bool,
}
pub fn parameters(o: &IndoorObject) -> FloorProgram {
    let mut r = stream(o.seed, 7485);
    FloorProgram {
        lid_width_fraction: r.random_range(0.48..0.76),
        lid_height_fraction: r.random_range(0.58..1.),
        rug_exponent: r.random_range(2.0..12.),
        rug_taper: r.random_range(-0.16..0.16),
        clipped_corners: r.random_bool(0.3),
    }
}

pub(crate) fn is_open(o: &IndoorObject) -> bool {
    o.variant & 8 == 0
}

fn service_size(o: &IndoorObject) -> Vec3 {
    if (o.variant & 7).is_multiple_of(3) {
        Vec3::new(o.size.x.min(o.size.z), o.size.y, o.size.x.min(o.size.z))
    } else {
        o.size
    }
}

pub(crate) fn port_local(o: &IndoorObject) -> Vec3 {
    Vec3::new(0., 0.017, service_size(o).z * 0.10)
}

pub fn build(a: &mut Assembly, o: &IndoorObject) {
    if o.kind == ObjectKind::Rug {
        rug(a, o);
    } else {
        outlet(a, o);
    }
}

fn triangle(g: &mut Geometry, p: Vec3, q: Vec3, r: Vec3) {
    let i = g.positions.len() as u32;
    let n = (q - p).cross(r - p).normalize();
    for v in [p, q, r] {
        g.positions.push(v.to_array());
        g.normals.push(n.to_array());
        g.uvs.push(v.xz().to_array());
    }
    g.indices.extend([i, i + 1, i + 2]);
}

/// Same physical contour used by geometry, support and placement.
pub(crate) fn rug_outline(o: &IndoorObject) -> Vec<Vec2> {
    let program = parameters(o);
    let exponent = program.rug_exponent;
    (0..if program.clipped_corners { 8 } else { 64 })
        .map(|i| {
            let angle = TAU * (i as f32 + if program.clipped_corners { 0.5 } else { 0. })
                / if program.clipped_corners { 8. } else { 64. };
            let f = |v: f32| v.signum() * v.abs().powf(2. / exponent);
            Vec2::new(
                f(angle.cos())
                    * o.size.x
                    * 0.5
                    * (1. - program.rug_taper.abs() + program.rug_taper * f(angle.sin())),
                f(angle.sin()) * o.size.z * 0.5,
            )
        })
        .collect()
}

fn rug(a: &mut Assembly, o: &IndoorObject) {
    let mut rng = stream(o.seed, 7340);
    // Superellipse exponent spans oval, capsule-like and softly squared forms.
    // Rotation/aspect/edge width are independent continuous parameters.
    let _ = rng.random_range(2.0_f32..7.0);
    let border = rng.random_range(0.025..0.095) / o.size.x.min(o.size.z);
    let inset = (1. - border * 2.).clamp(0.65, 0.97);
    let outline = rug_outline(o);
    let field = Surface::FabricAlt;
    let slot = super::super::materials::rugs::slot(o.finish_seed.unwrap_or(o.seed));
    let rug_label = format!("floormat#rug{slot}");
    let edge_label = format!("floormat#rug_binding{slot}");
    let top = |p: Vec2| Vec3::new(p.x, o.size.y, p.y);
    let bottom = |p: Vec2| Vec3::new(p.x, 0., p.y);
    // Keep Cartesian motif coordinates near the center and ease into the
    // outline-aligned border. One center fan stretches patches into wedges;
    // a bounded four-ring tessellation samples the smooth interior mapping.
    let uv = |p: Vec2, radius: f32| {
        let natural = p / Vec2::new(o.size.x, o.size.z);
        let boundary = natural / natural.abs().max_element() * 0.5;
        (Vec2::splat(0.5) + natural.lerp(boundary, radius * radius) * radius).to_array()
    };
    for i in 0..outline.len() {
        let p = outline[i];
        let q = outline[(i + 1) % outline.len()];
        let g = a.part(field, &rug_label);
        for ring in 0..4 {
            let inner = ring as f32 / 4.;
            let outer = (ring + 1) as f32 / 4.;
            let mut add = |points: [(Vec2, f32); 3]| {
                let start = g.uvs.len();
                let positions = points.map(|(p, r)| top(p * r * inset));
                triangle(g, positions[0], positions[1], positions[2]);
                for (index, (p, r)) in points.into_iter().enumerate() {
                    g.uvs[start + index] = uv(p, r);
                }
            };
            if ring == 0 {
                add([(p, 0.), (q, outer), (p, outer)]);
            } else {
                add([(p, inner), (q, inner), (q, outer)]);
                add([(p, inner), (q, outer), (p, outer)]);
            }
        }
        let g = a.part(field, &edge_label);
        triangle(g, top(p * inset), top(q * inset), top(q));
        triangle(g, top(p * inset), top(q), top(p));
        triangle(g, bottom(p), top(p), top(q));
        triangle(g, bottom(p), top(q), bottom(q));
        triangle(g, Vec3::ZERO, bottom(p), bottom(q));
    }
}

fn outlet(a: &mut Assembly, o: &IndoorObject) {
    let s = service_size(o);
    let round = (o.variant & 7).is_multiple_of(3);
    let finish = if (o.variant & 7).is_multiple_of(2) {
        Surface::Metal
    } else {
        Surface::Chrome
    };
    // Raised sealed rim: plate thickness is 4 mm, independent of hinged lid height.
    if round {
        a.part(finish, "floor_outlet").cylinder(
            s.x.min(s.z) * 0.5,
            0.004,
            Transform::from_translation(Vec3::Y * 0.002),
        );
    } else {
        a.box_part(
            finish,
            "floor_outlet",
            Vec3::Y * 0.002,
            Vec3::new(s.x, 0.004, s.z),
            0.001,
        );
    }
    if !is_open(o) {
        if round {
            a.part(finish, "floor_outlet").cylinder(
                s.x * 0.41,
                0.002,
                Transform::from_translation(Vec3::Y * 0.005),
            );
        } else {
            a.box_part(
                finish,
                "floor_outlet",
                Vec3::Y * 0.005,
                Vec3::new(s.x * 0.81, 0.002, s.z * 0.81),
                0.0005,
            );
        }
        a.box_part(
            Surface::Ink,
            "floor_outlet",
            Vec3::new(0., 0.0062, s.z * 0.25),
            Vec3::new(s.x * 0.14, 0.0004, 0.002),
            0.,
        );
        return;
    }
    a.box_part(
        Surface::Plastic,
        "floor_outlet",
        Vec3::new(0., 0.010, s.z * 0.10),
        Vec3::new(s.x * 0.48, 0.012, s.z * 0.48),
        0.003,
    );
    // A mains socket/USB service panel, inset dark openings with separated faces.
    let port = port_local(o);
    for dx in [-0.014, 0.014] {
        a.box_part(
            Surface::Ink,
            "floor_outlet",
            port + Vec3::new(dx, -0.0007, 0.),
            Vec3::new(0.005, 0.0004, 0.012),
            0.,
        );
    }
    a.part(Surface::Ink, "floor_outlet").cylinder(
        0.003,
        0.0004,
        Transform::from_translation(port + Vec3::new(0., -0.0007, 0.014)),
    );
    // Hinged upright cover is behind the socket and entirely inside its reserve.
    let hinge_z = -s.z * 0.27;
    a.part(finish, "floor_outlet").rod(
        Vec3::new(-s.x * 0.34, 0.013, hinge_z),
        Vec3::new(s.x * 0.34, 0.013, hinge_z),
        0.006,
    );
    let program = parameters(o);
    let height = (s.y - 0.016) * program.lid_height_fraction;
    a.part(finish, "floor_outlet").cuboid(
        Vec3::new(s.x * program.lid_width_fraction, height, 0.004),
        0.001,
        Transform::from_translation(Vec3::new(0., 0.014 + height * 0.5, hinge_z - 0.007)),
    );
    for x in [-0.38, 0.38] {
        a.part(Surface::Ink, "floor_outlet").cylinder(
            0.0025,
            0.0006,
            Transform::from_translation(Vec3::new(s.x * x, 0.0043, 0.)),
        );
    }
}

/// Electrical routing tests the occupied parts, not the empty air under an open
/// lid. The ordinary whole-object reserve still protects floor placement.
pub(crate) fn cable_hits(o: &IndoorObject, start: Vec3, end: Vec3, radius: f32) -> bool {
    let s = service_size(o);
    if !is_open(o) {
        return super::super::layout::segment_hits_box(
            start,
            end,
            Vec3::new(-s.x * 0.5, 0., -s.z * 0.5) - Vec3::splat(radius),
            Vec3::new(s.x * 0.5, s.y, s.z * 0.5) + Vec3::splat(radius),
        );
    }
    let p = parameters(o);
    let boxes = [
        (
            Vec3::new(-s.x * 0.5, 0., -s.z * 0.5),
            Vec3::new(s.x * 0.5, 0.004, s.z * 0.5),
        ),
        (
            Vec3::new(-s.x * 0.24, 0.004, -s.z * 0.14),
            Vec3::new(s.x * 0.24, 0.016, s.z * 0.34),
        ),
        (
            Vec3::new(
                -s.x * p.lid_width_fraction * 0.5,
                0.014,
                -s.z * 0.27 - 0.009,
            ),
            Vec3::new(
                s.x * p.lid_width_fraction * 0.5,
                0.014 + (s.y - 0.016) * p.lid_height_fraction,
                -s.z * 0.27 - 0.005,
            ),
        ),
        (
            Vec3::new(-s.x * 0.34 - 0.006, 0.007, -s.z * 0.27 - 0.006),
            Vec3::new(s.x * 0.34 + 0.006, 0.019, -s.z * 0.27 + 0.006),
        ),
    ];
    boxes.into_iter().any(|(lo, hi)| {
        super::super::layout::segment_hits_box(
            start,
            end,
            lo - Vec3::splat(radius),
            hi + Vec3::splat(radius),
        )
    })
}
