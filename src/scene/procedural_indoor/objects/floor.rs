//! Metric floor furnishings. Rug fields share edges with their bindings (no
//! overlaid coplanar decal); floor sockets expose the same port used by routing.
use super::*;
pub mod rug;
pub mod service;
pub(crate) use rug::outline as rug_outline;

/// Parameters of archived outlet and rug meshes. Current hardware and rug cuts
/// are available from [`service::parameters`] and [`rug::parameters`].
#[derive(Debug, Clone, serde::Serialize)]
pub struct FloorProgram {
    pub lid_width_fraction: f32,
    pub lid_height_fraction: f32,
    pub rug_exponent: f32,
    pub rug_taper: f32,
    pub clipped_corners: bool,
    pub outlet_finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> FloorProgram {
    let mut r = stream(o.seed, 7485);
    FloorProgram {
        lid_width_fraction: r.random_range(0.48..0.76),
        lid_height_fraction: r.random_range(0.58..1.),
        rug_exponent: r.random_range(2.0..12.),
        rug_taper: r.random_range(-0.16..0.16),
        clipped_corners: r.random_bool(0.3),
        outlet_finish: finish(o),
    }
}

fn finish(o: &IndoorObject) -> Surface {
    if o.kind != ObjectKind::FloorOutlet {
        return Surface::Metal;
    }
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x464c4f4f5246494e);
    [
        Surface::Metal,
        Surface::Metal,
        Surface::Chrome,
        Surface::Chrome,
        Surface::Plastic,
    ][f.random_range(0..5)]
}

pub(crate) fn is_open(o: &IndoorObject) -> bool {
    o.variant & 8 == 0
}

pub(crate) fn service_size(o: &IndoorObject) -> Vec3 {
    if (o.variant & 7).is_multiple_of(3) {
        Vec3::new(o.size.x.min(o.size.z), o.size.y, o.size.x.min(o.size.z))
    } else {
        o.size
    }
}

pub(crate) fn port_local(o: &IndoorObject) -> Option<Vec3> {
    if !is_open(o) {
        return None;
    }
    if let Some(p) = service::parameters(o) {
        return p.port(0);
    }
    Some(legacy_port(o))
}

fn legacy_port(o: &IndoorObject) -> Vec3 {
    Vec3::new(0., 0.017, service_size(o).z * 0.10)
}

pub fn build(a: &mut Assembly, o: &IndoorObject) {
    if o.kind == ObjectKind::Rug {
        rug::build(a, o);
    } else if let Some(p) = service::parameters(o) {
        service::build(a, o, &p);
    } else {
        outlet(a, o);
    }
}

fn outlet(a: &mut Assembly, o: &IndoorObject) {
    let s = service_size(o);
    let round = (o.variant & 7).is_multiple_of(3);
    let finish = parameters(o).outlet_finish;
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
    let port = legacy_port(o);
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
pub(crate) fn legacy_cable_hits(o: &IndoorObject, start: Vec3, end: Vec3, radius: f32) -> bool {
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn floor_box_finishes_do_not_determine_geometry_ports_or_operating_state() {
        let mut combinations = std::collections::BTreeSet::new();
        for seed in 0..64 {
            for variant in [0, 1, 2, 8, 9, 10] {
                let mut o = super::super::audit::fixture(ObjectKind::FloorOutlet, seed);
                o.variant = variant;
                o.size.y = if is_open(&o) { 0.13 } else { 0.006 };
                // The geometry audit fixture intentionally fixes appearance.
                // This test explicitly varies the independent finish input.
                o.finish_seed = Some(seed);
                let p = parameters(&o);
                combinations.insert((variant, format!("{:?}", p.outlet_finish)));
                let a = super::super::build_object(&o);
                super::super::audit::check(&o, &a).unwrap();
                let port = port_local(&o);
                let points = |a: Assembly| {
                    let mut p: Vec<_> = a.parts.into_values().flat_map(|g| g.positions).collect();
                    p.sort_by(|a, b| {
                        a[0].total_cmp(&b[0])
                            .then(a[1].total_cmp(&b[1]))
                            .then(a[2].total_cmp(&b[2]))
                    });
                    p
                };
                o.finish_seed = Some(99);
                assert_eq!(port, port_local(&o));
                assert_eq!(points(a), points(super::super::build_object(&o)));
            }
        }
        assert_eq!(combinations.len(), 18);
    }
}
