use super::*;
use crate::scene::procedural_indoor::{envelope::tests::ray_triangle, layout::IndoorLayout};

fn triangles(g: &Geometry) -> impl Iterator<Item = [Vec3; 3]> + '_ {
    g.indices
        .as_chunks::<3>()
        .0
        .iter()
        .map(|t| t.map(|i| g.positions[i as usize].into()))
}

#[test]
fn acoustic_fabric_owns_the_finished_corners_from_each_view() {
    // Both ends of this panel used to share 0.60 m of face with a paint cap.
    let s =
        IndoorManifest::generate_with_humans(93510, IndoorLayout::Mixed, 0.65, 3, 0.25).unwrap();
    let e = s.envelope.as_ref().unwrap();
    let mesh = super::super::build(&s);
    let part = e.walls.iter().find(|w| w.lining.is_some_and(|l| l.stock == crate::scene::procedural_indoor::envelope::lining::LiningStock::AcousticFabric) && {
        let i = w.edge; let v = e.footprint[i]; let t = e.footprint[(i+1)%e.footprint.len()];
        (v.y-t.y).abs()<1e-5 && v.y > -s.room_size.z*0.5 + 0.2
    }).unwrap();
    let tf = e.wall_transform(part.edge);
    let half =
        e.footprint[part.edge].distance(e.footprint[(part.edge + 1) % e.footprint.len()]) * 0.5;
    let n = tf.rotation * Vec3::Z;
    for side in [-1., 1.] {
        for x in [0.073, 0.187, 0.314, 0.476] {
            // Stay in the fabric field, including right beside the seam.
            let point = tf.transform_point(Vec3::new(side * (half - x), 1.473, 0.));
            for lateral in [-1.45, 0.16, 1.71] {
                let dir = (tf.rotation * Vec3::new(lateral, 0.083, -1.)).normalize();
                let origin = point - dir * 0.6;
                let hits: Vec<_> = mesh
                    .parts
                    .iter()
                    .filter(|((_, l), _)| l.split('#').next() == Some("wall"))
                    .flat_map(|((s, _), g)| {
                        triangles(g).filter_map(move |p| {
                            let t = ray_triangle(origin, dir, p)?;
                            ((t - 0.6).abs() < 0.0002
                                && (p[1] - p[0]).cross(p[2] - p[0]).normalize_or_zero().dot(n)
                                    > 0.98)
                                .then_some(*s)
                        })
                    })
                    .collect();
                assert_eq!(
                    hits,
                    [Surface::FabricAlt],
                    "side {side}, x {x}, lateral {lateral}"
                );
            }
        }
    }
}

#[test]
fn a_short_or_sloping_receiver_only_hides_the_part_of_the_end_it_covers() {
    for yaw in [0., 0.76] {
        let rot = Quat::from_rotation_y(yaw);
        let to_world = |x, y, z| rot * Vec3::new(x, y, z) + Vec3::new(3., 0., 2.);
        let outline = [
            (0., 0.),
            (1., 0.),
            (1., 1.),
            (1.25, 1.),
            (1.25, 0.),
            (5., 0.),
            (5., 4.),
            (0., 4.),
        ]
        .map(|(x, z)| to_world(x, 0., z).xz());
        let section = WallSection::new(
            &outline,
            1,
            |v| {
                let local = rot.inverse() * (Vec3::new(v.x, 0., v.y) - Vec3::new(3., 0., 2.));
                3. - 0.5 * (local.x - 1.)
            },
            |_| 0.55,
        );
        let mut g = Geometry::default();
        section.build(&[], 0., |p, n, _| wall_face(&mut g, p, n));
        for (x, y, visible) in [
            (1.131, 1.47, false),
            (1.131, 2.91, false),
            (1.131, 2.975, true),
            (1.368, 1.47, true),
            (1.131, 3.03, false),
        ] {
            let direction = rot * Vec3::NEG_Z;
            let near = triangles(&g)
                .filter_map(|p| ray_triangle(to_world(x, y, 1.3), direction, p))
                .any(|t| (t - 0.3).abs() < 1e-5);
            assert_eq!(near, visible, "yaw {yaw}, x {x}, elevation {y}");
        }
    }
}

#[test]
fn wall_returns_exist_at_the_opening_edges_only() {
    let outline = [
        Vec2::new(-2., 0.),
        Vec2::new(2., 0.),
        Vec2::new(2., 4.),
        Vec2::new(-2., 4.),
    ];
    let section = WallSection::new(&outline, 0, |_| 3., |_| 0.26);
    let hole = |x, y, top| WindowOpening {
        min: Vec2::new(x, y),
        max: Vec2::new(x + 0.7, top),
        glazing: None,
        columns: 1,
        transom: 0.,
        transoms: vec![],
        operable: false,
    };
    let mut g = Geometry::default();
    section.build(
        &[hole(-1.7, 0., 2.2), hole(0.45, 0.95, 2.2)],
        0.,
        |p, n, _| wall_face(&mut g, p, n),
    );
    let hits = |p, dir| {
        let mut ts: Vec<_> = triangles(&g)
            .filter_map(|t| ray_triangle(p, dir, t))
            .collect();
        ts.sort_unstable_by(f32::total_cmp);
        ts
    };
    // In the solid between these windows only roof and floor close the section.
    let top = hits(Vec3::new(-0.31, 0.12, -0.091), Vec3::Y);
    assert_eq!(top.len(), 1);
    assert!((top[0] - 2.88).abs() < 1e-5);
    let sill = hits(Vec3::new(0.72, 1.37, -0.091), Vec3::NEG_Y);
    assert!((sill[0] - 0.42).abs() < 1e-5);
    assert!(hits(Vec3::new(-1.36, 1.3, -0.109), Vec3::NEG_Y).is_empty());
    assert!(hits(Vec3::new(0.71, 1.44, 0.5), Vec3::NEG_Z).is_empty());
}

#[test]
fn facing_caps_at_a_narrow_notch_present_one_closed_exterior_skin() {
    for (left_height, right_height) in [(3.3, 3.), (3., 3.3), (3., 3.)] {
        let outline = [
            (-2., 0.),
            (0., 0.),
            (0., 1.3),
            (0.95, 1.3),
            (0.95, 0.),
            (3., 0.),
            (3., 4.),
            (-2., 4.),
        ]
        .map(|(x, y)| Vec2::new(x, y));
        let h = |p: Vec2| if p.x < 0.5 { left_height } else { right_height };
        let mut caps = Geometry::default();
        for edge in [1, 3] {
            WallSection::new(&outline, edge, h, |_| 0.68).build(&[], 0., |p, n, _| {
                if n.normalize().dot(Vec3::NEG_Z) > 0.999 && p.iter().all(|v| v.z.abs() < 1e-5) {
                    wall_face(&mut caps, p, n);
                }
            });
        }
        let count = |x, y| {
            triangles(&caps)
                .filter(|p| ray_triangle(Vec3::new(x, y, -1.), Vec3::Z, *p).is_some())
                .count()
        };
        for x in [0.071, 0.221, 0.301, 0.469, 0.511, 0.661, 0.731, 0.881] {
            assert_eq!(count(x, 1.731), 1, "{x}m at {left_height}/{right_height}");
            let tallest = if x < 0.27 {
                left_height
            } else if x > 0.68 {
                right_height
            } else {
                left_height.max(right_height)
            };
            assert_eq!(count(x, 3.148), usize::from(tallest > 3.148));
        }
    }
}
