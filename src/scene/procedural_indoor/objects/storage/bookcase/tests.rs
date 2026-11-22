use super::*;
use crate::scene::procedural_indoor::placement::{rotated_half, Footprint};

#[test]
fn upright_book_runs_use_both_ends_and_the_middle_of_shelves() {
    let mut bins = [0; 4];
    for seed in 0..128 {
        let o = audit::fixture(ObjectKind::Bookcase, seed);
        let p = parameters(&o);
        for (index, bay) in p.bays.iter().enumerate() {
            for level in 0..bay.levels.len() - 1 {
                let v = contents::placements(&o, &p, index, level);
                if v.is_empty() || !v.iter().all(|v| v.upright && v.lean == 0.) {
                    continue;
                }
                let left = v
                    .iter()
                    .map(|v| v.position.x - rotated_half(v.size.xz(), v.yaw).x)
                    .fold(f32::INFINITY, f32::min);
                let right = v
                    .iter()
                    .map(|v| v.position.x + rotated_half(v.size.xz(), v.yaw).x)
                    .fold(f32::NEG_INFINITY, f32::max);
                let slack = bay.right - bay.left - (right - left) - 0.008;
                if slack < 0.06 {
                    continue;
                }
                let u = (left - bay.left - 0.004) / slack;
                assert!((-1e-4..=1.0001).contains(&u));
                bins[((u * 4.) as usize).min(3)] += 1;
            }
        }
    }
    assert!(
        bins.iter().all(|&v| v > 20),
        "shelf row alignment collapsed: {bins:?}"
    );
}

#[test]
fn bookcase_contents_fit_real_shelf_compartments_and_replay() {
    for seed in 0..96 {
        for size in [Vec3::new(0.7, 1.35, 0.28), Vec3::new(1.6, 2.1, 0.52)] {
            let mut o = audit::fixture(ObjectKind::Bookcase, seed);
            o.size = size;
            let p = parameters(&o);
            let whole = build_object(&o);
            audit::check(&o, &whole).unwrap();
            for (index, b) in p.bays.iter().enumerate() {
                for level in 0..b.levels.len() - 1 {
                    let base = b.levels[level] + p.panel_m * 0.5;
                    let ceiling = b.levels[level + 1] - p.panel_m * 0.5;
                    let rear = -size.z * 0.5 + if p.closed_back { p.panel_m } else { 0. };
                    let children = contents::placements(&o, &p, index, level);
                    check_separation(
                        &children,
                        b,
                        base,
                        ceiling,
                        rear,
                        size.z * 0.5,
                        p.open_frame,
                    );
                    for child in children {
                        let mut mesh = Assembly::default();
                        contents::build(&mut mesh, &o, &child);
                        let (lo, hi) = mesh.bounds();
                        assert!(
                            lo.y >= base - 1e-5 && hi.y <= ceiling,
                            "{seed}: child clips shelf"
                        );
                        assert!(
                            lo.x >= b.left
                                && hi.x <= b.right
                                && lo.z >= rear
                                && hi.z <= size.z * 0.5,
                            "{seed}: child escapes bay: {lo:?}/{hi:?}"
                        );
                        assert!((lo.y - base).abs() < 0.001, "{seed}: floating child");
                    }
                }
            }
            let replay = build_object(&o);
            for (key, mesh) in whole.parts {
                assert_eq!(mesh.positions, replay.parts[&key].positions);
                assert_eq!(mesh.indices, replay.parts[&key].indices);
            }
        }
    }
}

/// Independent separating-axis oracle for the thickness/height section. A
/// neighbour can share an overhead footprint while touching only at its edge.
fn section(c: &contents::Content) -> [Vec2; 4] {
    let half = c.size.x * 0.5;
    let rotation = Mat2::from_angle(c.lean);
    [
        Vec2::new(-half, 0.),
        Vec2::new(half, 0.),
        Vec2::new(half, c.size.y),
        Vec2::new(-half, c.size.y),
    ]
    .map(|p| c.position.xy() + rotation * p)
}

fn signed_gap(a: [Vec2; 4], b: [Vec2; 4]) -> f32 {
    let extent = |v: [Vec2; 4], axis: Vec2| {
        let dots = v.map(|p| p.dot(axis));
        (
            dots.into_iter().fold(f32::INFINITY, f32::min),
            dots.into_iter().fold(f32::NEG_INFINITY, f32::max),
        )
    };
    let mut largest = f32::NEG_INFINITY;
    for points in [a, b] {
        for i in 0..4 {
            let axis = (points[(i + 1) % 4] - points[i]).perp().normalize();
            let (al, ah) = extent(a, axis);
            let (bl, bh) = extent(b, axis);
            largest = largest.max((bl - ah).max(al - bh));
        }
    }
    largest
}

pub(in super::super) fn check_separation(
    children: &[contents::Content],
    bay: &ShelfBay,
    base: f32,
    top: f32,
    rear: f32,
    front: f32,
    open: bool,
) {
    let mut occupied = Vec::new();
    for (i, c) in children.iter().enumerate() {
        if c.lean == 0. {
            let q = Footprint {
                center: c.position.xz(),
                half: rotated_half(c.size.xz(), c.yaw),
                yaw: c.yaw,
            };
            assert!(q.inside(Vec2::new(bay.left, rear), Vec2::new(bay.right, front), 0.));
            assert!(!occupied.iter().any(|a| q.overlaps(*a, 0.)));
            occupied.push(q);
            continue;
        }
        assert!(children
            .iter()
            .all(|b| b.lean.signum() == c.lean.signum() && b.yaw == 0.));
        let corners = section(c);
        let bearing = if corners[0].y < corners[1].y {
            corners[0]
        } else {
            corners[1]
        };
        let center = (corners[0] + corners[2]) * 0.5;
        assert!(
            (center.x - bearing.x) * c.lean.signum() < 0.,
            "book would tip away from its support"
        );
        for v in corners {
            assert!(v.x >= bay.left && v.x <= bay.right && v.y >= base - 1e-6 && v.y < top);
        }
        assert!(
            (corners
                .into_iter()
                .map(|p| p.y)
                .fold(f32::INFINITY, f32::min)
                - base)
                .abs()
                < 2e-6
        );
        for (j, other) in children[..i].iter().enumerate() {
            let gap = signed_gap(corners, section(other));
            assert!(gap >= -2e-6, "penetration {gap}");
            if i == j + 1 {
                assert!(gap < 0.0003, "unsupported gap {gap}");
                let z_overlap = ((c.position.z + c.size.z * 0.5)
                    .min(other.position.z + other.size.z * 0.5)
                    - (c.position.z - c.size.z * 0.5).max(other.position.z - other.size.z * 0.5))
                    / c.size.z.min(other.size.z);
                assert!(z_overlap > 0.8);
            }
        }
        if i == 0 {
            if let Some(stop) = &c.bookend {
                let object = audit::fixture(ObjectKind::Bookcase, c.seed);
                let mut mesh = Assembly::default();
                stop.build(&mut mesh, &object, c);
                let (lo, hi) = mesh.bounds();
                assert!((lo.y - base).abs() < 1e-5 && lo.z >= rear && hi.z <= front);
                let gap = signed_gap(
                    corners,
                    [
                        lo.xy(),
                        Vec2::new(hi.x, lo.y),
                        hi.xy(),
                        Vec2::new(lo.x, hi.y),
                    ],
                );
                assert!(
                    (-2e-6..0.001).contains(&gap),
                    "missing bookend contact: {gap}"
                );
            } else {
                assert!(!open);
                let gap = if c.lean > 0. {
                    corners
                        .iter()
                        .map(|p| p.x - bay.left)
                        .fold(f32::INFINITY, f32::min)
                } else {
                    corners
                        .iter()
                        .map(|p| bay.right - p.x)
                        .fold(f32::INFINITY, f32::min)
                };
                assert!(
                    (0.0..0.0004).contains(&gap),
                    "missing divider contact: {gap}"
                );
            }
        } else {
            assert!(c.bookend.is_none());
        }
    }
}

#[test]
fn leaning_runs_and_shared_editions_are_distinct_repeatable_choices() {
    let (mut rows, mut left, mut right, mut ends, mut dividers, mut series, mut plain_series) =
        (0, 0, 0, 0, 0, 0, 0);
    let mut max_lean: f32 = 0.;
    for seed in 0..192 {
        let o = audit::fixture(ObjectKind::Bookcase, seed);
        let p = parameters(&o);
        for (bay, b) in p.bays.iter().enumerate() {
            for level in 0..b.levels.len() - 1 {
                let row = contents::placements(&o, &p, bay, level);
                if row.len() < 2 || !row[0].upright {
                    continue;
                }
                rows += 1;
                let first = &row[0];
                if first.lean != 0. {
                    left += usize::from(first.lean > 0.);
                    right += usize::from(first.lean < 0.);
                    ends += usize::from(first.bookend.is_some());
                    dividers += usize::from(first.bookend.is_none());
                    max_lean = max_lean.max(row.last().unwrap().lean.abs());
                }
                let matched = row
                    .iter()
                    .filter(|c| c.edition.is_some())
                    .collect::<Vec<_>>();
                if let Some(sample) = matched.first() {
                    series += 1;
                    plain_series += usize::from(first.lean == 0.);
                    assert!(
                        matched.len() >= 2
                            && matched.iter().all(|b| b.edition == sample.edition
                                && b.size.y == sample.size.y
                                && b.size.z == sample.size.z)
                    );
                }
            }
        }
    }
    println!("bookcase run audit: rows={rows}, lean_left={left}, lean_right={right}, bookends={ends}, dividers={dividers}, editions={series}, upright_editions={plain_series}, max_lean_degrees={}",max_lean.to_degrees());
    assert!(left * 10 > rows && right * 10 > rows && (left + right) * 10 < rows * 7);
    assert!(ends * 14 > rows && dividers * 14 > rows);
    assert!(series * 4 > rows && plain_series * 10 > rows && max_lean.to_degrees() > 22.);
}

#[test]
fn binding_fits_both_orientations_with_metric_uvs_and_a_bounded_mesh() {
    let mut min = usize::MAX;
    let mut max = 0;
    let mut finishes = std::collections::BTreeSet::new();
    for seed in 0..96 {
        let mut o = audit::fixture(ObjectKind::Books, seed);
        o.size = Vec3::new(0.022 + seed as f32 * 0.00023, 0.235, 0.183);
        let mut a = Assembly::default();
        binding::volume(&mut a, seed, None, o.size, Transform::IDENTITY);
        let n = audit::check(&o, &a).unwrap().triangles;
        min = min.min(n);
        max = max.max(n);
        let (lo, hi) = a.bounds();
        assert!(((hi - lo) - o.size).abs().max_element() < 1e-6);
        assert!(a.parts.keys().all(|(_, k)| part_label(k) == "books"));
        finishes.extend(a.parts.keys().map(|(s, _)| *s));
        let size = o.size;
        o.size = Vec3::new(size.y, size.x, size.z);
        let tf = Transform::from_xyz(-size.y * 0.5, size.x * 0.5, 0.)
            .with_rotation(Quat::from_rotation_z(-FRAC_PI_2));
        let mut flat = Assembly::default();
        binding::volume(&mut flat, seed, None, size, tf);
        assert_eq!(audit::check(&o, &flat).unwrap().triangles, n);
        assert!(flat.bounds().0.y.abs() < 1e-6);
    }
    println!(
        "binding audit across 96 seeds: {min}..{max} triangles, {} surfaces",
        finishes.len()
    );
    assert!(min >= 24 && max <= 180 && finishes.len() >= 5);
}
