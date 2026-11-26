//! One physical cloth face shared by color and geometric annotations.
use super::*;
use crate::scene::procedural_indoor::materials::projection_cloth;

const CLASS: &str = "projection_screen";

pub(super) fn corners(lo: Vec3, hi: Vec3) -> [Vec3; 4] {
    [
        lo,
        Vec3::new(hi.x, lo.y, lo.z),
        hi,
        Vec3::new(lo.x, hi.y, lo.z),
    ]
}

pub(super) fn front(g: &mut Geometry, lo: Vec3, hi: Vec3) {
    if hi.x <= lo.x || hi.y <= lo.y {
        return;
    }
    let p = corners(lo, hi);
    g.vertex_quad(p, [Vec3::Z; 4], p.map(|v| [v.x, -v.y]));
}

fn band(g: &mut Geometry, a: [Vec3; 4], b: [Vec3; 4]) {
    for j in 0..4 {
        let k = (j + 1) % 4;
        let points = [a[j], a[k], b[k], b[j]];
        let normal = (points[1] - points[0]).cross(points[3] - points[0]);
        g.metric_quad(points, normal);
    }
}

fn cloth(a: &mut Assembly, p: &ScreenProgram, i: &Installation) {
    front(
        a.part(Surface::Paper, &projection_cloth::label(i.coating)),
        p.surface_lo,
        p.surface_hi,
    );
}

fn frame(a: &mut Assembly, o: &IndoorObject, p: &ScreenProgram, i: &Installation) {
    let z = p.sheet_front_m;
    let lo = Vec3::new(-i.housing_width_m * 0.5, p.sheet_bottom_m, z);
    let hi = Vec3::new(i.housing_width_m * 0.5, p.sheet_top_m, z);
    let edge = (p.surface_lo.x - lo.x) * 0.28;
    let rim = corners(lo, hi).map(|v| v.with_z(z + edge.min(o.size.z * 0.12)));
    let rear = rim.map(|v| v.with_z(-o.size.z * 0.5));
    let mouth = corners(
        p.surface_lo - Vec3::new(edge, edge, 0.),
        p.surface_hi + Vec3::new(edge, edge, 0.),
    )
    .map(|v| v.with_z(rim[0].z));
    let g = a.part(p.housing, CLASS);
    // The outer lip, sloped reveal and screen meet edge-to-edge.
    band(g, rear, rim);
    band(g, rim, mouth);
    band(
        a.part(Surface::Ink, projection_cloth::MASK),
        mouth,
        corners(p.surface_lo, p.surface_hi),
    );
    a.part(p.housing, CLASS).metric_quad(rear, Vec3::NEG_Z);
    cloth(a, p, i);
}

fn roll(a: &mut Assembly, p: &ScreenProgram, i: &Installation) {
    let bottom = p.sheet_bottom_m + 2. * p.bar_radius_m;
    let z = p.sheet_front_m;
    let lo = Vec3::new(p.sheet_center_x_m - p.sheet_width_m * 0.5, bottom, z);
    let hi = Vec3::new(p.sheet_center_x_m + p.sheet_width_m * 0.5, p.sheet_top_m, z);
    let g = a.part(Surface::Ink, projection_cloth::MASK);
    let back = corners(lo, hi).map(|v| v.with_z(z - 0.0008));
    band(g, back, corners(lo, hi));
    g.metric_quad(back, Vec3::NEG_Z);
    if p.surface_hi.y > p.surface_lo.y {
        front(g, lo, hi.with_y(p.surface_lo.y));
        front(g, lo.with_y(p.surface_hi.y), hi);
        front(
            g,
            lo.with_y(p.surface_lo.y),
            p.surface_hi.with_x(p.surface_lo.x),
        );
        front(
            g,
            p.surface_lo.with_x(p.surface_hi.x),
            hi.with_y(p.surface_hi.y),
        );
        cloth(a, p, i);
    } else {
        front(g, lo, hi);
    }
    hem(a, p, i);
}

pub(super) fn hem(a: &mut Assembly, p: &ScreenProgram, i: &Installation) {
    let z = p.sheet_front_m;
    a.part(p.trim, CLASS).cylinder(
        p.bar_radius_m,
        p.bar_width_m,
        Transform::from_xyz(p.sheet_center_x_m, p.sheet_bottom_m + p.bar_radius_m, z)
            .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
    );
    if i.pull_length_m > 0. {
        let x = i.pull_x_m;
        let h = i.pull_length_m;
        let y = p.sheet_bottom_m;
        let r = (h * 0.045).min(p.bar_radius_m * 0.25);
        // The two ends enter the rail. The loop has its own vertical clearance
        // even when the sheet is fully extended.
        let path = [
            (-0.18, 0.07),
            (-0.22, -0.62),
            (-0.12, -0.87),
            (0.12, -0.87),
            (0.22, -0.62),
            (0.18, 0.07),
        ]
        .map(|(dx, dy)| Vec3::new(x + h * dx, y + h * dy, z));
        a.part(Surface::Ink, projection_cloth::MASK)
            .tube(&path, r, 6);
    }
}

fn cassette(a: &mut Assembly, o: &IndoorObject, p: &ScreenProgram, i: &Installation) {
    let w = i.housing_width_m * 0.5;
    let h = p.case_height_m;
    let d = p.case_depth_m * 0.5;
    let y = p.sheet_top_m;
    let b = i.shell_bevel_m.min(h * 0.35).min(d * 0.58);
    let slot = p.sheet_front_m;
    let gap = (h * 0.075).clamp(0.0015, 0.006);
    // Section coordinates (depth, elevation). The folded return opens around the
    // real sheet. End caps close the extrusion without interior coplanar faces.
    let profile = [
        (slot - gap, 0.),
        (-d + b, 0.),
        (-d, b),
        (-d, h - b),
        (-d + b, h),
        (d - b, h),
        (d, h - b),
        (d, b),
        (d - b, 0.),
        (slot + gap, 0.),
        (slot + gap, h * 0.38),
        (slot - gap, h * 0.38),
    ];
    let point = |x, (z, v)| Vec3::new(x, y + v, z);
    let cap = p.cap_width_m;
    for (x0, x1, stock) in [
        (-w, -w + cap, p.trim),
        (-w + cap, w - cap, p.housing),
        (w - cap, w, p.trim),
    ] {
        for j in 0..profile.len() {
            let v = profile[j];
            let n = profile[(j + 1) % profile.len()];
            let (surf, label) = if j >= 9 {
                (Surface::Ink, projection_cloth::MASK)
            } else {
                (stock, CLASS)
            };
            a.part(surf, label).metric_quad(
                [point(x0, v), point(x1, v), point(x1, n), point(x0, n)],
                Vec3::new(0., n.0 - v.0, v.1 - n.1),
            );
        }
    }
    let g = a.part(p.trim, CLASS);
    // Three bands tessellate each chamfered end without a triangle fan.
    for x in [-w, w] {
        let n = Vec3::X * x.signum();
        for vs in [
            [(-d + b, 0.), (d - b, 0.), (d, b), (-d, b)],
            [(-d, b), (d, b), (d, h - b), (-d, h - b)],
            [(-d, h - b), (d, h - b), (d - b, h), (-d + b, h)],
        ] {
            g.metric_quad(vs.map(|v| point(x, v)), n);
        }
    }
    for x in p.mount_x_m {
        let rear = -o.size.z * 0.5;
        a.box_part(
            p.trim,
            CLASS,
            Vec3::new(x, y + h * 0.53, (rear - d) * 0.5),
            Vec3::new(cap.min(0.045), h * 0.53, -d - rear),
            0.001,
        );
    }
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &ScreenProgram,
    i: &Installation,
    version: u32,
) {
    if p.framed {
        frame(a, o, p, i);
    } else {
        cassette(a, o, p, i);
        if !deployment::build_if_selected(a, o, p, i, version) {
            roll(a, p, i);
        }
    }
}
