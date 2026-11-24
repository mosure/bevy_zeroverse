//! A cast frame, a seated liquid bulb and a pierced, stamped deflector.
//! Build the fitting once per run, then clock and instance it at each takeoff.
use super::super::sprinkler::{SprinklerRun, BOSS_TIP_M, DEFLECTOR_GAUGE_M};
use super::*;
use crate::scene::procedural_indoor::geometry::{ProfileJoints, RevolvedOutline};
use std::f32::consts::{PI, TAU};

fn deflector(g: &mut Geometry, stock: SprinklerRun) {
    let count = usize::from(stock.slots) * 4;
    let half_slot = stock.slot_width * 0.5;
    let height = |r: f32| stock.span + stock.crown * r;
    let outline: Vec<_> = (0..stock.slots)
        .flat_map(|i| {
            [
                (half_slot * 0.65, stock.slot_root),
                (half_slot, 1.),
                (1. - half_slot, 1.),
                (1. - half_slot * 0.65, stock.slot_root),
            ]
            .map(|(u, r)| {
                let (s, c) = ((f32::from(i) + u) * TAU / f32::from(stock.slots)).sin_cos();
                Vec3::new(stock.radius * r * c, height(r), stock.radius * r * s)
            })
        })
        .collect();
    for sign in [-1., 1.] {
        let base = g.positions.len() as u32;
        let y = sign * DEFLECTOR_GAUGE_M * 0.5;
        g.positions.push([0., stock.span + y, 0.]);
        g.normals.push((Vec3::Y * sign).to_array());
        g.uvs.push([0., 0.]);
        for (i, &p) in outline.iter().enumerate() {
            let grad = p.xz().normalize() * (stock.crown / stock.radius);
            g.positions.push(p.with_y(p.y + y).to_array());
            g.normals
                .push((Vec3::new(-grad.x, 1., -grad.y).normalize() * sign).to_array());
            g.uvs.push(p.xz().to_array());
            let a = base + 1 + i as u32;
            let b = base + 1 + ((i + 1) % count) as u32;
            g.indices.extend(if sign > 0. {
                [base, b, a]
            } else {
                [base, a, b]
            });
        }
    }
    let dy = Vec3::Y * (DEFLECTOR_GAUGE_M * 0.5);
    for (i, &p) in outline.iter().enumerate() {
        let q = outline[(i + 1) % count];
        g.metric_quad([p - dy, p + dy, q + dy, q - dy], Vec3::Y.cross(q - p));
    }
}

struct HeadMesh {
    frame: Geometry,
    bulb: Geometry,
}

impl HeadMesh {
    fn new(stock: SprinklerRun) -> Self {
        let mut frame = Geometry::default();
        let mut bulb = Geometry::default();
        let r = stock.radius;
        let h = stock.span;
        // The hex boss encloses the end of the riser and seats the bulb and bow.
        RevolvedOutline::polygon(6).sweep(
            &mut frame,
            &[
                (0., -0.008),
                (r * 0.42, -0.008),
                (r * 0.50, -0.006),
                (r * 0.50, h * 0.06),
                (r * 0.32, h * 0.14),
                (0., h * 0.14),
            ],
            ProfileJoints::Creased,
            Transform::IDENTITY,
        );
        for side in [-1., 1.] {
            let across = |v: f32, t: f32| Vec3::new(side * v * r, t * h, 0.);
            frame.tapered_tube(
                &[
                    across(0.28, 0.04),
                    across(stock.bow * 0.91, 0.24),
                    across(stock.bow, 0.56),
                    across(stock.bow * 0.78, 0.69),
                    across(0.105, 0.865),
                ],
                r * 0.14,
                r * 0.09,
                6,
            );
        }
        // A common boss joins both arms, retains the trigger and carries the sheet.
        frame.lathe(
            &[
                (0., h * 0.77),
                (r * 0.16, h * 0.77),
                (r * 0.19, h * 0.87),
                (r * 0.13, h * 0.94),
                (r * 0.13, h + BOSS_TIP_M),
                (0., h + BOSS_TIP_M),
            ],
            8,
            Transform::IDENTITY,
        );
        deflector(&mut frame, stock);
        RevolvedOutline::round(8, 0., 0).sweep(
            &mut bulb,
            &[
                (r * 0.075, h * 0.11),
                (r * 0.10, h * 0.20),
                (r * 0.158, h * 0.33),
                (r * 0.15, h * 0.59),
                (r * 0.089, h * 0.72),
                (r * 0.076, h * 0.80),
            ],
            ProfileJoints::Smooth,
            Transform::IDENTITY,
        );
        Self { frame, bulb }
    }
}

/// Rigid instances can share indices, metric UVs and the manufactured profile.
fn instance(dst: &mut Geometry, src: &Geometry, pos: Vec3, rot: Mat3) {
    let base = dst.positions.len() as u32;
    dst.positions.extend(
        src.positions
            .iter()
            .map(|&p| (pos + rot * Vec3::from(p)).to_array()),
    );
    dst.normals.extend(
        src.normals
            .iter()
            .map(|&n| (rot * Vec3::from(n)).to_array()),
    );
    dst.uvs.extend_from_slice(&src.uvs);
    dst.indices.extend(src.indices.iter().map(|i| base + i));
}

/// A long drop gets a swept reducing elbow. A close-coupled outlet seats
/// directly into the end tee when there is insufficient room for a bend.
fn takeoff(a: &mut Assembly, f: &Frame, main: Vec3, end: Vec3, face: Vec3, radius: f32) {
    let horizontal = Vec3::Z * end.z.signum();
    let vertical = Vec3::Y * (face.y - end.y).signum();
    let lift = (face.y - end.y).abs();
    let bend = (radius * 2.15).min(end.z.abs() * 0.45).min(lift - 0.011);
    if bend < radius * 1.25 {
        f.rod(a, Surface::Metal, main, end, radius);
        f.rod(a, Surface::Chrome, end, face, 0.006);
        return;
    }
    let start = end - horizontal * bend;
    let riser = end + vertical * bend;
    // Overlap the socket shoulders so no two exposed end caps share a plane.
    f.rod(a, Surface::Metal, main, start + horizontal * 0.002, radius);
    f.part(a, Surface::Metal).tapered_tube(
        &[
            start - horizontal * 0.002,
            start + horizontal * (bend * 0.5) + vertical * (bend * 0.134),
            start + horizontal * (bend * 0.866) + vertical * (bend * 0.5),
            riser,
            riser + vertical * 0.002,
        ],
        radius * 1.12,
        0.0074,
        8,
    );
    f.rod(a, Surface::Chrome, riser, face, 0.006);
}

pub(super) fn branches(
    a: &mut Assembly,
    f: &Frame,
    stock: SprinklerRun,
    radius: f32,
    upright: bool,
) {
    let head = HeadMesh::new(stock);
    let flip = if upright {
        Mat3::IDENTITY
    } else {
        Mat3::from_rotation_x(PI)
    };
    let (frame_surface, frame_label) = stock.frame.key();
    let (bulb_surface, bulb_label) = stock.bulb.key();
    for i in 0..stock.count {
        let site = stock.site(f.m, i);
        let main = Vec3::new(site.x, f.height * 0.64, 0.);
        let end = main.with_z(site.z);
        let face = end.with_y(stock.outlet_y(f.height, upright));
        takeoff(a, f, main, end, face, radius * 0.57);
        f.rod(
            a,
            Surface::Metal,
            main - Vec3::X * 0.020,
            main + Vec3::X * 0.020,
            radius + 0.003,
        );
        let rot = Mat3::from_rotation_y(site.clocking) * flip;
        instance(a.part(frame_surface, frame_label), &head.frame, face, rot);
        instance(a.part(bulb_surface, bulb_label), &head.bulb, face, rot);
    }
}

#[cfg(test)]
mod tests;
