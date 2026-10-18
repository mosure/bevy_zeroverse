//! Raised ceiling fields and supply/return terminals. Openings are geometry,
//! with separated backing, trim and louvers instead of overlapping face decals.
use super::*;

pub(super) fn coffers(
    a: &mut Assembly,
    f: &Frame,
    aspect: f32,
    beam: f32,
    recess: f32,
    timber: bool,
) -> Vec<Vec3> {
    let nx = (f.length / (f.m.pitch_m * aspect)).round().clamp(1., 5.) as usize;
    let nz = (f.width / f.m.pitch_m).round().clamp(1., 4.) as usize;
    let dx = (f.length - 0.02) / nx as f32;
    let dz = (f.width - 0.02) / nz as f32;
    let bw = dx.min(dz) * beam;
    let top = f.height * 0.92;
    let low = top - f.height * recess;
    let surface = if timber {
        Surface::Wood
    } else {
        Surface::Paint
    };
    for z in 0..=nz {
        let edge = (z as f32 - nz as f32 * 0.5) * dz;
        // Outer rails stay inside the module instead of extending half a beam.
        let z = edge
            + if z == 0 {
                bw * 0.5
            } else if z == nz {
                -bw * 0.5
            } else {
                0.
            };
        f.block(
            a,
            surface,
            Vec3::new(0., (top + low) * 0.5, z),
            Vec3::new(f.length - 0.02, top - low, bw),
            0.001,
        );
    }
    for iz in 0..nz {
        let cz = (iz as f32 + 0.5 - nz as f32 * 0.5) * dz;
        let zlo = cz - dz * 0.5 + if iz == 0 { bw } else { bw * 0.5 };
        let zhi = cz + dz * 0.5 - if iz + 1 == nz { bw } else { bw * 0.5 };
        for ix in 0..=nx {
            let x = (ix as f32 - nx as f32 * 0.5) * dx
                + if ix == 0 {
                    bw * 0.5
                } else if ix == nx {
                    -bw * 0.5
                } else {
                    0.
                };
            f.block(
                a,
                surface,
                Vec3::new(x, (top + low) * 0.5, (zlo + zhi) * 0.5),
                Vec3::new(bw, top - low, zhi - zlo),
                0.001,
            );
        }
        for ix in 0..nx {
            let cx = (ix as f32 + 0.5 - nx as f32 * 0.5) * dx;
            let xlo = cx - dx * 0.5 + if ix == 0 { bw } else { bw * 0.5 };
            let xhi = cx + dx * 0.5 - if ix + 1 == nx { bw } else { bw * 0.5 };
            // Higher backing and dark reveals make the field visibly recessed.
            let c = Vec3::new((xlo + xhi) * 0.5, top - 0.003, (zlo + zhi) * 0.5);
            f.block(
                a,
                Surface::Ink,
                c,
                Vec3::new(xhi - xlo, 0.006, zhi - zlo),
                0.,
            );
            f.block(
                a,
                Surface::Ceiling,
                c - Vec3::Y * 0.004,
                Vec3::new(xhi - xlo - 0.008, 0.002, zhi - zlo - 0.008),
                0.,
            );
        }
    }
    [-1., 1.]
        .map(|sign| Vec3::new(sign * (f.length * 0.5 - 0.01 - bw * 0.5), top, 0.))
        .to_vec()
}

pub(super) fn diffuser(
    a: &mut Assembly,
    f: &Frame,
    round: bool,
    blades: u8,
    angle: f32,
) -> Vec<Vec3> {
    let low = 0.006;
    let top = f.height * 0.76;
    if round {
        let r = f.width.min(f.length) * 0.46;
        f.part(a, Surface::Metal).lathe(
            &[(r * 0.80, low), (r, low), (r, top), (0., top)],
            32,
            Transform::IDENTITY,
        );
        f.part(a, Surface::Ink)
            .cylinder(r * 0.82, 0.003, Transform::from_xyz(0., top - 0.005, 0.));
        for i in 1..=blades {
            let radius = r * 0.79 * i as f32 / blades as f32;
            let width = r * 0.65 / blades as f32;
            let y = low + 0.002 + (top - low) * 0.30 * (1. - i as f32 / blades as f32);
            let dy = (width * angle.tan()).min((top - low) * 0.22);
            f.part(a, Surface::Paint).lathe(
                &[
                    (radius - width, y + dy),
                    (radius, y),
                    (radius, y + 0.002),
                    (radius - width, y + dy + 0.002),
                    (radius - width, y + dy),
                ],
                32,
                Transform::IDENTITY,
            );
        }
        f.part(a, Surface::Paint).cylinder(
            r * 0.79 / blades as f32 * 0.25,
            0.006,
            Transform::from_xyz(0., low + (top - low) * 0.28, 0.),
        );
    } else {
        let x = f.length * 0.47;
        let z = f.width * 0.47;
        let border = x.min(z) * 0.10;
        for sign in [-1., 1.] {
            f.block(
                a,
                Surface::Metal,
                Vec3::new(sign * (x - border * 0.5), (top + low) * 0.5, 0.),
                Vec3::new(border, top - low, z * 2.),
                0.001,
            );
            f.block(
                a,
                Surface::Metal,
                Vec3::new(0., (top + low) * 0.5, sign * (z - border * 0.5)),
                Vec3::new((x - border) * 2., top - low, border),
                0.001,
            );
        }
        f.block(
            a,
            Surface::Ink,
            Vec3::Y * (top - 0.003),
            Vec3::new((x - border) * 2., 0.004, (z - border) * 2.),
            0.,
        );
        let pitch = (x - border) * 2. / blades as f32;
        let width = pitch * 0.62;
        let y = low + width * angle.sin() * 0.5 + 0.003;
        // Fit tilted louvers to the available plenum even in shallow modules.
        let angle = angle.min(((top - low) * 0.70 / width).min(0.99).asin());
        for i in 0..blades {
            f.part(a, Surface::Paint).cuboid(
                Vec3::new(width, 0.003, (z - border) * 2.),
                0.,
                Transform::from_xyz(
                    (i as f32 + 0.5 - blades as f32 * 0.5) * pitch,
                    y.min(top * 0.5),
                    0.,
                )
                .with_rotation(Quat::from_rotation_z(angle)),
            );
        }
    }
    // A closed supply plenum and neck are part of the reserved body.
    f.block(
        a,
        Surface::Metal,
        Vec3::Y * (top + 0.008),
        Vec3::new(f.length * 0.62, 0.016, f.width * 0.62),
        0.002,
    );
    vec![
        Vec3::new(-f.length * 0.25, top + 0.016, 0.),
        Vec3::new(f.length * 0.25, top + 0.016, 0.),
    ]
}
