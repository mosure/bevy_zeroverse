//! The joint seal stays on the fixed casing. Latch hardware moves with the sash.
use super::*;

pub(super) fn seals(
    a: &mut Assembly,
    tf: Transform,
    pane: &PaneInstallation,
    lo: Vec2,
    hi: Vec2,
    z: f32,
) {
    let min = pane.min;
    let max = pane.max;
    let width = (lo.x - min.x) + 0.002;
    let center = (min + max) * 0.5;
    let build = a.part(Surface::Rubber, "window");
    let mut strip = |p: Vec2, size: Vec2| {
        build.cuboid(
            size.extend(0.003),
            0.,
            Transform::from_translation(tf.transform_point(p.extend(z))).with_rotation(tf.rotation),
        )
    };
    for x in [min.x + width * 0.5, max.x - width * 0.5] {
        strip(Vec2::new(x, center.y), Vec2::new(width, max.y - min.y));
    }
    for y in [min.y + width * 0.5, max.y - width * 0.5] {
        strip(
            Vec2::new(center.x, y),
            Vec2::new(hi.x - lo.x - 0.004, width),
        );
    }
}

pub(super) fn handle(
    stock: &SashStock,
    pane: &PaneInstallation,
    center: Vec2,
    rail: f32,
    face: f32,
    block: &mut impl FnMut(Surface, &str, Vec3, Vec3, f32),
) {
    let vertical = matches!(
        pane.hinge,
        SashHinge::LeftCasement | SashHinge::RightCasement
    );
    let axis = if vertical {
        -Vec2::Y
    } else if center.x < (pane.min.x + pane.max.x) * 0.5 {
        Vec2::X
    } else {
        -Vec2::X
    };
    let size = |across, along, depth| {
        if vertical {
            Vec3::new(across, along, depth)
        } else {
            Vec3::new(along, across, depth)
        }
    };
    let mut part =
        |p: Vec2, z, dims, bevel| block(stock.finish, "window#hardware", p.extend(z), dims, bevel);
    let length = stock.lever_length_m;
    let projection = stock.grip_projection_m;
    // The plate embeds 0.6 mm into the actual rail. The neck intersects the
    // plate and grip internally, leaving a real finger space behind the lever.
    part(
        center,
        face + 0.0019,
        size(rail * 0.68, length * 0.58, 0.005),
        0.0015,
    );
    part(
        center,
        face + projection * 0.5,
        size(rail * 0.44, 0.016, projection),
        0.001,
    );
    part(
        center + axis * (length * 0.5 - 0.009),
        face + projection,
        size((rail * 0.52).min(0.017), length, 0.012),
        0.004,
    );
}

/// Two rebate hinges connect the leaf to its casing even when the glass is open.
/// The shoe occupies the fit joint, the pin stays on the axis and the tongue
/// moves with the sash. No hinge volume is needed on a direct-glazed fixed light.
pub(super) fn hinges(
    a: &mut Assembly,
    wall: Transform,
    recess: f32,
    leaf: Transform,
    pane: &PaneInstallation,
    [back, rail_back, face]: [f32; 3],
) {
    let (lo, hi) = pane.bounds(true);
    let (start, end, into, pin_z) = match pane.hinge {
        SashHinge::LeftCasement => (lo, Vec2::new(lo.x, hi.y), Vec2::X, face),
        SashHinge::RightCasement => (Vec2::new(hi.x, lo.y), hi, -Vec2::X, face),
        SashHinge::TopAwning => (Vec2::new(lo.x, hi.y), hi, -Vec2::Y, back),
        SashHinge::BottomHopper => (lo, Vec2::new(hi.x, lo.y), Vec2::Y, face),
        SashHinge::Fixed => return,
    };
    let along = (end - start).normalize();
    let length = (0.043 + 0.016 * start.distance(end)).min(0.075);
    let gap = lo.x - pane.min.x;
    let radius = gap * 0.88;
    let metal = a.part(Surface::Metal, "window#hinge");
    let stationary = wall * Transform::from_xyz(0., 0., -recess);
    let moving = wall * leaf;
    for t in [0.18, 0.82] {
        let center = start.lerp(end, t);
        let shoe = (center - into * gap * 0.5).extend((back + face) * 0.5);
        metal.cuboid(
            (into.abs() * gap + along * length * 0.86).extend(face - back),
            0.,
            stationary * Transform::from_translation(shoe),
        );
        metal.lathe(
            &[
                (0., -length * 0.5),
                (radius, -length * 0.5),
                (radius, length * 0.5),
                (0., length * 0.5),
            ],
            8,
            stationary
                * Transform::from_translation(center.extend(pin_z))
                    .with_rotation(Quat::from_rotation_arc(Vec3::Y, along.extend(0.))),
        );
        // A thick glazing unit can project behind a shallow rail. The folded
        // tongue reaches from the exterior pin back into the actual metal.
        let rail_z = if pane.hinge == SashHinge::TopAwning {
            rail_back
        } else {
            face
        };
        let tongue = (center + into * 0.006).extend((pin_z + rail_z) * 0.5);
        metal.cuboid(
            (into.abs() * 0.014 + along * length * 0.52).extend((rail_z - pin_z).abs() + 0.003),
            0.,
            moving * Transform::from_translation(tongue),
        );
    }
}
