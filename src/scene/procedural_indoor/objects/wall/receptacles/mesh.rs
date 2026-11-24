//! Front faces are tiled around actual openings. Each pocket has an interior
//! wall and a bottom, so oblique color, depth and normals see the same relief.
use super::*;

use super::super::super::electrical::{edge, insert, pierced, Contact, Rect};

pub(super) fn build(b: &OutletBank, a: &mut Assembly, label: &str) {
    let s = b.case_size;
    let center = Vec2::Y * (s.y * 0.5);
    let shell = Rect::new(center, s.xy());
    let face = Rect::new(center, s.xy() - Vec2::splat(2. * b.edge_m));
    let rear = -0.35 * s.z;
    let plate = b.face_z_m - 0.0012;
    let g = a.part(b.finish, label);
    g.metric_quad(shell.corners().map(|p| p.extend(rear)), -Vec3::Z);
    edge(g, shell, rear, shell, plate - b.edge_m, false);
    edge(g, shell, plate - b.edge_m, face, plate, false);
    let bodies: Vec<_> = (0..b.plate_slots())
        .map(|i| Rect::new(b.slot_center(i), b.module_size))
        .collect();
    pierced(g, face, plate, &bodies);
    for (slot_index, body) in bodies.into_iter().enumerate() {
        if b.blank_slot == Some(slot_index) {
            let front = plate + 0.00075;
            let g = a.part(b.insert, label);
            edge(g, body, plate, body, front, false);
            body.face(g, front);
            continue;
        }
        let gang = slot_index - usize::from(b.blank_slot.is_some_and(|blank| slot_index > blank));
        edge(
            a.part(b.insert, label),
            body,
            plate,
            body,
            b.face_z_m,
            false,
        );
        let k = b.contact_scale;
        let contact = Contact {
            face: b.face,
            contact_scale: k,
            face_z_m: b.face_z_m,
            well_depth_m: b.well_depth_m,
            insert: b.insert,
            pin_sides: 24,
            well_sides: 24,
        };
        let mut centers = [Vec2::ZERO; 2];
        let centers = &mut centers[..b.rows];
        for (row, c) in centers.iter_mut().enumerate() {
            *c = b.socket_local(gang * b.rows + row).unwrap().truncate();
        }
        let accessory_y = if b.rows == 2 {
            b.gang_center(gang).y
        } else {
            body.hi.y - b.module_size.y * 0.13
        };
        let usb = b.usb.then(|| {
            let x = b.gang_center(gang).x
                - if b.switched {
                    b.module_size.x * 0.18
                } else {
                    0.
                };
            Rect::new(Vec2::new(x, accessory_y), Vec2::new(0.009, 0.0033) * k)
        });
        insert(a, &contact, label, body, centers, usb);
        if b.switched {
            let x = b.gang_center(gang).x + if b.usb { b.module_size.x * 0.21 } else { 0. };
            a.part(Surface::Plastic, label).cuboid(
                Vec3::new(0.009 * k, 0.008 * k, 0.0019),
                0.0006,
                Transform::from_xyz(x, accessory_y, b.face_z_m + 0.00045)
                    .with_rotation(Quat::from_rotation_x(0.09)),
            );
        }
    }
    if b.screws {
        for sign in [-1., 1.] {
            // Centre of the exposed margin, clear of both insert and bevel.
            let y = center.y + sign * (face.hi.y - center.y + b.module_size.y * 0.5) * 0.5;
            let r = (s.y * 0.019).min((face.hi.y - center.y - b.module_size.y * 0.5) * 0.31);
            let tf =
                Transform::from_xyz(0., y, plate).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
            a.part(Surface::Metal, label).lathe(
                &[(r, 0.), (r * 0.9, 0.00055), (0., 0.00055)],
                12,
                tf,
            );
        }
    }
}
