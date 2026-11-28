use super::*;
use crate::scene::procedural_indoor::geometry::housing_outline;

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = Cooler::for_object(o);
    let s = o.size;
    let deck = p.body_height_m;
    let tray = p.tray_height_m;
    let foot = 0.024 * deck;
    let body = match p.body {
        Surface::Metal => STEEL,
        Surface::Plastic => "appliance#finish301",
        _ => CLASS,
    };
    let tf = Transform::IDENTITY;
    let half = s.xz() * 0.483;
    for x in [-0.37 * s.x, 0.37 * s.x] {
        for z in [-0.36 * s.z, 0.36 * s.z] {
            a.part(Surface::Rubber, CLASS).cylinder(
                0.019,
                foot,
                Transform::from_xyz(x, foot * 0.5, z),
            );
        }
    }
    let cabinet_half = if p.loading == Loading::BottomCabinet {
        Vec2::new(half.x, half.y * 0.968)
    } else {
        half
    };
    a.part(p.body, body).housing(
        &[
            (foot, cabinet_half * 0.92),
            (foot + 0.016, cabinet_half * 0.97),
            (tray - 0.022, cabinet_half),
            (tray - 0.012, cabinet_half * 0.985),
        ],
        p.roundedness,
        tf,
    );
    if p.loading == Loading::BottomCabinet {
        // The removable front stands proud of a recessed plinth, with a shadow gap
        // down its perimeter. A finger pull bridges its two fixing bosses.
        let face_z = s.z * 0.473;
        let top = tray - 0.044;
        let bottom = foot + 0.029;
        a.box_part(
            p.body,
            body,
            Vec3::new(0., (top + bottom) * 0.5, face_z - 0.0045),
            Vec3::new(s.x * 0.805, top - bottom, 0.010),
            0.004,
        );
        let pull = if o.seed & 2 == 0 { -0.252 } else { 0.252 } * s.x;
        for x in [pull - 0.021, pull + 0.021] {
            a.box_part(
                Surface::Ink,
                CLASS,
                Vec3::new(x, top - 0.035, face_z + 0.0025),
                Vec3::new(0.007, 0.008, 0.005),
                0.001,
            );
        }
        a.box_part(
            Surface::Ink,
            CLASS,
            Vec3::new(pull, top - 0.035, face_z + 0.005),
            Vec3::new(0.049, 0.009, 0.004),
            0.001,
        );
    }
    let nozzle_len = 0.029 * deck;
    let control_drop = if p.buttons { nozzle_len } else { 0.0585 * deck };
    let underside = tray + p.cup_clearance_m + control_drop;
    let bay_center = (tray - 0.015 + underside) * 0.5;
    a.box_part(
        Surface::Ink,
        CLASS,
        Vec3::new(0., bay_center, -s.z * 0.246),
        Vec3::new(s.x * 0.88, underside - tray + 0.017, s.z * 0.42),
        0.009,
    );
    for sign in [-1., 1.] {
        a.box_part(
            p.body,
            body,
            Vec3::new(sign * s.x * 0.451, bay_center, -s.z * 0.171),
            Vec3::new(0.048 * s.x, underside - tray + 0.016, s.z * 0.474),
            0.004,
        );
    }
    for z in [0.023 * s.z, 0.396 * s.z] {
        a.box_part(
            Surface::Ink,
            CLASS,
            Vec3::new(0., tray - 0.009, z),
            Vec3::new(s.x * 0.78, 0.008, 0.009),
            0.0015,
        );
    }
    let bar_pitch = s.x * 0.735 / (p.tray_bars - 1) as f32;
    for i in 0..p.tray_bars {
        a.box_part(
            Surface::Metal,
            STEEL,
            Vec3::new(
                i as f32 * bar_pitch - s.x * 0.3675,
                tray - 0.0025,
                s.z * 0.210,
            ),
            Vec3::new(bar_pitch * 0.47, 0.005, s.z * 0.395),
            0.0015,
        );
    }
    let receiver_depth = deck * 0.046;
    let top = if p.loading == Loading::BottomCabinet {
        deck
    } else {
        deck - receiver_depth
    };
    a.part(p.body, body).housing(
        &[
            (underside, half * 0.94),
            (underside + 0.008, half),
            (top - 0.006, half),
            (top, half * 0.982),
        ],
        p.roundedness,
        tf,
    );
    let pitch = s.x * if p.taps == 3 { 0.253 } else { 0.32 };
    for i in 0..p.taps {
        let x = (i as f32 - (p.taps - 1) as f32 * 0.5) * pitch;
        let g = a.part(Surface::Chrome, CLASS);
        g.lathe(
            &[
                (0.0063, underside - 0.008),
                (0.0063, underside - nozzle_len),
                (0.0105, underside - nozzle_len),
                (0.015, underside - 0.004),
                (0.015, underside + 0.001),
            ],
            16,
            Transform::from_xyz(x, 0., p.outlet_z_m),
        );
        // Red child-lock tab on the hot control. The cold and ambient taps
        // retain a neutral switch and a distinct physical control position.
        let label = if i == 0 && p.taps > 1 {
            "appliance#finish300"
        } else {
            "appliance#finish303"
        };
        if p.buttons {
            let front =
                -crate::scene::procedural_indoor::geometry::housing_front_z(half, p.roundedness, x);
            a.box_part(
                Surface::Plastic,
                label,
                Vec3::new(x, underside + (top - underside) * 0.51, front - 0.001),
                Vec3::new(pitch * 0.43, 0.019, 0.007),
                0.003,
            );
        } else {
            a.box_part(
                Surface::Plastic,
                label,
                Vec3::new(x, underside - 0.018 * deck, p.outlet_z_m - 0.017),
                Vec3::new(0.012, 0.04 * deck, 0.010),
                0.003,
            );
            a.box_part(
                Surface::Plastic,
                label,
                Vec3::new(x, underside - 0.045 * deck, p.outlet_z_m - 0.021),
                Vec3::new(0.032, 0.027 * deck, 0.012),
                0.004,
            );
        }
    }
    if p.loading != Loading::BottomCabinet {
        let bore = p
            .bottle
            .as_ref()
            .map_or(0.035, |b| b.radius_m * 0.25 + 0.003);
        // A continuous annular deck ends at the open receiver. The solid top
        // of the lower head is the bottom of that well, below the bottle neck.
        let lip = bore * 1.88;
        let shoulder = deck - 0.011 * deck;
        let outer = housing_outline(p.roundedness);
        let round = housing_outline(2.);
        let point = |v: Vec2, y| Vec3::new(v.x, y, v.y);
        let g = a.part(p.body, body);
        for i in 0..outer.len() {
            let j = (i + 1) % outer.len();
            let (u, v) = (outer[i] * half * 0.982, outer[j] * half * 0.982);
            let (q, r) = (round[i] * lip, round[j] * lip);
            g.metric_quad(
                [
                    point(u, top),
                    point(v, top),
                    point(v, shoulder),
                    point(u, shoulder),
                ],
                point((u + v) * 0.5, 0.),
            );
            g.metric_quad(
                [
                    point(u, shoulder),
                    point(v, shoulder),
                    point(r, shoulder),
                    point(q, shoulder),
                ],
                Vec3::Y,
            );
        }
        a.part(Surface::Rubber, CLASS).lathe(
            &[
                (lip, shoulder),
                (lip - 0.004, shoulder + 0.003),
                (bore + 0.006, deck),
                (bore, deck),
                (bore - 0.002, deck - 0.003),
                (bore - 0.003, top),
            ],
            32,
            tf,
        );
        a.part(Surface::Ink, CLASS).lathe(
            &[
                (0.008, top),
                (0.008, top + 0.014),
                (0.004, top + 0.020),
                (0., top + 0.020),
            ],
            16,
            tf,
        );
        if let Some(bottle) = p.bottle {
            // The removable cap grips the neck bead. Its flange bears on
            // the receiver; the central bore remains open around the spike.
            let cap_top = deck - bottle.insertion_m + bottle.height_m * 0.12;
            a.part(
                Surface::Plastic,
                crate::scene::procedural_indoor::materials::glass::carboy::CAP,
            )
            .lathe(
                &[
                    (bore - 0.004, deck - 0.011),
                    (bore - 0.004, deck - 0.001),
                    (bore + 0.004, deck - 0.0007),
                    (bore + 0.004, deck + 0.003),
                    (bore - 0.0015, deck + 0.005),
                    (bottle.radius_m * 0.207 + 0.003, cap_top - 0.002),
                    (bottle.radius_m * 0.196, cap_top),
                ],
                32,
                tf,
            );
            bottle.build(a, deck);
        }
    }
}
