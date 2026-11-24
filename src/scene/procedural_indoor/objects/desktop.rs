//! Construction programs for input devices. Keys, controls and housings are
//! physical separate parts; construction is independent of the finish seed.
use super::*;
pub mod pointer;
pub(crate) mod ports;
pub(crate) mod tower;

#[derive(Debug, Clone, serde::Serialize)]
pub struct InputProgram {
    pub keyboard_split: bool,
    pub keyboard_rows: usize,
    pub keyboard_columns: usize,
    pub deck_fraction: f32,
    pub key_area_fraction: f32,
    pub row_stagger: f32,
    pub wrist_rest: bool,
    pub mouse_roundness: f32,
    pub mouse_crown: f32,
    pub mouse_waist: f32,
    pub button_fraction: f32,
    pub trackball: bool,
}
pub fn parameters(o: &IndoorObject) -> InputProgram {
    let mut r = stream(o.seed, 7400);
    let mut p = InputProgram {
        keyboard_split: r.random_bool(0.28),
        keyboard_rows: r.random_range(4..=6),
        keyboard_columns: r.random_range(10..=17),
        deck_fraction: r.random_range(0.35..0.67),
        key_area_fraction: r.random_range(0.88..0.97),
        row_stagger: r.random_range(-0.09..0.09),
        wrist_rest: r.random_bool(0.3),
        mouse_roundness: r.random_range(2.0..4.5),
        mouse_crown: r.random_range(2.0..4.0),
        mouse_waist: r.random_range(0.78..0.99),
        button_fraction: r.random_range(0.27..0.47),
        trackball: r.random_bool(0.18),
    };
    // Low-profile scissor decks expose only a few millimetres of key. Tall
    // cases put their additional height below the switch plate.
    let cap_limit = if p.deck_fraction > 0.52 {
        0.0024 + (0.67 - p.deck_fraction) * 0.028
    } else {
        0.007 + (0.52 - p.deck_fraction) * 0.030
    };
    p.deck_fraction = 1. - (o.size.y * (1. - p.deck_fraction)).min(cap_limit) / o.size.y.max(0.001);
    p.keyboard_rows = if p.keyboard_rows == 4 { 5 } else { 6 };
    p.keyboard_columns = super::keyboard::columns(
        o.size.x * p.key_area_fraction,
        p.keyboard_columns,
        p.keyboard_split,
    );
    p
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    if o.kind == ObjectKind::PcTower {
        tower::build(a, o, version);
        return;
    }
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    if o.kind == ObjectKind::Keyboard {
        super::keyboard::desktop(a, o, &p);
    } else if version >= pointer::INTRODUCED {
        pointer::build(a, o, &p);
        return;
    } else {
        let stock = super::super::materials::input_housing::Housing::for_object(o);
        let shell = stock.case_label(o);
        a.part(stock.surface, &shell).cushion(
            s * Vec3::new(p.mouse_waist, 0.87, 0.98),
            super::super::geometry::CushionProfile {
                roundness: p.mouse_roundness,
                crown: p.mouse_crown,
                dish: 0.,
            },
            Transform::from_xyz(0., s.y * 0.475, 0.),
        );
        // The bottom plate intersects the shell internally, giving broad desk contact.
        a.part(Surface::Rubber, l).housing(
            &[
                (0., Vec2::new(s.x * p.mouse_waist * 0.28, s.z * 0.29)),
                (
                    s.y * 0.12,
                    Vec2::new(s.x * p.mouse_waist * 0.32, s.z * 0.33),
                ),
            ],
            p.mouse_roundness,
            Transform::IDENTITY,
        );
        if p.trackball {
            a.part(Surface::Plastic, l).ellipsoid(
                Vec3::new(s.x * 0.22, s.y * 0.20, s.z * 0.15),
                Transform::from_xyz(0., s.y * 0.79, -s.z * 0.08),
            );
        } else {
            for side in [-1., 1.] {
                button(a.part(stock.surface, &shell), s, &p, side);
            }
            let z = -s.z * (0.45 - p.button_fraction * 0.52);
            a.part(Surface::Rubber, l).cylinder(
                (s.y * 0.17).min(0.009),
                (s.x * 0.105).min(0.009),
                Transform::from_xyz(0., roof(s, &p, 0., z) - (s.y * 0.105).min(0.006), z)
                    .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            );
        }
    }
    ports::input_socket(a, o, &p);
}

// Same analytic superellipsoid as the housing. Button caps follow its crown
// rather than floating as independent cushions above it.
fn roof(s: Vec3, p: &InputProgram, x: f32, z: f32) -> f32 {
    let radial = ((x / (s.x * p.mouse_waist * 0.5))
        .abs()
        .powf(p.mouse_roundness)
        + (z / (s.z * 0.49)).abs().powf(p.mouse_roundness))
    .powf(1. / p.mouse_roundness);
    s.y * (0.475
        + 0.435
            * (1. - radial.powf(p.mouse_crown))
                .max(0.)
                .powf(1. / p.mouse_crown))
}
fn button(g: &mut Geometry, s: Vec3, p: &InputProgram, side: f32) {
    const COLS: usize = 5;
    const ROWS: usize = 9;
    let offset = g.positions.len() as u32;
    for row in 0..ROWS {
        let z = -s.z * (0.45 - p.button_fraction * row as f32 / (ROWS - 1) as f32);
        let half = s.x
            * p.mouse_waist
            * 0.5
            * (1. - (z / (s.z * 0.49)).abs().powf(p.mouse_roundness)).powf(1. / p.mouse_roundness);
        for col in 0..COLS {
            let x =
                side * (s.x * 0.025 + (half * 0.80 - s.x * 0.025) * col as f32 / (COLS - 1) as f32);
            let e = 0.00001;
            let n = Vec3::new(
                -(roof(s, p, x + e, z) - roof(s, p, x - e, z)) / (2. * e),
                1.,
                -(roof(s, p, x, z + e) - roof(s, p, x, z - e)) / (2. * e),
            )
            .normalize();
            g.positions.push([x, roof(s, p, x, z) + 0.0008, z]);
            g.normals.push(n.to_array());
            g.uvs.push([x, z]);
        }
    }
    for row in 0..ROWS - 1 {
        for col in 0..COLS - 1 {
            let i = offset + (row * COLS + col) as u32;
            for mut tri in [
                [i, i + 1, i + 1 + COLS as u32],
                [i, i + 1 + COLS as u32, i + COLS as u32],
            ] {
                if side > 0. {
                    tri.swap(1, 2);
                }
                g.indices.extend(tri);
            }
        }
    }
}
