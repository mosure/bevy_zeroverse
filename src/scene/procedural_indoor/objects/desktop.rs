//! Construction programs for input devices. Keys, controls and housings are
//! physical separate parts; construction is independent of the finish seed.
use super::*;

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
    InputProgram {
        keyboard_split: r.random_bool(0.28),
        keyboard_rows: r.random_range(4..=6),
        keyboard_columns: r.random_range(10..=17),
        deck_fraction: r.random_range(0.35..0.67),
        key_area_fraction: r.random_range(0.68..0.91),
        row_stagger: r.random_range(0.08..0.40),
        wrist_rest: r.random_bool(0.3),
        mouse_roundness: r.random_range(2.0..4.5),
        mouse_crown: r.random_range(2.0..4.0),
        mouse_waist: r.random_range(0.78..0.99),
        button_fraction: r.random_range(0.27..0.47),
        trackball: r.random_bool(0.18),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let l = o.kind.class_name();
    if o.kind == ObjectKind::Keyboard {
        let groups = if p.keyboard_split { 2 } else { 1 };
        let width = s.x / groups as f32 * if groups == 2 { 0.94 } else { 1. };
        let depth = s.z * if p.wrist_rest { 0.76 } else { 1. };
        let z = if p.wrist_rest { -s.z * 0.12 } else { 0. };
        for half in 0..groups {
            let x = s.x * ((half as f32 + 0.5) / groups as f32 - 0.5);
            a.box_part(
                Surface::Metal,
                l,
                Vec3::new(x, s.y * p.deck_fraction * 0.5, z),
                Vec3::new(width, s.y * p.deck_fraction, depth),
                s.y * 0.12,
            );
            let cols = p.keyboard_columns / groups;
            let pitch = width * p.key_area_fraction / cols as f32;
            let row_pitch = depth * 0.81 / p.keyboard_rows as f32;
            for row in 0..p.keyboard_rows {
                for col in 0..cols {
                    // Space bars occupy three key positions, without overlapping keys.
                    let space = row == p.keyboard_rows - 1 && cols >= 6 && col == 2;
                    if row == p.keyboard_rows - 1 && cols >= 6 && (3..=4).contains(&col) {
                        continue;
                    }
                    let key_w = pitch * if space { 2.85 } else { 0.83 };
                    let xx = x
                        + (col as f32 - (cols - 1) as f32 * 0.5 + if space { 1. } else { 0. })
                            * pitch
                        + if row % 2 == 1 {
                            pitch * p.row_stagger * 0.4
                        } else {
                            -pitch * p.row_stagger * 0.4
                        };
                    let zz = z + (row as f32 - (p.keyboard_rows - 1) as f32 * 0.5) * row_pitch;
                    a.box_part(
                        Surface::Plastic,
                        l,
                        Vec3::new(xx, s.y * (1. + p.deck_fraction) * 0.5, zz),
                        Vec3::new(key_w, s.y * (1. - p.deck_fraction), row_pitch * 0.83),
                        s.y * 0.05,
                    );
                }
            }
        }
        if p.wrist_rest {
            a.box_part(
                Surface::Rubber,
                l,
                Vec3::new(0., s.y * 0.18, s.z * 0.38),
                Vec3::new(s.x * 0.93, s.y * 0.36, s.z * 0.22),
                s.y * 0.15,
            );
        }
    } else {
        a.part(Surface::Plastic, l).cushion(
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
            a.part(Surface::Chrome, l).ellipsoid(
                Vec3::new(s.x * 0.22, s.y * 0.20, s.z * 0.15),
                Transform::from_xyz(0., s.y * 0.79, -s.z * 0.08),
            );
        } else {
            for side in [-1., 1.] {
                button(a.part(Surface::Plastic, l), s, &p, side);
            }
            let z = -s.z * (0.45 - p.button_fraction * 0.52);
            a.part(Surface::Rubber, l).cylinder(
                s.y * 0.065,
                s.x * 0.095,
                Transform::from_xyz(0., roof(s, &p, 0., z), z)
                    .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
            );
        }
    }
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
