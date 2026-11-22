//! A continuous wrapped cover, exposed page block, and optionally printed spine.
//! The print replaces faces of the jacket so close views cannot reveal floating labels.
use super::*;
use crate::scene::procedural_indoor::materials::{letter_rows, variants};

struct Frame(Transform);
impl Frame {
    fn face(&self, g: &mut Geometry, p: [Vec3; 4], normal: Vec3) {
        g.metric_quad(
            p.map(|q| self.0.transform_point(q)),
            self.0.rotation * normal,
        );
    }
    fn spine_panel(&self, g: &mut Geometry, z: f32, x: [f32; 2], y: [f32; 2]) {
        self.face(
            g,
            [
                Vec3::new(x[0], y[0], z),
                Vec3::new(x[1], y[0], z),
                Vec3::new(x[1], y[1], z),
                Vec3::new(x[0], y[1], z),
            ],
            Vec3::Z,
        );
    }
}

fn point(v: Vec2, y: f32) -> Vec3 {
    Vec3::new(v.x, y, v.y)
}

/// Work in binding-local coordinates, with a shelf contact at y=0 and the spine at +z.
pub(super) fn volume(a: &mut Assembly, seed: u64, edition: Option<u64>, s: Vec3, tf: Transform) {
    let mut r = stream(edition.unwrap_or(seed), 0x424f_4f4b_4249_4e44);
    let paperback = r.random_bool(0.37);
    let cover = if paperback {
        [Surface::Paper, Surface::Plastic][r.random_range(0..2)]
    } else {
        [Surface::FabricAlt, Surface::Leather, Surface::Plastic][r.random_range(0..3)]
    };
    let label = format!("books#finish{}", r.random_range(0..variants::COUNT));
    let gauge = if paperback {
        r.random_range(0.00045..0.00095)
    } else {
        r.random_range(0.0014..0.0030)
    };
    let gauge = f32::min(gauge, s.x * 0.12);
    let half = s.x * 0.5;
    let fore = -s.z * 0.5;
    let spine = -fore;
    let crown = if paperback {
        r.random_range(0.012..0.055)
    } else {
        r.random_range(0.06..0.24)
    } * s.x;
    let flat = r.random_range(0.38..0.67);
    let overhang = if paperback {
        r.random_range(0.00025..0.0010)
    } else {
        r.random_range(0.0016..0.0038)
    };
    let mark = r.random_bool(0.67);
    let panel_h = f32::min(half * r.random_range(0.82..1.65), s.y * 0.18);
    let center_y = s.y * r.random_range(0.13..0.83);
    let ys = [center_y - panel_h * 0.5, center_y + panel_h * 0.5];
    let xs = [-half * flat, half * flat];
    let frame = Frame(tf);
    // Two gently curved shoulders lead from the cover boards to the printable face.
    let track = |width: f32, z: f32| {
        [
            Vec2::new(-width, fore),
            Vec2::new(-width, z - crown),
            Vec2::new(-width * 0.83, z - crown * 0.28),
            Vec2::new(-width * flat, z),
            Vec2::new(width * flat, z),
            Vec2::new(width * 0.83, z - crown * 0.28),
            Vec2::new(width, z - crown),
            Vec2::new(width, fore),
        ]
    };
    let outer = track(half, spine);
    let inner = track(half - gauge, spine - gauge);
    let g = a.part(cover, &label);
    for (i, (outside, inside)) in outer.windows(2).zip(inner.windows(2)).enumerate() {
        let (v, w) = (outside[0], outside[1]);
        let (u, t) = (inside[0], inside[1]);
        let edge = w - v;
        let normal = Vec3::new(-edge.y, 0., edge.x);
        if i == 3 && mark {
            frame.spine_panel(g, spine, xs, [0., ys[0]]);
            frame.spine_panel(g, spine, xs, [ys[1], s.y]);
        } else {
            frame.face(
                g,
                [point(v, 0.), point(w, 0.), point(w, s.y), point(v, s.y)],
                normal,
            );
        }
        frame.face(
            g,
            [point(u, 0.), point(t, 0.), point(t, s.y), point(u, s.y)],
            -normal,
        );
        for (y, n) in [(0., Vec3::NEG_Y), (s.y, Vec3::Y)] {
            frame.face(g, [point(v, y), point(w, y), point(t, y), point(u, y)], n);
        }
    }
    for i in [0, outer.len() - 1] {
        frame.face(
            g,
            [
                point(outer[i], 0.),
                point(inner[i], 0.),
                point(inner[i], s.y),
                point(outer[i], s.y),
            ],
            Vec3::NEG_Z,
        );
    }

    // Inset fore-edge and head/tail reveal the cover gauge. The page block
    // stops at the inside shoulder, even on the deepest rounded case.
    let back = spine - gauge - crown;
    let front = fore + overhang;
    a.part(Surface::Paper, &label).cuboid(
        Vec3::new(s.x - 2. * gauge - 0.0002, s.y - 2. * overhang, back - front),
        0.,
        tf.with_translation(tf.transform_point(Vec3::new(0., s.y * 0.5, (back + front) * 0.5))),
    );
    if mark {
        let stock = if r.random_bool(0.52) {
            Surface::Paper
        } else {
            cover
        };
        let ink = if stock != Surface::Paper && r.random_bool(0.5) {
            Surface::Paper
        } else {
            Surface::Ink
        };
        let glyph = letter_rows(r.random_range(b'A'..=b'Z')).unwrap();
        print_mark(&frame, a.part(stock, &label), spine, xs, ys, glyph, false);
        print_mark(&frame, a.part(ink, "books"), spine, xs, ys, glyph, true);
    }
}

/// Tile the jacket and its ink edge to edge. Merge consecutive cells in each
/// row, so a letter costs tens of triangles and never needs a sticker mesh.
fn print_mark(
    frame: &Frame,
    g: &mut Geometry,
    z: f32,
    xs: [f32; 2],
    ys: [f32; 2],
    glyph: [u8; 5],
    ink: bool,
) {
    let cell = ((xs[1] - xs[0]) / 5.).min((ys[1] - ys[0]) / 7.);
    let cx = (xs[0] + xs[1]) * 0.5;
    let cy = (ys[0] + ys[1]) * 0.5;
    let x = [
        xs[0],
        cx - 1.5 * cell,
        cx - 0.5 * cell,
        cx + 0.5 * cell,
        cx + 1.5 * cell,
        xs[1],
    ];
    let bottom = cy - 2.5 * cell;
    if !ink {
        frame.spine_panel(g, z, xs, [ys[0], bottom]);
        frame.spine_panel(g, z, xs, [bottom + 5. * cell, ys[1]]);
    }
    for (row, bits) in glyph.into_iter().rev().enumerate() {
        let y = [bottom + row as f32 * cell, bottom + (row + 1) as f32 * cell];
        let cells = [false, bits & 4 != 0, bits & 2 != 0, bits & 1 != 0, false];
        let mut start = 0;
        for end in 1..=cells.len() {
            if end == cells.len() || cells[end] != cells[start] {
                if cells[start] == ink {
                    frame.spine_panel(g, z, [x[start], x[end]], y);
                }
                start = end;
            }
        }
    }
}
