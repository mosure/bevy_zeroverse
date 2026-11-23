//! One typing layout for integrated and freestanding keyboards. Key widths are
//! in switch-pitch units; the deck fits the field in metres before meshing.
use super::*;
use crate::scene::procedural_indoor::materials::keycaps;
mod cap;
mod layout;
#[cfg(test)]
mod tests;

pub(super) fn columns(available: f32, choice: usize, split: bool) -> usize {
    if split {
        return 15;
    }
    // Leave the typing block intact on compact devices. Add a separate bank
    // only when a useful switch pitch fits across the manufactured case.
    let fits = |cols: usize| available >= units(cols) * 0.0169;
    if choice >= 15 && fits(22) {
        22
    } else if choice >= 13 && fits(19) {
        19
    } else if choice >= 12 && fits(18) {
        18
    } else {
        15
    }
}

fn units(cols: usize) -> f32 {
    match cols {
        22 => 22.70, // navigation and numeric banks
        19 => 19.35, // compact numeric bank
        18 => 18.35, // tenkeyless
        _ => 15.,
    }
}

struct Field {
    frame: Transform,
    pitch: Vec2,
    first: Vec2,
    half: Option<bool>,
    deck_y: f32,
    slope: f32,
}

struct Caps {
    gap: f32,
    taper: f32,
    dish: f32,
    height: f32,
}

impl Caps {
    fn draw(&self, g: &mut Geometry, field: &Field, key: layout::Key) {
        if field.half.is_some_and(|right| right != key.right) {
            return;
        }
        let center = field.first + field.pitch * (key.at + key.size * 0.5);
        let size = field.pitch * key.size - self.gap;
        cap::build(g, self, field, key.legend, center, size);
    }
}

pub(super) fn laptop(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &computers::LaptopProgram,
    hinge_z: f32,
    depth: f32,
) {
    let mut r = stream(o.seed, 0x4b455943415053);
    let width = o.size.x * p.keyboard_fraction;
    let available = depth * 0.57;
    let function = available > 0.089 && (p.keyboard_columns > 15 || r.random_bool(0.77));
    let rows = layout::rows(function);
    let nominal = r.random_range(0.0177..0.0198);
    let pitch = Vec2::new(
        (width / units(p.keyboard_columns as usize)).min(nominal),
        (available / rows).min(nominal),
    );
    let first = Vec2::new(
        -pitch.x * units(p.keyboard_columns as usize) * 0.5,
        hinge_z + depth * 0.35 - pitch.y * rows * 0.5,
    );
    let field = Field {
        frame: Transform::IDENTITY,
        pitch,
        first,
        half: None,
        deck_y: p.chassis_m + 0.003,
        slope: 0.,
    };
    let caps = Caps {
        gap: r.random_range(0.0012..0.0022),
        taper: r.random_range(0.00035..0.00085),
        dish: r.random_range(0.00008..0.00034),
        height: r.random_range(0.0015..0.0024),
    };
    let g = a.part(Surface::Plastic, keycaps::label(o));
    let stagger = r.random_range(-0.09..0.09);
    layout::visit(p.keyboard_columns as usize, function, stagger, |key| {
        caps.draw(g, &field, key)
    });
}

pub(super) fn desktop(a: &mut Assembly, o: &IndoorObject, p: &desktop::InputProgram) {
    let mut r = stream(o.seed, 0x4b455943415053);
    let s = o.size;
    let l = o.kind.class_name();
    let depth = s.z * if p.wrist_rest { 0.76 } else { 1. };
    let z = (depth - s.z) * 0.5;
    let deck_y = s.y * p.deck_fraction;
    let function = p.keyboard_rows == 6;
    let rows = layout::rows(function);
    let split = p.keyboard_split;
    let splay: f32 = if split {
        r.random_range(0.025..0.15)
    } else {
        0.
    };
    let caps = Caps {
        gap: r.random_range(0.0011..0.0024),
        taper: ((s.y - deck_y) * r.random_range(0.16..0.30)).min(0.0028),
        dish: ((s.y - deck_y) * r.random_range(0.07..0.16)).min(0.00135),
        height: s.y - deck_y,
    };
    let nominal = r.random_range(0.0178..0.0202);
    let bank = units(p.keyboard_columns);
    // Staggered rows split at different unit positions. The outer extents
    // of the left and right hands are 7.25 (+ stagger) and 9 units.
    let spans: &[f32] = if split {
        &[7.4, 9.]
    } else {
        std::slice::from_ref(&bank)
    };
    let total: f32 = spans.iter().sum();
    let gap = if split {
        s.x * r.random_range(0.014..0.030)
    } else {
        0.
    };
    let available = s.x - gap;
    let (sn, cs) = splay.sin_cos();
    let widest = available * spans.iter().copied().fold(0., f32::max) / total;
    let narrowest = available * spans[0] / total;
    // The entire rotated case (not just the caps) stays on its support.
    let fit = (narrowest / (narrowest * cs + depth * sn))
        .min(depth / (depth * cs + widest * sn))
        .min(1.);
    let bevel = s.y * 0.12;
    let pitch = Vec2::new(
        (available * fit * p.key_area_fraction / total)
            .min((narrowest * fit - 2. * bevel) / spans[0])
            .min(nominal),
        (depth * fit * 0.89 / rows).min(nominal),
    );
    let depth = depth * fit;
    let slope = r.random_range(0.0_f32..0.115).min(deck_y * 0.58 / depth);
    let stock = super::super::materials::input_housing::Housing::for_object(o);
    let case = stock.case_label(o);
    let mut left = -s.x * 0.5;
    for (i, &span) in spans.iter().enumerate() {
        let width = available * span / total;
        let tf = Transform::from_xyz(left + width * 0.5, 0., z)
            .with_rotation(Quat::from_rotation_y(if i == 0 { splay } else { -splay }));
        let body = a.part(stock.surface, &case);
        let start = body.positions.len();
        body.cuboid(
            Vec3::new(width * fit, deck_y, depth),
            bevel,
            Transform::from_xyz(0., deck_y * 0.5, 0.),
        );
        // Compress the case toward the typist while keeping its underside flat.
        // The inverse-transpose of this taper keeps edge highlights on the wedge.
        for (pos, normal) in body.positions[start..]
            .iter_mut()
            .zip(&mut body.normals[start..])
        {
            let v = Vec3::from_array(*pos);
            let n = Vec3::from_array(*normal);
            let scale = 1. - slope * (v.z + depth * 0.5) / deck_y;
            *pos = tf.transform_point(v.with_y(v.y * scale)).to_array();
            let ny = n.y / scale;
            *normal = (tf.rotation
                * Vec3::new(n.x, ny, n.z + slope * v.y / deck_y * ny).normalize())
            .to_array();
        }
        let first_unit = if i == 0 { 0. } else { 15. - span };
        let field = Field {
            frame: tf,
            pitch,
            first: -pitch * Vec2::new(first_unit + span * 0.5, rows * 0.5),
            half: split.then_some(i == 1),
            deck_y: deck_y - slope * depth * 0.5,
            slope,
        };
        let g = a.part(Surface::Plastic, keycaps::label(o));
        layout::visit(p.keyboard_columns, function, p.row_stagger, |key| {
            if key.legend == b' ' && split {
                for (offset, width, right) in [(0., 3.0, false), (3.0, 3.25, true)] {
                    caps.draw(
                        g,
                        &field,
                        layout::Key {
                            at: key.at + Vec2::X * offset,
                            size: key.size.with_x(width),
                            right,
                            ..key
                        },
                    );
                }
            } else {
                caps.draw(g, &field, key);
            }
        });
        left += width + gap;
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
}
