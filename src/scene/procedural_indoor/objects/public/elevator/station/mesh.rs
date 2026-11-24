use super::*;
use crate::scene::procedural_indoor::{
    geometry::housing_outline,
    materials::{appliance_panel::Glyph, indicator_lens_finish},
};

const LABEL: &str = "control_panel";

fn corners(xy: Vec2, size: Vec2, z: f32) -> [Vec3; 4] {
    [
        Vec2::NEG_ONE,
        Vec2::new(1., -1.),
        Vec2::ONE,
        Vec2::new(-1., 1.),
    ]
    .map(|p| (xy + size * p * 0.5).extend(z))
}

// A stamped bezel with an open face for the insert and no hidden rear cap.
fn bezel(g: &mut Geometry, xy: Vec2, size: Vec2, back: f32, front: f32) -> [Vec3; 4] {
    let base = corners(xy, size, back);
    let top = corners(xy, size - Vec2::splat((front - back) * 0.68), front);
    for i in 0..4 {
        let j = (i + 1) % 4;
        g.metric_quad([base[i], base[j], top[j], top[i]], Vec3::Z);
    }
    top
}

fn glyph_uv(p: Vec2, direction: Direction) -> [f32; 2] {
    let [lo, right, _, up] = Glyph::Start.uvs().map(Vec2::from_array);
    let st = match direction {
        Direction::Up => Vec2::new(p.y, -p.x),
        Direction::Down => Vec2::new(-p.y, p.x),
    } * 0.5
        + Vec2::splat(0.5);
    (lo + (right - lo) * st.x + (up - lo) * st.y).to_array()
}

fn button(
    a: &mut Assembly,
    station: &Station,
    center: Vec2,
    direction: Direction,
    outline: &[Vec2; 32],
) {
    let radius = station.button_diameter_m * 0.5;
    let z = station.face_z_m + station.plate_depth_m;
    let front = z + station.plate_depth_m * 0.59;
    let cap_radius = radius * 0.81;
    let label;
    let rim = if station.registered == Some(direction) {
        label = format!(
            "{LABEL}#finish{}",
            indicator_lens_finish(station.amber_confirmation)
        );
        a.part(Surface::Plastic, &label)
    } else {
        a.part(Surface::Chrome, LABEL)
    };
    for i in (0..32).step_by(2) {
        let j = (i + 2) % 32;
        rim.metric_quad(
            [
                (center + outline[i] * radius).extend(z),
                (center + outline[j] * radius).extend(z),
                (center + outline[j] * cap_radius).extend(front),
                (center + outline[i] * cap_radius).extend(front),
            ],
            Vec3::Z,
        );
    }
    let cap = a.part(Surface::Ink, &station.key_film.label(LABEL));
    let base = cap.positions.len() as u32;
    cap.positions.push(center.extend(front).to_array());
    cap.normals.push(Vec3::Z.to_array());
    cap.uvs.push(glyph_uv(Vec2::ZERO, direction));
    for (i, &p) in outline.iter().step_by(2).enumerate() {
        cap.positions
            .push((center + p * cap_radius).extend(front).to_array());
        cap.normals.push(Vec3::Z.to_array());
        cap.uvs.push(glyph_uv(p, direction));
        cap.indices
            .extend([base, base + 1 + i as u32, base + 1 + (i as u32 + 1) % 16]);
    }
}

fn readout(a: &mut Assembly, s: &Station) {
    let back = s.face_z_m
        + if s.mount == ReadoutMount::Integrated {
            s.plate_depth_m
        } else {
            0.
        };
    let front = back + s.plate_depth_m * 0.83;
    let xy = s.readout_center_m;
    let face = bezel(
        a.part(Surface::Ink, LABEL),
        xy,
        s.readout_size_m,
        back,
        front,
    );
    let width = face[1].x - face[0].x;
    let height = face[3].y - face[0].y;
    let digit_w = (height * 0.77).min(width * 0.43);
    let g = a.part(Surface::Ink, &s.display_film.label(LABEL));
    let mut cell = |x0: f32, x1: f32, glyph: Glyph| {
        let points = corners(
            Vec2::new((x0 + x1) * 0.5, xy.y),
            Vec2::new(x1 - x0, height),
            front,
        );
        g.vertex_quad(points, [Vec3::Z; 4], glyph.uvs());
    };
    let left = xy.x - digit_w * if s.displayed_floor >= 10 { 1. } else { 0.5 };
    cell(face[0].x, left, Glyph::Blank);
    let mut x = left;
    if s.displayed_floor >= 10 {
        cell(x, x + digit_w, Glyph::Number(s.displayed_floor / 10));
        x += digit_w;
    }
    cell(x, x + digit_w, Glyph::Number(s.displayed_floor % 10));
    cell(x + digit_w, face[1].x, Glyph::Blank);
}

pub(super) fn build(a: &mut Assembly, s: &Station) {
    let stock;
    let label = if s.plate_finish == Surface::Metal {
        stock = s.brushed_stock.label(LABEL);
        &stock
    } else {
        LABEL
    };
    let g = a.part(s.plate_finish, label);
    let front = bezel(
        g,
        s.plate_center_m,
        s.plate_size_m,
        s.face_z_m,
        s.face_z_m + s.plate_depth_m,
    );
    g.metric_quad(front, Vec3::Z);
    let outline = housing_outline(s.button_exponent);
    let offset = if s.call_directions[1].is_some() {
        (if s.horizontal { Vec2::NEG_X } else { Vec2::Y }) * (s.button_pitch_m * 0.5)
    } else {
        Vec2::ZERO
    };
    for (i, &direction) in s.call_directions.iter().flatten().enumerate() {
        button(
            a,
            s,
            s.button_center_m + if i == 0 { offset } else { -offset },
            direction,
            &outline,
        );
    }
    readout(a, s);
}
