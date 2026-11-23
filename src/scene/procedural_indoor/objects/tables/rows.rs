//! Purchased lecture and dining furniture. One product can be cut to different
//! bank lengths. Intermediate rails carry both the worktop and its field joints.
use super::*;

pub(super) fn specify(p: &mut TableProgram, kind: ObjectKind, r: &mut impl Rng) {
    let dining = kind == ObjectKind::Table;
    p.top_surface = match r.random_range(0..100) {
        0..=55 => Surface::Plastic, // embossed sheet laminate
        56..=89 => Surface::Wood,
        90..=96 if dining => Surface::Ceramic,
        _ if dining => Surface::Concrete,
        _ => Surface::Wood,
    };
    p.top_thickness = match p.top_surface {
        Surface::Plastic => r.random_range(0.023..0.037),
        Surface::Wood => r.random_range(0.026..0.045),
        _ => r.random_range(0.032..0.048),
    };
    p.edge_bevel = r.random_range(0.0015..0.006);
    p.frame_surface = if r.random_bool(0.20) {
        Surface::Chrome
    } else {
        Surface::Metal
    };
    p.span_pitch = r.random_range(1.08..1.56);
    p.end_overhang = r.random_range(0.20..0.40);
    p.bay_style = r.random_range(0..if dining { 6 } else { 5 });
}

pub(super) fn overhang(length: f32, p: &TableProgram) -> f32 {
    p.end_overhang.min(length * 0.21)
}

pub(super) fn stations(length: f32, p: &TableProgram) -> usize {
    (((length - 2. * overhang(length, p)) / p.span_pitch).ceil() as usize + 1).clamp(2, 32)
}

/// A field splice belongs over a crossrail. Look one rail ahead so a short
/// remainder can be part of the last leaf without leaving an unsupported splice.
pub(super) fn joints(o: &IndoorObject, p: &TableProgram) -> impl Iterator<Item = f32> {
    let length = o.size.xz().max_element();
    let count = stations(length, p);
    let end = overhang(length, p);
    let stock_length =
        stream(product_seed(o), u64::from_le_bytes(*b"RowSheet")).random_range(2.42..3.04);
    let mut start = -length * 0.5;
    (1..count - 1).filter_map(move |i| {
        let far = if i + 2 == count {
            length * 0.5
        } else {
            conference::station_offset(i + 1, count, length, end)
        };
        (far - start > stock_length).then(|| {
            start = conference::station_offset(i, count, length, end);
            start
        })
    })
}

pub(super) fn top(a: &mut Assembly, o: &IndoorObject, p: &TableProgram, outline: &[Vec2]) {
    let mut joints = joints(o, p).peekable();
    if joints.peek().is_some() {
        // One to two millimetres between adjoining finished edges.
        conference::cut_leaves(a, o, p, outline, p.edge_bevel * 0.10 + 0.00035, joints);
    } else {
        let tf = Transform::from_xyz(0., o.size.y - p.top_thickness * 0.5, 0.);
        top::TopStock::sample(o, p).build(a, o, p, outline, tf, 0);
    }
}

#[cfg(test)]
mod tests;
