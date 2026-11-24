//! Removable input drawers have a formed finger recess and a narrow seam
//! around the front, separate from the moving output above.
use super::*;
use crate::scene::procedural_indoor::objects::electrical::{edge, pierced, Rect};

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &PrinterProgram,
    front: f32,
    label: &str,
    trim: &str,
    case: Surface,
) {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"PaperBin"));
    if !r.random_bool(0.82) {
        return;
    }
    let body = p.body;
    let center = Vec2::new(0., p.lower_top * 0.47);
    let size = Vec2::new(
        body.x * r.random_range(0.80..0.94),
        p.lower_top * r.random_range(0.66..0.80),
    );
    let face = Rect::new(center, size);
    let grip = Rect::new(
        center + Vec2::new(size.x * r.random_range(-0.10..0.10), size.y * 0.13),
        Vec2::new(
            size.x * r.random_range(0.27..0.46),
            size.y * r.random_range(0.27..0.43),
        ),
    );
    let inset = Rect::new(
        (grip.hi + grip.lo) * 0.5,
        (grip.hi - grip.lo) * Vec2::new(0.93, 0.66),
    );
    let face_z = front + r.random_range(0.0055..0.009);
    let bottom_z = front + 0.0008;
    let g = a.part(case, label);
    pierced(g, face, face_z, &[grip]);
    edge(g, face, front, face, face_z, false);
    let liner = a.part(Surface::Plastic, trim);
    edge(liner, grip, face_z, inset, bottom_z, true);
    inset.face(liner, bottom_z);
}
