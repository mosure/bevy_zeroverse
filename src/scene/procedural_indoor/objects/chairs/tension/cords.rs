//! Original round-cord weave, also used to replay archived chair meshes.
use super::*;

pub(super) fn build(g: &mut Geometry, frame: &Frame, pitch_m: f32) {
    let rows = (frame.height / pitch_m).round().clamp(5., 18.) as usize;
    let cols = (frame.width / pitch_m).round().clamp(5., 14.) as usize;
    for row in 1..rows {
        let v = row as f32 / rows as f32;
        let width = frame.chord(v * 2. - 1.);
        let path: [_; 11] =
            std::array::from_fn(|i| frame.point((-1.0 + i as f32 * 0.2) * width, v));
        g.tube(&path, 0.0023, 5);
    }
    for col in 1..cols {
        let x = -1.0 + 2.0 * col as f32 / cols as f32;
        let height = frame.chord(x);
        let path: [_; 9] =
            std::array::from_fn(|i| frame.point(x, 0.5 + (i as f32 / 8.0 - 0.5) * height));
        g.tube(&path, 0.0023, 5);
    }
}
