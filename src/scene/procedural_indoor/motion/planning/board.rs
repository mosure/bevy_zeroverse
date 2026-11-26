//! Broad phase for a wing rotating on a fitted, sometimes tilted, teaching-board pin.
use super::*;

/// Enclose the planner's collision shapes for every admissible sub-stroke, including
/// their padding. The frame origin is the pin and its Y axis is the hinge axis.
pub(super) fn search_bounds(a: &Assembly, j: &Joint, object: Transform) -> Option<Bounds> {
    let inverse = j.frame.compute_affine().inverse();
    let (lo, hi) = bounds(
        j.parts
            .iter()
            .flat_map(|k| &a.parts[k].positions)
            .map(|p| inverse.transform_point3(Vec3::from_array(*p))),
    );
    if lo.y > hi.y {
        return None;
    }
    let radius = lo.xz().abs().max(hi.xz().abs()).length();
    // The fine sweeps enclose the endpoint corners in each mid-step frame.
    // Their independent X and Z extrema fit inside this square at every
    // heading; its diagonal is a bounding radius for the rotated square.
    let farthest = radius.hypot(lo.y.abs().max(hi.y.abs()));
    let pad = farthest * (1. - (0.5 * ROTATION_STEP_RADIANS).cos()) + 0.0002;
    let r = (radius + pad) * std::f32::consts::SQRT_2;
    Some(rotor::world_bounds(
        (Vec3::new(-r, lo.y - pad, -r), Vec3::new(r, hi.y + pad, r)),
        object.mul_transform(j.frame),
    ))
}
