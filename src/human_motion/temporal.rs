//! Continuous dressed-body clearance against articulated room reservations.
//! This extra certification is skipped entirely without accepted room motion.
use super::skin::{MotionFrame, MotionRig};
use bevy::prelude::*;

/// Upper bound on vertex speed over one keyframe interval (time in [0,1]).
/// Recursing through the exact local-offset/slerp interpolator bounds parent
/// rotation, changing offsets and scale. Skin weights form convex combinations.
fn speed(a: &MotionFrame, b: &MotionFrame, rig: &MotionRig) -> f32 {
    let n = a.bones.len();
    let mut linear = vec![0.; n];
    let mut angular = vec![0.; n];
    let mut maximum = 0_f32;
    for i in 0..n {
        let x = a.bones[i];
        let y = b.bones[i];
        if let Some(parent) = rig.parents[i] {
            let p = a.bones[parent];
            let q = b.bones[parent];
            let u = p.rotation.inverse() * (x.translation - p.translation) / p.scale;
            let v = q.rotation.inverse() * (y.translation - q.translation) / q.scale;
            let radius = u.length().max(v.length());
            let scale = p.scale.abs().max(q.scale.abs()).max_element();
            linear[i] = linear[parent]
                + angular[parent] * scale * radius
                + (q.scale - p.scale).length() * radius
                + scale * u.distance(v);
            angular[i] = angular[parent]
                + (p.rotation.inverse() * x.rotation)
                    .angle_between(q.rotation.inverse() * y.rotation);
        } else {
            linear[i] = x.translation.distance(y.translation);
            angular[i] = x.rotation.angle_between(y.rotation);
        }
        if let Some((lo, hi)) = rig.local_bounds[i] {
            let radius = lo.abs().max(hi.abs()).length();
            let scale = x.scale.abs().max(y.scale.abs()).max_element();
            maximum = maximum.max(
                linear[i] + angular[i] * scale * radius + (y.scale - x.scale).length() * radius,
            );
        }
    }
    maximum
}

pub(super) fn validate(
    frames: &[MotionFrame],
    rig: &MotionRig,
    reservations: &[(Vec3, Vec3)],
) -> Result<(), String> {
    if reservations.is_empty() {
        return Ok(());
    }
    for (index, pair) in frames.windows(2).enumerate() {
        let [a, b] = pair else { unreachable!() };
        let velocity = speed(a, b, rig);
        if !velocity.is_finite() {
            return Err("non-finite temporal clearance bound".into());
        }
        // Eight enclosing slabs are far cheaper than deforming every garment
        // vertex. Midpoint +/- speed * half interval encloses every render vertex.
        let pad = Vec3::splat(velocity / 16. + 0.002);
        for step in 0..8 {
            let pose = a.interpolate(b, (step as f32 + 0.5) / 8., rig);
            let Some((lo, hi)) = rig.bounds(&pose.bones) else {
                continue;
            };
            if reservations
                .iter()
                .any(|&(a, b)| super::validation::bounds_overlap((lo - pad, hi + pad), (a, b)))
            {
                return Err(format!(
                    "continuous dressed pose near frame {index} intersects articulated room travel"
                ));
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn between_frame_crossings_are_rejected_with_clear_endpoints() {
        let rig = MotionRig {
            parents: vec![None],
            annotation_indices: [0; 21],
            stature: 1.7,
            local_bounds: vec![Some((Vec3::splat(-0.02), Vec3::splat(0.02)))],
        };
        let f = |x| MotionFrame {
            bones: vec![Transform::from_xyz(x, 1., 0.)],
            joints: vec![],
            bounds: (Vec3::ZERO, Vec3::ZERO),
        };
        let frames = [f(-1.), f(1.)];
        assert!(validate(
            &frames,
            &rig,
            &[(Vec3::new(-0.01, 0., -0.1), Vec3::new(0.01, 2., 0.1))]
        )
        .is_err());
        assert!(validate(
            &frames,
            &rig,
            &[(Vec3::new(2., 0., -0.1), Vec3::new(3., 2., 0.1))]
        )
        .is_ok());
        assert!(validate(&frames, &rig, &[]).is_ok());
    }
}
