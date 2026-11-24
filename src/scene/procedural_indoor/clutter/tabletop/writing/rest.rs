//! The horizontal support line touches a barrel face, or the barrel and a pocket clip.
use super::*;

impl WritingProgram {
    pub(crate) fn table_rest(&self, seed: u64) -> (Transform, bool) {
        let mut r = stream(seed, u64::from_le_bytes(*b"ToolRest"));
        let on_clip = self.clip && r.random_bool(0.60);
        let normal = if on_clip {
            // Upper tangent from the outside corner of the clip to the
            // transverse barrel polygon. Reflect for the opposite stable roll.
            let c = self.clip_rings()[2][1].xz() / self.radius;
            let mut tangent = Vec2::X - c;
            for (point, _, _) in self.outline().meridians() {
                if tangent.perp_dot(point - c) > 0. {
                    tangent = point - c;
                }
            }
            let mut n = tangent.perp().normalize();
            if r.random_bool(0.5) {
                n.y = -n.y;
            }
            n
        } else {
            // A clip on the upper half cannot interfere with this bearing face.
            let n = self.sides as i32;
            let side = if self.clip {
                r.random_range(-n / 4..n / 4)
            } else {
                r.random_range(0..n)
            };
            Vec2::from_angle((side as f32 + 0.5) * TAU / self.sides as f32)
        };
        let mut rot = Quat::from_rotation_x(-FRAC_PI_2)
            * Quat::from_rotation_y(FRAC_PI_2 + normal.to_angle());
        let mut center_z = self.length * 0.5;
        if self.mechanism == 2 {
            // A marker cap is wider than its barrel. In a horizontal pose
            // the thin end would be cantilevered beyond the cap.
            let up = self.marker_support_normal(-normal, on_clip);
            let right = Vec3::Y.cross(up).normalize();
            rot = Quat::from_mat3(&Mat3::from_cols(right, up, right.cross(up))).inverse();
            let (near, far) = self.projected_extent(rot.inverse() * Vec3::Z);
            center_z = -(near + far) * 0.5;
        }
        let (left, right) = self.projected_extent(rot.inverse() * Vec3::X);
        let (bottom, _) = self.vertical_range(rot);
        (
            Transform::from_xyz(-(left + right) * 0.5, -bottom, center_z).with_rotation(rot),
            on_clip,
        )
    }

    /// The support plane bears on the last full-width section of both cap and
    /// barrel. A clip-supported roll also brings its outside corner onto it.
    fn marker_support_normal(&self, roll: Vec2, on_clip: bool) -> Vec3 {
        let (cap_y, barrel_y) = (self.cap_length(), self.length * MARKER_SHOULDER_HEIGHT);
        let plane = |roll: Vec2| {
            let cap_low = self.radius * self.outline().projected_range(roll).0;
            let pitch = (cap_low - MARKER_BARREL_RADIUS * cap_low) / (barrel_y - cap_y);
            (Vec3::new(roll.x, pitch, roll.y), cap_low + pitch * cap_y)
        };
        if !on_clip {
            return plane(roll).0.normalize();
        }
        let rings = self.clip_rings();
        let mut clear = roll;
        let mut touching = Vec2::X;
        for _ in 0..16 {
            let mid = (clear + touching) * 0.5;
            let (normal, height) = plane(mid);
            if rings.iter().flatten().any(|p| normal.dot(*p) < height) {
                touching = mid;
            } else {
                clear = mid;
            }
        }
        plane(clear).0.normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::{audit, build_object};

    #[test]
    fn loose_tools_rest_on_two_bearing_edges_across_barrel_and_clip_rolls() {
        let mut clip_rest = 0;
        for seed in 0..160 {
            let o = audit::fixture(ObjectKind::Pencil, seed);
            let p = parameters(&o);
            let (tf, on_clip) = p.table_rest(seed);
            let a = build_object(&o);
            let mut left = f32::INFINITY;
            let mut right = f32::NEG_INFINITY;
            let mut near = f32::INFINITY;
            let mut far = f32::NEG_INFINITY;
            for pt in a.parts.values().flat_map(|g| &g.positions) {
                if pt[1].abs() < 1e-6 {
                    left = left.min(pt[0]);
                    right = right.max(pt[0]);
                    let along = (tf.rotation.inverse() * (Vec3::from(*pt) - tf.translation)).y;
                    near = near.min(along);
                    far = far.max(along);
                }
            }
            assert!(
                near < p.length * 0.4 && far > p.length * 0.52,
                "cantilever {seed}"
            );
            assert!(
                right - left > p.radius * 0.15,
                "unstable support {seed} {left}/{right}"
            );
            if p.clip {
                let clip_low = p
                    .clip_rings()
                    .into_iter()
                    .flatten()
                    .map(|v| tf.transform_point(v).y)
                    .fold(f32::INFINITY, f32::min);
                assert_eq!(on_clip, clip_low < 1e-6);
            }
            clip_rest += usize::from(on_clip);
        }
        assert!((16..110).contains(&clip_rest));
    }
}
