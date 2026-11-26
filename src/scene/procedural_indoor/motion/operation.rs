//! An admitted rigid stroke can be traversed, held at its stop and retraced.
//! Sampling the stroke does not depend on frame order or repeat collision work.
use super::*;

/// A single outward stroke, dwell and return on the room’s normalized clock.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct OperationCycle {
    pub starts_at: f32,
    pub reaches_stop_at: f32,
    pub leaves_stop_at: f32,
    pub returns_at: f32,
}

impl OperationCycle {
    pub(super) fn sample(joint: &Joint, fraction: f32) -> Option<Self> {
        let operates = match joint.kind {
            MotionKind::TeachingBoard | MotionKind::Cabinet | MotionKind::Appliance => true,
            // Synchronized sliding leaves share a cycle; revolving wings keep their direction.
            MotionKind::Door => !joint.coupled || joint.angle.abs() < 1e-6,
            // A hinged laptop lid and a loose tabletop object both report as a prop.
            MotionKind::Prop => joint.contact == Contact::Free && joint.angle != 0.,
            _ => false,
        };
        if !operates {
            return None;
        }
        let mut r = stream(joint.program_seed, u64::from_le_bytes(*b"OpReturn"));
        if r.random::<f32>() >= fraction {
            return None;
        }
        let duration = r.random_range(0.39..0.95);
        let start = r.random_range(0.0..1. - duration);
        let dwell = r.random_range(0.12..0.43) * duration;
        let outward = (duration - dwell) * r.random_range(0.32..0.65);
        Some(Self {
            starts_at: start,
            reaches_stop_at: start + outward,
            leaves_stop_at: start + outward + dwell,
            returns_at: start + duration,
        })
    }

    /// Zero at rest, one at the admitted stop. Times are normalized scene progress.
    pub fn stroke_fraction_at(self, progress: f32) -> f32 {
        if progress <= self.starts_at || progress >= self.returns_at {
            return 0.;
        }
        let t = if progress < self.reaches_stop_at {
            (progress - self.starts_at) / (self.reaches_stop_at - self.starts_at)
        } else if progress > self.leaves_stop_at {
            (self.returns_at - progress) / (self.returns_at - self.leaves_stop_at)
        } else {
            return 1.;
        };
        t * t * (3. - 2. * t)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn operation_cycles_have_distinct_timing_nested_selection_and_exact_holds() {
        let mut selected = [0; 3];
        let mut midpoints = Vec::new();
        for seed in 0..384 {
            let joint = construction::rigid(
                MotionKind::Cabinet,
                seed,
                Vec3::ZERO,
                Vec3::Y,
                0.9,
                Vec3::ZERO,
            );
            let full = OperationCycle::sample(&joint, 1.).unwrap();
            for (i, p) in [0., 0.5, 1.].into_iter().enumerate() {
                if let Some(c) = OperationCycle::sample(&joint, p) {
                    assert_eq!(c, full);
                    selected[i] += 1;
                }
            }
            let c = full;
            assert!(
                0. <= c.starts_at
                    && c.starts_at < c.reaches_stop_at
                    && c.reaches_stop_at < c.leaves_stop_at
                    && c.leaves_stop_at < c.returns_at
                    && c.returns_at <= 1.
            );
            midpoints.push((c.reaches_stop_at + c.leaves_stop_at) * 0.5);
            for (t, val) in [
                (0., 0.),
                (c.starts_at, 0.),
                (c.reaches_stop_at, 1.),
                (c.leaves_stop_at, 1.),
                (c.returns_at, 0.),
                (1., 0.),
            ] {
                assert_eq!(c.stroke_fraction_at(t), val);
            }
            for s in [0.04, 0.25, 0.56, 0.91] {
                let out = c.stroke_fraction_at(c.starts_at.lerp(c.reaches_stop_at, s));
                let back = c.stroke_fraction_at(c.returns_at.lerp(c.leaves_stop_at, s));
                assert!((out - back).abs() < 0.00001);
            }
        }
        midpoints.sort_by(f32::total_cmp);
        assert_eq!((selected[0], selected[2]), (0, 384));
        assert!((150..235).contains(&selected[1]));
        assert!(midpoints[345] - midpoints[38] > 0.22);
        eprintln!("operation cycles across 384 seeds: half-policy {}, stop midpoint p10/p50/p90 {:.3}/{:.3}/{:.3}",selected[1],midpoints[38],midpoints[192],midpoints[345]);
    }

    #[test]
    fn cyclic_operation_is_limited_to_hardware_that_can_physically_return() {
        for (kind, contact, coupled, angle, can_return) in [
            (MotionKind::Door, Contact::Free, true, 0., true),
            (MotionKind::Door, Contact::Free, true, 1.3, false),
            (MotionKind::Door, Contact::Free, false, 0.5, true),
            (MotionKind::Appliance, Contact::Free, false, 0.5, true),
            (MotionKind::TeachingBoard, Contact::Free, false, 0.7, true),
            (MotionKind::Prop, Contact::Free, false, 0.3, true),
            (MotionKind::Prop, Contact::Surface, false, 0.3, false),
            (MotionKind::Chair, Contact::Floor, false, 0.7, false),
            (MotionKind::Cart, Contact::Floor, false, 0.3, false),
            (MotionKind::Fan, Contact::Free, false, 8., false),
            (MotionKind::Covering, Contact::Free, false, 0., false),
        ] {
            let mut joint = construction::rigid(kind, 2, Vec3::ZERO, Vec3::Y, angle, Vec3::X);
            joint.coupled = coupled;
            joint.contact = contact;
            assert_eq!(
                OperationCycle::sample(&joint, 1.).is_some(),
                can_return,
                "{kind:?}"
            );
        }
    }
}
