//! Continuous, room-seeded priors for low, seated, standing and elevated views.
//!
//! These are proposal distributions, not acceptance rules. Every lens/path still
//! passes the same primary-room, collision, semantic-coverage and multi-view
//! tests. Heights are measured above the actual supporting floor; truncated
//! logistic densities avoid piles of samples clamped onto a ceiling.
use super::super::layout::stream;
use super::IndoorCamera;
use bevy::prelude::*;
use rand::Rng;
use serde::Serialize;

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ViewpointProgram {
    /// Continuous mixture weights, ordered low/seated/standing/elevated.
    pub weights: [f32; 4],
    pub centers_metres: [f32; 4],
    /// Logistic scale; standard deviation is scale * pi / sqrt(3).
    pub scales_metres: [f32; 4],
    /// Probability of proposing a floor-parallel gaze instead of centering content.
    pub level_gaze_fraction: f32,
    /// Symmetric, continuously sampled pitch interval around the horizon.
    pub level_pitch_span_degrees: f32,
}
impl ViewpointProgram {
    pub fn sample(seed: u64) -> Self {
        let mut rng = stream(seed, 0x56494557504f494e);
        let mut weights = [
            // Low viewpoints need meaningful proposal mass: furnishing and
            // content-framing gates reject many of them. A thin low tail in a
            // standing-height prior effectively disappears after acceptance.
            rng.random_range(0.16..0.34),
            rng.random_range(0.16..0.40),
            rng.random_range(0.26..0.56),
            rng.random_range(0.08..0.32),
        ];
        let total: f32 = weights.iter().sum();
        for weight in &mut weights {
            *weight /= total;
        }
        Self {
            weights,
            centers_metres: [
                rng.random_range(0.76..1.00),
                rng.random_range(1.08..1.46),
                rng.random_range(1.45..1.88),
                rng.random_range(1.91..2.85),
            ],
            scales_metres: [
                rng.random_range(0.045..0.14),
                rng.random_range(0.08..0.21),
                rng.random_range(0.08..0.22),
                rng.random_range(0.17..0.46),
            ],
            level_gaze_fraction: rng.random_range(0.30..0.48),
            level_pitch_span_degrees: rng.random_range(2.0..7.0),
        }
    }
    pub(super) fn height(&self, floor: f32, upper: f32, rng: &mut impl Rng) -> Option<f32> {
        if !floor.is_finite() || !upper.is_finite() {
            return None;
        }
        let low = floor + 0.78;
        let upper = upper.min(floor + 3.25);
        if upper <= low {
            return None;
        }
        let mut draw = rng.random::<f32>();
        let mut component = self.weights.len() - 1;
        for (i, weight) in self.weights.iter().enumerate() {
            if draw < *weight {
                component = i;
                break;
            }
            draw -= weight;
        }
        let center = self.centers_metres[component];
        let scale = self.scales_metres[component];
        let cdf = |height: f32| 1. / (1. + (-(height - center) / scale).exp());
        let a = cdf(low - floor);
        let b = cdf(upper - floor);
        // A component can be far above a low slanted ceiling. Its truncated
        // tail still has a continuous density over all usable eye heights.
        let q = (a + (b - a) * rng.random::<f32>()).clamp(f32::EPSILON, 1. - f32::EPSILON);
        Some((floor + center + scale * (q / (1. - q)).ln()).clamp(low, upper))
    }

    pub(super) fn level_pitch(&self, rng: &mut impl Rng) -> Option<f32> {
        rng.random_bool(self.level_gaze_fraction as f64).then(|| {
            rng.random_range(-self.level_pitch_span_degrees..=self.level_pitch_span_degrees)
                .to_radians()
        })
    }

    /// Keep the scene-content azimuth and target distance in the floor plane.
    /// This is a proposal only: visibility, person framing and overlap are still
    /// tested afterwards. No height clamp or exact zero-pitch rail is introduced.
    pub(super) fn aim_at_pitch(origin: Vec3, target: Vec3, pitch: f32) -> Vec3 {
        target.with_y(origin.y + origin.xz().distance(target.xz()) * pitch.tan())
    }

    pub(super) fn level_gaze(&self, camera: &mut IndoorCamera, rng: &mut impl Rng) {
        // Free orientations belong to an explicit handheld program.
        if camera
            .motion
            .as_ref()
            .is_some_and(|m| m.orientations.is_some())
        {
            return;
        }
        if let Some(pitch) = self.level_pitch(rng) {
            Self::apply_pitch(camera, pitch);
        }
    }

    pub(super) fn apply_pitch(camera: &mut IndoorCamera, pitch: f32) {
        camera.target = Self::aim_at_pitch(camera.start, camera.target, pitch);
        if let Some(motion) = &mut camera.motion {
            // Use the endpoint's own height/distance, so a moving camera does
            // not pitch down just because it gets closer to its original target.
            motion.target_end = Self::aim_at_pitch(camera.end, motion.target_end, pitch);
        }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn room_priors_have_continuous_independent_parameters_and_replay() {
        let mut parameters = std::collections::BTreeSet::new();
        for seed in 0..128 {
            let p = ViewpointProgram::sample(seed);
            assert_eq!(p, ViewpointProgram::sample(seed));
            assert!((p.weights.iter().sum::<f32>() - 1.).abs() < 0.000001);
            assert!(p.weights.iter().all(|w| *w > 0.));
            parameters.insert(p.weights.map(f32::to_bits));
        }
        assert_eq!(parameters.len(), 128);
    }
    #[test]
    fn height_proposals_follow_the_supporting_floor_and_have_no_band_rails() {
        let p = ViewpointProgram::sample(204);
        for (floor, ceiling) in [(0., 3.25), (1.2, 3.25), (-0.25, 1.3)] {
            let mut rng = stream(900, 17);
            let mut replay_rng = rng.clone();
            let mut quantized = std::collections::BTreeSet::new();
            let mut at_boundary = 0;
            for _ in 0..1024 {
                let height = p.height(floor, ceiling, &mut rng).unwrap();
                assert_eq!(height, p.height(floor, ceiling, &mut replay_rng).unwrap());
                assert!((floor + 0.78..=ceiling.min(floor + 3.25)).contains(&height));
                quantized.insert((height * 10000.).round() as i32);
                if height == floor + 0.78 || height == ceiling.min(floor + 3.25) {
                    at_boundary += 1;
                }
            }
            assert!(quantized.len() > 800);
            assert!(
                at_boundary < 3,
                "density should not concentrate at clearance boundaries"
            );
        }
        assert!(p.height(1.0, 1.75, &mut stream(0, 0)).is_none());
        assert!(p.height(f32::NAN, 3.25, &mut stream(0, 0)).is_none());
        assert!(p.height(0., f32::NAN, &mut stream(0, 0)).is_none());
    }

    #[test]
    fn continuous_mixture_keeps_low_and_elevated_tail_mass() {
        let mut low = 0;
        let mut standing = 0;
        let mut elevated = 0;
        for seed in 0..128 {
            let p = ViewpointProgram::sample(seed);
            let mut rng = stream(seed, 31);
            for _ in 0..64 {
                let height = p.height(0., 3.25, &mut rng).unwrap();
                low += usize::from(height < 0.85);
                standing += usize::from((1.4..2.0).contains(&height));
                elevated += usize::from(height > 2.1);
            }
        }
        // Proposal support must survive practical rejection sampling. Realized
        // scene coverage is checked separately after all placement/view gates.
        assert!(low > 256, "low-view proposal tail disappeared: {low}/8192");
        assert!(
            standing > 1600,
            "ordinary eye heights disappeared: {standing}/8192"
        );
        assert!(
            elevated > 256,
            "elevated-view proposal tail disappeared: {elevated}/8192"
        );
    }

    #[test]
    fn level_gaze_keeps_motion_and_continuous_endpoint_pitch() {
        let mut camera = IndoorCamera {
            ceiling_mount: None,
            fisheye: None,
            principal_point_offset: None,
            start: Vec3::new(-2., 1.3, 0.),
            end: Vec3::new(-1., 1.6, 0.5),
            target: Vec3::new(1., 0.8, -3.),
            fov_degrees: 65.,
            motion: Some(super::super::CameraMotion {
                orientations: None,
                control: [Vec3::new(-1.8, 1.4, 0.1), Vec3::new(-1.3, 1.5, 0.4)],
                target_end: Vec3::new(1.2, 0.9, -2.8),
                roll: [-0.04, 0.03],
                route: Vec::new(),
            }),
        };
        let original = camera.clone();
        for pitch in [-0.08_f32, -0.015, 0.025, 0.075] {
            ViewpointProgram::apply_pitch(&mut camera, pitch);
            for time in [0., 1.] {
                let direction = camera.transform_at(time).rotation * Vec3::NEG_Z;
                assert!((direction.y.asin() - pitch).abs() < 1e-5);
            }
            assert_eq!(camera.start, original.start);
            assert_eq!(camera.end, original.end);
            assert_eq!(camera.target.xz(), original.target.xz());
            assert_eq!(
                camera.motion.as_ref().unwrap().control,
                original.motion.as_ref().unwrap().control
            );
        }
        let orientations = [Quat::from_rotation_x(0.2), Quat::from_rotation_y(0.4)];
        camera.motion.as_mut().unwrap().orientations = Some(orientations);
        let free = camera.clone();
        ViewpointProgram::sample(0).level_gaze(&mut camera, &mut stream(0, 5));
        assert_eq!(
            camera, free,
            "explicit handheld orientation must be preserved"
        );
    }

    #[test]
    fn accepted_views_keep_level_gazes_and_height_tails_with_overlap() {
        use super::super::super::{
            layout::{IndoorLayout, IndoorManifest},
            validation::validate_layout,
        };
        let mut level = 0;
        let mut down = 0;
        let mut up = 0;
        let mut low = 0;
        let mut high = 0;
        let mut total = 0;
        for seed in 0..64 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25)
                    .unwrap();
            validate_layout(&scene).unwrap();
            let policy = scene.camera_settings.multiview.as_ref().unwrap();
            assert!(scene
                .camera_overlap()
                .iter()
                .all(|pair| policy.accepts(&pair.samples)));
            for camera in &scene.cameras {
                for time in [0., 0.25, 0.5, 0.75, 1.] {
                    let pose = camera.transform_at(time);
                    let pitch = (pose.rotation * Vec3::NEG_Z)
                        .y
                        .clamp(-1., 1.)
                        .asin()
                        .to_degrees();
                    let height = pose.translation.y - scene.floor_height(pose.translation.xz());
                    level += usize::from(pitch.abs() <= 5.);
                    up += usize::from(pitch > 5.);
                    down += usize::from(pitch < -10.);
                    low += usize::from(height < 1.05);
                    high += usize::from(height > 2.1);
                    total += 1;
                }
            }
        }
        assert!(
            level * 100 >= total * 30,
            "near-horizontal views collapsed: {level}/{total}"
        );
        assert!(
            // Upward pitches beyond 5 degrees are a small tail of the existing
            // content-directed views; preserve that tail alongside level views.
            down * 100 >= total * 10 && up * 100 >= total * 2,
            "inclined views disappeared: down={down} up={up} total={total}"
        );
        assert!(
            low * 100 >= total * 3 && high * 100 >= total * 5,
            "height tails collapsed: low={low} high={high} total={total}"
        );
    }
}
