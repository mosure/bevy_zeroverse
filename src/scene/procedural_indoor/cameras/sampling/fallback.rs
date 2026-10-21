//! Rare, bounded recovery when fixed person-centred rings miss a narrow view.
//! Normal proposals are unchanged; geometry and acceptance gates are identical.
use super::*;
use crate::scene::procedural_indoor::cameras::CameraMotion;

impl IndoorManifest {
    pub(super) fn person_free_space_fallback(
        &self,
        people: &[&super::super::super::humans::IndoorHuman],
        rng: &mut rand_chacha::ChaCha8Rng,
        coverage: &super::super::coverage::Coverage,
    ) -> Option<IndoorCamera> {
        let (lo, hi) = if self.camera_settings.primary_room {
            self.primary_room_bounds()
        } else {
            (-self.room_size.xz() * 0.5, self.room_size.xz() * 0.5)
        };
        let distance = self.camera_settings.minimum_travel(self.seed, 0);
        // A deterministic metric grid covers free space missed by fixed-radius
        // rings. It runs only after both original proposal budgets are exhausted.
        // Floor-relative low views also retain head visibility past furniture.
        for x in 0..40 {
            for z in 0..40 {
                let p = lo + (hi - lo) * Vec2::new(x as f32 + 0.5, z as f32 + 0.5) / 40.;
                let floor = self.floor_height(p);
                for height in [0.78, 1.10, 1.55, 2.0] {
                    let start = Vec3::new(p.x, floor + height, p.y);
                    if !self.camera_clear(start) {
                        continue;
                    }
                    for human in people {
                        for joint in [2, 4] {
                            let target = human.transform().transform_point(human.joints[joint]);
                            if !self.camera_view_clear(start, target) {
                                continue;
                            }
                            for fov_degrees in [84., 96., 106.]
                                .map(|f| self.camera_settings.intrinsics.recovery(f))
                            {
                                let still = IndoorCamera {
                                    ceiling_mount: None,
                                    fisheye: self
                                        .camera_settings
                                        .lenses
                                        .as_ref()
                                        .and_then(|l| l.sample(self.seed, 0)),
                                    principal_point_offset: None,
                                    start,
                                    end: start,
                                    target,
                                    fov_degrees,
                                    motion: None,
                                };
                                if !coverage.suitable(&still, true) {
                                    continue;
                                }
                                for direction in [Vec3::X, Vec3::NEG_X, Vec3::Z, Vec3::NEG_Z] {
                                    let end = start + direction * distance;
                                    let mut camera = IndoorCamera {
                                        end,
                                        motion: (distance > 0.).then_some(CameraMotion {
                                            orientations: None,
                                            route: Vec::new(),
                                            control: [
                                                start.lerp(end, 1. / 3.),
                                                start.lerp(end, 2. / 3.),
                                            ],
                                            target_end: target,
                                            roll: [0.; 2],
                                        }),
                                        ..still.clone()
                                    };
                                    self.camera_settings.prepare_motion(
                                        self.seed,
                                        0,
                                        &mut camera,
                                        rng,
                                    );
                                    if self.camera_settings.accepts_motion(self.seed, 0, &camera)
                                        && self.camera_curve_clear(&camera)
                                        && coverage.suitable(&camera, true)
                                        && !self.cameras.iter().any(|c| {
                                            c.start.distance(camera.start) < 0.18
                                                && c.target.distance(camera.target) < 0.5
                                        })
                                    {
                                        return Some(camera);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        None
    }
}
