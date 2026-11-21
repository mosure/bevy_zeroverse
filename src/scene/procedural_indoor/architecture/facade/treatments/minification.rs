//! Freeze a single RGB raster policy for the rig. Repeated opaque blades and
//! open gaps can beat against the sensor even when their PBR textures are mipmapped.
use super::super::{Facade, Shade};
use crate::{calibration::CameraCalibration, scene::procedural_indoor::layout::IndoorManifest};
use bevy::prelude::*;
use bevy_zeroverse_capture::sampling::window_detail_budget;

#[derive(Clone, Copy)]
struct Cover {
    center: Vec3,
    width: Vec3,
    height: f32,
    pitch: f32,
}

impl Cover {
    fn projected(self, view: &Transform, sensor: &CameraCalibration) -> bool {
        let camera_from_room = view.rotation.inverse();
        let project = |p: Vec3| {
            sensor
                .project((camera_from_room * (p - view.translation)).to_array())
                .map(Vec2::from_array)
        };
        let pixels = sensor.image_size[0] as f32 * sensor.image_size[1] as f32;
        // Evaluate the local projected period perpendicular to the slats. The
        // norm of projected world-up alone overestimates it on an oblique wall.
        for fraction in [-0.30, 0., 0.30] {
            let p = self.center + self.width * fraction;
            let Some(uv) = project(p) else { continue };
            if !sensor.in_image(uv.to_array()) {
                continue;
            }
            let (Some(across), Some(up)) = (
                project(p + self.width * 0.05),
                project(p + Vec3::Y * self.pitch),
            ) else {
                continue;
            };
            let dx = (across - uv) * 20.;
            let projected_cell = dx.perp_dot(up - uv).abs();
            let spacing = projected_cell / dx.length().max(0.001);
            let area = projected_cell * self.height / self.pitch;
            if spacing < 3. && area > (pixels * 0.012).max(100.) {
                return true;
            }
        }
        false
    }
}

fn covers(f: &Facade, wall: Transform) -> impl Iterator<Item = Cover> + '_ {
    f.openings.iter().filter_map(move |o| {
        let p = f.treatment.as_ref()?;
        let span = o.max - o.min;
        let coverage = f.shade_coverage.clamp(0., 1.);
        let mut center = (o.min + o.max) * 0.5;
        let (width, height, pitch) = match f.shade {
            Shade::Venetian if coverage > 0.08 => {
                let height = (span.y - 0.123) * coverage;
                center.y = o.max.y - 0.085 - height * 0.5;
                let full = span.y - 0.123;
                let pitch = full / (full / p.slat_pitch).floor().clamp(2., 80.);
                (
                    span.x - 0.066,
                    height,
                    pitch * if p.slats.is_some() { 1. } else { coverage },
                )
            }
            Shade::PanelTrack => {
                let t = p.track?;
                let (n, panel, step) = t.panels(span.x);
                let width = panel + step * coverage * (n - 1) as f32;
                let shift = ((span.x - 0.060) - width) * 0.5;
                center.x += if super::PanelTrack::parks_right(p.stack, p.identity) {
                    shift
                } else {
                    -shift
                };
                (width, span.y - 0.123, t.reed_pitch_m)
            }
            _ => return None,
        };
        (height > pitch * 7. && width > 0.).then(|| Cover {
            center: wall.transform_point(center.extend(0.108)),
            width: wall.rotation * Vec3::X * width,
            height,
            pitch,
        })
    })
}

/// For automatic RGB captures below 720p only. One factor for all views and
/// timesteps preserves the sample-level sampling contract and stable targets.
pub(crate) fn rgb_factor(scene: &IndoorManifest, size: UVec2, count: usize, limit: u32) -> u32 {
    let cameras = &scene.cameras[..count.min(scene.cameras.len())];
    let factor = window_detail_budget(size.to_array(), cameras.len(), limit);
    if factor == 2 {
        return 2;
    }
    let mut all = Vec::new();
    if let Some(envelope) = &scene.envelope {
        for wall in &envelope.walls {
            if let Some(f) = &wall.facade {
                all.extend(covers(f, envelope.wall_transform(wall.edge)));
            }
        }
    } else if let Some(exterior) = &scene.exterior {
        for f in &exterior.facades {
            all.extend(covers(f, f.side.transform(scene.room_size)));
        }
    }
    if all.is_empty() {
        return 2;
    }
    let mut affected = 0;
    for c in cameras {
        let offset = c.principal_point_offset.unwrap_or_default();
        let projection = crate::camera::pinhole::projection(
            PerspectiveProjection {
                fov: c.fov_degrees.to_radians(),
                aspect_ratio: size.x as f32 / size.y as f32,
                ..default()
            },
            offset,
        );
        let sensor =
            crate::camera::lens::calibration(&projection, c.fisheye, offset, size.to_array());
        let Ok(sensor) = sensor else { continue };
        let mut trajectory = c.runtime_trajectory();
        let affected_camera = [0., 0.5, 1.].iter().any(|&t| {
            let pose = trajectory.sample(t);
            all.iter().any(|b| b.projected(&pose, &sensor))
        });
        affected += usize::from(affected_camera);
        if affected >= cameras.len().div_ceil(3) {
            return factor;
        }
    }
    2
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projected_blades_must_cross_the_sensor_and_be_large_enough_to_matter() {
        let k = CameraCalibration::centered_pinhole(256, 256, 1., 1.).unwrap();
        let blind = Cover {
            center: Vec3::new(0., 0., -8.),
            width: Vec3::X * 3.,
            height: 1.5,
            pitch: 0.04,
        };
        assert!(blind.projected(&Transform::IDENTITY, &k));
        assert!(!Cover {
            center: -blind.center,
            ..blind
        }
        .projected(&Transform::IDENTITY, &k));
        assert!(!Cover {
            center: Vec3::new(30., 0., -8.),
            ..blind
        }
        .projected(&Transform::IDENTITY, &k));
        assert!(!Cover {
            width: Vec3::X * 0.12,
            height: 0.15,
            ..blind
        }
        .projected(&Transform::IDENTITY, &k));
        assert!(!Cover {
            pitch: 0.3,
            ..blind
        }
        .projected(&Transform::IDENTITY, &k));
    }

    #[test]
    fn capture_policy_respects_rig_lenses_explicit_factors_and_annotation_only_requests() {
        use crate::{
            app::BevyZeroverseConfig,
            camera::lens::LensSampling,
            render::{ssaa::capture_factor, RenderMode},
            scene::{
                procedural_indoor::layout::{IndoorCamera, IndoorLayout},
                ZeroverseSceneType,
            },
        };
        let mut scene =
            IndoorManifest::generate_with_humans(31, IndoorLayout::Kitchen, 0.65, 1, 0.).unwrap();
        let env = scene.envelope.as_mut().unwrap();
        for wall in &mut env.walls {
            wall.facade = None;
        }
        let tf = env.wall_transform(env.walls[0].edge);
        env.walls[0].facade = Some(super::super::tests::facade(
            41,
            Shade::Venetian,
            Vec2::new(3., 2.4),
            0.7,
        ));
        let at = tf.transform_point(Vec3::new(0., 1.5, 0.));
        let eye = tf.transform_point(Vec3::new(0., 1.5, 7.));
        let camera = IndoorCamera {
            roll_radians: 0.,
            ceiling_mount: None,
            fisheye: None,
            principal_point_offset: Some(Vec2::new(0.04, -0.03)),
            start: eye,
            end: eye,
            target: at,
            fov_degrees: 60.,
            motion: None,
        };
        let mut config = BevyZeroverseConfig {
            scene_type: ZeroverseSceneType::ProceduralIndoor,
            num_cameras: 1,
            headless: true,
            render_mode: RenderMode::Color,
            render_modes: vec![RenderMode::Color],
            ..default()
        };
        let size = UVec2::splat(256);
        let fisheye = LensSampling {
            probability: 1.,
            fov_degrees: [150.; 2],
            ..default()
        }
        .sample(31, 0)
        .unwrap();
        for lens in [None, Some(fisheye)] {
            scene.cameras = vec![IndoorCamera {
                fisheye: lens,
                ..camera.clone()
            }];
            assert_eq!(capture_factor(size, &config, Some(&scene), 8192), 4);
            scene.cameras[0].target = 2. * eye - at;
            assert_eq!(capture_factor(size, &config, Some(&scene), 8192), 2);
        }
        scene.cameras = vec![camera.clone(); 4];
        config.num_cameras = 4;
        for c in &mut scene.cameras[1..] {
            c.target = 2. * eye - at;
        }
        assert_eq!(capture_factor(size, &config, Some(&scene), 8192), 2);
        scene.cameras[1] = camera;
        assert_eq!(capture_factor(size, &config, Some(&scene), 8192), 4);
        for f in 1..=4 {
            config.ssaa_factor = f;
            assert_eq!(capture_factor(size, &config, Some(&scene), 8192), f);
        }
        config.ssaa_factor = 0;
        config.render_modes = vec![RenderMode::Semantic];
        assert_eq!(capture_factor(size, &config, Some(&scene), 8192), 2);
    }

    #[test]
    fn small_indoor_rigs_have_bounded_sample_and_attachment_cost() {
        assert_eq!(window_detail_budget([256; 2], 3, 8192), 4);
        assert_eq!(window_detail_budget([512; 2], 3, 8192), 3);
        assert_eq!(window_detail_budget([512; 2], 12, 8192), 2);
        assert_eq!(window_detail_budget([768; 2], 2, 8192), 2);
        assert_eq!(window_detail_budget([256; 2], 3, 512), 2);
    }
}
