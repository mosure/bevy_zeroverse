//! CPU-only schematic and prediction overlay example.
//! cargo run --example schematic --no-default-features --features multi_threaded
use bevy::prelude::*;
use bevy_zeroverse::{
    annotation::schematic::{Overlay, RenderOptions, Schematic},
    scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest},
};
fn main() -> anyhow::Result<()> {
    if std::env::args().any(|a| a == "--lenses") {
        return lens_examples();
    }
    for seed in [47_586_113, 47_359_576, 47_062_729] {
        let scene = IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 4, 0.25)
            .map_err(anyhow::Error::msg)?;
        let plan = Schematic::from_manifest(&scene, 0.4).map_err(anyhow::Error::msg)?;
        plan.write(
            format!("out/schematic/{seed}"),
            RenderOptions::default(),
            Overlay::default(),
        )?;
        let mut predicted = plan.cameras[0].clone();
        predicted.label = "Predicted C0".into();
        let matrix = Mat4::from_cols_array_2d(&predicted.world_from_view);
        predicted.world_from_view = (Mat4::from_translation(Vec3::new(0.5, 0., 0.35))
            * matrix
            * Mat4::from_rotation_y(0.12))
        .to_cols_array_2d();
        predicted.path.clear();
        plan.write(
            format!("out/schematic/{seed}-prediction"),
            RenderOptions::default(),
            Overlay {
                cameras: vec![predicted],
                ..default()
            },
        )?;
    }
    Ok(())
}

/// CPU-only inspection of the same mixed-lens annotations used by the viewer.
fn lens_examples() -> anyhow::Result<()> {
    use bevy_zeroverse::camera::lens::LensSampling;
    for seed in [400, 401] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 3, 0.25)
                .map_err(anyhow::Error::msg)?;
        let mut settings = scene.camera_settings.clone();
        settings.lenses = Some(LensSampling::default());
        scene
            .resample_cameras(3, settings, 1.)
            .map_err(anyhow::Error::msg)?;
        let plan = Schematic::from_manifest(&scene, 0.4).map_err(anyhow::Error::msg)?;
        plan.write(
            format!("out/schematic/lenses/room-{seed}"),
            RenderOptions::default(),
            Overlay::default(),
        )?;
    }
    use bevy_zeroverse::{
        annotation::schematic::{Camera, Footprint, FootprintKind},
        calibration::{Fisheye, FisheyeFit},
    };
    // A calibration fixture gives each lens room to inspect its shape. It is
    // deliberately separate from the generated-room examples above.
    let mut plan = Schematic {
        schema_version: 1,
        seed: 0,
        trajectory_progress: 0.,
        time_seconds: None,
        footprints: vec![Footprint {
            label: "Calibration fixture".into(),
            instance_id: None,
            points: vec![[-5., 0., -4.], [5., 0., -4.], [5., 0., 4.], [-5., 0., 4.]],
            kind: FootprintKind::Floor,
        }],
        cameras: vec![],
        poses: vec![],
    };
    for (i, (x, z, degrees, fit, ceiling)) in [
        (-3., -1.5, None, FisheyeFit::Diagonal, false),
        (0., -1.5, Some(140.), FisheyeFit::Diagonal, false),
        (3., -1.5, Some(180.), FisheyeFit::Circular, false),
        (-3., 2., Some(210.), FisheyeFit::Circular, false),
        (0., 2., Some(190.), FisheyeFit::Circular, true),
        (3., 2., Some(160.), FisheyeFit::Elliptical, false),
    ]
    .into_iter()
    .enumerate()
    {
        let mut tf = Transform::from_xyz(x, if ceiling { 3. } else { 1.5 }, z);
        if ceiling {
            tf = tf.looking_at(Vec3::new(x, 0., z), Vec3::Z);
        }
        plan.cameras.push(Camera {
            label: format!("C{i}"),
            world_from_view: tf.to_matrix().to_cols_array_2d(),
            fov_y: 65_f32.to_radians(),
            aspect: 1.,
            principal_point_offset: None,
            path: vec![],
            calibration: degrees.map(|degrees: f32| {
                Fisheye {
                    coefficients: [0.; 4],
                    max_angle: degrees.to_radians() * 0.5,
                }
                .calibration(
                    [640, 480],
                    fit,
                    [0.; 2],
                    if fit == FisheyeFit::Elliptical {
                        1.15
                    } else {
                        1.
                    },
                )
                .unwrap()
            }),
        });
    }
    let options = RenderOptions {
        width: 1200,
        height: 1000,
        ..default()
    };
    plan.write(
        "out/schematic/lenses/calibration",
        options,
        Overlay::default(),
    )?;
    let mut predicted = plan.cameras[4].clone();
    predicted.label = "Predicted C4".into();
    predicted.world_from_view[3][0] += 0.5;
    predicted.world_from_view[3][2] += 0.3;
    plan.write(
        "out/schematic/lenses/prediction",
        options,
        Overlay {
            cameras: vec![predicted],
            ..default()
        },
    )?;
    Ok(())
}
