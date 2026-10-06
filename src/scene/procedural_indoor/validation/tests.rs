use super::*;
use crate::scene::procedural_indoor::{
    layout::{IndoorObject, ObjectKind},
    objects,
};

#[test]
fn small_valid_captures_reject_insufficient_pixels_without_panicking() {
    use crate::sample::{AnnotationPrecision, View};
    for [width, height] in [[1, 1], [1, 8], [8, 1], [4, 256], [256, 4]] {
        let mut view = View {
            world_from_view: Mat4::IDENTITY.to_cols_array_2d(),
            fovy: 1.1,
            calibration: Some(
                crate::calibration::CameraCalibration::centered_pinhole(
                    width,
                    height,
                    1.1,
                    width as f32 / height as f32,
                )
                .unwrap(),
            ),
            depth: vec![0; (width * height * 16) as usize],
            position: vec![0; (width * height * 16) as usize],
            normal: vec![0; (width * height * 16) as usize],
            ..Default::default()
        };
        let check = |view: &View, precision| {
            validate_annotations_with_precision(view, [[-5.; 3], [5.; 3]], width, height, precision)
                .unwrap_err()
        };
        for precision in [
            AnnotationPrecision::Float16Hdr,
            AnnotationPrecision::Float32Geometry,
        ] {
            assert_eq!(
                check(&view, precision),
                "insufficient visible annotation pixels"
            );
        }
        view.normal[..4].copy_from_slice(&f32::NAN.to_ne_bytes());
        assert_eq!(
            check(&view, AnnotationPrecision::Float16Hdr),
            "non-finite annotation"
        );
        view.normal.clear();
        assert_eq!(
            check(&view, AnnotationPrecision::Float16Hdr),
            "wrong annotation buffer length"
        );
    }
}

#[test]
fn annotation_value_borrowing_preserves_bits_and_unaligned_inputs() {
    let values = [0.0_f32, -0.0, 1.25, -17.5, f32::MIN_POSITIVE, f32::MAX];
    let bytes = bytemuck::cast_slice(&values);
    let borrowed = annotation_values(bytes, bytes.len()).unwrap();
    assert!(matches!(borrowed, std::borrow::Cow::Borrowed(_)));
    assert_eq!(borrowed.as_ptr(), values.as_ptr());
    let mut storage = vec![0; bytes.len() + 4];
    let offset = (0..4)
        .find(|&offset| {
            !(storage.as_ptr() as usize + offset).is_multiple_of(std::mem::align_of::<f32>())
        })
        .unwrap();
    storage[offset..offset + bytes.len()].copy_from_slice(bytes);
    let owned = annotation_values(&storage[offset..offset + bytes.len()], bytes.len()).unwrap();
    assert!(matches!(owned, std::borrow::Cow::Owned(_)));
    for decoded in [&*borrowed, &*owned] {
        assert_eq!(
            decoded.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
    }
    assert!(annotation_values(bytes, bytes.len() - 1).is_err());
    assert!(annotation_values(&bytes[..bytes.len() - 1], bytes.len() - 1).is_err());
    // Finite scans include trailing channels, even when the pixel's validity is zero.
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let corrupt = [0.0_f32, 0., 0., 0., 0., 0., bad, 0.];
        let bytes = bytemuck::cast_slice(&corrupt);
        assert_eq!(
            annotation_values(bytes, bytes.len()).unwrap_err(),
            "non-finite annotation"
        );
    }
}

#[test]
fn percentile_selection_matches_sorted_total_order_exactly() {
    for count in [100, 101, 137, 4096, 28672] {
        let mut values: Vec<_> = (0..count)
            .map(|i| {
                if i % 17 == 0 {
                    -0.0_f32
                } else if i % 13 == 0 {
                    0.0
                } else {
                    ((i * 7919) % 8093) as f32 / 31.0 - 130.0
                }
            })
            .collect();
        let mut sorted = values.clone();
        sorted.sort_by(f32::total_cmp);
        let expected = sorted[(count as f32 * 0.99) as usize];
        assert_eq!(percentile99(&mut values).to_bits(), expected.to_bits());
    }
}

#[test]
fn float32_annotation_qualification_rejects_single_pixel_outliers() {
    use crate::sample::{AnnotationPrecision, View};
    let size = 64;
    let calibration =
        crate::calibration::CameraCalibration::centered_pinhole(size, size, 1.1, 1.).unwrap();
    let aabb = [[-5.; 3], [5.; 3]];
    let mut view = View {
        world_from_view: Mat4::IDENTITY.to_cols_array_2d(),
        fovy: 1.1,
        calibration: Some(calibration.clone()),
        ..Default::default()
    };
    for y in 0..size {
        for x in 0..size {
            let p = Vec3::from_array(
                calibration
                    .unproject([x as f32 + 0.5, y as f32 + 0.5], 2.)
                    .unwrap(),
            );
            let p = (p + Vec3::splat(5.)) / 10.;
            view.position
                .extend_from_slice(bytemuck::cast_slice(&[p.x, p.y, p.z, 1.]));
            view.depth
                .extend_from_slice(bytemuck::cast_slice(&[2.0_f32, 2., 2., 1.]));
            view.normal
                .extend_from_slice(bytemuck::cast_slice(&[0.5_f32, 0.5, 1., 1.]));
        }
    }
    let check = |view: &View| {
        validate_annotations_with_precision(
            view,
            aabb,
            size,
            size,
            AnnotationPrecision::Float32Geometry,
        )
    };
    let report = check(&view).unwrap();
    assert_eq!(report.checked_pixels, (size * size) as usize);
    assert!(report.reprojection_max_pixels < 0.001);
    // Both an unsampled border pixel and a sampled pixel hidden below p99
    // must fail; neither a sampling gap nor a percentile may conceal a defect.
    for pixel in [0, (2 * size + 2) as usize] {
        let mut corrupt = view.clone();
        let offset = pixel * 16;
        let value = f32::from_ne_bytes(corrupt.position[offset..offset + 4].try_into().unwrap());
        corrupt.position[offset..offset + 4].copy_from_slice(&(value + 0.001).to_ne_bytes());
        assert!(check(&corrupt)
            .unwrap_err()
            .contains("annotation alignment failed"));
        let mut corrupt = view.clone();
        corrupt.depth[offset..offset + 4].copy_from_slice(&2.001_f32.to_ne_bytes());
        assert!(check(&corrupt)
            .unwrap_err()
            .contains("annotation alignment failed"));
    }
}

#[test]
fn wall_mounts_touch_real_back_faces_on_oblique_and_axis_aligned_walls() {
    let mut kinds = std::collections::BTreeSet::new();
    let mut oblique = 0;
    for seed in 0..128 {
        let scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.8, 0, 0.).unwrap();
        validate_layout(&scene).unwrap();
        for o in &scene.objects {
            let Some(offset) = objects::wall::mount_offset(o) else {
                continue;
            };
            assert!(scene.wall_attachment_clear(o));
            let (lo, hi) = objects::build_object(o).bounds();
            // Independent mesh measurement: closest rear face is 1 mm off the
            // analytic wall, and all foreground relief fits the declared box.
            assert!(
                (lo.z + offset - 0.001).abs() < 1e-5,
                "seed={seed} {:?}: rear {lo:?}, mount={offset}",
                o.kind
            );
            assert!(hi.z <= o.size.z * 0.5 + 0.003);
            assert!(hi.y <= o.size.y + 0.003);
            kinds.insert(format!("{:?}", o.kind));
            oblique += usize::from(o.yaw.sin().abs().min(o.yaw.cos().abs()) > 0.01);
        }
        if let Some(o) = scene
            .objects
            .iter()
            .find(|o| objects::wall::mount_offset(o).is_some())
        {
            let mut detached = scene.clone();
            detached.objects[o.id].position += Quat::from_rotation_y(o.yaw) * Vec3::Z * 0.18;
            assert!(validate_layout(&detached)
                .unwrap_err()
                .contains("unmounted"));
            detached.objects[o.id] = o.clone();
            detached.objects[o.id].yaw += std::f32::consts::PI;
            assert!(validate_layout(&detached)
                .unwrap_err()
                .contains("unmounted"));
        }
    }
    assert_eq!(kinds.len(), 6);
    assert!(oblique > 50, "only rectangular mounting was exercised");
}

#[test]
fn saucers_and_clock_relief_fit_narrow_and_shallow_reservations() {
    let mut saucers = 0;
    let mut digital = 0;
    let mut analog = 0;
    for seed in 0..128 {
        for (kind, size) in [
            (ObjectKind::Mug, Vec3::new(0.101, 0.073, 0.109)),
            (ObjectKind::Mug, Vec3::new(0.138, 0.129, 0.076)),
            (ObjectKind::Clock, Vec3::new(0.22, 0.22, 0.045)),
            (ObjectKind::Clock, Vec3::new(0.43, 0.43, 0.045)),
            (ObjectKind::Whiteboard, Vec3::new(0.7, 0.65, 0.12)),
            (ObjectKind::Whiteboard, Vec3::new(2.3, 1.35, 0.18)),
            (ObjectKind::WallArt, Vec3::new(0.35, 1.25, 0.045)),
            (ObjectKind::Display, Vec3::new(0.85, 0.55, 0.07)),
        ] {
            let o = IndoorObject {
                id: 0,
                kind,
                position: Vec3::ZERO,
                size,
                yaw: 0.,
                variant: 0,
                seed,
                solid: false,
                support: None,
                neighbor: false,
                interaction_target: None,
            };
            if kind == ObjectKind::Mug {
                saucers += usize::from(
                    crate::scene::procedural_indoor::clutter::beverages::parameters(&o).style >= 3,
                );
            }
            if kind == ObjectKind::Clock {
                digital += usize::from(objects::clocks::parameters(&o).digital);
                analog += usize::from(!objects::clocks::parameters(&o).digital);
            }
            let a = objects::build_object(&o);
            let (lo, hi) = a.bounds();
            assert!(
                lo.cmpge(Vec3::new(-size.x * 0.5, 0., -size.z * 0.5) - Vec3::splat(1e-5))
                    .all()
                    && hi
                        .cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(1e-5))
                        .all(),
                "{kind:?}, seed={seed}, size={size:?}: {lo:?}..{hi:?}"
            );
        }
    }
    assert!(saucers > 64 && digital > 32 && analog > 128);
}

#[test]
fn tall_supported_props_must_fit_beneath_the_entire_sloped_roof() {
    let mut scene =
        IndoorManifest::generate_with_humans(202, IndoorLayout::Mixed, 0.6, 0, 0.).unwrap();
    let mut support = scene.objects[0].clone();
    scene.objects.clear();
    let roof = scene.ceiling_height(Vec2::ZERO);
    support.id = 0;
    support.kind = ObjectKind::Cabinet;
    support.size = Vec3::new(1., roof - 0.4, 1.);
    support.position = Vec3::ZERO;
    support.yaw = 0.;
    scene.objects.push(support.clone());
    let mut prop = support.clone();
    prop.id = 1;
    prop.kind = ObjectKind::Plant;
    prop.support = Some(0);
    prop.solid = false;
    prop.position = Vec3::Y * support.size.y;
    prop.size = Vec3::new(0.24, 0.7, 0.24);
    for yaw in [0., 0.3, 1.4, 2.7] {
        prop.yaw = yaw;
        assert!(!scene.prop_clear(&prop, &support, 0.012));
        prop.size.y = 0.1;
        assert!(scene.prop_clear(&prop, &support, 0.012));
        prop.size.y = 0.7;
    }
}

#[test]
fn invalid_base_scene_scalars_fail_before_dependent_program_queries() {
    let scene = IndoorManifest::generate_with_humans(0, IndoorLayout::Mixed, 0.6, 0, 0.).unwrap();
    let mut bright = scene.clone();
    bright.daylight_lux = 150_000.;
    bright
        .program
        .as_mut()
        .unwrap()
        .domain
        .as_mut()
        .unwrap()
        .photometry
        .sun_lux = 150_000.;
    validate_layout(&bright).unwrap();
    for field in 0..8 {
        let mut invalid = scene.clone();
        match field {
            0 => invalid.room_size.x = 0.,
            1 => invalid.room_size.y = f32::NAN,
            2 => invalid.world_yaw = f32::INFINITY,
            3 => invalid.density = f32::NAN,
            4 => invalid.human_density = -1.,
            5 => invalid.daylight_lux = f32::NAN,
            6 => invalid.target_lux = -1.,
            _ => invalid.light_kelvin = f32::INFINITY,
        }
        assert!(validate_layout(&invalid)
            .unwrap_err()
            .contains("invalid scene dimensions"));
    }
}

#[test]
fn geometry_qualification_catches_hidden_mesh_overruns_and_records_coverage() {
    for seed in [202, 1_013_005] {
        let mut scene =
            IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.8, 0, 0.).unwrap();
        let stats = validate_geometry(&scene).unwrap();
        assert_eq!(stats.object_envelopes_checked, scene.objects.len());
        assert_eq!(
            stats.object_envelopes_by_kind.values().sum::<usize>(),
            scene.objects.len()
        );
        assert!(stats.object_envelope_max_overrun_metres <= 0.003);
        // A bad reservation must fail even if the mesh is finite and its
        // topology/normals are valid. Do not resize geometry to conceal it.
        scene.objects[0].kind = ObjectKind::FloorLamp;
        scene.objects[0].size = Vec3::new(0.1, 1.65, 0.1);
        assert!(validate_geometry(&scene)
            .unwrap_err()
            .contains("escapes its reserved envelope"));
        scene.objects[0].size.y = f32::NAN;
        assert!(validate_geometry(&scene)
            .unwrap_err()
            .contains("invalid object geometry dimensions"));
    }
}
