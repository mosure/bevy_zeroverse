use super::*;
use crate::scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest};
use bevy::prelude::*;

fn config() -> BevyZeroverseConfig {
    BevyZeroverseConfig {
        scene_type: ZeroverseSceneType::ProceduralIndoor,
        width: 256.,
        height: 256.,
        num_cameras: 4,
        headless: true,
        ..default()
    }
}
fn request_fixture() -> CapacityRequest {
    CapacityRequest {
        budget: AuditBudget {
            manifests: 16,
            mesh_checks: 0,
            rendered_rooms: 0,
        },
        ..Default::default()
    }
}
#[test]
fn plan_is_seed_independent_but_configuration_and_capture_specific() {
    let a = config();
    let mut b = a.clone();
    b.indoor_seed = Some(99);
    b.gizmos = !a.gizmos;
    b.editor = !a.editor;
    let request = request_fixture();
    let pa = a.capacity_plan(&request, default_capture()).unwrap();
    assert_eq!(pa, b.capacity_plan(&request, default_capture()).unwrap());
    b.indoor_camera = Some("{\"motion_fraction\":1.0}".into());
    assert_eq!(pa, b.capacity_plan(&request, default_capture()).unwrap());
    b.indoor_camera = Some("{\"motion_fraction\":0.0}".into());
    let pb = b.capacity_plan(&request, default_capture()).unwrap();
    assert_ne!(pa.population.config_sha256, pb.population.config_sha256);
    assert_eq!(
        pa.factors
            .iter()
            .find(|f| f.id == "camera_motion")
            .unwrap()
            .state,
        CapturedFactorState::AdmissionOnly
    );
    assert_eq!(
        pb.factors
            .iter()
            .find(|f| f.id == "camera_motion")
            .unwrap()
            .state,
        CapturedFactorState::Inactive
    );
    assert_eq!(
        pa.factors
            .iter()
            .find(|f| f.id == "human_motion")
            .unwrap()
            .state,
        CapturedFactorState::Inactive
    );
    b.indoor_camera = None;
    b.annotation_glass = crate::render::glass::AnnotationGlass::Through;
    assert!(b.capacity_plan(&request, default_capture()).is_err());
}
#[test]
fn strict_schedule_and_wire_validation() {
    let request = request_fixture();
    let config = config();
    let mut capture = default_capture();
    capture.progress = vec![0., 0.];
    assert!(config.capacity_plan(&request, capture).is_err());
    let mut request = request;
    request.confidence = f64::NAN;
    assert!(config.capacity_plan(&request, default_capture()).is_err());
    request.confidence = 0.95;
    request.unit = SampleUnit::DressedScene;
    let plan = config.capacity_plan(&request, default_capture()).unwrap();
    assert_eq!(
        plan.factors
            .iter()
            .find(|f| f.id == "camera_initial")
            .unwrap()
            .state,
        CapturedFactorState::AdmissionOnly
    );
    request.unit = SampleUnit::TemporalClip;
    assert!(config.capacity_plan(&request, default_capture()).is_err());
    let mut wire = serde_json::to_value(request_fixture()).unwrap();
    wire["guess_capacity"] = json!(true);
    assert!(serde_json::from_value::<CapacityRequest>(wire).is_err());
}
#[test]
fn descriptor_is_joint_seed_free_and_intervention_sensitive() {
    let mut config = config();
    config.indoor_human_density = 0.;
    config.indoor_layout = IndoorLayout::Conference;
    let request = CapacityRequest {
        seed_policy: SeedPolicy::Fixed { seed: 612 },
        ..request_fixture()
    };
    let plan = config.capacity_plan(&request, default_capture()).unwrap();
    let original = generate(&plan, 0).unwrap();
    let observation = observe(&original, &plan, 0, false).unwrap();
    let SampleOutcome::Accepted {
        descriptors,
        family_sha256,
        ..
    } = observation.outcome
    else {
        panic!()
    };
    let base = &descriptors[&request.definitions[0].id];
    let mut appearance = original.clone();
    appearance.appearance = None;
    appearance
        .apply_appearance(AppearanceSettings {
            material_detail: 0.25,
            illumination_scale: 0.1,
            ..default()
        })
        .unwrap();
    let SampleOutcome::Accepted {
        descriptors: other,
        family_sha256: other_family,
        ..
    } = observe(&appearance, &plan, 0, false).unwrap().outcome
    else {
        panic!()
    };
    let other = &other[&request.definitions[0].id];
    assert_eq!(family_sha256, other_family);
    assert_eq!(base.blocks["architecture"], other.blocks["architecture"]);
    assert_eq!(base.blocks["furniture"], other.blocks["furniture"]);
    assert_ne!(base.blocks["lighting"], other.blocks["lighting"]);
    assert_ne!(base.blocks["materials"], other.blocks["materials"]);
    assert_ne!(base.class_sha256, other.class_sha256);
    // Recipe seed changes alone are not a descriptor dimension. Evaluated
    // microstructure requires rendered evidence, not hidden seed entropy.
    assert_eq!(
        descriptor::canonical(json!({"id":1,"seed":42,"color":[0.2,0.3,0.4]}), 0.05).unwrap(),
        descriptor::canonical(json!({"id":999,"seed":1000,"color":[0.2,0.3,0.4]}), 0.05).unwrap()
    );
    assert!(serde_json::to_string(&observation.seed).is_ok());
}
#[test]
fn static_motion_does_not_multiply_captured_states() {
    let mut config = config();
    config.indoor_human_density = 0.;
    config.indoor_camera = Some("{\"motion_fraction\":0.0}".into());
    let request = CapacityRequest {
        seed_policy: SeedPolicy::Fixed { seed: 40 },
        ..request_fixture()
    };
    let plan = config.capacity_plan(&request, default_capture()).unwrap();
    let scene = generate(&plan, 0).unwrap();
    let a = observe(&scene, &plan, 0, false).unwrap();
    let mut repeated = plan.clone();
    repeated.population.capture.progress = vec![0., 0.5, 1.];
    let b = observe(&scene, &repeated, 0, false).unwrap();
    // Repeated identical captured transforms are canonicalized as one state.
    let (
        SampleOutcome::Accepted { descriptors: a, .. },
        SampleOutcome::Accepted { descriptors: b, .. },
    ) = (a.outcome, b.outcome)
    else {
        panic!()
    };
    assert_eq!(a, b);
}
#[test]
fn manifest_adapter_matches_production_constructor() {
    let config = config();
    let request = CapacityRequest {
        seed_policy: SeedPolicy::Fixed { seed: 14 },
        ..request_fixture()
    };
    let plan = config.capacity_plan(&request, default_capture()).unwrap();
    let scene = generate(&plan, 0).unwrap();
    let mut reference = IndoorManifest::generate_with_humans(
        14,
        config.indoor_layout,
        config.indoor_density,
        0,
        config.indoor_human_density,
    )
    .unwrap();
    reference.apply_appearance(default()).unwrap();
    reference.resample_cameras(4, default(), 1.).unwrap();
    assert_eq!(scene, reference);
}

fn rendered_fixture(progress: Vec<f32>) -> (CapacityPlan, crate::sample::Sample) {
    use crate::{
        render::color::ColorEncoding,
        sample::{Sample, View},
    };
    use bevy_zeroverse_capture::calibration::CameraCalibration;
    let mut config = config();
    config.width = 4.;
    config.height = 4.;
    config.num_cameras = 1;
    config.indoor_human_density = 0.;
    config.indoor_camera = Some("{\"motion_fraction\":0.0}".into());
    let request = CapacityRequest {
        seed_policy: SeedPolicy::Fixed { seed: 55 },
        budget: AuditBudget {
            manifests: 1,
            rendered_rooms: 1,
            mesh_checks: 0,
        },
        definitions: vec![DiversityDefinition {
            id: "pixel-grid".into(),
            descriptor: DescriptorKind::RenderedGrid,
            grid_side: 2,
            ..CapacityRequest::default().definitions.remove(0)
        }],
        ..Default::default()
    };
    let plan = config
        .capacity_plan(
            &request,
            CapturePlan {
                image_size: [4, 4],
                cameras: 1,
                progress: progress.clone(),
                modes: ["color", "normal", "position", "semantic"]
                    .map(String::from)
                    .into(),
                ..default_capture()
            },
        )
        .unwrap();
    let scene = generate(&plan, 0).unwrap();
    let (lo, hi) = scene.primary_region().shell_bounds();
    let camera = &scene.cameras[0];
    let point = (camera.transform_at(0.).translation - lo) / (hi - lo);
    let plane = |pixel: [f32; 4]| {
        (0..16)
            .flat_map(|_| pixel.into_iter().flat_map(f32::to_ne_bytes))
            .collect::<Vec<u8>>()
    };
    let mut k =
        CameraCalibration::centered_pinhole(4, 4, camera.fov_degrees.to_radians(), 1.).unwrap();
    k.k[0][2] += camera.principal_point_offset.unwrap_or_default().x * 4.;
    k.k[1][2] += camera.principal_point_offset.unwrap_or_default().y * 4.;
    let views = progress
        .into_iter()
        .map(|t| View {
            calibration: Some(k.clone()),
            trajectory_progress: Some(t),
            world_from_view: camera.transform_at(t).to_matrix().to_cols_array_2d(),
            color: plane([0.2, 0.3, 0.4, 1.]),
            normal: plane([0.5, 1., 0.5, 1.]),
            semantic: plane([0.1, 0.2, 0.3, 1.]),
            position: plane([point.x, point.y, point.z, 1.]),
            ..default()
        })
        .collect();
    let sample = Sample {
        views,
        view_dim: 1,
        aabb: [lo.to_array(), hi.to_array()],
        color_encoding: ColorEncoding::TonemappedLinear,
        indoor_render_metadata: Some(
            json!({"build_provenance":{"source_sha256":plan.population.generator.source_sha256,"capture_engine":plan.population.capture_engine}}),
        ),
        indoor: Some(scene),
        ..default()
    };
    (plan, sample)
}
#[test]
fn completed_render_contract_rejects_incomplete_stale_and_wrong_camera_delivery() {
    let (plan, sample) = rendered_fixture(vec![0.]);
    let observe = |s: &crate::sample::Sample| observe_rendered(s, &plan, 0, false);
    assert!(observe(&sample).is_ok());
    let mut broken = sample.clone();
    broken.views.clear();
    assert!(observe(&broken).is_err());
    let mut broken = sample.clone();
    broken.indoor.as_mut().unwrap().cameras.clear();
    assert!(observe(&broken).is_err());
    let mut broken = sample.clone();
    broken.views[0].normal.pop();
    assert!(observe(&broken).is_err());
    let mut broken = sample.clone();
    broken.views[0].world_from_view[0][0] = f32::NAN;
    assert!(observe(&broken).is_err());
    let mut broken = sample.clone();
    broken.views[0].calibration.as_mut().unwrap().k[0][0] *= 1.3;
    assert!(observe(&broken).is_err());
    let mut broken = sample.clone();
    broken.views[0].color[..4].copy_from_slice(&f32::NAN.to_ne_bytes());
    assert!(observe(&broken).is_err());
    let mut broken = sample.clone();
    broken.indoor_render_metadata.as_mut().unwrap()["build_provenance"]["source_sha256"] =
        json!("old");
    assert!(observe(&broken).is_err());
    let mut timed = plan;
    timed.population.capture.seconds = Some(vec![1.]);
    assert!(observe_rendered(&sample, &timed, 0, false).is_err());
}
#[test]
fn repeated_static_renders_have_one_captured_state_and_appearance_changes_are_observed() {
    let (one, sample) = rendered_fixture(vec![0.]);
    let (many, repeat) = rendered_fixture(vec![0., 0.5, 1.]);
    let descriptors =
        |s: &crate::sample::Sample, p: &CapacityPlan| match observe_rendered(s, p, 0, false)
            .unwrap()
            .outcome
        {
            SampleOutcome::Accepted { descriptors, .. } => descriptors,
            _ => panic!(),
        };
    assert_eq!(descriptors(&sample, &one), descriptors(&repeat, &many));
    let mut changed = sample.clone();
    for v in changed.views[0].color.as_chunks_mut::<16>().0.iter_mut() {
        v[..4].copy_from_slice(&0.8_f32.to_ne_bytes());
    }
    assert_ne!(descriptors(&sample, &one), descriptors(&changed, &one));
}
