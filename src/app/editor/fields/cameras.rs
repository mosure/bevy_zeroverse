//! Detailed controls kept with their owning policy.
use super::*;
pub(super) fn append(state: &EditorState, v: &mut Vec<Item>) {
    let d = &state.draft;
    if state.view.page == Page::Cameras {
        if !d["indoor_camera"]["multiview"].is_null() {
            v.extend([
                num("@baseline","Camera baseline",0.,1.,0.01,"0 = close stereo; 1 = wider spacing and more varied visible targets. Accounts for camera count; overlap and clearance remain required. Replaces advanced values below."),
                section("ADVANCED RIG","Custom constraints override the baseline program. Impossible requests report an error."),
                choices("@overlap_graph","Visibility connections",&[("Reference camera","reference"),("Connected graph","connected")],"A connected graph allows overlap through intermediate cameras. The same spanning edges must pass along the paths. Explicit overlap strata require Reference camera."),
                num("/indoor_camera/multiview/min_overlap","Minimum proxy overlap",0.,1.,0.01,"Placement estimate; measured rendered overlap may differ."),
                num("/indoor_camera/multiview/min_baseline","Minimum pair spacing (m)",0.01,10.,0.01,"Between every pair of views."),
                num("/indoor_camera/multiview/min_reference_baseline","Minimum distance to view 0 (m)",0.,20.,0.05,"Metric reference baseline."),
                num("/indoor_camera/multiview/max_baseline","Maximum distance to view 0 (m)",0.01,30.,0.05,"Upper bound in room space."),
                num("/indoor_camera/multiview/min_spread","Group spread",0.,1.,0.01,"For 3+ views: discourages a line of cameras."),
                num("/indoor_camera/multiview/trajectory_variation","Independent path variation",0.,1.,0.01,"0 permits rigid shared camera motion; 1 requests varied paths."),
                num("/indoor_camera/multiview/target_variation","View coverage variation",0.,1.,0.01,"0 keeps shared look targets. Higher values sample visible surface regions and prefer distinct, useful viewpoints; selection accounts for camera count."),
                toggle("@mixture","Sample overlap strata","Mix high, low and no-overlap pairs instead of one overlap threshold."),
            ]);
            if !d["indoor_camera"]["overlap_mixture"].is_null() {
                for (i, label) in [
                    "High-overlap weight",
                    "Low-overlap weight",
                    "No-overlap weight",
                ]
                .iter()
                .enumerate()
                {
                    v.push(num(
                        &format!("/indoor_camera/overlap_mixture/weights/{i}"),
                        label,
                        0.,
                        1.,
                        0.01,
                        "Relative sampling weight; all three must not be zero.",
                    ));
                }
            }
        }
        v.extend([
            section("LENSES", "Vertical perspective FOV; wide lenses retain full calibration and annotation support."),
            num("/indoor_camera/intrinsics/fov_y_min_degrees","Minimum vertical FOV",15.,150.,1.,"Lower bound for capture lenses. Horizontal coverage also depends on image aspect."),
            num("/indoor_camera/intrinsics/fov_y_max_degrees","Maximum vertical FOV",15.,150.,1.,"145° includes ultra-wide perspective lenses. Near-plane clearance is checked across the complete camera path."),
            num("@lens_bias","Lens preference · narrow / wide",-1.,1.,0.01,"Continuously shifts the distribution toward narrow (−1) or wide (+1) lenses. Independent of spatial baseline. Editing uses the continuous distribution."),
            num("@lens_spread","Lens spread",0.,1.,0.01,"0 fixes every lens at the median; 1 spans the full FOV range. Continuous PDF in log focal space; placement and overlap can bias accepted lenses."),
            Item::Distribution(super::super::distribution_preview::DistributionKind::Perspective),
            num("@lens_x","Principal-point variation X",0.,0.2,0.005,"Seeded optical-center displacement, as a fraction of image width. Zero keeps the centered pinhole."),
            num("@lens_y","Principal-point variation Y",0.,0.2,0.005,"Seeded optical-center displacement, as a fraction of image height. Full K is exported for each view."),
        ]);
        v.extend([
                section("FISHEYE LENSES","A separate angular FOV distribution. Set probability to zero to preserve the pinhole-only path; changing these controls never regenerates the scene."),
                num("@fisheye_probability","Fisheye camera probability",0.,1.,0.01,"Studio defaults to 50%. Assignment is seeded per camera before visibility planning; lens parameters stay fixed through time."),
                num("/indoor_camera/lenses/fov_degrees/0","Fisheye minimum diameter FOV",40.,220.,1.,"Angular domain; diagonal or circular sensor fit. Not a perspective vertical FOV."),
                num("/indoor_camera/lenses/fov_degrees/1","Fisheye maximum diameter FOV",40.,220.,1.,"Values above 180 degrees include rearward rays."),
                num("/indoor_camera/lenses/fov_bias","Fisheye preference · narrow / wide",-1.,1.,0.01,"Continuously shifts the angular FOV median toward narrow (−1) or wide (+1). Zero is neutral."),
                num("/indoor_camera/lenses/fov_spread","Fisheye FOV spread",0.,1.,0.01,"0 fixes the diameter FOV at the median; 1 spans the configured angular range. Lens distortion and sensor shape still vary independently."),
                Item::Distribution(super::super::distribution_preview::DistributionKind::Fisheye),
                num("/indoor_camera/lenses/circular_probability","Circular image probability",0.,1.,0.01,"Contained circular image with equal focal axes. Circular + elliptical probability must be at most one."),
                num("/indoor_camera/lenses/elliptical_probability","Elliptical image probability",0.,1.,0.01,"Contained ellipse with sampled focal-axis scaling. Remaining probability uses full-frame / diagonal fisheye."),
                num("/indoor_camera/lenses/distortion_strength","Angular distortion variation",0.,1.,0.01,"0 is equidistant; 1 samples a bounded monotonic polynomial family."),
                num("/indoor_camera/lenses/anisotropy/0","Fisheye minimum aspect scaling",0.8,1.25,0.01,"Sensor focal-axis anisotropy, independent of image aspect."),
                num("/indoor_camera/lenses/anisotropy/1","Fisheye maximum aspect scaling",0.8,1.25,0.01,"Full pixel calibration is exported."),
            ]);
        v.extend([
            section("CAMERA MOTION","Independent of furniture/human animation and camera spacing. Intrinsics remain fixed through time."),
            choices("@path_preset","Trajectory preset",&[("Custom","custom"),("Short handheld","handheld"),("Explore the room","explore")],"Presets set travel bounds; all fields remain editable."),
            num("/indoor_camera/path_length_min","Minimum travel (m)",0.,20.,0.05,"Minimum path length for moving cameras. Use Moving cameras = 0% to freeze poses without losing travel settings."),
            num("/indoor_camera/path_length_max","Maximum travel (m)",0.,30.,0.05,"Swept collision checks always apply."),
            num("/indoor_camera/long_path_fraction","Long-route probability",0.,1.,0.01,"Fraction of proposals that explore farther through the room."),
            toggle("/indoor_camera/primary_room","Stay in primary room","Bounds capture trajectories to the reconstruction target."),
            toggle("@handheld","Independent handheld increments","Translation and yaw/pitch/roll sampled independently from the shared look target."),
            text("/indoor_camera/duration_seconds","Physical duration (seconds)","Empty leaves physical time unspecified. Positive number maps normalized progress to seconds."),
        ]);
        if !d["indoor_camera"]["handheld"].is_null() {
            for (key, units, names, bound) in [
                ("translation_m", "m", ["Right", "Up", "Forward"], 10.),
                ("rotation_degrees", "deg", ["Yaw", "Pitch", "Roll"], 90.),
            ] {
                for (i, name) in names.iter().enumerate() {
                    for (j, endpoint) in ["min", "max"].iter().enumerate() {
                        v.push(num(
                            &format!("/indoor_camera/handheld/{key}/{i}/{j}"),
                            &format!("{name} {endpoint} ({units})"),
                            -bound,
                            bound,
                            0.01,
                            "Signed endpoint increment. Minimum must not exceed maximum.",
                        ));
                    }
                }
            }
            v.push(num(
                "/indoor_camera/handheld/reverse_probability",
                "Reverse path probability",
                0.,
                1.,
                0.01,
                "Reverses positions and orientations together.",
            ));
        }
    }
}
