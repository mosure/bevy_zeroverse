//! Independent ceiling controls, displayed alongside the room camera policy.
use super::*;
pub(super) fn append(v: &mut Vec<Item>) {
    v.extend([
        section("cameras.mount", "Ceiling mix & mounting", "Each view samples its mounting type before placement. Fractions describe a dataset probability, not an exact count in a small rig."),
        num("/indoor_camera/ceiling_mount/probability", "Ceiling camera probability", 0., 1., 0.01, "0 = all room cameras; 1 = all ceiling cameras. Intermediate values mix both. Lens and motion roles are sampled independently."),
        num("/indoor_camera/ceiling_mount/standoff_m/0", "Minimum ceiling standoff (m)", 0.08, 1., 0.01, "Optical-center distance from the ceiling. Must exceed twice the body radius plus 5 mm."),
        num("/indoor_camera/ceiling_mount/standoff_m/1", "Maximum ceiling standoff (m)", 0.08, 1., 0.01, "Sampled before placement retries."),
        num("/indoor_camera/ceiling_mount/body_radius_m", "Camera body radius (m)", 0.02, 0.20, 0.005, "Physical clearance reservation. No body is rendered into its own capture."),
        num("/indoor_camera/ceiling_mount/wall_clearance_m", "Wall clearance (m)", 0.20, 2., 0.01, "Avoid walls, fixtures, suspension, ducts and furniture."),
        section("cameras.mount-motion", "Ceiling trajectories", "Ceiling rails translate below one supporting plane, looking perpendicular to it. Swept clearance and shared visibility remain required throughout the path."),
        num("/indoor_camera/ceiling_mount/motion_fraction", "Moving ceiling camera fraction", 0., 1., 0.01, "0 fixes mounted cameras; 1 requests rail motion for every mounted camera. Independent of room, human and furnishing motion."),
        num("/indoor_camera/ceiling_mount/travel_m/0", "Minimum ceiling travel (m)", 0.01, 20., 0.05, "Positive metric length. An impossible request reports a placement error, never silently freezes the camera."),
        num("/indoor_camera/ceiling_mount/travel_m/1", "Maximum ceiling travel (m)", 0.01, 30., 0.05, "Log-uniform travel distribution; rails cannot cross changes in supporting ceiling plane."),
        section("cameras.mount-lenses", "Ceiling lens distribution", "Separate from room lenses. Each mounted camera can be fisheye or perspective; full calibration is exported."),
    ]);
    for (key, label, lo, hi, step) in [
        (
            "lenses/probability",
            "Ceiling fisheye probability",
            0.,
            1.,
            0.01,
        ),
        (
            "lenses/fov_degrees/0",
            "Fisheye minimum diameter FOV",
            40.,
            220.,
            1.,
        ),
        (
            "lenses/fov_degrees/1",
            "Fisheye maximum diameter FOV",
            40.,
            220.,
            1.,
        ),
        (
            "lenses/fov_bias",
            "Fisheye preference · narrow / wide",
            -1.,
            1.,
            0.01,
        ),
        ("lenses/fov_spread", "Fisheye FOV spread", 0., 1., 0.01),
        (
            "intrinsics/fov_y_min_degrees",
            "Perspective minimum vertical FOV",
            15.,
            150.,
            1.,
        ),
        (
            "intrinsics/fov_y_max_degrees",
            "Perspective maximum vertical FOV",
            15.,
            150.,
            1.,
        ),
        (
            "intrinsics/bias",
            "Perspective preference · narrow / wide",
            -1.,
            1.,
            0.01,
        ),
        ("intrinsics/spread", "Perspective FOV spread", 0., 1., 0.01),
        (
            "lenses/circular_probability",
            "Circular image probability",
            0.,
            1.,
            0.01,
        ),
        (
            "lenses/elliptical_probability",
            "Elliptical image probability",
            0.,
            1.,
            0.01,
        ),
        (
            "lenses/distortion_strength",
            "Angular distortion variation",
            0.,
            1.,
            0.01,
        ),
        (
            "lenses/anisotropy/0",
            "Minimum aspect scaling",
            0.8,
            1.25,
            0.01,
        ),
        (
            "lenses/anisotropy/1",
            "Maximum aspect scaling",
            0.8,
            1.25,
            0.01,
        ),
    ] {
        v.push(num(&format!("/indoor_camera/ceiling_mount/{key}"), label, lo, hi, step, "Mounted-camera prior. Placement retries preserve the selected lens. Circular and elliptical probabilities must sum to at most one."));
    }
}
