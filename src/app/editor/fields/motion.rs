//! Population, model-driven human behavior, and mechanical motion stay separate.
use super::*;

pub(super) fn append(state: &EditorState, v: &mut Vec<Item>) {
    if state.view.page != Page::People || state.draft["scene_type"] != "ProceduralIndoor" {
        return;
    }
    human(state, v);
    furnishings(state, v);
}

fn human(state: &EditorState, v: &mut Vec<Item>) {
    v.extend([
        section(
            "people.human-motion",
            "Human motion",
            "Optional ARDY trajectories. Static people remain available.",
        ),
        toggle(
            "@motion",
            "Generate human motion",
            "Cached models load only after an enabled policy is applied.",
        ),
    ]);
    if !cfg!(feature = "human_motion") {
        v.push(Item::Note("Unavailable in this build. Enable --features human_motion to use ARDY. Imported policies remain available in Capture & tools.".into()));
        return;
    }
    if state.draft["human_motion"].is_null() {
        return;
    }
    for (name, key, max, step) in [
        ("Moving fraction", "fraction", 1., 0.01),
        ("Navigate the room", "locomotion_fraction", 1., 0.01),
        ("Walk / action / resume", "sequence_fraction", 1., 0.01),
        ("Energetic travel", "energetic_fraction", 1., 0.01),
        ("Maximum moving people", "max_actors", 16., 1.),
    ] {
        v.push(num(
            &format!("/human_motion/{key}"),
            name,
            if key == "max_actors" { 1. } else { 0. },
            max,
            step,
            "Applies on generation; collision rejection can retain a person as static.",
        ));
    }
    v.push(toggle(
        "/human_motion/dense_trajectory",
        "Constrain complete navigation path",
        "Waypoints are planned after furniture placement.",
    ));
    v.push(section(
        "people.model",
        "Motion model & quality",
        "Inference quality, clip duration and batching. Edits never initialize models.",
    ));
    for (name, key, min, max, step) in [
        ("Clip frames (20 Hz)", "frames", 40., 640., 4.),
        ("Diffusion steps", "diffusion_steps", 1., 10., 1.),
        ("History frames", "history_frames", 0., 160., 4.),
        ("Inference batch size", "batch_size", 1., 8., 1.),
        ("Attempts per person", "max_attempts", 1., 3., 1.),
        ("Text guidance", "text_guidance", 0., 10., 0.1),
        ("Trajectory guidance", "trajectory_guidance", 0., 10., 0.1),
    ] {
        v.push(num(
            &format!("/human_motion/{key}"),
            name,
            min,
            max,
            step,
            "Applies on generation; collision rejection can retain a person as static.",
        ));
    }
    v.push(toggle(
        "/human_motion/strict",
        "Fail capture on rejected motion",
        "Otherwise the requested actor stays static.",
    ));
    v.push(Item::Action(
        "Play motion at model speed",
        super::super::Action::ModelSpeed,
    ));
    v.push(section(
        "people.prompts",
        "Behavior & prompt sampling",
        "Relative weights; at least one family must be nonzero.",
    ));
    for (name, key) in [
        ("Travel / chair transitions", "locomotion"),
        ("Hand gestures", "gesture"),
        ("Exercise", "exercise"),
        ("Dance", "dance"),
        ("Floor actions", "floor"),
        ("Idle / looking", "idle"),
    ] {
        v.push(num(
            &format!("/human_motion/prompt_sampling/{key}"),
            name,
            0.,
            10.,
            0.1,
            "Continuous procedural prompt sampling weight.",
        ));
    }
    v.extend([
        num(
            "/human_motion/prompt_sampling/style_fraction",
            "Posture & gaze variation",
            0.,
            1.,
            0.01,
            "Fraction of prompts with an additional style cue.",
        ),
        num(
            "/human_motion/prompt_sampling/max_sequence_actions",
            "Maximum action stops",
            1.,
            2.,
            1.,
            "Stops during compound walking sequences.",
        ),
        section(
            "people.custom-motion",
            "Custom human trajectories",
            "Override sampled actions with explicit actors, prompts and waypoints.",
        ),
        json_field(
            "/human_motion/trajectories",
            "Actor trajectories JSON",
            "Array of actor_id, prompt and frame-indexed waypoints; [] uses sampled prompts.",
        ),
    ]);
}

fn furnishings(state: &EditorState, v: &mut Vec<Item>) {
    v.extend([
        section(
            "people.furnishings",
            "Furnishing motion",
            "Chairs, doors, coverings, fans and portable items. Independent of human models.",
        ),
        toggle(
            "@indoor_motion",
            "Animate furnishings",
            "Apply to generate clearance-checked motion; no models are loaded.",
        ),
    ]);
    if state.draft["indoor_motion"].is_null() {
        return;
    }
    v.push(num(
        "/indoor_motion/max_mechanisms",
        "Mechanism limit",
        1.,
        128.,
        1.,
        "Bounds preparation and animation cost.",
    ));
    v.push(section(
        "people.motion-selection",
        "Which furnishings move",
        "Fractions of eligible objects. Occupied or fully blocked objects remain static.",
    ));
    for (key, label) in [
        ("chair_fraction", "Moving chairs"),
        ("door_fraction", "Moving doors"),
        ("cabinet_fraction", "Moving cabinets"),
        ("covering_fraction", "Moving blinds & curtains"),
        ("fan_fraction", "Spinning fans"),
        ("appliance_fraction", "Opening appliances"),
        ("prop_fraction", "Moving portable items"),
        ("cart_fraction", "Moving carts"),
    ] {
        v.push(num(
            &format!("/indoor_motion/{key}"),
            label,
            0.,
            1.,
            0.01,
            "Selection probability; constrained motions shorten or swivel.",
        ));
    }
    v.push(section(
        "people.motion-travel",
        "Travel & rotation limits",
        "Maximum movement. Clearance and support checks may shorten it.",
    ));
    for (key, label, max, step) in [
        ("chair_travel_m", "Chair travel (m)", 1.5, 0.05),
        ("chair_swivel_degrees", "Chair swivel (degrees)", 120., 5.),
        ("prop_travel_m", "Portable item travel (m)", 0.75, 0.01),
        (
            "prop_rotation_degrees",
            "Portable item rotation (degrees)",
            180.,
            5.,
        ),
        ("cart_travel_m", "Cart travel (m)", 2., 0.05),
    ] {
        v.push(num(&format!("/indoor_motion/{key}"), label, 0., max, step,
            "Shorter paths and other headings are tried when blocked. Connected cables stay attached."));
    }
    v.push(section(
        "people.fans",
        "Ceiling-fan speed",
        "Revolutions across the shared timeline; playback speed determines real-time speed.",
    ));
    for (index, label) in [(0, "Minimum turns"), (1, "Maximum turns")] {
        v.push(num(
            &format!("/indoor_motion/fan_turns/{index}"),
            label,
            0.,
            120.,
            0.25,
            "Minimum must not exceed maximum; zero maximum disables spinning.",
        ));
    }
}
