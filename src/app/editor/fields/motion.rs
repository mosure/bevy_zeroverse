//! Detailed controls kept with their owning policy.
use super::*;
pub(super) fn append(state: &EditorState, v: &mut Vec<Item>) {
    let d = &state.draft;
    let indoor = d["scene_type"] == "ProceduralIndoor";
    if indoor && state.view.page == Page::People {
        v.push(section(
            "ROOM MOTION",
            "Optional deterministic chair, door, cabinet, covering and ceiling-fan motion.",
        ));
        v.push(toggle(
            "@indoor_motion",
            "Animate furnishings",
            "Apply to generate clearance-checked motion; no models are loaded.",
        ));
        if !d["indoor_motion"].is_null() {
            for (key, label) in [
                ("chair_fraction", "Moving chair fraction"),
                ("door_fraction", "Moving door fraction"),
                ("cabinet_fraction", "Moving cabinet fraction"),
                ("covering_fraction", "Moving covering fraction"),
                ("fan_fraction", "Spinning fan fraction"),
                ("appliance_fraction", "Opening appliance fraction"),
                ("prop_fraction", "Moving portable item fraction"),
                ("cart_fraction", "Moving cart fraction"),
            ] {
                v.push(num(
                    &format!("/indoor_motion/{key}"),
                    label,
                    0.,
                    1.,
                    0.01,
                    "Selects eligible mechanisms. Constrained motions shorten or swivel; occupied or fully blocked objects remain static.",
                ));
            }
            v.push(num(
                "/indoor_motion/chair_travel_m",
                "Chair travel (m)",
                0.,
                1.5,
                0.05,
                "Maximum chair rolling/sliding travel; shorter paths and other headings are tried when blocked.",
            ));
            v.push(num(
                "/indoor_motion/chair_swivel_degrees",
                "Chair swivel (degrees)",
                0.,
                120.,
                5.,
                "Maximum chair turn or upper-seat swivel; allows rotation-only motion in tight spaces.",
            ));
            v.push(num(
                "/indoor_motion/max_mechanisms",
                "Mechanism limit",
                1.,
                128.,
                1.,
                "Bounds preparation and animation cost.",
            ));
            for (key, label, maximum, step) in [
                ("prop_travel_m", "Portable item travel (m)", 0.75, 0.01),
                (
                    "prop_rotation_degrees",
                    "Portable item rotation (degrees)",
                    180.,
                    5.,
                ),
                ("cart_travel_m", "Cart travel (m)", 2., 0.05),
            ] {
                v.push(num(&format!("/indoor_motion/{key}"), label, 0., maximum, step,
                    "Upper bound; clearance and support checks shorten blocked motion. Connected cables stay attached."));
            }
            for (index, label) in [(0, "Fan turns minimum"), (1, "Fan turns maximum")] {
                v.push(num(&format!("/indoor_motion/fan_turns/{index}"),label,
                    0.,120.,0.25,"Revolutions per shared timeline. Minimum must not exceed maximum; zero maximum disables spinning."));
            }
        }
    }
    if state.view.page == Page::People && !cfg!(feature = "human_motion") {
        v.push(Item::Note("Human motion is unavailable in this build. Enable --features human_motion to use ARDY. Imported policies remain available in Advanced.".into()));
    }
    if state.view.page == Page::People
        && cfg!(feature = "human_motion")
        && !d["human_motion"].is_null()
    {
        v.push(Item::Note("Models download once through burn_human loaders; configuration edits do not initialize inference.".into()));
        for (name, key, min, max, step) in [
            ("Moving fraction", "fraction", 0., 1., 0.01),
            ("Navigate the room", "locomotion_fraction", 0., 1., 0.01),
            ("Walk / action / resume", "sequence_fraction", 0., 1., 0.01),
            ("Energetic travel", "energetic_fraction", 0., 1., 0.01),
            ("Maximum moving people", "max_actors", 1., 16., 1.),
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
        v.extend([
            toggle(
                "/human_motion/dense_trajectory",
                "Constrain complete navigation path",
                "Waypoints are planned after furniture placement.",
            ),
            toggle(
                "/human_motion/strict",
                "Fail capture on rejected motion",
                "Otherwise the requested actor stays static.",
            ),
            section(
                "PROMPT DISTRIBUTION",
                "Relative weights; at least one family must be nonzero.",
            ),
        ]);
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
        v.push(num(
            "/human_motion/prompt_sampling/style_fraction",
            "Posture & gaze variation",
            0.,
            1.,
            0.01,
            "Fraction of prompts with an additional style cue.",
        ));
        v.push(num(
            "/human_motion/prompt_sampling/max_sequence_actions",
            "Maximum action stops",
            1.,
            2.,
            1.,
            "Stops during compound walking sequences.",
        ));
        v.push(json_field(
            "/human_motion/trajectories",
            "Custom actor trajectories",
            "Array of actor_id, prompt and frame-indexed waypoints; [] uses sampled prompts.",
        ));
    }
}
