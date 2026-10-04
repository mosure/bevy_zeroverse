//! Seeded motion grammar. Text choices carry timing, clearance and conditioning
//! metadata; stylistic variety never adds a contradictory action or missing prop.
mod catalog;
mod sequence;
#[cfg(test)]
mod tests;

use super::HumanMotionConfig;
use burn_human_motion::Waypoint;
use rand::Rng;
pub(super) use sequence::walking_sequence;
use serde::{Deserialize, Serialize};

pub const PROMPT_PROGRAM_VERSION: u32 = 3;
const STANDING_PELVIS: f32 = 0.94;
const MAX_PROMPT_BYTES: usize = 230;
const MAX_PROMPT_WORDS: usize = 36;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Family {
    Locomotion,
    Gesture,
    Exercise,
    Dance,
    Floor,
    Idle,
}
impl Family {
    const ALL: [Self; 6] = [
        Self::Locomotion,
        Self::Gesture,
        Self::Exercise,
        Self::Dance,
        Self::Floor,
        Self::Idle,
    ];
    pub fn name(self) -> &'static str {
        match self {
            Self::Locomotion => "locomotion",
            Self::Gesture => "gesture",
            Self::Exercise => "exercise",
            Self::Dance => "dance",
            Self::Floor => "floor",
            Self::Idle => "idle",
        }
    }
}

/// Relative proposal weights, not guaranteed admission quotas. Zero excludes a
/// family, including actions from that family inside compound sequences.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PromptSamplingConfig {
    pub locomotion: f32,
    pub gesture: f32,
    pub exercise: f32,
    pub dance: f32,
    pub floor: f32,
    pub idle: f32,
    pub style_fraction: f32,
    pub max_sequence_actions: usize,
}
impl Default for PromptSamplingConfig {
    fn default() -> Self {
        Self {
            locomotion: 4.0,
            gesture: 1.0,
            exercise: 0.8,
            dance: 0.5,
            floor: 0.35,
            idle: 0.5,
            style_fraction: 0.35,
            max_sequence_actions: 2,
        }
    }
}
impl PromptSamplingConfig {
    pub(super) fn weight(&self, family: Family) -> f32 {
        match family {
            Family::Locomotion => self.locomotion,
            Family::Gesture => self.gesture,
            Family::Exercise => self.exercise,
            Family::Dance => self.dance,
            Family::Floor => self.floor,
            Family::Idle => self.idle,
        }
    }
    pub fn validate(&self) -> Result<(), String> {
        let weights = Family::ALL.map(|f| self.weight(f));
        if weights
            .iter()
            .any(|w| !w.is_finite() || !(0.0..=100.0).contains(w))
            || weights.iter().sum::<f32>() <= 0.0
            || !(0.0..=1.0).contains(&self.style_fraction)
            || !(1..=2).contains(&self.max_sequence_actions)
        {
            return Err("prompt_sampling requires nonnegative finite weights (at least one positive), style_fraction in [0,1], and max_sequence_actions in 1..=2".into());
        }
        Ok(())
    }
    pub(super) fn sample_family(&self, rng: &mut impl Rng) -> Family {
        weighted_family(self, &Family::ALL, rng).expect("validated positive family weights")
    }
}
fn weighted_family(
    config: &PromptSamplingConfig,
    families: &[Family],
    rng: &mut impl Rng,
) -> Option<Family> {
    let total: f32 = families.iter().map(|&f| config.weight(f)).sum();
    if total <= 0.0 {
        return None;
    }
    let mut draw = rng.random_range(0.0..total);
    for &f in families {
        draw -= config.weight(f);
        if draw < 0.0 {
            return Some(f);
        }
    }
    families
        .iter()
        .rev()
        .copied()
        .find(|&f| config.weight(f) > 0.0)
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PromptRecipe {
    pub version: u32,
    pub family: Family,
    pub gait: Option<String>,
    pub travel_speed_mps: Option<f32>,
    pub style: Option<String>,
    pub arm_style: Option<String>,
    pub gaze: Option<String>,
    pub actions: Vec<ActionPhase>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub navigation: Option<super::NavigationRecipe>,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ActionPhase {
    pub family: Family,
    pub action: String,
    pub hand: String,
    pub side: String,
    pub repetitions: usize,
    pub amplitude: f32,
    pub start_frame: usize,
    pub end_frame: usize,
    pub peak_pelvis_height: f32,
    pub clearance_radius: f32,
}

pub(super) struct Draft {
    pub behavior: String,
    pub text: String,
    pub recipe: PromptRecipe,
    pub radius: f32,
    pub headroom: f32,
}
pub(super) struct SampledAction {
    spec: &'static catalog::Action,
    phrase: String,
    phase: ActionPhase,
    frames: usize,
}

fn choose<'a>(items: &'a [&str], rng: &mut impl Rng) -> &'a str {
    items[rng.random_range(0..items.len())]
}
fn recipe(family: Family) -> PromptRecipe {
    PromptRecipe {
        version: PROMPT_PROGRAM_VERSION,
        family,
        gait: None,
        travel_speed_mps: None,
        style: None,
        arm_style: None,
        gaze: None,
        actions: Vec::new(),
        navigation: None,
    }
}
fn fits(text: &str) -> bool {
    text.len() <= MAX_PROMPT_BYTES && text.split_whitespace().count() <= MAX_PROMPT_WORDS
}
fn finish(
    core: String,
    recipe: &mut PromptRecipe,
    config: &PromptSamplingConfig,
    allow_gaze: bool,
    rng: &mut impl Rng,
) -> String {
    let mut text = core;
    // A single optional modifier keeps conditioning focused on the action.
    // Previously a clip could request a gait, folded arms, stiff posture and a
    // gaze change simultaneously, often contradicting the timed action stops.
    if rng.random_bool(config.style_fraction as f64) {
        let choice = rng.random_range(0..if allow_gaze { 3 } else { 1 });
        let (suffix, field) = match choice {
            1 => (
                choose(
                    &[
                        "with relaxed arm swings",
                        "with small arm swings",
                        "with one hand at the waist",
                    ],
                    rng,
                ),
                &mut recipe.arm_style,
            ),
            2 => (
                choose(&["looking ahead", "glancing left", "glancing right"], rng),
                &mut recipe.gaze,
            ),
            _ => (
                choose(&["carefully", "with relaxed shoulders", "smoothly"], rng),
                &mut recipe.style,
            ),
        };
        if fits(&format!("{text} {suffix}.")) {
            text.push(' ');
            text.push_str(suffix);
            *field = Some(suffix.into());
        }
    }
    text.push('.');
    debug_assert!(
        fits(&text),
        "generated prompt exceeds grammar budget: {text}"
    );
    text
}

pub(super) fn sample_action(
    config: &PromptSamplingConfig,
    family: Option<Family>,
    frames: usize,
    headroom: f32,
    seated: bool,
    rng: &mut impl Rng,
) -> Option<SampledAction> {
    let candidates: Vec<_> = catalog::ACTIONS
        .iter()
        .filter(|a| {
            family.is_none_or(|f| f == a.family)
                && config.weight(a.family) > 0.0
                && a.minimum_frames <= frames
                && a.headroom <= headroom
                && (!seated || a.seated)
        })
        .collect();
    let families: Vec<_> = Family::ALL
        .into_iter()
        .filter(|&f| candidates.iter().any(|a| a.family == f))
        .collect();
    let family = weighted_family(config, &families, rng)?;
    let actions: Vec<_> = candidates
        .into_iter()
        .filter(|a| a.family == family)
        .collect();
    let spec = actions[rng.random_range(0..actions.len())];
    let count = if spec.repeatable {
        rng.random_range(1..=(frames / spec.minimum_frames).min(3))
    } else {
        1
    };
    let hand = choose(&["left", "right"], rng);
    let side = choose(&["left", "right"], rng);
    let amplitude = rng.random_range(0.0..1.0);
    let phrase = spec
        .phrase
        .replace("{hand}", hand)
        .replace("{side}", side)
        .replace("{count}", ["once", "twice", "three times"][count - 1])
        .replace("{size}", if amplitude < 0.5 { "small" } else { "wide" })
        .replace("{rhythm}", choose(&["steady", "syncopated", "gentle"], rng));
    let peak = spec.peak_height[0] + (spec.peak_height[1] - spec.peak_height[0]) * amplitude;
    let minimum = count * spec.minimum_frames;
    let duration = rng.random_range(minimum..=(minimum + minimum / 4).min(frames));
    Some(SampledAction {
        spec,
        phrase,
        frames: duration,
        phase: ActionPhase {
            family,
            action: spec.id.into(),
            hand: hand.into(),
            side: side.into(),
            repetitions: count,
            amplitude,
            start_frame: 0,
            end_frame: duration,
            peak_pelvis_height: peak,
            clearance_radius: spec.radius * (0.9 + 0.1 * amplitude),
        },
    })
}

/// Interior action keys keep a stationary pelvis corridor. Individual cycles
/// have standing/seated recovery keys, not a single sustained low pose.
fn action_keys(
    action: &SampledAction,
    anchor: &Waypoint,
    start: usize,
    end: usize,
    seated: bool,
) -> Vec<Waypoint> {
    let base = if seated { 0.61 } else { STANDING_PELVIS };
    let mut keys = Vec::new();
    for i in 0..=action.phase.repetitions * 2 {
        let t = i as f32 / (action.phase.repetitions * 2) as f32;
        let peak = i % 2 == 1;
        let heading = anchor.heading.map(|h| {
            h + if peak && action.spec.turn {
                std::f32::consts::FRAC_PI_2
                    * if action.phase.side == "left" {
                        -1.0
                    } else {
                        1.0
                    }
            } else {
                0.0
            }
        });
        keys.push(Waypoint {
            frame: start + ((end - start) as f32 * t).round() as usize,
            position: anchor.position.with_y(if peak && !seated {
                action.phase.peak_pelvis_height
            } else {
                base
            }),
            heading,
            constrain_height: true,
        });
    }
    keys
}

pub(super) fn stationary(
    mut action: SampledAction,
    points: &mut Vec<Waypoint>,
    config: &HumanMotionConfig,
    seated: bool,
    rng: &mut impl Rng,
) -> Draft {
    let last = config.frames - 1;
    let start = ((last - action.frames.min(last)) as f32 * rng.random_range(0.15..0.75)) as usize;
    let end = (start + action.frames).min(last);
    let anchor = points[0].clone();
    let mut timed = std::collections::BTreeMap::new();
    for mut w in [points[0].clone(), points.last().unwrap().clone()] {
        w.position.y = if seated { 0.61 } else { STANDING_PELVIS };
        w.constrain_height = true;
        timed.insert(w.frame, w);
    }
    for w in action_keys(&action, &anchor, start, end, seated) {
        timed.insert(w.frame, w);
    }
    *points = timed.into_values().collect();
    action.phase.start_frame = start;
    action.phase.end_frame = end;
    if seated {
        action.phase.peak_pelvis_height = 0.61;
    }
    let mut recipe = recipe(action.spec.family);
    recipe.actions.push(action.phase.clone());
    let prefix = if seated {
        "A seated person"
    } else {
        "A person"
    };
    let text = finish(
        format!("{prefix} {}", action.phrase),
        &mut recipe,
        &config.prompt_sampling,
        false,
        rng,
    );
    Draft {
        behavior: if seated {
            "seated_gesture".into()
        } else {
            action.spec.id.into()
        },
        text,
        recipe,
        radius: action.phase.clearance_radius,
        headroom: action.spec.headroom,
    }
}

pub(super) fn chair(behavior: &str, config: &HumanMotionConfig, rng: &mut impl Rng) -> Draft {
    let clause = if behavior == "sit" {
        choose(
            &[
                "steps back and sits on the chair",
                "backs up and lowers onto the chair",
                "bends both knees and sits down",
            ],
            rng,
        )
    } else {
        choose(
            &[
                "rises from the chair and steps forward",
                "leans forward and stands from the chair",
                "stands up from the chair and straightens",
            ],
            rng,
        )
    };
    let mut recipe = recipe(Family::Locomotion);
    let text = finish(
        format!("A person {clause}"),
        &mut recipe,
        &config.prompt_sampling,
        false,
        rng,
    );
    Draft {
        behavior: behavior.into(),
        text,
        recipe,
        radius: 0.30,
        headroom: 0.0,
    }
}

struct Gait {
    id: &'static str,
    verb: &'static str,
    noun: &'static str,
    minimum_speed: f32,
    maximum_speed: f32,
    heading_offset: f32,
    headroom: f32,
    elastic_height: bool,
}
const GAITS: &[Gait] = &[
    Gait {
        id: "walk",
        verb: "walks",
        noun: "walking",
        minimum_speed: 0.0,
        maximum_speed: 1.8,
        heading_offset: 0.0,
        headroom: 0.0,
        elastic_height: false,
    },
    Gait {
        id: "tiptoe",
        verb: "tiptoes",
        noun: "tiptoeing",
        minimum_speed: 0.0,
        maximum_speed: 0.8,
        heading_offset: 0.0,
        headroom: 0.12,
        elastic_height: false,
    },
    Gait {
        id: "shuffle",
        verb: "shuffles",
        noun: "shuffling",
        minimum_speed: 0.0,
        maximum_speed: 0.65,
        heading_offset: 0.0,
        headroom: 0.0,
        elastic_height: false,
    },
    Gait {
        id: "march",
        verb: "marches",
        noun: "marching",
        minimum_speed: 0.25,
        maximum_speed: 1.25,
        heading_offset: 0.0,
        headroom: 0.10,
        elastic_height: false,
    },
    Gait {
        id: "backward_walk",
        verb: "walks backward",
        noun: "walking backward",
        minimum_speed: 0.0,
        maximum_speed: 0.7,
        heading_offset: std::f32::consts::PI,
        headroom: 0.0,
        elastic_height: false,
    },
    Gait {
        id: "side_step",
        verb: "steps sideways",
        noun: "stepping sideways",
        minimum_speed: 0.0,
        maximum_speed: 0.75,
        heading_offset: std::f32::consts::FRAC_PI_2,
        headroom: 0.0,
        elastic_height: false,
    },
    Gait {
        id: "skip",
        verb: "skips",
        noun: "skipping",
        minimum_speed: 0.35,
        maximum_speed: 1.6,
        heading_offset: 0.0,
        headroom: 0.55,
        elastic_height: true,
    },
    Gait {
        id: "jog",
        verb: "jogs",
        noun: "jogging",
        minimum_speed: 0.75,
        maximum_speed: 2.0,
        heading_offset: 0.0,
        headroom: 0.30,
        elastic_height: true,
    },
];
fn gait(speed: f32, headroom: f32, energetic_fraction: f32, rng: &mut impl Rng) -> &'static Gait {
    let energetic = rng.random_bool(energetic_fraction as f64);
    let choices: Vec<_> = GAITS
        .iter()
        .filter(|g| {
            g.elastic_height == energetic
                && speed >= g.minimum_speed
                && speed <= g.maximum_speed
                && g.headroom <= headroom
        })
        .collect();
    if choices.is_empty() {
        &GAITS[0]
    } else {
        choices[rng.random_range(0..choices.len())]
    }
}
fn condition_gait(points: &mut [Waypoint], gait: &Gait) {
    for w in points {
        w.position.y = STANDING_PELVIS;
        w.constrain_height = !gait.elastic_height;
        w.heading = w.heading.map(|h| h + gait.heading_offset);
    }
}
fn pace(speed: f32) -> &'static str {
    if speed < 0.45 {
        "slowly"
    } else if speed < 0.95 {
        "steadily"
    } else {
        "briskly"
    }
}
pub(super) fn locomotion(
    behavior: &str,
    points: &mut [Waypoint],
    config: &HumanMotionConfig,
    headroom: f32,
    rng: &mut impl Rng,
) -> Draft {
    let distance: f32 = points
        .windows(2)
        .map(|p| {
            p[0].position
                .with_y(0.0)
                .distance(p[1].position.with_y(0.0))
        })
        .sum();
    let speed = distance / ((config.frames - 1) as f32 / 20.0);
    let gait = if matches!(behavior, "walk" | "return") {
        gait(speed, headroom, config.energetic_fraction, rng)
    } else {
        &GAITS[0]
    };
    condition_gait(points, gait);
    let mut recipe = recipe(Family::Locomotion);
    recipe.gait = Some(gait.id.into());
    recipe.travel_speed_mps = Some(speed);
    let destination = match behavior {
        "enter" => " through the doorway into the room",
        "leave" => " through the doorway out of the room",
        "return" => ", turns around and comes back",
        _ => "",
    };
    let text = finish(
        format!("A person {} {}{destination}", gait.verb, pace(speed)),
        &mut recipe,
        &config.prompt_sampling,
        true,
        rng,
    );
    Draft {
        behavior: if matches!(behavior, "walk" | "return") {
            gait.id.into()
        } else {
            behavior.into()
        },
        text,
        recipe,
        radius: 0.40,
        headroom: gait.headroom,
    }
}
