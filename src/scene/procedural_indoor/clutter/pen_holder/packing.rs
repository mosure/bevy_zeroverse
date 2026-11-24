//! Tilt a tool, lower it to the actual cup floor, then place a barrel vertex on
//! the lip. Admission checks the full tilted shaft, clip and neighboring tools.
use super::*;
use std::f32::consts::TAU;

pub(super) fn settle(
    owner_seed: u64,
    body_finish: Option<u64>,
    envelope: Vec3,
    pocket: &Pocket,
) -> Vec<Pen> {
    let mut set = Vec::with_capacity(pocket.count);
    let upper_tilt = (pocket.span.min_element() * 0.84 / (pocket.height - pocket.floor))
        .atan()
        .clamp(0.09, 0.52);
    let mut stock_rng = stream(owner_seed, u64::from_le_bytes(*b"PenStock"));
    for _ in 0..pocket.count {
        let seed = stock_rng.random();
        let mut r = stream(seed, u64::from_le_bytes(*b"PenStand"));
        let finish = if stock_rng.random_bool(0.30) {
            body_finish.unwrap_or(owner_seed)
        } else {
            seed ^ body_finish.unwrap_or(0)
        };
        let width = stock_rng.random_range(0.007..0.013_f32);
        let mut tool = writing::stock(seed, finish, Vec3::new(width, width, envelope.y));
        tool.length = envelope.y * stock_rng.random_range(0.81..0.99);
        let preferred_axis_up = r.random_bool(if tool.clip { 0.27 } else { 0.48 });
        let slot = variants::object_slot(tool.finish, seed, Some(finish));
        let phase = r.random_range(0.0..TAU);
        let original_length = tool.length;
        for attempt in 0..32 {
            // First try the sampled bunch, then sweep the rim in both resting
            // orientations. Late proposals can use a shorter or longer tool
            // of the same product to put its barrel, rather than its tip, on the lip.
            let axis_up = if attempt < 5 || attempt % 2 == 0 {
                preferred_axis_up
            } else {
                !preferred_axis_up
            };
            if attempt >= 12 {
                tool.length = (original_length * r.random_range(0.92..1.09))
                    .clamp(envelope.y * 0.80, envelope.y * 0.99);
            }
            let (edge, lip) = rim_position(pocket, &set, phase, attempt, &mut r);
            let heading = Vec2::from_angle(r.random_range(-0.72..0.72)).rotate(edge.normal);
            let estimated_center = lip - tool.radius * edge.normal;
            let run = (envelope.xz() * 0.5 - tool.radius - heading.signum() * estimated_center)
                / heading.abs().max(Vec2::splat(1e-5));
            let depth = pocket.height - pocket.floor;
            let upper = upper_tilt
                .min((run.min_element().max(0.) / (tool.length - depth).max(0.004)).atan());
            let lower = ((pocket.mouth - pocket.base) * edge.distance
                / (depth * heading.dot(edge.normal)))
            .atan()
            .max(0.025);
            if upper <= lower {
                continue;
            }
            let u = r.random::<f32>();
            // With a crowded floor, a shallower lean keeps the foot near its
            // own side instead of crossing the other shafts near the centre.
            let tilt = lower
                + (upper - lower)
                    * if set.len() >= 2 && attempt >= 4 {
                        u * u
                    } else {
                        u
                    };
            let (sin, cos) = tilt.sin_cos();
            let up = Vec3::new(heading.x * sin, cos, heading.y * sin);
            let axis = if axis_up { up } else { -up };
            let slope = up.xz() / up.y;
            // At constant height this radius is the farthest point on a tilted
            // cylindrical section in the wall-normal direction. Rotate a real
            // stock vertex onto it even for pencils with only six flat faces.
            let x = Vec3::new(edge.normal.x, -edge.normal.dot(slope), edge.normal.y).normalize();
            let rot = Quat::from_mat3(&Mat3::from_cols(x, axis, x.cross(axis))).normalize();
            let (lo, hi) = tool.vertical_range(rot);
            let y = pocket.floor - lo;
            if y + hi > envelope.y {
                continue;
            }
            let Some((bearing, transition_y)) = tool.shaft_at(pocket.height, y, axis.y) else {
                continue;
            };
            let along = (pocket.height - y - x.y * bearing) / axis.y;
            let start = lip - axis.xz() * along - x.xz() * bearing;
            let origin = Vec3::new(start.x, y, start.y);
            let end = origin + axis * tool.length;
            let (base, top) = if axis_up {
                (origin, end)
            } else {
                (end, origin)
            };
            let candidate = Pen {
                base,
                top,
                contact: Vec3::new(lip.x, pocket.height, lip.y),
                rotation: rot,
                upper_end: tool.upper_end(axis_up),
                bounds: clearance::ToolBounds::new(
                    &tool,
                    Transform::from_translation(origin).with_rotation(rot),
                ),
                tool: tool.clone(),
                finish_slot: slot,
            };
            let pad = candidate.clearance_radius();
            if (base.xz().abs().max(top.xz().abs()) + pad)
                .cmpgt(envelope.xz() * 0.5)
                .any()
            {
                // A pointed end or a side-facing clip can fit where the
                // circumscribed capsule extends outside the placement envelope.
                if [Vec3::X, Vec3::Z].into_iter().any(|axis| {
                    let (a, b) = tool.projected_extent(rot.inverse() * axis);
                    let center = origin.dot(axis);
                    let half = envelope.dot(axis) * 0.5;
                    center + a < -half || center + b > half
                }) {
                    continue;
                }
            }
            if set
                .iter()
                .any(|other| !candidate.bounds.separate(other.bounds))
            {
                continue;
            }
            if fits(pocket, &candidate, slope, bearing, transition_y) {
                set.push(candidate);
                break;
            }
        }
    }
    set
}

/// Keep initial poses grouped when the sampled heading permits it. For a
/// crowded rim, choose among a few widely spaced, jittered contacts.
fn rim_position<'a>(
    pocket: &'a Pocket,
    settled: &[Pen],
    phase: f32,
    attempt: usize,
    r: &mut impl Rng,
) -> (&'a Edge, Vec2) {
    let lip_scale = pocket.inner_scale(pocket.height);
    let mut proposal = || {
        let direction = if attempt < 5 {
            pocket.heading + r.random_range(-pocket.spread..pocket.spread)
        } else {
            phase + 2.399_963_1 * (attempt - 5) as f32 + r.random_range(-1.0..1.0)
        };
        let index = (direction.rem_euclid(TAU) / TAU * pocket.edges.len() as f32) as usize
            % pocket.edges.len();
        let edge = &pocket.edges[index];
        let point = edge.start.lerp(edge.end, r.random_range(0.04..0.96)) * lip_scale;
        (edge, point)
    };
    let mut best = proposal();
    if attempt >= 4 && settled.len() >= 2 {
        let clearance = |pt: Vec2| {
            settled
                .iter()
                .map(|q| pt.distance_squared(q.contact.xz()))
                .fold(f32::INFINITY, f32::min)
        };
        let mut best_clearance = clearance(best.1);
        for _ in 0..2 {
            let other = proposal();
            let other_clearance = clearance(other.1);
            if other_clearance > best_clearance {
                best = other;
                best_clearance = other_clearance;
            }
        }
    }
    best
}

fn fits(pocket: &Pocket, pen: &Pen, slope: Vec2, bearing: f32, transition_y: f32) -> bool {
    // Each wall is piecewise linear in height. The support distance of the
    // horizontal ellipse of a tilted cylinder is also analytic. It suffices
    // to check the floor, belly, lip and change of barrel diameter.
    let radius = pen.tool.radius;
    let floor = pocket.floor;
    let floor_scale = pocket.inner_scale(floor);
    let lip_scale = pocket.inner_scale(pocket.height);
    let belly_scale = pocket.inner_scale(pocket.belly_y);
    let transition =
        (bearing < radius && transition_y > floor).then(|| pocket.inner_scale(transition_y));
    for edge in &pocket.edges {
        let n = edge.normal;
        let drift = n.dot(slope);
        let projected = (1. + drift * drift).sqrt();
        let at_base = n.dot(pen.base.xz());
        let clears = |y: f32, r: f32, scale: f32| {
            at_base + drift * (y - pen.base.y) + r * projected <= edge.distance * scale + 1e-7
        };
        if !clears(floor, radius, floor_scale)
            || !clears(pocket.height, bearing, lip_scale)
            || !clears(
                pocket.belly_y,
                if pocket.belly_y > transition_y {
                    bearing
                } else {
                    radius
                },
                belly_scale,
            )
            || transition.is_some_and(|scale| !clears(transition_y, radius, scale))
        {
            return false;
        }
    }
    if pen.tool.clip {
        let tf = pen.transform();
        let sections = pen
            .tool
            .clip_rings()
            .map(|s| s.map(|pt| tf.transform_point(pt)));
        for pair in sections.windows(2) {
            if pair.iter().flatten().all(|pt| pt.y > pocket.height) {
                continue;
            }
            // Extending the top frustum above the lip is conservative for a
            // partly submerged clip segment. Higher segments are unrestricted.
            if pair.iter().flatten().any(|pt| {
                pocket
                    .edges
                    .iter()
                    .any(|e| e.normal.dot(pt.xz()) > e.distance * pocket.inner_scale(pt.y) + 1e-7)
            }) {
                return false;
            }
        }
    }
    true
}
