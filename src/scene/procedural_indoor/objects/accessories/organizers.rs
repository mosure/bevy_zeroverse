//! Unequal desktop compartments and document trays with supported, fitted contents.
use super::*;
use crate::scene::procedural_indoor::clutter::pen_holder::{self, Pen};
mod legacy;
pub mod sheets;
mod shell;
#[cfg(test)]
mod tests;
pub(crate) const INTRODUCED: u32 = 228;

#[derive(Debug, Clone, serde::Serialize)]
pub struct OrganizerProgram {
    pub weights: Vec<f32>,
    pub heights: Vec<f32>,
    /// Requested occupancy. A compartment without room for its stock stays empty.
    pub filled: Vec<bool>,
    pub front_fraction: f32,
    pub wall_fraction: f32,
    pub finish: Surface,
}

pub fn parameters(o: &IndoorObject) -> OrganizerProgram {
    parameters_at(o, INTRODUCED)
}
fn parameters_at(o: &IndoorObject, version: u32) -> OrganizerProgram {
    let mut r = stream(o.seed, 0x4f5247414e495a45);
    let count = if o.kind == ObjectKind::DocumentTray {
        r.random_range(1..=4)
    } else {
        r.random_range(1..=5)
    };
    OrganizerProgram {
        weights: (0..count).map(|_| r.random_range(0.55..1.60)).collect(),
        heights: (0..count).map(|_| r.random_range(0.36..0.96)).collect(),
        filled: (0..count).map(|_| r.random_bool(0.66)).collect(),
        front_fraction: r.random_range(0.12..0.78),
        wall_fraction: r.random_range(0.014..0.032),
        finish: {
            let material = if version >= INTRODUCED {
                stream(
                    o.finish_seed.unwrap_or(o.seed),
                    u64::from_le_bytes(*b"TrayBody"),
                )
                .random_range(0..3)
            } else {
                r.random_range(0..3)
            };
            [Surface::Metal, Surface::Plastic, Surface::Wood][material]
        },
    }
}

#[derive(Clone, Copy)]
pub(super) struct Bay {
    index: usize,
    x: f32,
    base: f32,
    width: f32,
    depth: f32,
    height: f32,
    wall_h: f32,
    t: f32,
    inner: Vec2,
    last: bool,
}

impl OrganizerProgram {
    fn bays(&self, o: &IndoorObject, version: u32) -> impl Iterator<Item = Bay> + '_ {
        let s = o.size;
        let trays = o.kind == ObjectKind::DocumentTray;
        // The reservation includes above-rim tool overhang in a shallow caddy.
        let depth = if !trays && version >= INTRODUCED {
            s.z * stream(o.seed, u64::from_le_bytes(*b"CaddyBed")).random_range(0.75..0.92)
        } else {
            s.z
        };
        let sum = self.weights.iter().sum::<f32>();
        let narrowest = self.weights.iter().copied().fold(f32::INFINITY, f32::min) / sum;
        let t = (s.min_element() * self.wall_fraction)
            .min(if trays { s.y } else { s.x } * narrowest * 0.12);
        let mut cursor = if trays { 0. } else { -s.x * 0.5 };
        self.weights.iter().enumerate().map(move |(i, weight)| {
            let width = if trays { s.x } else { s.x * weight / sum };
            let height = if trays {
                s.y * weight / sum
            } else {
                s.y * self.heights[i]
            };
            let base = if trays { cursor } else { 0. };
            let x = if trays { 0. } else { cursor + width * 0.5 };
            cursor += if trays { height } else { width };
            Bay {
                index: i,
                x,
                base,
                width,
                depth,
                height,
                t,
                wall_h: if trays {
                    (height - t) * self.front_fraction
                } else {
                    height - t
                },
                inner: Vec2::new(width - t * 4.4, depth - t * 4.4),
                last: i + 1 == self.weights.len(),
            }
        })
    }
}

#[derive(Debug, serde::Serialize)]
#[serde(tag = "contents", rename_all = "snake_case")]
pub enum Contents {
    Empty,
    Tools { bay_x_m: f32, tools: Vec<Pen> },
    Paper(sheets::PaperStack),
}

fn contents(o: &IndoorObject, p: &OrganizerProgram, b: &Bay) -> Contents {
    if !p.filled[b.index] {
        return Contents::Empty;
    }
    let tray = o.kind == ObjectKind::DocumentTray;
    let mut r = stream(o.seed, u64::from_le_bytes(*b"DeskBays") + b.index as u64);
    if !tray && !r.random_bool(0.42) {
        let clear = Vec2::new(b.width - 2.2 * b.t, b.depth - 2. * b.t);
        let tools = pen_holder::compartment_tools(
            o,
            b.index,
            Vec3::new(clear.x, o.size.y, o.size.z),
            clear,
            b.t,
            b.height,
        );
        return if tools.is_empty() {
            Contents::Empty
        } else {
            Contents::Tools {
                bay_x_m: b.x,
                tools,
            }
        };
    }
    // A narrow compartment can be too small even for an index slip.
    sheets::PaperStack::fit(o, b).map_or(Contents::Empty, Contents::Paper)
}

/// Physical stock and occupancy, sampled by the same program that constructs each bay.
pub fn compartments(o: &IndoorObject, p: &OrganizerProgram) -> Vec<Contents> {
    p.bays(o, INTRODUCED).map(|b| contents(o, p, &b)).collect()
}

pub fn metadata(o: &IndoorObject) -> serde_json::Value {
    let p = parameters(o);
    let mut result = serde_json::json!(p);
    let bays = compartments(o, &p);
    result["requested_filled"] = result["filled"].take();
    result["filled"] = serde_json::json!(bays
        .iter()
        .map(|b| !matches!(b, Contents::Empty))
        .collect::<Vec<_>>());
    result["compartments"] = serde_json::json!(bays);
    if o.kind == ObjectKind::DocumentTray {
        result["rim"] = serde_json::json!(shell::TrayProfile::sample(o));
    }
    result
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters_at(o, version);
    let s = o.size;
    let trays = o.kind == ObjectKind::DocumentTray;
    let label = o.kind.class_name();
    let current = version >= INTRODUCED;
    let profile = (current && trays).then(|| shell::TrayProfile::sample(o));
    let mut legacy_rng = stream(o.seed, 0x4f5247434f4e5445);
    for b in p.bays(o, version) {
        let current_contents = current.then(|| contents(o, &p, &b));
        let tool_bay = matches!(current_contents, Some(Contents::Tools { .. }));
        let wall_bevel = if tool_bay { 0. } else { b.t * 0.12 };
        let Bay {
            index: i,
            x,
            base,
            width,
            t,
            wall_h,
            ..
        } = b;
        if !current && trays && i == 0 && p.weights.len() > 1 {
            legacy::tray_frame(a, o, &p, t);
        }
        if trays || i == 0 {
            a.box_part(
                p.finish,
                label,
                Vec3::new(0., base + t * 0.5, 0.),
                Vec3::new(s.x, t, b.depth),
                t * 0.2,
            );
        }
        if let Some(profile) = &profile {
            profile.build(a, o, &p, &b);
        } else {
            for side in [-1., 1.] {
                a.box_part(
                    p.finish,
                    label,
                    Vec3::new(
                        x + side * (width * 0.5 - t * 0.6),
                        base + t + wall_h * 0.5,
                        0.,
                    ),
                    Vec3::new(t, wall_h, b.depth),
                    wall_bevel,
                );
            }
        }
        a.box_part(
            p.finish,
            label,
            Vec3::new(x, base + t + wall_h * 0.5, -b.depth * 0.5 + t * 0.5),
            Vec3::new(width - t * 2.2, wall_h, t),
            wall_bevel,
        );
        if !trays {
            let front = if tool_bay {
                wall_h
            } else {
                wall_h * p.front_fraction
            };
            a.box_part(
                p.finish,
                label,
                Vec3::new(x, t + front * 0.5, b.depth * 0.5 - t * 0.5),
                Vec3::new(width - t * 2.2, front, t),
                wall_bevel,
            );
        }
        if let Some(built) = current_contents {
            match built {
                Contents::Empty => {}
                Contents::Paper(stack) => stack.build(a),
                Contents::Tools { bay_x_m, tools } => {
                    for pen in tools {
                        pen.draw(a, label, Vec3::X * bay_x_m);
                    }
                }
            }
        } else if p.filled[i] {
            if trays || legacy_rng.random_bool(0.42) {
                legacy::paper(a, o, &b, &mut legacy_rng);
            } else {
                legacy::tools(a, o, &b, &mut legacy_rng);
            }
        }
    }
}
