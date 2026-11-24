//! Frame joints follow sectional depth and transportable lengths. Load-bearing
//! stations are measured along each frame independently of the cushion pitch.
use super::*;
use crate::scene::procedural_indoor::geometry::stock::{GrainAxis, TimberCuts};

mod hardware;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize)]
pub(crate) enum Base {
    Tapered,
    Hairpin,
    CrossSled,
    Plinth,
    Pedestal,
    LengthwiseSled,
    Block,
}

#[derive(Debug)]
pub(crate) struct Section {
    pub left: f32,
    pub right: f32,
    /// Negative Z is the front of the seat.
    pub front: f32,
    pub back: f32,
    end_inset: f32,
    front_inset: f32,
    rear_inset: f32,
    intervals: usize,
    center_levelers: bool,
}

impl Section {
    fn station_x(&self, i: usize) -> f32 {
        let span = self.right - self.left - 2. * self.end_inset;
        self.left + self.end_inset + span * i as f32 / self.intervals as f32
    }
    fn ends_z(&self) -> [f32; 2] {
        [self.front + self.front_inset, self.back - self.rear_inset]
    }
    fn center(&self) -> Vec3 {
        Vec3::new(
            (self.left + self.right) * 0.5,
            0.,
            (self.front + self.back) * 0.5,
        )
    }
}

#[derive(Debug)]
pub(crate) struct Foundation {
    pub base: Base,
    pub stock: Surface,
    pub sections: Vec<Section>,
    /// Radius of the stock for bent frames, half-width for cut legs.
    half_width: f32,
    foot_ratio: f32,
    rake: f32,
    aspect: f32,
    glide_height: f32,
}

impl Foundation {
    pub(crate) fn new(o: &IndoorObject, p: &SofaProgram, widths: &[f32]) -> Self {
        let mut joints = stream(o.seed, u64::from_le_bytes(*b"SofaJoin"));
        let mut hardware = stream(o.seed, u64::from_le_bytes(*b"SofaFoot"));
        let mut finish = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"SofaStok"),
        );
        let base = match p.base_style {
            0 => Base::Tapered,
            1 => Base::Hairpin,
            2 => Base::CrossSled,
            3 => Base::Plinth,
            4 if o.kind == ObjectKind::Chair => Base::Pedestal,
            4 => Base::LengthwiseSled,
            _ => Base::Block,
        };
        let stock = match base {
            Base::Tapered | Base::Plinth | Base::Block => [
                Surface::WoodEdge,
                Surface::Wood,
                Surface::Metal,
                Surface::Plastic,
            ][finish.random_range(0..4)],
            _ if finish.random_bool(0.28) => Surface::Chrome,
            _ => Surface::Metal,
        };
        let split_probability = match joints.random_range(0..5) {
            0 => 1.,
            1 | 2 => joints.random_range(0.12..0.72),
            _ => 0.,
        };
        let max_piece = joints.random_range(1.76..3.55);
        let max_span = hardware.random_range(1.04..1.68);
        let end_inset = hardware.random_range(0.072_f32..0.166);
        let front_inset = hardware.random_range(0.065_f32..0.155);
        let rear_inset = hardware.random_range(0.065_f32..0.155);
        let levelers = hardware.random_bool(0.55)
            && matches!(base, Base::Tapered | Base::Hairpin | Base::Block);
        let mut sections = Vec::with_capacity(widths.len());
        let mut left = -o.size.x * 0.5 + p.arm_width;
        let mut right = left;
        let back = if p.recliner {
            // The moving footrest lies outside the stationary chassis.
            recliner::base_center_z(o, p) + recliner::BASE_DEPTH * 0.5
        } else {
            o.size.z * 0.5
        };
        for (i, &width) in widths.iter().enumerate() {
            right += width;
            let depth = if p.recliner {
                recliner::BASE_DEPTH
            } else {
                module_depth(o, p, i)
            };
            let next = widths.get(i + 1);
            if next.is_some_and(|&w| {
                module_depth(o, p, i + 1) == depth
                    && right - left + w <= max_piece
                    && !joints.random_bool(split_probability)
            }) {
                continue;
            }
            let inset = end_inset.min((right - left) * 0.19);
            let span = right - left - 2. * inset;
            sections.push(Section {
                left,
                right,
                front: back - depth,
                back,
                end_inset: inset,
                front_inset: front_inset.min(depth * 0.20),
                rear_inset: rear_inset.min(depth * 0.20),
                intervals: (span / max_span).ceil().max(1.) as usize,
                center_levelers: levelers,
            });
            left = right;
        }
        Self {
            base,
            stock,
            sections,
            half_width: match base {
                Base::Hairpin => hardware.random_range(0.0052..0.0082),
                Base::CrossSled | Base::LengthwiseSled => hardware.random_range(0.009..0.015),
                Base::Block => hardware.random_range(0.030..0.049),
                _ => hardware.random_range(0.023..0.039),
            },
            foot_ratio: hardware.random_range(0.48..0.80),
            rake: hardware.random_range(0.006..0.032),
            aspect: hardware.random_range(0.79..1.22),
            glide_height: hardware.random_range(0.006..0.013),
        }
    }

    /// Number of discrete contact pads. Plinths and disc pedestals bear
    /// directly on the floor and have no separate glides.
    pub(crate) fn glide_count(&self) -> usize {
        if matches!(self.base, Base::Pedestal | Base::Plinth) {
            return 0;
        }
        self.sections
            .iter()
            .map(|s| 4 + (s.intervals - 1) * if s.center_levelers { 1 } else { 2 })
            .sum()
    }

    pub(super) fn metadata(&self) -> serde_json::Value {
        let sections: Vec<_> = self
            .sections
            .iter()
            .map(|s| {
                let mut section = serde_json::json!({
                    "left_m": s.left, "right_m": s.right, "front_m": s.front, "back_m": s.back,
                });
                if self.base != Base::Pedestal {
                    section["support_z_m"] = serde_json::json!(s.ends_z());
                    if self.base == Base::Plinth {
                        section["support_x_m"] =
                            serde_json::json!([s.station_x(0), s.station_x(s.intervals)]);
                    } else {
                        section["support_x_m"] = serde_json::json!((0..=s.intervals)
                            .map(|i| s.station_x(i))
                            .collect::<Vec<_>>());
                        section["center_levelers"] =
                            serde_json::json!(s.center_levelers && s.intervals > 1);
                    }
                }
                section
            })
            .collect();
        serde_json::json!({"base":self.base, "stock":self.stock, "sections":sections, "glide_count":self.glide_count()})
    }

    pub(super) fn build(&self, a: &mut Assembly, o: &IndoorObject, p: &SofaProgram) {
        let label = o.kind.class_name();
        let frame_stock = if p.exposed_frame && !p.recliner {
            Surface::WoodEdge
        } else {
            p.upholstery
        };
        let mut cuts = TimberCuts::new(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"SofaCuts"),
        );
        let top = p.seat_height - p.cushion;
        let (joint, edge, bevel) = if p.recliner {
            (0., 0., 0.035)
        } else {
            (0.004, 0.018, p.roundness * 0.5)
        };
        for s in &self.sections {
            let tf = Transform::from_translation(s.center().with_y((p.leg_height + top) * 0.5));
            cuts.cuboid(
                a.part(frame_stock, label),
                frame_stock,
                Vec3::new(
                    s.right - s.left - joint,
                    top - p.leg_height,
                    s.back - s.front - edge,
                ),
                bevel,
                tf,
                GrainAxis::X,
            );
            self.hardware(a, label, p.leg_height, s, &mut cuts);
        }
    }
}
