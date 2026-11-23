//! A fitted flight has a clear approach, one stock specification, and a list of
//! walkable edges. The floor patches remain the sole source of support heights.
use super::{EnvelopeProgram, FloorPatch, IndoorManifest};
use crate::scene::procedural_indoor::{layout::stream, materials::Surface};
use bevy::prelude::*;
use rand::Rng;
use serde::{Deserialize, Serialize};

#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepAxis {
    X,
    Z,
}

impl StepAxis {
    pub(super) fn index(self) -> usize {
        match self {
            Self::X => 0,
            Self::Z => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepNosing {
    Integral,
    Timber,
    Resilient,
    MetalWithInsert,
}

impl StepNosing {
    pub(super) fn surface(self) -> Option<Surface> {
        match self {
            Self::Integral => None,
            Self::Timber => Some(Surface::Wood),
            Self::Resilient => Some(Surface::Rubber),
            Self::MetalWithInsert => Some(Surface::Metal),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StepEdge {
    pub coordinate_m: f32,
    /// Along the edge, measured on the axis perpendicular to the flight.
    pub span_m: Vec2,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StepInstallation {
    pub run_axis: StepAxis,
    pub riser_surface: Surface,
    pub nosing: StepNosing,
    pub tread_band_m: f32,
    pub face_return_m: f32,
    /// Width of each exposed metal shoulder beside a flush abrasive insert.
    pub insert_margin_m: f32,
    pub walk_edges: Vec<StepEdge>,
    /// Clear route from the base landing through a short flight onto its platform.
    /// Lecture aisles have their own circulation plan.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub passage: Option<Rect>,
}

impl StepInstallation {
    pub(crate) fn sample(seed: u64, floor_style: u32, axis: StepAxis) -> Self {
        let mut rng = stream(seed, u64::from_le_bytes(*b"StepTrim"));
        use StepNosing::*;
        let (facing, trims) = match floor_style {
            0 => (
                [Surface::Wood, Surface::Paint, Surface::Wood],
                [Timber, Integral, Timber, MetalWithInsert, Resilient],
            ),
            1 => (
                [Surface::Floor, Surface::Rubber, Surface::Wood],
                [
                    Resilient,
                    MetalWithInsert,
                    Resilient,
                    Integral,
                    MetalWithInsert,
                ],
            ),
            _ => (
                [Surface::Concrete, Surface::Concrete, Surface::Ceramic],
                [
                    Integral,
                    Resilient,
                    MetalWithInsert,
                    Integral,
                    MetalWithInsert,
                ],
            ),
        };
        let tread_band_m = rng.random_range(0.032..0.074);
        Self {
            run_axis: axis,
            riser_surface: facing[rng.random_range(0..facing.len())],
            nosing: trims[rng.random_range(0..trims.len())],
            tread_band_m,
            face_return_m: rng.random_range(0.016..0.044),
            insert_margin_m: tread_band_m * rng.random_range(0.10..0.23),
            walk_edges: Vec::new(),
            passage: None,
        }
    }

    pub(super) fn is_walk_edge(&self, coordinate: f32, across: f32) -> bool {
        self.walk_edges.iter().any(|edge| {
            (coordinate - edge.coordinate_m).abs() < 1e-5
                && (edge.span_m.x..=edge.span_m.y).contains(&across)
        })
    }

    pub(super) fn valid(&self, envelope: &EnvelopeProgram) -> bool {
        matches!(
            self.riser_surface,
            Surface::Paint
                | Surface::Wood
                | Surface::Rubber
                | Surface::Floor
                | Surface::Concrete
                | Surface::Ceramic
        ) && (0.025..=0.085).contains(&self.tread_band_m)
            && (0.01..=0.05).contains(&self.face_return_m)
            && (0.002..=self.tread_band_m * 0.30).contains(&self.insert_margin_m)
            && self.passage.is_none_or(|p| {
                p.min.is_finite()
                    && p.max.is_finite()
                    && p.size().min_element() >= 0.70
                    && super::polygon::box_inside(&envelope.footprint, p.min, p.max, 0.)
            })
            && !self.walk_edges.is_empty()
            && self.walk_edges.len() <= envelope.floor_patches.len() * 2
            && self.walk_edges.iter().enumerate().all(|(i, edge)| {
                !self.walk_edges[..i].contains(edge) && self.edge_is_backed(envelope, edge)
            })
    }

    fn edge_is_backed(&self, envelope: &EnvelopeProgram, edge: &StepEdge) -> bool {
        let axis = self.run_axis.index();
        let cross = 1 - axis;
        // A full tread/stage boundary provides both the lateral extent and
        // the clearance from the clipped perimeter needed by the inset band.
        if !envelope.floor_patches.iter().any(|p| {
            [p.min[axis], p.max[axis]].contains(&edge.coordinate_m)
                && edge.span_m == Vec2::new(p.min[cross], p.max[cross])
        }) || edge.span_m.y - edge.span_m.x < 0.55
        {
            return false;
        }
        let heights = |offset| {
            let mut point = Vec2::ZERO;
            point[cross] = offset;
            point[axis] = edge.coordinate_m - 0.001;
            let left = envelope.floor_height(point);
            point[axis] = edge.coordinate_m + 0.001;
            [left, envelope.floor_height(point)]
        };
        let datum = heights((edge.span_m.x + edge.span_m.y) * 0.5);
        if !(0.07..=0.21).contains(&(datum[0] - datum[1]).abs()) {
            return false;
        }
        // A neighboring patch may meet only part of a long edge. Check both
        // ends of every intersecting interval, not just the flight's midpoint.
        envelope.floor_patches.iter().all(|p| {
            if p.min[axis] > edge.coordinate_m + 0.001 || p.max[axis] < edge.coordinate_m - 0.001 {
                return true;
            }
            [p.min[cross], p.max[cross]].into_iter().all(|cut| {
                if cut <= edge.span_m.x || cut >= edge.span_m.y {
                    return true;
                }
                heights(cut.next_down()) == datum && heights(cut.next_up()) == datum
            })
        })
    }
}

/// Select among four entrances with an actual landing inside the room and clear
/// of portals. Draw from a separate stream so furniture/material RNG is not
/// advanced by the number of candidate directions tested.
pub(super) fn short_flight(
    room: &IndoorManifest,
    envelope: &mut EnvelopeProgram,
    lo: Vec2,
    hi: Vec2,
    steps: usize,
    rise: f32,
    tread: f32,
) -> bool {
    let mut rng = stream(room.seed, u64::from_le_bytes(*b"StepPath"));
    let first = rng.random_range(0..4);
    let stride = if rng.random_bool(0.5) { 1 } else { 3 };
    let cross_fraction = rng.random_range(0.30..0.70);
    let clear_width = rng.random_range(0.84..1.06);
    let landing = rng.random_range(0.75..1.03);
    let zone = room.primary_room_bounds();
    for i in 0..4 {
        let side = (first + stride * i) % 4;
        let axis = if side % 2 == 0 {
            StepAxis::X
        } else {
            StepAxis::Z
        };
        let k = axis.index();
        let cross = 1 - k;
        let positive = side < 2;
        let entry = if positive { hi[k] } else { lo[k] };
        let inward = if positive { -1. } else { 1. };
        let boundary = |j: usize| entry + inward * j as f32 * tread;
        let center = (lo[cross] + (hi[cross] - lo[cross]) * cross_fraction)
            .clamp(lo[cross] + clear_width * 0.5, hi[cross] - clear_width * 0.5);
        let outer = entry - inward * landing;
        let inner = boundary(steps) + inward * 0.42;
        let mut passage = Rect { min: lo, max: hi };
        passage.min[k] = outer.min(inner);
        passage.max[k] = outer.max(inner);
        passage.min[cross] = center - clear_width * 0.5;
        passage.max[cross] = center + clear_width * 0.5;
        if !passage.min.cmpge(zone.0).all()
            || !passage.max.cmple(zone.1).all()
            || !envelope.feature_clear(room, passage.min, passage.max, 0.06)
        {
            continue;
        }

        let mut installation = StepInstallation::sample(room.seed, room.floor_style, axis);
        installation.passage = Some(passage);
        let patch = |a: f32, b: f32, height: f32| {
            let mut p = FloorPatch {
                min: lo,
                max: hi,
                height,
            };
            p.min[k] = a.min(b);
            p.max[k] = a.max(b);
            p
        };
        let far_end = if positive { lo[k] } else { hi[k] };
        envelope
            .floor_patches
            .push(patch(far_end, boundary(steps), rise * steps as f32));
        for j in 0..steps {
            envelope
                .floor_patches
                .push(patch(boundary(j), boundary(j + 1), rise * j as f32));
            installation.walk_edges.push(StepEdge {
                coordinate_m: boundary(j + 1),
                span_m: Vec2::new(lo[cross], hi[cross]),
            });
        }
        envelope.step_installation = Some(installation);
        return true;
    }
    false
}
