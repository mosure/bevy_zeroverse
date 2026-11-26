//! Size the enclosing foyer around its flight, turning space and gallery. The
//! recorded mezzanine is the same one used for meshes, support and clearance.
use super::*;
use crate::scene::procedural_indoor::layout::stream;
use rand::Rng;

pub(crate) struct Section {
    pub room_size: Vec3,
    deck: FloorPatch,
    stair_min: Vec2,
    stair_max: Vec2,
    axis: usize,
    ascending_positive: bool,
    steps: u32,
    landing: Option<Landing>,
    abutment: DeckAbutment,
}

impl Section {
    pub fn sample(seed: u64) -> Self {
        let mut r = stream(seed, u64::from_le_bytes(*b"StairBay"));
        let height = r.random_range(2.82..3.48_f32);
        let steps = (height / r.random_range(0.148..0.177)).ceil() as u32;
        let landing = r.random_bool(0.56).then(|| Landing {
            after_riser: r.random_range(steps / 3..=2 * steps / 3),
            extra_run: r.random_range(1.0..1.65),
        });
        let run = steps as f32 * r.random_range(0.278..0.345)
            + landing.as_ref().map_or(0., |l| l.extra_run);
        let width = r.random_range(1.18..1.85);
        let bay = r.random_range(2.85..4.65);
        let rail_margin = r.random_range(0.24..0.64);
        let breadth = width + bay + rail_margin;
        let deck_depth = r.random_range(2.55..4.10);
        let front = r.random_range(2.85..4.75);
        let back = r.random_range(2.60..3.95);
        let left = r.random_range(2.65..3.90);
        let right = r.random_range(2.65..3.90);
        let axis = usize::from(r.random_bool(0.5));
        let ascending_positive = r.random_bool(0.5);
        let direction = if ascending_positive { 1. } else { -1. };
        let orient = |along, across| {
            if axis == 0 {
                Vec2::new(along, across)
            } else {
                Vec2::new(across, along)
            }
        };
        let length = front + run + deck_depth + back;
        let span = left + breadth + right;
        let xz = orient(length, span);
        let foot = direction * (front - length * 0.5);
        let top = foot + direction * run;
        let far = top + direction * deck_depth;
        let side = left - span * 0.5;
        // Leave the shared +Z entrance at ground level. On X-running
        // stairs the gallery therefore meets the rear wall. Both side walls
        // are available for Z-running flights.
        let bay_low = axis == 0 || r.random_bool(0.5);
        let flight_side = side + if bay_low { bay } else { rail_margin };
        let wall_edge = span * 0.5 - WALL_JOINT_M;
        let (low, high) = if bay_low {
            (-wall_edge, side + breadth)
        } else {
            (side, wall_edge)
        };
        Self {
            room_size: Vec3::new(xz.x, height + r.random_range(3.85..4.75), xz.y),
            deck: FloorPatch {
                min: orient(top.min(far), low),
                max: orient(top.max(far), high),
                height,
            },
            abutment: if bay_low {
                DeckAbutment::Low
            } else {
                DeckAbutment::High
            },
            stair_min: orient(foot.min(top), flight_side),
            stair_max: orient(foot.max(top), flight_side + width),
            axis,
            ascending_positive,
            steps,
            landing,
        }
    }

    pub fn flight(self, seed: u64) -> Mezzanine {
        let mut r = stream(seed, u64::from_le_bytes(*b"StairFrm"));
        let beam = r.random_range(0.12..0.23);
        let mut construction =
            Construction::sample(seed, self.ascending_positive, self.landing, beam);
        // Match the walking surface to its substrate. A timber gallery can
        // carry boards or a fitted floor covering. Cast flights can be left
        // concrete or topped, while steel stringers can take metal pans.
        let finishes: &[Surface] = match construction.frame_surface {
            Surface::Wood => &[Surface::Wood, Surface::Wood, Surface::Floor],
            Surface::Concrete => &[Surface::Concrete, Surface::Floor, Surface::Wood],
            _ => &[Surface::Metal, Surface::Wood, Surface::Floor],
        };
        construction.tread_surface = finishes[r.random_range(0..finishes.len())];
        if r.random_bool(0.69) {
            construction.deck_surface = construction.tread_surface;
        }
        if r.random_bool(0.34) {
            construction.nosing_depth = 0.; // Integral edge, uninterrupted by a rubber insert.
        }
        construction.wall_abutment = Some(self.abutment);
        Mezzanine {
            deck: self.deck,
            thickness: r.random_range(0.17..0.28),
            stair_min: self.stair_min,
            stair_max: self.stair_max,
            stair_axis: self.axis,
            steps: self.steps,
            rail_height: r.random_range(1.04..1.21),
            construction: Some(construction),
        }
    }
}

// Expansion joint between the interior wall datum and the slab. Only this
// verified abutment may omit its balustrade.
pub(super) const WALL_JOINT_M: f32 = 0.008;

impl Mezzanine {
    pub(crate) fn abuts_wall_segment(&self, p: Vec2, q: Vec2) -> bool {
        let Some(side) = self.construction.as_ref().and_then(|c| c.wall_abutment) else {
            return false;
        };
        let along = self.stair_axis;
        let cross = 1 - along;
        let face = match side {
            DeckAbutment::Low => self.deck.min[cross] - WALL_JOINT_M,
            DeckAbutment::High => self.deck.max[cross] + WALL_JOINT_M,
        };
        (p[cross] - face).abs() < 0.001
            && (q[cross] - face).abs() < 0.001
            && self.deck.min[along] >= p[along].min(q[along]) + 0.10
            && self.deck.max[along] <= p[along].max(q[along]) - 0.10
    }
}

/// Keep corners outside the continuous peripheral access aisles and preserve
/// the full shared entrance wall at +Z.
pub fn outline(span: Vec2, rng: &mut impl Rng) -> Vec<Vec2> {
    let half = span * 0.5;
    let mut corners = Vec::with_capacity(6);
    for sign in [-1., 1.] {
        let corner = Vec2::new(sign * half.x, -half.y);
        if rng.random_bool(0.62) {
            let a = corner - Vec2::X * sign * rng.random_range(0.50..1.22);
            let b = corner + Vec2::Y * rng.random_range(0.45..1.16);
            corners.extend(if sign < 0. { [b, a] } else { [a, b] });
        } else {
            corners.push(corner);
        }
    }
    corners.extend([half, Vec2::new(-half.x, half.y)]);
    corners
}
