//! Convex, piecewise-linear inner wall and rim used by the writing-tool contact solver.
use super::*;

pub(super) struct Edge {
    pub start: Vec2,
    pub end: Vec2,
    pub normal: Vec2,
    pub distance: f32,
}

pub(super) struct Pocket {
    pub edges: Vec<Edge>,
    pub span: Vec2,
    pub floor: f32,
    pub height: f32,
    pub belly_y: f32,
    pub base: f32,
    pub mouth: f32,
    pub wall: f32,
    pub count: usize,
    pub heading: f32,
    pub spread: f32,
}

impl Pocket {
    pub fn rectangle(span: Vec2, floor: f32, height: f32, r: &mut impl Rng) -> Self {
        let half = span * 0.5;
        let corners = [
            -half,
            Vec2::new(half.x, -half.y),
            half,
            Vec2::new(-half.x, half.y),
        ];
        let edges = (0..4)
            .map(|i| {
                let start = corners[i];
                let end = corners[(i + 1) % 4];
                let delta = end - start;
                let normal = Vec2::new(delta.y, -delta.x).normalize();
                Edge {
                    start,
                    end,
                    normal,
                    distance: normal.dot(start),
                }
            })
            .collect();
        Self {
            edges,
            span,
            floor,
            height,
            belly_y: 0.5 * (height + floor),
            base: 1.,
            mouth: 1.,
            wall: 0.,
            count: r.random_range(1..=(span.x / 0.012).floor().clamp(1., 3.) as usize),
            heading: r.random_range(0.0..std::f32::consts::TAU),
            spread: r.random_range(0.35..std::f32::consts::PI),
        }
    }
    /// The two linear sections admit an exact half-space test at their endpoints.
    pub fn inner_scale(&self, y: f32) -> f32 {
        let fraction = if y <= self.belly_y {
            self.base + (1. - self.base) * (y - self.floor) / (self.belly_y - self.floor)
        } else {
            1. + (self.mouth - 1.) * (y - self.belly_y) / (self.height - self.belly_y)
        };
        fraction - self.wall
    }
}
