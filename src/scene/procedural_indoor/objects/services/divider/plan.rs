//! Hinge centres and support positions in the admitted floor footprint (X,Z).
use super::*;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
pub enum Arrangement {
    Single,
    Straight,
    Angled,
    Curved,
    Zigzag,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct Plan {
    pub arrangement: Arrangement,
    /// Clear distance at a straight joint, including the middle pin.
    pub joint_gap_m: f32,
    /// Left outside end, interior hinge centres, then right outside end in object-local metres.
    pub hinge_centres_xz: Vec<Vec2>,
}

#[derive(Clone, Copy)]
pub(super) struct Bay {
    pub frame: Transform,
    pub width: f32,
}

impl Plan {
    pub(super) fn sample(o: &IndoorObject, weights: &[f32]) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"DivHinge"));
        let n = weights.len();
        let arrangement = if n == 1 {
            Arrangement::Single
        } else if r.random_bool(0.29) {
            Arrangement::Straight
        } else if n == 2 {
            Arrangement::Angled
        } else if r.random_bool(0.53) {
            Arrangement::Zigzag
        } else {
            Arrangement::Curved
        };
        let sign = if r.random_bool(0.5) { 1. } else { -1. };
        let wave = match arrangement {
            Arrangement::Angled => [-1., 1., 0.],
            Arrangement::Zigzag => [1., -1., 1.],
            Arrangement::Curved => [-1., r.random_range(-0.24..0.24), 1.],
            Arrangement::Straight | Arrangement::Single => [0.; 3],
        };
        let width = o.size.x * 0.955;
        let total: f32 = weights.iter().sum();
        let mut points = Vec::with_capacity(n + 1);
        let mut point = Vec2::new(-0.5 * width, 0.);
        points.push(point);
        let mut lo = 0_f32;
        let mut hi = 0_f32;
        for (weight, wave) in weights.iter().zip(wave) {
            let run = width * weight / total;
            let slope = sign * wave * r.random_range(0.21..0.76);
            point += Vec2::new(run, slope * run);
            lo = lo.min(point.y);
            hi = hi.max(point.y);
            points.push(point);
        }
        // A room can reserve a shallow footprint for an otherwise wide screen.
        // Compress the excursion, leaving sufficient depth for standing feet.
        let max_excursion = o.size.z * r.random_range(0.32..0.53);
        let scale = max_excursion / (hi - lo).max(max_excursion);
        for p in &mut points {
            p.y = (p.y - 0.5 * (hi + lo)) * scale;
        }
        Self {
            arrangement,
            joint_gap_m: r.random_range(0.006..0.013),
            hinge_centres_xz: points,
        }
    }

    pub(super) fn axis(&self, i: usize) -> Vec2 {
        (self.hinge_centres_xz[i + 1] - self.hinge_centres_xz[i]).normalize()
    }

    /// Signed turn between the bays on either side of an interior hinge.
    pub(crate) fn joint_angle(&self, i: usize) -> f32 {
        let vertices = &self.hinge_centres_xz;
        let before = vertices[i] - vertices[i - 1];
        let after = vertices[i + 1] - vertices[i];
        before.perp_dot(after).atan2(before.dot(after))
    }

    pub(super) fn bay(&self, i: usize, frame_depth: f32) -> Bay {
        let points = &self.hinge_centres_xz;
        let axis = self.axis(i);
        // The two inner corners approach one another when the panels turn.
        // Shorten each square-cut stile by exactly half the bend allowance.
        let setback = |j| {
            if j == 0 || j + 1 == points.len() {
                0.
            } else {
                0.5 * (self.joint_gap_m + frame_depth * (0.5 * self.joint_angle(j)).abs().tan())
            }
        };
        let a = points[i] + axis * setback(i);
        let b = points[i + 1] - axis * setback(i + 1);
        let mid = (a + b) * 0.5;
        Bay {
            frame: Transform::from_xyz(mid.x, 0., mid.y)
                .with_rotation(Quat::from_rotation_y(-axis.y.atan2(axis.x))),
            width: (b - a).length(),
        }
    }

    /// Keep a support out of a hinge gap, fully under the bearing rail.
    pub(super) fn support(&self, x: f32, depth: f32, margin: f32) -> (Vec2, f32) {
        let last = self.hinge_centres_xz.len() - 2;
        let i = self
            .hinge_centres_xz
            .windows(2)
            .position(|q| x <= q[1].x)
            .unwrap_or(last);
        let bay = self.bay(i, depth);
        let axis = self.axis(i);
        let center = bay.frame.translation.xz();
        let reach = (bay.width * 0.5 - margin).max(0.);
        let t = ((x - center.x) / axis.x).clamp(-reach, reach);
        (center + t * axis, -axis.y.atan2(axis.x))
    }
}
