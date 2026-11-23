//! Wall-hung sliding doors. The panel, its two trolleys and its end stops all
//! use the same longitudinal travel and transverse track centre.
use super::*;
mod construction;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
pub(super) struct Lane {
    closed: f32,
    travel: f32,
    pub(super) depth: f32,
    pub(super) leaf_width: f32,
}

impl Lane {
    pub(super) fn x_at(self, opening: f32) -> f32 {
        self.closed + self.travel * opening
    }
}

pub(super) fn lanes(
    d: &DoorInstallation,
) -> impl DoubleEndedIterator<Item = Lane> + ExactSizeIterator + '_ {
    let p = &d.program;
    let count = if p.mechanism == DoorMechanism::Telescoping {
        2
    } else {
        1
    };
    // An overlay panel closes *over* its jambs. Size the laps to the setback
    // from the wall and the separation of the two planes at a meeting stile.
    let spacing = p.thickness + 0.014;
    let setback = 0.030 + (count - 1) as f32 * spacing;
    let outer_edge = d.width * 0.5 + 0.016 + 0.50 * setback;
    let meeting_lap = 0.018 + 0.65 * spacing;
    let leaf_width = if count == 1 {
        2. * outer_edge
    } else {
        outer_edge + meeting_lap * 0.5
    };
    // Even the trailing edge clears the jamb when both panels are parked.
    let parked = p.handedness * (d.width * 0.5 + 0.008 + leaf_width * 0.5);
    (0..count).map(move |i| {
        let closed = if count == 1 {
            0.
        } else {
            (2. * i as f32 - 1.) * (outer_edge - leaf_width * 0.5)
        };
        Lane {
            closed,
            travel: parked - closed,
            // The wallward leaf passes 22 mm partition skirting throughout its run.
            depth: d.wall_depth * 0.5 + 0.030 + p.thickness * (i as f32 + 0.5) + i as f32 * 0.014,
            leaf_width,
        }
    })
}

pub(super) struct Track {
    /// Cross-section dimensions in metres. Height is relative to the panel head.
    half_width: f32,
    slot_half: f32,
    gauge: f32,
    drop: f32,
    radius: f32,
    inset: f32,
    fascia: bool,
}

/// Full panel sweep along the wall, including space to pass the parking jamb.
/// Its far side depends on both handedness and the number of stacked leaves.
pub(super) fn parking_span(d: &DoorInstallation) -> Vec2 {
    let end = d.width * 0.5 + 0.06;
    let mut span = Vec2::new(-end, end);
    for lane in lanes(d) {
        let half = lane.leaf_width * 0.5 + 0.02;
        let park = lane.x_at(1.);
        span.x = span.x.min(lane.closed.min(park) - half);
        span.y = span.y.max(lane.closed.max(park) + half);
    }
    span
}

const STOP_LENGTH: f32 = 0.012;
const CLEAT_HEIGHT: f32 = 0.008;
// Behind the moving leaves, the backplate must bear on the jamb header.
const BACKPLATE_BOTTOM: f32 = -0.016;

impl Track {
    pub(super) fn for_program(p: &DoorProgram) -> Option<Self> {
        if !matches!(
            p.mechanism,
            DoorMechanism::Sliding | DoorMechanism::Telescoping
        ) {
            return None;
        }
        // Use immutable stock measurements so archived or animated doors
        // retain their hardware when the opening, location or direction changes.
        let stock = u64::from(p.thickness.to_bits()) << 32 | u64::from(p.stile.to_bits());
        let mut r = stream(stock, 0x534c_4944_4552);
        Some(Self {
            half_width: p.thickness * r.random_range(0.38..0.45),
            slot_half: p.thickness * 0.105,
            gauge: r.random_range(0.0024..0.0036),
            drop: r.random_range(0.014..0.027),
            radius: r.random_range(0.011..0.018),
            inset: r.random_range(0.085..0.16),
            fascia: r.random_bool(0.48),
        })
    }

    fn hanger_offset(&self, width: f32) -> f32 {
        width * 0.5 - self.inset.min(width * 0.23)
    }

    fn axle_height(&self) -> f32 {
        self.drop + self.gauge + self.radius
    }

    fn top(&self) -> f32 {
        self.axle_height() + self.radius + 0.004 + self.gauge
    }

    fn stops(&self, lane: Lane) -> Vec2 {
        let closed = lane.x_at(0.);
        let open = lane.x_at(1.);
        let end = self.hanger_offset(lane.leaf_width) + self.radius + 0.002 + STOP_LENGTH * 0.5;
        Vec2::new(closed.min(open) - end, closed.max(open) + end)
    }

    fn span(&self, d: &DoorInstallation) -> Vec2 {
        // Reach both supporting jambs, but extend over the parking side only
        // as far as the last trolley and its bumper actually travel.
        let end = d.width * 0.5 + 0.031;
        let mut span = Vec2::new(-end, end);
        for lane in lanes(d) {
            let stops = self.stops(lane);
            span.x = span.x.min(stops.x - STOP_LENGTH * 0.5 - 0.005);
            span.y = span.y.max(stops.y + STOP_LENGTH * 0.5 + 0.005);
        }
        span
    }

    fn outer_depth(&self, d: &DoorInstallation) -> f32 {
        lanes(d).next_back().unwrap().depth + self.half_width + if self.fascia { 0.009 } else { 0. }
    }

    fn head(d: &DoorInstallation) -> f32 {
        d.height + FLOOR_GAP
    }

    pub(super) fn overhead_bounds(&self, d: &DoorInstallation) -> (Vec3, Vec3) {
        let span = self.span(d);
        let head = Self::head(d);
        let bottom = head + BACKPLATE_BOTTOM;
        let top = head + self.top() + CLEAT_HEIGHT;
        let inside = d.wall_depth * 0.5;
        let outside = self.outer_depth(d);
        let tf = d.transform();
        Leaf {
            transform: tf.with_translation(tf.transform_point(Vec3::new(
                (span.x + span.y) * 0.5,
                (bottom + top) * 0.5,
                d.program.swing_side * (inside + outside) * 0.5,
            ))),
            size: Vec3::new(span.y - span.x, top - bottom, outside - inside),
        }
        .bounds(0.001)
    }
}
