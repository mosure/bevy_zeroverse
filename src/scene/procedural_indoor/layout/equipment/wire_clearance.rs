//! Reuse each intersecting cable's world-space samples and capsule clearance
//! throughout a route check. Clearance uses both jackets symmetrically.
use super::*;

pub(super) enum Peer<F> {
    Rigid(F),
    Wire(Wire),
}

pub(super) struct Wire {
    lo: Vec3,
    hi: Vec3,
    points: Option<Vec<Vec3>>,
    gap_squared: f64,
    plug: Option<(Vec3, Vec3)>,
}
impl Wire {
    pub fn new(other: &IndoorObject, radius: f32) -> Self {
        let (lo, hi) = other.bounds();
        let tf = other.transform();
        let gap = f64::from(radius + wires::bound_radius(other));
        Self {
            lo: lo - radius,
            hi: hi + radius,
            points: other
                .cable_path
                .as_ref()
                .map(|path| path.iter().map(|&p| tf.transform_point(p)).collect()),
            gap_squared: gap * gap,
            plug: wires::MainsPlug::world(other).map(|plug| {
                let (lo, hi) = plug.bounds();
                (lo - radius, hi + radius)
            }),
        }
    }
    pub fn hits(&self, start: Vec3, end: Vec3) -> bool {
        if start.min(end).cmpgt(self.hi).any() || start.max(end).cmplt(self.lo).any() {
            return false;
        }
        if self
            .plug
            .is_some_and(|(lo, hi)| segment_hits_box(start, end, lo, hi))
        {
            return true;
        }
        self.points.as_ref().is_none_or(|points| {
            points.windows(2).any(|seg| {
                wires::segment_distance_squared(start, end, seg[0], seg[1]) <= self.gap_squared
            })
        })
    }
}

pub(super) fn overlaps((a, b): (Vec3, Vec3), (c, d): (Vec3, Vec3)) -> bool {
    a.cmplt(d).all() && b.cmpgt(c).all()
}

pub(super) fn plug_clear(
    scene: &IndoorManifest,
    cable: &IndoorObject,
    plug: wires::MainsPlug,
    target: &IndoorObject,
    columns: &[(Vec3, Vec3)],
    partitions: &[(Vec3, Vec3)],
    humans: &[(Vec3, Vec3, f32)],
) -> bool {
    let bounds = plug.bounds();
    let (lo, hi) = bounds;
    if columns
        .iter()
        .chain(partitions)
        .any(|&b| overlaps(bounds, b))
        || humans
            .iter()
            .any(|&(a, b, r)| segment_hits_box(a, b, lo - r, hi + r))
    {
        return false;
    }
    let edges = plug.outer_edges();
    for (a, b) in edges {
        if scene.envelope.as_ref().map_or_else(
            || {
                [a, b]
                    .iter()
                    .any(|p| p.xz().abs().cmpgt(scene.room_size.xz() * 0.5).any())
            },
            |env| {
                !super::super::super::envelope::polygon::segment_inside(
                    &env.footprint,
                    a.xz(),
                    b.xz(),
                    0.,
                )
            },
        ) || [a, (a + b) * 0.5, b]
            .iter()
            .any(|p| p.y < scene.floor_height(p.xz()) || p.y > scene.ceiling_height(p.xz()))
        {
            return false;
        }
    }
    !scene.objects.iter().any(|other| {
        if other.id == cable.id || other.id == target.id || !overlaps(bounds, other.bounds()) {
            return false;
        }
        if other.kind != ObjectKind::Cable {
            return true;
        }
        if wires::MainsPlug::world(other).is_some_and(|p| overlaps(bounds, p.bounds())) {
            return true;
        }
        let tf = other.transform();
        let r = wires::bound_radius(other);
        other.cable_path.as_ref().is_none_or(|path| {
            path.windows(2).any(|s| {
                segment_hits_box(
                    tf.transform_point(s[0]),
                    tf.transform_point(s[1]),
                    lo - r,
                    hi + r,
                )
            })
        })
    })
}
