//! Physical planar mounting faces. Closed beams and rib undersides can support
//! CCTV; the nominal roof behind them cannot. Open trusses keep their collision
//! reservations but are not approximated as solid mounting plates.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::ceiling::framing::{FrameMember, FrameSection},
    layout::IndoorManifest,
};

#[derive(Clone, Copy)]
struct Patch {
    origin: Vec3,
    along: Vec3,
    across: Vec3,
}
impl Patch {
    fn underside(member: &FrameMember) -> Option<Self> {
        let [lo, hi] = match member.section {
            FrameSection::OpenWeb => return None,
            FrameSection::Fluted => [0.22, 0.78],
            FrameSection::Solid | FrameSection::ISection => [0., 1.],
        };
        let origin = member.point(0., lo, member.depth);
        Some(Self {
            origin,
            along: member.point(1., lo, member.depth) - origin,
            across: member.point(0., hi, member.depth) - origin,
        })
    }
    fn at(self, p: Vec2) -> Option<(Vec3, Vec3)> {
        let offset = p - self.origin.xz();
        let t = offset.dot(self.along.xz()) / self.along.xz().length_squared();
        let u = offset.dot(self.across.xz()) / self.across.xz().length_squared();
        if !(0.0..=1.).contains(&t) || !(0.0..=1.).contains(&u) {
            return None;
        }
        let normal = self.along.cross(self.across).normalize();
        Some((
            self.origin + self.along * t + self.across * u,
            if normal.y > 0. { -normal } else { normal },
        ))
    }
    fn bounds(self) -> (Vec2, Vec2) {
        let opposite = self.origin + self.along + self.across;
        (
            self.origin.xz().min(opposite.xz()),
            self.origin.xz().max(opposite.xz()),
        )
    }
}

fn patches(scene: &IndoorManifest) -> impl Iterator<Item = Patch> + '_ {
    scene
        .ceiling_installation
        .iter()
        .filter_map(|c| c.framing.as_ref())
        .flat_map(|f| &f.members)
        .filter_map(Patch::underside)
}

/// A lower bound on the vertical distance between a ceiling mount and a room
/// origin. Use the lowest supporting face and highest floor anywhere in the
/// room, even if those two points could never be used together.
pub(super) fn impossible_metric_baseline(
    scene: &IndoorManifest,
    max_standoff: f32,
    limit: f32,
) -> Option<f32> {
    use super::super::viewpoint::MAX_EYE_HEIGHT_M;
    if scene.room_size.y - max_standoff - MAX_EYE_HEIGHT_M <= limit {
        return None;
    }
    let half = scene.room_size.xz() * 0.5;
    let mut face = scene.minimum_ceiling_height(-half, half);
    if let Some(grid) =
        crate::scene::procedural_indoor::architecture::ceiling::CeilingGrid::for_scene(scene)
    {
        face = face.min(scene.room_size.y - grid.tile.drop_m);
    } else {
        for patch in patches(scene) {
            let low = patch.origin.y + patch.along.y.min(0.) + patch.across.y.min(0.);
            face = face.min(low);
        }
    }
    let mut floor = scene.rug_supports.iter().map(|r| r.top).fold(0., f32::max);
    if let Some(e) = &scene.envelope {
        floor = e
            .floor_patches
            .iter()
            .map(|p| p.height)
            .fold(floor, f32::max);
        if let Some(m) = &e.mezzanine {
            floor = floor.max(m.deck.height);
        }
    }
    let separation = face - max_standoff - (floor + MAX_EYE_HEIGHT_M);
    (separation > limit + 1e-4).then_some(separation)
}

pub(super) fn at(scene: &IndoorManifest, p: Vec2) -> (Vec3, Vec3) {
    if let Some(grid) =
        crate::scene::procedural_indoor::architecture::ceiling::CeilingGrid::for_scene(scene)
    {
        return (
            Vec3::new(p.x, grid.mount_height(scene, p), p.y),
            Vec3::NEG_Y,
        );
    }
    let height = scene.ceiling_height(p);
    let normal = scene.envelope.as_ref().map_or(Vec3::NEG_Y, |e| {
        if height < e.roof_height(scene.room_size, p) - 1e-5 {
            Vec3::NEG_Y
        } else {
            Vec3::new(
                -e.ceiling_drop.x / scene.room_size.x,
                -1.,
                -e.ceiling_drop.y / scene.room_size.z,
            )
            .normalize()
        }
    });
    patches(scene).filter_map(|patch| patch.at(p)).fold(
        (Vec3::new(p.x, height, p.y), normal),
        |lowest, candidate| {
            if candidate.0.y < lowest.0.y {
                candidate
            } else {
                lowest
            }
        },
    )
}

/// Frame edges are support discontinuities, just like soffit edges. Testing
/// both sides of every crossing prevents a rail bridging an unmounted gap,
/// including a gap too thin to be found by uniform temporal samples.
pub(super) fn rail_boundaries(scene: &IndoorManifest) -> impl Iterator<Item = (Vec2, Vec2)> + '_ {
    patches(scene).map(Patch::bounds)
}

#[cfg(test)]
mod tests;
