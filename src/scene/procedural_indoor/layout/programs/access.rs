//! Standing/browsing space belongs to placement only. Rays and cameras see the actual millwork.
use super::*;
use crate::scene::procedural_indoor::footprint::Footprint;
use crate::scene::procedural_indoor::objects::public::furnishing::lectern;

fn area(o: &IndoorObject) -> Option<(Vec3, Vec3)> {
    let (side, front, rear) = match o.kind {
        ObjectKind::Lectern if lectern::installed(o) => (
            ((0.70 - o.size.x) * 0.5).max(0.06),
            lectern::standing_depth(o),
            0.,
        ),
        ObjectKind::DrinkingFountain => (0.075, 0.77, 0.),
        ObjectKind::WaterCooler => (
            0.075,
            crate::scene::procedural_indoor::objects::public::cooler::approach_m(o.seed),
            0.055,
        ),
        ObjectKind::LibraryStack => {
            let depth =
                crate::scene::procedural_indoor::objects::storage::library_stack::browsing_depth_m(
                    o,
                );
            (0.08, depth, depth)
        }
        ObjectKind::Workbench => (
            0.09,
            0.76 + 0.34 * f32::from((o.seed >> 24) as u16) / f32::from(u16::MAX),
            0.055,
        ),
        ObjectKind::ArtistEasel => (
            0.09,
            crate::scene::procedural_indoor::objects::public::easel::working_clearance(o),
            0.04,
        ),
        ObjectKind::ReceptionDesk => {
            let fraction = |bits: u16| f32::from(bits) / f32::from(u16::MAX);
            (
                0.08,
                0.90 + 0.40 * fraction(o.seed as u16),
                0.87 + 0.26 * fraction((o.seed >> 32) as u16),
            )
        }
        _ => return None,
    };
    let size = Vec3::new(
        o.size.x + 2. * side,
        o.size.y.max(1.90),
        o.size.z + (front + rear),
    );
    let shift = (front - rear) * 0.5;
    let center = if shift == 0. {
        o.position
    } else {
        o.transform().transform_point(Vec3::Z * shift)
    };
    let center = if o.kind == ObjectKind::DrinkingFountain {
        center
            - Vec3::Y
                * crate::scene::procedural_indoor::objects::public::fountain::mount_height_m(o.seed)
    } else {
        center
    };
    Some((center, size))
}

pub(in crate::scene::procedural_indoor::layout) fn envelope(
    o: &IndoorObject,
) -> Option<IndoorObject> {
    let (position, size) = area(o)?;
    Some(IndoorObject {
        position,
        size,
        ..o.clone()
    })
}

/// A later aisle or door approach must leave the working face of an installed fixture usable.
pub(in crate::scene::procedural_indoor::layout) fn bounds(o: &IndoorObject) -> (Vec3, Vec3) {
    envelope(o).as_ref().unwrap_or(o).bounds()
}

pub(in crate::scene::procedural_indoor::layout) fn blocks(
    a: &IndoorObject,
    b: &IndoorObject,
) -> bool {
    fn one(fixture: &IndoorObject, other: &IndoorObject) -> bool {
        let Some((center, size)) = area(fixture) else {
            return false;
        };
        if !other.solid
            || other.support.is_some()
            || other.position.y + other.size.y <= center.y + 0.03
            || other.position.y >= center.y + size.y
        {
            return false;
        }
        if fixture.kind == ObjectKind::ReceptionDesk
            && other.kind == ObjectKind::Chair
            && other.interaction_target == Some(fixture.id)
        {
            let offset = other.position - fixture.position;
            if offset.dot(fixture.transform().rotation * Vec3::Z) < -fixture.size.z * 0.5 {
                return false;
            }
        }
        Footprint::oriented(center.xz(), size.xz(), fixture.yaw)
            .overlaps(Footprint::object(other), 0.)
    }
    one(a, b) || one(b, a)
}
