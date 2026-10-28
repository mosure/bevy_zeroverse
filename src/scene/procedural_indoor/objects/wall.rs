//! Wall hardware in a local +Z facing frame. Recesses are inset dark geometry.
use super::*;

/// Root-to-wall offset. Geometry rear faces differ for plates and clock cases.
/// A millimetre gap avoids coplanar faces while retaining a plausible mounting.
pub(crate) fn mount_offset(o: &IndoorObject) -> Option<f32> {
    let fraction = match o.kind {
        ObjectKind::Display
        | ObjectKind::Whiteboard
        | ObjectKind::WallArt
        | ObjectKind::ProjectionScreen
        | ObjectKind::WallShelf
        | ObjectKind::NoticeBoard
        | ObjectKind::WallSconce
        | ObjectKind::FirePullStation => 0.5,
        ObjectKind::ExitSign
            if equipment::signals::exit::mount(o) == equipment::signals::exit::ExitMount::Wall =>
        {
            0.5
        }
        ObjectKind::StringLights if !equipment::signals::string_program(o).suspended => 0.5,
        ObjectKind::FireAlarm => {
            if o.variant % 4 == 3 {
                0.46
            } else {
                0.5
            }
        }
        ObjectKind::Speaker if !o.solid && o.support.is_none() => {
            return Some(accessories::speaker::case_depth(o) * 0.5 + 0.001)
        }
        ObjectKind::Clock => return Some(clocks::case_depth(o) * 0.5 + 0.001),
        ObjectKind::WallOutlet | ObjectKind::LightSwitch => 0.35,
        _ => return None,
    };
    Some(o.size.z * fraction + 0.001)
}

mod fixtures;
pub(super) use fixtures::build as fixture;
pub(crate) use fixtures::outlet_port_local;
pub use fixtures::{parameters, FixtureProgram};
