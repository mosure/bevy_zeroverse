//! Indexed outlet positions, in the same order as the rendered inserts.
use super::super::floor::service;
use super::*;

pub(crate) enum MainsBank {
    Wall(super::super::wall::OutletBank),
    Floor(service::FloorBox),
    Strip(power_strip::StripProgram),
}
impl MainsBank {
    pub fn for_target(target: &IndoorObject) -> Option<Self> {
        match target.kind {
            ObjectKind::WallOutlet => super::super::wall::parameters(target).bank.map(Self::Wall),
            ObjectKind::FloorOutlet => service::parameters(target)
                .filter(|p| p.open)
                .map(Self::Floor),
            ObjectKind::PowerStrip => Some(Self::Strip(power_strip::parameters(target))),
            _ => None,
        }
    }
    pub fn socket_count(&self) -> usize {
        match self {
            Self::Wall(b) => b.socket_count(),
            Self::Floor(b) => b.socket_count,
            Self::Strip(b) => b.socket_count,
        }
    }
    pub fn local(&self, size: Vec3, index: usize) -> Option<Vec3> {
        match self {
            Self::Wall(b) => b.socket_local(index),
            Self::Floor(b) => b.port(index),
            Self::Strip(b) => b.mains_port(size, index),
        }
    }
}

/// A routed plug enters a well only on an indexed mains connection.
pub(crate) fn recessed_connection(target: &IndoorObject, socket: Option<u8>) -> bool {
    socket.is_some()
        && match target.kind {
            ObjectKind::FloorOutlet => {
                service::face(target) == Some(super::super::wall::OutletFace::RoundWell)
            }
            ObjectKind::WallOutlet => target.variant % 3 == 1,
            ObjectKind::PowerStrip => {
                super::super::wall::OutletFace::from_variant(target.variant)
                    == Some(super::super::wall::OutletFace::RoundWell)
            }
            _ => false,
        }
}

pub(crate) fn target_port(
    target: &IndoorObject,
    low_voltage: bool,
    socket: Option<u8>,
) -> Option<Vec3> {
    let Some(index) = socket else {
        if service::installed(target) {
            return None;
        }
        return super::power_port(target, low_voltage);
    };
    if low_voltage {
        return None;
    }
    let local = MainsBank::for_target(target)?.local(target.size, usize::from(index))?;
    Some(target.transform().transform_point(local))
}
