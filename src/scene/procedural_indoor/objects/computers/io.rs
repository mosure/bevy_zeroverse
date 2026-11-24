//! Device-local sockets. A single USB-C receptacle is the mating point for
//! both charging and docked display routes. Animating the lid or changing the
//! housing finish leaves the installed edge connectors alone.
use super::*;
mod mesh;
#[cfg(test)]
mod tests;

pub(crate) const FITTED_IO: u32 = 1 << 8;
const MAX_PORTS: u32 = 6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Connector {
    UsbC,
    UsbA,
    Hdmi,
    Audio,
}

impl Connector {
    /// Nominal aperture width, height and pocket depth in metres.
    fn dimensions(self) -> (Vec2, f32) {
        match self {
            Self::UsbC => (Vec2::new(0.0086, 0.0031), 0.0061),
            Self::UsbA => (Vec2::new(0.0131, 0.0055), 0.0085),
            Self::Hdmi => (Vec2::new(0.0145, 0.0050), 0.0070),
            Self::Audio => (Vec2::splat(0.0038), 0.0085),
        }
    }
}

#[derive(Clone, Copy, Debug, serde::Serialize)]
pub struct Port {
    pub connector: Connector,
    /// Mouth centre in object coordinates. The sign of X selects the wall.
    pub center_m: Vec3,
}

#[derive(Debug, serde::Serialize)]
pub struct Bank {
    /// Charging/dock socket first, followed by the fitted auxiliary sockets.
    pub ports: Vec<Port>,
}

/// Sample the dock by itself: routing never has to build or pack the other ports.
fn dock_site(seed: u64) -> (f32, f32) {
    let mut r = stream(seed, u64::from_le_bytes(*b"DockSite"));
    let side = if r.random_bool(0.5) { -1. } else { 1. };
    (side, r.random_range(0.14..0.62))
}

fn dock(o: &IndoorObject, p: &LaptopProgram, deck: Deck) -> Port {
    let (side, station) = dock_site(o.seed);
    Port {
        connector: Connector::UsbC,
        center_m: Vec3::new(
            side * o.size.x * 0.5,
            0.003 + p.chassis_m * 0.5,
            deck.rear + deck.depth * station,
        ),
    }
}

pub(crate) fn charge_normal(o: &IndoorObject) -> Vec3 {
    if o.variant & FITTED_IO == 0 {
        Vec3::X
    } else {
        Vec3::X * dock_site(o.seed).0
    }
}

pub(crate) fn charge_port(o: &IndoorObject) -> Vec3 {
    let p = parameters(o);
    let deck = Deck::fit(o, &p);
    if o.variant & FITTED_IO != 0 {
        dock(o, &p, deck).center_m
    } else {
        // Older manifests archive leads to the first printed mark on the right.
        Vec3::new(
            o.size.x * 0.5 + 0.001,
            p.chassis_m * 0.55 + 0.003,
            deck.rear + deck.depth * 0.17,
        )
    }
}

impl Bank {
    pub(super) fn fit(o: &IndoorObject, p: &LaptopProgram, deck: Deck) -> Option<Self> {
        if o.variant & FITTED_IO == 0 {
            return None;
        }
        let mut r = stream(o.seed, u64::from_le_bytes(*b"IoLayout"));
        let charge = dock(o, p, deck);
        let count = p.port_count.clamp(1, MAX_PORTS);
        let mut ports = Vec::with_capacity(count as usize);
        ports.push(charge);
        let flat_height = p.chassis_m - 2. * p.corner_m.min(0.002);
        let min_z = deck.rear + p.corner_m + 0.002;
        let max_z = deck.rear + deck.depth - p.corner_m - 0.002;
        // A device can concentrate connectors on one edge, or populate both.
        let same_side = r.random_range(0.19..0.80);
        for _ in 1..count {
            let choice = [
                Connector::UsbC,
                Connector::UsbA,
                Connector::Hdmi,
                Connector::Audio,
            ][r.random_range(0..4)];
            let unique = matches!(choice, Connector::Hdmi | Connector::Audio);
            let connector = if choice.dimensions().0.y + 0.0014 > flat_height
                || unique && ports.iter().any(|p| p.connector == choice)
            {
                Connector::UsbC
            } else {
                choice
            };
            let size = connector.dimensions().0;
            let headroom = (flat_height - size.y - 0.0008).max(0.);
            let y = charge.center_m.y + r.random_range(-0.35..0.35) * headroom;
            let gap = r.random_range(0.005..0.014);
            let favored = charge.center_m.x * if r.random_bool(same_side) { 1. } else { -1. };
            let lo = min_z + size.x * 0.5;
            let hi = max_z - size.x * 0.5;
            if hi <= lo {
                continue;
            }
            for attempt in 0..8 {
                let x = if attempt < 4 { favored } else { -favored };
                let z = r.random_range(lo..hi);
                if ports.iter().all(|p| {
                    p.center_m.x != x
                        || (p.center_m.z - z).abs()
                            > gap + (size.x + p.connector.dimensions().0.x) * 0.5
                }) {
                    ports.push(Port {
                        connector,
                        center_m: Vec3::new(x, y, z),
                    });
                    break;
                }
            }
        }
        Some(Self { ports })
    }
}
