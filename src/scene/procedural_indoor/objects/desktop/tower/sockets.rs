//! Socket lips, reveals and contacts share one opening with the cut chassis.
//! The cable router needs only the rear layout, independent of any mesh build.
use super::face::{clipped, corners, Plane};
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;
use ports::TowerJack;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum Connector {
    Button,
    UsbA,
    UsbC,
    Video,
    Audio,
    Appliance,
    Barrel,
}
impl Connector {
    pub(super) fn size(self) -> Vec2 {
        use crate::scene::procedural_indoor::objects::computers::io::Connector as Io;
        match self {
            Self::UsbA => Io::UsbA.dimensions().0,
            Self::UsbC => Io::UsbC.dimensions().0,
            Self::Video => Io::Hdmi.dimensions().0,
            Self::Audio => Io::Audio.dimensions().0,
            Self::Button => Vec2::splat(0.010),
            Self::Barrel => Vec2::splat(0.0075),
            Self::Appliance => Vec2::new(0.025, 0.017),
        }
    }
    pub(super) fn depth(self) -> f32 {
        match self {
            Self::Button => 0.0014,
            Self::Appliance => 0.009,
            Self::Barrel | Self::Audio => 0.0055,
            Self::UsbA => 0.008,
            Self::UsbC => 0.006,
            Self::Video => 0.007,
        }
    }
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub(crate) struct Socket {
    pub connector: Connector,
    pub aperture: Rect,
}
impl Socket {
    pub(super) fn new(connector: Connector, center: Vec2) -> Self {
        Self {
            connector,
            aperture: Rect::from_center_size(center, connector.size()),
        }
    }
    pub(super) fn build(
        self,
        a: &mut Assembly,
        plane: Plane,
        stock: &Housing,
        case: &str,
        trim: &str,
    ) {
        let r = self.aperture;
        let half = r.size().min_element() * 0.5;
        let cut = half
            * match self.connector {
                Connector::Button | Connector::Barrel | Connector::Audio => 0.58,
                Connector::UsbC => 0.76,
                Connector::Video | Connector::Appliance => 0.48,
                Connector::UsbA => 0.13,
            };
        let mut rim = clipped(r, cut);
        if matches!(self.connector, Connector::Video | Connector::Appliance) {
            // Only one edge is keyed on the stamped shell.
            let top = clipped(r, cut * 0.18);
            rim[3..7].copy_from_slice(&top[3..7]);
        }
        for (k, corner) in corners(r).into_iter().enumerate() {
            let i = 2 * k;
            cap_triangle(
                a.part(stock.surface, case),
                plane,
                [corner, rim[i], rim[(i + 7) % 8]],
                0.,
            );
        }
        let metal = self.connector != Connector::Button;
        let outer = r.size();
        let lip = if self.connector == Connector::Appliance {
            0.0016
        } else {
            0.00042
        };
        let throat = rim.map(|p| r.center() + (p - r.center()) * (Vec2::ONE - 2. * lip / outer));
        let d = self.connector.depth();
        let depth = if metal { d } else { 0.00055 };
        for i in 0..8 {
            let j = (i + 1) % 8;
            let direction = plane.point(r.center(), 0.) - plane.point((rim[i] + rim[j]) * 0.5, 0.);
            let l = if metal {
                a.part(Surface::Metal, "pc_tower")
            } else {
                a.part(Surface::Plastic, trim)
            };
            l.metric_quad(
                [
                    plane.point(rim[i], 0.),
                    plane.point(rim[j], 0.),
                    plane.point(throat[j], 0.0002),
                    plane.point(throat[i], 0.0002),
                ],
                plane.normal(),
            );
            a.part(Surface::Rubber, "pc_tower").metric_quad(
                [
                    plane.point(throat[i], 0.0002),
                    plane.point(throat[j], 0.0002),
                    plane.point(throat[j], depth),
                    plane.point(throat[i], depth),
                ],
                direction,
            );
            let cap = if metal {
                a.part(Surface::Rubber, "pc_tower")
            } else {
                a.part(Surface::Plastic, trim)
            };
            cap_triangle(cap, plane, [r.center(), throat[i], throat[j]], depth);
        }
        if matches!(
            self.connector,
            Connector::UsbA | Connector::UsbC | Connector::Video
        ) {
            let y = if self.connector == Connector::UsbA {
                outer.y * 0.19
            } else {
                0.
            };
            let center = plane.point(r.center() + Vec2::Y * y, d * 0.70);
            a.box_part(
                Surface::Plastic,
                trim,
                center,
                Vec3::new(outer.x * 0.69, outer.y * 0.17, d * 0.60),
                0.,
            );
        }
    }
}

pub(super) fn cap_triangle(g: &mut Geometry, plane: Plane, uv: [Vec2; 3], d: f32) {
    let p = uv.map(|q| plane.point(q, d));
    let outward = plane.normal();
    let b = g.positions.len() as u32;
    g.positions.extend(p.map(|v| v.to_array()));
    g.normals.extend([outward.to_array(); 3]);
    g.uvs.extend(uv.map(|v| v.to_array()));
    g.indices
        .extend(if (p[1] - p[0]).cross(p[2] - p[0]).dot(outward) > 0. {
            [b, b + 1, b + 2]
        } else {
            [b, b + 2, b + 1]
        });
}

struct RearLayout {
    power_sign: f32,
    lateral: f32,
    height: f32,
    spacing: f32,
    usb_high: bool,
}
impl RearLayout {
    fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"RearJack"));
        Self {
            power_sign: if r.random_bool(0.5) { 1. } else { -1. },
            lateral: r.random_range(0.09..0.16),
            height: r.random_range(0.15..0.23),
            spacing: r.random_range(0.019..0.024),
            usb_high: r.random_bool(0.67),
        }
    }
    fn socket(&self, o: &IndoorObject, j: TowerJack) -> Socket {
        let compact = o.size.y < o.size.x * 0.67;
        let kind = match j {
            TowerJack::Mains if compact => Connector::Barrel,
            TowerJack::Mains => Connector::Appliance,
            TowerJack::Video(_) => Connector::Video,
            _ => Connector::UsbA,
        };
        let center = if compact {
            let i = match j {
                TowerJack::Mains => -1.,
                TowerJack::Video(false) => 0.,
                TowerJack::Video(true) => 1.,
                TowerJack::Keyboard => 2.,
                TowerJack::Mouse => 3.,
            };
            let x = if i < 0. {
                -o.size.x * (0.20 + self.lateral)
            } else {
                (i - 1.5) * self.spacing + o.size.x * self.lateral
            };
            Vec2::new(
                x * self.power_sign,
                o.size.y * (0.38 + self.height + if i >= 2. { 0.055 } else { 0. }),
            )
        } else {
            let (x, y) = match j {
                TowerJack::Mains => (-o.size.x * self.lateral, self.height),
                TowerJack::Video(b) => (
                    o.size.x * self.lateral + (if b { 0.5 } else { -0.5 }) * self.spacing,
                    0.44 + self.height * 0.18,
                ),
                TowerJack::Keyboard | TowerJack::Mouse => (
                    o.size.x * self.lateral
                        + (if matches!(j, TowerJack::Mouse) {
                            0.5
                        } else {
                            -0.5
                        }) * self.spacing,
                    0.72 + self.height * 0.15,
                ),
            };
            Vec2::new(
                x * self.power_sign,
                o.size.y * if self.usb_high { y } else { 1. - y },
            )
        };
        Socket::new(kind, center)
    }
}

pub(crate) fn rear_center(o: &IndoorObject, j: TowerJack) -> Vec3 {
    RearLayout::sample(o)
        .socket(o, j)
        .aperture
        .center()
        .extend(-o.size.z * 0.5)
}
pub(super) fn rear_sockets(o: &IndoorObject) -> [Socket; 5] {
    let layout = RearLayout::sample(o);
    TowerJack::ALL.map(|j| layout.socket(o, j))
}
pub(super) fn rear(
    a: &mut Assembly,
    o: &IndoorObject,
    plane: Plane,
    face: Rect,
    stock: &Housing,
    case: &str,
) {
    let sockets = rear_sockets(o);
    let exhaust = super::exhaust::Exhaust::fit(o, face, &sockets);
    let mut cuts = [Rect::default(); 7];
    for (c, s) in cuts.iter_mut().zip(sockets) {
        *c = s.aperture;
    }
    let mut n = 5;
    if let Some(vent) = exhaust {
        cuts[n] = vent.opening;
        n += 1;
        if let Some(r) = vent.expansion {
            cuts[n] = r;
            n += 1;
        }
    }
    plane.sheet(a.part(stock.surface, case), face, &cuts[..n]);
    let trim = stock.trim_label(o);
    for socket in sockets {
        socket.build(a, plane, stock, case, &trim);
    }
    if let Some(vent) = exhaust {
        let grille = crate::scene::procedural_indoor::materials::speaker::perforated_pc(o);
        vent.build(a, plane, stock, case, grille);
    }
}
