//! Pressed, carved and injection-moulded seats, supported at the chair's existing frame joints.
use super::*;

mod shell;
#[cfg(test)]
mod tests;

const SEAT_Y: f32 = 0.47;
const SIDE_CARRIER_Y: f32 = 0.427;
const SIDE_CARRIER_HEIGHT: f32 = 0.024;
const CROSSBAR_Y: f32 = 0.423;
const CROSSBAR_HEIGHT: f32 = 0.018;

pub(super) fn carrier_underside(style: u8) -> f32 {
    if style == 5 {
        SIDE_CARRIER_Y - SIDE_CARRIER_HEIGHT * 0.5
    } else {
        CROSSBAR_Y - CROSSBAR_HEIGHT * 0.5
    }
}

/// Bring each carrier around the rear cut edge before it rises to the back.
/// A direct rail-to-back diagonal would penetrate the sitting surface.
pub(super) fn back_stays(g: &mut Geometry, width: f32, depth: f32, sockets: [Vec3; 2]) {
    let rear = depth * 0.5 + 0.030;
    for (side, socket) in [-1., 1.].into_iter().zip(sockets) {
        let x = side * width * 0.43;
        let elbow = Vec3::new(x, 0.538, rear);
        let to_socket = socket - elbow;
        // Trim each tangent by the same length before rounding the upper
        // turn. The backrest's offset varies substantially between products.
        let trim = (to_socket.length() * 0.42).min(0.038);
        let incoming = elbow - Vec3::Y * trim;
        let outgoing = elbow + to_socket.normalize() * trim;
        g.tapered_tube(
            &[
                Vec3::new(x, 0.423, depth * 0.35),
                Vec3::new(x, 0.423, rear - 0.028),
                Vec3::new(x, 0.4312, rear - 0.0082),
                Vec3::new(x, 0.451, rear),
                incoming,
                (incoming + 2. * elbow + outgoing) * 0.25,
                outgoing,
                socket,
            ],
            0.011,
            0.0085,
            8,
        );
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum SeatStock {
    PressedPlywood,
    CarvedTimber,
    MouldedPolymer,
}

impl SeatStock {
    pub(super) fn style_id(self) -> u8 {
        match self {
            Self::PressedPlywood => 3,
            Self::CarvedTimber => 4,
            Self::MouldedPolymer => 5,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize)]
pub struct FormedSeat {
    pub stock: SeatStock,
    /// Seats ordered for one table can share tooling while using different cuts of stock.
    pub table_order: bool,
    /// Vertical thickness at the centre of the sitting surface.
    pub gauge_m: f32,
    pub side_rise_m: f32,
    pub rear_rise_m: f32,
    pub nose_drop_m: f32,
    pub corner_radius_m: f32,
    /// Positive narrows the rear, negative narrows the front.
    pub width_taper: f32,
    pub front_sweep_m: f32,
    /// Rotation of the material's metre-space coordinates around the sitting surface.
    pub chart_angle_radians: f32,
}

impl FormedSeat {
    pub(super) fn sample(
        o: &IndoorObject,
        surface: Surface,
        r: &mut rand_chacha::ChaCha8Rng,
    ) -> Option<Self> {
        use SeatStock::*;
        if !matches!(
            (o.variant, surface),
            (3, Surface::Wood) | (4, Surface::Plastic)
        ) {
            return None;
        }
        let mut order = o
            .finish_seed
            .zip(o.interaction_target)
            .filter(|_| r.random_bool(0.82))
            .map(|(lot, table)| {
                stream(
                    lot,
                    0x5345_4154_4c4f_5400 ^ (table as u64).rotate_left(17) ^ u64::from(o.variant),
                )
            });
        let table_order = order.is_some();
        let r = order.as_mut().unwrap_or(r);
        let stock = match surface {
            Surface::Plastic => MouldedPolymer,
            _ if r.random_bool(0.56) => PressedPlywood,
            _ => CarvedTimber,
        };
        let (gauge, rise, drop, corner) = match stock {
            PressedPlywood => (0.008..0.015, 0.008..0.019, 0.009..0.027, 0.043..0.104),
            CarvedTimber => (0.021..0.034, 0.005..0.016, 0.005..0.016, 0.024..0.085),
            MouldedPolymer => (0.0045..0.008, 0.010..0.022, 0.012..0.034, 0.048..0.114),
        };
        Some(Self {
            stock,
            table_order,
            gauge_m: r.random_range(gauge),
            side_rise_m: r.random_range(rise),
            rear_rise_m: r.random_range(0.002..0.007),
            nose_drop_m: r.random_range(drop),
            corner_radius_m: r.random_range(corner),
            width_taper: r.random_range(-0.075..0.15),
            front_sweep_m: r.random_range(0.003..0.036),
            chart_angle_radians: r.random_range(-0.16..0.16)
                + if r.random_bool(0.34) { FRAC_PI_2 } else { 0. },
        })
    }

    /// Height and outward normal at an X/Z location. A pressed sheet follows its
    /// upper surface underneath. Carving leaves more stock at the high edges.
    fn skin(&self, xz: Vec2, half: Vec2, bottom: bool) -> (Vec3, Vec3) {
        let uv = xz / half;
        let rear = uv.y.max(0.);
        let front = (-uv.y).max(0.);
        let mut side_rise = self.side_rise_m;
        let mut rear_rise = self.rear_rise_m;
        let mut drop = self.nose_drop_m;
        if bottom && self.stock == SeatStock::CarvedTimber {
            side_rise *= 0.20;
            rear_rise *= 0.30;
            drop *= 0.45;
        }
        let height = SEAT_Y + side_rise * uv.x * uv.x + rear_rise * rear * rear
            - drop * front * front * front;
        let normal = Vec3::new(
            -2. * side_rise * uv.x / half.x,
            1.,
            -(2. * rear_rise * rear + 3. * drop * front * front) / half.y,
        )
        .normalize();
        if bottom {
            (Vec3::new(xz.x, height - self.gauge_m, xz.y), -normal)
        } else {
            (Vec3::new(xz.x, height, xz.y), normal)
        }
    }

    pub(super) fn build(&self, a: &mut Assembly, p: &ChairProgram, width: f32, depth: f32) {
        let half = Vec2::new(width, depth) * 0.5;
        self.shell(a, p.seat_surface, half);
        // Cantilever and trestle frames already have full-length upper rails.
        // Join the open tops of the splayed and sled legs on the other frames.
        if matches!(p.base_style, 4 | 5) {
            let length = depth * if p.base_style == 4 { 0.80 } else { 0.97 };
            for side in [-1., 1.] {
                a.box_part(
                    p.frame_surface,
                    "chair",
                    Vec3::new(side * width * 0.43, SIDE_CARRIER_Y, 0.),
                    Vec3::new(0.029, SIDE_CARRIER_HEIGHT, length),
                    0.0025,
                );
            }
        }
        for end in [-1., 1.] {
            let z = end * depth * 0.25;
            a.box_part(
                p.frame_surface,
                "chair",
                Vec3::new(0., CROSSBAR_Y, z),
                Vec3::new(width * 0.88, CROSSBAR_HEIGHT, 0.030),
                0.,
            );
            if self.stock != SeatStock::MouldedPolymer {
                for side in [-1., 1.] {
                    let x = side * width * 0.29;
                    let top = self.skin(Vec2::new(x, z), half, true).0.y + 0.0012;
                    a.box_part(
                        p.frame_surface,
                        "chair",
                        Vec3::new(x, (top + 0.431) * 0.5, z),
                        Vec3::new(0.035, top - 0.431, 0.029),
                        0.,
                    );
                }
            }
        }
        if self.stock == SeatStock::MouldedPolymer {
            for side in [-1., 1.] {
                self.rib(a.part(p.seat_surface, "chair"), half, side);
            }
        }
    }

    /// Thin integral ribs carry the polymer shell onto the two transverse frame rails.
    fn rib(&self, g: &mut Geometry, half: Vec2, side: f32) {
        let mut previous = None;
        for (z_fraction, bearing) in [
            (-0.70, false),
            (-0.54, true),
            (-0.46, true),
            (0., false),
            (0.46, true),
            (0.54, true),
            (0.70, false),
        ] {
            let z = z_fraction * half.y;
            let section = [0, 1, 2, 3].map(|i| {
                let x = side * half.x * 0.58 + if i == 0 || i == 3 { -0.0025 } else { 0.0025 };
                let shell = self.skin(Vec2::new(x, z), half, true).0.y;
                let y = if i >= 2 {
                    shell + 0.0012
                } else if bearing {
                    0.431
                } else {
                    shell - 0.008
                };
                Vec3::new(x, y, z)
            });
            if let Some(prev) = previous {
                let prev: [Vec3; 4] = prev;
                for (i, outward) in [-Vec3::Y, Vec3::X, Vec3::Y, -Vec3::X]
                    .into_iter()
                    .enumerate()
                {
                    let j = (i + 1) % 4;
                    g.metric_quad([prev[i], prev[j], section[j], section[i]], outward);
                }
            } else {
                g.metric_quad(section, -Vec3::Z);
            }
            previous = Some(section);
        }
        g.metric_quad(previous.unwrap(), Vec3::Z);
    }
}
