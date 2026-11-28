//! Wall-fixed practice goals. +Z faces the players, Y=0 is the bottom of the net.
use super::*;
use crate::scene::procedural_indoor::materials::sport;

mod board;
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 295;
pub(crate) const MIN_PROJECTION: f32 = 0.98;
pub(crate) const MAX_PROJECTION: f32 = 1.74;
const CLASS: &str = "sports_equipment";
const TIES: usize = 12;

pub(super) fn dimensions(r: &mut impl Rng) -> Vec3 {
    Vec3::new(
        r.random_range(1.34..1.84),
        r.random_range(1.22..1.47),
        r.random_range(MIN_PROJECTION..MAX_PROJECTION),
    )
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct BasketballGoal {
    pub arched: bool,
    pub timber: bool,
    pub upper_braces: bool,
    pub board_depth: f32,
    pub shoulder: f32,
    pub frame_span: f32,
    pub bore_radius: f32,
    pub steel_radius: f32,
    pub net_drop: f32,
    pub outlet_radius: f32,
    pub cord_radius: f32,
    pub net_twist: f32,
}

impl BasketballGoal {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut kit = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"Goal Kit"),
        );
        let mut net = stream(o.seed, u64::from_le_bytes(*b"GoalTwst"));
        Self {
            arched: kit.random_bool(0.42),
            timber: kit.random_bool(0.34),
            upper_braces: kit.random(),
            board_depth: kit.random_range(0.027..0.049),
            shoulder: kit.random_range(0.46..0.68),
            frame_span: o.size.x * kit.random_range(0.24..0.36),
            // The 32-sided mesh also clears the minimum bore between vertices.
            bore_radius: kit.random_range(0.2262..0.2294),
            steel_radius: kit.random_range(0.008..0.010),
            net_drop: kit.random_range(0.40..0.45),
            outlet_radius: kit.random_range(0.126..0.16),
            cord_radius: kit.random_range(0.0018..0.0031),
            net_twist: net.random_range(-0.19..0.19),
        }
    }
    /// Top of the steel in local coordinates. Installation uses this datum.
    pub fn rim_height(self) -> f32 {
        self.net_drop + self.cord_radius + 2. * self.steel_radius
    }
    fn rim(self, size: Vec3) -> (Vec3, f32) {
        let radius = self.bore_radius + self.steel_radius;
        (
            Vec3::new(
                0.,
                self.rim_height() - self.steel_radius,
                size.z * 0.5 - radius - self.steel_radius,
            ),
            radius,
        )
    }
    pub(crate) fn face_z(self, size: Vec3) -> f32 {
        self.rim(size).0.z - self.bore_radius - 0.151
    }
    /// Solid areas for placement scoring. Do not turn the open basket or the
    /// air behind the backboard into an opaque box in the visibility estimate.
    pub(crate) fn view_boxes(self, size: Vec3, mut emit: impl FnMut(Vec3, Vec3)) {
        let levels = self.board_levels(size);
        for panel in levels.windows(2) {
            let [(y0, w0), (y1, w1)] = [panel[0], panel[1]];
            emit(
                Vec3::new(
                    0.,
                    (y0 + y1) * 0.5,
                    self.face_z(size) - 0.5 * self.board_depth,
                ),
                Vec3::new(w0.min(w1), (y1 - y0) * 0.5, 0.5 * self.board_depth),
            );
        }
    }
}

fn radial(center: Vec3, r: f32, angle: f32) -> Vec3 {
    let (s, c) = angle.sin_cos();
    center + Vec3::new(r * c, 0., r * s)
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let g = BasketballGoal::for_object(o);
    g.build_board(a, o.size);
    let (rim, radius) = g.rim(o.size);
    let face = g.face_z(o.size);
    let rear = face - g.board_depth;
    let wall = -o.size.z * 0.5;
    let low = g.rim_height() + 0.024;
    let high = g.rim_height() + (o.size.y - g.rim_height()) * 0.68;
    // The rim bolts into a crossmember spanning both rear mounting channels.
    a.box_part(
        Surface::Metal,
        CLASS,
        Vec3::new(0., low - 0.029, rear - 0.018),
        Vec3::new(2. * g.frame_span - 0.057, 0.064, 0.036),
        0.,
    );
    for x in [-g.frame_span, g.frame_span] {
        a.box_part(
            Surface::Metal,
            CLASS,
            Vec3::new(x, (low + high) * 0.5, wall + 0.014),
            Vec3::new(0.075, high - low + 0.088, 0.028),
            0.,
        );
        a.box_part(
            Surface::Metal,
            CLASS,
            Vec3::new(x, (low + high) * 0.5, rear - 0.018),
            Vec3::new(0.057, high - low + 0.072, 0.036),
            0.,
        );
        let (arm, brace) = if g.upper_braces {
            (low, high)
        } else {
            (high, low)
        };
        let hub = Vec3::new(x, arm, rear - 0.022);
        let metal = a.part(Surface::Metal, CLASS);
        for y in [arm, brace] {
            metal.tube(&[Vec3::new(x, y, wall + 0.027), hub], 0.024, 8);
        }
    }
    let paint = a.part(Surface::Metal, sport::RIM);
    paint.cuboid(
        Vec3::new(0.16, 0.13, 0.014),
        0.,
        Transform::from_xyz(0., rim.y - 0.046, face + 0.007),
    );
    for side in [-1., 1.] {
        paint.tube(
            &[
                Vec3::new(side * 0.055, rim.y - 0.080, face + 0.012),
                radial(rim, radius, -FRAC_PI_2 + side * 0.27),
            ],
            0.0085,
            6,
        );
    }
    let ring =
        std::array::from_fn::<_, 33, _>(|i| radial(rim, radius, TAU * (i % 32) as f32 / 32.));
    paint.tube(&ring, g.steel_radius, 8);

    let net = a.part(Surface::Fabric, sport::TAPE);
    const ROWS: usize = 5;
    for tie in 0..TIES {
        for turn in [-1., 1.] {
            let path = std::array::from_fn::<_, { ROWS + 1 }, _>(|j| {
                let t = j as f32 / ROWS as f32;
                let r = g.outlet_radius + (radius - g.outlet_radius) * (1. - t).powi(2);
                let angle =
                    (tie as f32 + turn * 0.5 * j as f32) * TAU / TIES as f32 + g.net_twist * t;
                radial(rim - Vec3::Y * (g.steel_radius + g.net_drop * t), r, angle)
            });
            net.tube(&path, g.cord_radius, 4);
        }
    }
}
