//! Metric millwork. Local +Z faces arrivals, -Z is the seated operator side.
use super::*;

#[derive(Clone, Copy, Debug, Serialize)]
pub enum Fascia {
    Veneer,
    FramedPanels,
    Battens,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Counter {
    pub worktop_m: f32,
    pub stock_m: f32,
    pub ledge_depth_m: f32,
    /// Signed width of the lowered signing wing. Negative is left, zero is full-height.
    pub signing_wing_m: f32,
    pub drawer_left: bool,
    pub drawer_width_m: f32,
    pub fascia: Fascia,
    pub panel_pitch_m: f32,
    pub body: Surface,
    pub cap: Surface,
}

impl Counter {
    pub fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"FrontBay"));
        let mut f = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"FrontMat"),
        );
        let wing = if r.random_bool(0.68) {
            let width = r.random_range(0.57..0.98_f32).min(o.size.x * 0.42);
            if r.random_bool(0.5) {
                width
            } else {
                -width
            }
        } else {
            0.
        };
        Self {
            worktop_m: r.random_range(0.718..0.79),
            stock_m: r.random_range(0.025..0.043),
            ledge_depth_m: r.random_range(0.245..0.35),
            signing_wing_m: wing,
            drawer_left: if wing == 0. {
                r.random_bool(0.5)
            } else {
                wing > 0.
            },
            drawer_width_m: r.random_range(0.30..0.45),
            fascia: match f.random_range(0..3) {
                0 => Fascia::Veneer,
                1 => Fascia::FramedPanels,
                _ => Fascia::Battens,
            },
            panel_pitch_m: f.random_range(0.068..0.119),
            body: [Surface::Wood, Surface::Paint, Surface::Accent][f.random_range(0..3)],
            cap: [Surface::Wood, Surface::Plastic, Surface::Ceramic][f.random_range(0..3)],
        }
    }

    pub fn ledge(&self, o: &IndoorObject) -> Rect {
        let h = o.size.xz() * 0.5;
        Rect::from_corners(
            Vec2::new(-h.x - self.signing_wing_m.min(0.), h.y - self.ledge_depth_m),
            Vec2::new(h.x - self.signing_wing_m.max(0.), h.y),
        )
    }
    pub fn worktop(&self, o: &IndoorObject) -> Rect {
        let h = o.size.xz() * 0.5;
        Rect::from_corners(-h, Vec2::new(h.x, h.y - self.ledge_depth_m))
    }
    pub fn signing_wing(&self, o: &IndoorObject) -> Option<Rect> {
        let h = o.size.xz() * 0.5;
        let raised = self.ledge(o);
        let (left, right) = if self.signing_wing_m < 0. {
            (-h.x, raised.min.x)
        } else if self.signing_wing_m > 0. {
            (raised.max.x, h.x)
        } else {
            return None;
        };
        Some(Rect::from_corners(
            Vec2::new(left, -h.y),
            Vec2::new(right, h.y),
        ))
    }

    pub fn supports(&self, o: &IndoorObject, height: f32, lo: Vec2, hi: Vec2, margin: f32) -> bool {
        let inside = |r: Rect| {
            (lo - Vec2::splat(margin)).cmpge(r.min).all()
                && (hi + Vec2::splat(margin)).cmple(r.max).all()
        };
        if (height - o.size.y).abs() <= 0.002 {
            inside(self.ledge(o))
        } else {
            (height - self.worktop_m).abs() <= 0.002
                && (inside(self.worktop(o)) || self.signing_wing(o).is_some_and(inside))
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = Counter::sample(o);
    let raised = p.ledge(o);
    let rear = p.worktop(o);
    let t = p.stock_m;
    let half = o.size.xz() * 0.5;
    let l = o.kind.class_name();
    let mut cuts = TimberCuts::new(o.finish_seed.unwrap_or(o.seed), 0);
    let mut board = |s: Surface, pos: Vec3, size: Vec3, grain: GrainAxis| {
        a.board(&mut cuts, (s, l), pos, size, 0.0025, grain);
    };
    // Two abutting slabs at working height and an independently supported transaction shelf.
    board(
        p.cap,
        Vec3::new(0., p.worktop_m - t * 0.5, rear.center().y),
        Vec3::new(o.size.x, t, rear.height()),
        GrainAxis::X,
    );
    if let Some(wing) = p.signing_wing(o) {
        board(
            p.cap,
            Vec3::new(wing.center().x, p.worktop_m - t * 0.5, raised.center().y),
            Vec3::new(wing.width(), t, p.ledge_depth_m),
            GrainAxis::X,
        );
    }
    board(
        p.cap,
        Vec3::new(raised.center().x, o.size.y - t * 0.5, raised.center().y),
        Vec3::new(raised.width(), t, raised.height()),
        GrainAxis::X,
    );
    let plinth = 0.077;
    let z = raised.min.y + t * 0.5;
    // The vertical spine starts at the rear edge of the ledge, directly against the worktop.
    let w = raised.width() - 0.038;
    let top = o.size.y - t;
    let h = top - plinth;
    board(
        p.body,
        Vec3::new(raised.center().x, (top + plinth) * 0.5, z),
        Vec3::new(w, h, t),
        GrainAxis::X,
    );
    board(
        Surface::Metal,
        Vec3::new(raised.center().x, plinth * 0.5, z + 0.019),
        Vec3::new(w - 0.036, plinth, t + 0.06),
        GrainAxis::X,
    );
    // Seams and relief stay recessed under the cap. Adjacent front faces are separated physically.
    let front = z + t * 0.5;
    // Return cheeks support the cap near its front edge. They stop above the working
    // surface, leaving the lower signing wing open to visitors.
    let return_depth = raised.max.y - front - 0.035;
    let return_h = top - p.worktop_m;
    for x in [raised.min.x + 0.044, raised.max.x - 0.044] {
        board(
            p.body,
            Vec3::new(x, p.worktop_m + return_h * 0.5, front + return_depth * 0.5),
            Vec3::new(t, return_h, return_depth),
            GrainAxis::Y,
        );
    }
    match p.fascia {
        Fascia::Veneer => {}
        Fascia::FramedPanels | Fascia::Battens => {
            let slats = matches!(p.fascia, Fascia::Battens);
            let n = (w / if slats {
                p.panel_pitch_m
            } else {
                0.57 + p.panel_pitch_m
            })
            .round()
            .max(2.) as usize;
            let pitch = (w - 0.124) / n as f32;
            let width = if slats { pitch * 0.45 } else { pitch - 0.012 };
            for i in 0..n {
                let x = raised.center().x - (w - 0.124) * 0.5 + pitch * (i as f32 + 0.5);
                board(
                    if slats { Surface::Wood } else { p.body },
                    Vec3::new(x, (top + plinth) * 0.5, front + 0.008),
                    Vec3::new(width, h - 0.026, 0.016),
                    if slats { GrainAxis::Y } else { GrainAxis::X },
                );
            }
        }
    }
    let sign = if p.drawer_left { -1. } else { 1. };
    let x = sign * (half.x - 0.054 - p.drawer_width_m * 0.5);
    let side_x = -sign * (half.x - 0.047);
    let ch = p.worktop_m - t;
    let d = rear.height() - 0.09;
    let cz = rear.center().y + 0.008;
    board(
        p.body,
        Vec3::new(x, ch * 0.5, cz),
        Vec3::new(p.drawer_width_m, ch, d),
        GrainAxis::Y,
    );
    board(
        p.body,
        Vec3::new(side_x, ch * 0.5, cz),
        Vec3::new(t, ch, d),
        GrainAxis::Y,
    );
    let face = cz - d * 0.5;
    let mut y = plinth;
    for fraction in [0.46, 0.31, 0.23] {
        let height = (ch - plinth) * fraction;
        board(
            p.body,
            Vec3::new(x, y + height * 0.5, face - 0.008),
            Vec3::new(p.drawer_width_m - 0.012, height - 0.007, 0.016),
            GrainAxis::X,
        );
        board(
            Surface::Metal,
            Vec3::new(x, y + height * 0.74, face - 0.025),
            Vec3::new(p.drawer_width_m * 0.37, 0.014, 0.018),
            GrainAxis::X,
        );
        y += height;
    }
}

#[cfg(test)]
mod tests;
