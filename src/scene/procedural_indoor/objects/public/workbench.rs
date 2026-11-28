//! Standing shop stations. The operator and movable face jaw are on local +Z.
use super::*;
mod construction;
#[cfg(test)]
mod tests;
mod top;

pub(crate) const VERSION: u32 = 299;
const BORE_CHAMFER_M: f32 = 0.0012;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
pub enum WorkingSurface {
    DogHoles,
    AssemblyTop,
    CuttingMat,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct FaceVise {
    pub center_x_m: f32,
    pub jaw_width_m: f32,
    pub jaw_thickness_m: f32,
    pub opening_m: f32,
    pub handle_angle_rad: f32,
    pub handle_slide_m: f32,
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Bench {
    pub surface: WorkingSurface,
    pub frame_finish: Surface,
    pub stock_m: f32,
    pub post_m: f32,
    pub front_z_m: f32,
    pub leg_setback_m: f32,
    pub shelf_y_m: Option<f32>,
    pub slatted_shelf: bool,
    pub vise: Option<FaceVise>,
    pub bore_pitch_m: f32,
    pub bore_radius_m: f32,
    pub two_bore_rows: bool,
    pub laminations: usize,
}

impl Bench {
    pub fn sample(o: &IndoorObject) -> Self {
        let mut kit = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"BenchKit"),
        );
        let mut r = stream(o.seed, u64::from_le_bytes(*b"BenchFit"));
        let surface = match kit.random_range(0..100) {
            0..=54 => WorkingSurface::DogHoles,
            55..=75 => WorkingSurface::AssemblyTop,
            _ => WorkingSurface::CuttingMat,
        };
        let vise = (surface == WorkingSurface::DogHoles).then(|| {
            let jaw_width_m = r.random_range(0.235..0.385);
            let side = if r.random_bool(0.72) { -1. } else { 1. };
            FaceVise {
                center_x_m: side
                    * (o.size.x * r.random_range(0.27..0.38))
                        .min(o.size.x * 0.5 - jaw_width_m * 0.5 - 0.038),
                jaw_width_m,
                jaw_thickness_m: r.random_range(0.042..0.071),
                opening_m: r.random_range(0.018..0.115),
                handle_angle_rad: r.random_range(-std::f32::consts::PI..std::f32::consts::PI),
                handle_slide_m: r.random_range(-0.024..0.024),
            }
        });
        let wood = kit.random_bool(if vise.is_some() { 0.88 } else { 0.34 });
        Self {
            surface,
            vise,
            stock_m: kit.random_range(0.033..0.078),
            frame_finish: if wood {
                Surface::Wood
            } else {
                [Surface::Metal, Surface::Paint, Surface::Accent][kit.random_range(0..3)]
            },
            post_m: if wood {
                kit.random_range(0.065..0.104)
            } else {
                kit.random_range(0.039..0.065)
            },
            leg_setback_m: r.random_range(0.13..0.25),
            front_z_m: o.size.z * 0.5
                - vise.map_or(0., |v| v.opening_m + v.jaw_thickness_m + 0.055),
            shelf_y_m: kit.random_bool(0.70).then(|| r.random_range(0.155..0.355)),
            slatted_shelf: kit.random_bool(if wood { 0.72 } else { 0.18 }),
            bore_pitch_m: kit.random_range(0.118..0.213),
            bore_radius_m: kit.random_range(0.0095..0.0128),
            two_bore_rows: kit.random_bool(0.53),
            laminations: kit.random_range(6..=10),
        }
    }

    pub fn deck(&self, o: &IndoorObject) -> Rect {
        Rect::from_corners(
            -o.size.xz() * 0.5,
            Vec2::new(o.size.x * 0.5, self.front_z_m),
        )
    }

    pub fn bores_per_row(&self, o: &IndoorObject) -> usize {
        if self.vise.is_none() {
            return 0;
        }
        ((o.size.x - 0.16) / self.bore_pitch_m).floor() as usize
    }

    fn bored_strip(&self, strip: usize) -> bool {
        self.vise.is_some() && (strip == self.laminations - 2 || self.two_bore_rows && strip == 1)
    }

    pub(crate) fn supports_top(&self, o: &IndoorObject, lo: Vec2, hi: Vec2, margin: f32) -> bool {
        let deck = self.deck(o);
        if lo.cmplt(deck.min + margin).any() || hi.cmpgt(deck.max - margin).any() {
            return false;
        }
        // A thin part may bridge a bore, but may not be balanced entirely over its mouth.
        // A footprint wider than the diameter in either dimension cannot fall through.
        let mouth = self.bore_radius_m + BORE_CHAMFER_M;
        let n = self.bores_per_row(o) as f32;
        if n == 0. || (hi - lo).max_element() >= 2. * mouth {
            return true;
        }
        let center = (hi + lo) * 0.5;
        let j = (center.x / self.bore_pitch_m + (n - 1.) * 0.5)
            .round()
            .clamp(0., n - 1.);
        let x = (j - (n - 1.) * 0.5) * self.bore_pitch_m;
        for i in [1, self.laminations - 2] {
            if !self.bored_strip(i) {
                continue;
            }
            let z = deck.min.y + deck.height() * (i as f32 + 0.5) / self.laminations as f32;
            let delta = (center - Vec2::new(x, z)).abs() + (hi - lo) * 0.5;
            if delta.length_squared() <= mouth.powi(2) {
                return false;
            }
        }
        true
    }

    /// Signed distance above the floor of stock held edge-up between the two jaws.
    pub fn clamped_bottom(o: &IndoorObject) -> f32 {
        o.size.y - 0.054
    }

    pub(crate) fn grips(&self, bench: &IndoorObject, item: &IndoorObject) -> bool {
        let Some(v) = self.vise else {
            return false;
        };
        if item.kind != ObjectKind::TimberOffcut {
            return false;
        }
        let local = bench
            .transform()
            .compute_affine()
            .inverse()
            .transform_point3(item.position);
        let turn = item.yaw - bench.yaw;
        // The mitred end may miss one cheek. Require the full square-ended span
        // to overlap the jaw near the centre and seat above the steel guides.
        let setback = mitre_setback(item).abs();
        (local.y - Self::clamped_bottom(bench)).abs() < 0.001
            && item.size.y > 0.062
            && turn.sin().abs() < 0.0005
            && (item.size.z - v.opening_m).abs() < 0.0005
            && (local.z - self.front_z_m - v.opening_m * 0.5).abs() < 0.0005
            && (local.x - v.center_x_m).abs() + 0.034 + setback <= item.size.x * 0.5
    }
}

pub fn mitre_setback(o: &IndoorObject) -> f32 {
    let mut r = stream(o.seed, u64::from_le_bytes(*b"SawnEnds"));
    if r.random_bool(0.55) {
        0.
    } else {
        r.random_range(-0.82..0.82) * o.size.z.min(o.size.x * 0.38)
    }
}

pub(super) fn offcut(a: &mut Assembly, o: &IndoorObject) {
    let grain = o.finish_seed.unwrap_or(o.seed);
    let mut cuts = TimberCuts::new(grain, 1);
    let chart = cuts.origin();
    let g = a.part(Surface::Wood, o.kind.class_name());
    let (x, z) = (o.size.x * 0.5, o.size.z * 0.5);
    let m = mitre_setback(o);
    let q = [
        Vec3::new(-x, 0., -z),
        Vec3::new(x - m.max(0.), 0., -z),
        Vec3::new(x + m.min(0.), 0., z),
        Vec3::new(-x, 0., z),
    ];
    let up = q.map(|p| p + Vec3::Y * o.size.y);
    g.stock_quad(q, -Vec3::Y, Vec3::X, Some(chart));
    g.stock_quad(up, Vec3::Y, Vec3::X, Some(chart));
    for i in 0..4 {
        let j = (i + 1) % 4;
        let wall = [q[i], q[j], up[j], up[i]];
        let normal = Vec3::Y.cross(q[j] - q[i]).normalize();
        if i % 2 == 1 {
            g.metric_quad(wall, normal);
        } else {
            g.stock_quad(wall, normal, Vec3::X, Some(chart));
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = Bench::sample(o);
    let mut cuts = TimberCuts::new(o.finish_seed.unwrap_or(o.seed), 0);
    top::build(a, o, &p, &mut cuts);
    construction::base(a, o, &p, &mut cuts);
    if let Some(v) = p.vise {
        construction::vise(a, o, &p, v, &mut cuts);
    }
}
