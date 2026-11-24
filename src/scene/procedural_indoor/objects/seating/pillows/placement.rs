use super::{
    cut::{Cut, Form},
    *,
};
use crate::scene::procedural_indoor::geometry::CushionSurface;

#[derive(serde::Serialize)]
pub(crate) struct Installed {
    pub seat_module: usize,
    pub propped: bool,
    pub center_m: Vec3,
    pub rotation: Quat,
    pub back_gap_m: Option<f32>,
    pub(super) cut: Cut,
    pub(super) stock: Stock,
}

struct Pad {
    // Center in the common backrest frame. Depth is positive towards the sitter.
    center: Vec3,
    half: Vec3,
    surface: CushionSurface,
}

struct Seat {
    center: Vec3,
    half: Vec3,
    surface: CushionSurface,
}

struct Supports {
    seats: Vec<Seat>,
    back: Vec<Pad>,
    into_back: Quat,
    frame_front: f32,
}
impl Supports {
    fn new(o: &IndoorObject, p: &SofaProgram, widths: &[f32]) -> Self {
        let mut seats = Vec::with_capacity(p.modules);
        upholstery::seats(o, p, widths, |center, size| {
            let (center, size, profile) = upholstery::main_pad(p, center, size);
            seats.push(Seat {
                center,
                half: size * 0.5,
                surface: CushionSurface::new(size, profile),
            });
        });
        let into_back = Quat::from_rotation_x(FRAC_PI_2 - p.back_tilt);
        let mut back = Vec::with_capacity(p.back_modules * 2);
        let w = (o.size.x - p.arm_width * 2.) / p.back_modules as f32;
        for i in 0..p.back_modules {
            let x = (i as f32 - (p.back_modules - 1) as f32 * 0.5) * w;
            let top = upholstery::back_top(o, p, x, p.seat_height - p.cushion);
            upholstery::back_pads(o, p, x, w, top, |size, profile, tf| {
                back.push(Pad {
                    center: into_back * tf.translation,
                    half: size * 0.5,
                    surface: CushionSurface::new(size, profile),
                });
            });
        }
        Self {
            seats,
            back,
            into_back,
            frame_front: o.size.z * 0.5 - 0.115,
        }
    }

    fn gap(&self, points: &[Vec3], center: Vec3) -> Option<f32> {
        let center = self.into_back * center;
        let mut gap = f32::INFINITY;
        for &v in points {
            let pt = v + center;
            for pad in &self.back {
                let q = pt - pad.center;
                if q.y - pad.half.y >= gap || q.x.abs() > pad.half.x || q.z.abs() > pad.half.z {
                    continue;
                }
                if let Some(height) = pad.surface.height(q.xz()) {
                    gap = gap.min(q.y - height);
                }
            }
        }
        gap.is_finite().then_some(gap)
    }
}

impl Seat {
    fn height(&self, points: &[Vec3], center: Vec2, min_y: f32) -> Option<f32> {
        let center = center - self.center.xz();
        let mut rise = f32::NEG_INFINITY;
        let mut bears = false;
        for &v in points {
            if self.half.y - v.y <= rise && (bears || v.y >= min_y + 0.022) {
                continue;
            }
            if let Some(h) = self.surface.height(center + v.xz()) {
                rise = rise.max(h - v.y);
                bears |= v.y < min_y + 0.022;
            }
        }
        bears.then_some(self.center.y + rise - 0.0008)
    }
}

struct Pose {
    points: Vec<Vec3>,
    lo: Vec3,
    hi: Vec3,
}
impl Pose {
    fn new(form: &Form, turn: Quat) -> Self {
        let points: Vec<_> = form.points.iter().map(|&v| turn * v).collect();
        let (lo, hi) = points.iter().fold(
            (Vec3::splat(f32::INFINITY), Vec3::splat(f32::NEG_INFINITY)),
            |(lo, hi), &v| (lo.min(v), hi.max(v)),
        );
        Self { points, lo, hi }
    }

    fn settle(
        &self,
        s: &Supports,
        seat: &Seat,
        mut center: Vec3,
        propped: bool,
    ) -> Option<(Vec3, Option<f32>)> {
        let back_points: Vec<_> = self.points.iter().map(|v| s.into_back * *v).collect();
        let normal_z = (s.into_back * Vec3::NEG_Z).y;
        let mut gap = None;
        for step in 0..4 {
            center.y = seat.height(&self.points, center.xz(), self.lo.y)?;
            gap = s.gap(&back_points, center);
            if !propped {
                break;
            }
            let d = gap?;
            if d.abs() < 0.0009 || step == 3 {
                break;
            }
            center.z = (center.z + (d - 0.0003) / normal_z).min(s.frame_front - self.hi.z - 0.0005);
        }
        if center.z + self.hi.z > s.frame_front + 0.0001
            || gap.is_some_and(|d| d < -0.0025)
            || propped && gap? > 0.0035
        {
            return None;
        }
        Some((center, gap))
    }
}

pub(super) fn install(
    o: &IndoorObject,
    p: &SofaProgram,
    widths: &[f32],
    mut emit: impl FnMut(Installed, &Form),
) {
    let mut rng = stream(o.seed, u64::from_le_bytes(*b"ThrowFit"));
    let target = rng.random_range(1..=4.min(p.modules + 1));
    let supports = Supports::new(o, p, widths);
    let mut accepted = Vec::<Rect>::with_capacity(target);
    for index in 0..target {
        let lean = rng.random_bool(0.58);
        let near_arm = rng.random_bool(0.44);
        let left = rng.random_bool(0.5);
        for attempt in 0..8 {
            let module = if near_arm && attempt < 3 && supports.seats.len() > 1 {
                if left {
                    0
                } else {
                    supports.seats.len() - 1
                }
            } else {
                rng.random_range(0..supports.seats.len())
            };
            let seat = &supports.seats[module];
            let max = (p.pillow_width_m * 1.7).min(seat.half.x * 1.60);
            let min = (p.pillow_width_m * 0.73).max(0.19);
            if max <= min {
                continue;
            }
            let w = rng.random_range(min..max);
            let size = Vec3::new(
                w,
                w * rng.random_range(0.18..0.45),
                w * rng.random_range(0.47..1.14),
            );
            let cut = Cut::sample(size, &mut rng);
            let propped = lean && attempt < 5;
            let pitch = if propped {
                rng.random_range(0.71..1.41)
            } else {
                rng.random_range(0.015..0.31)
            };
            let yaw = if propped {
                rng.random_range(-0.48..0.48)
            } else {
                rng.random_range(-PI..PI)
            };
            let turn = Quat::from_euler(
                EulerRot::YXZ,
                yaw,
                -pitch,
                p.pillow_roll + rng.random_range(-0.20..0.20),
            );
            let form = Form::new(cut);
            let pose = Pose::new(&form, turn);
            let pad = 0.012;
            let lo = seat.center.xz() - seat.half.xz() + Vec2::splat(pad) - pose.lo.xz();
            let hi = Vec2::new(
                seat.center.x + seat.half.x - pad,
                supports.frame_front - 0.001,
            ) - pose.hi.xz();
            if lo.cmpge(hi).any() {
                continue;
            }
            let mut side = rng.random::<f32>();
            if near_arm && attempt < 3 {
                side = if left { side * side } else { 1. - side * side };
            }
            let x = lo.x + (hi.x - lo.x) * side;
            let z = if propped {
                hi.y
            } else {
                rng.random_range(lo.y..hi.y)
            };
            let Some((at, gap)) = pose.settle(&supports, seat, Vec3::new(x, 0., z), propped) else {
                continue;
            };
            if at.y + pose.hi.y > o.size.y - 0.003 || at.z < lo.y {
                continue;
            }
            let footprint = Rect::from_corners(at.xz() + pose.lo.xz(), at.xz() + pose.hi.xz());
            if accepted
                .iter()
                .any(|r| !r.inflate(0.008).intersect(footprint).is_empty())
            {
                continue;
            }
            accepted.push(footprint);
            emit(
                Installed {
                    seat_module: module,
                    propped,
                    center_m: at,
                    rotation: turn,
                    back_gap_m: gap,
                    cut,
                    stock: Stock::sample(o, p, index),
                },
                &form,
            );
            break;
        }
    }
}
