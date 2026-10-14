use super::*;
use std::f32::consts::{FRAC_PI_2, PI, TAU};

struct Builder<'a> {
    a: &'a mut Assembly,
    tf: Transform,
    lo: Vec2,
    hi: Vec2,
    p: &'a ShadeProgram,
    label: String,
    coverage: f32,
    tilt: f32,
    fixed_angle: Option<f32>,
}
impl Builder<'_> {
    fn width(&self) -> f32 {
        self.hi.x - self.lo.x
    }
    fn height(&self) -> f32 {
        self.hi.y - self.lo.y - 0.095
    }
    fn top(&self) -> f32 {
        self.hi.y - 0.058
    }
    fn center(&self) -> f32 {
        (self.lo.x + self.hi.x) * 0.5
    }
    fn block(&mut self, surface: Surface, c: Vec3, size: Vec3, bevel: f32) {
        self.a.part(surface, &self.label).cuboid(
            size,
            bevel,
            Transform::from_translation(self.tf.transform_point(c)).with_rotation(self.tf.rotation),
        );
    }
    fn tube(&mut self, surface: Surface, points: &[Vec3], radius: f32) {
        self.a.part(surface, &self.label).tube(
            &points
                .iter()
                .map(|p| self.tf.transform_point(*p))
                .collect::<Vec<_>>(),
            radius,
            6,
        );
    }
    fn header(&mut self, kind: Shade) {
        let y = self.hi.y - 0.026;
        if kind == Shade::Curtain {
            self.tube(
                Surface::Metal,
                &[Vec3::new(self.lo.x, y, 0.13), Vec3::new(self.hi.x, y, 0.13)],
                0.01,
            );
        } else if kind == Shade::Roller && self.p.identity.is_multiple_of(2) {
            self.tube(
                self.p.fabric,
                &[
                    Vec3::new(self.lo.x + 0.018, y, 0.12),
                    Vec3::new(self.hi.x - 0.018, y, 0.12),
                ],
                0.021,
            );
        } else {
            self.block(
                Surface::Plastic,
                Vec3::new(self.center(), y, 0.115),
                Vec3::new(self.width(), 0.042, 0.050),
                0.003,
            );
        }
        for x in [self.lo.x + 0.01, self.hi.x - 0.01] {
            self.block(
                Surface::Metal,
                Vec3::new(x, y, 0.075),
                Vec3::new(0.016, 0.042, 0.13),
                0.002,
            );
        }
    }
    fn control(&mut self, vertical: bool) {
        let x = if self.p.identity.is_multiple_of(2) {
            self.hi.x - 0.008
        } else {
            self.lo.x + 0.008
        };
        let top = self.top();
        let bottom = top - (self.height() * 0.70).min(1.35);
        if vertical {
            self.tube(
                Surface::Plastic,
                &[Vec3::new(x, top, 0.172), Vec3::new(x, bottom, 0.172)],
                0.003,
            );
        } else {
            let points = [
                Vec3::new(x, top, 0.164),
                Vec3::new(x, bottom + 0.01, 0.164),
                Vec3::new(x - 0.006, bottom, 0.164),
                Vec3::new(x - 0.012, bottom + 0.01, 0.164),
                Vec3::new(x - 0.012, top, 0.164),
            ];
            self.tube(Surface::Fabric, &points, 0.0012);
        }
    }
    fn venetian(&mut self) {
        let n = (self.height() / self.p.slat_pitch).floor().clamp(2., 80.) as usize;
        // Preserve every slat when raised; the stack compresses at the header.
        let pitch = self.height() / n as f32;
        let step = 0.004 + (pitch - 0.004) * self.coverage;
        let depth = (pitch * 1.07).min(0.092);
        let tilt = self.fixed_angle.unwrap_or(self.tilt * self.coverage.sqrt());
        let width = self.width() - 0.038;
        let center = self.center();
        let top = self.top() - depth * 0.5;
        for i in 0..n {
            let tf = Transform::from_translation(self.tf.transform_point(Vec3::new(
                center,
                top - i as f32 * step,
                0.108,
            )))
            .with_rotation(self.tf.rotation * Quat::from_rotation_x(-FRAC_PI_2 + tilt));
            let g = self.a.part(self.p.slat_material, &self.label);
            let first_uv = g.uvs.len();
            cloth::sheet(g, tf, [1, 6], Vec2::new(width, depth), |u, v| {
                Vec3::new(
                    (u - 0.5) * width,
                    (v - 0.5) * depth,
                    0.003 * (1. - (v * 2. - 1.).powi(2)),
                )
            });
            for uv in &mut g.uvs[first_uv..] {
                uv[1] += i as f32 * pitch;
            }
        }
        let bottom = top - (n - 1) as f32 * step - depth * 0.5 - 0.006;
        self.block(
            self.p.slat_material,
            Vec3::new(center, bottom, 0.108),
            Vec3::new(width, 0.012, depth),
            0.001,
        );
        for x in [
            self.lo.x + self.width() * 0.18,
            self.hi.x - self.width() * 0.18,
        ] {
            for z in [0.108 - depth * 0.48, 0.108 + depth * 0.48] {
                self.tube(
                    Surface::Fabric,
                    &[Vec3::new(x, self.top(), z), Vec3::new(x, bottom, z)],
                    0.0013,
                );
            }
        }
    }
    fn vertical(&mut self) {
        let width = self.width() - 0.04;
        let count = (width / (self.p.slat_pitch * 1.9).min(0.13))
            .ceil()
            .clamp(2., 80.) as usize;
        let pitch = width / count as f32;
        let slat = (pitch * 1.025).min(0.15);
        let angle = self.tilt;
        // Stacked vanes rotate nearly edge-on so adjacent lamellae do not overlap.
        let angle = if angle < 0. { -1. } else { 1. }
            * (angle.abs().max(0.12) * self.coverage + 1.50 * (1. - self.coverage));
        let angle = self.fixed_angle.unwrap_or(angle);
        let stack_pitch = slat * angle.cos().abs() + 0.003;
        let step = stack_pitch.max(pitch * self.coverage).min(pitch);
        let span = (count - 1) as f32 * step;
        let half_x = slat * angle.cos().abs() * 0.5 + 0.004;
        let left = self.lo.x + 0.02 + half_x;
        let right = self.hi.x - 0.02 - half_x;
        let left_count = ((count as f32 * self.p.split).round() as usize).clamp(1, count - 1);
        let height = self.height() - 0.01;
        let top = self.top() - 0.01;
        for i in 0..count {
            let x = match self.p.stack {
                Stack::Left => left + i as f32 * step,
                Stack::Right => right - span + i as f32 * step,
                Stack::Pair if i < left_count => left + i as f32 * step,
                Stack::Pair => right - (count - 1 - i) as f32 * step,
            };
            let tf = Transform::from_translation(self.tf.transform_point(Vec3::new(
                x,
                top - height * 0.5,
                0.108,
            )))
            .with_rotation(self.tf.rotation * Quat::from_rotation_y(angle));
            cloth::sheet(
                self.a.part(self.p.fabric, &self.label),
                tf,
                [4, 1],
                Vec2::new(slat, height),
                |u, v| {
                    Vec3::new(
                        (u - 0.5) * slat,
                        (v - 0.5) * height,
                        0.002 * (1. - (u * 2. - 1.).powi(2)),
                    )
                },
            );
            self.tube(
                Surface::Metal,
                &[
                    Vec3::new(x, self.top() + 0.01, 0.108),
                    Vec3::new(x, top, 0.108),
                ],
                0.002,
            );
        }
    }
    fn shade(&mut self, kind: Shade) {
        let available = self.height();
        let min_stack = match kind {
            Shade::Roman => 0.10_f32,
            Shade::Pleated => 0.045,
            _ => 0.012,
        }
        .min(available * 0.35);
        let height = min_stack + (available - min_stack) * self.coverage;
        let width = self.width() - 0.035;
        let top = self.top();
        let center = self.center();
        let pitch = self.p.fold_pitch * if kind == Shade::Roman { 2.1 } else { 0.40 };
        let folds = (available / pitch).ceil().clamp(2., 32.) as usize;
        let rows = if kind == Shade::Roller { 2 } else { folds * 6 };
        let depth = self.p.fold_depth
            * if kind == Shade::Roman {
                1. - self.coverage * 0.93
            } else {
                0.65
            };
        let sag = if kind == Shade::Roman {
            self.p.hem_sag.min(height * 0.08)
        } else {
            0.
        };
        cloth::sheet(
            self.a.part(self.p.fabric, &self.label),
            self.tf,
            // Roller/pleated profiles are exactly linear across their width;
            // additional columns would only repeat the same surface and UVs.
            [if kind == Shade::Roman { 8 } else { 1 }, rows],
            Vec2::new(width, available),
            |u, v| {
                let z = match kind {
                    Shade::Roller => 0.112,
                    Shade::Roman => 0.108 + depth * (1. - (v * folds as f32 * TAU).cos()) * 0.5,
                    _ => 0.10 + depth * (1. - ((v * folds as f32).fract() * 2. - 1.).abs()),
                };
                Vec3::new(
                    center + (u - 0.5) * width,
                    top - height + v * height + sag * (u * 2. - 1.).powi(2) * (1. - v),
                    z,
                )
            },
        );
        if kind != Shade::Roman {
            self.block(
                Surface::Plastic,
                Vec3::new(center, top - height - 0.007, 0.112),
                Vec3::new(width + 0.006, 0.014, 0.018),
                0.002,
            );
        }
    }
    fn curtain(&mut self) {
        let ranges = match self.p.stack {
            Stack::Pair => vec![
                (self.lo.x, self.lo.x + self.width() * self.p.split, true),
                (self.lo.x + self.width() * self.p.split, self.hi.x, false),
            ],
            Stack::Left => vec![(self.lo.x, self.hi.x, true)],
            Stack::Right => vec![(self.lo.x, self.hi.x, false)],
        };
        let top = self.top();
        let height = self.height();
        for (left, right, left_stack) in ranges {
            let span = right - left - 0.014;
            let width = span * (0.12 + 0.88 * self.coverage);
            let x0 = if left_stack {
                left + 0.007
            } else {
                right - 0.007 - width
            };
            let folds = (span / self.p.fold_pitch).ceil().clamp(2., 18.) as usize;
            let depth = self.p.fold_depth;
            let tieback = self.p.tieback * (1. - self.coverage);
            let hem_sag = self.p.hem_sag;
            let shape = |u: f32, v: f32| {
                let pinch = (-(v - 0.38).powi(2) / 0.025).exp() * tieback;
                let gathering = width * (1. - 0.72 * pinch);
                let x = if left_stack {
                    x0 + u * gathering
                } else {
                    x0 + width - (1. - u) * gathering
                };
                // Unequal folds and changing amplitude avoid a rigid accordion.
                let across = u + 0.10 * (u * TAU).sin() / TAU + 0.035 * (u * 3. * TAU).sin() / TAU;
                let phase = across * folds as f32 * TAU + 0.38 * (v * PI).sin() * (u * PI).sin();
                let wave = phase.cos() * 0.82 + (phase * 0.5 + 0.8).cos() * 0.18;
                let z = 0.110
                    + depth * wave * (0.65 + 0.35 * (1. - v))
                    + 0.006 * (v * PI).sin() * (u * 3. * PI).sin();
                Vec3::new(
                    x,
                    top - height + height * v + hem_sag * (1. - v) * (0.5 + 0.5 * phase.sin()),
                    z,
                )
            };
            cloth::sheet(
                self.a.part(self.p.fabric, &self.label),
                self.tf,
                [folds * 6, 6],
                Vec2::new(span * self.p.fullness, height),
                shape,
            );
            for i in 0..=folds {
                let p = shape(i as f32 / folds as f32, 1.);
                self.tube(
                    Surface::Metal,
                    &[p, Vec3::new(p.x, self.hi.y - 0.026, 0.13)],
                    0.002,
                );
            }
            if tieback > 0.01 {
                let a = shape(0., 0.38);
                let b = shape(1., 0.38);
                self.tube(
                    self.p.fabric,
                    &[
                        a + Vec3::Z * 0.035,
                        (a + b) * 0.5 + Vec3::Z * 0.045,
                        b + Vec3::Z * 0.035,
                    ],
                    0.003,
                );
            }
        }
    }
}

pub(super) fn build(
    a: &mut Assembly,
    tf: Transform,
    f: &Facade,
    o: &WindowOpening,
    p: &ShadeProgram,
) {
    build_with_angle(a, tf, f, o, p, None);
}
pub(super) fn build_with_angle(
    a: &mut Assembly,
    tf: Transform,
    f: &Facade,
    o: &WindowOpening,
    p: &ShadeProgram,
    fixed_angle: Option<f32>,
) {
    let label = format!(
        "{}#finish{}",
        if f.shade == Shade::Curtain {
            "curtain"
        } else {
            "blinds"
        },
        p.finish_slot
    );
    let mut b = Builder {
        a,
        tf,
        lo: o.min + Vec2::splat(0.014),
        hi: o.max - Vec2::splat(0.014),
        p,
        label,
        coverage: f.shade_coverage.clamp(0., 1.),
        tilt: f.shade_tilt.clamp(-1.2, 1.2),
        fixed_angle,
    };
    b.header(f.shade);
    match f.shade {
        Shade::Venetian => b.venetian(),
        Shade::Vertical => b.vertical(),
        Shade::Curtain => b.curtain(),
        Shade::Roller | Shade::Roman | Shade::Pleated => b.shade(f.shade),
        Shade::None => return,
    }
    if f.shade != Shade::Curtain {
        b.control(f.shade == Shade::Vertical);
    }
}
