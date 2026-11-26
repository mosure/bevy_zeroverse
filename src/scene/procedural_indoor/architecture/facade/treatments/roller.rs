use super::*;
use crate::scene::procedural_indoor::motion::RollerFeed;

pub(in super::super) fn feed_plane(tf: Transform, opening: &WindowOpening) -> RollerFeed {
    RollerFeed::new(tf, opening.max.y - 0.014 - super::RAIL_DROP)
}

const AXLE_Z: f32 = 0.101;
const CORD_RADIUS: f32 = 0.0013;

impl Builder<'_> {
    pub(super) fn fitted_roller_head(&mut self, stock: RollerStock) {
        let y = self.top();
        let width = self.width() - 0.030;
        let center = self.center();
        let along_x = Quat::from_rotation_z(-FRAC_PI_2);
        let center_at = |x| Transform::from_xyz(x, y, AXLE_Z).with_rotation(along_x);
        let radius = stock.radius_m;
        match stock.head {
            RollerHead::SquareCassette => self.block(
                Surface::Metal,
                Vec3::new(center, y, AXLE_Z),
                Vec3::new(width, radius * 1.8, radius * 2.),
                0.003,
            ),
            head => {
                let (surface, label) = if head == RollerHead::Wrapped {
                    (self.p.fabric, &self.cloth_label)
                } else {
                    (Surface::Metal, &self.label)
                };
                let half = width * 0.5;
                self.a.part(surface, label).lathe(
                    &[(0., -half), (radius, -half), (radius, half), (0., half)],
                    16,
                    self.tf.mul_transform(center_at(center)),
                );
            }
        }
        for (right, edge, sign) in [(false, self.lo.x, 1.), (true, self.hi.x, -1.)] {
            // The return reaches the jamb, and the bearing spans the cheek and
            // housing end. Bury the ends of the shaft to avoid coincident caps.
            self.block(
                Surface::Metal,
                Vec3::new(edge - sign * 0.005, y, 0.004),
                Vec3::new(0.018, 0.034, 0.008),
                0.,
            );
            self.block(
                Surface::Metal,
                Vec3::new(edge + sign * 0.0015, y, AXLE_Z * 0.54 + 0.002),
                Vec3::new(0.003, 0.027, AXLE_Z * 1.08),
                0.,
            );
            let x = edge + sign * 0.0085;
            let drive = matches!(
                (right, stock.drive),
                (false, RollerDrive::LeftLoop) | (true, RollerDrive::RightLoop)
            );
            let wheel = stock.wheel_radius_m;
            let outer = if drive {
                wheel + CORD_RADIUS * 1.5
            } else {
                0.007
            };
            let groove = if drive { wheel - CORD_RADIUS } else { outer };
            let tf = self.tf.mul_transform(center_at(x));
            self.a.part(Surface::Plastic, &self.label).lathe(
                &[
                    (0., -0.0075),
                    (outer, -0.0075),
                    (outer, -0.0023),
                    (groove, -0.0015),
                    (groove, 0.0015),
                    (outer, 0.0023),
                    (outer, 0.0075),
                    (0., 0.0075),
                ],
                12,
                tf,
            );
        }
    }

    pub(super) fn roller_loop(&mut self, stock: RollerStock, moving: bool) {
        let Some(drop) = stock.loop_drop(self.height()) else {
            return;
        };
        let x = if stock.drive == RollerDrive::LeftLoop {
            self.lo.x + 0.0085
        } else {
            self.hi.x - 0.0085
        };
        let top = self.top();
        let radius = stock.wheel_radius_m;
        // Two tangent strands, an arc seated in the clutch groove and a free
        // return bend. The control is stationary even while the web is drawing.
        const STEPS: usize = 8;
        let mut points = [Vec3::ZERO; 2 * (STEPS + 1) + 1];
        for (i, point) in points[..2 * (STEPS + 1)].iter_mut().enumerate() {
            let lower = i > STEPS;
            let a = PI * ((i % (STEPS + 1)) as f32 / STEPS as f32 + f32::from(lower));
            let (sin, cos) = a.sin_cos();
            *point = self.tf.transform_point(Vec3::new(
                x,
                top - if lower { drop } else { 0. } + radius * sin,
                AXLE_Z + radius * cos,
            ));
        }
        points[2 * (STEPS + 1)] = points[0];
        let stationary = moving.then(|| format!("{}#roller_drive", self.label));
        self.a
            .part(
                Surface::Plastic,
                stationary.as_deref().unwrap_or(&self.label),
            )
            .tube(&points, CORD_RADIUS, 5);
    }

    pub(super) fn roller_head(&mut self) {
        let top = self.top();
        if self.p.identity.is_multiple_of(2) {
            // A cloth-covered roll, supported at both ends. Its exposed lower
            // arc hides the transition between the visible web and stored yarn.
            self.tube(
                self.p.fabric,
                &[
                    Vec3::new(self.lo.x + 0.006, top, 0.101),
                    Vec3::new(self.hi.x - 0.006, top, 0.101),
                ],
                0.042,
            );
        } else {
            self.block(
                Surface::Metal,
                Vec3::new(self.center(), top - 0.003, 0.111),
                Vec3::new(self.width(), 0.068, 0.062),
                0.003,
            );
        }
    }

    pub(super) fn roller(&mut self, moving: bool, other_coverage: Option<f32>) {
        let available = self.height();
        let height = 0.044 + (available - 0.044) * self.coverage;
        let width = self.width() - 0.035;
        let center = self.center();
        let rail = self.top();
        let warp = 0.13 * self.p.fold_depth;
        let length = if moving { available } else { height };
        // Only an articulated roller needs vertices for cloth still on the roll.
        // A row crossing the feed plane fits entirely behind either head profile.
        let uniform_rows = if moving {
            (available / 0.018).ceil() as usize
        } else {
            1
        };
        // Away from the portion that crosses the feed the web is straight in
        // both position and material UV. Only retain dense rows inside the swept
        // interval. Both endpoints use exactly the same material coordinates.
        let (first_row, last_row) = if let Some(other) =
            other_coverage.filter(|_| moving && self.p.roller.is_some() && uniform_rows >= 3)
        {
            let row = |c: f32| (0.044 + (available - 0.044) * c) / available * uniform_rows as f32;
            (
                (row(self.coverage.min(other)).floor() as usize).clamp(1, uniform_rows - 1),
                (row(self.coverage.max(other)).ceil() as usize).clamp(1, uniform_rows - 1),
            )
        } else {
            (1, uniform_rows - 1)
        };
        let rows = last_row + 2 - first_row;
        let body_label = format!("{}#roller", self.cloth_label);
        let g = self.a.part(self.p.fabric, &body_label);
        let first = g.uvs.len();
        let columns = 4;
        let shape = |u: f32, v: f32| {
            let above_rail = v * length - height;
            Vec3::new(
                center + (u - 0.5) * width,
                rail + RollerFeed::pack(above_rail),
                0.112 + warp * (u * PI).sin() * (u * TAU + 0.6).sin(),
            )
        };
        let cells = [columns, rows];
        let uv_size = Vec2::new(width, length);
        if self.p.roller.is_some() {
            cloth::sheet_rows::<true>(
                g,
                self.tf,
                cells,
                uv_size,
                0.0012,
                |v| {
                    if rows == uniform_rows || v == 0. || v == 1. {
                        return v;
                    }
                    let index = (v * rows as f32).round() as usize;
                    (index + first_row - 1) as f32 / uniform_rows as f32
                },
                shape,
            );
        } else {
            cloth::sheet(g, self.tf, cells, uv_size, shape);
        }
        if !moving {
            let faces = 2 * (columns + 1) * (rows + 1);
            for uv in &mut g.uvs[first..first + faces] {
                uv[1] += available - height;
            }
        }
        let (surface, hem, depth) =
            self.p
                .roller
                .map_or((Surface::Plastic, 0.014, 0.018), |stock| {
                    (
                        if stock.metal_hem {
                            Surface::Metal
                        } else {
                            Surface::Plastic
                        },
                        stock.hem_height_m,
                        stock.hem_depth_m,
                    )
                });
        self.block(
            surface,
            Vec3::new(center, rail - height - hem * 0.5, 0.112),
            Vec3::new(width + 0.006, hem, depth),
            0.002,
        );
    }
}
