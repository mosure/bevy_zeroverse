//! Grounded courtyard shade structures. The same installation controls the
//! planted headroom, beam bearings, slat openings and fabric attachments.
use super::*;
use crate::scene::procedural_indoor::geometry::Geometry;
mod canopy;
#[cfg(test)]
mod tests;

#[derive(Clone, Copy)]
enum Roof {
    Rafters,
    Louvres {
        tilt: f32,
    },
    Canvas {
        covered: usize,
        from_right: bool,
        sag: f32,
    },
}

struct Screen {
    bottom: f32,
    height: f32,
    vertical: bool,
    pitch: f32,
    solid: f32,
}

pub(super) struct Program {
    stock: Surface,
    post: f32,
    supports: Vec2,
    beam_bottom: f32,
    beam_height: f32,
    roof_half: Vec2,
    bays: usize,
    blade: Vec2,
    roof: Roof,
    screen: Option<Screen>,
}

impl Program {
    pub(super) fn sample(p: &Parcel) -> Self {
        let mut r = stream(p.seed, u64::from_le_bytes(*b"Pergolas"));
        let stock = if r.random_bool(0.61) {
            Surface::WoodEdge
        } else {
            Surface::Metal
        };
        let supports =
            p.size * Vec2::new(r.random_range(0.397..0.437), r.random_range(0.391..0.435));
        let roof_half =
            p.size * Vec2::new(r.random_range(0.457..0.485), r.random_range(0.456..0.486));
        let post = if stock == Surface::Metal {
            r.random_range(0.083..0.139)
        } else {
            r.random_range(0.105..0.174)
        };
        let beam_bottom = r.random_range(2.37..3.24);
        let beam_height = (supports.x * r.random_range(0.045..0.075) + 0.052).clamp(0.127, 0.29);
        let choice = r.random::<f32>();
        let target_pitch = if choice < 0.37 {
            r.random_range(0.25..0.56)
        } else if choice < 0.69 {
            r.random_range(0.18..0.36)
        } else {
            r.random_range(0.59..1.10)
        };
        let bays = (2. * supports.x / target_pitch).round().clamp(3., 32.) as usize;
        let pitch = 2. * supports.x / bays as f32;
        let (roof, width, depth) = if choice < 0.37 {
            (
                Roof::Rafters,
                r.random_range(0.049..0.104),
                r.random_range(0.09..0.19),
            )
        } else if choice < 0.69 {
            (
                Roof::Louvres {
                    tilt: r.random_range(-0.89..0.89),
                },
                pitch * r.random_range(0.48..0.88),
                r.random_range(0.019..0.049),
            )
        } else {
            (
                Roof::Canvas {
                    covered: (bays as f32 * r.random_range(0.36_f32..1.09))
                        .round()
                        .clamp(1., bays as f32) as usize,
                    from_right: r.random(),
                    sag: r.random_range(0.042..0.133),
                },
                r.random_range(0.040..0.081),
                r.random_range(0.060..0.115),
            )
        };
        let screen = r.random_bool(0.47).then(|| {
            let bottom = r.random_range(0.12..0.49);
            Screen {
                bottom,
                height: r.random_range(1.14..2.18) - bottom,
                vertical: r.random(),
                pitch: r.random_range(0.139..0.276),
                solid: r.random_range(0.35..0.72),
            }
        });
        Self {
            stock,
            post,
            supports,
            beam_bottom,
            beam_height,
            roof_half,
            bays,
            blade: Vec2::new(width.min(1.75 * (roof_half.x - supports.x)), depth),
            roof,
            screen,
        }
    }

    fn pitch(&self) -> f32 {
        2. * self.supports.x / self.bays as f32
    }
    fn beam_top(&self) -> f32 {
        self.beam_bottom + self.beam_height
    }
    fn blade_geometry(&self) -> (Quat, f32) {
        let angle = if let Roof::Louvres { tilt } = self.roof {
            tilt
        } else {
            0.
        };
        let (s, c) = angle.sin_cos();
        let half_height = 0.5 * (self.blade.x * s.abs() + self.blade.y * c);
        (
            Quat::from_rotation_z(angle),
            self.beam_top() - 0.007 + half_height,
        )
    }

    pub(super) fn headroom(&self) -> f32 {
        if let Roof::Canvas { sag, .. } = self.roof {
            self.beam_bottom
                .min(self.fabric_height() - self.pitch() * sag - canopy::GAUGE)
        } else {
            self.beam_bottom
        }
    }
    fn fabric_height(&self) -> f32 {
        self.blade_geometry().1 + self.blade.y * 0.26
    }

    pub(super) fn measure(&self, out: &mut Vec<(&'static str, f32)>) {
        out.extend([
            ("exterior_pergola_clear_height_m", self.headroom()),
            ("exterior_pergola_batten_pitch_m", self.pitch()),
        ]);
        if let Roof::Canvas { covered, .. } = self.roof {
            out.push((
                "exterior_pergola_covered_bay_fraction",
                covered as f32 / self.bays as f32,
            ));
        }
        if let Roof::Louvres { tilt } = self.roof {
            out.push(("exterior_pergola_louvre_degrees", tilt.to_degrees()));
        }
        if let Some(s) = &self.screen {
            out.push(("exterior_pergola_screen_height_m", s.height));
        }
    }
}

fn block(g: &mut Geometry, tf: &Transform, p: Vec3, size: Vec3, rotation: Quat, bevel: f32) {
    g.cuboid(
        size,
        bevel,
        Transform::from_translation(tf.transform_point(p)).with_rotation(tf.rotation * rotation),
    );
}

pub(super) fn build(a: &mut Assembly, p: &Parcel) {
    let plan = Program::sample(p);
    let tf =
        Transform::from_xyz(p.center.x, 0., p.center.y).with_rotation(Quat::from_rotation_y(p.yaw));
    let stock = |s| super::super::super::materials::exterior::label(s, "other_structure", p.seed);
    let (x, z) = (plan.supports.x, plan.supports.y);
    let g = a.part(Surface::Metal, &stock(Surface::Metal));
    for u in [-x, x] {
        for v in [-z, z] {
            // Finish grade is +30 mm; the post tenon enters the base plate.
            block(
                g,
                &tf,
                Vec3::new(u, 0.040, v),
                Vec3::new(plan.post + 0.040, 0.024, plan.post + 0.040),
                Quat::IDENTITY,
                0.003,
            );
        }
    }
    let g = a.part(plan.stock, &stock(plan.stock));
    let top = plan.beam_bottom + 0.007;
    for v in [-z, z] {
        for u in [-x, x] {
            block(
                g,
                &tf,
                Vec3::new(u, (top + 0.048) * 0.5, v),
                Vec3::new(plan.post, top - 0.048, plan.post),
                Quat::IDENTITY,
                0.003,
            );
        }
        block(
            g,
            &tf,
            Vec3::new(0., plan.beam_bottom + plan.beam_height * 0.5, v),
            Vec3::new(2. * plan.roof_half.x, plan.beam_height, plan.post + 0.014),
            Quat::IDENTITY,
            0.003,
        );
    }
    let (rotation, height) = plan.blade_geometry();
    for i in 0..=plan.bays {
        block(
            g,
            &tf,
            Vec3::new(-x + i as f32 * plan.pitch(), height, 0.),
            Vec3::new(plan.blade.x, plan.blade.y, 2. * plan.roof_half.y),
            rotation,
            0.,
        );
    }
    if let Some(s) = &plan.screen {
        let half = x - plan.post * 0.5 + 0.003;
        let z = -z;
        if s.vertical {
            for h in [s.bottom + 0.065 * s.height, s.bottom + 0.935 * s.height] {
                block(
                    g,
                    &tf,
                    Vec3::new(0., h, z),
                    Vec3::new(2. * half, 0.072, 0.050),
                    Quat::IDENTITY,
                    0.002,
                );
            }
            let n = ((2. * half / s.pitch).round() as usize).max(2);
            let step = 2. * half / n as f32;
            for i in 0..n {
                block(
                    g,
                    &tf,
                    Vec3::new(
                        -half + (i as f32 + 0.5) * step,
                        s.bottom + s.height * 0.5,
                        z + 0.033,
                    ),
                    Vec3::new(step * s.solid, s.height, 0.020),
                    Quat::IDENTITY,
                    0.,
                );
            }
        } else {
            let n = ((s.height / s.pitch).round() as usize).max(2);
            let step = s.height / n as f32;
            for i in 0..n {
                block(
                    g,
                    &tf,
                    Vec3::new(0., s.bottom + (i as f32 + 0.5) * step, z),
                    Vec3::new(2. * half, step * s.solid, 0.035),
                    Quat::IDENTITY,
                    0.,
                );
            }
        }
    }
    if let Roof::Canvas {
        covered,
        from_right,
        sag,
    } = plan.roof
    {
        let g = a.part(Surface::FabricAlt, &stock(Surface::FabricAlt));
        let start = if from_right { plan.bays - covered } else { 0 };
        let hem = plan.blade.x * 0.5 - 0.003;
        for i in start..start + covered {
            let x0 = -x + i as f32 * plan.pitch();
            canopy::panel(
                g,
                &tf,
                x0 + hem,
                x0 + plan.pitch() - hem,
                z - 0.022,
                plan.fabric_height(),
                plan.pitch() * sag,
            );
        }
    }
}
