//! A formed substrate and its occupant-facing surface share a closed perimeter.
//! All dimensions are in metres; a fully folded seat clears the front skin.
use super::*;
mod topology;

#[derive(Clone, Copy, Debug, serde::Serialize)]
struct CarryGrip {
    center: Vec2,
    half: Vec2,
    roundness: f32,
}

impl CarryGrip {
    fn frame_half(self) -> Vec2 {
        let to_edge = Vec2::new(
            1. - self.center.x.abs(),
            self.center.y.min(1. - self.center.y),
        );
        // The adjacent land follows the shallow ends of the opening. Keep it
        // narrow in U where the outer wings of a tall shell turn down.
        let margin = Vec2::new((0.20 * self.half.x).min(0.04), 0.65 * self.half.y);
        self.half + margin.min(0.6 * (to_edge - self.half))
    }

    fn panel_uv(self, q: Vec2, frame: Vec2) -> Vec2 {
        let lower = Vec2::new(-1., 0.) - (self.center - frame);
        let upper = Vec2::ONE - (self.center + frame);
        self.center
            + frame * q.clamp(Vec2::NEG_ONE, Vec2::ONE)
            + lower * (-q - Vec2::ONE).max(Vec2::ZERO)
            + upper * (q - Vec2::ONE).max(Vec2::ZERO)
    }

    fn uv(self, q: Vec2) -> Vec2 {
        let rounding = (Vec2::ONE - self.roundness * q.yx() * q.yx()).map(f32::sqrt);
        self.center + self.half * q * rounding
    }
}

#[derive(Debug, serde::Serialize)]
pub(super) struct Backrest {
    pub(super) z: f32,
    bottom: f32,
    height: f32,
    width: f32,
    corner: f32,
    waist: f32,
    taper: f32,
    cup: f32,
    rake: f32,
    pad: f32,
    gauge: f32,
    grip: Option<CarryGrip>,
}

impl Backrest {
    pub(super) fn sample(o: &IndoorObject, w: f32, z: f32, fixed: bool, r: &mut impl Rng) -> Self {
        let bottom = r.random_range(0.545..0.615);
        let available = o.size.z * 0.5 - z - 0.006;
        let cup = available * r.random_range(0.26..0.43);
        let mut back = Self {
            z,
            bottom,
            height: o.size.y - 0.010 - bottom,
            width: w * r.random_range(0.89..1.05),
            corner: r.random_range(0.075..0.195),
            waist: r.random_range(0.02..0.15),
            taper: r.random_range(0.015..0.14),
            cup,
            rake: available * r.random_range(0.23..0.38),
            pad: if fixed {
                cup * r.random_range(0.40..0.88)
            } else {
                0.
            },
            gauge: available * r.random_range(0.09..0.15),
            grip: None,
        };
        if !fixed {
            back.grip = back.sample_grip(o.seed);
        }
        back
    }

    fn sample_grip(&self, seed: u64) -> Option<CarryGrip> {
        let mut r = stream(seed, u64::from_le_bytes(*b"GripMill"));
        if !r.random_bool(0.58) {
            return None;
        }
        let center = Vec2::new(r.random_range(-0.07..0.07), r.random_range(0.70..0.84));
        let span = self.width
            * (1. - self.waist * 4. * center.y * (1. - center.y) - self.taper * center.y);
        let height = self.height * (1. - 2. * self.corner * center.x.powi(4));
        let half = Vec2::new(
            r.random_range(0.085..0.145) / span,
            r.random_range(0.025..0.043) / (2. * height),
        );
        // Fit a hand-sized aperture with an intact web above and on both sides.
        (half.x + center.x.abs() < 0.53 && (1. - center.y - half.y) * height > 0.022).then(|| {
            CarryGrip {
                center,
                half,
                roundness: r.random_range(0.25..0.50),
            }
        })
    }

    pub(super) fn grip_dimensions(&self) -> Option<Vec2> {
        let grip = self.grip?;
        let point = |q| {
            let uv = grip.uv(q);
            self.sample_skin(uv.x, uv.y, false).0
        };
        Some(Vec2::new(
            point(Vec2::X).x - point(Vec2::NEG_X).x,
            point(Vec2::Y).y - point(Vec2::NEG_Y).y,
        ))
    }

    /// Normal and position for a continuous local shell. Front bulging changes
    /// its thickness inside the sewn edge without pushing into the folding arc.
    fn sample_skin(&self, u: f32, v: f32, rear: bool) -> (Vec3, Vec3) {
        let wave = 4. * v * (1. - v);
        let slope = 4. - 8. * v;
        let hw = self.width * 0.5;
        let span = 1. - self.waist * wave - self.taper * v;
        let corner = self.corner * u.powi(4);
        let bow = self.cup - if rear { 0. } else { self.pad * wave };
        let point = Vec3::new(
            hw * u * span,
            self.bottom + self.height * (v + corner * (1. - 2. * v)),
            self.z + self.rake * v + (1. - u * u) * bow + if rear { self.gauge } else { 0. },
        );
        let du = Vec3::new(
            hw * span,
            self.height * 4. * self.corner * u.powi(3) * (1. - 2. * v),
            -2. * u * bow,
        );
        let dv = Vec3::new(
            hw * u * (-self.waist * slope - self.taper),
            self.height * (1. - 2. * corner),
            self.rake
                - if rear {
                    0.
                } else {
                    (1. - u * u) * self.pad * slope
                },
        );
        let n = du.cross(dv).normalize();
        (point, if rear { n } else { -n })
    }

    pub(super) fn mount(&self, side: f32) -> Vec3 {
        self.sample_skin(side * 0.81, 0.43, true).0
    }

    /// The stay bears against a plate on the rear face of the pressed shell.
    /// Offsetting along the local normal keeps the round tube behind the skin.
    pub(super) fn stay_plate(&self, side: f32) -> (Transform, Vec3, Vec3) {
        let (contact, outward) = self.sample_skin(side * 0.79, 0.43, true);
        let size = Vec3::new(0.035, 0.048, 0.010);
        let pose = Transform::from_translation(contact + outward * (size.z * 0.5 - 0.001))
            .looking_to(-outward, Vec3::Y);
        (pose, size, contact + outward * 0.017)
    }

    pub(super) fn build(&self, a: &mut Assembly, face: Surface, shell: Surface) {
        let mesh = topology::get(self.grip.is_some());
        let frame_half = self.grip.map(CarryGrip::frame_half).unwrap_or_default();
        let mut rim_positions = [[Vec3::ZERO; 2]; topology::MAX_VERTICES];
        for (skin, rear) in [false, true].into_iter().enumerate() {
            let g = a.part(if rear { shell } else { face }, "chair");
            let start = g.positions.len() as u32;
            for (i, &node) in mesh.nodes.iter().enumerate() {
                let (p, n) = if rear && self.pad == 0. && face == shell {
                    // A rigid back is one pressed sheet of constant Z gauge.
                    // Its front vertices immediately precede these rear vertices.
                    let normal = g.normals[start as usize - mesh.nodes.len() + i];
                    (
                        rim_positions[i][0] + Vec3::Z * self.gauge,
                        -Vec3::from_array(normal),
                    )
                } else {
                    let uv = match self.grip {
                        Some(grip) if i < mesh.slot_start => grip.panel_uv(node, frame_half),
                        Some(grip) => grip.uv(node),
                        None => node,
                    };
                    self.sample_skin(uv.x, uv.y, rear)
                };
                rim_positions[i][skin] = p;
                g.positions.push(p.to_array());
                g.normals.push(n.to_array());
                g.uvs
                    .push([if rear { p.x } else { -p.x }, p.y - self.bottom]);
            }
            for &[i, j, k] in &mesh.triangles {
                let tri = if rear { [i, j, k] } else { [i, k, j] };
                g.indices.extend(tri.map(|index| start + index));
            }
        }
        let g = a.part(shell, "chair");
        for ring in [&mesh.perimeter, &mesh.hole] {
            let Some((&first, rest)) = ring.split_first() else {
                continue;
            };
            let mut prev = rim_positions[first];
            let mut along = 0.;
            for &node in rest.iter().chain(std::iter::once(&first)) {
                let next = rim_positions[node];
                let distance = along + prev[0].distance(next[0]);
                let n = (next[0] - prev[0]).cross(prev[1] - prev[0]).normalize();
                let start = g.positions.len() as u32;
                for (p, uv) in [
                    (prev[0], [along, 0.]),
                    (next[0], [distance, 0.]),
                    (next[1], [distance, next[1].z - next[0].z]),
                    (prev[1], [along, prev[1].z - prev[0].z]),
                ] {
                    g.positions.push(p.to_array());
                    g.normals.push(n.to_array());
                    g.uvs.push(uv);
                }
                g.indices
                    .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
                prev = next;
                along = distance;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{assert_closed, inside_triangle};
    use super::*;
    use crate::scene::procedural_indoor::objects::audit;

    #[test]
    fn carry_openings_are_clear_through_both_skins_with_solid_webs_and_mounts() {
        let mut openings = 0;
        for seed in 0..192 {
            let mut o = audit::fixture(ObjectKind::Chair, seed);
            o.variant = 10;
            o.size = Vec3::new(
                0.56 + 0.022 * (seed % 11) as f32,
                0.85 + 0.033 * (seed % 10) as f32,
                0.59 + 0.033 * (seed % 7) as f32,
            );
            let seat = TeachingSeat::sample(&o);
            let b = seat.back;
            let mut shell = Assembly::default();
            b.build(&mut shell, seat.face_stock, seat.shell_stock);
            audit::check(&o, &shell).unwrap();
            assert_closed(&shell);
            let Some(grip) = b.grip else {
                continue;
            };
            openings += 1;
            let size = b.grip_dimensions().unwrap();
            assert!((0.0849..0.1451).contains(&size.x) && (0.0249..0.0431).contains(&size.y));
            let hit = |uv: Vec2| {
                let p = b.sample_skin(uv.x, uv.y, false).0.truncate();
                shell.parts.values().any(|g| {
                    g.indices.as_chunks::<3>().0.iter().any(|t| {
                        inside_triangle(
                            t.map(|i| Vec3::from_array(g.positions[i as usize]).truncate()),
                            p,
                        )
                    })
                })
            };
            for q in [
                Vec2::ZERO,
                0.6 * Vec2::X,
                0.6 * Vec2::NEG_X,
                0.55 * Vec2::Y,
                0.55 * Vec2::NEG_Y,
            ] {
                assert!(
                    !hit(grip.center + q * grip.half),
                    "obstructed grip on chair {seed}"
                );
            }
            for d in [Vec2::X, Vec2::NEG_X, Vec2::Y, Vec2::NEG_Y] {
                assert!(
                    hit(grip.center + d * (grip.half + Vec2::splat(0.045))),
                    "missing web on chair {seed}"
                );
            }
            assert!(hit(Vec2::new(0.81, 0.43)) && hit(Vec2::new(-0.81, 0.43)));
            o.variant = 11;
            assert!(TeachingSeat::sample(&o).back.grip.is_none());
        }
        assert!((75..145).contains(&openings), "{openings}");
    }
}
