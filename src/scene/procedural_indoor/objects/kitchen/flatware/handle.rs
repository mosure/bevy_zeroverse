use super::*;
use crate::scene::procedural_indoor::materials::variants::object_slot;
use rand_chacha::ChaCha8Rng;

/// A fitted timber or moulded grip over the metal shank. Dimensions are
/// fractions of the individual utensil so a venue can stock matching sets.
#[derive(Debug, Serialize)]
pub struct FittedGrip {
    pub material: Surface,
    pub length_fraction: f32,
    pub waist: f32,
    pub butt: f32,
}

impl FittedGrip {
    pub(super) fn sample(r: &mut ChaCha8Rng) -> Option<Self> {
        r.random_bool(0.39).then(|| Self {
            material: if r.random_bool(0.44) {
                Surface::Wood
            } else {
                Surface::Plastic
            },
            length_fraction: r.random_range(0.55..0.77),
            waist: r.random_range(0.70..0.97),
            butt: r.random_range(0.53..0.86),
        })
    }

    pub(super) fn build(
        &self,
        a: &mut Assembly,
        o: &IndoorObject,
        p: &FlatwareProgram,
        metal_label: &str,
        span: std::ops::Range<f32>,
        socket_y: f32,
    ) {
        let tail = span.start;
        let run = span.end - tail;
        let radius = o.size.x * (p.handle_width_fraction * 0.5 + 0.06);
        let height = o.size.y * 0.40;
        let length = run * self.length_fraction;
        let stock = o.finish_seed.unwrap_or(o.seed);
        let slot = object_slot(self.material, stock, None);
        a.part(self.material, &format!("kitchen_utensil#finish{slot}"))
            .lathe(
                &[
                    (0., 0.),
                    (radius * self.butt, 0.),
                    (radius, length * 0.08),
                    (radius * self.waist, length * 0.57),
                    (radius * 0.84, length * 0.94),
                    (radius * 0.69, length),
                    (0., length),
                ],
                12,
                Transform::from_xyz(0., height, tail)
                    .with_rotation(Quat::from_rotation_x(std::f32::consts::FRAC_PI_2))
                    .with_scale(Vec3::new(1., 1., height / radius)),
            );
        // Start inside the grip. The metal emerges through a closed shoulder
        // and meets the raised spoon rim or the heel of a fork.
        let start = length * 0.89;
        let dz = run - start;
        let dy = socket_y - height;
        let reach = dz.hypot(dy);
        let shank = radius * 0.48;
        a.part(p.metal, metal_label).lathe(
            &[
                (0., 0.),
                (shank, 0.),
                (shank * 0.75, reach * 0.58),
                (shank * if p.tines == 0 { 0.68 } else { 0.95 }, reach),
                (0., reach),
            ],
            8,
            Transform::from_xyz(0., height, tail + start)
                .with_rotation(Quat::from_rotation_x(dz.atan2(dy)))
                .with_scale(Vec3::new(1., 1., o.size.y * 0.15 / shank)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::objects::audit;

    #[test]
    fn finished_sets_share_handles_and_rest_on_the_table() {
        let mut styles = [0; 3];
        let mut heads = [0; 3];
        for stock in 0..96 {
            let mut utensil = audit::fixture(ObjectKind::Flatware, stock);
            utensil.finish_seed = Some(stock);
            utensil.size = Vec3::new(0.033, 0.009, 0.186);
            let p = parameters(&utensil);
            utensil.seed = utensil.seed.wrapping_add(1037);
            let other = parameters(&utensil);
            assert_eq!(p.metal, other.metal);
            assert_eq!(p.handle_width_fraction, other.handle_width_fraction);
            assert_eq!(
                serde_json::json!(p.fitted_grip),
                serde_json::json!(other.fitted_grip)
            );
            let legacy = parameters_at(&utensil, FITTED_GRIP_VERSION - 1);
            assert!(legacy.fitted_grip.is_none());
            assert_eq!(other.tines, legacy.tines);
            heads[match other.tines {
                0 => 0,
                3 => 1,
                4 => 2,
                _ => unreachable!(),
            }] += 1;
            let mut assembly = Assembly::default();
            build(&mut assembly, &utensil, FITTED_GRIP_VERSION);
            audit::check(&utensil, &assembly).unwrap();
            if let Some(grip) = other.fitted_grip {
                styles[if grip.material == Surface::Wood { 1 } else { 2 }] += 1;
                assert_eq!(assembly.parts.len(), 2);
                let fitted = assembly
                    .parts
                    .iter()
                    .find(|((surface, _), _)| *surface == grip.material)
                    .unwrap()
                    .1;
                let minimum = fitted
                    .positions
                    .iter()
                    .map(|p| p[1])
                    .reduce(f32::min)
                    .unwrap();
                assert!(minimum.abs() < 1e-5);
            } else {
                styles[0] += 1;
                assert_eq!(assembly.parts.len(), 1);
            }
            let (lo, hi) = assembly.bounds();
            assert!(lo.y >= -1e-5 && hi.y <= utensil.size.y + 1e-5);
        }
        assert!(styles.into_iter().chain(heads).all(|n| n >= 6));
    }
}
