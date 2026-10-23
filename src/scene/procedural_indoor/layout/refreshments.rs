//! Functional refreshment worktops: equipment first, optional companions and
//! supplies second. No fixed left/right pair or mandatory clutter density.
use super::*;
use crate::scene::procedural_indoor::{objects, placement::SurfacePattern};

#[derive(Debug, Clone, Serialize)]
pub(crate) struct RefreshmentProgram {
    pub primary: ObjectKind,
    pub companion_probability: f32,
    pub accessories: usize,
    pub occupied_fraction: f32,
    pub appliance_yaw_spread: f32,
}

pub(crate) fn parameters(
    support: &IndoorObject,
    required: Option<ObjectKind>,
) -> RefreshmentProgram {
    let mut r = stream(support.seed, 0x5245_4652_4553_4850);
    RefreshmentProgram {
        primary: required.unwrap_or_else(|| {
            [
                ObjectKind::CoffeeMaker,
                ObjectKind::Kettle,
                ObjectKind::DrinkDispenser,
            ][r.random_range(0..3)]
        }),
        companion_probability: r.random_range(0.08..0.72),
        accessories: r.random_range(0..=7),
        occupied_fraction: r.random_range(0.22..0.58),
        appliance_yaw_spread: r.random_range(0.04..0.32),
    }
}

impl IndoorManifest {
    pub(super) fn dress_refreshment_counter(
        &mut self,
        support: &IndoorObject,
        required: Option<ObjectKind>,
    ) {
        let p = parameters(support, required);
        let mut r = stream(support.seed, 0x5245_4652_4553_4853);
        let pattern = SurfacePattern::sample(&mut r);
        let bounds = if support.kind == ObjectKind::KitchenCounter {
            objects::kitchen::counter_top_bounds(support)
        } else {
            (-support.size.xz() * 0.5, support.size.xz() * 0.5)
        };
        let mut choices = vec![p.primary];
        let mut companions = [
            ObjectKind::CoffeeMaker,
            ObjectKind::DrinkDispenser,
            ObjectKind::Kettle,
        ];
        use rand::seq::SliceRandom;
        companions.shuffle(&mut r);
        for kind in companions {
            if kind != p.primary && r.random_bool(p.companion_probability as f64) {
                choices.push(kind);
            }
        }
        for _ in 0..p.accessories {
            choices.push(
                [
                    ObjectKind::Mug,
                    ObjectKind::CoffeeCup,
                    ObjectKind::Dishware,
                    ObjectKind::PaperTowel,
                    ObjectKind::FruitBowl,
                    ObjectKind::KitchenTowel,
                ][r.random_range(0..6)],
            );
        }
        let mut occupied = 0.;
        for (index, kind) in choices.into_iter().enumerate() {
            if index > 0
                && occupied >= p.occupied_fraction * (bounds.1 - bounds.0).element_product()
            {
                break;
            }
            let seed = r.random();
            let size = match kind {
                ObjectKind::CoffeeMaker => objects::hospitality::coffee::dimensions(seed),
                ObjectKind::DrinkDispenser => objects::public::dimensions(kind, seed),
                ObjectKind::Kettle => Vec3::new(
                    r.random_range(0.16..0.23),
                    r.random_range(0.19..0.30),
                    r.random_range(0.23..0.31),
                ),
                ObjectKind::Mug | ObjectKind::CoffeeCup => {
                    // Cup handles require more width than the vessel itself.
                    let d = r.random_range(0.065..0.10);
                    Vec3::new(
                        d * if kind == ObjectKind::Mug { 1.32 } else { 1. },
                        r.random_range(0.08..0.15),
                        d,
                    )
                }
                _ => objects::kitchen::dimensions(kind, seed),
            };
            let appliance = matches!(
                kind,
                ObjectKind::CoffeeMaker | ObjectKind::DrinkDispenser | ObjectKind::Kettle
            );
            // Barista equipment faces the staff aisle, self-service taps face
            // guests; kitchen equipment faces the accessible cabinet front.
            let heading = if support.kind == ObjectKind::ServiceCounter
                && kind != ObjectKind::DrinkDispenser
            {
                std::f32::consts::PI
            } else {
                0.
            };
            for attempt in 0..24 {
                let yaw = if appliance {
                    heading
                        + if attempt >= 16 {
                            0.
                        } else {
                            r.random_range(-p.appliance_yaw_spread..p.appliance_yaw_spread)
                        }
                } else {
                    pattern.yaw(&mut r)
                };
                let Some(proposal) = pattern.propose(bounds, size.xz(), yaw, 0.018, &mut r) else {
                    continue;
                };
                let position = support.transform().transform_point(Vec3::new(
                    proposal.center.x,
                    support.size.y,
                    proposal.center.y,
                ));
                let mut o = self.candidate(kind, position, size, support.yaw + yaw, &mut r);
                o.seed = seed;
                o.support = Some(support.id);
                o.solid = false;
                o.neighbor = support.neighbor;
                if !self.permits_table_item(support, kind) || !self.prop_clear(&o, support, 0.012) {
                    continue;
                }
                if let Some(program) = self
                    .assembly_diagnostics
                    .programs
                    .iter_mut()
                    .find(|p| p.members.contains(&support.id))
                {
                    program.members.push(o.id);
                }
                occupied += size.x * size.z;
                self.objects.push(o);
                break;
            }
        }
    }
}

#[cfg(test)]
mod tests;
