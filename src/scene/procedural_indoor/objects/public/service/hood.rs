//! Extraction hoods fit a replaceable intake and work lights to the canopy.
//! The independently fitted flue reaches the installation's measured roof contact.
use super::*;
use crate::scene::procedural_indoor::materials::{coated_stock, variants};
mod intake;
mod legacy;
mod lighting;
pub use intake::{FilterDeck, FilterStyle, LampStyle};
pub const FITTED_INTAKE: u32 = 248;

#[derive(Clone, Debug, serde::Serialize)]
pub struct HoodProgram {
    pub construction: u8,
    /// Stock draw used by archived proportional canopies and current metric
    /// sizes. Use `canopy_top` for the installed height in metres.
    pub canopy_fraction: f32,
    pub chimney_fraction: Vec2,
    pub chimney_x: f32,
    pub taper: f32,
    /// Catalogue requests. `FilterDeck::fit` returns the installed counts.
    pub filters: usize,
    pub lights: usize,
    pub finish: Surface,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub coated_finish_slot: Option<usize>,
}

impl HoodProgram {
    pub(crate) fn chimney(&self, size: Vec3) -> (Vec3, Vec2) {
        (
            Vec3::new(size.x * self.chimney_x, 0., -size.z * 0.10),
            size.xz() * self.chimney_fraction,
        )
    }
    /// Top of the manufactured canopy above the bottom of the installed assembly.
    /// The remaining rise belongs to the flue, including below tall ceilings.
    pub fn canopy_top(&self, height: f32, version: u32) -> f32 {
        if version < FITTED_INTAKE {
            return 0.018 + height * self.canopy_fraction;
        }
        let (lo, hi) = match self.construction {
            0 => (0.095, 0.182),
            1 => (0.178, 0.340),
            2 => (0.115, 0.238),
            _ => (0.142, 0.295),
        };
        let t = (self.canopy_fraction - 0.19) / (0.48 - 0.19);
        (lo + (hi - lo) * t).min(height * 0.80)
    }
}

pub fn hood_parameters(o: &IndoorObject) -> HoodProgram {
    parameters_at(
        o,
        crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
    )
}

fn parameters_at(o: &IndoorObject, version: u32) -> HoodProgram {
    let mut r = stream(o.seed, 0x484f_4f44_5348_4150);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x484f_4f44_4649_4e49);
    let stock = [Surface::Metal, Surface::Plastic, Surface::Chrome][f.random_range(0..3)];
    let coated_finish_slot = if version >= FITTED_INTAKE && stock == Surface::Plastic {
        coated_stock::slot(
            Surface::Metal,
            variants::object_slot(Surface::Metal, o.seed, o.finish_seed),
        )
    } else {
        None
    };
    HoodProgram {
        construction: r.random_range(0..4),
        canopy_fraction: r.random_range(0.19..0.48),
        chimney_fraction: Vec2::new(r.random_range(0.20..0.48), r.random_range(0.28..0.52)),
        chimney_x: r.random_range(-0.12..0.12),
        taper: r.random_range(0.45..0.78),
        filters: r.random_range(1..=3),
        lights: r.random_range(1..=3),
        finish: if coated_finish_slot.is_some() {
            Surface::Metal
        } else {
            stock
        },
        coated_finish_slot,
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let p = parameters_at(o, version);
    let s = o.size;
    let fitted = version >= FITTED_INTAKE;
    let base = if fitted { intake::ROOF } else { 0.018 };
    let top = p.canopy_top(s.y, version);
    let (center, chimney) = p.chimney(s);
    let coated;
    let surface = p.finish;
    let label = if let Some(slot) = p.coated_finish_slot {
        coated = format!("{}#finish{slot}", o.kind.class_name());
        coated.as_str()
    } else {
        o.kind.class_name()
    };
    match p.construction {
        0 => a.box_part(
            surface,
            label,
            Vec3::Y * (base + top) * 0.5,
            Vec3::new(s.x * 0.98, top - base, s.z * 0.98),
            0.010,
        ),
        1 => a.part(surface, label).housing(
            &[
                (base, s.xz() * 0.49),
                (base + 0.025, s.xz() * 0.49),
                (top, s.xz() * (p.taper * 0.49)),
            ],
            10.,
            Transform::IDENTITY,
        ),
        2 => {
            // Elliptical glass canopy attached to the filter/motor housing.
            if fitted {
                intake::glass_wing(a, o, center.xz(), chimney, base);
            } else {
                a.part(Surface::Glass, label).housing(
                    &[(base, s.xz() * 0.49), (base + 0.014, s.xz() * 0.49)],
                    2.6,
                    Transform::IDENTITY,
                );
            }
            a.box_part(
                surface,
                label,
                center.with_y((top + base) * 0.5),
                Vec3::new(chimney.x * 1.2, top - base, chimney.y * 1.2),
                0.008,
            );
        }
        _ => a.part(surface, label).housing(
            &[
                (base, s.xz() * 0.39),
                (base + (top - base) * 0.3, s.xz() * 0.48),
                (top, s.xz() * (p.taper * 0.45)),
            ],
            2.2,
            Transform::IDENTITY,
        ),
    }
    // Overlap the real canopy volume, never leave the flue floating above it.
    let flue_bottom = if fitted {
        base + (top - base) * 0.4
    } else {
        base
    };
    a.box_part(
        surface,
        label,
        center.with_y((s.y + flue_bottom) * 0.5),
        Vec3::new(chimney.x, s.y - flue_bottom, chimney.y),
        0.005,
    );
    if fitted {
        FilterDeck::fit(o, &p).build(a, o);
    } else {
        legacy::face(a, o, &p, base, center, chimney);
    }
}

#[cfg(test)]
mod tests;
