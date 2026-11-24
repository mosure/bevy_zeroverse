//! Wet-formed mineral board. Continuous pigment, porosity and metric surface
//! relief share the same field; no directional illumination is baked into it.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::ceiling::{CeilingShape, TileFacing},
    layout::stream,
    preparation::AssetStore,
};
use rand::Rng;
use std::collections::BTreeMap;

const SIZE: usize = 128;
const GRID_SLOT: usize = 123;
pub(crate) const GRID_LABEL: &str = "ceiling#finish123";
const FACINGS: [TileFacing; 3] = [
    TileFacing::FineFiber,
    TileFacing::Pinhole,
    TileFacing::Fissured,
];

impl TileFacing {
    fn slot(self) -> usize {
        match self {
            Self::FineFiber => 120,
            Self::Pinhole => 121,
            Self::Fissured => 122,
        }
    }
    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::FineFiber => "ceiling#finish120",
            Self::Pinhole => "ceiling#finish121",
            Self::Fissured => "ceiling#finish122",
        }
    }
}

pub(super) fn supports(surface: Surface, slot: usize) -> bool {
    (surface == Surface::Ceiling && (120..=122).contains(&slot))
        || (surface == Surface::Metal && slot == GRID_SLOT)
}

/// Stamp irregular depressions into a periodic height field. Cost follows the
/// area of the marks, without a per-pixel search over the whole sheet.
fn pores(facing: TileFacing, period_m: f32, r: &mut impl Rng) -> Vec<f32> {
    let mut field = vec![0_f32; SIZE * SIZE];
    let per_metre = SIZE as f32 / period_m;
    let density = match facing {
        TileFacing::FineFiber => r.random_range(6200.0..13000.),
        TileFacing::Pinhole => r.random_range(10000.0..23000.),
        TileFacing::Fissured => r.random_range(4600.0..9300.),
    };
    for _ in 0..(density * period_m * period_m).round() as usize {
        let center = Vec2::new(
            r.random_range(0.0..SIZE as f32),
            r.random_range(0.0..SIZE as f32),
        );
        let width = r.random_range(0.00062..0.00185) * per_metre;
        let length = width
            * match facing {
                TileFacing::FineFiber => r.random_range(1.1..2.6),
                TileFacing::Pinhole => r.random_range(1.0..1.48),
                TileFacing::Fissured => r.random_range(1.45..5.6),
            };
        let axis = Vec2::from_angle(r.random_range(0.0..std::f32::consts::PI));
        let bend = r.random_range(-0.48..0.48);
        let strength = r.random_range(0.44..1.0);
        let radius = axis.abs() * length + axis.perp().abs() * (width * 1.5);
        let lo = (center - radius).floor().as_ivec2();
        let hi = (center + radius).ceil().as_ivec2();
        for y in lo.y..=hi.y {
            for x in lo.x..=hi.x {
                let v = Vec2::new(x as f32 + 0.5, y as f32 + 0.5) - center;
                let u = v.dot(axis) / length;
                let w = v.dot(axis.perp()) / width - bend * (1. - u * u);
                let weight = (1. - u * u - w * w).clamp(0., 1.) * strength;
                let idx =
                    y.rem_euclid(SIZE as i32) as usize * SIZE + x.rem_euclid(SIZE as i32) as usize;
                field[idx] = field[idx].max(weight);
            }
        }
    }
    field
}

fn board_material(
    seed: u64,
    facing: TileFacing,
    images: &mut impl AssetStore<Image>,
) -> StandardMaterial {
    let mut r = stream(
        seed,
        u64::from_le_bytes(*b"TileFace") ^ facing.slot() as u64,
    );
    let period = r.random_range(0.145..0.235);
    let pore_field = pores(facing, period, &mut r);
    let value = if r.random_bool(0.09) {
        r.random_range(0.20..0.39)
    } else {
        r.random_range(0.85..0.96)
    };
    let warmth = r.random_range(-0.009..0.026);
    let pigment = Vec3::new(value + warmth, value, value - warmth);
    let relief = r.random_range(0.00055..0.00155)
        * if facing == TileFacing::FineFiber {
            0.48
        } else {
            1.
        };
    let pigment_loss = r.random_range(0.23..0.48)
        * if facing == TileFacing::FineFiber {
            0.48
        } else {
            1.
        };
    let grain_seed = r.random();
    sheet::sheet_material::<SIZE>(images, Vec2::splat(period), 0., 1., |p| {
        let coord = (p * SIZE as f32).as_uvec2();
        let pore = pore_field[coord.y as usize * SIZE + coord.x as usize];
        let fiber = periodic_noise(p.x, p.y, 51, 47, grain_seed) - 0.5;
        let cloud = periodic_noise(p.x, p.y, 6, 7, grain_seed ^ 1) - 0.5;
        let color = pigment * (1. - pigment_loss * pore + fiber * 0.059 + cloud * 0.033);
        (
            color,
            0.00032 * fiber - relief * pore,
            0.84 + 0.092 * pore + 0.045 * fiber,
        )
    })
}

fn grid_material(seed: u64) -> StandardMaterial {
    let mut r = stream(seed, u64::from_le_bytes(*b"TeeCoat "));
    let value = if r.random_bool(0.13) {
        r.random_range(0.15..0.38)
    } else {
        r.random_range(0.77..0.95)
    };
    let warmth = r.random_range(-0.016..0.025);
    StandardMaterial {
        base_color: Color::srgb(value + warmth, value, value - warmth),
        perceptual_roughness: r.random_range(0.48..0.76),
        ..default()
    }
}

pub(super) fn install(
    scene: &IndoorManifest,
    images: &mut impl AssetStore<Image>,
    materials: &mut impl AssetStore<StandardMaterial>,
    selection: Option<&MaterialSelection>,
    variants: &mut BTreeMap<(Surface, usize), Handle<StandardMaterial>>,
) {
    // Staged captures pass the realized part set. The standalone full-material
    // builder obtains the same keys from the recorded installation.
    let requested = |key, facing: Option<TileFacing>| {
        selection.map_or_else(
            || {
                scene.ceiling_installation.iter().any(|c| {
                    c.room_grid.as_ref().is_some_and(|grid| facing.is_none_or(|f|f == grid.tile.facing))
                    || c.modules.iter().any(|m| matches!(m.detail.as_ref().map(|d| &d.shape),
                        Some(CeilingShape::LayIn {tile,..}) if facing.is_none_or(|f| f==tile.facing)))
                })
            },
            |s| s.finishes.contains(&key),
        )
    };
    let seed = scene.material_seed();
    for facing in FACINGS {
        let key = (Surface::Ceiling, facing.slot());
        if requested(key, Some(facing)) {
            variants.insert(key, materials.add(board_material(seed, facing, images)));
        }
    }
    let key = (Surface::Metal, GRID_SLOT);
    let coated_register = selection.is_none()
        && scene.ceiling_installation.iter().any(|c| {
            c.modules.iter().any(|m| {
                m.duct_register_stock(scene.generator_version)
                    .is_some_and(|s| s.coated)
            })
        });
    if requested(key, None) || coated_register {
        variants.insert(key, materials.add(grid_material(seed)));
    }
}

#[cfg(test)]
mod tests;
