//! Physical product size and installation regularity, resolved once into the
//! existing CPU/GPU recipe. No new texture, shader branch or per-texel cost.
use super::*;

fn log_uniform(rng: &mut impl Rng, min: f32, max: f32) -> f32 {
    rng.random_range(min.ln()..max.ln()).exp()
}

pub(super) fn period(seed: u64, seamless: bool) -> f32 {
    let mut rng = stream(seed, 0x464c5343414c45);
    if seamless {
        log_uniform(&mut rng, 0.55, 3.2)
    } else {
        log_uniform(&mut rng, 1.6, 6.4)
    }
}

pub(super) fn resolve(r: &mut MaterialRecipe, style: u32) {
    let mut format = stream(r.seed, 0x464c464f524d4154);
    let mut installation = stream(r.seed, 0x464c524547554c);
    let f = r.flooring.as_mut().expect("resolved floor installation");
    match f.layout {
        FloorLayout::Straight | FloorLayout::BrokenBond => {
            let width = match style {
                0 => log_uniform(&mut format, 0.065, 0.40),
                1 => log_uniform(&mut format, 0.25, 1.0),
                _ => log_uniform(&mut format, 0.07, 1.6),
            };
            let aspect = if style == 0 {
                log_uniform(&mut format, 3.5, 18.)
            } else if format.random_bool(0.40) {
                // Manufactured square stock remains common, alongside slabs,
                // rectangular carpet tiles and small-format mineral mosaics.
                log_uniform(&mut format, 0.97, 1.03)
            } else {
                log_uniform(&mut format, 0.4, 3.)
            };
            let length = (width * aspect).clamp(
                [0.32, 0.25, 0.08][style as usize],
                [2.8, 1.5, 2.2][style as usize],
            );
            // Avoid collapsing long/wide stock onto the minimum count or fine
            // stock onto the maximum. Every atlas closes with 2..48 cells/axis.
            r.period_m = r
                .period_m
                .clamp(2.05 * width.max(length), 48. * width.min(length));
            f.cells = [width, length].map(|m| (r.period_m / m).round().clamp(2., 48.) as u32);
        }
        FloorLayout::Basket => {
            let block = log_uniform(&mut format, 0.20, 0.95);
            let cells = (r.period_m / block).round().clamp(2., 48.) as u32;
            f.cells = [cells; 2];
        }
        FloorLayout::Herringbone => {
            let width = if style == 0 {
                log_uniform(&mut format, 0.06, 0.27)
            } else {
                log_uniform(&mut format, 0.08, 0.50)
            };
            let unit = 2 * f.craft.as_ref().unwrap().plank_ratio;
            let block = width * unit as f32;
            let lo = (1.6 / block).ceil().max(1.);
            let hi = (6.4 / block).floor().min((48 / unit) as f32).max(lo);
            let blocks = (r.period_m / block).round().clamp(lo, hi) as u32;
            f.cells = [blocks * unit; 2];
            r.period_m = block * blocks as f32;
        }
        FloorLayout::Chevron => {
            let lane = log_uniform(&mut format, 0.30, 1.2);
            let course = log_uniform(&mut format, 0.065, 0.28);
            f.cells = [
                (r.period_m / lane / 2.).round().clamp(1., 24.) as u32 * 2,
                (r.period_m / course).round().clamp(2., 48.) as u32,
            ];
            let slope = format.random_range(0.5..1.5);
            f.craft.as_mut().unwrap().chevron_rise =
                (slope * f.cells[1] as f32 / f.cells[0] as f32).clamp(0.25, 3.);
        }
        FloorLayout::Hexagonal => {
            let width = log_uniform(&mut format, 0.08, 0.80);
            r.period_m = r.period_m.max(width * 4.);
            let nx = (r.period_m / width).round().clamp(4., 48.) as u32;
            f.cells = [nx, (nx as f32 / 3_f32.sqrt()).round().max(2.) as u32];
        }
        FloorLayout::Seamless => {}
    }
    if matches!(f.layout, FloorLayout::Straight | FloorLayout::BrokenBond) && style != 1 {
        // Include both uniform manufactured products and genuinely mixed cut
        // stock. Shared interval boundaries keep every installation gap-free.
        let variable = installation.random_bool(if style == 0 { 0.80 } else { 0.35 });
        if variable {
            let amount = installation.random_range(0.12_f32..1.);
            let minimum = if style == 0 {
                [0.04, 0.12]
            } else {
                [0.04, 0.04]
            };
            let desired = [
                installation.random_range(0.0..0.24),
                installation.random_range(0.08..0.39),
            ];
            f.stock.as_mut().unwrap().cut_jitter = std::array::from_fn(|i| {
                let size = r.period_m / f.cells[i] as f32;
                (desired[i] * amount).min((0.5 * (1. - minimum[i] / size)).clamp(0., 0.4))
            });
        }
    }
    if style == 1 {
        f.tile_definition = if installation.random_bool(0.40) {
            0.
        } else {
            installation.random_range(0.0_f32..1.).powf(1.6)
        };
        // Broadloom must not retain a hidden tiled gloss field when its visual
        // seams are absent. Resolve this into sampled fields; old recipes replay.
        f.piece_roughness *= f.tile_definition;
        let c = r.carpet.as_mut().unwrap();
        let mut dye = stream(r.seed, 0x464c4d4f544946);
        c.repeats = std::array::from_fn(|_| {
            let spacing = log_uniform(&mut dye, 0.10, 3.0);
            (r.period_m / spacing).round().clamp(1., 16.) as u32
        });
        let d = c.dye.as_mut().unwrap();
        d.dye_cells = std::array::from_fn(|_| {
            let scale = log_uniform(&mut dye, 0.15, 3.0);
            (r.period_m / scale).round().clamp(1., 16.) as u32
        });
    }
}

#[cfg(test)]
mod tests;
