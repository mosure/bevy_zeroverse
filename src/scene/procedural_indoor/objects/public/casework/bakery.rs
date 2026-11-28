//! A batch shares its recipe and portion size. Sales and handling leave gaps on the tray.
use super::{
    stocking::{tray_rng, BakeryTray},
    *,
};
mod portion;
use rand::seq::SliceRandom;

pub const VERSION: u32 = 327;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bake {
    Biscuit,
    Roll,
    Bagel,
    Muffin,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct Portion {
    pub center_m: Vec2,
    pub yaw_rad: f32,
    /// Reserved radius and headroom above the tray. The liner fits inside both.
    pub radius_m: f32,
    pub height_m: f32,
    pub seed: u64,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct BakeryBatch {
    pub level: usize,
    pub tray: BakeryTray,
    pub shelf_y_m: f32,
    pub recipe: Bake,
    pub arranged_in_rows: bool,
    pub portions: Vec<Portion>,
}

impl BakeryBatch {
    pub(super) fn fit(
        o: &IndoorObject,
        level: usize,
        shelf_y_m: f32,
        headroom: f32,
        bounds: (Vec2, Vec2),
    ) -> Option<Self> {
        let tray = BakeryTray::sample(headroom, bounds, &mut tray_rng(o.seed, level))?;
        let room = headroom - tray.deck_m;
        let mut r = stream(o.seed, u64::from_le_bytes(*b"BakeSell") + level as u64);
        let recipe = match r.random_range(0..4) {
            1 => Bake::Roll,
            2 => Bake::Bagel,
            3 if room >= 0.053 => Bake::Muffin,
            _ => Bake::Biscuit,
        };
        let radius: f32 = match recipe {
            Bake::Biscuit => r.random_range(0.028..0.047),
            Bake::Roll => r.random_range(0.036..0.054),
            Bake::Bagel => r.random_range(0.043..0.060),
            Bake::Muffin => r.random_range(0.033..0.050),
        };
        let rise = match recipe {
            Bake::Biscuit => 0.54,
            Bake::Roll => r.random_range(0.78..1.20),
            Bake::Bagel => r.random_range(0.52..0.86),
            Bake::Muffin => r.random_range(1.08..1.55),
        };
        let usable = tray.size_m - Vec2::splat(0.016);
        let cells = (usable / (2.12 * radius + 0.004))
            .floor()
            .as_uvec2()
            .max(UVec2::ONE);
        let arranged_in_rows = r.random_bool(0.58);
        let inventory = if r.random_bool(0.07) {
            0
        } else {
            r.random_range(1..=8)
        };
        let mut portions = Vec::<Portion>::with_capacity(inventory);
        // Traverse a shuffled subset of row positions. Sold portions can come from any row.
        let mut slots: Vec<_> = if arranged_in_rows {
            (0..cells.x * cells.y).collect()
        } else {
            Vec::new()
        };
        slots.shuffle(&mut r);
        for i in 0..inventory {
            let rr = radius * r.random_range(0.90..1.06);
            let free = usable * 0.5 - rr;
            if free.min_element() <= 0. {
                continue;
            }
            let fits = |q: Vec2| {
                portions
                    .iter()
                    .all(|p| p.center_m.distance_squared(q) >= (rr + p.radius_m + 0.004).powi(2))
            };
            let center = if arranged_in_rows {
                let Some(&n) = slots.get(i) else {
                    break;
                };
                let cell = usable / cells.as_vec2();
                let home = (Vec2::new((n % cells.x) as f32, (n / cells.x) as f32) + 0.5) * cell
                    - usable * 0.5;
                let q = home
                    + (cell * 0.5 - rr).max(Vec2::ZERO)
                        * Vec2::new(r.random_range(-0.87..0.87), r.random_range(-0.87..0.87));
                fits(q).then_some(q)
            } else {
                (0..12)
                    .map(|_| free * Vec2::new(r.random_range(-1.0..1.0), r.random_range(-1.0..1.0)))
                    .find(|&q| fits(q))
            };
            let Some(center) = center else {
                continue;
            };
            portions.push(Portion {
                center_m: center,
                yaw_rad: r.random_range(-PI..PI),
                radius_m: rr,
                height_m: (rr * rise * r.random_range(0.91..1.08)).min(room - 0.001),
                seed: r.random(),
            });
        }
        Some(Self {
            level,
            tray,
            shelf_y_m,
            recipe,
            arranged_in_rows,
            portions,
        })
    }

    pub(super) fn build(&self, a: &mut Assembly) {
        let mut batch = Assembly::default();
        self.tray.build(&mut batch);
        for p in &self.portions {
            let at = Transform::from_xyz(p.center_m.x, self.tray.deck_m, p.center_m.y)
                .with_rotation(Quat::from_rotation_y(p.yaw_rad));
            p.build(&mut batch, self.recipe, at);
        }
        append_at(
            a,
            batch,
            Transform::from_xyz(self.tray.center_m.x, self.shelf_y_m, self.tray.center_m.y)
                .with_rotation(Quat::from_rotation_y(self.tray.yaw_rad)),
        );
    }
}

#[cfg(test)]
mod tests;
