//! Yoga practice props. Dimensions include the sewn handle when fitted.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::{ProfileJoints, RevolvedOutline},
    materials::garment_stock,
    objects::bags::webbing_strap,
};
#[cfg(test)]
mod tests;

pub(crate) const VERSION: u32 = 317;
const LABEL: &str = "fitness_equipment";

pub(crate) fn dimensions(kind: ObjectKind, seed: u64) -> Vec3 {
    let mut r = stream(seed, u64::from_le_bytes(*b"YogaSize"));
    match kind {
        ObjectKind::YogaBlock => {
            let (long, high, wide) = match r.random_range(0..10) {
                0..=1 => (0.153, 0.101, 0.115), // compact
                2..=4 => (0.222, 0.072, 0.116), // slim
                _ => (0.230, 0.101, 0.151),
            };
            Vec3::new(long, high, wide) * r.random_range(0.94..1.065)
        }
        ObjectKind::YogaBolster => {
            let (width, height, length) = match r.random_range(0..10) {
                0..=3 => (0.242, 0.218, 0.675), // round, compressed under its own weight
                4..=5 => (0.175, 0.105, 0.610), // lean
                _ => (0.296, 0.134, 0.700),
            };
            Vec3::new(width, height, length) * r.random_range(0.91..1.075)
        }
        _ => unreachable!("practice prop"),
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Block {
    pub on_edge: bool,
    pub facing: Stock,
    pub inlay: Option<Stock>,
    pub corner_m: f32,
    pub bevel_m: f32,
    pub stripe_fraction: f32,
}

impl Block {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut lot = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"YogaFoam"),
        );
        let facing = match lot.random_range(0..10) {
            0..=2 => Stock::NaturalCork,
            3 => Stock::ToastedCork,
            _ => Stock::Pebbled,
        };
        Self {
            // All three products have a broader base than thickness. The
            // placed envelope records which face is down, including in replay.
            on_edge: o.size.y > o.size.z,
            facing,
            inlay: (facing == Stock::Pebbled && lot.random_bool(0.36)).then_some(Stock::Ribbed),
            corner_m: lot
                .random_range(0.008..0.024_f32)
                .min(o.size.min_element() * 0.19),
            bevel_m: lot
                .random_range(0.004..0.012_f32)
                .min(o.size.min_element() * 0.10),
            stripe_fraction: lot.random_range(0.11..0.27),
        }
    }

    fn build(self, a: &mut Assembly, o: &IndoorObject) {
        let s = if self.on_edge { o.size.xzy() } else { o.size };
        let contour = rounded_outline(s.xz() * 0.5, self.corner_m);
        let key = self.facing.equipment_key();
        let body = a.part(key.0, key.1);
        let mut tf = Transform::from_xyz(0., o.size.y * 0.5, 0.);
        if self.on_edge {
            tf.rotation = Quat::from_rotation_x(FRAC_PI_2);
        }
        let Some(inlay) = self.inlay else {
            body.profile_slab(&contour, s.y, self.bevel_m, tf);
            return;
        };
        // The laminate is one continuous shell. Each side interval has one
        // owner, including at grazing angles and in depth/normal captures.
        let profile = Geometry::slab_profile(&contour, s.y, self.bevel_m);
        for sign in [-1., 1.] {
            body.slab_cap(
                &contour,
                profile.1,
                Vec2::ZERO,
                sign * s.y * 0.5,
                &tf,
                |p| p,
            );
        }
        body.slab_edge_sections(
            &contour,
            s.y,
            profile,
            &tf,
            |_, along, rings| rings.map(|(y, _)| Vec2::new(along, y)),
            |_| true,
        );
        let half = s.y * 0.5;
        let band = half * self.stripe_fraction;
        let inlay = inlay.equipment_key();
        for (low, high, stock) in [
            (-half + profile.0, -band, key),
            (-band, band, inlay),
            (band, half - profile.0, key),
        ] {
            let g = a.part(stock.0, stock.1);
            let mut along = 0.;
            for i in 0..contour.len() {
                let p = contour[i];
                let q = contour[(i + 1) % contour.len()];
                let end = along + p.distance(q);
                let normal = tf.rotation * Vec3::new(q.y - p.y, 0., p.x - q.x).normalize();
                let at = |v: Vec2, y| tf.transform_point(Vec3::new(v.x, y, v.y));
                g.vertex_quad(
                    [at(p, low), at(p, high), at(q, high), at(q, low)],
                    [normal; 4],
                    [[along, low], [along, high], [end, high], [end, low]],
                );
                along = end;
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize)]
pub struct Bolster {
    pub section_exponent: f32,
    pub cover: Surface,
    pub dye_slot: usize,
    pub gusset_slot: usize,
    pub handle_m: f32,
    pub handle_width_m: f32,
}

impl Bolster {
    pub fn for_object(o: &IndoorObject) -> Self {
        let mut lot = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"YogaSewn"),
        );
        let mut cut = stream(o.seed, u64::from_le_bytes(*b"CarryTab"));
        let dye_slot = garment_stock::sample_slot(&mut lot);
        Self {
            section_exponent: if o.size.y >= o.size.x * 0.74 {
                lot.random_range(2.0..2.65)
            } else {
                lot.random_range(3.6..6.5)
            },
            cover: if lot.random_bool(0.63) {
                Surface::Fabric
            } else {
                Surface::FabricAlt
            },
            dye_slot,
            gusset_slot: if lot.random_bool(0.34) {
                garment_stock::contrasting_slot(dye_slot, &mut lot)
            } else {
                dye_slot
            },
            handle_m: if cut.random_bool(0.82) {
                cut.random_range(0.025..0.065)
            } else {
                0.
            },
            handle_width_m: cut.random_range(0.015..0.030),
        }
    }

    fn build(self, a: &mut Assembly, s: Vec3) {
        let start = -s.z * 0.5;
        let end = s.z * 0.5 - self.handle_m;
        let seam = (s.y * 0.060).min(0.013);
        let r = s.x * 0.5;
        let section =
            RevolvedOutline::superellipse(24, self.section_exponent, Vec2::new(1., s.y / s.x));
        let profile = [
            (0., start),
            (0.83 * r, start),
            (0.965 * r, start + seam),
            (0.950 * r, start + 1.4 * seam),
            (r, start + s.z * 0.17),
            (r, end - s.z * 0.17),
            (0.950 * r, end - 1.4 * seam),
            (0.965 * r, end - seam),
            (0.83 * r, end),
            (0., end),
        ];
        let tf =
            Transform::from_xyz(0., s.y * 0.5, 0.).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
        let gusset = format!("{LABEL}#finish{}#uv_local", self.gusset_slot);
        let cover = format!("{LABEL}#finish{}#uv_local", self.dye_slot);
        for (range, label) in [(0..2, &gusset), (2..7, &cover), (7..9, &gusset)] {
            section.sweep_range(
                a.part(self.cover, label),
                &profile,
                range,
                ProfileJoints::Smooth,
                tf,
            );
        }
        if self.handle_m > 0. {
            // Ends enter the sewn end panel. The loop is hollow and has
            // physical tape thickness and metre-scaled UVs on both faces.
            let y = s.y * 0.51;
            let span = s.x * 0.22;
            let thickness = 0.0028;
            let tip = s.z * 0.5 - thickness * 0.5;
            let points = [
                Vec3::new(-span, y, end - 0.002),
                Vec3::new(-span * 0.86, y, tip - self.handle_m * 0.13),
                Vec3::new(0., y, tip),
                Vec3::new(span * 0.86, y, tip - self.handle_m * 0.13),
                Vec3::new(span, y, end - 0.002),
            ];
            // The wide dimension of the tape is vertical at the end gusset.
            webbing_strap(
                a.part(self.cover, &gusset),
                &points,
                thickness,
                self.handle_width_m,
                Vec3::Y,
                true,
            );
        }
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    if o.kind == ObjectKind::YogaBlock {
        Block::for_object(o).build(a, o);
    } else {
        Bolster::for_object(o).build(a, o.size);
    }
}
