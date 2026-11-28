//! Table glasses have a solid foot and an open bowl. A manufacturing lot controls
//! the glass itself, while a separate draw per serving determines the water level.
use super::*;
use crate::scene::procedural_indoor::materials::glass::tableware::{self, Tint};

#[derive(Clone, Copy, Debug, Eq, PartialEq, serde::Serialize)]
pub(crate) enum Form {
    Rocks,
    Highball,
    Bistro,
    Stemless,
    Goblet,
}
impl Form {
    pub(crate) const ALL: [Self; 5] = [
        Self::Rocks,
        Self::Highball,
        Self::Bistro,
        Self::Stemless,
        Self::Goblet,
    ];
}

#[derive(Clone, Copy)]
pub(crate) struct Stock {
    form: Form,
    lot: u64,
    pub(crate) size: Vec3,
}
impl Stock {
    pub(crate) fn new(form: Form, lot: u64) -> Self {
        let mut r = stream(lot, u64::from_le_bytes(*b"GlassSku"));
        let (diameter, height) = match form {
            Form::Rocks => (0.072..0.096, 0.070..0.103),
            Form::Highball => (0.059..0.080, 0.121..0.167),
            Form::Bistro => (0.065..0.092, 0.077..0.133),
            Form::Stemless => (0.072..0.098, 0.094..0.133),
            Form::Goblet => (0.069..0.096, 0.148..0.224),
        };
        let d = r.random_range(diameter);
        Self {
            form,
            lot,
            size: Vec3::new(d, r.random_range(height), d),
        }
    }
    pub(crate) fn catalog(seed: u64) -> Self {
        Self::new(Form::ALL[(seed % 5) as usize], seed)
    }
    pub(crate) fn apply(self, o: &mut IndoorObject) {
        o.variant = self.form as u32;
        o.finish_seed = Some(self.lot);
    }
}

#[derive(Debug, serde::Serialize)]
pub(crate) struct Glass {
    pub(crate) form: Form,
    pub(crate) tint: Tint,
    sides: u32,
    pub(crate) wall_m: f32,
    /// Height of the bowl floor above the table, including a stem when present.
    floor_m: f32,
    heel_radius: f32,
    mouth_radius: f32,
    shoulder_height: f32,
    stem: Option<Stem>,
    /// Relative to usable bowl depth, not to overall height or nominal volume.
    pub(crate) water_depth_fraction: Option<f32>,
}

#[derive(Debug, serde::Serialize)]
struct Stem {
    radius_m: f32,
    foot_radius: f32,
}

impl Glass {
    pub(crate) fn for_object(o: &IndoorObject) -> Self {
        let mut stock = stream(
            o.finish_seed.unwrap_or(o.seed),
            u64::from_le_bytes(*b"GlassCut"),
        );
        let mut serving = stream(o.seed, u64::from_le_bytes(*b"GlassH2O"));
        let form = Form::ALL[o.variant as usize % Form::ALL.len()];
        let h = o.size.y;
        let radius = o.size.x.min(o.size.z) * 0.5;
        let (heel, mouth, floor, shoulder) = match form {
            Form::Rocks => (0.80..0.98, 0.96..1., 0.075..0.18, 0.55..0.83),
            Form::Highball => (0.68..0.90, 0.97..1., 0.038..0.092, 0.65..0.86),
            Form::Bistro => (0.53..0.73, 0.95..1., 0.063..0.13, 0.57..0.76),
            Form::Stemless => (0.39..0.58, 0.69..0.88, 0.035..0.08, 0.33..0.53),
            Form::Goblet => (0.16..0.25, 0.70..0.91, 0.32..0.51, 0.32..0.55),
        };
        let stem = (form == Form::Goblet).then(|| Stem {
            radius_m: stock.random_range(0.0029..0.0047_f32).min(radius * 0.12),
            foot_radius: stock.random_range(0.70..0.95),
        });
        let wall_m: f32 = match form {
            Form::Stemless | Form::Goblet => stock.random_range(0.00095..0.00185),
            _ => stock.random_range(0.00145..0.0031),
        };
        Self {
            form,
            stem,
            tint: Tint::sample(&mut stock),
            sides: if form == Form::Bistro {
                [8, 10, 12][stock.random_range(0..3)]
            } else {
                32
            },
            wall_m: wall_m.min(radius * 0.095).min(h * 0.025),
            floor_m: stock.random_range(floor) * h,
            heel_radius: stock.random_range(heel) * radius,
            mouth_radius: stock.random_range(mouth) * radius,
            shoulder_height: stock.random_range(shoulder),
            water_depth_fraction: serving
                .random_bool(0.69)
                .then(|| serving.random_range(0.11..0.89)),
        }
    }

    fn build(&self, a: &mut Assembly, size: Vec3) {
        let d = size.x.min(size.z);
        let radius = d * 0.5;
        let height = size.y;
        let tf = Transform::from_scale(Vec3::new(size.x / d, 1., size.z / d));
        let outline = super::outline(self.sides);
        let floor = self.floor_m;
        let heel = self.heel_radius;
        let lip = self.mouth_radius;
        let gauge = self.wall_m;
        let mut profile = Vec::with_capacity(28);
        profile.push((0., 0.));
        if let Some(stem) = &self.stem {
            let foot = stem.foot_radius * radius;
            let shaft = stem.radius_m;
            profile.extend([
                (foot * 0.95, 0.),
                (foot, 0.0014),
                (foot * 0.98, 0.0026),
                (foot * 0.42, floor * 0.069),
                (shaft * 1.35, floor * 0.12),
                (shaft, floor * 0.19),
                (shaft, floor * 0.78),
                (shaft * 1.20, floor * 0.88),
            ]);
        } else {
            profile.extend([(heel * 0.86, 0.), (heel * 0.95, floor * 0.15)]);
        }
        let depth = height - gauge * 0.4 - floor;
        let shoulder = self.shoulder_height;
        let bulb = matches!(self.form, Form::Stemless | Form::Goblet);
        let body = [
            (heel, floor),
            (
                heel + (radius - heel) * if bulb { 0.62 } else { 0.28 },
                floor + depth * shoulder * 0.30,
            ),
            (
                heel + (radius - heel) * if bulb { 0.91 } else { 0.67 },
                floor + depth * shoulder * 0.66,
            ),
            (radius, floor + depth * shoulder),
            (
                radius + (lip - radius) * 0.28,
                floor + depth * (shoulder + (1. - shoulder) * 0.49),
            ),
            (lip, height - gauge * 0.4),
        ];
        profile.extend(body);
        profile.extend([(lip - gauge * 0.30, height), (lip - gauge * 0.72, height)]);
        let level = self.water_depth_fraction.map(|f| floor + depth * f);
        let mut contact_edge = None;
        for i in (0..body.len()).rev() {
            let (r, y) = body[i];
            profile.push((r - gauge, y));
            if let (Some(level), Some(j)) = (level, i.checked_sub(1)) {
                let (below, low_y) = body[j];
                if low_y <= level && level < y {
                    let t = (level - low_y) / (y - low_y);
                    let joint = (below + t * (r - below) - gauge, level);
                    // Use the very same vertex on the wall and water, even on flat facets.
                    if (y - level) < 0.00006 {
                        contact_edge = Some(profile.len() - 1);
                    } else if (level - low_y) < 0.00006 {
                        contact_edge = Some(profile.len());
                    } else {
                        profile.push(joint);
                        contact_edge = Some(profile.len() - 1);
                    }
                }
            }
        }
        profile.push((0., floor));
        let dry = a.part(Surface::ContainerGlass, self.tint.label());
        let end = profile.len() - 1;
        if let Some(joint) = contact_edge {
            outline.sweep_range(dry, &profile, 0..joint, ProfileJoints::Smooth, tf);
            outline.sweep_range(
                a.part(Surface::ContainerGlass, tableware::IMMERSED),
                &profile,
                joint..end,
                ProfileJoints::Smooth,
                tf,
            );
            let (r, y) = profile[joint];
            let dip = 0.00065_f32.min(gauge * 0.55);
            outline.sweep(
                a.part(Surface::Liquid, "drink_container"),
                &[(r, y), (r - dip * 2., y - dip), (0., y - dip)],
                ProfileJoints::Smooth,
                tf,
            );
        } else {
            outline.sweep(dry, &profile, ProfileJoints::Smooth, tf);
        }
    }
}

pub(crate) fn build(a: &mut Assembly, o: &IndoorObject) {
    Glass::for_object(o).build(a, o.size);
}

#[cfg(test)]
mod tests;
