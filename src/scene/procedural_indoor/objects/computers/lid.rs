//! The display, front surround and camera window share a single manufactured lid.
use super::*;
use crate::scene::procedural_indoor::materials::input_housing::LidStock;

mod camera;
#[cfg(test)]
mod tests;

pub(crate) const INTRODUCED: u32 = 262;

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Bezel {
    Molded,
    BondedGlass,
    CaseMatched,
}

impl Bezel {
    fn stock(self) -> Option<LidStock> {
        match self {
            Self::Molded => Some(LidStock::Molded),
            Self::BondedGlass => Some(LidStock::Bonded),
            Self::CaseMatched => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CameraWindow {
    LensOnly,
    WideWindow,
    ShutterOpen,
    ShutterClosed,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct CameraModule {
    pub window: CameraWindow,
    /// Relative to the lid centreline, positive toward the operator's right.
    pub offset_x_m: f32,
    pub width_m: f32,
    pub height_m: f32,
    pub aperture_radius_m: f32,
    pub aperture_offset_x_m: f32,
}

#[derive(Debug, Clone, Copy, serde::Serialize)]
pub struct LidFace {
    pub bezel: Bezel,
    pub side_bezel_m: f32,
    pub top_bezel_m: f32,
    pub chin_m: f32,
    pub active_size_m: Vec2,
    pub screen_recess_m: f32,
    pub camera: Option<CameraModule>,
}

impl LaptopProgram {
    /// Resolved front assembly. Earlier manifests replay the uniform bezel.
    pub fn lid_face(&self, o: &IndoorObject, version: u32) -> Option<LidFace> {
        (o.kind == ObjectKind::Laptop && version >= INTRODUCED)
            .then(|| LidFace::fit(o, self, Deck::fit(o, self)))
    }
}

impl LidFace {
    fn fit(o: &IndoorObject, p: &LaptopProgram, deck: Deck) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"LidGlass"));
        let bezel = match r.random_range(0..100) {
            0..=41 => Bezel::BondedGlass,
            42..=57 => Bezel::CaseMatched,
            _ => Bezel::Molded,
        };
        let side = (p.bezel_m * r.random_range(0.48..1.05))
            .max(0.0028)
            .min(o.size.x * 0.08);
        let top = (side * r.random_range(0.64..1.30) + r.random_range(0.0014..0.0058))
            .max(0.0048)
            .min(deck.height * 0.15);
        let chin = (side * r.random_range(0.72..1.46) + r.random_range(0.001..0.013))
            .min(deck.height * 0.19);
        let edge = p.corner_m.min(0.0016).min(p.lid_m * 0.45);
        let land = side.min(top).min(chin);
        let screen_recess_m = match bezel {
            Bezel::BondedGlass => 0.,
            Bezel::Molded | Bezel::CaseMatched => r
                .random_range(0.00035_f32..0.00165)
                .min(p.lid_m * 0.26)
                .min((land - edge - 0.0004).max(0.) / 0.68),
        };
        Self {
            bezel,
            side_bezel_m: side,
            top_bezel_m: top,
            chin_m: chin,
            active_size_m: Vec2::new(o.size.x - 2. * side, deck.height - top - chin),
            screen_recess_m,
            camera: CameraModule::sample(o, top, top - edge - screen_recess_m * 0.68),
        }
    }

    pub(crate) fn stocks(self) -> impl Iterator<Item = LidStock> {
        self.bezel.stock().into_iter().chain(
            self.camera
                .into_iter()
                .flat_map(|cam| [cam.surround(), cam.aperture_stock()]),
        )
    }
}

pub(super) fn build(
    a: &mut Assembly,
    o: &IndoorObject,
    program: &LaptopProgram,
    housing: &Housing,
    deck: Deck,
    lid: Transform,
) {
    let face = LidFace::fit(o, program, deck);
    let half = Vec2::new(o.size.x, deck.height) * 0.5;
    // XZ slab charts become XY lid charts, with increasing V down the image.
    let tf =
        lid * Transform::from_xyz(0., half.y, 0.).with_rotation(Quat::from_rotation_x(FRAC_PI_2));
    let outer = palmrest::rounded_outline(half, program.corner_m, Vec2::ZERO);
    let profile = Geometry::slab_profile(&outer, program.lid_m, program.corner_m.min(0.0016));
    let z = program.lid_m * 0.5;
    let case_label = housing.case_label(o);
    let shell = a.part(housing.surface, &case_label);
    shell.slab_cap(&outer, profile.1, Vec2::ZERO, -z, &tf, |p| p);
    shell.slab_edge(&outer, program.lid_m, profile, &tf);

    let center = Vec2::Y * ((face.top_bezel_m - face.chin_m) * 0.5);
    // A narrow border must follow the outside corner. A square optical
    // corner can otherwise cut across the rolled edge of a thin bezel.
    let land = face.side_bezel_m.min(face.top_bezel_m).min(face.chin_m);
    let free_land = (land - profile.0 - face.screen_recess_m * 0.68).max(0.);
    let radius = (program.corner_m - free_land).max(program.corner_m.min(land * 0.52));
    let glass = palmrest::rounded_outline(face.active_size_m * 0.5, radius, center);
    let lip = if face.screen_recess_m > 0. {
        let bevel = face.screen_recess_m * 0.68;
        palmrest::rounded_outline(face.active_size_m * 0.5 + bevel, radius + bevel, center)
    } else {
        glass
    };
    let surround = face
        .bezel
        .stock()
        .map_or((housing.surface, case_label.as_str()), |stock| {
            (Surface::Plastic, stock.label())
        });
    let at = |p: Vec2, z| tf.transform_point(Vec3::new(p.x, z, p.y));
    let forward = tf.rotation * Vec3::Y;
    let image_z = z - face.screen_recess_m;
    let panel = |i: usize, j: usize| [outer[i] * profile.1, outer[j] * profile.1, lip[j], lip[i]];
    let bezel = a.part(surround.0, surround.1);
    // Straight segment across the top of the rounded outline.
    const HEADER: usize = 8;
    for i in 0..outer.len() {
        let j = (i + 1) % outer.len();
        if i != HEADER || face.camera.is_none() {
            bezel.metric_quad(panel(i, j).map(|p| at(p, z)), forward);
        }
        if face.screen_recess_m > 0. {
            bezel.metric_quad(
                [
                    at(lip[i], z),
                    at(lip[j], z),
                    at(glass[j], image_z),
                    at(glass[i], image_z),
                ],
                forward,
            );
        }
    }
    if let Some(camera) = face.camera {
        camera.build(a, panel(HEADER, HEADER + 1), &tf, z, surround);
    }
    a.part(Surface::Screen, o.kind.class_name())
        .slab_cap(&glass, 1., center, image_z, &tf, |p| p - center);
}
