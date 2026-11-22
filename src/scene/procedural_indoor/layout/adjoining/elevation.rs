//! Punched windows and continuous skirting on the outside of the adjoining bay.
use super::*;
use crate::scene::procedural_indoor::{
    architecture::{
        facade::{reveal::RevealProgram, Facade, FacadeSide, Shade, WindowJoinery, WindowOpening},
        trim::TrimProgram,
    },
    envelope::polygon,
    materials::{glazing::GlazingProgram, Surface},
};
use rand::seq::IndexedRandom;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdjoiningElevation {
    /// Continue the building's installed base moulding along the bay perimeter.
    pub continue_skirting: bool,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub windows: Option<AdjoiningWindows>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AdjoiningWindows {
    /// Index in AdjoiningSpace::footprint. Edge zero is the interior partition.
    /// Aperture coordinates are metres from this edge's midpoint in its wall frame.
    pub edge: usize,
    pub facade: Facade,
}

impl AdjoiningWindows {
    pub(crate) fn endpoints(&self, scene: &IndoorManifest) -> Option<(Vec2, Vec2)> {
        let outline = scene.adjoining_space().footprint(scene.room_size);
        (self.edge > 0 && self.edge < outline.len())
            .then(|| (outline[self.edge], outline[(self.edge + 1) % outline.len()]))
    }

    pub(crate) fn validate(&self, scene: &IndoorManifest) -> Result<(), String> {
        let (a, b) = self
            .endpoints(scene)
            .ok_or("invalid adjoining window wall")?;
        let envelope = scene
            .envelope
            .as_ref()
            .ok_or("adjoining windows require an envelope")?;
        if scene.adjoining.is_none()
            || (self.facade.wall_thickness_m() - envelope.wall_thickness()).abs() > 1e-5
            || self.facade.recess + self.facade.frame_depth * 0.5 > envelope.wall_thickness()
        {
            return Err("adjoining glazing does not fit its wall section".into());
        }
        self.facade.validate_span(
            a.distance(b),
            scene.ceiling_height(a).min(scene.ceiling_height(b)),
        )
    }
}

impl AdjoiningElevation {
    pub(in crate::scene::procedural_indoor::layout) fn sample(scene: &IndoorManifest) -> Self {
        let continue_skirting = TrimProgram::for_scene(scene)
            .and_then(|t| t.base.as_ref())
            .is_some()
            && stream(scene.seed, u64::from_le_bytes(*b"baytrim ")).random_bool(0.84);
        let mut rng = stream(scene.seed, u64::from_le_bytes(*b"baysash "));
        // Interior circulation can be entirely enclosed. This draw is independent
        // of the furniture, primary glazing, camera rig and time of day.
        let windows = rng
            .random_bool(0.58)
            .then(|| Self::windows(scene, &mut rng))
            .flatten();
        Self {
            continue_skirting,
            windows,
        }
    }

    fn windows(scene: &IndoorManifest, rng: &mut ChaCha8Rng) -> Option<AdjoiningWindows> {
        let envelope = scene.envelope.as_ref()?;
        let thickness = envelope.wall_thickness();
        let outline = scene.adjoining_space().footprint(scene.room_size);
        // Select by usable frontage so a short return does not receive an entire
        // wall's worth of glazing. Reserve both perpendicular wall sections.
        let edges: Vec<_> = polygon::edges(&outline)
            .enumerate()
            .skip(1)
            .filter_map(|(edge, (a, b))| {
                let available = a.distance(b) - 2. * (thickness + 0.15);
                (available >= 0.88).then_some(((edge, a, b), available))
            })
            .collect();
        let &((edge, a, b), available) = edges.choose_weighted(rng, |e| e.1).ok()?;
        let height = scene.ceiling_height(a).min(scene.ceiling_height(b));
        let width = (available * rng.random_range(0.34..0.94))
            .max(0.88)
            .min(rng.random_range(4.8..8.2));
        let offset = (available - width) * rng.random_range(-0.5..0.5);
        let count = (width / rng.random_range(1.45..3.15)).round().clamp(1., 3.) as usize;
        let pier = rng.random_range(0.16..0.43);
        let pitch = (width + pier) / count as f32;
        let clear = pitch - pier;
        let head = height - rng.random_range(0.16..0.47);
        let sill = if rng.random_bool(0.32) {
            head - rng.random_range(0.48..0.96)
        } else {
            rng.random_range(0.56..1.21)
        };
        let identity = rng.random();
        let frame_width = rng.random_range(0.030..0.067);
        let frame_depth = rng.random_range(0.046..0.093);
        let (reveal, recess) = RevealProgram::sample(identity, thickness, frame_depth, frame_width);
        let mut joinery = WindowJoinery::sample(identity);
        joinery.reveal = Some(reveal);
        let columns = (clear / rng.random_range(0.68..1.58)).round().clamp(1., 4.) as u32;
        let mut transom = if head - sill > 1.3 && rng.random_bool(0.34) {
            rng.random_range(0.57..0.78)
        } else {
            0.
        };
        let mut transoms = Vec::new();
        let clear_height = head - sill - 2. * frame_width;
        if clear_height > 3.25 {
            // Tall public volumes use stacked glazing units. Keep their rails
            // on a shared datum across the installation, as in a curtain wall.
            let mut rails = stream(identity, u64::from_le_bytes(*b"bayrails"));
            let rows = (clear_height / rails.random_range(2.25..3.05))
                .ceil()
                .clamp(2., 4.) as usize;
            let mut levels =
                (1..rows).map(|i| (i as f32 + rails.random_range(-0.09..0.09)) / rows as f32);
            transom = levels.next().unwrap();
            transoms.extend(levels);
        }
        let openings = (0..count)
            .map(|i| {
                let left = offset - width * 0.5 + i as f32 * pitch;
                WindowOpening {
                    min: Vec2::new(left, sill),
                    max: Vec2::new(left + clear, head),
                    columns,
                    transom,
                    transoms: transoms.clone(),
                    operable: false,
                    glazing: Some(GlazingProgram::sample(identity, i, false)),
                }
            })
            .collect();
        Some(AdjoiningWindows {
            edge,
            facade: Facade {
                // As for polygonal primary facades, Rear is the canonical local
                // frame. The recorded polygon edge supplies the world orientation.
                side: FacadeSide::Rear,
                openings,
                joinery: Some(joinery),
                frame: [Surface::Metal, Surface::WoodEdge, Surface::Plastic]
                    [rng.random_range(0..3)],
                frame_width,
                frame_depth,
                recess,
                sill_projection: rng.random_range(0.022..0.105),
                shade: Shade::None,
                shade_coverage: 0.,
                shade_tilt: 0.,
                treatment: None,
            },
        })
    }
}
