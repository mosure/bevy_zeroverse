//! Finish-side construction inside the structural wall section. The room-side
//! datum stays fixed so wall-mounted objects, glazing and clearances still fit.
use super::*;
use crate::scene::procedural_indoor::{
    layout::{stream, IndoorLayout},
    materials::{variants, Surface},
};
use rand::Rng;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LiningStock {
    Veneer,
    AcousticFabric,
    CeramicSlab,
}

impl LiningStock {
    pub(crate) fn surface(self) -> Surface {
        match self {
            Self::Veneer => Surface::Wood,
            Self::AcousticFabric => Surface::FabricAlt,
            Self::CeramicSlab => Surface::Ceramic,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WallLining {
    pub stock: LiningStock,
    pub height_m: f32,
    pub module_width_m: f32,
    /// Fraction of a stock module at the wall origin.
    pub phase: f32,
    pub joint_width_m: f32,
    pub joint_depth_m: f32,
    pub finish_slot: usize,
}

impl WallLining {
    pub(super) fn valid(&self) -> bool {
        (0.55..=2.25).contains(&self.height_m)
            && (0.30..=1.60).contains(&self.module_width_m)
            && (0.0..1.0).contains(&self.phase)
            && (0.002..=0.02).contains(&self.joint_width_m)
            && (0.001..=0.012).contains(&self.joint_depth_m)
            && self.finish_slot < variants::COUNT
    }
}

pub(super) fn install(e: &mut EnvelopeProgram, scene: &IndoorManifest) {
    use IndoorLayout::*;
    use LiningStock::*;
    let mut r = stream(scene.seed, 0x57414c4c4c4f5453);
    if !r.random_bool(0.58) {
        return;
    }
    let stock = match (scene.layout, r.random_range(0..10)) {
        (Kitchen | Kitchenette | Cafeteria | Breakroom, 0..=7) => CeramicSlab,
        (Training | LectureHall | Classroom | OpenOffice, 0..=6) => AcousticFabric,
        (Lounge | Conference | Reception | Library, 0..=6) => Veneer,
        (_, 0..=3) => AcousticFabric,
        (_, 4..=6) => CeramicSlab,
        _ => Veneer,
    };
    let (height, pitch, width, depth) = match stock {
        Veneer => (0.74..1.68, 0.36..1.22, 0.004..0.013, 0.002..0.008),
        AcousticFabric => (1.06..2.15, 0.44..1.36, 0.005..0.018, 0.003..0.010),
        CeramicSlab => (0.82..1.72, 0.43..1.54, 0.003..0.009, 0.0015..0.004),
    };
    let series = WallLining {
        stock,
        height_m: r.random_range(height),
        module_width_m: r.random_range(pitch),
        phase: 0.,
        joint_width_m: r.random_range(width),
        joint_depth_m: r.random_range(depth),
        finish_slot: r.random_range(0..variants::COUNT),
    };
    let mut candidates = Vec::new();
    for (index, wall) in e.walls.iter().enumerate() {
        let a = e.footprint[wall.edge];
        let b = e.footprint[(wall.edge + 1) % e.footprint.len()];
        let span = a.distance(b);
        let area = span * series.height_m;
        let covered = wall.facade.as_ref().map_or(0., |f| {
            f.openings
                .iter()
                .map(|o| {
                    (o.max.x - o.min.x) * (o.max.y.min(series.height_m) - o.min.y.max(0.)).max(0.)
                })
                .sum::<f32>()
        });
        let solid = area - covered;
        if span > 2. && solid > area * 0.38 && solid > 2.2 {
            candidates.push((index, solid.sqrt() * r.random_range(0.22_f32..1.)));
        }
    }
    candidates.sort_unstable_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    let count = if r.random_bool(0.35) { 2 } else { 1 };
    for (i, _) in candidates.into_iter().take(count) {
        e.walls[i].lining = Some(WallLining {
            phase: r.random(),
            ..series
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archived_plain_walls_replay_and_invalid_finish_geometry_is_rejected() {
        let plain: EnvelopeWall = serde_json::from_str(r#"{"edge":2,"facade":null}"#).unwrap();
        assert!(plain.lining.is_none());
        assert!(serde_json::to_value(plain).unwrap().get("lining").is_none());
        let mut sample = WallLining {
            stock: LiningStock::CeramicSlab,
            height_m: 1.25,
            module_width_m: 0.8,
            phase: 0.28,
            joint_width_m: 0.008,
            joint_depth_m: 0.003,
            finish_slot: 1,
        };
        assert!(sample.valid());
        sample.module_width_m = 0.;
        assert!(!sample.valid());
        sample.module_width_m = 0.8;
        sample.joint_depth_m = f32::NAN;
        assert!(!sample.valid());
    }
}
