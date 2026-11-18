//! One room-wide partition installation. The neighboring volume uses actual
//! plaster piers/spandrels instead of always being a floor-to-ceiling glass wall.
pub use super::shared_openings::SharedOpenings;
use super::{facade::*, *};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InteriorGlazing {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub shared_openings: Option<SharedOpenings>,
    pub sill_m: f32,
    pub header_m: f32,
    pub pier_fraction: f32,
    pub opening_pitch_m: f32,
    pub mullion_pitch_m: f32,
    pub transom_fraction: f32,
    pub frame_width_m: f32,
    pub frame_depth_m: f32,
    pub frame: Surface,
    pub joinery: WindowJoinery,
}

impl InteriorGlazing {
    pub fn sample(seed: u64) -> Self {
        let mut r = stream(seed, 3163);
        Self {
            shared_openings: Some(SharedOpenings::sample(seed)),
            sill_m: 0.028 + 1.25 * r.random_range(0.0_f32..1.).powi(2),
            header_m: r.random_range(0.06..0.60),
            pier_fraction: r.random_range(0.025..0.28),
            opening_pitch_m: r.random_range(1.8..6.),
            mullion_pitch_m: r.random_range(0.65..2.6),
            transom_fraction: if r.random_bool(0.55) {
                r.random_range(0.24..0.82)
            } else {
                0.
            },
            frame_width_m: r.random_range(0.024..0.072),
            frame_depth_m: r.random_range(0.045..0.10),
            frame: [Surface::WoodEdge, Surface::Metal, Surface::Plastic][r.random_range(0..3)],
            joinery: WindowJoinery::sample(seed.wrapping_add(3163)),
        }
    }
    pub fn for_scene(scene: &IndoorManifest) -> Option<&Self> {
        scene
            .program
            .as_ref()?
            .finishes
            .as_ref()?
            .interior_glazing
            .as_ref()
    }
    pub fn validate(&self) -> Result<(), String> {
        self.joinery.validate()?;
        if self.shared_openings.is_some_and(|s| !s.valid())
            || !(0.02..=1.3).contains(&self.sill_m)
            || !(0.05..=0.65).contains(&self.header_m)
            || !(0.02..=0.30).contains(&self.pier_fraction)
            || !(1.5..=6.5).contains(&self.opening_pitch_m)
            || !(0.6..=2.8).contains(&self.mullion_pitch_m)
            || !(self.transom_fraction == 0. || (0.2..=0.85).contains(&self.transom_fraction))
            || !(0.02..=0.075).contains(&self.frame_width_m)
            || !(0.04..=0.12).contains(&self.frame_depth_m)
            || !matches!(
                self.frame,
                Surface::WoodEdge | Surface::Metal | Surface::Plastic
            )
        {
            return Err("invalid interior glazing installation".into());
        }
        Ok(())
    }
}

/// Wall coordinates look inward from +Z, so the entry-door coordinate is negated.
pub fn shared_facade(scene: &IndoorManifest, height: f32) -> Facade {
    let hx = scene.room_size.x * 0.5;
    let door = -scene.door_x;
    let p = InteriorGlazing::for_scene(scene);
    let mut openings = Vec::new();
    let design = p.and_then(|p| p.shared_openings);
    let sill = p.map_or(0.05, |p| p.sill_m);
    let head = height - p.map_or(0.15, |p| p.header_m);
    let band = design.map_or(Vec2::new(sill, head), |s| {
        Vec2::splat(sill) + s.height_interval * (head - sill)
    });
    let fits_frame =
        design.is_none() || band.y - band.x >= 2. * p.map_or(0.035, |p| p.frame_width_m) + 0.09;
    for (side, (lo, hi)) in [
        (-hx + 0.10, door - scene.entry_width() * 0.5 - 0.05),
        (door + scene.entry_width() * 0.5 + 0.05, hx - 0.10),
    ]
    .into_iter()
    .enumerate()
    {
        let (lo, hi) = if let Some(design) = design {
            let v = design.jamb_intervals[side] * (hi - lo);
            if side == 0 {
                (hi - v.y, hi - v.x)
            } else {
                (lo + v.x, lo + v.y)
            }
        } else {
            (lo, hi)
        };
        if !fits_frame {
            continue;
        }
        if hi - lo <= if p.is_some() { 0.40 } else { 0.18 } {
            continue;
        }
        let count = p.map_or(1, |p| {
            ((hi - lo) / p.opening_pitch_m).round().clamp(1., 6.) as usize
        });
        let pitch = (hi - lo) / count as f32;
        for i in 0..count {
            let pier = p.map_or(0., |p| pitch * p.pier_fraction * 0.5);
            let left = lo + i as f32 * pitch + pier;
            let right = lo + (i + 1) as f32 * pitch - pier;
            if p.is_some() && right - left < 0.35 {
                continue;
            }
            openings.push(WindowOpening {
                glazing: None,
                min: Vec2::new(left, band.x),
                max: Vec2::new(right, band.y),
                columns: ((right - left) / p.map_or(1.3, |p| p.mullion_pitch_m))
                    .ceil()
                    .clamp(1., if p.is_some() { 6. } else { f32::INFINITY })
                    as u32,
                transom: p.map_or(0., |p| p.transom_fraction),
                transoms: vec![],
                operable: false,
            });
        }
    }
    Facade {
        joinery: p.map(|p| p.joinery.clone()),
        side: FacadeSide::Rear,
        openings,
        frame: p.map_or(Surface::Metal, |p| p.frame),
        frame_width: p.map_or(0.035, |p| p.frame_width_m),
        frame_depth: p.map_or(0.075, |p| p.frame_depth_m),
        recess: 0.06,
        sill_projection: 0.,
        treatment: None,
        shade: Shade::None,
        shade_coverage: 0.,
        shade_tilt: 0.,
    }
}

/// Both rough openings and wall-mounted equipment use the same limiting roof height.
pub(crate) fn partition_height(scene: &IndoorManifest) -> f32 {
    let half = scene.room_size.xz() * 0.5;
    scene
        .ceiling_height(Vec2::new(-half.x, half.y))
        .min(scene.ceiling_height(half))
}

#[cfg(test)]
mod shared_tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::IndoorLayout;

    #[test]
    fn fixed_wall_admits_asymmetric_high_broad_and_private_glazing() {
        let mut scene =
            IndoorManifest::generate_with_humans(11, IndoorLayout::Conference, 0.5, 0, 0.).unwrap();
        let h = partition_height(&scene);
        let mut counts = [0; 4];
        let mut spans = std::collections::BTreeSet::new();
        for seed in 0..256 {
            let installed = InteriorGlazing::sample(seed);
            installed.validate().unwrap();
            let sill = installed.sill_m;
            scene
                .program
                .as_mut()
                .unwrap()
                .finishes
                .as_mut()
                .unwrap()
                .interior_glazing = Some(installed);
            let facade = shared_facade(&scene, h);
            let mut sides = [false; 2];
            let mut area = 0.;
            for o in &facade.openings {
                assert!(o.min.y >= sill && o.max.y <= h && o.max.cmpgt(o.min).all());
                let left = o.max.x <= -scene.door_x - scene.entry_width() * 0.5;
                let right = o.min.x >= -scene.door_x + scene.entry_width() * 0.5;
                assert!(left || right);
                sides[usize::from(right)] = true;
                area += (o.max - o.min).element_product();
            }
            counts[0] += usize::from(!sides[0] && !sides[1]);
            counts[1] += usize::from(sides[0] != sides[1]);
            counts[2] += usize::from(sides[0] && sides[1]);
            counts[3] += usize::from(
                !facade.openings.is_empty() && facade.openings.iter().all(|o| o.min.y > h * 0.48),
            );
            spans.insert((area * 100.).round() as usize);
        }
        assert!(counts.iter().all(|count| *count >= 10), "{counts:?}");
        assert!(spans.len() > 155);
        let mut legacy = serde_json::to_value(InteriorGlazing::sample(42)).unwrap();
        legacy.as_object_mut().unwrap().remove("shared_openings");
        let legacy: InteriorGlazing = serde_json::from_value(legacy).unwrap();
        assert!(legacy.shared_openings.is_none());
        let sill = legacy.sill_m;
        scene
            .program
            .as_mut()
            .unwrap()
            .finishes
            .as_mut()
            .unwrap()
            .interior_glazing = Some(legacy);
        let windows = shared_facade(&scene, h).openings;
        assert!(!windows.is_empty() && windows.iter().all(|o| o.min.y == sill));
    }
}
