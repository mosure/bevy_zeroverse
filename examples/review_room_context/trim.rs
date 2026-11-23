//! Aim at the installed skirting and cornice from clear positions inside the room.
use super::*;
use bevy_zeroverse::scene::procedural_indoor::{
    architecture::trim::TrimProgram, layout::segment_hits_box,
};

pub(super) fn targets(scene: &IndoorManifest) -> Result<[(Vec3, Vec3); 3]> {
    let e = scene
        .envelope
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("missing envelope"))?;
    let trim =
        TrimProgram::for_scene(scene).ok_or_else(|| anyhow::anyhow!("missing trim program"))?;
    let mut inspection = scene.clone();
    inspection.camera_settings.primary_room = false;
    let n = e.footprint.len();
    let structures = e.structural_boxes(scene.room_size);
    let mut views = [(Vec3::ZERO, Vec3::ZERO); 3];
    for (view, pose) in views.iter_mut().enumerate() {
        let crown = trim.base.is_none() || view == 1 && trim.crown.is_some();
        let section = if crown { &trim.crown } else { &trim.base };
        let section = section
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("room has no trim"))?;
        let mut found = None;
        for offset in 0..n {
            let i = (offset + if view == 2 { n / 2 } else { 0 }) % n;
            let v = e.footprint[i];
            let left = (v - e.footprint[(i + n - 1) % n]).normalize().perp();
            let right = (e.footprint[(i + 1) % n] - v).normalize().perp();
            let inside = (left + right).normalize();
            let look = v + inside * section.depth_m;
            let level = if crown {
                e.roof_height(scene.room_size, look) - section.height_m * 0.5
            } else {
                section.height_m * 0.5
            };
            let target = Vec3::new(look.x, level, look.y);
            for distance in [1.35, 1.8, 2.35] {
                let footprint = v + inside * distance;
                let y = if crown {
                    (level - 0.85).min(scene.ceiling_height(footprint) - 1.)
                } else {
                    0.86
                };
                let eye = Vec3::new(footprint.x, y, footprint.y);
                if inspection.camera_clear(eye)
                    && !scene.objects.iter().filter(|o| o.solid).any(|o| {
                        let (lo, hi) = o.bounds();
                        segment_hits_box(eye, target, lo, hi)
                    })
                    && !structures
                        .iter()
                        .any(|&(lo, hi)| segment_hits_box(eye, target, lo, hi))
                {
                    found = Some((eye, target));
                    break;
                }
            }
            if found.is_some() {
                break;
            }
        }
        *pose = found
            .ok_or_else(|| anyhow::anyhow!("no visible trim corner in room {}", scene.seed))?;
    }
    Ok(views)
}
