use super::*;
use crate::scene::procedural_indoor::layout::ObjectKind;

/// Weight the installed work surfaces that can actually receive a ceiling
/// light. This decision is made once, before any view is sampled.
pub(super) fn assign(
    scene: &IndoorManifest,
    mounts: &[Vec3],
    ids: &[usize],
    banks: &mut [Bank],
    windows: &[DaylightOpening],
    borrowed_lux: f32,
) -> f32 {
    let program = scene.program.as_ref().unwrap();
    let deck = scene
        .envelope
        .as_ref()
        .and_then(|e| e.mezzanine.as_ref())
        .map(|m| &m.deck);
    let occupied = ((scene.target_lux - 40.) / 110.).clamp(0., 1.);
    let mut total = 0.;
    for surface in scene.objects.iter().filter(|o| {
        !o.neighbor
            && matches!(
                o.kind,
                ObjectKind::Desk
                    | ObjectKind::Table
                    | ObjectKind::KitchenCounter
                    | ObjectKind::ServiceCounter
                    | ObjectKind::Lectern
            )
    }) {
        let Some(zone) = zone_at(&program.zones, surface.position.xz()) else {
            continue;
        };
        let weight = (surface.size.x * surface.size.z).sqrt();
        let span = surface.size.x.max(surface.size.z);
        let points = if span > 3.2 {
            &[-0.32, 0.32][..]
        } else {
            &[0.][..]
        };
        let axis = if surface.size.x > surface.size.z {
            Vec3::X
        } else {
            Vec3::Z
        };
        let frame = surface.transform();
        for &along in points {
            let point = frame.transform_point(axis * span * along + Vec3::Y * surface.size.y);
            let bank = mounts
                .iter()
                .zip(ids)
                .filter_map(|(&lamp, &id)| {
                    let throw = lamp.y - point.y;
                    let delta = lamp.xz() - point.xz();
                    let distance_sq = delta.length_squared();
                    if banks[id].zone != zone || throw <= 0. || distance_sq > (2. * throw).powi(2) {
                        return None;
                    }
                    if deck.is_some_and(|d| {
                        let t = (d.height - point.y) / throw;
                        (0.0..1.).contains(&t) && d.contains(point.xz() + t * delta)
                    }) {
                        return None;
                    }
                    Some((id, distance_sq + throw * throw))
                })
                .min_by(|a, b| a.1.total_cmp(&b.1));
            if let Some((id, _)) = bank {
                let weight = weight / points.len() as f32;
                let daylight = borrowed_lux
                    + windows
                        .iter()
                        .map(|w| w.signal(point.xz(), zone, &program.zones))
                        .sum::<f32>();
                let deficit = (1. - daylight / (0.65 * scene.target_lux)).clamp(0., 1.);
                banks[id].task_demand += weight * occupied * deficit;
                total += weight;
            }
        }
    }
    total
}
