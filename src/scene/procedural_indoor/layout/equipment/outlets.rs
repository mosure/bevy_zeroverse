//! Choose among the actual free outlets on a fitted plate or strip. Reordering
//! this small list does not advance the cable shape or room placement streams.
use super::*;
use equipment::receptacles::MainsBank;

pub(super) fn available(
    scene: &IndoorManifest,
    source: &IndoorObject,
    target: &IndoorObject,
    low: bool,
) -> Vec<(Option<u8>, Vec3)> {
    let mut occupied = scene
        .objects
        .iter()
        .filter(|c| c.kind == ObjectKind::Cable && c.cable_target == Some(target.id))
        .filter(|c| {
            c.interaction_target
                .and_then(|id| scene.objects.get(id))
                .is_some_and(|s| (s.kind == ObjectKind::Laptop) == low)
        })
        .map(|c| usize::from(c.cable_target_socket.unwrap_or(0)));
    if let Some(bank) = (!low).then(|| MainsBank::for_target(target)).flatten() {
        let mut used = 0u32;
        for i in occupied {
            if let Some(bit) = 1u32.checked_shl(i as u32) {
                used |= bit;
            }
        }
        let tf = target.transform();
        let mut choices: Vec<_> = (0..bank.socket_count())
            .filter(|&i| used & (1 << i) == 0)
            .filter_map(|i| {
                Some((
                    Some(i as u8),
                    tf.transform_point(bank.local(target.size, i)?),
                ))
            })
            .collect();
        choices.shuffle(&mut stream(
            source.seed ^ target.seed,
            u64::from_le_bytes(*b"OutletIx"),
        ));
        choices
    } else if occupied.next().is_none() {
        equipment::power_port(target, low)
            .map(|p| (None, p))
            .into_iter()
            .collect()
    } else {
        Vec::new()
    }
}
