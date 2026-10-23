use super::*;

pub(super) fn content_size(kind: ObjectKind, seed: u64, r: &mut impl Rng) -> Vec3 {
    if matches!(kind, ObjectKind::Bread | ObjectKind::Dishware) {
        super::super::kitchen::dimensions(kind, seed)
    } else if kind == ObjectKind::Sculpture {
        // Display-case miniatures, distinct from floor/plinth-scale exhibits.
        Vec3::new(
            r.random_range(0.12..0.28),
            r.random_range(0.15..0.45),
            r.random_range(0.10..0.26),
        )
    } else {
        super::super::super::items::sizing::loose(kind, r).expect("case content dimensions")
    }
}
