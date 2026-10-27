//! Dimensioned office equipment and belongings with independently sampled mechanisms.
use super::*;
pub mod boxes;
pub mod printer;
pub mod racks;
#[cfg(test)]
mod storage_tests;
#[derive(Debug, Clone, serde::Serialize)]
pub struct UtilityProgram {
    pub construction: u8,
    pub taper: f32,
    pub roundness: f32,
    pub split: f32,
    pub lid_fraction: f32,
    pub handle_span: f32,
    pub arms: usize,
    pub branch_height: f32,
}
pub fn parameters(o: &IndoorObject) -> UtilityProgram {
    let mut r = stream(o.seed, 7430);
    UtilityProgram {
        construction: r.random_range(0..4),
        taper: r.random_range(0.72..0.98),
        roundness: r.random_range(3.5..9.),
        split: r.random_range(0.42..0.72),
        lid_fraction: r.random_range(0.07..0.19),
        handle_span: r.random_range(0.28..0.58),
        arms: r.random_range(3..=6),
        branch_height: r.random_range(0.62..0.84),
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::Printer => printer::build(a, o),
        ObjectKind::StorageBox => boxes::build(a, o),
        ObjectKind::CoatRack => racks::build(a, o),
        ObjectKind::Bag => super::bags::build(a, o),
        _ => unreachable!(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn utilities_keep_all_parts_inside_placement_envelopes() {
        for kind in [
            ObjectKind::Printer,
            ObjectKind::StorageBox,
            ObjectKind::CoatRack,
            ObjectKind::Bag,
        ] {
            for seed in 0..12 {
                let size = if kind == ObjectKind::CoatRack {
                    Vec3::new(0.55, 1.5, 0.55)
                } else {
                    Vec3::new(0.4, 0.28, 0.3)
                };
                let o = IndoorObject {
                    finish_seed: None,
                    id: 0,
                    kind,
                    position: Vec3::ZERO,
                    size,
                    yaw: 0.0,
                    variant: 0,
                    seed,
                    solid: true,
                    support: None,
                    neighbor: false,
                    interaction_target: None,
                    cable_path: None,
                    cable_target: None,
                };
                let a = super::super::build_object(&o);
                let (lo, hi) = a.bounds();
                assert!(lo.is_finite() && hi.is_finite());
                assert!(
                    lo.cmpge(Vec3::new(-size.x * 0.5, 0.0, -size.z * 0.5) - Vec3::splat(0.001))
                        .all(),
                    "{kind:?} lower {lo:?}"
                );
                assert!(
                    hi.cmple(Vec3::new(size.x * 0.5, size.y, size.z * 0.5) + Vec3::splat(0.001))
                        .all(),
                    "{kind:?} upper {hi:?}"
                );
            }
        }
    }
}
