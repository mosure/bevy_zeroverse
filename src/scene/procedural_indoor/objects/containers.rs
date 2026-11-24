//! Hollow containers with bounded, continuously sampled proportions.
use super::*;

pub mod bins;
pub use bins::{parameters as bin_parameters, BinProgram};

pub(super) fn bin(a: &mut Assembly, o: &IndoorObject) {
    bins::build(a, o);
}

pub mod tissue;
pub use tissue::{parameters as tissue_parameters, TissueProgram};

pub(super) fn tissues(a: &mut Assembly, o: &IndoorObject) {
    tissue::build(a, o);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{
        layout::{IndoorLayout, IndoorManifest},
        validation,
    };
    #[test]
    fn container_and_fixture_programs_are_bounded_replayable_and_diverse() {
        let mut scene =
            IndoorManifest::generate_with_humans(42, IndoorLayout::Mixed, 0.65, 0, 0.).unwrap();
        let template = scene.objects[0].clone();
        scene.objects.clear();
        let mut bins = std::collections::BTreeSet::new();
        let mut switches = std::collections::BTreeSet::new();
        let mut tissues = std::collections::BTreeSet::new();
        for seed in 0..128 {
            for (kind, size) in [
                (ObjectKind::TrashCan, Vec3::new(0.27, 0.32, 0.26)),
                (ObjectKind::TrashCan, Vec3::new(0.46, 0.68, 0.42)),
                (ObjectKind::TissueBox, Vec3::new(0.14, 0.08, 0.10)),
                (ObjectKind::TissueBox, Vec3::new(0.24, 0.14, 0.16)),
                (ObjectKind::WallOutlet, Vec3::new(0.075, 0.10, 0.018)),
                (ObjectKind::WallOutlet, Vec3::new(0.225, 0.125, 0.018)),
                (ObjectKind::LightSwitch, Vec3::new(0.075, 0.10, 0.018)),
                (ObjectKind::LightSwitch, Vec3::new(0.225, 0.125, 0.018)),
            ] {
                let mut o = template.clone();
                o.kind = kind;
                o.size = size;
                o.seed = seed;
                o.id = scene.objects.len();
                o.position = Vec3::ZERO;
                o.support = None;
                o.solid = false;
                assert_eq!(bin_parameters(&o), bin_parameters(&o));
                bins.insert(bin_parameters(&o).lid);
                switches.insert(wall::parameters(&o).style);
                tissues.insert(serde_json::to_string(&tissue_parameters(&o)).unwrap());
                let a = build_object(&o);
                let (lo, hi) = a.bounds();
                assert!(
                    lo.y < 0.003,
                    "missing support contact {kind:?} {seed}: {lo}"
                );
                assert!(
                    a.parts.values().map(|g| g.indices.len() / 3).sum::<usize>() < 5000,
                    "unbounded mesh budget"
                );
                if kind == ObjectKind::TissueBox {
                    assert!(hi.y > size.y * 0.94, "tissue disappeared");
                }
                scene.objects.push(o);
            }
        }
        validation::validate_geometry(&scene).unwrap();
        assert_eq!(bins.len(), 5);
        assert_eq!(switches.len(), 6);
        assert_eq!(tissues.len(), 128);
    }
}
