use super::*;
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    match o.kind {
        ObjectKind::Sculpture => super::sculpture::build(a, o),
        ObjectKind::InformationStand => super::signage::stand(a, o),
        ObjectKind::ExhibitionPanel => super::signage::exhibition(a, o),
        _ => unreachable!(),
    }
}
