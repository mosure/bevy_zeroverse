use super::*;
pub(super) fn build(a: &mut Assembly, o: &IndoorObject, p: &PublicItemProgram) {
    let label = o.kind.class_name();
    match o.kind {
        ObjectKind::ExhibitPlinth => {
            panel(
                a,
                o,
                Surface::Rubber,
                label,
                [0., 0.025, 0.],
                [0.90, 0.05, 0.90],
            );
            panel(a, o, p.finish, label, [0., 0.50, 0.], [0.97, 0.90, 0.97]);
            panel(
                a,
                o,
                Surface::Ceramic,
                label,
                [0., 0.975, 0.],
                [1., 0.05, 1.],
            );
        }
        ObjectKind::Sculpture => super::sculpture::build(a, o),
        ObjectKind::InformationStand => super::signage::stand(a, o),
        ObjectKind::ExhibitionPanel => super::signage::exhibition(a, o),
        _ => unreachable!(),
    }
}
