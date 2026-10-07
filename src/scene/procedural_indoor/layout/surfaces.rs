//! Seeded work-area transforms and mixed clustered/uniform surface proposals.
//! All proposals still pass the same true-outline and peer-overlap checks.
use super::*;
impl IndoorManifest {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn place_prop(
        &mut self,
        support: &IndoorObject,
        kind: ObjectKind,
        offset: Vec3,
        size: Vec3,
        yaw: f32,
        rng: &mut ChaCha8Rng,
    ) {
        let mut program = stream(support.seed, 906);
        let handed = if program.random_bool(0.5) { -1. } else { 1. };
        let angle = program.random_range(-0.20..0.20);
        let spread = program.random_range(0.78..1.05);
        let disorder = program.random_range(0.015..0.11);
        let shift = Vec3::new(
            program.random_range(-0.12..0.12) * support.size.x.min(1.8),
            0.,
            program.random_range(-0.12..0.10) * support.size.z.min(1.0),
        );
        let mut offset = shift
            + Quat::from_rotation_y(angle)
                * Vec3::new(offset.x * handed * spread, 0., offset.z * spread);
        offset += Vec3::new(
            rng.random_range(-disorder..disorder),
            0.,
            rng.random_range(-disorder..disorder),
        );
        let mut yaw = angle + yaw * handed + rng.random_range(-disorder * 2.5..disorder * 2.5);
        if matches!(kind, ObjectKind::Keyboard | ObjectKind::Mouse) {
            if let Some(monitor) = self
                .objects
                .iter()
                .find(|o| o.support == Some(support.id) && o.kind == ObjectKind::Monitor)
            {
                // Keep input devices on the user's side of the actual display.
                let local = Vec3::new(
                    if kind == ObjectKind::Mouse {
                        handed * 0.25
                    } else {
                        0.
                    },
                    0.,
                    if kind == ObjectKind::Mouse {
                        0.27
                    } else {
                        0.26
                    },
                );
                let world = monitor.position + Quat::from_rotation_y(monitor.yaw) * local;
                offset = support
                    .transform()
                    .compute_affine()
                    .inverse()
                    .transform_point3(world)
                    .with_y(0.);
                yaw = monitor.yaw - support.yaw + rng.random_range(-0.12..0.12);
            }
        }
        self.prop(support, kind, offset, size, yaw, rng);
    }
}

pub(super) fn scatter_proposal(
    objects: &[IndoorObject],
    support: &IndoorObject,
    kind: ObjectKind,
    size: Vec3,
    rng: &mut ChaCha8Rng,
) -> (Vec3, f32) {
    use rand::seq::IteratorRandom;
    let mut offset = Vec3::new(
        rng.random_range(-0.47..0.47) * support.size.x,
        0.,
        rng.random_range(-0.47..0.47) * support.size.z,
    );
    let mut yaw = rng.random_range(-std::f32::consts::PI..std::f32::consts::PI);
    // Accessories form loose, variably rotated clusters around an existing
    // work item; the uniform component still explores the entire usable top.
    if rng.random_bool(0.48) {
        if let Some(anchor) = objects
            .iter()
            .filter(|o| {
                o.support == Some(support.id)
                    && matches!(
                        o.kind,
                        ObjectKind::Notebook
                            | ObjectKind::Notepad
                            | ObjectKind::Laptop
                            | ObjectKind::Books
                    )
            })
            .choose(rng)
        {
            let angle = rng.random_range(0.0..std::f32::consts::TAU);
            let r = 0.5 * (anchor.size.xz().length() + size.xz().length())
                + rng.random_range(0.025..0.15);
            let local = support
                .transform()
                .compute_affine()
                .inverse()
                .transform_point3(anchor.position);
            offset = (local + Vec3::new(angle.cos() * r, 0., angle.sin() * r)).with_y(0.);
            if matches!(
                kind,
                ObjectKind::Pencil | ObjectKind::Phone | ObjectKind::Notepad
            ) {
                yaw = anchor.yaw - support.yaw + rng.random_range(-0.45..0.45);
            }
        }
    }
    (offset, yaw)
}
