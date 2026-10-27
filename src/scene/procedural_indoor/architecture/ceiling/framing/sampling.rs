use super::*;
use crate::scene::procedural_indoor::envelope::polygon;

pub(super) fn admissible(
    scene: &IndoorManifest,
    m: &FrameMember,
    r: &super::super::sampling::Reservations,
) -> bool {
    let (lo, hi) = m.bounds();
    lo.y >= super::super::highest_floor(scene, lo.xz(), hi.xz()) + MIN_HEADROOM_M
        && scene.envelope.as_ref().is_none_or(|e| {
            polygon::box_inside(&e.footprint, lo.xz(), hi.xz(), 0.16)
                && !e.soffits.iter().any(|p| {
                    p.overlaps(lo.xz(), hi.xz())
                        && !(lo.xz().cmpge(p.min).all() && hi.xz().cmple(p.max).all())
                })
        })
        && m.top
            .iter()
            .all(|p| p.y <= scene.ceiling_height(p.xz()) - 0.0005)
        && r.structure
            .iter()
            .chain(&r.partitions)
            .chain(&r.fixture_bodies)
            .all(|&b| !overlap((lo, hi), b, 0.006))
}

#[allow(clippy::too_many_arguments)]
fn member(
    scene: &IndoorManifest,
    lo: Vec2,
    hi: Vec2,
    axis: usize,
    drop: f32,
    depth: f32,
    role: FrameRole,
    section: FrameSection,
    material: Surface,
    finish: u8,
    pitch: f32,
) -> FrameMember {
    let points = if axis == 0 {
        [lo, Vec2::new(hi.x, lo.y), hi, Vec2::new(lo.x, hi.y)]
    } else {
        [lo, Vec2::new(lo.x, hi.y), hi, Vec2::new(hi.x, lo.y)]
    };
    FrameMember {
        top: points.map(|p| Vec3::new(p.x, scene.ceiling_height(p) - drop, p.y)),
        depth,
        role,
        section,
        material,
        finish,
        web_pitch: pitch,
    }
}

pub(super) fn sample(
    scene: &IndoorManifest,
    reserved: &super::super::sampling::Reservations,
) -> CeilingFraming {
    let mut r = stream(scene.seed, 0x524f4f464652414d);
    let timber = r.random_bool(0.36);
    let material = if timber {
        Surface::Wood
    } else {
        [Surface::Metal, Surface::Paint][r.random_range(0..2)]
    };
    let mut plan = CeilingFraming {
        roof_material: if timber {
            Surface::Wood
        } else {
            [Surface::Concrete, Surface::Metal][r.random_range(0..2)]
        },
        members: Vec::new(),
    };
    let Some(program) = &scene.program else {
        return plan;
    };
    for zone in &program.zones {
        let lo = zone.min + Vec2::splat(0.28);
        let hi = zone.max - Vec2::splat(0.28);
        let extent = hi - lo;
        if extent.min_element() < 1.4 {
            continue;
        }
        let axis = usize::from(r.random_bool(0.5));
        let across = 1 - axis;
        let roof = scene.minimum_ceiling_height(lo, hi);
        let available = roof - super::super::highest_floor(scene, lo, hi) - 2.39;
        if available < 0.22 {
            continue;
        }
        let rib_depth = r.random_range(0.025_f32..0.065).min(available * 0.12);
        let purlin_depth = r.random_range(0.08_f32..0.18).min(available * 0.22);
        let depth = r
            .random_range(0.28_f32..0.75)
            .min(available - rib_depth - purlin_depth);
        let width = if timber {
            r.random_range(0.12..0.25)
        } else {
            r.random_range(0.10..0.22)
        };
        let pitch = r.random_range(1.55..3.1);
        let count = (extent[across] / pitch).floor().clamp(1., 8.) as usize;
        let phase = r.random_range(0.15..0.85);
        let finish = r.random_range(0..12);
        let section = if timber {
            FrameSection::Solid
        } else if r.random_bool(0.55) {
            FrameSection::ISection
        } else {
            FrameSection::OpenWeb
        };
        let web_pitch = r.random_range(0.45..1.25);
        for i in 0..count {
            if plan.members.len() >= MAX_MEMBERS {
                break;
            }
            let center = lo[across] + extent[across] * (i as f32 + phase) / count as f32;
            let mut a = lo;
            let mut b = hi;
            a[across] = center - width * 0.5;
            b[across] = center + width * 0.5;
            // Shorten around existing architecture and luminaires. Subsequent
            // service proposals adapt to this structure, not vice versa.
            for _ in 0..4 {
                let m = member(
                    scene,
                    a,
                    b,
                    axis,
                    0.003 + rib_depth + purlin_depth,
                    depth,
                    FrameRole::Rafter,
                    section,
                    material,
                    finish,
                    web_pitch,
                );
                if admissible(scene, &m, reserved) {
                    let ends = [m.point(0.035, 0.5, 0.), m.point(0.965, 0.5, 0.)];
                    plan.members.push(m);
                    for p in ends {
                        if plan.members.len() >= MAX_MEMBERS {
                            break;
                        }
                        let mut half = Vec2::splat(width * 0.5);
                        half[axis] = 0.08;
                        let bearing = member(
                            scene,
                            p.xz() - half,
                            p.xz() + half,
                            axis,
                            0.001,
                            // Embed the foot 2 mm into the rafter. It avoids a
                            // coplanar underside if a purlin meets this bearing.
                            rib_depth + purlin_depth + 0.004,
                            FrameRole::Bearing,
                            FrameSection::Solid,
                            material,
                            finish,
                            web_pitch,
                        );
                        if admissible(scene, &bearing, reserved) {
                            plan.members.push(bearing);
                        }
                    }
                    break;
                }
                let inset = (b[axis] - a[axis]) * 0.09;
                a[axis] += inset;
                b[axis] -= inset;
                if b[axis] - a[axis] < 1.0 {
                    break;
                }
            }
        }
        for (role, along, pitch, width, drop, depth, section) in [
            (
                FrameRole::Purlin,
                across,
                r.random_range(0.55..1.15),
                r.random_range(0.055..0.10),
                0.003 + rib_depth,
                purlin_depth,
                FrameSection::Solid,
            ),
            (
                FrameRole::RoofRib,
                axis,
                r.random_range(0.26..0.62),
                r.random_range(0.055..0.14),
                0.001,
                rib_depth,
                FrameSection::Fluted,
            ),
        ] {
            let count = (extent[1 - along] / pitch).floor().clamp(1., 56.) as usize;
            let phase = r.random_range(0.12..0.88);
            for i in 0..count {
                if plan.members.len() >= MAX_MEMBERS {
                    break;
                }
                let center = lo[1 - along] + extent[1 - along] * (i as f32 + phase) / count as f32;
                let mut a = lo;
                let mut b = hi;
                a[1 - along] = center - width * 0.5;
                b[1 - along] = center + width * 0.5;
                let m = member(
                    scene,
                    a,
                    b,
                    along,
                    drop,
                    depth,
                    role,
                    section,
                    if role == FrameRole::RoofRib {
                        plan.roof_material
                    } else {
                        material
                    },
                    finish,
                    web_pitch,
                );
                if admissible(scene, &m, reserved) {
                    plan.members.push(m);
                }
            }
        }
    }
    connect_structure(scene, reserved, &mut plan);
    plan
}

/// Partial bays must not leave secondary members floating. Fit checked roof
/// brackets, then omit any component with no connection to the structure.
fn connect_structure(
    scene: &IndoorManifest,
    reserved: &super::super::sampling::Reservations,
    plan: &mut CeilingFraming,
) {
    let connected = plan.connected();
    for (index, connected) in connected.into_iter().enumerate() {
        if connected || plan.members.len() + 2 > MAX_MEMBERS {
            continue;
        }
        let m = &plan.members[index];
        let edge = m.top[1] - m.top[0];
        let axis = usize::from(edge.z.abs() > edge.x.abs());
        let width = m.top[0].xz().distance(m.top[3].xz());
        let drop = scene.ceiling_height(m.top[0].xz()) - m.top[0].y;
        for inset in [0.035, 0.18, 0.32] {
            let supports = [inset, 1. - inset].map(|t| {
                let p = m.point(t, 0.5, 0.);
                let mut half = Vec2::splat(width * 0.5);
                half[axis] = 0.08;
                member(
                    scene,
                    p.xz() - half,
                    p.xz() + half,
                    axis,
                    0.001,
                    drop + 0.001,
                    FrameRole::Bearing,
                    FrameSection::Solid,
                    m.material,
                    m.finish,
                    m.web_pitch,
                )
            });
            if supports.iter().all(|s| admissible(scene, s, reserved)) {
                plan.members.extend(supports);
                break;
            }
        }
    }
    let mut connected = plan.connected().into_iter();
    plan.members.retain(|_| connected.next().unwrap());
}
