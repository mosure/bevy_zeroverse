//! Decide which neighboring elevations warrant interiors from the permanent
//! primary-room openings. Camera count, position and time never select detail.
use super::*;

pub(super) fn viewed_sides(parcels: &[Parcel], windows: &[WindowView]) -> Vec<Option<usize>> {
    let mut ranked = Vec::new();
    for (index, p) in parcels.iter().enumerate().filter(|(_, p)| p.kind == 0) {
        let best = windows
            .iter()
            .filter_map(|w| {
                let delta = p.center - w.center;
                let distance = delta.length();
                let alignment = w.normal.dot(delta.normalize_or_zero());
                let gap = distance - 0.5 * p.size.min_element();
                if alignment <= 0.60 || gap > 13.0 {
                    return None;
                }
                // Side numbering follows the elevation builder (+Z,+X,-Z,-X).
                let view = Quat::from_rotation_y(-p.yaw) * Vec3::new(-delta.x, 0., -delta.y);
                let side = if view.z.abs() >= view.x.abs() {
                    if view.z >= 0. {
                        0
                    } else {
                        2
                    }
                } else if view.x >= 0. {
                    1
                } else {
                    3
                };
                Some((gap + 4.0 * (1.0 - alignment), side))
            })
            .min_by(|a, b| a.0.total_cmp(&b.0));
        if let Some((score, side)) = best {
            ranked.push((score, index, side));
        }
    }
    ranked.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
    let mut selected = vec![None; parcels.len()];
    for (_, index, side) in ranked.into_iter().take(3) {
        selected[index] = Some(side);
    }
    selected
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn selects_only_near_facing_elevations_in_each_buildings_local_frame() {
        let site = |x, z, yaw| Parcel {
            center: Vec2::new(x, z),
            size: Vec2::splat(3.),
            yaw,
            kind: 0,
            seed: 3,
        };
        let sites = [
            site(0., 5., 0.),
            site(0., 8., std::f32::consts::FRAC_PI_2),
            site(-4., 9., std::f32::consts::PI),
            site(0., 12., 0.),
            site(0., 19., 0.),
            site(0., -5., 0.),
            site(5., 1., 0.),
        ];
        let primary = [WindowView {
            center: Vec2::ZERO,
            normal: Vec2::Y,
            width: 2.,
            sill: 0.7,
            head: 2.5,
        }];
        assert_eq!(
            viewed_sides(&sites, &primary),
            [Some(2), Some(1), Some(0), None, None, None, None]
        );
        assert_eq!(viewed_sides(&sites, &[]), vec![None; sites.len()]);
    }
}
