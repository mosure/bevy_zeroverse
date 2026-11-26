use super::*;

/// A punched rear intake and removable service cover sit inside the admitted
/// envelope. The dark cavity is behind the slats, not painted on the cover.
pub(super) fn base(
    a: &mut Assembly,
    o: &IndoorObject,
    finish: (Surface, &str),
    toe: f32,
    top: f32,
) {
    let (surface, label) = finish;
    let s = o.size;
    let rear = -s.z * 0.5;
    let recess = 0.019;
    let face = rear + 0.004;
    let mid_y = (toe + top) * 0.5;
    let height = top - toe;
    let mut r = stream(o.seed, u64::from_le_bytes(*b"ColdVent"));
    let vent_width = s.x * r.random_range(0.26..0.48);
    let left = -s.x * 0.5;
    let right = -left;
    let vent_left = left + (s.x - vent_width) * r.random_range(0.09..0.91);
    let vent_right = vent_left + vent_width;
    let bezel = height * r.random_range(0.12..0.21);
    board(
        a,
        surface,
        label,
        Vec3::new(0., mid_y, recess * 0.5),
        Vec3::new(s.x, height, s.z - recess),
    );
    let mut cover = |x0: f32, x1: f32, y0: f32, y1: f32| {
        board(
            a,
            surface,
            label,
            Vec3::new(
                (x0 + x1) * 0.5,
                (y0 + y1) * 0.5,
                (rear + face + recess) * 0.5,
            ),
            Vec3::new(x1 - x0, y1 - y0, rear + recess - face),
        );
    };
    cover(left, vent_left, toe, top);
    cover(vent_right, right, toe, top);
    cover(vent_left, vent_right, toe, toe + bezel);
    cover(vent_left, vent_right, top - bezel, top);
    let x = (vent_left + vent_right) * 0.5;
    let inside = height - 2. * bezel;
    board(
        a,
        Surface::Rubber,
        FIXTURE,
        Vec3::new(x, mid_y, rear + recess - 0.0015),
        Vec3::new(vent_width, inside, 0.003),
    );
    let slats = (inside / 0.030).round().clamp(3., 9.) as usize;
    let spacing = inside / slats as f32;
    for row in 0..slats {
        board(
            a,
            surface,
            label,
            Vec3::new(x, toe + bezel + (row as f32 + 0.5) * spacing, face + 0.004),
            Vec3::new(vent_width, spacing * 0.54, 0.008),
        );
    }
}

pub(super) fn ends(
    a: &mut Assembly,
    o: &IndoorObject,
    case: RefrigeratedCase,
    finish: (Surface, &str),
    base: f32,
    canopy: f32,
) {
    let (surface, label) = finish;
    let s = o.size;
    let back = -s.z * 0.5;
    let front = -back;
    for (side, gauge) in [-1., 1.].into_iter().zip(case.end_glass_m) {
        let x = side * (s.x - case.end_thickness_m) * 0.5;
        let mut panel = |low: f32, high: f32, rear: f32, nose: f32| {
            board(
                a,
                surface,
                label,
                Vec3::new(x, (low + high) * 0.5, (rear + nose) * 0.5),
                Vec3::new(case.end_thickness_m, high - low, nose - rear),
            );
        };
        let Some(gauge) = gauge else {
            panel(base, canopy, back, front);
            continue;
        };
        let sill = case.deck_height_m + case.end_sill_m;
        let head = canopy - 0.025;
        let rear = front - s.z * case.end_window_depth_fraction;
        let nose = front - 0.025;
        panel(base, sill, back, front);
        panel(head, canopy, back, front);
        panel(sill, head, back, rear);
        panel(sill, head, nose, front);
        let glass = format!("{FIXTURE}#finish{}", gauge.slot());
        board(
            a,
            Surface::ContainerGlass,
            &glass,
            Vec3::new(x, (sill + head) * 0.5, (rear + nose) * 0.5),
            Vec3::new(gauge.metres(), head - sill, nose - rear),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::{materials, objects::audit};
    #[test]
    fn retail_case_end_windows_are_physical_openings_with_the_archived_sheet_gauge() {
        let mut ends_seen = [false; 3];
        for seed in 0..48 {
            let mut o = audit::fixture(ObjectKind::GroceryDisplay, seed);
            o.size = Vec3::new(1.85, 1.67, 0.82);
            o.variant |= REFRIGERATED;
            let c = refrigerated_case(&o).unwrap();
            ends_seen[c.end_glass_m.iter().flatten().count()] = true;
            let mut a = Assembly::default();
            let canopy = o.size.y - c.canopy_height_m;
            ends(
                &mut a,
                &o,
                c,
                (Surface::Metal, FIXTURE),
                c.deck_height_m - 0.035,
                canopy,
            );
            let front = o.size.z * 0.5;
            let mid_z = front - o.size.z * c.end_window_depth_fraction * 0.48;
            let sill = c.deck_height_m + c.end_sill_m;
            let mid_y = sill + (canopy - 0.025 - sill) * 0.37;
            for (sign, gauge) in [-1., 1.].into_iter().zip(c.end_glass_m) {
                // Intersect the two end faces along X. Other case parts cannot
                // stand in for an opaque panel that accidentally fills this aperture.
                for (y, expecting_glass) in [(mid_y, gauge.is_some()), (sill - 0.022, false)] {
                    let q = Vec2::new(y, mid_z);
                    let mut faces = Vec::new();
                    for ((surface, label), g) in &a.parts {
                        for tri in g.indices.as_chunks::<3>().0 {
                            let p = tri.map(|idx| Vec3::from_array(g.positions[idx as usize]));
                            if p.iter().any(|v| v.x * sign <= 0.)
                                || (p[0].x - p[1].x).abs() + (p[0].x - p[2].x).abs() > 1e-6
                            {
                                continue;
                            }
                            let [a, b, c] = p.map(|v| v.yz());
                            let sides = [
                                (b - a).perp_dot(q - a),
                                (c - b).perp_dot(q - b),
                                (a - c).perp_dot(q - c),
                            ];
                            if sides.iter().all(|&x| x >= 0.) || sides.iter().all(|&x| x <= 0.) {
                                faces.push((
                                    p[0].x,
                                    *surface,
                                    materials::finish_slot(*surface, label),
                                ));
                            }
                        }
                    }
                    assert_eq!(faces.len(), 2, "seed {seed} side {sign}");
                    let thickness = (faces[0].0 - faces[1].0).abs();
                    if expecting_glass {
                        let gauge = gauge.unwrap();
                        assert!(faces
                            .iter()
                            .all(|&(_, s, f)| s == Surface::ContainerGlass
                                && f == Some(gauge.slot())));
                        assert!((thickness - gauge.metres()).abs() < 1e-6);
                    } else {
                        assert!(faces.iter().all(|&(_, s, _)| s == Surface::Metal));
                        assert!((thickness - c.end_thickness_m).abs() < 1e-6);
                    }
                }
            }
        }
        assert!(ends_seen.into_iter().all(|seen| seen));
    }
}
