use super::*;

pub(super) struct Riser {
    pub start: Vec3,
    pub elbow: Vec3,
    pub hinge: Option<Vec3>,
    pub attachment: Vec3,
    pub depth_radius: f32,
}

pub(super) fn fit(
    o: &IndoorObject,
    p: &DisplayProgram,
    d: &DisplayGeometry,
    section: Option<&enclosure::PanelSection>,
    foot: Vec3,
    radius: f32,
) -> Riser {
    let tf = d.panel_transform;
    let x = d.width_m * p.mount_offset;
    let sag = |x: f32| {
        let r = d.curvature_radius_m;
        if r > 0. {
            r - (r * r - x * x).sqrt()
        } else {
            0.
        }
    };
    let half = d.panel_m * 0.5;
    let back = tf.transform_point(Vec3::new(x, -d.height_m * 0.22, sag(x) - half + 0.0006));
    let Some(section) = section else {
        return Riser {
            start: foot,
            elbow: Vec3::new(
                foot.x,
                back.y * p.mount.elbow_fraction,
                tf.translation
                    .z
                    .clamp(-o.size.z * 0.5 + radius * 2., o.size.z * 0.5 - radius * 2.),
            ),
            hinge: None,
            attachment: back,
            depth_radius: radius,
        };
    };

    // A diagonally cut rectangular post can project its half-diagonal below
    // the centre of a thin base. Seat it inside the foot above the desk plane.
    let start = foot.with_y(foot.y.max(radius * 1.3));
    let up = tf.rotation * Vec3::Y;
    let front = tf.rotation * Vec3::Z;
    // Every point on a curved panel lies forward of its tangent at the centre.
    // Reserve space for the entire post even when the panel leans over it.
    let rear_z = tf.translation.z - up.z.abs() * d.height_m * 0.5 - front.z * half;
    let available = rear_z + o.size.z * 0.5;
    if available < 0.003
        || p.mount.elbow_fraction < 0.36
        || (p.mount.riser_style == 0 && available < 3.8 * radius)
    {
        // Integrated pedestal socket in the underside of the front shell.
        let shell_depth = d.panel_m * section.front_shell_fraction;
        let z = sag(x) + half - (shell_depth + section.edge_bevel_m) * 0.5;
        let attachment = tf.transform_point(Vec3::new(x, -d.height_m * 0.5 + 0.0006, z));
        // Enter normal to the underside. A diagonal stem can cross several
        // millimetres of bezel before reaching the same recessed fixing point.
        let depth_radius = radius.min((shell_depth - section.edge_bevel_m) * 0.45);
        let drop = attachment.y * (1. - p.mount.elbow_fraction) / up.y;
        let room = o.size.z * 0.5 + attachment.z * up.z.signum() - 1.6 * depth_radius;
        let drop = if up.z.abs() > 1e-5 {
            drop.min(room / up.z.abs())
        } else {
            drop
        };
        return Riser {
            start,
            elbow: attachment - up * drop,
            hinge: None,
            attachment,
            depth_radius,
        };
    }
    // Molded blades can be broad across the desk while staying shallow behind
    // the display. The final short standoff runs forward to the fixing point.
    let depth_radius = radius.min(available / 3.8);
    let post_z = rear_z - 1.9 * depth_radius;
    let lowest_y = tf.translation.y
        - up.y * d.height_m * 0.5
        - front.y.abs() * (sag(d.width_m * 0.5) + half + 0.0013);
    Riser {
        start,
        elbow: Vec3::new(
            foot.x,
            (back.y * p.mount.elbow_fraction).min(lowest_y - 2. * radius),
            post_z,
        ),
        hinge: Some(Vec3::new(back.x, back.y, post_z)),
        attachment: back,
        depth_radius,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hit(from: Vec3, to: Vec3, tri: [Vec3; 3]) -> Option<Vec3> {
        let [a, b, c] = tri;
        let ray = to - from;
        let u = b - a;
        let v = c - a;
        let p = ray.cross(v);
        let det = u.dot(p);
        if det.abs() < 1e-12 {
            return None;
        }
        let delta = from - a;
        let q = delta.cross(u);
        let s = delta.dot(p) / det;
        let t = ray.dot(q) / det;
        let along = v.dot(q) / det;
        (s >= 0. && t >= 0. && s + t <= 1. && (0.0..=1.).contains(&along))
            .then(|| from + along * ray)
    }

    #[test]
    fn the_stand_meets_the_emitted_shell_only_at_its_mounting_socket() {
        let mut routes = [0; 2];
        for seed in (0..128).chain([8305910593332966396_u64]) {
            let mut o = audit::fixture(ObjectKind::Monitor, seed);
            if seed == 8305910593332966396 {
                o.size = Vec3::new(0.43913418, 0.30333877, 0.19676454);
            }
            let p = display_parameters(&o);
            let d = geometry_parameters(&o, &p);
            let mut housing = Assembly::default();
            let section = enclosure::build(&mut housing, &o, &p, &d);
            let h = p.mount.base_height;
            let radius = p.mount.riser_radius.min(h * 0.46);
            let span = Vec2::new(d.width_m * p.stand_width, o.size.z * p.stand_depth);
            let foot_z = match p.stand {
                1 => -span.y * 0.37,
                5 => -(span.y - span.min_element() * p.mount.rail_fraction) * 0.5,
                _ => 0.,
            };
            let foot = Vec3::new(0., h * 0.5, foot_z);
            let r = fit(&o, &p, &d, Some(&section), foot, radius);
            assert!(r.depth_radius > 0.);
            routes[usize::from(r.hinge.is_some())] += 1;
            let mut origin = r.start;
            let mut hits = 0;
            for end in [Some(r.elbow), r.hinge, Some(r.attachment)]
                .into_iter()
                .flatten()
            {
                for ((surface, _), g) in &housing.parts {
                    for tri in g.indices.as_chunks::<3>().0 {
                        let tri = tri.map(|i| Vec3::from_array(g.positions[i as usize]));
                        if let Some(point) = hit(origin, end, tri) {
                            assert_eq!(*surface, p.casing);
                            assert!(point.distance(r.attachment) < 0.002, "{seed}: {point:?}");
                            hits += 1;
                        }
                    }
                }
                origin = end;
            }
            assert_eq!(
                hits, 1,
                "stand is disconnected or pierces the shell in seed {seed}"
            );
        }
        assert!(routes.into_iter().all(|count| count >= 16));
    }
}
