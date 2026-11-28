use super::*;

pub(super) fn build(a: &mut Assembly, p: &BasketStack) {
    let resin = format!("{CLASS}#finish{}", p.dye_slot);
    let plated = p.shell == BasketShell::WeldedWire && !p.coated_wire;
    let stock = if plated {
        (Surface::Metal, PLATING)
    } else {
        (Surface::Plastic, resin.as_str())
    };
    if p.carrier != Carrier::Floor {
        dolly(a, p);
    }
    // Every layer is modeled, including the bottom behind the apertures of its
    // neighbor. Different cameras can see through different parts of the stack.
    let g = a.part(stock.0, stock.1);
    let v0 = g.positions.len();
    let i0 = g.indices.len();
    let y = p.rail_top_m;
    match p.shell {
        BasketShell::Molded => shell::molded(g, p, y),
        BasketShell::WeldedWire => shell::wire(g, p, y),
    }
    let top = y + p.body_height_m;
    // Shoulders rest on the rim underneath at exactly one nesting pitch.
    let foot = top - p.nesting_pitch_m;
    let height = 0.011_f32.min(p.nesting_pitch_m * 0.40);
    let t = 1. - (p.nesting_pitch_m - height * 0.5) / p.body_height_m;
    let half = p.bottom_half_m.lerp(p.rim_half_m, t);
    let n = p.openings[0] as f32;
    let (rib, inner) = match p.shell {
        BasketShell::Molded => (1. - 2. / n, half.y - p.wall_m),
        // Weld tabs onto a real rising wire. Both end walls have n+2 rods.
        BasketShell::WeldedWire => (1. - 2. / (n + 1.), half.y - p.wall_m * 1.8),
    };
    let x = rib * (1. - p.chamfer_fraction) * half.x;
    let outer = p.rim_half_m.y + 0.0008;
    for sx in [-1., 1.] {
        for sz in [-1., 1.] {
            g.cuboid(
                Vec3::new(0.011, height, outer - inner),
                0.001,
                Transform::from_xyz(sx * x, foot + height * 0.5, sz * (outer + inner) * 0.5),
            );
        }
    }
    // The manufactured shells match exactly. Tessellate and shade the first
    // once, then append translated vertices to the same semantic/material batch.
    let v1 = g.positions.len();
    let i1 = g.indices.len();
    let vertices = v1 - v0;
    g.positions.reserve((p.count - 1) * vertices);
    g.normals.reserve((p.count - 1) * vertices);
    g.uvs.reserve((p.count - 1) * vertices);
    g.indices.reserve((p.count - 1) * (i1 - i0));
    for layer in 1..p.count {
        let dy = layer as f32 * p.nesting_pitch_m;
        for idx in v0..v1 {
            let [x, y, z] = g.positions[idx];
            g.positions.push([x, y + dy, z]);
        }
        g.normals.extend_from_within(v0..v1);
        g.uvs.extend_from_within(v0..v1);
        let offset = (vertices * layer) as u32;
        for idx in i0..i1 {
            g.indices.push(g.indices[idx] + offset);
        }
    }
    for layer in 0..p.count {
        let top = p.bottom_m(layer) + p.body_height_m;
        let angles = if layer + 1 == p.count {
            p.top_handle_angles_rad
        } else {
            [0.; 2]
        };
        for (i, angle) in angles.into_iter().enumerate() {
            handle(a, p, top, if i == 0 { -1. } else { 1. }, angle);
        }
    }
}

fn handle(a: &mut Assembly, p: &BasketStack, top: f32, side: f32, angle: f32) {
    let pivot_y = top - 0.0075;
    let x = p.rim_half_m.x + 0.007;
    let pivot_z = side * 0.011;
    let (s, c) = angle.sin_cos();
    let reach = p.handle_reach();
    let point = |xx: f32, t: f32| Vec3::new(xx, pivot_y + s * t, pivot_z + side * c * t);
    let (surface, radius) = if p.shell == BasketShell::Molded {
        (Surface::Ink, 0.0044)
    } else {
        (Surface::Chrome, 0.0029)
    };
    for sign in [-1., 1.] {
        a.part(surface, CLASS).cylinder(
            0.0075,
            0.014,
            Transform::from_xyz(sign * (x - 0.005), pivot_y, pivot_z)
                .with_rotation(Quat::from_rotation_z(FRAC_PI_2)),
        );
    }
    a.part(surface, CLASS).tube(
        &[
            point(-x, 0.),
            point(-x, reach - 0.010),
            point(-x + 0.010, reach),
            point(x - 0.010, reach),
            point(x, reach - 0.010),
            point(x, 0.),
        ],
        radius,
        8,
    );
    // An overmolded palm grip shares the moving crossbar centerline.
    a.part(Surface::Rubber, CLASS).tube(
        &[point(-x * 0.34, reach), point(x * 0.34, reach)],
        0.0065,
        8,
    );
}

fn dolly(a: &mut Assembly, p: &BasketStack) {
    let h = p.rail_top_m;
    let (x, z) = (p.rim_half_m.x + 0.014, p.rim_half_m.y + 0.025);
    let frame = (Surface::Metal, "retail_fixture#finish111#uv_local");
    for sign in [-1., 1.] {
        a.box_part(
            frame.0,
            frame.1,
            Vec3::new(0., h - 0.007, sign * z),
            Vec3::new(2. * x + 0.012, 0.014, 0.014),
            0.002,
        );
        a.box_part(
            frame.0,
            frame.1,
            Vec3::new(sign * x, h - 0.007, 0.),
            Vec3::new(0.012, 0.014, 2. * z - 0.014),
            0.002,
        );
        // The bearers span the small footprint at the base of a tapered basket.
        a.box_part(
            frame.0,
            frame.1,
            Vec3::new(sign * p.bottom_half_m.x * 0.56, h - 0.006, 0.),
            Vec3::new(0.023, 0.012, 2. * z - 0.014),
            0.0015,
        );
    }
    let r = p.wheel_radius_m;
    for sx in [-1., 1.] {
        for sz in [-1., 1.] {
            let center = Vec3::new(sx * (x - r - 0.006), r, sz * (z - r - 0.008));
            let axle = Quat::from_rotation_x(FRAC_PI_2);
            a.part(Surface::Rubber, CADDY).cylinder(
                r,
                0.016,
                Transform::from_translation(center).with_rotation(axle),
            );
            for side in [-1., 1.] {
                a.box_part(
                    frame.0,
                    frame.1,
                    center + Vec3::new(0., (h - r) * 0.5 - 0.006, side * 0.010),
                    Vec3::new(0.014, h - r - 0.005, 0.003),
                    0.001,
                );
            }
            // Bracket plate joins the perimeter rail to the two caster cheeks.
            let near = sz * (z - 0.007);
            a.box_part(
                frame.0,
                frame.1,
                Vec3::new(center.x, h - 0.018, (center.z + near) * 0.5),
                Vec3::new(0.027, 0.010, (near - center.z).abs() + 0.018),
                0.002,
            );
        }
    }
    if p.carrier == Carrier::PushDolly {
        let y = p.push_handle_height_m - 0.010;
        let back = -z;
        a.part(frame.0, frame.1).tube(
            &[
                Vec3::new(-x, h - 0.008, back),
                Vec3::new(-x, y - 0.028, back),
                Vec3::new(-x + 0.030, y, back),
                Vec3::new(x - 0.030, y, back),
                Vec3::new(x, y - 0.028, back),
                Vec3::new(x, h - 0.008, back),
            ],
            0.007,
            10,
        );
        a.part(Surface::Rubber, CADDY).tube(
            &[Vec3::new(-x * 0.69, y, back), Vec3::new(x * 0.69, y, back)],
            0.010,
            10,
        );
    }
}
