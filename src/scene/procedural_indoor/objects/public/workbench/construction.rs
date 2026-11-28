use super::*;

pub(super) fn base(a: &mut Assembly, o: &IndoorObject, p: &Bench, cuts: &mut TimberCuts) {
    let label = o.kind.class_name();
    let top = o.size.y - p.stock_m;
    let d = p.deck(o);
    let leg = p.post_m;
    let x = d.width() * 0.5 - p.leg_setback_m;
    let rear = d.min.y + leg * 0.5 + 0.045;
    let front = d.max.y - leg * 0.5 - 0.067;
    let z = (front + rear) * 0.5;
    let bay = front - rear - leg;
    let foot = 0.016;
    for xx in [-x, x] {
        for zz in [rear, front] {
            a.box_part(
                Surface::Rubber,
                label,
                Vec3::new(xx, foot * 0.5, zz),
                Vec3::new(leg + 0.003, foot, leg + 0.003),
                0.002,
            );
            a.board(
                cuts,
                (p.frame_finish, label),
                Vec3::new(xx, (top + foot) * 0.5, zz),
                Vec3::new(leg, top - foot, leg),
                0.0023,
                GrainAxis::Y,
            );
        }
    }
    let rail = leg * 0.76;
    let stretcher = p.shelf_y_m.unwrap_or(0.22);
    for h in [top - 0.043, stretcher] {
        for xx in [-x, x] {
            a.board(
                cuts,
                (p.frame_finish, label),
                Vec3::new(xx, h, z),
                Vec3::new(rail, 0.078, bay),
                0.0018,
                GrainAxis::Z,
            );
        }
        for zz in [rear, front] {
            a.board(
                cuts,
                (p.frame_finish, label),
                Vec3::new(0., h, zz),
                Vec3::new(2. * x - leg, 0.078, rail),
                0.0018,
                GrainAxis::X,
            );
        }
    }
    if p.shelf_y_m.is_some() {
        let length = 2. * x - leg - 0.004;
        let depth = front - rear;
        let count = if p.slatted_shelf {
            (length / 0.13).ceil() as usize
        } else {
            1
        };
        let pitch = length / count as f32;
        for k in 0..count {
            let at = -length * 0.5 + pitch * (k as f32 + 0.5);
            a.board(
                cuts,
                (
                    if p.slatted_shelf {
                        Surface::Wood
                    } else {
                        p.frame_finish
                    },
                    label,
                ),
                Vec3::new(at, stretcher + 0.039 + 0.009, z),
                Vec3::new(pitch - if count > 1 { 0.014 } else { 0. }, 0.018, depth),
                0.0014,
                if count > 1 {
                    GrainAxis::Z
                } else {
                    GrainAxis::X
                },
            );
        }
    }
}

pub(super) fn vise(
    a: &mut Assembly,
    o: &IndoorObject,
    p: &Bench,
    v: FaceVise,
    cuts: &mut TimberCuts,
) {
    let l = o.kind.class_name();
    let (y, z, x) = (o.size.y, p.front_z_m, v.center_x_m);
    // The fixed chop extends the front lip of the top down to the guides.
    let cheek_bottom = y - 0.175;
    let fixed_height = y - p.stock_m - cheek_bottom;
    a.board(
        cuts,
        (Surface::Wood, l),
        Vec3::new(x, cheek_bottom + fixed_height * 0.5, z - 0.021),
        Vec3::new(v.jaw_width_m + 0.025, fixed_height, 0.042),
        0.001,
        GrainAxis::X,
    );
    let outer = z + v.opening_m + v.jaw_thickness_m;
    a.board(
        cuts,
        (Surface::Wood, l),
        Vec3::new(x, cheek_bottom + 0.085, outer - v.jaw_thickness_m * 0.5),
        Vec3::new(v.jaw_width_m, 0.17, v.jaw_thickness_m),
        0.0015,
        GrainAxis::X,
    );
    let metal = a.part(Surface::Metal, l);
    let screw_y = y - 0.139;
    for (xx, yy, r) in [
        (x, screw_y, 0.012),
        (x - v.jaw_width_m * 0.32, y - 0.097, 0.008),
        (x + v.jaw_width_m * 0.32, y - 0.097, 0.008),
    ] {
        metal.rod(
            Vec3::new(xx, yy, z - 0.003),
            Vec3::new(xx, yy, z + v.opening_m + 0.003),
            r,
        );
    }
    metal.rod(
        Vec3::new(x, screw_y, outer - 0.002),
        Vec3::new(x, screw_y, outer + 0.048),
        0.021,
    );
    let (sn, cs) = v.handle_angle_rad.sin_cos();
    let axis = Vec3::new(cs, sn, 0.);
    let hub = Vec3::new(x, screw_y, outer + 0.033);
    let center = hub + axis * v.handle_slide_m;
    let ends = [center - axis * 0.103, center + axis * 0.103];
    metal.rod(ends[0], ends[1], 0.0063);
    for (end, sign) in [(ends[0], -1.), (ends[1], 1.)] {
        metal.rod(end - axis * sign * 0.002, end + axis * sign * 0.009, 0.010);
    }
}
