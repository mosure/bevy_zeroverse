use super::*;

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let b = BleacherBank::for_object(o);
    let color = format!("{CLASS}#finish{}", b.seat_slot);
    let mut cuts = TimberCuts::new(o.seed, 0);
    for i in 0..b.rows {
        let front = b.front(i);
        let y = b.deck_height(i);
        b.treads(i, |center, size| {
            a.board(
                &mut cuts,
                (b.deck_stock, CLASS),
                center,
                size,
                0.,
                GrainAxis::X,
            )
        });
        // A nosebeam under each deck carries the half-step on two brackets.
        let beam_z = front - b.pitch * 0.46 - 0.045;
        a.box_part(
            Surface::Metal,
            CLASS,
            Vec3::new(0., y - 0.052 - 0.035, beam_z),
            Vec3::new(b.width - 0.035, 0.070, 0.09),
            0.,
        );
        for side in [-1., 1.] {
            let x = b.aisle_center + side * (b.aisle_width * 0.5 - 0.09);
            let low = y - b.rise * 0.5 - 0.052;
            a.box_part(
                Surface::Metal,
                CLASS,
                Vec3::new(x, low - 0.018, front - b.pitch * 0.28),
                Vec3::new(0.04, 0.036, b.pitch * 0.51),
                0.,
            );
            a.box_part(
                Surface::Metal,
                CLASS,
                Vec3::new(x, (low + y - 0.052) * 0.5, beam_z + 0.056),
                Vec3::new(0.04, b.rise * 0.5, 0.022),
                0.,
            );
        }

        for (lo, hi) in b.spans() {
            let mid = (lo + hi) * 0.5;
            let z = b.seat_z(i);
            let top = y + b.seat_height;
            if b.seat_stock == SeatStock::Timber {
                a.board(
                    &mut cuts,
                    (Surface::Wood, CLASS),
                    Vec3::new(mid, top - 0.022, z),
                    Vec3::new(hi - lo, 0.044, b.seat_depth),
                    0.009,
                    GrainAxis::X,
                );
            } else {
                let slots = b.modules(lo, hi);
                let pitch = (hi - lo) / slots as f32;
                let g = a.part(Surface::Plastic, &color);
                for n in 0..slots {
                    bucket(
                        g,
                        Vec3::new(lo + (n as f32 + 0.5) * pitch, top, z),
                        pitch - 0.012,
                        b.seat_depth,
                    );
                }
            }
            // A continuous steel seat beam and brackets carry the span to the deck.
            let beam_top = top
                - if b.seat_stock == SeatStock::Timber {
                    0.044
                } else {
                    0.060
                };
            a.box_part(
                Surface::Metal,
                CLASS,
                Vec3::new(mid, beam_top - 0.022, z),
                Vec3::new(hi - lo - 0.035, 0.044, 0.085),
                0.,
            );
            let count = ((hi - lo) / 1.02).ceil() as usize;
            for n in 0..=count {
                let x = lo + 0.105 + (hi - lo - 0.21) * n as f32 / count as f32;
                let h = beam_top - 0.044 - y;
                a.box_part(
                    Surface::Metal,
                    CLASS,
                    Vec3::new(x, y + h * 0.5, z),
                    Vec3::new(0.036, h, 0.064),
                    0.,
                );
            }
        }
        // Under each section, runners transfer load to the floor. Staggering the
        // leg depth makes the retractable frames visibly nest from the ends.
        let supports = (b.width / 2.1).ceil() as usize;
        let z = front - b.pitch * 0.71;
        for n in 0..=supports {
            let x = (n as f32 / supports as f32 - 0.5) * (b.width - 0.19 - 0.032 * i as f32);
            let h = y - 0.052 - 0.038;
            a.box_part(
                Surface::Metal,
                CLASS,
                Vec3::new(x, 0.038 + h * 0.5, z),
                Vec3::new(0.055, h, 0.059),
                0.,
            );
            a.box_part(
                Surface::Metal,
                CLASS,
                Vec3::new(x, 0.019, front - 0.56 * b.pitch),
                Vec3::new(0.069, 0.038, 0.69 * b.pitch),
                0.,
            );
        }
    }
    guards(a.part(Surface::Metal, CLASS), b);
}

// Low-cost closed moulding. The centre is dished between raised sides and a
// curved forward lip. The three-dimensional profile reads at oblique angles.
fn bucket(g: &mut Geometry, p: Vec3, width: f32, depth: f32) {
    let zs = [-0.5, -0.31, 0.16, 0.40, 0.5];
    let dys = [0., -0.010, -0.012, -0.021, -0.042];
    let point = |x: usize, j: usize| {
        p + Vec3::new(
            (x as f32 - 1.) * width * 0.5,
            dys[j] + if x == 1 { -0.007 } else { 0. },
            zs[j] * depth,
        )
    };
    let bottom = |v: Vec3| Vec3::new(v.x, p.y - 0.060, v.z);
    for j in 0..4 {
        for x in 0..2 {
            g.metric_quad(
                [
                    point(x, j),
                    point(x + 1, j),
                    point(x + 1, j + 1),
                    point(x, j + 1),
                ],
                Vec3::Y,
            );
        }
        for (x, normal) in [(0, -Vec3::X), (2, Vec3::X)] {
            let a = point(x, j);
            let b = point(x, j + 1);
            g.metric_quad([a, b, bottom(b), bottom(a)], normal);
        }
    }
    for (j, normal) in [(0, -Vec3::Z), (4, Vec3::Z)] {
        for x in 0..2 {
            let a = point(x, j);
            let b = point(x + 1, j);
            g.metric_quad([a, b, bottom(b), bottom(a)], normal);
        }
    }
    g.metric_quad(
        [
            bottom(point(0, 0)),
            bottom(point(2, 0)),
            bottom(point(2, 4)),
            bottom(point(0, 4)),
        ],
        -Vec3::Y,
    );
}

fn guards(g: &mut Geometry, b: BleacherBank) {
    let rear = b.front(b.rows) + 0.026;
    let y = b.deck_height(b.rows - 1);
    let radius = 0.021;
    let top = y + GUARD_HEIGHT - radius;
    let half = b.width * 0.5 - 0.026;
    let posts = (b.width / 1.35).ceil() as usize;
    for i in 0..=posts {
        let x = half * (2. * i as f32 / posts as f32 - 1.);
        g.tube(&[Vec3::new(x, y, rear), Vec3::new(x, top, rear)], radius, 6);
    }
    for rail in [top, top - 0.32, top - 0.61] {
        g.tube(
            &[Vec3::new(-half, rail, rear), Vec3::new(half, rail, rear)],
            radius,
            6,
        );
    }
    for x in [-half, half] {
        let first_z = b.front(0) - 0.12;
        let first = Vec3::new(x, b.deck_height(0), first_z);
        let upper = Vec3::new(x, top, rear);
        let lower = first + Vec3::Y * (GUARD_HEIGHT - radius);
        g.tube(&[first, lower], radius, 6);
        for drop in [0., 0.34, 0.64] {
            g.tube(&[lower - Vec3::Y * drop, upper - Vec3::Y * drop], radius, 6);
        }
    }
}
