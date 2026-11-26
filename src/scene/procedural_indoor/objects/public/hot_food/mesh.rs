//! The deck and pan lips tile the chassis top. Rays through an open pan hit
//! its contents or recessed floor. There is no solid worktop behind the food.
use super::*;
use crate::scene::procedural_indoor::geometry::housing_outline;

pub(super) const STEEL: &str = "appliance#finish111#uv_local";
pub(super) const LIP: f32 = 0.004;
const EXPONENT: f32 = 8.;

fn at(p: Vec2, y: f32) -> Vec3 {
    Vec3::new(p.x, y, p.y)
}

fn deck_strip(g: &mut Geometry, lo: Vec2, hi: Vec2, y: f32) {
    if (hi - lo).min_element() <= 0.00001 {
        return;
    }
    let first = g.positions.len();
    g.metric_quad(
        [
            at(lo, y),
            at(Vec2::new(lo.x, hi.y), y),
            at(hi, y),
            at(Vec2::new(hi.x, lo.y), y),
        ],
        Vec3::Y,
    );
    for (p, uv) in g.positions[first..].iter().zip(&mut g.uvs[first..]) {
        *uv = [p[0], p[2]];
    }
}

fn cap(
    g: &mut Geometry,
    pan: &Pan,
    half: Vec2,
    y: f32,
    outline: &[Vec2],
    mut uv: impl FnMut(Vec2) -> [f32; 2],
) {
    let base = g.positions.len() as u32;
    for v in std::iter::once(Vec2::ZERO).chain(outline.iter().copied()) {
        g.positions.push(at(pan.center + half * v, y).to_array());
        g.normals.push(Vec3::Y.to_array());
        g.uvs.push(uv(v));
    }
    for i in 0..outline.len() as u32 {
        g.indices.extend([
            base,
            base + 1 + (i + 1) % outline.len() as u32,
            base + 1 + i,
        ]);
    }
}

fn pan(a: &mut Assembly, p: &Pan, deck: f32, outline: &[Vec2; 32]) {
    let g = a.part(Surface::Metal, STEEL);
    let bottom = p.floor(deck);
    let lip = deck + LIP;
    let ring = |v: Vec2| {
        [
            at(p.center + 0.5 * p.size * (v / v.abs().max_element()), deck),
            at(p.center + p.outer() * v, deck),
            at(p.center + p.outer() * v, lip),
            at(p.center + p.inner() * v, lip),
            at(p.center + p.bottom() * v, bottom),
        ]
    };
    let mut prev = ring(outline[31]);
    for &dir in outline {
        let next = ring(dir);
        let out = at(dir, 0.);
        for (i, normal) in [Vec3::Y, out, Vec3::Y, -out].into_iter().enumerate() {
            g.metric_quad([prev[i], next[i], next[i + 1], prev[i + 1]], normal);
        }
        prev = next;
    }
    cap(g, p, p.bottom(), bottom, outline, |v| {
        (p.center + p.bottom() * v).to_array()
    });
    if p.contents == Contents::Covered {
        // Nest the underside below the rim. Only the rolled shoulder bears
        // on the lip, avoiding a second visible coplanar deck.
        let outer = p.outer() - Vec2::splat(0.0008);
        let tf = Transform::from_translation(at(p.center, lip));
        g.housing(
            &[
                (-0.002, p.inner() - Vec2::splat(0.001)),
                (0.0015, outer),
                (0.004, outer),
                (0.009, outer * 0.80),
            ],
            EXPONENT,
            tf,
        );
        let w = p.size.x.min(0.24) * 0.22;
        let handles = [
            Vec3::new(-w, 0.009, 0.),
            Vec3::new(-w, 0.022, 0.),
            Vec3::new(-w * 0.84, 0.031, 0.),
            Vec3::new(w * 0.84, 0.031, 0.),
            Vec3::new(w, 0.022, 0.),
            Vec3::new(w, 0.009, 0.),
        ];
        g.tube(&handles.map(|v| tf.transform_point(v)), 0.0038, 8);
    } else if let Some(recipe) = p.contents.soup() {
        let y = bottom + (lip - bottom) * p.fill_fraction;
        // Use the same tessellated circumference as the bowl at this height.
        let half = p.bottom().lerp(p.inner(), p.fill_fraction);
        let mut r = stream(p.seed, u64::from_le_bytes(*b"LadleMap"));
        let span = r.random_range(0.46..0.78);
        let offset = Vec2::new(r.random_range(-0.08..0.08), r.random_range(-0.08..0.08));
        let sign = if r.random_bool(0.5) { 1. } else { -1. };
        cap(
            a.part(Surface::Ceramic, broth::LABEL),
            p,
            half,
            y,
            outline,
            |v| recipe.uv(Vec2::splat(0.5) + offset + span * 0.5 * Vec2::new(v.x * sign, v.y)),
        );
    } else if p.contents == Contents::PotatoGratin {
        gratin::build(a, p, deck, outline);
    } else if matches!(
        p.contents,
        Contents::RoastPotatoes | Contents::GlazedCarrots
    ) {
        produce::build(a, p, bottom);
    }
}

pub(in super::super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let plan = HotFoodWell::for_object(o);
    let h = plan.deck_height;
    let half = Vec2::new(o.size.x * 0.5, o.size.z * 0.5 - 0.016);
    let foot = 0.019;
    // The folded chassis has one exterior skin. The deck tiles its open top.
    let corners = [
        -half,
        Vec2::new(-half.x, half.y),
        half,
        Vec2::new(half.x, -half.y),
    ];
    let g = a.part(Surface::Metal, STEEL);
    g.metric_quad(corners.map(|p| at(p, foot)), Vec3::NEG_Y);
    for i in 0..4 {
        let (p, q) = (corners[i], corners[(i + 1) % 4]);
        g.metric_quad(
            [at(p, foot), at(q, foot), at(q, h), at(p, h)],
            at((p + q) * 0.5, 0.),
        );
    }
    for sign in [-1., 1.] {
        for other in [-1., 1.] {
            a.part(Surface::Rubber, "appliance").cylinder(
                0.018,
                foot,
                Transform::from_xyz(
                    sign * (half.x - 0.034),
                    foot * 0.5,
                    other * (half.y - 0.036),
                ),
            );
        }
    }
    let top = half;
    let pitch = o.size.x / plan.wells as f32;
    let gn = GN.min(Vec2::new(pitch - 0.018, o.size.z - 0.060));
    let g = a.part(Surface::Metal, STEEL);
    deck_strip(g, -top, Vec2::new(top.x, -gn.y * 0.5), h);
    deck_strip(g, Vec2::new(-top.x, gn.y * 0.5), top, h);
    let mut edge = -top.x;
    for n in 0..plan.wells {
        let center = (n as f32 + 0.5) * pitch - half.x;
        deck_strip(
            g,
            Vec2::new(edge, -gn.y * 0.5),
            Vec2::new(center - gn.x * 0.5, gn.y * 0.5),
            h,
        );
        edge = center + gn.x * 0.5;
    }
    deck_strip(
        g,
        Vec2::new(edge, -gn.y * 0.5),
        Vec2::new(top.x, gn.y * 0.5),
        h,
    );
    let outline = housing_outline(EXPONENT);
    for p in &plan.pans {
        pan(a, p, h, &outline);
    }
    let back = -half.y;
    for n in 0..plan.wells {
        let x = (n as f32 + 0.5) * pitch - half.x;
        let axis = Quat::from_rotation_x(-std::f32::consts::FRAC_PI_2);
        a.part(Surface::Rubber, "control_panel").cylinder(
            0.022,
            0.014,
            Transform::from_xyz(x, h * 0.60, back - 0.007).with_rotation(axis),
        );
        a.box_part(
            Surface::Plastic,
            "control_panel",
            Vec3::new(x + 0.054, h * 0.60, back - 0.003),
            Vec3::new(0.016, 0.025, 0.006),
            0.002,
        );
    }
    let gland = power_port(o);
    a.part(Surface::Rubber, "appliance").cylinder(
        0.009,
        0.010,
        Transform::from_translation(gland + Vec3::Z * 0.005)
            .with_rotation(Quat::from_rotation_x(FRAC_PI_2)),
    );
    if let Some(guard) = plan.guard {
        let lower = Vec3::new(0., h + guard.clearance, half.y - 0.012);
        // The finished upper edge, including the pane thickness, stays inside
        // the admitted envelope. The staff side stays open for replacing pans.
        let upper = Vec3::new(0., o.size.y - 0.004, lower.z - guard.setback);
        let travel = upper - lower;
        let tilt = Quat::from_rotation_x(travel.z.atan2(travel.y));
        let gauge = if guard.thickness < 0.0055 {
            "appliance#finish65"
        } else {
            "appliance#finish66"
        };
        a.part(Surface::ContainerGlass, gauge).cuboid(
            Vec3::new(o.size.x - 0.054, travel.length(), guard.thickness),
            0.,
            Transform::from_translation((lower + upper) * 0.5).with_rotation(tilt),
        );
        for side in [-1., 1.] {
            let x = side * (half.x - 0.009);
            let post = Vec3::new(x, h, lower.z - guard.setback * 0.14);
            let mid = lower.lerp(upper, 0.14) + Vec3::X * x;
            let high = lower.lerp(upper, 0.88) + Vec3::X * x;
            let steel = a.part(Surface::Metal, STEEL);
            steel.rod(post, mid, 0.006);
            steel.rod(mid, high, 0.006);
            for tip in [mid, high] {
                steel.rod(tip, tip - Vec3::X * side * 0.025, 0.004);
            }
        }
    }
}
