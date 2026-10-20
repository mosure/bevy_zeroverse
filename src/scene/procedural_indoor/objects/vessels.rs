//! Continuous hollow vessel profiles with a real base, wall and open mouth.
use super::*;
#[derive(Clone, Debug, serde::Serialize)]
pub struct VesselProgram {
    pub profile: u8,
    pub foot: f32,
    pub belly: f32,
    pub belly_height: f32,
    pub shoulder_height: f32,
    pub neck: f32,
    pub mouth: f32,
    pub flutes: u32,
    pub relief: f32,
    pub wall_fraction: f32,
    pub finish: Surface,
}
pub fn parameters(o: &IndoorObject) -> VesselProgram {
    let mut r = stream(o.seed, 0x56455353454c);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x5645535346494e);
    VesselProgram {
        profile: r.random_range(0..6),
        foot: r.random_range(0.32..0.72),
        belly: r.random_range(0.70..1.0),
        belly_height: r.random_range(0.24..0.58),
        shoulder_height: r.random_range(0.66..0.85),
        neck: r.random_range(0.19..0.68),
        mouth: r.random_range(0.72..1.30),
        flutes: r.random_range(6..=28),
        relief: if r.random_bool(0.38) {
            r.random_range(0.012..0.042)
        } else {
            0.
        },
        wall_fraction: r.random_range(0.025..0.070),
        finish: [
            Surface::Ceramic,
            Surface::Terracotta,
            Surface::ContainerGlass,
            Surface::Metal,
        ][f.random_range(0..4)],
    }
}
pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let s = o.size;
    let bowl = o.kind == ObjectKind::Bowl;
    let r = s.x.min(s.z) * 0.47;
    let knots = if bowl {
        vec![
            (0., p.foot),
            (0.18, p.foot * 1.15),
            (p.belly_height, p.belly),
            (1., 0.95),
        ]
    } else {
        let neck = if p.profile == 1 {
            p.belly * 0.94
        } else {
            p.neck
        };
        let belly = if p.profile == 2 { neck * 1.10 } else { p.belly };
        let waist = if p.profile == 3 { neck * 0.85 } else { belly };
        let mouth = (neck * p.mouth).min(0.98);
        vec![
            (0., p.foot),
            (0.10, p.foot * 1.05),
            (p.belly_height, belly),
            ((p.belly_height + p.shoulder_height) * 0.5, waist),
            (p.shoulder_height, neck),
            (1., mouth),
        ]
    };
    let radius_at = |y: f32| {
        let pair = knots
            .windows(2)
            .find(|v| y <= v[1].0)
            .unwrap_or(&knots[knots.len() - 2..]);
        let t = ((y - pair[0].0) / (pair[1].0 - pair[0].0)).clamp(0., 1.);
        let t = t * t * (3. - 2. * t);
        r * (pair[0].1 + (pair[1].1 - pair[0].1) * t)
    };
    let wall = r * p.wall_fraction;
    let mut profile = Vec::with_capacity(45);
    profile.push((0., 0.));
    for i in 0..=20 {
        let t = i as f32 / 20.;
        profile.push((radius_at(t), s.y * t));
    }
    for i in (1..=20).rev() {
        let t = i as f32 / 20.;
        profile.push(((radius_at(t) - wall).max(wall), s.y * t));
    }
    profile.push((0., s.y * 0.05));
    if p.relief > 0. {
        a.part(p.finish, o.kind.class_name()).fluted_lathe(
            &profile,
            64,
            p.relief,
            p.flutes,
            Transform::IDENTITY,
        );
    } else {
        a.part(p.finish, o.kind.class_name()).lathe(
            &profile,
            if p.profile == 4 { 10 } else { 32 },
            Transform::IDENTITY,
        );
    }
}
