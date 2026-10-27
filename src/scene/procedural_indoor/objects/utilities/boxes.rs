//! Open and lidded storage with real cavities, attached handles and fitted seams.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub struct BoxProgram {
    pub body: Vec3,
    pub taper: f32,
    pub roundness: f32,
    pub wall: f32,
    pub floor: f32,
    pub lid: f32,
    pub lid_overhang: f32,
    pub seam: f32,
    pub handle_height: f32,
    pub handle_span: f32,
    pub handle_drop: f32,
    pub handle_reach: f32,
    pub handle_radius: f32,
    pub handles: bool,
    pub latches: bool,
    pub finish: Surface,
    pub lid_finish: Surface,
    pub hardware: Surface,
}

pub fn parameters(o: &IndoorObject) -> BoxProgram {
    let mut r = stream(o.seed, 0x424f5850524f4752);
    let mut f = stream(o.finish_seed.unwrap_or(o.seed), 0x424f5846494e4953);
    let s = o.size;
    let seam = s.y * r.random_range(0.004..0.012);
    let lid = if r.random_bool(0.64) {
        s.y * r.random_range(0.035..0.14)
    } else {
        0.
    };
    let body = Vec3::new(
        s.x * r.random_range(0.81..0.88),
        s.y - lid - if lid > 0. { seam } else { 0. },
        s.z * r.random_range(0.85..0.94),
    );
    let finish = [
        Surface::Plastic,
        Surface::Paper,
        Surface::FabricAlt,
        Surface::Wood,
    ][f.random_range(0..4)];
    BoxProgram {
        body,
        taper: r.random_range(0.74..0.99),
        roundness: r.random_range(3.5..12.),
        wall: body.min_element() * r.random_range(0.018..0.040),
        floor: body.y * r.random_range(0.024..0.060),
        lid,
        lid_overhang: r.random_range(1.018..1.05),
        seam,
        handle_height: r.random_range(0.60..0.82),
        handle_span: body.z * r.random_range(0.32..0.56),
        handle_drop: body.y * r.random_range(0.06..0.16),
        handle_reach: s.x * r.random_range(0.025..0.048),
        handle_radius: s.min_element() * r.random_range(0.011..0.020),
        handles: r.random_bool(0.72),
        latches: lid > 0. && r.random_bool(0.40),
        finish,
        lid_finish: if f.random_bool(0.74) {
            finish
        } else {
            Surface::Plastic
        },
        hardware: [Surface::Metal, Surface::Rubber, Surface::Leather][f.random_range(0..3)],
    }
}

impl BoxProgram {
    pub fn half_at(&self, y: f32) -> Vec2 {
        self.body.xz() * 0.5 * (self.taper + (1. - self.taper) * y / self.body.y)
    }
    /// Attachments follow the actual rounded wall, not its bounding rectangle.
    pub fn handle_path(&self, side: f32) -> Vec<Vec3> {
        let y = self.body.y * self.handle_height;
        let half = self.half_at(y);
        (0..=12)
            .map(|i| {
                let t = i as f32 / 12.;
                let z = (t - 0.5) * self.handle_span;
                let x = half.x
                    * (1. - (z / half.y).abs().powf(self.roundness)).powf(1. / self.roundness);
                let curve = (PI * t).sin();
                Vec3::new(
                    side * (x + self.handle_reach * curve),
                    y - self.handle_drop * curve,
                    z,
                )
            })
            .collect()
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject) {
    let p = parameters(o);
    let l = o.kind.class_name();
    let top = p.body.xz() * 0.5;
    let bottom = top * p.taper;
    // The reversed inner rings end at the cavity floor. housing() caps only
    // the underside and cavity floor, leaving the mouth genuinely open.
    a.part(p.finish, l).housing(
        &[
            (0., bottom),
            (p.body.y, top),
            (p.body.y, top - Vec2::splat(p.wall)),
            (p.floor, p.half_at(p.floor) - Vec2::splat(p.wall)),
        ],
        p.roundness,
        Transform::IDENTITY,
    );
    if p.lid > 0. {
        let y = p.body.y + p.seam;
        let h = top * p.lid_overhang;
        // A recessed underside rests on the wall rim and connects to the cap.
        // The visible seam must not imply an unsupported floating lid.
        a.part(p.lid_finish, l).housing(
            &[
                (p.body.y, top - Vec2::splat(p.wall * 0.35)),
                (y + p.lid * 0.20, top - Vec2::splat(p.wall * 0.35)),
            ],
            p.roundness,
            Transform::IDENTITY,
        );
        a.part(p.lid_finish, l).housing(
            &[(y, h * 0.98), (y + p.lid * 0.75, h), (y + p.lid, h * 0.98)],
            p.roundness,
            Transform::IDENTITY,
        );
        if p.latches {
            for side in [-1., 1.] {
                a.box_part(
                    p.hardware,
                    l,
                    Vec3::new(
                        side * p.body.x * 0.27,
                        p.body.y + p.seam * 0.5,
                        top.y * 0.99,
                    ),
                    Vec3::new(
                        p.body.x * 0.085,
                        p.lid * 0.95 + p.seam * 2.,
                        p.body.z * 0.045,
                    ),
                    p.wall * 0.25,
                );
            }
        }
    }
    if p.handles {
        for side in [-1., 1.] {
            a.part(p.hardware, l)
                .tube(&p.handle_path(side), p.handle_radius, 8);
        }
    }
}
