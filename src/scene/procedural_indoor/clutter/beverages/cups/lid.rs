//! Snap-on takeaway lid with a formed skirt and an off-axis drinking aperture.
use super::*;

#[derive(Clone, Debug, serde::Serialize)]
pub(crate) struct SipLid {
    pub rise_m: f32,
    pub gauge_m: f32,
    pub deck_fraction: f32,
    pub port_offset_fraction: f32,
    pub port_half_fraction: Vec2,
}

impl SipLid {
    pub(super) fn sample(r: &mut impl Rng, height: f32) -> Self {
        Self {
            rise_m: (height * r.random_range(0.055..0.115)).min(0.017),
            gauge_m: r.random_range(0.00050..0.00094),
            deck_fraction: r.random_range(0.77..0.88),
            port_offset_fraction: r.random_range(0.48..0.73),
            port_half_fraction: Vec2::new(r.random_range(0.17..0.29), r.random_range(0.055..0.105)),
        }
    }

    pub(super) fn draw(&self, a: &mut Assembly, f: &CupForm, rim: f32, label: &str) {
        let tf = f.transform();
        let g = a.part(Surface::Plastic, label);
        let deck = self.deck_fraction * rim;
        let lip = f.height_m;
        let top = lip + self.rise_m;
        let t = self.gauge_m;
        let low = lip - 1.9 * f.lip_m;
        let contact = lip - 0.50 * f.lip_m;
        // The skirt touches the rolled paper rim on a ring. The underside,
        // skirt and top share perimeter edges and leave the drinking port open.
        outline(32).sweep(
            g,
            &[
                (deck, top - t),
                (rim * 0.92, lip + self.rise_m * 0.24),
                (rim + t * 0.08, lip + t * 0.7),
                (rim, contact),
                (rim + t * 0.20, low),
                (rim + t * 1.3, low),
                (rim + t * 1.7, contact),
                (rim + t * 1.2, lip + t * 1.5),
                (rim * 0.94, lip + self.rise_m * 0.35),
                (deck, top),
            ],
            ProfileJoints::Creased,
            tf,
        );
        super::super::end_plate::PiercedPlate::new(
            Vec2::splat(deck),
            Vec2::Y * (self.port_offset_fraction * deck),
            self.port_half_fraction * deck,
        )
        .faces(g, top, top - t, tf);
    }
}
