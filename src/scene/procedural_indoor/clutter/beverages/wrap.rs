//! Split the actual container skin at the two edges of a bonded label. A single
//! surface owns any point on the package, including across ribs and shoulders.
use super::*;
use crate::scene::procedural_indoor::materials::package_print::Print;

pub(super) struct Wrap {
    pub band: Vec2,
    pub print: Print,
    pub phase: f32,
}
impl Wrap {
    pub fn new(o: &IndoorObject, band: Vec2) -> Self {
        Self {
            band,
            print: Print::for_object(o),
            phase: stream(o.seed, u64::from_le_bytes(*b"WrapSeam")).random_range(0.0..1.),
        }
    }
    /// Keep a cut on a straight facet, clear of its corner. Both materials then
    /// have identical seam positions and slopes, even on deeply ribbed stock.
    fn cut(stations: &mut Vec<(f32, f32)>, y: f32) -> usize {
        let i = stations.partition_point(|&(_, h)| h < y);
        let (r0, h0) = stations[i - 1];
        let (r1, h1) = stations[i];
        let clearance = 0.000003_f32.min((h1 - h0) * 0.25);
        let y = y.clamp(h0 + clearance, h1 - clearance);
        let u = (y - h0) / (h1 - h0);
        stations.insert(i, (r0 + (r1 - r0) * u, y));
        i
    }
}

pub(super) fn side(
    a: &mut Assembly,
    skin: (Surface, &str),
    mut stations: Vec<(f32, f32)>,
    wrap: Option<Wrap>,
    tf: Transform,
) {
    let round = outline(32);
    let body = a.part(skin.0, skin.1);
    let Some(wrap) = wrap else {
        round.sweep(body, &stations, ProfileJoints::Smooth, tf);
        return;
    };
    let low = Wrap::cut(&mut stations, wrap.band.x);
    let high = Wrap::cut(&mut stations, wrap.band.y);
    round.sweep(body, &stations[..=low], ProfileJoints::Smooth, tf);
    round.sweep(body, &stations[high..], ProfileJoints::Smooth, tf);
    let g = a.part(wrap.print.surface(), &wrap.print.label());
    let start = g.positions.len();
    round.sweep(g, &stations[low..=high], ProfileJoints::Smooth, tf);
    // Derive the UV limits from the emitted rings. A yaw quaternion can round
    // the transformed Y coordinate by a few ULPs.
    let lo = g.positions[start][1];
    let hi = g.positions.last().expect("label has two rings")[1];
    for (i, (v, uv)) in g.positions[start..]
        .iter()
        .zip(&mut g.uvs[start..])
        .enumerate()
    {
        // Looking at the front of the cylinder, U increases to the right and
        // the top scanline sits at its top. One printed sheet wraps one turn.
        *uv = [
            wrap.phase + 1. - (i % 33) as f32 / 32.,
            (hi - v[1]) / (hi - lo),
        ];
    }
}
