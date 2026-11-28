//! A shared filtered chart for the cut face of a wound sheet. Its UV radius
//! encodes layer pitch; the through-bore and telescoping end are geometry.
use super::*;
use std::sync::OnceLock;

/// Number of wraps between the chart origin and the edge of its inscribed disk.
pub(crate) const CHART_TURNS: f32 = 18.;

pub(super) fn maps(images: &mut impl AssetStore<Image>) -> [Handle<Image>; 2] {
    static EDGE: OnceLock<[Image; 2]> = OnceLock::new();
    let maps = EDGE.get_or_init(|| {
        let (_, normal, orm) =
            sheet::metric_maps::<128>(Vec2::splat(2. * CHART_TURNS * 0.006), |uv| {
                let p = (uv - 0.5) * 2.;
                let r = p.length();
                let phase = CHART_TURNS * r - p.y.atan2(p.x) / std::f32::consts::TAU;
                let lips = (0.5 + 0.5 * (std::f32::consts::TAU * phase).cos()).powi(3);
                let coverage = field::smooth(r * 30.) * field::smooth((1. - r) * 12.);
                let cut = lips * coverage;
                (Vec3::ONE, -0.00032 * cut, 0.69 + 0.14 * cut)
            });
        [
            mip_image(normal, 128, MapType::Normal),
            mip_image(orm, 128, MapType::Data),
        ]
    });
    maps.each_ref().map(|map| images.add(map.clone()))
}
