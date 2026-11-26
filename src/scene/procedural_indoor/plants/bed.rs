//! A raised garden island with a recessed growing bed. The fitted object includes
//! the entire planting so people, camera paths and entrance aisles clear it.
use super::*;
use crate::scene::procedural_indoor::{
    geometry::{ProfileJoints, RevolvedOutline},
    layout::ObjectKind,
    objects::append_at,
};

#[derive(serde::Serialize)]
pub(crate) struct Bed {
    pub rim_height_m: f32,
    pub soil_height_m: f32,
    rim_width_m: f32,
    exponent: f32,
    edge: Surface,
    medium: soil::Medium,
}

impl Bed {
    pub(crate) fn sample(o: &IndoorObject) -> Self {
        let mut r = stream(o.seed, u64::from_le_bytes(*b"YardSoil"));
        let edge = [Surface::Concrete, Surface::Terracotta, Surface::Metal][r.random_range(0..3)];
        // Taller specimens need more growing medium in the above-grade trough.
        let rim_height_m = (o.size.y * r.random_range(0.10..0.22) + 0.04).min(o.size.y * 0.36);
        Self {
            rim_height_m,
            soil_height_m: rim_height_m - r.random_range(0.025..0.065_f32).min(rim_height_m * 0.4),
            rim_width_m: if edge == Surface::Metal {
                r.random_range(0.012..0.029)
            } else {
                r.random_range(0.055..0.13)
            },
            exponent: r.random_range(1.85..6.2),
            edge,
            medium: [
                soil::Medium::Loam,
                soil::Medium::BarkChips,
                soil::Medium::MineralGravel,
            ][r.random_range(0..3)],
        }
    }
}

pub(super) fn build(a: &mut Assembly, o: &IndoorObject, version: u32) {
    let bed = Bed::sample(o);
    let radius = o.size.xz() * 0.5;
    let lip = Vec2::splat(bed.rim_width_m).min(radius * 0.20);
    let inner = radius - lip;
    let outside = RevolvedOutline::superellipse(32, bed.exponent, radius);
    let inside = RevolvedOutline::superellipse(32, bed.exponent, inner);
    let rim = a.part(bed.edge, "plant");
    let at = |p: Vec2, y| Vec3::new(p.x, y, p.y);
    outside.sweep(
        rim,
        &[(1., 0.), (1., bed.rim_height_m)],
        ProfileJoints::Creased,
        Transform::IDENTITY,
    );
    inside.sweep(
        rim,
        &[(1., bed.rim_height_m), (1., bed.soil_height_m)],
        ProfileJoints::Creased,
        Transform::IDENTITY,
    );
    for i in 0..32 {
        let p = i as f32 / 32.;
        let q = (i + 1) as f32 / 32.;
        rim.metric_quad(
            [
                outside.perimeter_point(p),
                outside.perimeter_point(q),
                inside.perimeter_point(q),
                inside.perimeter_point(p),
            ]
            .map(|point| at(point, bed.rim_height_m)),
            Vec3::Y,
        );
    }
    inside.sweep(
        a.part(Surface::Soil, &bed.medium.label(soil::Layer::Bed)),
        &[(1., bed.soil_height_m), (0., bed.soil_height_m)],
        ProfileJoints::Creased,
        Transform::IDENTITY,
    );

    let mut r = stream(o.seed, u64::from_le_bytes(*b"YardRoot"));
    let sign = if r.random_bool(0.5) { 1. } else { -1. };
    // One dominant crown and one or two lower, staggered companions. Bound
    // each crown before meshing. Every stem starts at the finished soil level.
    let count = if o.size.x / o.size.z > 1.72 && r.random_bool(0.63) {
        3
    } else {
        2
    };
    let mut child = o.clone();
    child.kind = ObjectKind::Plant;
    for i in 0..count {
        let x = sign * ((i as f32 + r.random_range(0.36..0.64)) / count as f32 - 0.5) * 1.55;
        let z: f32 = r.random_range(-0.20..0.20);
        let spread =
            (inner.x * (1. - x.abs())).min(inner.y * (1. - z.abs())) * r.random_range(1.35..1.76);
        let height = (o.size.y - bed.soil_height_m)
            * if i == 0 {
                1.
            } else {
                r.random_range(0.27..0.62)
            };
        child.variant = if i == 0 {
            o.variant
        } else {
            [4, 9, 13][r.random_range(0..3)]
        };
        child.seed = r.random();
        child.size = Vec3::new(spread, height, spread);
        let mut crown = Assembly::default();
        build_plant(&mut crown, &child, version, Roots::Bed);
        // Some stems and lower leaves turn down through grade. Cut only
        // geometry reaching the bottom plate, below the opaque soil surface.
        let bottom = 0.008 - bed.soil_height_m;
        for g in crown.parts.values_mut() {
            if g.positions.iter().any(|v| v[1] < bottom) {
                g.clip_half_space(|p| f64::from(p.y) - f64::from(bottom));
            }
        }
        let tf = Transform::from_xyz(x * inner.x, bed.soil_height_m, z * inner.y)
            .with_rotation(Quat::from_rotation_y(r.random_range(-PI..PI)));
        append_at(a, crown, tf);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn courtyard_bed_has_buried_roots_recessed_soil_and_bounded_crowns() {
        for seed in 0..48 {
            let mut o = crate::scene::procedural_indoor::objects::audit::fixture(
                ObjectKind::PlantingBed,
                seed,
            );
            o.variant = [6, 13, 10, 4, 15, 1, 14, 3, 8, 9, 7][seed as usize % 11];
            let p = Bed::sample(&o);
            let mut a = Assembly::default();
            build(
                &mut a,
                &o,
                crate::scene::procedural_indoor::layout::GENERATOR_VERSION,
            );
            let (lo, hi) = a.bounds();
            assert!(
                lo.y.abs() < 1e-4 && hi.y <= o.size.y + 0.002,
                "{seed}: {lo} {hi}"
            );
            assert!(
                lo.xz()
                    .abs()
                    .max(hi.xz().abs())
                    .cmple(o.size.xz() * 0.5 + Vec2::splat(0.002))
                    .all(),
                "{seed}"
            );
            let fill = &a.parts[&(Surface::Soil, p.medium.label(soil::Layer::Bed).into_owned())];
            assert!(fill
                .positions
                .iter()
                .all(|pos| (pos[1] - p.soil_height_m).abs() < 1e-5));
            assert!(p.soil_height_m < p.rim_height_m - 0.02);
            // Ferns and succulents have green rachises or fleshy stems. A real
            // planted triangle must reach the soil with any such tissue.
            let rooted = a
                .parts
                .iter()
                .filter(|((s, _), _)| *s != p.edge && *s != Surface::Soil)
                .any(|(_, g)| {
                    g.indices.as_chunks::<3>().0.iter().any(|tri| {
                        let y = tri.map(|i| g.positions[i as usize][1]);
                        y.iter().any(|v| *v <= p.soil_height_m + 0.002)
                            && y.iter().any(|v| *v >= p.soil_height_m - 0.002)
                    })
                });
            assert!(rooted, "{seed}: floating planting at {}", p.soil_height_m);
        }
    }
}
