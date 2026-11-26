use super::*;
use crate::scene::procedural_indoor::envelope::steps::{StepAxis, StepEdge, StepInstallation};
pub fn size_room(seed: u64, activity: IndoorLayout, size: &mut Vec3) {
    if !activity.is_public() {
        return;
    }
    if activity == IndoorLayout::Gymnasium {
        *size = crate::scene::procedural_indoor::program::court::room_size(seed);
        return;
    }
    if activity == IndoorLayout::StairHall {
        *size =
            crate::scene::procedural_indoor::envelope::mezzanine::stair_hall::Section::sample(seed)
                .room_size;
        return;
    }
    if activity == IndoorLayout::Hallway {
        *size = super::super::super::program::hallway::Section::sample(seed).size();
        return;
    }
    let mut r = stream(seed, 8220);
    let (w, d) = match activity {
        IndoorLayout::Courtyard => ((10.5, 16.0), (11.5, 17.5)),
        IndoorLayout::Restaurant => ((9.0, 15.5), (10.0, 18.0)),
        IndoorLayout::CyclingStudio => ((10.4, 14.6), (11.0, 16.5)),
        IndoorLayout::FitnessRoom => ((8.2, 13.6), (9.5, 16.8)),
        IndoorLayout::GroceryStore => ((10.8, 16.2), (12.0, 18.7)),
        IndoorLayout::ClothingStore => ((9.8, 15.3), (11.0, 17.6)),
        IndoorLayout::LectureHall => ((10.0, 16.0), (12.0, 18.0)),
        IndoorLayout::PublicSuite => ((17.0, 22.0), (16.0, 21.0)),
        IndoorLayout::Cafeteria => ((10.0, 15.0), (10.0, 15.0)),
        IndoorLayout::ElevatorLanding => ((8.0, 13.0), (7.5, 11.0)),
        _ => ((8.5, 13.5), (8.5, 13.5)),
    };
    *size = Vec3::new(
        r.random_range(w.0..w.1),
        if activity == IndoorLayout::Courtyard {
            r.random_range(3.15..4.35)
        } else if activity == IndoorLayout::LectureHall {
            r.random_range(5.5..6.2)
        } else {
            r.random_range(3.6..4.8)
        },
        r.random_range(d.0..d.1),
    );
}
pub fn prepare_architecture(scene: &mut IndoorManifest) {
    if !scene.layout.is_public() {
        return;
    }
    if scene.layout == IndoorLayout::Gymnasium {
        // Keep the playing volume continuous from the floor to the high roof.
        let e = scene.envelope.as_mut().unwrap();
        e.floor_patches.clear();
        e.step_installation = None;
        e.mezzanine = None;
        e.pillars.clear();
        e.soffits.clear();
        e.ceiling_drop = e.ceiling_drop.clamp(Vec2::splat(-0.48), Vec2::splat(0.48));
        return;
    }
    if scene.layout == IndoorLayout::Courtyard {
        // Its load-bearing colonnade and open roof are designed as one envelope.
        return;
    }
    if scene.layout == IndoorLayout::StairHall {
        let routes = furnishing::stair_hall::routes(scene);
        let e = scene.envelope.as_mut().unwrap();
        e.pillars.retain(|p| {
            let pad = Vec2::splat(p.radius + 0.12);
            routes
                .iter()
                .all(|r| (p.center + pad).cmple(r.min).any() || (p.center - pad).cmpge(r.max).any())
        });
        return;
    }
    // Service and audience programs choose their levels before any objects. Random
    // depressions cannot bisect a service run, audience row or elevator threshold.
    if let Some(e) = scene.envelope.as_mut() {
        e.floor_patches.clear();
        e.step_installation = None;
        e.mezzanine = None;
        if matches!(
            scene.layout,
            IndoorLayout::ElevatorLanding | IndoorLayout::PublicSuite | IndoorLayout::Hallway
        ) {
            e.ceiling_drop = e.ceiling_drop.clamp(Vec2::splat(-0.35), Vec2::splat(0.35));
            for soffit in &mut e.soffits {
                soffit.height = soffit.height.max(3.1);
            }
        }
        // Peripheral columns remain; a central column cannot occupy a required aisle.
        if scene.layout == IndoorLayout::Hallway {
            let hall = &scene.program.as_ref().unwrap().zones[0];
            e.pillars.retain(|p| {
                let pad = Vec2::splat(p.radius + 0.35);
                p.center.cmplt(hall.min - pad).any() || p.center.cmpgt(hall.max + pad).any()
            });
        } else {
            e.pillars
                .retain(|p| p.center.x.abs() > scene.room_size.x * 0.32);
        }
    }
    if scene.layout == IndoorLayout::PublicSuite {
        let p = scene.program.as_mut().unwrap();
        let mut right = p.zones[0].clone();
        let mut r = stream(scene.seed, 8221);
        let axis = r.random_range(0..2);
        let split = r.random_range(-0.06..0.06) * scene.room_size.xz()[axis];
        let (a, b) = suite_activities(&mut r);
        p.zones[0].max[axis] = split;
        p.zones[0].activity = a;
        right.min[axis] = split;
        right.activity = b;
        p.zones.push(right);
    }
    if scene.layout != IndoorLayout::LectureHall {
        return;
    }
    if let Some(e) = scene.envelope.as_mut() {
        e.ceiling_drop = e.ceiling_drop.clamp(Vec2::splat(-0.5), Vec2::splat(0.5));
        e.soffits.clear();
    }
    let zone = scene.program.as_ref().unwrap().zones[0].clone();
    let (lo, hi) = usable_region(scene, &zone);
    let p = ProgramParameters::sample(scene.seed, 0, scene.density);
    let rows = p.lecture_rows(lo.y + 3., hi.y - 1.);
    let half_aisle = p.aisle_metres * 0.5;
    let center = (lo.x + hi.x) * 0.5;
    let e = scene.envelope.as_mut().unwrap();
    let mut steps = StepInstallation::sample(scene.seed, scene.floor_style, StepAxis::Z);
    for (row, &[front, back]) in rows.iter().enumerate().skip(1) {
        // The shared plan reuses each boundary exactly, avoiding one-ULP cracks.
        let height = row as f32 * p.tier_rise_metres;
        for (x0, x1) in [(lo.x, center - half_aisle), (center + half_aisle, hi.x)] {
            e.floor_patches.push(FloorPatch {
                min: Vec2::new(x0, front),
                max: Vec2::new(x1, back),
                height,
            });
        }
        // Two ordinary risers per seating tier; the long landing belongs to the
        // same support level as its row. No camera/human is placed inside a riser.
        for (z0, z1, h) in [
            (front, front + 0.32, height - p.tier_rise_metres * 0.5),
            (front + 0.32, back, height),
        ] {
            steps.walk_edges.push(StepEdge {
                coordinate_m: z0,
                span_m: Vec2::new(center - half_aisle, center + half_aisle),
            });
            e.floor_patches.push(FloorPatch {
                min: Vec2::new(center - half_aisle, z0),
                max: Vec2::new(center + half_aisle, z1),
                height: h,
            });
        }
    }
    // Raised teaching platform with a full-width single step, clear of side access.
    let stage_half = (hi.x - lo.x) * 0.38;
    steps.walk_edges.push(StepEdge {
        coordinate_m: lo.y + 2.1,
        span_m: Vec2::new(center - stage_half, center + stage_half),
    });
    e.step_installation = Some(steps);
    e.floor_patches.push(FloorPatch {
        min: Vec2::new(center - stage_half, lo.y),
        max: Vec2::new(center + stage_half, lo.y + 2.1),
        height: 0.14,
    });
}

/// Compatible public adjacencies, with both orientations and continuous room
/// proportions. Each half executes its complete functional builder. Lecture
/// halls keep their full tier/support domain rather than fitting half a hall.
fn suite_activities(rng: &mut ChaCha8Rng) -> (IndoorLayout, IndoorLayout) {
    use IndoorLayout::*;
    let public = [
        ElevatorLanding,
        Cafe,
        Cafeteria,
        Classroom,
        Showroom,
        Gallery,
    ];
    let a = public[rng.random_range(0..public.len())];
    let peers: &[IndoorLayout] = match a {
        ElevatorLanding => &[Cafe, Cafeteria, Classroom, Showroom, Gallery],
        Cafe => &[ElevatorLanding, Classroom, Showroom, Gallery],
        Cafeteria => &[ElevatorLanding, Classroom, Gallery],
        Classroom => &[ElevatorLanding, Cafe, Cafeteria, Showroom, Gallery],
        Showroom => &[ElevatorLanding, Cafe, Classroom, Gallery],
        Gallery => &[ElevatorLanding, Cafe, Cafeteria, Classroom, Showroom],
        _ => unreachable!("suite anchor catalog"),
    };
    (a, peers[rng.random_range(0..peers.len())])
}
pub fn support_regions(scene: &IndoorManifest) -> Vec<SupportRegion> {
    use super::super::super::envelope::polygon;
    let half = scene.room_size.xz() * 0.5;
    let rectangle = vec![
        -half,
        Vec2::new(half.x, -half.y),
        half,
        Vec2::new(-half.x, half.y),
    ];
    let footprint = scene
        .envelope
        .as_ref()
        .map_or(rectangle.as_slice(), |e| e.footprint.as_slice());
    let mut result = vec![SupportRegion {
        name: "ground".into(),
        min: -half,
        max: half,
        elevation_metres: 0.,
        triangles: vec![],
        overhead: false,
    }];
    let mut xs = vec![-half.x, half.x];
    let mut zs = vec![-half.y, half.y];
    if let Some(e) = &scene.envelope {
        for (i, p) in e.floor_patches.iter().enumerate() {
            xs.extend([p.min.x, p.max.x]);
            zs.extend([p.min.y, p.max.y]);
            result.push(SupportRegion {
                name: format!("floor_{i}"),
                min: p.min,
                max: p.max,
                elevation_metres: p.height,
                triangles: vec![],
                overhead: false,
            });
        }
    }
    xs.sort_by(f32::total_cmp);
    xs.dedup();
    zs.sort_by(f32::total_cmp);
    zs.dedup();
    let triangles = polygon::triangles(footprint);
    for x in xs.windows(2) {
        for z in zs.windows(2) {
            let lo = Vec2::new(x[0], z[0]);
            let hi = Vec2::new(x[1], z[1]);
            let center = (lo + hi) * 0.5;
            let region = scene
                .envelope
                .as_ref()
                .and_then(|e| e.floor_patches.iter().position(|p| p.contains(center)))
                .map_or(0, |i| i + 1);
            for triangle in &triangles {
                let poly = polygon::clip_box(triangle, lo, hi);
                for i in 1..poly.len().saturating_sub(1) {
                    let t = [poly[0], poly[i], poly[i + 1]];
                    if (t[1] - t[0]).perp_dot(t[2] - t[0]).abs() > 1e-8 {
                        result[region].triangles.push(t);
                    }
                }
            }
        }
    }
    if let Some(m) = scene.envelope.as_ref().and_then(|e| e.mezzanine.as_ref()) {
        let mut upper = |name, lo: Vec2, hi: Vec2, elevation_metres| {
            result.push(SupportRegion {
                name,
                min: lo,
                max: hi,
                elevation_metres,
                triangles: vec![
                    [lo, Vec2::new(hi.x, lo.y), hi],
                    [lo, hi, Vec2::new(lo.x, hi.y)],
                ],
                overhead: true,
            });
        };
        upper("mezzanine".into(), m.deck.min, m.deck.max, m.deck.height);
        if scene.layout == IndoorLayout::StairHall {
            for (i, (lo, hi)) in m.treads().enumerate() {
                upper(format!("stair_tread_{i}"), lo.xz(), hi.xz(), hi.y);
            }
        }
    }
    result
}
