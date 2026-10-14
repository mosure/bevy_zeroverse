use super::*;
pub fn size_room(seed: u64, activity: IndoorLayout, size: &mut Vec3) {
    if !activity.is_public() {
        return;
    }
    let mut r = stream(seed, 8220);
    let (w, d) = match activity {
        IndoorLayout::LectureHall => ((10.0, 16.0), (12.0, 18.0)),
        IndoorLayout::PublicSuite => ((17.0, 22.0), (16.0, 21.0)),
        IndoorLayout::Cafeteria => ((10.0, 15.0), (10.0, 15.0)),
        IndoorLayout::ElevatorLanding => ((8.0, 13.0), (7.5, 11.0)),
        _ => ((8.5, 13.5), (8.5, 13.5)),
    };
    *size = Vec3::new(
        r.random_range(w.0..w.1),
        if activity == IndoorLayout::LectureHall {
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
    // Service and audience programs choose their levels before any objects. Random
    // depressions cannot bisect a service run, audience row or elevator threshold.
    if let Some(e) = scene.envelope.as_mut() {
        e.floor_patches.clear();
        e.mezzanine = None;
        if matches!(
            scene.layout,
            IndoorLayout::ElevatorLanding | IndoorLayout::PublicSuite
        ) {
            e.ceiling_drop = e.ceiling_drop.clamp(Vec2::splat(-0.35), Vec2::splat(0.35));
            for soffit in &mut e.soffits {
                soffit.height = soffit.height.max(3.1);
            }
        }
        // Peripheral columns remain; a central column cannot occupy a required aisle.
        e.pillars
            .retain(|p| p.center.x.abs() > scene.room_size.x * 0.32);
    }
    if scene.layout == IndoorLayout::PublicSuite {
        let p = scene.program.as_mut().unwrap();
        let mut right = p.zones[0].clone();
        let mut r = stream(scene.seed, 8221);
        let axis = r.random_range(0..2);
        let split = r.random_range(-0.06..0.06) * scene.room_size.xz()[axis];
        let pairs = [
            (IndoorLayout::Cafe, IndoorLayout::Gallery),
            (IndoorLayout::Cafe, IndoorLayout::Showroom),
            (IndoorLayout::ElevatorLanding, IndoorLayout::Gallery),
            (IndoorLayout::Classroom, IndoorLayout::Showroom),
            (IndoorLayout::Cafeteria, IndoorLayout::Gallery),
        ];
        let (a, b) = pairs[r.random_range(0..pairs.len())];
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
    let rows = ((hi.y - lo.y - 4.0) / p.row_pitch_metres)
        .floor()
        .clamp(2., 7.) as usize;
    let start = lo.y + 3.0;
    let half_aisle = p.aisle_metres * 0.5;
    let center = (lo.x + hi.x) * 0.5;
    let e = scene.envelope.as_mut().unwrap();
    for row in 1..rows {
        let front = start + row as f32 * p.row_pitch_metres;
        // Use the same arithmetic as the following row's front. Accumulating
        // pitch introduced one-ULP cracks with ambiguous support ownership.
        let back = start + (row + 1) as f32 * p.row_pitch_metres;
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
            e.floor_patches.push(FloorPatch {
                min: Vec2::new(center - half_aisle, z0),
                max: Vec2::new(center + half_aisle, z1),
                height: h,
            });
        }
    }
    // Raised teaching platform with a full-width single step, clear of side access.
    e.floor_patches.push(FloorPatch {
        min: Vec2::new(center - (hi.x - lo.x) * 0.38, lo.y),
        max: Vec2::new(center + (hi.x - lo.x) * 0.38, lo.y + 2.1),
        height: 0.14,
    });
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
        let lo = m.deck.min;
        let hi = m.deck.max;
        result.push(SupportRegion {
            name: "mezzanine".into(),
            min: lo,
            max: hi,
            elevation_metres: m.deck.height,
            triangles: vec![
                [lo, Vec2::new(hi.x, lo.y), hi],
                [lo, hi, Vec2::new(lo.x, hi.y)],
            ],
            overhead: true,
        });
    }
    result
}
