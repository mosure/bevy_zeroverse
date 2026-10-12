//! Retained 2D triangle batches; only moving overlays are rebuilt during playback.
use crate::annotation::schematic::{
    self as plan, Footprint, FootprintKind, Overlay, RenderOptions, Schematic,
};
use crate::scene::procedural_indoor::envelope::polygon;
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology},
    prelude::*,
};

pub(super) struct Label {
    pub position: Vec2,
    pub text: String,
    pub size: f32,
    pub color: Color,
}
pub(super) struct Drawing {
    points: Vec<[f32; 3]>,
    colors: Vec<[f32; 4]>,
    pub labels: Vec<Label>,
    size: Vec2,
}
fn color(hex: &str) -> Color {
    Color::Srgba(Srgba::hex(hex).expect("constant schematic color"))
}
impl Drawing {
    fn new(options: &RenderOptions) -> Self {
        Self {
            points: vec![],
            colors: vec![],
            labels: vec![],
            size: Vec2::new(options.width as f32, options.height as f32),
        }
    }
    fn raw_triangle(&mut self, points: [Vec2; 3], color: Color) {
        for v in points {
            self.points
                .push([v.x - self.size.x * 0.5, self.size.y * 0.5 - v.y, 0.]);
            self.colors.push(color.to_linear().to_f32_array());
        }
    }
    fn triangle(&mut self, points: [Vec2; 3], color: Color) {
        let lo = Vec2::new(12., 42.);
        let hi = self.size - Vec2::new(12., 46.);
        if points
            .iter()
            .all(|p| p.cmpge(lo).all() && p.cmple(hi).all())
        {
            self.raw_triangle(points, color);
            return;
        }
        // Outliers are clipped without changing the ground-truth metric extent.
        let p = polygon::clip_box(&points, lo, hi);
        for i in 1..p.len().saturating_sub(1) {
            self.raw_triangle([p[0], p[i], p[i + 1]], color);
        }
    }
    fn line(&mut self, a: Vec2, b: Vec2, width: f32, color: Color, dashed: bool) {
        let length = a.distance(b);
        if length < 0.001 {
            return;
        }
        if dashed {
            for i in 0..((length / 10.).ceil() as usize).min(2048) {
                let from = (i as f32 * 10.) / length;
                let to = ((i as f32 * 10. + 6.) / length).min(1.);
                self.line(a.lerp(b, from), a.lerp(b, to), width, color, false);
            }
            return;
        }
        let n = (b - a).normalize().perp() * width * 0.5;
        self.triangle([a - n, b - n, b + n], color);
        self.triangle([a - n, b + n, a + n], color);
    }
    fn dot(&mut self, p: Vec2, r: f32, color: Color) {
        for i in 0..16 {
            let angle = std::f32::consts::TAU / 16.;
            self.triangle(
                [
                    p,
                    p + Vec2::from_angle(i as f32 * angle) * r,
                    p + Vec2::from_angle((i + 1) as f32 * angle) * r,
                ],
                color,
            );
        }
    }
    fn label(&mut self, p: Vec2, text: &str, size: f32, color: Color) {
        if p.is_finite() {
            self.labels.push(Label {
                position: p,
                text: text.into(),
                size,
                color,
            });
        }
    }
    fn annotation_label(&mut self, p: Vec2, text: &str, size: f32, color: Color) {
        if p.cmpge(Vec2::new(12., 42. + size)).all()
            && p.cmple(self.size - Vec2::new(12., 46.)).all()
        {
            self.label(p, text, size, color);
        }
    }
    fn footprint(
        &mut self,
        f: &Footprint,
        projection: &plan::Projection,
        pred: bool,
        labels: bool,
    ) -> Result<(), String> {
        if f.points.iter().flatten().any(|v| !v.is_finite()) {
            return Err("non-finite schematic footprint".into());
        }
        let p: Vec<_> = f
            .points
            .iter()
            .map(|p| Vec2::from_array(projection.project(*p)))
            .collect();
        let (fill, stroke, width) = match f.kind {
            FootprintKind::Floor => ("ffffff", "293f43", 3.),
            FootprintKind::Level => ("e9e7dc", "b7b3a4", 0.7),
            FootprintKind::Mezzanine => ("dce7f0", "6287a1", 1.2),
            FootprintKind::Wall => ("455b60", "455b60", 4.),
            FootprintKind::Window => ("ffffff", "4eabc4", 4.),
            FootprintKind::Door => ("ffffff", "ffffff", 5.),
            FootprintKind::Furniture => ("d7c4a8", "8e7e64", 1.),
            FootprintKind::Prop => ("e3dfd6", "9a978e", 0.65),
            FootprintKind::Plant => ("b1c99c", "648451", 1.),
        };
        if p.len() > 2 {
            if p.len() > 4096 {
                return Err("schematic polygon exceeds vertex budget".into());
            }
            let triangles = polygon::triangles(&p);
            if polygon::area(&p) > 0.00001 && triangles.is_empty() {
                return Err("invalid schematic polygon".into());
            }
            for t in triangles {
                self.triangle(t, color(fill).with_alpha(0.85));
            }
        }
        let stroke = color(if pred { "bb297c" } else { stroke });
        for pair in p.windows(2) {
            self.line(pair[0], pair[1], width, stroke, pred);
        }
        if p.len() > 2 {
            self.line(*p.last().unwrap(), p[0], width, stroke, pred);
        }
        if labels
            && !p.is_empty()
            && matches!(f.kind, FootprintKind::Furniture | FootprintKind::Mezzanine)
        {
            self.annotation_label(
                p.iter().sum::<Vec2>() / p.len() as f32 + Vec2::new(3., -3.),
                &f.label,
                9.,
                color("435457"),
            );
        }
        Ok(())
    }
    pub fn into_mesh(self) -> Mesh {
        let n = self.points.len() as u32;
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        );
        mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, self.points);
        mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, self.colors);
        mesh.insert_indices(Indices::U32((0..n).collect()));
        mesh
    }
}
pub(super) fn background(
    plan: &Schematic,
    options: &RenderOptions,
    predictions: &Overlay,
) -> Result<Drawing, String> {
    let projection = plan.projection(options)?;
    let mut d = Drawing::new(options);
    for (pred, shapes) in [(false, &plan.footprints), (true, &predictions.footprints)] {
        for f in shapes {
            d.footprint(f, &projection, pred, options.labels)?;
        }
    }
    d.label(
        Vec2::new(24., 27.),
        &format!("ROOM {}  /  TOP-DOWN", plan.seed),
        16.,
        color("203337"),
    );
    let y = options.height as f32 - 26.;
    let x = 24. + projection.pixels_per_metre;
    let corners = [
        Vec2::new(24., y - 1.5),
        Vec2::new(x, y - 1.5),
        Vec2::new(x, y + 1.5),
        Vec2::new(24., y + 1.5),
    ];
    d.raw_triangle([corners[0], corners[1], corners[2]], color("203337"));
    d.raw_triangle([corners[0], corners[2], corners[3]], color("203337"));
    d.label(Vec2::new(24., y - 7.), "1 m", 11., color("203337"));
    d.label(
        Vec2::new(x + 18., y + 4.),
        "+X right / +Z down",
        11.,
        color("52666b"),
    );
    if predictions != &Overlay::default() {
        d.label(
            Vec2::new(options.width as f32 - 185., options.height as f32 - 25.),
            "Dashed: predictions",
            11.,
            color("bb297c"),
        );
    }
    Ok(d)
}
pub(super) fn moving(
    plan: &Schematic,
    options: &RenderOptions,
    predictions: &Overlay,
) -> Result<Drawing, String> {
    let projection = plan.projection(options)?;
    let project = |p| Vec2::from_array(projection.project(p));
    let mut d = Drawing::new(options);
    for (pred, poses) in [(false, &plan.poses), (true, &predictions.poses)] {
        for pose in poses {
            if pose.joints.len() != pose.parents.len()
                || pose.joints.iter().flatten().any(|v| !v.is_finite())
            {
                return Err("invalid schematic pose".into());
            }
            let c = color(if pred { "bb297c" } else { "a64d37" });
            for (i, parent) in pose.parents.iter().enumerate() {
                if *parent < -1 || *parent >= pose.joints.len() as i64 || *parent == i as i64 {
                    return Err("invalid schematic joint parent".into());
                }
                if *parent >= 0 {
                    d.line(
                        project(pose.joints[i]),
                        project(pose.joints[*parent as usize]),
                        2.,
                        c,
                        pred,
                    );
                }
                d.dot(project(pose.joints[i]), 2.3, c);
            }
            if let Some(p) = pose.joints.first() {
                d.annotation_label(project(*p) + Vec2::new(7., 15.), &pose.label, 11., c);
            }
        }
    }
    for (pred, cameras) in [(false, &plan.cameras), (true, &predictions.cameras)] {
        for (i, camera) in cameras.iter().enumerate() {
            let footprint = plan::camera_footprint(camera, options.frustum_length)?;
            let c = color(if pred {
                "#bb297c"
            } else {
                plan::camera_color(i)
            });
            let m = Mat4::from_cols_array_2d(&camera.world_from_view);
            let origin = project(m.transform_point3(Vec3::ZERO).to_array());
            if options.camera_paths {
                if camera.path.iter().flatten().any(|v| !v.is_finite()) {
                    return Err("non-finite schematic path".into());
                }
                for pair in camera.path.windows(2) {
                    d.line(
                        project(pair[0]),
                        project(pair[1]),
                        1.5,
                        c.with_alpha(0.65),
                        pred,
                    );
                }
            }
            let p: Vec<_> = footprint.into_iter().map(project).collect();
            for j in 1..p.len().saturating_sub(1) {
                d.triangle([p[0], p[j], p[j + 1]], c.with_alpha(0.10));
            }
            for j in 0..p.len() {
                d.line(p[j], p[(j + 1) % p.len()], 1.4, c, pred);
            }
            d.line(
                origin,
                project(
                    m.transform_point3(Vec3::NEG_Z * options.frustum_length)
                        .to_array(),
                ),
                2.5,
                c,
                pred,
            );
            d.dot(origin, 5.7, Color::WHITE);
            d.dot(origin, 5., c);
            d.annotation_label(origin + Vec2::new(8., -8.), &camera.label, 13., c);
        }
    }
    Ok(d)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::procedural_indoor::layout::{IndoorLayout, IndoorManifest};
    #[test]
    fn diagram_batches_cover_concave_rooms_and_reject_invalid_predictions() {
        let options = RenderOptions::default();
        for seed in 0..64 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 4, 0.)
                    .unwrap();
            let plan = Schematic::from_manifest(&scene, 0.5).unwrap();
            let mut predictions = Overlay {
                cameras: plan.cameras.clone(),
                ..default()
            };
            let static_batch = background(&plan, &options, &predictions).unwrap();
            let moving_batch = moving(&plan, &options, &predictions).unwrap();
            for d in [static_batch, moving_batch] {
                assert!(!d.points.is_empty());
                assert!(d.points.iter().flatten().all(|v| v.is_finite()));
                assert_eq!(d.points.len(), d.colors.len());
                assert!(d.points.len() < 200_000, "unbounded schematic geometry");
            }
            predictions.cameras[0].aspect = f32::NAN;
            assert!(moving(&plan, &options, &predictions).is_err());
            predictions.cameras.clear();
            predictions.poses.push(plan::Pose {
                label: "prediction".into(),
                joints: vec![[0.; 3]],
                parents: vec![2],
            });
            assert!(moving(&plan, &options, &predictions).is_err());
        }
    }
    #[test]
    #[ignore = "local CPU preparation timing; writes qualification evidence"]
    fn compare_live_schematic_preparation() {
        let options = RenderOptions {
            width: 980,
            height: 804,
            ..default()
        };
        let mut rows = vec![];
        for seed in 0..8 {
            let scene =
                IndoorManifest::generate_with_humans(seed, IndoorLayout::Mixed, 0.65, 4, 0.5)
                    .unwrap();
            let mut plan = Schematic::from_manifest(&scene, 0.).unwrap();
            let now = std::time::Instant::now();
            std::hint::black_box(
                background(&plan, &options, &Overlay::default())
                    .unwrap()
                    .into_mesh(),
            );
            let static_ms = now.elapsed().as_secs_f64() * 1000.;
            let mut cpu_ms = 0.;
            let mut mesh_ms = 0.;
            for frame in 0..12 {
                for (c, source) in plan.cameras.iter_mut().zip(&scene.cameras) {
                    c.world_from_view = (Mat4::from_rotation_y(scene.world_yaw)
                        * source.transform_at(frame as f32 / 11.).to_matrix())
                    .to_cols_array_2d();
                }
                let now = std::time::Instant::now();
                std::hint::black_box(plan.rgba(&options, &Overlay::default()).unwrap());
                cpu_ms += now.elapsed().as_secs_f64() * 1000.;
                let now = std::time::Instant::now();
                std::hint::black_box(
                    moving(&plan, &options, &Overlay::default())
                        .unwrap()
                        .into_mesh(),
                );
                mesh_ms += now.elapsed().as_secs_f64() * 1000.;
            }
            rows.push(serde_json::json!({"seed":seed,"frames":12,"static_mesh_ms":static_ms,"cpu_svg_frame_ms":cpu_ms/12.,"dynamic_mesh_frame_ms":mesh_ms/12.}));
        }
        std::fs::create_dir_all("out/schematic/gpu").unwrap();
        std::fs::write(
            "out/schematic/gpu/preparation.json",
            serde_json::to_vec_pretty(&rows).unwrap(),
        )
        .unwrap();
        println!("{}", serde_json::to_string_pretty(&rows).unwrap());
    }
}
