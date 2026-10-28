use super::*;

#[test]
fn clipped_wall_sliver_preserves_geometry_with_a_planar_normal() {
    // Seed 14: the last two vertices delimit a valid clipped roof corner.
    // World-coordinate quantization makes its cross-product tilt by 30 deg.
    let points = [
        Vec3::new(-5.3169694, 4.6075935, 6.7387934),
        Vec3::new(-4.773689, 5.503965, -6.7387934),
        Vec3::new(-4.7736893, 5.503965, -6.738787),
    ];
    let direction = (points[1] - points[0]).with_y(0.);
    let normal = direction.cross(Vec3::Y).normalize();
    let mut original = Geometry::default();
    let mut planar = Geometry::default();
    face(&mut original, &points, normal);
    wall_face(&mut planar, &points, normal);
    assert_eq!(planar.positions, original.positions);
    assert_eq!(planar.indices, original.indices);
    assert_eq!(planar.indices.len(), 3, "do not discard the thin triangle");
    assert!(original.normals[0][1].abs() > 0.4);
    for (n, uv) in planar.normals.iter().zip(&planar.uvs) {
        assert!(Vec3::from(*n).abs_diff_eq(normal, 1e-6));
        assert_eq!(n[1], 0.);
        assert!(uv.iter().all(|v| v.is_finite()));
    }
    let [a, b, c] = points.map(|p| p.as_dvec3());
    assert!((b - a).cross(c - a).dot(normal.as_dvec3()) > 0.);
}
