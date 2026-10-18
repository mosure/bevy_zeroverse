//! Four independent shadow tests with the scalar Moller-Trumbore arithmetic.
//! This only answers boolean occlusion; closest-hit order and UVs stay scalar.
use super::{TransportTriangle, Vec3};
use bevy::prelude::Vec4;

pub(super) fn occluded(
    triangles: &[TransportTriangle; 4],
    origin: Vec3,
    direction: Vec3,
    max: f32,
) -> bool {
    let ab = [
        Vec4::from_array(triangles.map(|t| t.ab.x)),
        Vec4::from_array(triangles.map(|t| t.ab.y)),
        Vec4::from_array(triangles.map(|t| t.ab.z)),
    ];
    let ac = [
        Vec4::from_array(triangles.map(|t| t.ac.x)),
        Vec4::from_array(triangles.map(|t| t.ac.y)),
        Vec4::from_array(triangles.map(|t| t.ac.z)),
    ];
    let p = [
        direction.y * ac[2] - direction.z * ac[1],
        direction.z * ac[0] - direction.x * ac[2],
        direction.x * ac[1] - direction.y * ac[0],
    ];
    let det = ab[0] * p[0] + ab[1] * p[1] + ab[2] * p[2];
    let mut valid = det.abs().cmpge(Vec4::splat(1e-9));
    if !valid.any() {
        return false;
    }
    let inverse = det.recip();
    let t = [
        Vec4::splat(origin.x) - Vec4::from_array(triangles.map(|t| t.a.x)),
        Vec4::splat(origin.y) - Vec4::from_array(triangles.map(|t| t.a.y)),
        Vec4::splat(origin.z) - Vec4::from_array(triangles.map(|t| t.a.z)),
    ];
    let u = (t[0] * p[0] + t[1] * p[1] + t[2] * p[2]) * inverse;
    valid &= u.cmpge(Vec4::ZERO) & u.cmple(Vec4::ONE);
    if !valid.any() {
        return false;
    }
    let q = [
        t[1] * ab[2] - t[2] * ab[1],
        t[2] * ab[0] - t[0] * ab[2],
        t[0] * ab[1] - t[1] * ab[0],
    ];
    let v = (direction.x * q[0] + direction.y * q[1] + direction.z * q[2]) * inverse;
    valid &= !v.cmplt(Vec4::ZERO) & !(u + v).cmpgt(Vec4::ONE);
    if !valid.any() {
        return false;
    }
    let distance = (ac[0] * q[0] + ac[1] * q[1] + ac[2] * q[2]) * inverse;
    (valid & distance.cmpgt(Vec4::splat(0.0002)) & distance.cmplt(Vec4::splat(max))).any()
}

#[cfg(test)]
mod tests;
