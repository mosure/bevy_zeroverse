//! Small procedural mesh vocabulary. UVs are measured in metres, not object extents.
use bevy::{
    asset::RenderAssetUsages,
    mesh::{Indices, PrimitiveTopology, VertexAttributeValues},
    prelude::*,
};
use std::f32::consts::TAU;

#[derive(Default)]
pub struct Geometry {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub uvs: Vec<[f32; 2]>,
    pub indices: Vec<u32>,
}

impl Geometry {
    fn vertex(&mut self, p: Vec3, n: Vec3, uv: Vec2, tf: &Transform) {
        self.positions.push(tf.transform_point(p).to_array());
        self.normals.push(
            (tf.rotation * (n / tf.scale))
                .normalize_or_zero()
                .to_array(),
        );
        self.uvs.push(uv.to_array());
    }

    /// Rounded edges catch highlights without requiring a high-poly imported mesh.
    pub fn cuboid(&mut self, size: Vec3, bevel: f32, tf: Transform) {
        let half = size * 0.5;
        let radius = bevel.max(0.0).min(size.min_element() * 0.45);
        let core = half - Vec3::splat(radius);
        for (n, u, v) in [
            (Vec3::X, -Vec3::Z, Vec3::Y),
            (-Vec3::X, Vec3::Z, Vec3::Y),
            (Vec3::Y, Vec3::X, -Vec3::Z),
            (-Vec3::Y, Vec3::X, Vec3::Z),
            (Vec3::Z, Vec3::X, Vec3::Y),
            (-Vec3::Z, -Vec3::X, Vec3::Y),
        ] {
            let du = half.dot(u.abs());
            let dv = half.dot(v.abs());
            let dn = half.dot(n.abs());
            let a = if radius > 0.0 {
                vec![-du, -du + radius, du - radius, du]
            } else {
                vec![-du, du]
            };
            let b = if radius > 0.0 {
                vec![-dv, -dv + radius, dv - radius, dv]
            } else {
                vec![-dv, dv]
            };
            let start = self.positions.len() as u32;
            for &y in &b {
                for &x in &a {
                    let p = n * dn + u * x + v * y;
                    let closest = p.clamp(-core, core);
                    let normal = if radius > 0.0 {
                        (p - closest).normalize()
                    } else {
                        n
                    };
                    let p = if radius > 0.0 {
                        closest + normal * radius
                    } else {
                        p
                    };
                    self.vertex(p, normal, Vec2::new(x + du, dv - y), &tf);
                }
            }
            let nx = a.len() as u32;
            for y in 0..b.len() as u32 - 1 {
                for x in 0..nx - 1 {
                    let i = start + y * nx + x;
                    self.indices
                        .extend([i, i + 1, i + nx + 1, i, i + nx + 1, i + nx]);
                }
            }
        }
    }

    pub fn mesh(&mut self, mesh: Mesh, tf: Transform) {
        let Some(VertexAttributeValues::Float32x3(pos)) = mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            return;
        };
        let Some(VertexAttributeValues::Float32x3(norm)) = mesh.attribute(Mesh::ATTRIBUTE_NORMAL)
        else {
            return;
        };
        let Some(VertexAttributeValues::Float32x2(uv)) = mesh.attribute(Mesh::ATTRIBUTE_UV_0)
        else {
            return;
        };
        let offset = self.positions.len() as u32;
        for ((p, n), uv) in pos.iter().zip(norm).zip(uv) {
            self.vertex(
                Vec3::from_array(*p),
                Vec3::from_array(*n),
                Vec2::from_array(*uv),
                &tf,
            );
        }
        if let Some(indices) = mesh.indices() {
            self.indices
                .extend(indices.iter().map(|i| i as u32 + offset));
        } else {
            self.indices
                .extend((0..pos.len() as u32).map(|i| i + offset));
        }
    }

    pub fn ellipsoid(&mut self, radii: Vec3, tf: Transform) {
        self.mesh(
            Sphere::new(1.0).mesh().uv(20, 12),
            tf.with_scale(tf.scale * radii),
        );
    }

    /// A surface of revolution; profile goes from bottom outside to top, then inside.
    pub fn lathe(&mut self, profile: &[(f32, f32)], segments: u32, tf: Transform) {
        // Duplicate rings at profile creases: a pot rim must not average its outer
        // normal with the inward-facing inner wall or invert the lighting normal.
        for pair in profile.windows(2) {
            let offset = self.positions.len() as u32;
            let tangent = Vec2::new(pair[1].0 - pair[0].0, pair[1].1 - pair[0].1).normalize();
            for &(r, y) in pair {
                for i in 0..=segments {
                    let a = i as f32 / segments as f32 * TAU;
                    let normal = Vec3::new(tangent.y * a.cos(), -tangent.x, tangent.y * a.sin());
                    self.vertex(
                        Vec3::new(r * a.cos(), y, r * a.sin()),
                        normal,
                        Vec2::new(a * pair[0].0.max(pair[1].0).max(0.001), y + r),
                        &tf,
                    );
                }
            }
            for i in 0..segments {
                let a = offset + i;
                let b = a + segments + 1;
                if pair[0].0 > 0.00001 {
                    self.indices.extend([a, b, a + 1]);
                }
                if pair[1].0 > 0.00001 {
                    self.indices.extend([a + 1, b, b + 1]);
                }
            }
        }
    }

    pub fn cylinder(&mut self, radius: f32, height: f32, tf: Transform) {
        self.lathe(
            &[
                (0.0, -height * 0.5),
                (radius, -height * 0.5),
                (radius, height * 0.5),
                (0.0, height * 0.5),
            ],
            20,
            tf,
        );
    }

    pub fn rod(&mut self, a: Vec3, b: Vec3, radius: f32) {
        let d = b - a;
        if d.length_squared() < 1e-10 {
            return;
        }
        self.cylinder(
            radius,
            d.length(),
            Transform::from_translation((a + b) * 0.5)
                .with_rotation(Quat::from_rotation_arc(Vec3::Y, d.normalize())),
        );
    }

    /// Curved leaf blade with an actual silhouette, central fold, and tapered tip.
    pub fn leaf(&mut self, length: f32, width: f32, curl: f32, tf: Transform) {
        let offset = self.positions.len() as u32;
        let rows = 9;
        for row in 0..=rows {
            let t = row as f32 / rows as f32;
            let half = width * 0.5 * (std::f32::consts::PI * t).sin().max(0.012);
            for side in [-1.0, 0.0, 1.0] {
                let x = half * side;
                let y = curl * t * t + half * side.abs() * 0.30;
                let p = Vec3::new(x, y, -length * t);
                let n = Vec3::new(-side * 0.30, 1.0, 2.0 * curl * t / length).normalize();
                self.vertex(p, n, Vec2::new((side + 1.0) * 0.5, t), &tf);
            }
        }
        for row in 0..rows {
            for x in 0..2 {
                let i = offset + row * 3 + x;
                self.indices.extend([i, i + 1, i + 4, i, i + 4, i + 3]);
            }
        }
    }

    /// Ergonomic chair shell, curved horizontally and reclined vertically.
    pub fn chair_back(&mut self, width: f32, height: f32, tf: Transform) {
        let offset = self.positions.len() as u32;
        for row in 0..=8 {
            let y = height * row as f32 / 8.0;
            for col in 0..=12 {
                let t = col as f32 / 12.0 * 2.0 - 1.0;
                let x = width * 0.5 * t * (1.0 - 0.09 * (y / height));
                let z = 0.065 * (1.0 - t * t) + y * 0.13;
                let n = Vec3::new(-0.26 * t / width, 0.13, -1.0).normalize();
                self.vertex(Vec3::new(x, y, z), n, Vec2::new(x + width * 0.5, y), &tf);
            }
        }
        for row in 0..8 {
            for col in 0..12 {
                let i = offset + row * 13 + col;
                self.indices.extend([i, i + 14, i + 1, i, i + 13, i + 14]);
            }
        }
        // Close the padded shell instead of leaving a single infinitely thin sheet.
        let count = 9 * 13;
        let inner = self.positions.len() as u32;
        for i in offset..offset + count {
            let p = Vec3::from_array(self.positions[i as usize]);
            let n = Vec3::from_array(self.normals[i as usize]);
            self.positions.push((p - n * 0.035).to_array());
            self.normals.push((-n).to_array());
            self.uvs.push(self.uvs[i as usize]);
        }
        for row in 0..8 {
            for col in 0..12 {
                let i = inner + row * 13 + col;
                self.indices.extend([i, i + 1, i + 14, i, i + 14, i + 13]);
            }
        }
        let mut rim: Vec<u32> = (0..13).collect();
        rim.extend((1..9).map(|r| r * 13 + 12));
        rim.extend((0..12).rev().map(|c| 8 * 13 + c));
        rim.extend((1..8).rev().map(|r| r * 13));
        for k in 0..rim.len() {
            let i = rim[k];
            let j = rim[(k + 1) % rim.len()];
            let p = Vec3::from_array(self.positions[(offset + i) as usize]);
            let q = Vec3::from_array(self.positions[(offset + j) as usize]);
            let r = Vec3::from_array(self.positions[(inner + i) as usize]);
            let n = (q - p).cross(r - p).normalize();
            let start = self.positions.len() as u32;
            for (index, uv) in [
                (offset + i, [0.0, 0.0]),
                (offset + j, [(q - p).length(), 0.0]),
                (inner + j, [(q - p).length(), 0.035]),
                (inner + i, [0.0, 0.035]),
            ] {
                self.positions.push(self.positions[index as usize]);
                self.normals.push(n.to_array());
                self.uvs.push(uv);
            }
            self.indices
                .extend([start, start + 1, start + 2, start, start + 2, start + 3]);
        }
    }

    pub fn into_mesh(self) -> Mesh {
        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::default(),
        )
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, self.positions)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, self.normals)
        .with_inserted_attribute(Mesh::ATTRIBUTE_UV_0, self.uvs)
        .with_inserted_indices(Indices::U32(self.indices));
        mesh.generate_tangents()
            .expect("procedural indoor meshes have valid normals and UVs");
        mesh
    }
}
