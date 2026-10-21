use super::*;

// Frozen triangle-partition implementation: verifies the compact-index builder
// preserves spatial partitions, conservative bounds and all triangle payloads.
fn reference_split(triangles: &mut [Triangle], base: usize, nodes: &mut Vec<Node>) -> u32 {
    let id = nodes.len() as u32;
    nodes.push(Node::default());
    let mut lo = Vec3::splat(f32::INFINITY);
    let mut hi = Vec3::splat(f32::NEG_INFINITY);
    for t in triangles.iter() {
        for p in t.p {
            let p = Vec3::new(p[0], p[1], p[2]);
            lo = lo.min(p);
            hi = hi.max(p);
        }
    }
    let eps = Vec3::splat(0.00001) * (hi.abs().max(lo.abs()) + Vec3::ONE);
    let mut node = Node {
        lo: (lo - eps).extend(0.).to_array(),
        hi: (hi + eps).extend(0.).to_array(),
        info: [0, 0, base as u32, triangles.len() as u32],
    };
    if triangles.len() > 8 {
        let d = hi - lo;
        let axis = if d.x >= d.y && d.x >= d.z {
            0
        } else if d.y >= d.z {
            1
        } else {
            2
        };
        let mid = triangles.len() / 2;
        triangles.select_nth_unstable_by(mid, |a, b| {
            let ca: f32 = a.p.iter().map(|p| p[axis]).sum();
            let cb: f32 = b.p.iter().map(|p| p[axis]).sum();
            ca.total_cmp(&cb).then(a.info[3].cmp(&b.info[3]))
        });
        let (a, b) = triangles.split_at_mut(mid);
        node.info = [
            reference_split(a, base, nodes),
            reference_split(b, base + mid, nodes),
            0,
            0,
        ];
    }
    nodes[id as usize] = node;
    id
}

fn triangles(count: usize, coincident: bool) -> Vec<Triangle> {
    (0..count)
        .map(|i| {
            let p = if coincident {
                Vec3::ZERO
            } else {
                Vec3::new(
                    ((i * 37) % 113) as f32 * 0.13 - 7.,
                    ((i * 11) % 71) as f32 * 0.07,
                    ((i * 19) % 97) as f32 * 0.19 - 8.,
                )
            };
            Triangle {
                p: [
                    p.extend(0.).to_array(),
                    (p + Vec3::X * 0.07).extend(0.).to_array(),
                    (p + Vec3::Y * 0.03).extend(0.).to_array(),
                ],
                n: [Vec3::Z.extend(0.).to_array(); 3],
                info: [i as u32 % 29, i as u32 % 3, i as u32 % 5, i as u32],
            }
        })
        .collect()
}

#[test]
fn compact_builder_preserves_reference_tree_and_triangle_bits() {
    for count in [0, 1, 8, 9, 73, 1024, 5000] {
        for coincident in [false, true] {
            let input = triangles(count, coincident);
            let mut reference = input.clone();
            let mut expected = Vec::new();
            if count > 0 {
                reference_split(&mut reference, 0, &mut expected);
            }
            let (actual, nodes) = finish(input.clone());
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&nodes),
                bytemuck::cast_slice::<_, u8>(&expected),
                "tree count={count} coincident={coincident}"
            );
            for node in &nodes {
                let [_, _, start, len] = node.info;
                if len == 0 {
                    continue;
                }
                let range = start as usize..(start + len) as usize;
                let mut a = actual[range.clone()].to_vec();
                let mut b = reference[range].to_vec();
                a.sort_unstable_by_key(|t| t.info[3]);
                b.sort_unstable_by_key(|t| t.info[3]);
                assert_eq!(
                    bytemuck::cast_slice::<_, u8>(&a),
                    bytemuck::cast_slice::<_, u8>(&b)
                );
            }
            let (replay, _) = finish(input);
            assert_eq!(
                bytemuck::cast_slice::<_, u8>(&actual),
                bytemuck::cast_slice::<_, u8>(&replay)
            );
        }
    }
}

#[test]
fn shared_vertex_transform_matches_per_triangle_expansion() {
    let mut geometry = ExtractedGeometry::default();
    for i in 0..3 {
        let world = Mat4::from_scale_rotation_translation(
            Vec3::new(0.3 + i as f32, 2., 0.7),
            Quat::from_euler(EulerRot::XYZ, 0.4, -0.7, 1.2),
            Vec3::new(3., -2., i as f32),
        );
        geometry.instances.push(Instance {
            world_from_local: world.to_cols_array_2d(),
            normal_from_local: world.inverse().transpose().to_cols_array_2d(),
            semantic: 7 + i,
            padding: [0; 3],
        });
        let start = geometry.indices.len() as u32;
        let base = geometry.vertices.len() as u32;
        for p in [Vec3::ZERO, Vec3::X, Vec3::Y, Vec3::ONE] {
            geometry.vertices.push(Vertex {
                position: p.to_array(),
                normal: (Vec3::Z + p * 0.2).normalize().to_array(),
                instance: i,
            });
        }
        geometry
            .indices
            .extend([0, 1, 2, 1, 3, 2].map(|v| v + base));
        geometry.batches.push(Batch {
            indices: start..start + 6,
            cull: i as usize,
            layers: RenderLayers::layer(i as usize),
        });
    }
    let (triangles, _) = build(&geometry);
    for t in triangles {
        let offset = t.info[3] as usize * 3;
        for (j, &index) in geometry.indices[offset..offset + 3].iter().enumerate() {
            let v = geometry.vertices[index as usize];
            let instance = geometry.instances[v.instance as usize];
            let p = Mat4::from_cols_array_2d(&instance.world_from_local)
                .transform_point3(Vec3::from_array(v.position))
                .extend(0.)
                .to_array();
            let n = Mat4::from_cols_array_2d(&instance.normal_from_local)
                .transform_vector3(Vec3::from_array(v.normal))
                .normalize_or_zero()
                .extend(0.)
                .to_array();
            assert_eq!(t.p[j], p);
            assert_eq!(t.n[j], n);
            assert_eq!(t.info[..3], [instance.semantic, v.instance, v.instance]);
        }
    }
}
