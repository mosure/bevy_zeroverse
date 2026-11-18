//! Keep ray intersection data and hit-only attributes in separate storage
//! bindings. Both use the same BVH leaf order and retain all four lanes.
use super::*;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub(super) struct Surface {
    p: [[f32; 4]; 3],
    info: [u32; 4],
}

impl From<&Triangle> for Surface {
    fn from(triangle: &Triangle) -> Self {
        Self {
            p: triangle.p,
            info: triangle.info,
        }
    }
}

type Attributes = [[f32; 4]; 3];

pub(in crate::render::ground_truth) struct TreeBuffers {
    pub surfaces: Buffer,
    pub attributes: Option<Buffer>,
    pub nodes: Buffer,
    pub node_count: u32,
    pub bytes: u64,
}

impl TreeBuffers {
    /// Each pass caches at most one pipeline per record layout.
    pub(in crate::render::ground_truth::angular) fn layout_index(tree: Option<&Self>) -> usize {
        usize::from(tree.is_some_and(|t| t.attributes.is_some()))
    }
}

fn binding_size<T>(count: usize, limit: u64, label: &str) -> Result<u64, String> {
    // An empty read-only array still needs a binding large enough for one element.
    let bytes = (count.max(1) as u64).saturating_mul(size_of::<T>() as u64);
    if bytes > limit {
        Err(format!(
            "fisheye {label} requires {bytes} bytes, above the adapter binding limit of {limit}"
        ))
    } else {
        Ok(bytes)
    }
}

fn upload_array<T: bytemuck::Pod>(
    device: &RenderDevice,
    queue: &RenderQueue,
    label: &str,
    values: &[T],
) -> Buffer {
    let zero = T::zeroed();
    let contents = if values.is_empty() {
        bytemuck::bytes_of(&zero)
    } else {
        bytemuck::cast_slice(values)
    };
    super::super::super::super::upload_buffer(
        device,
        queue,
        &BufferInitDescriptor {
            label: Some(label),
            contents,
            usage: BufferUsages::STORAGE,
        },
    )
}

/// Keep the normal capture path in one ordered upload. Larger meshes use two
/// independently bounded bindings with exactly the same leaf order and values.
enum Records {
    Interleaved(Vec<Triangle>),
    Split {
        surfaces: Vec<Surface>,
        attributes: Vec<Attributes>,
    },
}
pub(in crate::render::ground_truth::angular) struct PackedTree {
    records: Records,
    nodes: Vec<Node>,
}
impl PackedTree {
    pub(in crate::render::ground_truth::angular) fn build(
        triangles: Vec<Triangle>,
        limits: &wgpu::Limits,
    ) -> Result<Self, String> {
        let limit = limits
            .max_storage_buffer_binding_size
            .min(limits.max_buffer_size);
        let fits_interleaved =
            triangles.len().max(1) as u64 <= limit / size_of::<Triangle>() as u64;
        if !fits_interleaved {
            binding_size::<Surface>(triangles.len(), limit, "triangle positions and labels")?;
            binding_size::<Attributes>(triangles.len(), limit, "triangle attributes")?;
        }
        let (order, nodes) = tree_order(&triangles, true);
        binding_size::<Node>(nodes.len(), limit, "BVH nodes")?;
        let pool = bulk::pool(order.len());
        let ordered = |i: usize| &triangles[order[i].index as usize];
        let records = if fits_interleaved {
            Records::Interleaved(bulk::collect(order.len(), pool, |i| *ordered(i)))
        } else {
            Records::Split {
                surfaces: bulk::collect(order.len(), pool, |i| Surface::from(ordered(i))),
                attributes: bulk::collect(order.len(), pool, |i| ordered(i).n),
            }
        };
        Ok(Self { records, nodes })
    }
    pub(in crate::render::ground_truth::angular) fn upload(
        self,
        device: &RenderDevice,
        queue: &RenderQueue,
    ) -> TreeBuffers {
        let (surfaces, attributes, bytes) = match self.records {
            Records::Interleaved(triangles) => (
                upload_array(device, queue, "angular_triangles", &triangles),
                None,
                triangles.len().max(1) * size_of::<Triangle>(),
            ),
            Records::Split {
                surfaces,
                attributes,
            } => (
                upload_array(
                    device,
                    queue,
                    "angular_triangle_positions_and_labels",
                    &surfaces,
                ),
                Some(upload_array(
                    device,
                    queue,
                    "angular_triangle_hit_attributes",
                    &attributes,
                )),
                surfaces.len().max(1) * size_of::<Surface>()
                    + attributes.len().max(1) * size_of::<Attributes>(),
            ),
        };
        TreeBuffers {
            surfaces,
            attributes,
            bytes: (bytes + self.nodes.len().max(1) * size_of::<Node>()) as u64,
            node_count: self.nodes.len() as u32,
            nodes: upload_array(device, queue, "angular_bvh", &self.nodes),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn positions_labels_normals_and_flow_validity_share_the_same_leaf_order() {
        let input: Vec<_> = (0..171u32)
            .map(|id| {
                let x = ((id * 57) % 89) as f32;
                Triangle {
                    p: [[x, 0., -0., 0.], [x + 1., 0., 0., 0.], [x, 0.5, 0., 0.]],
                    n: std::array::from_fn(|k| {
                        [
                            id as f32,
                            k as f32,
                            -0.,
                            if (id as usize + k).is_multiple_of(3) {
                                0.0
                            } else {
                                1.0
                            },
                        ]
                    }),
                    info: [id % 31, id % 3, id / 9, id],
                }
            })
            .collect();
        let regular = PackedTree::build(input.clone(), &wgpu::Limits::default()).unwrap();
        let Records::Interleaved(ordered) = regular.records else {
            panic!("small rooms should retain the interleaved layout")
        };
        let limits = wgpu::Limits {
            max_storage_buffer_binding_size: (input.len() * size_of::<Surface>()) as u64,
            ..Default::default()
        };
        let packed = PackedTree::build(input.clone(), &limits).unwrap();
        let Records::Split {
            surfaces,
            attributes,
        } = packed.records
        else {
            panic!("fixture must exercise the split layout")
        };
        assert_eq!(surfaces.len(), input.len());
        assert_eq!(attributes.len(), input.len());
        assert!(surfaces
            .iter()
            .enumerate()
            .any(|(i, t)| t.info[3] as usize != i));
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&regular.nodes),
            bytemuck::cast_slice::<_, u8>(&packed.nodes)
        );
        for ((surface, attr), interleaved) in surfaces.iter().zip(&attributes).zip(&ordered) {
            assert_eq!(surface.info, interleaved.info);
            assert_eq!(bytemuck::bytes_of(attr), bytemuck::bytes_of(&interleaved.n));
            let source = &input[surface.info[3] as usize];
            assert_eq!(
                bytemuck::bytes_of(surface),
                bytemuck::bytes_of(&Surface::from(source))
            );
            assert_eq!(bytemuck::bytes_of(attr), bytemuck::bytes_of(&source.n));
        }
    }

    #[test]
    fn seed_403_records_fit_in_webgpu_minimum_storage_bindings_without_quantization() {
        let triangles = 1_218_047;
        let limit = 128 * 1024 * 1024;
        assert_eq!((size_of::<Surface>(), size_of::<Attributes>()), (64, 48));
        assert!(binding_size::<Triangle>(triangles, limit, "old interleaved triangle").is_err());
        assert_eq!(
            binding_size::<Surface>(triangles, limit, "positions").unwrap(),
            triangles as u64 * 64
        );
        assert_eq!(
            binding_size::<Attributes>(triangles, limit, "attributes").unwrap(),
            triangles as u64 * 48
        );
        assert!(binding_size::<Surface>(limit as usize / 64 + 1, limit, "positions").is_err());
        assert_eq!(binding_size::<Node>(0, limit, "nodes").unwrap(), 48);
    }
}
