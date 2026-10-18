use super::*;

fn geometry() -> ExtractedGeometry {
    ExtractedGeometry {
        generation: 1,
        vertices: vec![
            Vertex {
                position: [1., 2., 3.],
                normal: [0., 1., 0.],
                instance: 0
            };
            3
        ],
        instances: vec![Instance {
            world_from_local: Mat4::IDENTITY.to_cols_array_2d(),
            normal_from_local: Mat4::IDENTITY.to_cols_array_2d(),
            semantic: 1,
            padding: [0; 3],
        }],
        indices: vec![0, 1, 2],
        topology: vec![ObjectTopology {
            entity: Entity::PLACEHOLDER,
            mesh: Handle::<Mesh>::default().id(),
            vertices: 0..3,
            indices_hash: 7,
        }],
        ..default()
    }
}

#[test]
fn stationary_flow_requires_exact_generation_and_transforms() {
    let mut geometry = geometry();
    let snapshot = Snapshot::new(&geometry);
    assert!(snapshot.matches(&geometry));
    geometry.generation += 1;
    assert!(
        !snapshot.matches(&geometry),
        "an unqualified mesh change cannot reuse flow"
    );
    geometry.deformation = Some(deformation::VertexUpdate {
        base_generation: 1,
        ranges: vec![],
    });
    assert!(snapshot.matches(&geometry), "unchanged skin pose");
    geometry.deformation.as_mut().unwrap().ranges.push(0..3);
    assert!(!snapshot.matches(&geometry), "changed skin pose");
    geometry.deformation.as_mut().unwrap().ranges.clear();
    geometry.deformation.as_mut().unwrap().base_generation = 0;
    assert!(
        !snapshot.matches(&geometry),
        "missing intermediate generation"
    );
    geometry.generation = 1;
    geometry.instances[0].world_from_local[3][0] = -0.0;
    assert!(
        !snapshot.matches(&geometry),
        "transform comparison preserves signed zero"
    );
    geometry.instances[0].world_from_local = Mat4::IDENTITY.to_cols_array_2d();
    geometry.instances.push(geometry.instances[0]);
    assert!(!snapshot.matches(&geometry), "instance insertion");
}

#[test]
fn moving_to_stationary_flow_rebuilds_once_then_reuses_exact_correspondence() {
    let mut geometry = geometry();
    let mut snapshot = Snapshot::new(&geometry);
    let mut stationary_pair = false;
    let mut retained = Vec::new();
    let mut builds = 0;
    for translation in [0., 0., 0.5, 0.5, 0.5, 0.5, 1., 1., 1.] {
        geometry.instances[0].world_from_local =
            Mat4::from_translation(Vec3::new(translation, 0., 0.)).to_cols_array_2d();
        let original = Snapshot::new(&geometry);
        let expected = snapshot.correspondence(&original);
        let unchanged = snapshot.matches(&geometry);
        if !(unchanged && stationary_pair) {
            retained = snapshot.correspondence(if unchanged { &snapshot } else { &original });
            builds += 1;
        }
        assert_eq!(
            bytemuck::cast_slice::<_, u8>(&retained),
            bytemuck::cast_slice::<_, u8>(&expected)
        );
        stationary_pair = unchanged;
        snapshot = original;
    }
    assert_eq!(
        builds, 5,
        "moving -> stationary must discard the preceding moving pair"
    );
}
