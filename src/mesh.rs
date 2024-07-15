use bevy::{
    prelude::*,
    render::mesh::VertexAttributeValues,
};
use noise::{NoiseFn, Perlin};
use rand::Rng;


pub fn displace_vertices_with_noise(mesh: &mut Mesh, frequency: f32, scale: f32) {
    let rng = &mut rand::thread_rng();

    let perlin_x = Perlin::new(rng.gen());
    let perlin_y = Perlin::new(rng.gen());
    let perlin_z = Perlin::new(rng.gen());

    let mut positions_attr = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .unwrap()
        .as_float3()
        .unwrap()
        .to_vec();

    for position in positions_attr.iter_mut() {
        let coords = [
            (position[0] * frequency) as f64,
            (position[1] * frequency) as f64,
            (position[2] * frequency) as f64,
        ];

        let n_x = perlin_x.get(coords) * scale as f64;
        position[0] += n_x as f32;

        let n_y = perlin_y.get(coords) * scale as f64;
        position[1] += n_y as f32;

        let n_z = perlin_z.get(coords) * scale as f64;
        position[2] += n_z as f32;
    }

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, VertexAttributeValues::Float32x3(positions_attr));
}
