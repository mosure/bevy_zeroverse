import argparse
import random
import os
import numpy as np
from manifold3d import Manifold, OpType
import open3d as o3d
import trimesh

def generate_random_object(num_primitives=5, scale_noise=1.0, position_noise=4.0, rotation_noise=60.0):
    primitives = []
    boolean_operations = [OpType.Add, OpType.Add, OpType.Subtract]  # OpType.Intersect can be added if needed

    for i in range(num_primitives):
        shape_type = random.choice(['cube', 'cylinder', 'sphere', 'tetrahedron'])
        scale = (2 + random.uniform(-scale_noise, scale_noise),
                 2 + random.uniform(-scale_noise, scale_noise),
                 2 + random.uniform(-scale_noise, scale_noise))
        position = (random.uniform(-position_noise, position_noise),
                    random.uniform(-position_noise, position_noise),
                    random.uniform(-position_noise, position_noise))
        rotation = (random.uniform(-rotation_noise, rotation_noise),
                    random.uniform(-rotation_noise, rotation_noise),
                    random.uniform(-rotation_noise, rotation_noise))
        boolean_op = random.choice(boolean_operations)

        if shape_type == 'cube':
            primitive = Manifold.cube((1, 1, 1)).scale(scale).translate(position).rotate(rotation)
        elif shape_type == 'cylinder':
            primitive = Manifold.cylinder(1, 1, 1, 50).scale(scale).translate(position).rotate(rotation)
        elif shape_type == 'sphere':
            primitive = Manifold.sphere(1, 50).scale(scale).translate(position).rotate(rotation)
        elif shape_type == 'tetrahedron':
            primitive = Manifold.tetrahedron().scale(scale).translate(position).rotate(rotation)

        primitives.append((primitive, boolean_op))

    # Apply Boolean operations sequentially
    result = primitives[0][0]
    for primitive, op in primitives[1:]:
        if op == OpType.Add:
            result += primitive
        elif op == OpType.Subtract:
            result -= primitive
        elif op == OpType.Intersect:
            result ^= primitive

    return result

def load_pbr_texture(root_path):
    # Get all material classes
    material_classes = [d for d in os.listdir(root_path) if os.path.isdir(os.path.join(root_path, d))]

    if not material_classes:
        raise ValueError(f"No material classes found in {root_path}")

    # Select a random material class
    selected_class = random.choice(material_classes)
    class_path = os.path.join(root_path, selected_class)

    # Get all materials within the selected class
    materials = [d for d in os.listdir(class_path) if os.path.isdir(os.path.join(class_path, d))]

    if not materials:
        raise ValueError(f"No materials found in {class_path}")

    # Select a random material
    selected_material = random.choice(materials)
    material_path = os.path.join(class_path, selected_material)

    # Define the texture types we're looking for
    texture_types = {
        "baseColorTexture": ["basecolor.png", "diffuse.png"],
        "metallicRoughnessTexture": ["metallic_roughness.png"],
        "normalTexture": ["normal.png"],
        "occlusionTexture": ["ao.png"],
        "emissiveTexture": ["emissive.png"],
        "roughnessTexture": ["roughness.png"],
        "displacementTexture": ["displacement.png", "height.png"]
    }

    texture_files = {}

    # Find the textures
    for texture_type, possible_names in texture_types.items():
        for file in os.listdir(material_path):
            if file.lower() in possible_names:
                texture_files[texture_type] = os.path.join(material_path, file)
                break

    print(f"Selected material: {selected_class}/{selected_material}")
    return texture_files

def save_mesh_as_obj(mesh, filename):
    mesh.to_mesh().save_obj(filename)

def plot_mesh_with_open3d(glb_file_path):
    # Load the GLB file
    scene = o3d.io.read_triangle_mesh(glb_file_path)

    # Compute vertex normals if they're not already present
    if not scene.has_vertex_normals():
        scene.compute_vertex_normals()

    # Create a visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add the loaded scene to the visualizer
    vis.add_geometry(scene)

    # Optimize the camera view
    vis.get_view_control().set_front([0, 0, -1])
    vis.get_view_control().set_up([0, 1, 0])
    vis.get_view_control().set_zoom(0.7)

    # Run the visualizer
    vis.run()
    vis.destroy_window()

def manifold2trimesh(manifold):
    mesh = manifold.to_mesh()

    vertices = np.array(mesh.vert_properties)[:, :3]
    faces = np.array(mesh.tri_verts).reshape(-1, 3)

    # Generate simple UV coordinates based on vertex positions
    uv_coordinates = vertices[:, :2]  # Use X and Y coordinates as U and V
    uv_coordinates = (uv_coordinates - uv_coordinates.min(axis=0)) / (uv_coordinates.max(axis=0) - uv_coordinates.min(axis=0))

    return trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        visual=trimesh.visual.TextureVisuals(uv=uv_coordinates)
    )

def apply_pbr_material(trimesh_mesh, root_path):
    texture_files = load_pbr_texture(root_path)

    material = trimesh.visual.material.PBRMaterial()

    if "baseColorTexture" in texture_files:
        material.baseColorTexture = trimesh.load(texture_files["baseColorTexture"])
    if "metallicRoughnessTexture" in texture_files:
        material.metallicRoughnessTexture = trimesh.load(texture_files["metallicRoughnessTexture"])
    if "normalTexture" in texture_files:
        material.normalTexture = trimesh.load(texture_files["normalTexture"])
    if "occlusionTexture" in texture_files:
        material.occlusionTexture = trimesh.load(texture_files["occlusionTexture"])
    if "emissiveTexture" in texture_files:
        material.emissiveTexture = trimesh.load(texture_files["emissiveTexture"])

    # Handle roughness separately if it's not combined with metallic
    # if "roughnessTexture" in texture_files:
    #     material.roughnessTexture = trimesh.load(texture_files["roughnessTexture"])

    # Handle displacement if needed
    if "displacementTexture" in texture_files:
        # You might need to implement displacement mapping separately
        pass

    trimesh_mesh.visual = trimesh.visual.TextureVisuals(material=material)

def save_mesh_as_glb(mesh, filename, material_path):
    trimesh_mesh = manifold2trimesh(mesh)
    apply_pbr_material(trimesh_mesh, material_path)
    trimesh_mesh.export(filename)

def main(args):
    random_object = generate_random_object(args.num_primitives, args.scale_noise, args.position_noise, args.rotation_noise)

    output_filename = 'random_object.glb'
    save_mesh_as_glb(random_object, output_filename, args.material_path)

    if args.view:
        plot_mesh_with_open3d(output_filename)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate a random 3D object with Manifold3D.')

    parser.add_argument('--num_primitives', type=int, default=20, help='Number of primitives to combine.')
    parser.add_argument('--scale_noise', type=float, default=1.5, help='Noise level for scaling primitives.')
    parser.add_argument('--position_noise', type=float, default=4.0, help='Noise level for positioning primitives.')
    parser.add_argument('--rotation_noise', type=float, default=60.0, help='Noise level for rotating primitives.')
    parser.add_argument('--material_path', type=str, default='D:\\data\\mat-synth\\data_resize\\test\\', help='Path to PBR texture images.')
    parser.add_argument('--view', action='store_true', help='View the generated mesh.')

    args = parser.parse_args()
    main(args)
