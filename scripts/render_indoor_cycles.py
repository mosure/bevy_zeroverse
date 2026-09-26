#!/usr/bin/env python3
"""Independent offline reference for indoor_validate --export-reference.

Run with Blender, not system Python:
  blender -b --factory-startup --python scripts/render_indoor_cycles.py -- \
    --scene out/run/seed_000000/reference/scene.json --samples 256 --device OPTIX

Keeps the actual mesh assemblies, maps, world transforms and camera calibration.
Cycles traces emission, indirect diffuse/specular light and solid glass. Raster
spot/point proxies and the preconvolved indoor reflection cubemap are deliberately
replaced by emissive surfaces and the baker's external hemispherical sky. Those
model differences are recorded, so image differences are not called renderer error.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from indoor_reference_contract import sky_radiance, validate_snapshot

import bpy
from mathutils import Matrix, Vector
import numpy as np


def matrix(columns):
    return Matrix(columns).transposed()


def connect(tree, a, output, b, input_name):
    tree.links.new(a.outputs[output], b.inputs[input_name])


def material(definition, directory, index):
    m = bpy.data.materials.new(f"material_{index:03}")
    m.use_nodes = True
    tree = m.node_tree
    tree.nodes.clear()
    output = tree.nodes.new("ShaderNodeOutputMaterial")
    shader = tree.nodes.new("ShaderNodeBsdfPrincipled")
    connect(tree, shader, "BSDF", output, "Surface")
    shader.inputs["Base Color"].default_value = definition["base_color"]
    shader.inputs["Roughness"].default_value = definition["roughness"]
    shader.inputs["Metallic"].default_value = definition["metallic"]
    shader.inputs["Anisotropic"].default_value = definition.get("anisotropy_strength", 0.0)
    shader.inputs["Anisotropic Rotation"].default_value = definition.get("anisotropy_rotation", 0.0) / (2 * math.pi)
    shader.inputs["IOR"].default_value = definition["ior"]
    f0 = ((definition["ior"] - 1) / (definition["ior"] + 1)) ** 2
    shader.inputs["Specular IOR Level"].default_value = min(
        1, 0.5 * 0.16 * definition["reflectance"] ** 2 / max(f0, 1e-8))
    shader.inputs["Transmission Weight"].default_value = definition["specular_transmission"]
    shader.inputs["Emission Color"].default_value = definition["emissive"]
    shader.inputs["Emission Strength"].default_value = 1
    # Metric UV transform, followed by Bevy (top-left) -> Blender (bottom-left).
    uv = tree.nodes.new("ShaderNodeTexCoord")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    connect(tree, uv, "UV", separate, "Vector")
    rows = np.array(definition["uv_matrix"]).T
    combine = tree.nodes.new("ShaderNodeCombineXYZ")
    for axis, row in zip(("X", "Y"), rows):
        products = []
        for source, coefficient in zip(("X", "Y"), row[:2]):
            node = tree.nodes.new("ShaderNodeMath")
            node.operation = "MULTIPLY"
            connect(tree, separate, source, node, 0)
            node.inputs[1].default_value = coefficient * (-1 if axis == "Y" else 1)
            products.append(node)
        addition = tree.nodes.new("ShaderNodeMath")
        addition.operation = "ADD"
        connect(tree, products[0], 0, addition, 0)
        connect(tree, products[1], 0, addition, 1)
        shift = tree.nodes.new("ShaderNodeMath")
        shift.operation = "ADD"
        connect(tree, addition, 0, shift, 0)
        shift.inputs[1].default_value = 1 - row[2] if axis == "Y" else row[2]
        connect(tree, shift, 0, combine, axis)

    def texture(name):
        item = definition[name]
        if item is None:
            return None
        image = bpy.data.images.load(str(directory / item["path"]), check_existing=True)
        image.colorspace_settings.name = "sRGB" if item["srgb"] else "Non-Color"
        node = tree.nodes.new("ShaderNodeTexImage")
        node.image = image
        node.extension = "REPEAT"
        node.interpolation = "Linear"
        connect(tree, combine, "Vector", node, "Vector")
        return node

    base = texture("base_color_texture")
    base_product = None
    if base:
        multiply = tree.nodes.new("ShaderNodeVectorMath")
        multiply.operation = "MULTIPLY"
        connect(tree, base, "Color", multiply, 0)
        multiply.inputs[1].default_value = definition["base_color"][:3]
        connect(tree, multiply, "Vector", shader, "Base Color")
        base_product = multiply
    normal = texture("normal_map_texture")
    if normal:
        node = tree.nodes.new("ShaderNodeNormalMap")
        connect(tree, normal, "Color", node, "Color")
        connect(tree, node, "Normal", shader, "Normal")
    data = texture("metallic_roughness_texture")
    if data:
        split = tree.nodes.new("ShaderNodeSeparateColor")
        connect(tree, data, "Color", split, "Color")
        for channel, target, factor in [
            ("Green", "Roughness", definition["roughness"]),
            ("Blue", "Metallic", definition["metallic"]),
        ]:
            node = tree.nodes.new("ShaderNodeMath")
            node.operation = "MULTIPLY"
            connect(tree, split, channel, node, 0)
            node.inputs[1].default_value = factor
            connect(tree, node, 0, shader, target)
    emission = texture("emissive_texture")
    if emission:
        multiply = tree.nodes.new("ShaderNodeVectorMath")
        multiply.operation = "MULTIPLY"
        connect(tree, emission, "Color", multiply, 0)
        multiply.inputs[1].default_value = definition["emissive"][:3]
        connect(tree, multiply, "Vector", shader, "Emission Color")
    if definition["diffuse_transmission"] > 0:
        translucent = tree.nodes.new("ShaderNodeBsdfTranslucent")
        translucent.inputs["Color"].default_value = definition["base_color"]
        if base_product:
            connect(tree, base_product, "Vector", translucent, "Color")
        mix = tree.nodes.new("ShaderNodeMixShader")
        mix.inputs[0].default_value = definition["diffuse_transmission"]
        connect(tree, shader, "BSDF", mix, 1)
        connect(tree, translucent, "BSDF", mix, 2)
        connect(tree, mix, "Shader", output, "Surface")
    if definition["alpha_mode"] != "Opaque":
        raise ValueError("Use native Auto quality for optical references; alpha approximations are unsupported")
    if definition["unlit"]:
        raise ValueError("Unlit materials have no physical reference mapping")
    return m


def load_geometry(document, directory, materials):
    raw = (directory / "geometry.bin").read_bytes()
    if len(raw) != document["geometry_bytes"]:
        raise ValueError("Truncated or mismatched geometry.bin")
    meshes = []
    triangles = 0
    for index, definition in enumerate(document["meshes"]):
        n, k, offset = (definition[x] for x in ("vertices", "indices", "offset"))
        arrays = []
        for width in (3, 3, 2):
            arrays.append(np.frombuffer(raw, dtype="<f4", count=n * width, offset=offset).reshape(n, width))
            offset += n * width * 4
        positions, normals, uvs = arrays
        indices = np.frombuffer(raw, dtype="<u4", count=k, offset=offset).reshape(-1, 3)
        if not all(np.isfinite(a).all() for a in arrays) or indices.max() >= n:
            raise ValueError("Invalid exported geometry")
        mesh = bpy.data.meshes.new(f"mesh_{index:04}")
        mesh.from_pydata(positions.tolist(), [], indices.tolist())
        mesh.update()
        mesh.polygons.foreach_set("use_smooth", [True] * len(mesh.polygons))
        mesh.normals_split_custom_set_from_vertices(normals.tolist())
        uv = mesh.uv_layers.new(name="UVMap")
        uv.data.foreach_set("uv", uvs[indices.reshape(-1)].reshape(-1))
        meshes.append(mesh)
        triangles += len(indices)
    for index, instance in enumerate(document["instances"]):
        mesh = meshes[instance["mesh"]]
        # Export currently has one material per mesh; object linking also supports
        # future mesh instancing with independently assigned material roles.
        if len(mesh.materials) == 0:
            mesh.materials.append(materials[instance["material"]])
        obj = bpy.data.objects.new(f'{index:04}_{instance["name"]}', mesh)
        bpy.context.collection.objects.link(obj)
        obj.material_slots[0].link = "OBJECT"
        obj.material_slots[0].material = materials[instance["material"]]
        obj.matrix_world = matrix(instance["world_from_mesh"])
        # Use ordinary transport. Cycles' approximate MNEE path failed the
        # planar-glass sunlight calibration on both CPU and OptiX.
        obj.cycles.is_caustics_caster = False
        obj.cycles.is_caustics_receiver = False
    return triangles


def lighting(document, manifest):
    # Same hemisphere radiance as the native baker, integrated through real
    # openings here. Preconvolved indoor environment lighting is not a sky.
    sky = sky_radiance(manifest)
    world = bpy.data.worlds.new("exterior_hemisphere")
    world.cycles.is_caustics_light = False
    bpy.context.scene.world = world
    world.use_nodes = True
    tree = world.node_tree
    background = tree.nodes.get("Background")
    background.inputs["Color"].default_value = (*sky, 1)
    coordinate = tree.nodes.new("ShaderNodeTexCoord")
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    connect(tree, coordinate, "Normal", separate, "Vector")
    positive = tree.nodes.new("ShaderNodeMath")
    # World Normal points against the escaping ray. Verified by the analytic
    # +Y receiver in validate_cycles_radiometry.py (rho * hemispherical radiance).
    positive.operation = "LESS_THAN"
    connect(tree, separate, "Y", positive, 0)
    connect(tree, positive, 0, background, "Strength")
    for index, definition in enumerate(document["lights"]):
        if definition["kind"] != "sun":
            continue  # Physical emitters replace the raster proxies; no double counting.
        data = bpy.data.lights.new(f"sun_{index}", "SUN")
        data.energy = definition["illuminance"]
        data.color = definition["color"][:3]
        data.angle = math.radians(0.53)
        data.cycles.is_caustics_light = False
        obj = bpy.data.objects.new(data.name, data)
        bpy.context.collection.objects.link(obj)
        obj.matrix_world = matrix(definition["world_from_light"])


def setup_device(device):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    if device == "CPU":
        scene.cycles.device = "CPU"
        return ["CPU"]
    preferences = bpy.context.preferences.addons["cycles"].preferences
    preferences.compute_device_type = device
    preferences.get_devices()
    selected = []
    for item in preferences.devices:
        item.use = item.type == device
        if item.use:
            selected.append(item.name)
    if not selected:
        raise RuntimeError(f"No {device} device; explicit CPU selection is required for fallback")
    scene.cycles.device = "GPU"
    return selected


def geometry_audit(document, directory):
    """Independent Blender BVH first-surface intersections versus native float32 labels."""
    bpy.context.view_layer.update()
    depsgraph = bpy.context.evaluated_depsgraph_get()
    scene = bpy.context.scene
    width, height = document["image_size"]
    bounds = document.get("annotation_aabb")
    if bounds is None:  # Initial v1 exports stored the same bounds in capture.json.
        bounds = json.loads((directory.parent / "capture.json").read_text())["aabb"]
    bounds = np.array(bounds, dtype=np.float64)
    results = []
    for view in document["cameras"]:
        path = directory.parent / f'view_{view["index"]:02}_position.rgba32f'
        if not path.exists():
            continue
        native = np.fromfile(path, dtype="<f4").reshape(height, width, 4)
        transform = matrix(view["world_from_view"])
        origin = transform.translation
        fy = height / (2 * math.tan(view["fov_y"] / 2))
        errors = []
        hit_mismatches = 0
        tested = 0
        for y in range(9, height, 23):
            for x in range(11, width, 23):
                direction = (transform.to_3x3() @ Vector(((x + .5 - width / 2) / fy,
                             (height / 2 - y - .5) / fy, -1))).normalized()
                hit, point, *_ = scene.ray_cast(depsgraph, origin, direction, distance=view["far"])
                valid = native[y, x, 3] > .5
                tested += 1
                hit_mismatches += int(hit != valid)
                if hit and valid:
                    native_world = native[y, x, :3] * (bounds[1] - bounds[0]) + bounds[0]
                    errors.append(float(np.linalg.norm(np.array(point) - native_world)))
        results.append({"view": view["index"], "tested": tested, "matched_hits": len(errors),
                        "hit_mismatches": hit_mismatches,
                        "position_error_p99_m": float(np.quantile(errors, .99)) if errors else None,
                        "position_error_max_m": max(errors, default=None)})
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", choices=["OPTIX", "CUDA", "CPU"], default="OPTIX")
    parser.add_argument("--samples", type=int, default=256)
    parser.add_argument("--bounces", type=int, default=12)
    parser.add_argument("--denoise", action="store_true")
    parser.add_argument("--save-blend", action="store_true")
    parser.add_argument("--camera-indices", type=int, nargs="+")
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--radiometry-report", type=Path, required=True)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    if not 1 <= args.samples <= 65536 or not 1 <= args.bounces <= 64:
        parser.error("samples must be 1..65536 and bounces 1..64")
    calibration = json.loads(args.radiometry_report.read_text())
    if not calibration.get("passed") or calibration.get("approximate_caustics"):
        raise ValueError("A passing ordinary-path radiometry control is required")
    if calibration["backend"] != args.device or calibration["blender"] != bpy.app.version_string:
        raise ValueError("Radiometry control must match this Blender version and backend")
    args.scene = args.scene.resolve()
    document = json.loads(args.scene.read_text())
    if document["format"] != "zeroverse-reference-v1":
        raise ValueError("Unsupported export version")
    directory = args.scene.parent
    manifest = json.loads((directory.parent / "manifest.json").read_text())
    validate_snapshot(document, manifest)
    output = (args.output or directory / f"cycles_{args.samples}").resolve()
    output.mkdir(parents=True, exist_ok=False)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    started = time.monotonic()
    devices = setup_device(args.device)
    materials = [material(m, directory, i) for i, m in enumerate(document["materials"])]
    triangles = load_geometry(document, directory, materials)
    lighting(document, manifest)
    scene = bpy.context.scene
    scene.render.resolution_x, scene.render.resolution_y = document["image_size"]
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.use_persistent_data = True
    scene.cycles.samples = args.samples
    scene.cycles.use_adaptive_sampling = False  # Fixed samples make convergence comparisons interpretable.
    scene.cycles.use_denoising = args.denoise
    scene.cycles.max_bounces = args.bounces
    scene.cycles.diffuse_bounces = args.bounces
    scene.cycles.glossy_bounces = args.bounces
    scene.cycles.transmission_bounces = args.bounces
    scene.cycles.transparent_max_bounces = args.bounces
    scene.cycles.sample_clamp_direct = 0
    scene.cycles.sample_clamp_indirect = 0
    scene.cycles.blur_glossy = 0
    scene.cycles.use_guiding = args.device == "CPU"
    scene.view_settings.view_transform = "AgX"
    scene.view_settings.look = "AgX - Medium High Contrast"
    scene.view_settings.exposure = -document["ev100"] - math.log2(1.2)
    scene.view_settings.gamma = 1
    data = bpy.data.cameras.new("calibrated_camera")
    camera = bpy.data.objects.new(data.name, data)
    bpy.context.collection.objects.link(camera)
    scene.camera = camera
    data.sensor_fit = "VERTICAL"
    data.sensor_height = 32
    data.dof.use_dof = False  # Same pinhole contract as native labels.
    audit = geometry_audit(document, directory)
    views = []
    for view in document["cameras"]:
        if args.camera_indices is not None and view["index"] not in args.camera_indices:
            continue
        camera.matrix_world = matrix(view["world_from_view"])
        data.lens = data.sensor_height / (2 * math.tan(view["fov_y"] / 2))
        data.clip_start, data.clip_end = view["near"], view["far"]
        scene.cycles.seed = (document["seed"] * 1009 + view["index"] * 1013 + args.seed_offset) % 2147483647
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.color_mode = "RGBA"
        scene.render.filepath = str(output / f'view_{view["index"]:02}.exr')
        before = time.monotonic()
        bpy.ops.render.render(write_still=True)
        # Portable top-left-origin float output lets ordinary NumPy tooling
        # compare linear radiance without an OpenEXR/Blender Python dependency.
        linear = bpy.data.images.load(scene.render.filepath, check_existing=False)
        width, height = document["image_size"]
        values = np.array(linear.pixels[:], dtype=np.float32).reshape(height, width, 4)
        if not np.isfinite(values).all():
            raise RuntimeError("Non-finite Cycles radiance")
        values[::-1].astype("<f4").tofile(output / f'view_{view["index"]:02}.rgba32f')
        bpy.data.images.remove(linear)
        result = bpy.data.images["Render Result"]
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
        result.save_render(str(output / f'view_{view["index"]:02}.png'), scene=scene)
        views.append({"index": view["index"], "seconds": time.monotonic() - before,
                      "seed": scene.cycles.seed, "samples": args.samples})
    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(output / "scene.blend"))
    hashes = {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in sorted(directory.glob("textures/*.png"))}
    for p in [args.scene, directory / "geometry.bin"]:
        hashes[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    hashes["../manifest.json"] = hashlib.sha256((directory.parent / "manifest.json").read_bytes()).hexdigest()
    aligned = bool(audit) and all(r["hit_mismatches"] == 0 and
                                  r["position_error_p99_m"] is not None and
                                  r["position_error_p99_m"] < .001 for r in audit)
    report = {"renderer": "Blender Cycles", "blender_version": bpy.app.version_string,
              "blender_hash": bpy.app.build_hash.decode(), "devices": devices,
              "source_sha256": hashes, "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "contract_script_sha256": hashlib.sha256(Path(__file__).with_name("indoor_reference_contract.py").read_bytes()).hexdigest(),
              "radiometry": calibration,
              "radiometry_sha256": hashlib.sha256(args.radiometry_report.read_bytes()).hexdigest(),
              "seed": document["seed"], "capture_engine": document["capture_engine"],
              "samples": args.samples, "bounces": args.bounces, "denoised": args.denoise,
              "shadow_caustics": "Ordinary path tracing; approximate MNEE disabled after failed optical control",
              "path_guiding": args.device == "CPU",
              "triangles": triangles, "instances": len(document["instances"]),
              "image_size": document["image_size"], "views": views, "geometry_audit": audit,
              "geometry_alignment_passed": aligned if audit else None,
              "seconds": time.monotonic() - started,
              "linear_exr": "Unexposed scene-linear RGB photometric proxy radiance; no display transform",
              "linear_raw": "Same unexposed radiance; little-endian RGBA32F; top-left pixel origin",
              "png": "AgX Medium High Contrast; exposure=log2(2^-EV100/1.2); sRGB",
              "differences_from_native": ["Full diffuse/specular/refraction paths", "Finite solar disk",
                  "Actual emissive surfaces replace raster spot/point light proxies",
                  "Visible upper-hemisphere sky replaces indoor reflection cubemap",
                  "No SSAO, bloom, FXAA, probe interpolation or range truncation",
                  "Blender Principled BSDF and AgX differ from Bevy PBR and tone mapping",
                  "Glass absorption is not mapped; RGB transport is not spectral"],
              "state_of_the_art_claim": False}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if audit and not aligned:
        raise RuntimeError("Reference geometry/camera alignment failed; inspect report.json")
    print("ZERO_REFERENCE_COMPLETE", output, flush=True)


if __name__ == "__main__":
    main()
