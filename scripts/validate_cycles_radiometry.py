#!/usr/bin/env python3
"""Analytic radiometric controls for the optional Cycles reference bridge.

blender -b --factory-startup --python scripts/validate_cycles_radiometry.py -- --output out/cycles_controls
CPU is the default; a GPU backend can be qualified explicitly. The narrow solar
disk needs many samples through glass. Approximate MNEE is an optional diagnostic.
"""
import argparse
import json
import math
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from indoor_reference_contract import absorption_coefficients

import bpy
from mathutils import Vector
import numpy as np


def slab_diffuse_reflectance(ior):
    """Cosine-weighted Fresnel reflection of a lossless parallel-sided slab."""
    x, weights = np.polynomial.legendre.leggauss(256)
    cosine = (x + 1) / 2
    transmitted = np.sqrt(1 - (1 - cosine*cosine) / (ior*ior))
    fresnel = .5 * (((cosine-ior*transmitted)/(cosine+ior*transmitted))**2
                    + ((ior*cosine-transmitted)/(ior*cosine+transmitted))**2)
    return float(np.dot(weights / 2, 2*cosine * (2*fresnel/(1+fresnel))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["CPU", "OPTIX", "CUDA"], default="CPU")
    parser.add_argument("--glass-samples", type=int, default=262144)
    parser.add_argument("--seed", type=int, default=73)
    parser.add_argument("--approximate-caustics", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1:])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    reports = []
    diffuse_reflection = slab_diffuse_reflectance(1.5)
    # The 18% floor and the slab reflect light back to each other. Include
    # their infinite-plane radiosity series, not just the first transmitted ray.
    through_glass = 100 * .18 / math.pi * (1 - .04) / (1 + .04)
    through_glass /= 1 - .18 * diffuse_reflection
    for kind, expected in [("emission", 4), ("sun", 100 * .18 / math.pi),
                           ("upper_hemisphere", .18), ("glass", 4 * (1 - .04) / (1 + .04)),
                           ("sun_glass", through_glass),
                           ("absorption", 4*sum(c**0.5 for c in (.4,.7,.9))/3)]:
        bpy.ops.wm.read_factory_settings(use_empty=True)
        scene = bpy.context.scene
        scene.render.engine = "CYCLES"
        scene.cycles.device = "CPU" if args.device == "CPU" else "GPU"
        if args.device != "CPU":
            preferences = bpy.context.preferences.addons["cycles"].preferences
            preferences.compute_device_type = args.device
            preferences.get_devices()
            for item in preferences.devices:
                item.use = item.type == args.device
            if not any(item.use for item in preferences.devices):
                raise RuntimeError(f"No {args.device} device")
        scene.cycles.samples = args.glass_samples if kind == "sun_glass" else 512
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.cycles.sample_clamp_direct = 0
        scene.cycles.sample_clamp_indirect = 0
        scene.cycles.max_bounces = 32
        scene.cycles.transmission_bounces = 32
        scene.cycles.seed = args.seed
        scene.cycles.blur_glossy = 0
        scene.cycles.use_guiding = args.device == "CPU" and not args.approximate_caustics
        scene.cycles.guiding_training_samples = min(32768, scene.cycles.samples)
        scene.render.resolution_x = scene.render.resolution_y = 64
        scene.render.resolution_percentage = 100
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.view_settings.view_transform = "Standard"
        scene.view_settings.exposure = 0
        camera_data = bpy.data.cameras.new("camera")
        camera_data.type = "ORTHO"
        camera_data.ortho_scale = .5
        camera = bpy.data.objects.new("camera", camera_data)
        bpy.context.collection.objects.link(camera)
        camera.location = (0, 0, 2)
        scene.camera = camera
        bpy.ops.mesh.primitive_plane_add(size=100)
        plane = bpy.context.object
        plane.cycles.is_caustics_receiver = args.approximate_caustics
        m = bpy.data.materials.new("control")
        m.use_nodes = True
        tree = m.node_tree
        tree.nodes.clear()
        target = tree.nodes.new("ShaderNodeOutputMaterial")
        if kind in ("emission", "glass", "absorption"):
            shader = tree.nodes.new("ShaderNodeEmission")
            shader.inputs["Color"].default_value = (1, 1, 1, 1)
            shader.inputs["Strength"].default_value = 4
        else:
            shader = tree.nodes.new("ShaderNodeBsdfDiffuse")
            shader.inputs["Color"].default_value = (.18, .18, .18, 1)
            shader.inputs["Roughness"].default_value = 0
        tree.links.new(shader.outputs[0], target.inputs["Surface"])
        plane.data.materials.append(m)
        if kind in ("sun", "sun_glass"):
            light_data = bpy.data.lights.new("sun", "SUN")
            light_data.energy = 100
            light_data.angle = math.radians(.53)
            light_data.cycles.is_caustics_light = args.approximate_caustics
            light = bpy.data.objects.new("sun", light_data)
            bpy.context.collection.objects.link(light)
        if kind == "upper_hemisphere":
            # Same coordinate node as render_indoor_cycles.py, with the receiver
            # aligned to +Y. A constant hemispherical sky has irradiance pi*L.
            plane.rotation_euler[0] = -math.pi / 2
            camera.location = (0, 2, 0)
            camera.rotation_euler = Vector((0, -1, 0)).to_track_quat('-Z', 'Y').to_euler()
            world = bpy.data.worlds.new("hemisphere")
            scene.world = world
            world.use_nodes = True
            tree = world.node_tree
            bg = tree.nodes.get("Background")
            bg.inputs["Color"].default_value = (1, 1, 1, 1)
            coord = tree.nodes.new("ShaderNodeTexCoord")
            separate = tree.nodes.new("ShaderNodeSeparateXYZ")
            positive = tree.nodes.new("ShaderNodeMath")
            positive.operation = "LESS_THAN"
            tree.links.new(coord.outputs["Normal"], separate.inputs["Vector"])
            tree.links.new(separate.outputs["Y"], positive.inputs[0])
            tree.links.new(positive.outputs[0], bg.inputs["Strength"])
        if kind in ("glass", "sun_glass", "absorption"):
            bpy.ops.mesh.primitive_cube_add(size=1, location=(0, 0, .2))
            glass = bpy.context.object
            glass.cycles.is_caustics_caster = args.approximate_caustics
            glass.scale = (100, 100, .008)
            # Cycles MNEE requires the smooth-normal shader flag. Preserve the
            # slab's exact flat face normals through custom corner normals,
            # matching the actual-scene bridge rather than rounding the cube.
            normals = [None] * len(glass.data.loops)
            for polygon in glass.data.polygons:
                polygon.use_smooth = True
                for loop in polygon.loop_indices:
                    normals[loop] = tuple(polygon.normal)
            glass.data.normals_split_custom_set(normals)
            gm = bpy.data.materials.new("glass")
            gm.use_nodes = True
            bsdf = gm.node_tree.nodes.get("Principled BSDF")
            bsdf.inputs["Base Color"].default_value = (1, 1, 1, 1)
            bsdf.inputs["Roughness"].default_value = 0
            bsdf.inputs["IOR"].default_value = 1.5
            bsdf.inputs["Transmission Weight"].default_value = 1
            if kind == "absorption":
                # Isolate volume attenuation from Fresnel with transparent
                # boundaries. Actual slab thickness .008 m, reference .016 m.
                gm.node_tree.nodes.remove(bsdf)
                bsdf = gm.node_tree.nodes.new("ShaderNodeBsdfTransparent")
                output_node = gm.node_tree.nodes.get("Material Output")
                gm.node_tree.links.new(bsdf.outputs[0], output_node.inputs["Surface"])
                sigma = absorption_coefficients((.4,.7,.9), .016)
                density = max(sigma)
                volume = gm.node_tree.nodes.new("ShaderNodeVolumeAbsorption")
                volume.inputs["Color"].default_value = (*[1-s/density for s in sigma],1)
                volume.inputs["Density"].default_value = density
                gm.node_tree.links.new(volume.outputs[0], output_node.inputs["Volume"])
            glass.data.materials.append(gm)
            if kind == "sun_glass":
                # Observe the diffuse receiver from below the slab, measuring
                # illumination transmission separately from camera transmission.
                camera.location.z = .1
                camera_data.clip_start = .001
        scene.render.filepath = str(output / f"{kind}.exr")
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(scene.render.filepath, check_existing=False)
        pixels = np.array(img.pixels[:], dtype=np.float32).reshape(64, 64, 4)[8:-8, 8:-8, :3]
        measured = float(pixels.mean())
        error = abs(measured / expected - 1)
        luminance = pixels @ np.array([.2126, .7152, .0722])
        standard_error = float(luminance.std(ddof=1) / math.sqrt(luminance.size))
        reports.append({"control": kind, "expected_radiance": expected,
                        "measured_radiance": measured, "relative_error": error,
                        "samples": scene.cycles.samples, "seed": args.seed,
                        "pixel_standard_deviation": float(pixels.std()),
                        "mean_standard_error": standard_error,
                        "passed": error < .025 and standard_error / expected < .02})
    report = {"blender": bpy.app.version_string, "backend": args.device,
              "approximate_caustics": args.approximate_caustics,
              "slab_diffuse_reflectance": diffuse_reflection,
              "sun_glass_model": "Two-interface Fresnel transmission and floor/slab radiosity feedback; finite solar disk differs from normal incidence by less than 0.01%",
              "limits": "Pixel standard error is a Monte Carlo diagnostic, not a bound on all rendering bias",
              "radiometry": "RGB photometric proxy scalar units; no wavelength-to-lumen conversion",
              "controls": reports, "passed": all(r["passed"] for r in reports)}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print("ZERO_RADIOMETRY", json.dumps(report), flush=True)
    if not report["passed"]:
        raise RuntimeError("Radiometry bridge calibration failed")


if __name__ == "__main__":
    main()
