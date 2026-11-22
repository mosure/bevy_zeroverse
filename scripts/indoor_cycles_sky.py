"""Wire the native room-local hemisphere texture into the Cycles world."""
import bpy


def attach(tree, background, coordinate, positive, document, directory):
    sky = document.get("exterior_sky")
    if not sky:
        return
    if sky["mapping"] != "upper_hemisphere_octahedral_v1":
        raise ValueError("Unknown reference sky mapping")
    material = document["materials"][sky["material"]]
    texture = material.get("emissive_texture")
    if not texture:
        raise ValueError("Directional sky material has no emissive texture")

    def math_node(operation, a, b=None):
        node = tree.nodes.new("ShaderNodeMath")
        node.operation = operation
        for index, value in enumerate((a, b)):
            if isinstance(value, bpy.types.NodeSocket):
                tree.links.new(value, node.inputs[index])
            elif value is not None:
                node.inputs[index].default_value = value
        return node.outputs[0]

    rotate = tree.nodes.new("ShaderNodeVectorRotate")
    rotate.rotation_type = "Y_AXIS"
    rotate.inputs["Angle"].default_value = -document["world_yaw"]
    tree.links.new(coordinate.outputs["Normal"], rotate.inputs["Vector"])
    separate = tree.nodes.new("ShaderNodeSeparateXYZ")
    tree.links.new(rotate.outputs["Vector"], separate.inputs[0])
    x, y, z = separate.outputs
    # The world Normal points against the escaping ray in this Y-up scene.
    horizontal = math_node("ADD", math_node("ABSOLUTE", x), math_node("ABSOLUTE", z))
    norm = math_node("ADD", horizontal, math_node("MAXIMUM", math_node("MULTIPLY", y, -1), 0))
    image = bpy.data.images.load(str(directory / texture["path"]), check_existing=False)
    image.colorspace_settings.name = "sRGB" if texture["srgb"] else "Non-Color"
    width, height = image.size
    if width != height or width < 2:
        raise ValueError("Invalid directional sky texture dimensions")
    scale = math_node("DIVIDE", 0.5 * (width - 1) / width, math_node("MAXIMUM", norm, 1e-8))
    uv = tree.nodes.new("ShaderNodeCombineXYZ")
    # Blender's decoded PNG origin is bottom-left.
    tree.links.new(math_node("SUBTRACT", 0.5, math_node("MULTIPLY", math_node("ADD", x, z), scale)), uv.inputs[0])
    tree.links.new(math_node("ADD", 0.5, math_node("MULTIPLY", math_node("SUBTRACT", x, z), scale)), uv.inputs[1])
    sampler = tree.nodes.new("ShaderNodeTexImage")
    sampler.image, sampler.extension, sampler.interpolation = image, "EXTEND", "Linear"
    tree.links.new(uv.outputs[0], sampler.inputs[0])
    emission = tree.nodes.new("ShaderNodeVectorMath")
    emission.operation = "MULTIPLY"
    emission.inputs[1].default_value = material["emissive"][:3]
    tree.links.new(sampler.outputs["Color"], emission.inputs[0])
    tree.links.new(emission.outputs["Vector"], background.inputs["Color"])
    # The finite RGB horizon continues until the modelled grade comes into
    # view. Diffuse irradiance comes exclusively from the upper hemisphere.
    path = tree.nodes.new("ShaderNodeLightPath")
    camera_transmission = math_node(
        "MULTIPLY", path.outputs["Is Transmission Ray"],
        math_node("LESS_THAN", path.outputs["Diffuse Depth"], 0.5),
    )
    visible = math_node("MAXIMUM", path.outputs["Is Camera Ray"], camera_transmission)
    tree.links.new(math_node("MAXIMUM", positive.outputs[0], visible), background.inputs["Strength"])
