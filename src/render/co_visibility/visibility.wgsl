struct Camera {
    clip_from_world: mat4x4<f32>,
    view_from_world: mat4x4<f32>,
    world_from_view: mat4x4<f32>,
    limits: vec4<f32>, // near, far, width, height
    rgb: vec4<f32>, // additive sRGB code in [0,1]
}
struct Cameras {
    views: array<Camera, 16>,
    info: vec4<u32>, // count
}
@group(0) @binding(0) var<uniform> cameras: Cameras;
@group(0) @binding(1) var world_depth: texture_2d_array<f32>;
@group(0) @binding(2) var normal_semantic: texture_2d_array<f32>;
@group(0) @binding(3) var output: texture_storage_2d_array<rgba32float, write>;

@compute @workgroup_size(8, 8, 1)
fn visibility(@builtin(global_invocation_id) id: vec3<u32>) {
    let source = id.z;
    if source >= cameras.info.x { return; }
    let camera = cameras.views[source];
    if id.x >= u32(camera.limits.z) || id.y >= u32(camera.limits.w) { return; }
    let pixel = vec2<i32>(id.xy);
    let point = textureLoad(world_depth, pixel, i32(source), 0);
    if point.w <= 0.0 {
        textureStore(output, pixel, i32(source), vec4<f32>(0.0));
        return;
    }
    let source_normal = normalize((camera.world_from_view * vec4<f32>(
        textureLoad(normal_semantic, pixel, i32(source), 0).xyz * 2.0 - 1.0, 0.0)).xyz);
    var mask = 0u;
    for (var other_index = 0u; other_index < cameras.info.x; other_index++) {
        if other_index == source { continue; }
        let other = cameras.views[other_index];
        let clip = other.clip_from_world * vec4<f32>(point.xyz, 1.0);
        let depth = -(other.view_from_world * vec4<f32>(point.xyz, 1.0)).z;
        if clip.w <= 0.0 || depth < other.limits.x || depth > other.limits.y { continue; }
        let uv = clip.xy / clip.w * vec2<f32>(0.5, -0.5) + 0.5;
        if any(uv < vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) { continue; }
        let target_pixel = vec2<i32>(uv * other.limits.zw);
        let hit = textureLoad(world_depth, target_pixel, i32(other_index), 0);
        if hit.w <= 0.0 { continue; }
        let target_normal = normalize((other.world_from_view * vec4<f32>(
            textureLoad(normal_semantic, target_pixel, i32(other_index), 0).xyz * 2.0 - 1.0, 0.0)).xyz);
        // Comparing tangent planes avoids false occlusion on sloped surfaces
        // caused by comparing depths along different pixel-center rays.
        let delta = hit.xyz - point.xyz;
        let tolerance = max(0.001, 0.0001 * depth);
        if max(abs(dot(delta, source_normal)), abs(dot(delta, target_normal))) <= tolerance {
            mask |= 1u << other_index;
        }
    }
    textureStore(output, pixel, i32(source), vec4<f32>(f32(mask), f32(countOneBits(mask)), 1.0, 0.0));
}
