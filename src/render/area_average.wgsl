fn area_average(input_image: texture_2d<f32>, pixel: vec2<i32>, factor: u32) -> vec4<f32> {
    let origin = pixel * i32(factor);
    var value = vec4<f32>(0.0);
    for (var y = 0u; y < factor; y++) {
        for (var x = 0u; x < factor; x++) {
            value += textureLoad(input_image, origin + vec2<i32>(i32(x), i32(y)), 0);
        }
    }
    return value / f32(factor * factor);
}
