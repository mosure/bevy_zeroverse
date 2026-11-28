fn area_average(input_image: texture_2d<f32>, pixel: vec2<i32>, factor: u32) -> vec4<f32> {
    let origin = pixel * i32(factor);
    var value = vec4<f32>(0.0);
    // The capture resolver normally uses native resolution or a 2x box.
    // Keep the original row-major additions, including the initial zero.
    if factor == 1u {
        return value + textureLoad(input_image, origin, 0);
    }
    if factor == 2u {
        value += textureLoad(input_image, origin, 0);
        value += textureLoad(input_image, origin + vec2<i32>(1, 0), 0);
        value += textureLoad(input_image, origin + vec2<i32>(0, 1), 0);
        value += textureLoad(input_image, origin + vec2<i32>(1, 1), 0);
        return value * 0.25;
    }
    // The validated larger factors are 3 and 4. Keep one loop over rows.
    for (var y = 0u; y < factor; y++) {
        let row = origin + vec2<i32>(0, i32(y));
        value += textureLoad(input_image, row, 0);
        value += textureLoad(input_image, row + vec2<i32>(1, 0), 0);
        value += textureLoad(input_image, row + vec2<i32>(2, 0), 0);
        if factor == 4u {
            value += textureLoad(input_image, row + vec2<i32>(3, 0), 0);
        }
    }
    return value / f32(factor * factor);
}
