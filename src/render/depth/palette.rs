//! Inject the shared display palette into the renderer shader.
use bevy_zeroverse_capture::depth::TURBO_COEFFICIENTS;

pub(super) fn shader(source: &str) -> String {
    let coefficients = TURBO_COEFFICIENTS
        .iter()
        .enumerate()
        .map(|(i, c)| {
            format!(
                "const TURBO_{i}: vec3<f32> = vec3<f32>({:?}, {:?}, {:?});\n",
                c[0], c[1], c[2]
            )
        })
        .collect::<String>();
    source.replace("// TURBO_COEFFICIENTS", &coefficients)
}
