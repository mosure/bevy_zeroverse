//! Small shader integrations for the pinned Bevy PBR version. Keep StandardMaterial
//! (and its depth, semantic, flow and voxel contracts), without a renderer fork.
use bevy::{prelude::*, shader::Source};

pub(super) struct IndoorShadingPlugin;

/// Offline validation controls. Dense references are deliberately expensive;
/// normal viewers and dataset generation always use `Default`.
#[derive(Resource, Default, Debug, Clone, Copy, clap::ValueEnum, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum GlassFilter {
    #[default]
    Default,
    Legacy,
    Reference,
    LegacyReference,
}

#[derive(Resource)]
pub(crate) struct IndoorShaders {
    handles: [Handle<Shader>; 3],
    installed: [bool; 3],
}

impl IndoorShaders {
    pub(crate) fn ready(&self) -> bool {
        self.installed.into_iter().all(|done| done)
    }
}

impl Plugin for IndoorShadingPlugin {
    fn build(&self, app: &mut App) {
        let Some(server) = app.world().get_resource::<AssetServer>() else {
            return;
        };
        let handles = [
            server.load("embedded://bevy_pbr/render/pbr_fragment.wgsl"),
            server.load("embedded://bevy_pbr/transmission/transmission.wgsl"),
            server.load("embedded://bevy_pbr/render/pbr_functions.wgsl"),
        ];
        app.init_resource::<GlassFilter>()
            .insert_resource(IndoorShaders {
                handles,
                installed: [false; 3],
            })
            .add_systems(PreUpdate, install.run_if(pending));
    }
}

fn pending(state: Res<IndoorShaders>) -> bool {
    !state.ready()
}

fn install(
    mut shaders: ResMut<Assets<Shader>>,
    mut state: ResMut<IndoorShaders>,
    filter: Res<GlassFilter>,
) {
    for i in 0..3 {
        if state.installed[i] {
            continue;
        }
        let Some(mut shader) = shaders.get_mut(&state.handles[i]) else {
            continue;
        };
        let Source::Wgsl(source) = &shader.source else {
            panic!("Expected Bevy 0.19.1 WGSL PBR libraries");
        };
        let updated = if i == 0 {
            clearcoat_prepass(&anisotropy_prepass(source))
        } else if i == 2 {
            attenuation(source)
        } else {
            match *filter {
                GlassFilter::Default => transmission_filter(source),
                GlassFilter::Legacy => source.to_string(),
                GlassFilter::Reference => transmission_filter(source)
                    .replace(
                        "let taps = 2 * #{SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS};",
                        "let taps = 2048;",
                    )
                    .replace("let taps = 16;", "let taps = 2048;"),
                GlassFilter::LegacyReference => legacy_reference(source),
            }
        };
        shader.source = Source::Wgsl(updated.into());
        state.installed[i] = true;
    }
}

fn attenuation(source: &str) -> String {
    // StandardMaterial defines attenuation_color as the remaining transmission
    // at attenuation_distance. Beer-Lambert therefore needs -ln(T)/distance;
    // the upstream pow(1-T,e) approximation severely under-absorbs tinted panes.
    let old = "pow(1.0 - in.material.attenuation_color.rgb, vec3<f32>(E)) / in.material.attenuation_distance";
    assert!(
        source.contains(old),
        "Bevy absorption changed: requalify glass transport"
    );
    source.replace(old, "-log(clamp(in.material.attenuation_color.rgb, vec3(0.000001), vec3(1.0))) / max(in.material.attenuation_distance, 0.000001)")
}

fn legacy_reference(source: &str) -> String {
    // Integrate 64 rotations of the *same* upstream 32-tap kernel, including
    // both checkerboard radii. Increasing upstream num_taps changes that kernel
    // and would confound a noise comparison with a blur-width change.
    let replacements = [
        ("i < num_taps;", "i < num_taps * 64;"),
        ("(i >> 3u)", "((i % num_taps) >> 3u)"),
        (
            "(random_angle + f32(current_spiral) / f32(num_spirals))",
            "(f32(i / num_taps) / 64.0 + f32(current_spiral) / f32(num_spirals))",
        ),
        (
            "f32(pixel_checkboard) * 0.1",
            "f32((i / num_taps) % 2) * 0.1",
        ),
        ("result /= f32(num_taps);", "result /= f32(num_taps * 64);"),
    ];
    replacements
        .into_iter()
        .fold(source.to_string(), |text, (old, new)| {
            assert!(
                text.contains(old),
                "Bevy transmission changed: requalify reference integrator"
            );
            text.replace(old, new)
        })
}

fn anisotropy_prepass(source: &str) -> String {
    // Upstream initializes the tangent frame only when it does NOT load normals
    // from the prepass. With SSAO, hair otherwise normalizes a zero vector. Move
    // the existing anisotropy implementation outside that conditional, retaining
    // bindless/textured/material flags and using the same MikkTSpace frame.
    let start = source
        .find("        // Take anisotropy into account.")
        .expect("Bevy PBR changed: requalify anisotropy prepass integration");
    let end_marker = "#endif  // LOAD_PREPASS_NORMALS";
    let end = start
        + source[start..]
            .find(end_marker)
            .expect("Bevy PBR changed: missing prepass conditional");
    let block = source[start..end]
        // Constant anisotropy also works without Bevy's optional texture
        // bindings (notably the reduced native/WebGPU feature set here).
        .replace("#ifdef PBR_ANISOTROPY_TEXTURE_SUPPORTED\n", "")
        .replace("#endif  // PBR_ANISOTROPY_TEXTURE_SUPPORTED\n", "")
        .replace("        // Adjust based on the anisotropy map", "#ifdef PBR_ANISOTROPY_TEXTURE_SUPPORTED\n        // Adjust based on the anisotropy map")
        .replace("        pbr_input.anisotropy_strength =", "#endif\n        pbr_input.anisotropy_strength =")
        .replace(
        "let anisotropy_T = normalize(TBN *",
        "let anisotropy_frame = pbr_functions::calculate_tbn_mikktspace(pbr_input.world_normal, in.world_tangent);\n        let anisotropy_T = normalize(anisotropy_frame *",
    );
    assert!(block.contains("let anisotropy_frame ="));
    format!(
        "{}{}\n{}{}",
        &source[..start],
        end_marker,
        block,
        &source[end + end_marker.len()..]
    )
}

fn clearcoat_prepass(source: &str) -> String {
    // The prepass stores only the base-layer normal. Bevy initializes the
    // separate coat frame inside !LOAD_PREPASS_NORMALS, leaving it zero with
    // SSAO. A smooth coat must use the geometric normal, not the bumped base.
    let prepass = "    pbr_input.N = prepass_utils::prepass_normal(in.position, 0u);";
    assert!(
        source.contains(prepass),
        "Bevy prepass changed: requalify clearcoat"
    );
    let source = source.replace(
        prepass,
        &format!("{prepass}\n    pbr_input.clearcoat_N = normalize(pbr_input.world_normal);"),
    );
    // Preserve optional coat normal maps too, with the same UVs, flags and
    // tangent frame as the regular path. This also works when downstream
    // consumers enable Bevy's additional material texture features.
    let begin = "#ifdef STANDARD_MATERIAL_CLEARCOAT\n\n        // Note:";
    let end = "#endif  // STANDARD_MATERIAL_CLEARCOAT\n";
    let start = source.find(begin).expect("Bevy clearcoat block changed");
    let finish = start + source[start..].find(end).expect("unclosed clearcoat block") + end.len();
    let block = source[start..finish].replace(
        "            TBN,",
        "            pbr_functions::calculate_tbn_mikktspace(pbr_input.world_normal, in.world_tangent),",
    );
    let without = format!("{}{}", &source[..start], &source[finish..]);
    let marker = "#endif  // LOAD_PREPASS_NORMALS";
    assert!(without.contains(marker));
    without.replace(
        marker,
        &format!("{marker}\n#ifdef VERTEX_UVS\n#ifdef VERTEX_TANGENTS\n{block}\n#endif\n#endif"),
    )
}

fn transmission_filter(source: &str) -> String {
    // Keep upstream Snell refraction, IOR, Fresnel and exposure. Quadrature and
    // coverage are corrected here; attenuation is corrected separately above.
    let marker = "fn fetch_transmissive_background(offset_position:";
    let start = source
        .find(marker)
        .expect("Bevy transmission changed: requalify glass quadrature");
    let body = start + source[start..].find('{').unwrap();
    let mut depth = 0;
    let end = source[body..]
        .char_indices()
        .find_map(|(offset, c)| {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            (depth == 0).then_some(body + offset + 1)
        })
        .expect("unclosed Bevy transmission function");
    format!(
        "{}{}{}",
        &source[..start],
        include_str!("shading/transmission.wgsl"),
        &source[end..]
    )
}
