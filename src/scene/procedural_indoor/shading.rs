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
    handles: [Handle<Shader>; 2],
    installed: [bool; 2],
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
            server.load("embedded://bevy_pbr/render/pbr_fragment.wesl"),
            server.load("embedded://bevy_pbr/transmission.wesl"),
        ];
        app.init_resource::<GlassFilter>()
            .insert_resource(IndoorShaders {
                handles,
                installed: [false; 2],
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
    for i in 0..2 {
        if state.installed[i] {
            continue;
        }
        let Some(mut shader) = shaders.get_mut(&state.handles[i]) else {
            continue;
        };
        let Source::Wesl(source) = &shader.source else {
            panic!("Expected Bevy 0.20 WESL PBR libraries");
        };
        let updated = if i == 0 {
            clearcoat_prepass(&anisotropy_prepass(source))
        } else {
            match *filter {
                GlassFilter::Default => transmission_filter(source),
                GlassFilter::Legacy => source.to_string(),
                GlassFilter::Reference => transmission_filter(source)
                    .replace(
                        "let taps = 2 * constants::SCREEN_SPACE_SPECULAR_TRANSMISSION_BLUR_TAPS;",
                        "let taps = 2048;",
                    )
                    .replace("let taps = 16;", "let taps = 2048;"),
                GlassFilter::LegacyReference => legacy_reference(source),
            }
        };
        shader.source = Source::Wesl(updated.into());
        state.installed[i] = true;
    }
}

// Bevy 0.20 implements Beer-Lambert absorption directly (T^(thickness/distance));
// the previous absorption correction is no longer needed.

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

/// End of a WESL statement block, including nested runtime/conditional blocks.
fn block_end(source: &str, start: usize) -> usize {
    let body = start + source[start..].find('{').expect("WESL block");
    let mut depth = 0;
    source[body..]
        .char_indices()
        .find_map(|(offset, c)| {
            match c {
                '{' => depth += 1,
                '}' => depth -= 1,
                _ => {}
            }
            (depth == 0).then_some(body + offset + 1)
        })
        .expect("closed WESL block")
}

fn anisotropy_prepass(source: &str) -> String {
    // SSAO supplies base normals, but must not suppress the tangent frame.
    let marker =
        "@if(PBR_ANISOTROPY_TEXTURE_SUPPORTED && VERTEX_TANGENTS && STANDARD_MATERIAL_ANISOTROPY)";
    let start = source.find(marker).expect("Bevy anisotropy block changed");
    let end = block_end(source, start);
    let block = source[start..end]
        .replace(marker, "@if(VERTEX_TANGENTS && STANDARD_MATERIAL_ANISOTROPY)")
        .replace("        if ((flags & pbr_types::STANDARD_MATERIAL_FLAGS_ANISOTROPY_TEXTURE_BIT)", "        @if(PBR_ANISOTROPY_TEXTURE_SUPPORTED)\n        if ((flags & pbr_types::STANDARD_MATERIAL_FLAGS_ANISOTROPY_TEXTURE_BIT)")
        .replace("let anisotropy_T = normalize(TBN *", "let anisotropy_frame = pbr_functions::calculate_tbn_mikktspace(pbr_input.world_normal, in.world_tangent);\n        let anisotropy_T = normalize(anisotropy_frame *");
    // The following brace closes !LOAD_PREPASS_NORMALS. Insert immediately after.
    let outer_end = end + source[end..].find('}').expect("prepass block end") + 1;
    format!(
        "{}{}\n{}{}",
        &source[..start],
        &source[end..outer_end],
        block,
        &source[outer_end..]
    )
}

fn clearcoat_prepass(source: &str) -> String {
    let marker = "@if(VERTEX_UVS && VERTEX_TANGENTS && STANDARD_MATERIAL_CLEARCOAT)";
    let start = source.find(marker).expect("Bevy clearcoat block changed");
    let end = block_end(source, start);
    let block = source[start..end].replace("            TBN,", "            pbr_functions::calculate_tbn_mikktspace(pbr_input.world_normal, in.world_tangent),");
    let outer_end = end + source[end..].find('}').expect("prepass block end") + 1;
    let without = format!(
        "{}{}\n{}{}",
        &source[..start],
        &source[end..outer_end],
        block,
        &source[outer_end..]
    );
    let prepass = "    pbr_input.N = prepass_utils::prepass_normal(in.position, 0u);";
    assert!(
        without.contains(prepass),
        "Bevy prepass changed: requalify clearcoat"
    );
    // Braces keep both statements inside the conditional branch.
    without.replace(prepass, &format!("    {{\n{prepass}\n    pbr_input.clearcoat_N = normalize(pbr_input.world_normal);\n    }}"))
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
        include_str!("shading/transmission.wesl"),
        &source[end..]
    )
}
