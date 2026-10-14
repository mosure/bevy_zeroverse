use super::*;
use crate::graphics::{begin, save, text};
fn box_text(svg: &mut String, x: f64, y: f64, w: f64, title: &str, body: &str, color: &str) {
    svg.push_str(&format!(r##"<rect x="{x}" y="{y}" width="{w}" height="100" rx="10" fill="{color}" stroke="#bac7cc"/>"##));
    text(svg, x + 18., y + 36., 22, title);
    text(svg, x + 18., y + 69., 17, body);
}
fn arrow(svg: &mut String, a: (f64, f64), b: (f64, f64)) {
    svg.push_str(&format!(r##"<path d="M{} {} L{} {}" stroke="#71828a" stroke-width="3" fill="none" marker-end="url(#arrow)"/>"##,a.0,a.1,b.0,b.1));
}
pub(super) fn write(output: &Path, receipt: &Value) -> Result<()> {
    let mut svg = begin(1560, 780);
    svg.push_str(r##"<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#71828a"/></marker></defs>"##);
    text(
        &mut svg,
        32.,
        42.,
        29,
        "Seeded programs → immutable preparation → readiness → calibrated views",
    );
    text(&mut svg,32.,76.,18,"Shared bounded CPU pool · stable assembly order · GPU material and lighting caches · no quality reduction");
    box_text(
        &mut svg,
        32.,
        108.,
        460.,
        "Layout + appearance + cameras",
        "Architecture, placements, lights, paths, validation",
        "#e5ecec",
    );
    box_text(
        &mut svg,
        552.,
        108.,
        455.,
        "Parallel geometry builders",
        "Architecture | objects | AnnyBody + clothing",
        "#dbece5",
    );
    box_text(
        &mut svg,
        1067.,
        108.,
        455.,
        "Consumed material selection",
        "Only used finish programs; stable handles",
        "#e5ecec",
    );
    arrow(&mut svg, (492., 158.), (549., 158.));
    arrow(&mut svg, (1007., 158.), (1064., 158.));
    box_text(
        &mut svg,
        120.,
        280.,
        580.,
        "Material recipes + CPU fallback maps",
        "GPU recipes, transport albedo, normal/roughness mips",
        "#f8e7d7",
    );
    box_text(
        &mut svg,
        820.,
        280.,
        650.,
        "Transport → group plan → parallel meshes",
        "Shared triangles, BVH, mesh attributes and tangents",
        "#e3e3f7",
    );
    arrow(&mut svg, (1295., 211.), (1140., 277.));
    arrow(&mut svg, (1090., 210.), (550., 277.));
    text(&mut svg, 36., 270., 19, "FORK");
    text(&mut svg, 36., 417., 19, "JOIN");
    box_text(
        &mut svg,
        300.,
        424.,
        960.,
        "Bind transport → HDR reflections → lighting requests → group binding",
        "One coherent room, with semantic/instance attachments retained",
        "#dceaf4",
    );
    arrow(&mut svg, (410., 383.), (600., 421.));
    arrow(&mut svg, (1145., 383.), (960., 421.));
    box_text(
        &mut svg,
        32.,
        578.,
        460.,
        "Install + upload + GPU bake",
        "Maps/mips, GI; optional motion inference",
        "#e8e2f2",
    );
    box_text(
        &mut svg,
        552.,
        578.,
        455.,
        "Explicit readiness gates",
        "Assets, pipelines, lighting, motion, settle",
        "#e8e2f2",
    );
    box_text(
        &mut svg,
        1067.,
        578.,
        455.,
        "RGB + geometric raster → output",
        "Shared identities; async readback / tensor",
        "#dbece5",
    );
    arrow(&mut svg, (560., 527.), (260., 575.));
    arrow(&mut svg, (492., 628.), (549., 628.));
    arrow(&mut svg, (1007., 628.), (1064., 628.));
    text(&mut svg,32.,735.,19,"Native: next rooms prepare while current views render/read back. Browser: cooperative work on the event loop.");
    svg.push_str("</g></svg>");
    save(&svg, &output.join("pipeline_engine.svg"))?;
    let cpu = super::paper::serial_run(receipt)?;
    let stats = &cpu["statistics"];
    let total = stats["total"]["mean"].as_f64().unwrap();
    let mut svg = begin(1560, 690);
    text(&mut svg, 34., 45., 28, "Where preparation wall time goes");
    text(&mut svg,34.,78.,18,"Sequential CPU study · production GPU material recipes · mean milliseconds per room (share of total)");
    let groups: [(&str, &[&str], &str); 8] = [
        (
            "Layout + validation",
            &["layout_seconds", "validation_seconds"],
            "#647d88",
        ),
        ("Camera sampling", &["cameras_seconds"], "#9d7960"),
        ("Geometry assemblies", &["assembly_seconds"], "#24846d"),
        (
            "Parallel material / transport / meshes",
            &["parallel_assets_seconds"],
            "#7765ac",
        ),
        (
            "Transport binding",
            &["transport_binding_seconds"],
            "#789ab0",
        ),
        (
            "HDR reflection environment",
            &["environment_seconds"],
            "#c17637",
        ),
        (
            "Lighting request preparation",
            &["gi_request_seconds"],
            "#af5480",
        ),
        (
            "Remaining stages + bookkeeping",
            &[
                "motion_planning_seconds",
                "material_selection_seconds",
                "serial_meshes_seconds",
                "group_binding_seconds",
                "other_seconds",
            ],
            "#91a69e",
        ),
    ];
    for (i, (label, keys, color)) in groups.into_iter().enumerate() {
        let mean: f64 = keys
            .iter()
            .map(|k| stats["wall"][k]["mean"].as_f64().unwrap())
            .sum();
        let y = 120. + i as f64 * 64.;
        text(&mut svg, 34., y + 26., 19, label);
        svg.push_str(&format!(r##"<rect x="440" y="{y}" width="760" height="36" fill="#e4e9e7"/><rect x="440" y="{y}" width="{}" height="36" fill="{color}"/>"##,760.*mean/total));
        text(
            &mut svg,
            1230.,
            y + 26.,
            20,
            &format!("{:.1} ms  ({:.1}%)", 1000. * mean, 100. * mean / total),
        );
    }
    text(
        &mut svg,
        440.,
        669.,
        19,
        &format!(
            "100% = {:.1} ms mean task lifetime; overlapping branches are counted once.",
            1000. * total
        ),
    );
    svg.push_str("</g></svg>");
    save(&svg, &output.join("pipeline_profile.svg"))
}
