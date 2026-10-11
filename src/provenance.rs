//! Build provenance describes the executable that captured the data, not the
//! checkout of a later report generator. Lock versions include optional packages.
pub fn capture_provenance() -> serde_json::Value {
    serde_json::json!({
        "schema_version": 1,
        "semantic_taxonomy_version": crate::render::semantic::SemanticLabel::TAXONOMY_VERSION,
        "semantic_palette": crate::render::semantic::SemanticLabel::ALL.iter().map(|label| {
            let color = label.color().to_srgba();
            serde_json::json!({"id":crate::render::ground_truth::semantic_id(label),"label":label.as_str(),"srgb8":([color.red,color.green,color.blue].map(|v| (v*255.).round() as u8))})
        }).collect::<Vec<_>>(),
        "capture_engine": crate::CAPTURE_ENGINE_IDENTITY,
        "generator_version": bevy_zeroverse_capture::GENERATOR_VERSION,
        "crate_version": env!("CARGO_PKG_VERSION"),
        "source_sha256": env!("ZEROVERSE_SOURCE_SHA256"),
        "target": env!("ZEROVERSE_CAPTURE_TARGET"),
        "locked_package_versions": serde_json::from_str::<serde_json::Value>(include_str!(concat!(env!("OUT_DIR"), "/capture-packages.json"))).unwrap(),
        "features": {"human_motion":cfg!(feature="human_motion"), "web":cfg!(feature="web"), "multi_threaded":cfg!(feature="multi_threaded")},
    })
}
